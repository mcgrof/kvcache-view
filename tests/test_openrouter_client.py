# SPDX-License-Identifier: MIT
import datetime as dt
import unittest
from unittest.mock import Mock
from scripts.openrouter.client import Client, DataError, merge_snapshot, validate_rankings


class SnapshotTests(unittest.TestCase):
    def payload(self, rows):
        return {"data": rows, "meta": {"as_of": "2026-10-07T10:00:00Z", "start_date": "2026-10-01", "end_date": "2026-10-02"}}

    def row(self, day, model, tokens):
        return {"date": day, "model_permaslug": model, "total_tokens": str(tokens)}

    def test_refresh_replaces_whole_date_not_only_returned_models(self):
        old = self.payload([self.row("2026-09-30", "a", 1), self.row("2026-10-01", "old-top-model", 40)])
        new = self.payload([self.row("2026-10-01", "new-model", 20), self.row("2026-10-01", "other", 30)])
        result = merge_snapshot(old, new, dt.date(2026, 10, 1), dt.date(2026, 10, 2), "text", "https://openrouter.ai/api/v1/datasets/rankings-daily")
        self.assertEqual({r["model_permaslug"] for r in result["data"]}, {"a", "new-model", "other"})
        self.assertNotIn("2026-10-02", {r["date"] for r in result["data"]})
        self.assertEqual(len(result["meta"]["snapshots"]), 1)

    def test_invalid_rows_do_not_pass(self):
        for token in [-1, "1.2", True]:
            with self.subTest(token=token), self.assertRaises(DataError):
                p = self.payload([{"date": "2026-10-01", "model_permaslug": "a", "total_tokens": token}])
                validate_rankings(p, dt.date(2026, 10, 1), dt.date(2026, 10, 2))

    def test_duplicate_rejected(self):
        r = self.row("2026-10-01", "a", 1)
        with self.assertRaises(DataError):
            validate_rankings(self.payload([r, r]), dt.date(2026, 10, 1), dt.date(2026, 10, 2))

    def test_credential_destination_and_missing_key(self):
        client = Client("DO_NOT_LOG_TEST_SECRET", interval=0)
        with self.assertRaises(DataError) as context:
            client.get("https://example.com/api/v1/elsewhere", authenticated=True)
        self.assertNotIn("DO_NOT_LOG_TEST_SECRET", str(context.exception))
        with self.assertRaises(DataError):
            Client(interval=0).get("https://openrouter.ai/api/v1/datasets/rankings-daily", authenticated=True)

    def test_invalid_credentials_are_rejected_without_echoing_the_value(self):
        secret = "DO_NOT_LOG_TEST_SECRET"
        for suffix in ("\n", "\r", "\x00", " ", "\t", "\u200b", "\u00e9"):
            with self.subTest(suffix=repr(suffix)), self.assertRaises(DataError) as context:
                Client(secret + suffix)
            self.assertNotIn(secret, str(context.exception))

    def test_upstream_header_validation_error_does_not_expose_key(self):
        secret = "DO_NOT_LOG_TEST_SECRET"
        client = Client(secret, interval=0)
        client.opener = Mock()
        client.opener.open.side_effect = ValueError("Invalid header value: Bearer " + secret)
        with self.assertRaises(DataError) as context:
            client.get("https://openrouter.ai/api/v1/datasets/rankings-daily", authenticated=True)
        self.assertNotIn(secret, str(context.exception))

    def test_refresh_cannot_drop_previously_observed_dates(self):
        old = self.payload([self.row("2026-10-01", "a", 100), self.row("2026-10-02", "a", 200)])
        for rows in ([], [self.row("2026-10-02", "a", 201)]):
            with self.subTest(rows=rows), self.assertRaises(DataError):
                merge_snapshot(old, self.payload(rows), dt.date(2026, 10, 1), dt.date(2026, 10, 2), "text", "https://openrouter.ai/api/v1/datasets/rankings-daily")
        self.assertEqual(len(old["data"]), 2)

    def test_refresh_cannot_drop_a_previously_reported_tail(self):
        old = self.payload([self.row("2026-10-01", "a", 100), self.row("2026-10-01", "other", 200)])
        new = self.payload([self.row("2026-10-01", "a", 101)])
        with self.assertRaises(DataError):
            merge_snapshot(old, new, dt.date(2026, 10, 1), dt.date(2026, 10, 2), "text", "https://openrouter.ai/api/v1/datasets/rankings-daily")

    def test_refresh_cannot_shorten_previously_complete_top_50(self):
        rows = [self.row("2026-10-01", f"model-{i}", 100) for i in range(50)]
        rows.append(self.row("2026-10-01", "other", 10))
        old = self.payload(rows)
        new = self.payload(rows[1:])
        with self.assertRaises(DataError):
            merge_snapshot(old, new, dt.date(2026, 10, 1), dt.date(2026, 10, 2), "1M", "https://openrouter.ai/api/v1/datasets/rankings-daily")

    def test_sparse_fresh_days_and_existing_sparse_corrections_remain_valid(self):
        first = self.payload([self.row("2026-10-01", "a", 100), self.row("2026-10-01", "b", 100)])
        validate_rankings(first, dt.date(2026, 10, 1), dt.date(2026, 10, 2))
        old = merge_snapshot(None, first, dt.date(2026, 10, 1), dt.date(2026, 10, 2), "1M", "https://openrouter.ai/api/v1/datasets/rankings-daily")
        new = self.payload([self.row("2026-10-01", "a", 150)])
        result = merge_snapshot(old, new, dt.date(2026, 10, 1), dt.date(2026, 10, 2), "1M", "https://openrouter.ai/api/v1/datasets/rankings-daily")
        self.assertEqual(result["data"], new["data"])
        self.assertNotIn("2026-10-02", {r["date"] for r in result["data"]})


if __name__ == "__main__":
    unittest.main()
