# SPDX-License-Identifier: MIT
"""Regressions for calendar comparisons, independent filters and missingness."""

import copy
from datetime import date, timedelta
import json
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

from scripts.openrouter.client import BUCKETS, update_rankings
from scripts.openrouter.context_study import analyze_context_study


def month_rows(month, counts):
    start = date.fromisoformat(month + "-01")
    result = {}
    day = start
    while day.month == start.month:
        result[day.isoformat()] = dict(counts)
        day += timedelta(days=1)
    return result


def fixture():
    daily = {}
    for bucket in BUCKETS:
        daily[bucket] = {
            **month_rows("2026-01", {"regular_attention": 40, "hybrid_linear": 20, "unknown": 10, "other": 30}),
            **month_rows("2026-09", {"regular_attention": 60, "hybrid_linear": 80, "unknown": 20, "other": 40}),
        }
    return {"daily": daily}


def datasets(study):
    return {item["bucket"]: item for item in study["datasets"]}


class ContextStudyTests(unittest.TestCase):
    def test_calendar_normalization_and_independent_bucket_denominators(self):
        analysis = fixture()
        # Overall traffic is larger and grows more slowly than this bucket.
        analysis["daily"]["text"] = {
            **month_rows("2026-01", {"other": 1000}),
            **month_rows("2026-09", {"other": 1500}),
        }
        study = analyze_context_study(analysis)
        data = datasets(study)
        comparison = data["1M"]["comparison"]
        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["baseline"]["expected_days"], 31)
        self.assertEqual(comparison["current"]["expected_days"], 30)
        self.assertEqual(comparison["baseline"]["total_tokens"], 3100)
        self.assertEqual(comparison["current"]["total_tokens"], 6000)
        self.assertEqual(comparison["daily_growth_percent"], 100)
        self.assertEqual(comparison["daily_growth_multiple"], 2)
        self.assertAlmostEqual(comparison["total_growth_percent"], (6000 / 3100 - 1) * 100)
        self.assertEqual(data["text"]["comparison"]["daily_growth_percent"], 50)
        self.assertEqual(comparison["baseline"]["share_of_text_percent"], 10)
        self.assertAlmostEqual(comparison["current"]["share_of_text_percent"], 100 * 200 / 1500)
        self.assertAlmostEqual(comparison["share_of_text_change_pp"], 10 / 3)
        self.assertEqual(data["1M"]["monthly"][-1]["index_january_100"], 200)
        self.assertIsNone(data["1M"]["monthly"][1]["index_january_100"])

    def test_incomplete_month_suppresses_metrics_instead_of_extrapolating(self):
        analysis = fixture()
        del analysis["daily"]["1M"]["2026-09-15"]
        data = datasets(analyze_context_study(analysis))
        comparison = data["1M"]["comparison"]
        current = comparison["current"]
        self.assertFalse(comparison["comparable"])
        self.assertEqual(current["days"], 29)
        self.assertEqual(current["missing_dates"], ["2026-09-15"])
        self.assertEqual(current["observed_total_tokens"], 5800)
        for key in ("total_tokens", "mean_daily_tokens", "index_january_100", "share_of_text_percent", "unknown_share_percent"):
            self.assertIsNone(current[key])
        for key in ("daily_growth_percent", "daily_growth_multiple", "total_growth_percent", "share_of_text_change_pp"):
            self.assertIsNone(comparison[key])
        self.assertTrue(all(row["share_percent"] is None for row in current["architecture"]))
        self.assertTrue(data["text"]["comparison"]["comparable"])

    def test_missing_overall_day_only_suppresses_ratio_not_complete_bucket(self):
        analysis = fixture()
        del analysis["daily"]["text"]["2026-01-15"]
        comparison = datasets(analyze_context_study(analysis))["1M"]["comparison"]
        self.assertEqual(comparison["daily_growth_percent"], 100)
        self.assertIsNone(comparison["baseline"]["share_of_text_percent"])
        self.assertIsNone(comparison["share_of_text_change_pp"])

    def test_architecture_visibility_and_tail_stay_in_denominator(self):
        study = analyze_context_study(fixture())
        comparison = datasets(study)["1M"]["comparison"]
        current = comparison["current"]
        arch = {item["category"]: item for item in current["architecture"]}
        changes = {item["category"]: item for item in comparison["architecture"]}
        self.assertEqual(arch["hybrid_linear"]["share_percent"], 40)
        self.assertEqual(current["unknown_share_percent"], 10)
        self.assertEqual(current["tail_share_percent"], 20)
        self.assertEqual(current["named_share_percent"], 80)
        self.assertEqual(changes["hybrid_linear"]["daily_growth_percent"], 300)
        self.assertEqual(changes["hybrid_linear"]["share_change_pp"], 20)
        self.assertIsNone(arch["pure_mamba"]["tokens"])
        self.assertIsNone(arch["pure_mamba"]["share_percent"])
        self.assertIsNone(changes["pure_mamba"]["daily_growth_percent"])

    def test_censored_disappearance_is_not_a_measured_decline_to_zero(self):
        analysis = fixture()
        analysis["daily"]["1M"].update(month_rows("2026-09", {"other": 200}))
        comparison = datasets(analyze_context_study(analysis))["1M"]["comparison"]
        changes = {item["category"]: item for item in comparison["architecture"]}
        self.assertIsNone(changes["hybrid_linear"]["daily_growth_percent"])
        self.assertIsNone(changes["hybrid_linear"]["share_change_pp"])

    def test_explicit_zero_is_distinct_from_missing_and_zero_baseline(self):
        analysis = fixture()
        analysis["daily"]["1M"].update(month_rows("2026-09", {"other": 0}))
        analysis["daily"]["10M"] = {
            **month_rows("2026-01", {"other": 0}),
            **month_rows("2026-09", {"other": 100}),
        }
        data = datasets(analyze_context_study(analysis))
        comparison = data["1M"]["comparison"]
        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["current"]["total_tokens"], 0)
        self.assertEqual(comparison["current"]["mean_daily_tokens"], 0)
        self.assertEqual(comparison["current"]["index_january_100"], 0)
        self.assertEqual(comparison["daily_growth_percent"], -100)
        self.assertIsNone(comparison["current"]["tail_share_percent"])
        self.assertIsNone(data["10M"]["comparison"]["daily_growth_percent"])
        self.assertIsNone(data["10M"]["monthly"][-1]["index_january_100"])

    def test_missing_dataset_has_all_nine_months_and_no_fabricated_zero(self):
        analysis = fixture()
        del analysis["daily"]["10M"]
        study = analyze_context_study(analysis)
        data = datasets(study)
        self.assertEqual(study["bucket_order"], list(BUCKETS))
        self.assertEqual(len(data["10M"]["monthly"]), 9)
        january = data["10M"]["monthly"][0]
        self.assertEqual(january["days"], 0)
        self.assertEqual(len(january["missing_dates"]), 31)
        self.assertIsNone(january["observed_total_tokens"])
        self.assertIsNone(january["total_tokens"])
        self.assertFalse(study["reconciliation"][0]["complete"])
        self.assertIsNone(study["reconciliation"][0]["bucket_sum_tokens"])

    def test_reconciliation_reports_overlap_numerically_without_normalizing(self):
        study = analyze_context_study(fixture())
        january = study["reconciliation"][0]
        self.assertTrue(january["complete"])
        self.assertEqual(january["text_tokens"], 3100)
        self.assertEqual(january["bucket_sum_tokens"], 15500)
        self.assertEqual(january["difference_tokens"], 12400)
        self.assertEqual(january["bucket_sum_to_text_percent"], 500)
        self.assertEqual(january["daily_checks"], {
            "days_compared": 31, "days_equal": 0, "max_absolute_difference_tokens": 400,
        })
        analysis = fixture()
        for bucket in BUCKETS[1:]:
            analysis["daily"][bucket] = month_rows("2026-01", {"other": 20})
        january = analyze_context_study(analysis)["reconciliation"][0]
        self.assertEqual(january["difference_tokens"], 0)
        self.assertEqual(january["daily_checks"]["days_equal"], 31)

    def test_output_is_json_safe_deterministic_and_does_not_mutate_input(self):
        analysis = fixture()
        original = copy.deepcopy(analysis)
        first = analyze_context_study(analysis)
        second = analyze_context_study({"daily": dict(reversed(list(analysis["daily"].items())))})
        self.assertEqual(first, second)
        self.assertEqual(analysis, original)
        json.dumps(first, allow_nan=False)

    def test_all_supported_source_filters_are_sent_without_relabeling(self):
        class FakeClient:
            def __init__(self):
                self.urls = []

            def get(self, url, authenticated=False):
                self.urls.append((url, authenticated))
                return {
                    "data": [{"date": "2026-01-01", "model_permaslug": "other", "total_tokens": "1"}],
                    "meta": {"start_date": "2026-01-01", "end_date": "2026-01-01", "as_of": "2026-10-18T00:00:00Z"},
                }

        client = FakeClient()
        day = date(2026, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            for bucket in BUCKETS:
                update_rankings(client, directory, bucket, day, day)
        self.assertEqual(len(client.urls), 6)
        for bucket, (url, authenticated) in zip(BUCKETS, client.urls):
            params = parse_qs(urlparse(url).query)
            self.assertTrue(authenticated)
            self.assertEqual(params["period"], ["day"])
            self.assertEqual(params["modality"], ["text"])
            self.assertEqual(params.get("context_bucket"), None if bucket == "text" else [bucket])


if __name__ == "__main__":
    unittest.main()
