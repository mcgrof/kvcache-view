# SPDX-License-Identifier: MIT
"""Calendar alignment, censoring and contributor provenance regressions."""

import copy
from datetime import date, timedelta
import json
import unittest

from scripts.openrouter.architecture_study import analyze_architecture_study
from scripts.openrouter.client import BUCKETS


def month_rows(month, counts):
    day = start = date.fromisoformat(month + "-01")
    result = {}
    while day.month == start.month:
        result[day.isoformat()] = dict(counts)
        day += timedelta(days=1)
    return result


def fixture():
    return {"end_date": "2026-10-18", "daily": {
        bucket: {
            **month_rows("2026-01", {"local_global_attention": 40, "hybrid_linear": 10, "unknown": 20, "attention_unspecified": 10, "other": 20}),
            **month_rows("2026-09", {"local_global_attention": 60, "hybrid_linear": 80, "unknown": 20, "attention_unspecified": 0, "other": 40}),
        } for bucket in BUCKETS
    }}


def dataset(study, bucket="text"):
    return next(row for row in study["datasets"] if row["bucket"] == bucket)


def architecture(month):
    return {row["category"]: row for row in month["architecture"]}


class ArchitectureStudyTests(unittest.TestCase):
    def test_full_calendar_daily_normalization_and_denominator(self):
        study = analyze_architecture_study(fixture())
        comparison = dataset(study)["comparison"]
        self.assertTrue(comparison["comparable"])
        self.assertEqual(comparison["baseline"]["total_tokens"], 3100)
        self.assertEqual(comparison["current"]["total_tokens"], 6000)
        self.assertEqual(comparison["daily_growth_multiple"], 2)
        self.assertEqual(comparison["current"]["unresolved_share_percent"], 10)
        self.assertEqual(comparison["current"]["tail_share_percent"], 20)
        self.assertEqual(comparison["baseline"]["unresolved_share_percent"], 30)
        self.assertEqual(comparison["baseline"]["attention_unspecified_share_percent"], 10)
        self.assertEqual(architecture(comparison["current"])["hybrid_linear"]["share_percent"], 40)
        changes = {row["category"]: row for row in comparison["architecture"]}
        self.assertEqual(changes["hybrid_linear"]["daily_growth_multiple"], 8)
        self.assertEqual(changes["hybrid_linear"]["share_change_pp"], 30)
        self.assertEqual(architecture(comparison["current"])["hybrid_linear"]["index_january_100"], 800)

    def test_incomplete_month_suppresses_analysis_without_extrapolation(self):
        attention = fixture()
        del attention["daily"]["1M"]["2026-09-15"]
        data = dataset(analyze_architecture_study(attention), "1M")
        current = data["comparison"]["current"]
        self.assertFalse(data["comparison"]["comparable"])
        self.assertEqual(current["days"], 29)
        self.assertEqual(current["observed_total_tokens"], 5800)
        self.assertEqual(current["missing_dates"], ["2026-09-15"])
        for key in ("total_tokens", "mean_daily_tokens", "unresolved_share_percent", "tail_share_percent", "share_of_text_percent"):
            self.assertIsNone(current[key])
        self.assertTrue(all(row["share_percent"] is None for row in current["architecture"]))
        self.assertTrue(all(row["daily_growth_multiple"] is None for row in data["comparison"]["architecture"]))

    def test_censored_absence_and_explicit_zero_remain_distinct(self):
        attention = fixture()
        attention["daily"]["text"].update(month_rows("2026-09", {"local_global_attention": 0, "other": 100}))
        comparison = dataset(analyze_architecture_study(attention))["comparison"]
        current = architecture(comparison["current"])
        self.assertEqual(current["local_global_attention"]["tokens"], 0)
        self.assertEqual(current["local_global_attention"]["share_percent"], 0)
        self.assertIsNone(current["hybrid_linear"]["tokens"])
        self.assertFalse(current["hybrid_linear"]["observed"])
        changes = {row["category"]: row for row in comparison["architecture"]}
        self.assertEqual(changes["local_global_attention"]["daily_growth_percent"], -100)
        self.assertIsNone(changes["hybrid_linear"]["daily_growth_percent"])
        self.assertIsNone(changes["hybrid_linear"]["share_change_pp"])

    def test_missing_dataset_and_zero_baseline_never_fabricate_growth(self):
        attention = fixture()
        del attention["daily"]["10M"]
        attention["daily"]["text"].update(month_rows("2026-01", {"local_global_attention": 0, "other": 100}))
        study = analyze_architecture_study(attention)
        self.assertEqual(study["bucket_order"], list(BUCKETS))
        self.assertEqual(len(dataset(study, "10M")["monthly"]), 9)
        self.assertIsNone(dataset(study, "10M")["monthly"][0]["observed_total_tokens"])
        changes = {row["category"]: row for row in dataset(study)["comparison"]["architecture"]}
        self.assertIsNone(changes["local_global_attention"]["daily_growth_multiple"])
        self.assertIsNone(architecture(dataset(study)["monthly"][-1])["local_global_attention"]["index_january_100"])

    def test_context_ratio_preserves_independent_denominator(self):
        attention = fixture()
        attention["daily"]["text"] = {
            **month_rows("2026-01", {"other": 1000}),
            **month_rows("2026-09", {"other": 1500}),
        }
        study = analyze_architecture_study(attention)
        comparison = dataset(study, "1M")["comparison"]
        self.assertEqual(comparison["baseline"]["share_of_text_percent"], 10)
        self.assertAlmostEqual(comparison["current"]["share_of_text_percent"], 100 * 200 / 1500)
        del attention["daily"]["text"]["2026-09-15"]
        comparison = dataset(analyze_architecture_study(attention), "1M")["comparison"]
        self.assertTrue(comparison["comparable"])
        self.assertIsNone(comparison["current"]["share_of_text_percent"])

    def test_contributors_use_exact_september_rows_and_keep_variants(self):
        attention = fixture()
        attention["inventory"] = [{
            "model": model, "name": model, "attention_style": category,
            "attention_sources": ["https://example.com/reviewed-config"],
            "attention_confidence": "verified", "attention_notes": "Exact reviewed entry",
            "tokens_30d": 999999,
        } for model, category in [
            ("local/model", "local_global_attention"),
            ("linear/model", "hybrid_linear"), ("linear/model:free", "hybrid_linear"),
            ("private/model", "unknown"), ("attention/model", "attention_unspecified"),
        ]]
        raw = {"text": [{"date": day, "model_permaslug": model, "total_tokens": count}
            for day in month_rows("2026-09", {})
            for model, count in [("local/model", 60), ("linear/model", 50), ("linear/model:free", 30), ("private/model", 20), ("attention/model", 0), ("other", 40)]]}
        raw["text"].append({"date": "2026-10-01", "model_permaslug": "linear/model", "total_tokens": 999999})
        result = dataset(analyze_architecture_study(attention, raw))["contributors"]
        linear = next(row for row in result if row["category"] == "hybrid_linear")
        self.assertTrue(linear["available"])
        self.assertEqual(linear["observed_model_count"], 2)
        self.assertEqual([row["tokens"] for row in linear["models"]], [1500, 900])
        self.assertEqual(linear["models"][0]["sources"], ["https://example.com/reviewed-config"])
        self.assertEqual(linear["models"][0]["share_of_category_percent"], 62.5)
        self.assertEqual(linear["models"][0]["mean_daily_tokens"], 50)
        pure = next(row for row in result if row["category"] == "pure_linear")
        self.assertTrue(pure["available"])
        self.assertFalse(pure["observed"])
        self.assertIsNone(pure["category_tokens"])
        raw["text"][0]["total_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "do not match"):
            analyze_architecture_study(attention, raw)

    def test_contributors_unavailable_without_raw_data_not_replaced_with_inventory(self):
        result = dataset(analyze_architecture_study(fixture()))["contributors"]
        self.assertTrue(all(not row["available"] and not row["models"] for row in result))

    def test_deterministic_json_and_no_mutation(self):
        attention = fixture()
        original = copy.deepcopy(attention)
        result = analyze_architecture_study(attention)
        self.assertEqual(result, analyze_architecture_study(attention))
        self.assertEqual(attention, original)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
