# SPDX-License-Identifier: MIT
"""Synthetic regressions for censored rankings and evidence-based categories."""

import copy
from datetime import date, timedelta
import json
import unittest

from scripts.openrouter.analysis import analyze


def evidence(category):
    return {
        "category": category,
        "confidence": "verified",
        "sources": ["https://example.org/model-card"],
        "notes": "Fixture architecture evidence.",
    }


def row(day, model, tokens):
    return {"date": day, "model_permaslug": model, "total_tokens": str(tokens)}


def days(start, count):
    beginning = date.fromisoformat(start)
    return [(beginning + timedelta(days=i)).isoformat() for i in range(count)]


class AnalysisTests(unittest.TestCase):
    def test_reviewed_canonical_ids_and_serving_variants_count_once_each(self):
        registry = {
            "vendor/model": evidence("hybrid_linear"),
            "vendor/model-20260701": evidence("hybrid_linear"),
        }
        catalog = [{"id": "vendor/model", "canonical_slug": "vendor/model-20260701"}]
        rows = [
            row("2026-07-01", "vendor/model-20260701", 100),
            row("2026-07-01", "vendor/model-20260701:free", 200),
            row("2026-07-01", "vendor/model:batch", 300),
            row("2026-07-01", "other", 50),
        ]
        result = analyze({"text": rows}, registry, catalog, "2026-07-01")
        current = result["windows"]["text"]["7"]["current"]
        self.assertEqual(current["tokens"]["hybrid_linear"], 600)
        self.assertEqual(current["total"], 650)
        self.assertEqual(result["historical_text_models"], 3)
        inventory = {item["model"]: item for item in result["inventory"]}
        self.assertEqual(inventory["vendor/model-20260701:free"]["classification_match"], "exact_alias")
        self.assertIsNone(inventory["vendor/model"]["tokens_all"])
        self.assertIsNone(inventory["vendor/model"]["tokens_30d"])
        self.assertFalse(inventory["vendor/model"]["observed_text"])

    def test_future_release_is_unknown_despite_family_name(self):
        registry = {"vendor/mamba-3": evidence("hybrid_mamba")}
        rows = [row("2026-07-01", "vendor/mamba-4", 100)]
        result = analyze({"text": rows}, registry, [], "2026-07-01")
        self.assertEqual(result["daily"]["text"]["2026-07-01"], {"unknown": 100})
        self.assertEqual(result["quality"]["text"]["unreviewed_models"], ["vendor/mamba-4"])

    def test_nonstandard_suffix_is_not_automatically_an_alias(self):
        registry = {"vendor/model": evidence("hybrid_mamba")}
        rows = [row("2026-07-01", "vendor/model:experimental", 100)]
        result = analyze({"text": rows}, registry, [], "2026-07-01")
        self.assertEqual(result["daily"]["text"]["2026-07-01"], {"unknown": 100})

    def test_exact_historical_evidence_wins_over_moving_catalog_alias(self):
        registry = {
            "vendor/current": evidence("hybrid_linear"),
            "vendor/dated-20250101": evidence("regular_attention"),
        }
        catalog = [{"id": "vendor/current", "canonical_slug": "vendor/dated-20250101"}]
        result = analyze(
            {"text": [row("2026-07-01", "vendor/dated-20250101", 100)]},
            registry, catalog, "2026-07-01",
        )
        self.assertEqual(result["daily"]["text"]["2026-07-01"], {"regular_attention": 100})

    def test_refreshed_moving_alias_does_not_classify_a_new_checkpoint(self):
        registry = {
            "vendor/current": evidence("hybrid_linear"),
            "vendor/checkpoint-20260101": evidence("hybrid_linear"),
        }
        catalog = [{"id": "vendor/current", "canonical_slug": "vendor/checkpoint-20260701"}]
        rows = [
            row("2026-07-01", "vendor/checkpoint-20260701", 100),
            row("2026-07-01", "vendor/checkpoint-20260701:free", 200),
            row("2026-07-01", "vendor/checkpoint-20260101", 50),
        ]
        result = analyze({"text": rows}, registry, catalog, "2026-07-01")
        self.assertEqual(result["daily"]["text"]["2026-07-01"], {
            "hybrid_linear": 50, "unknown": 300,
        })
        inventory = {item["model"]: item for item in result["inventory"]}
        self.assertEqual(inventory["vendor/checkpoint-20260701"]["confidence"], "unreviewed")
        self.assertTrue(inventory["vendor/checkpoint-20260701"]["current_catalog"])

    def test_conflicting_serving_variant_evidence_resolves_to_unknown(self):
        registry = {
            "vendor/ambiguous:free": evidence("hybrid_linear"),
            "vendor/ambiguous:batch": evidence("hybrid_mamba"),
        }
        result = analyze(
            {"text": [row("2026-07-01", "vendor/ambiguous", 100)]},
            registry, [], "2026-07-01",
        )
        self.assertEqual(result["daily"]["text"]["2026-07-01"], {"unknown": 100})
        item = next(i for i in result["inventory"] if i["model"] == "vendor/ambiguous")
        self.assertEqual(item["confidence"], "conflicting_alias_evidence")

    def test_equal_length_growth_and_share_denominator_include_tail(self):
        registry = {"vendor/model": evidence("hybrid_mamba")}
        rows = []
        for i, day in enumerate(days("2026-01-01", 14)):
            rows.extend([
                row(day, "vendor/model", 100 if i < 7 else 200),
                row(day, "other", 100),
            ])
        result = analyze({"text": rows}, registry, [], "2026-01-14")
        window = result["windows"]["text"]["7"]
        self.assertTrue(window["comparable"])
        self.assertEqual(window["growth_percent"]["hybrid_mamba"], 100)
        self.assertEqual(window["platform_growth_percent"], 50)
        self.assertAlmostEqual(window["current"]["shares"]["hybrid_mamba"], 200 / 3)
        self.assertAlmostEqual(window["share_change_pp"]["hybrid_mamba"], 50 / 3)
        self.assertIsNone(window["growth_percent"]["pure_mamba"])
        self.assertNotIn("pure_mamba", window["current"]["observed_categories"])

    def test_censored_category_disappearance_does_not_report_minus_100(self):
        registry = {"vendor/model": evidence("hybrid_mamba")}
        rows = [row(day, "other", 100) for day in days("2026-01-01", 14)]
        rows += [row(day, "vendor/model", 100) for day in days("2026-01-01", 7)]
        result = analyze({"text": rows}, registry, [], "2026-01-14")
        window = result["windows"]["text"]["7"]
        self.assertTrue(window["comparable"])
        self.assertIsNone(window["growth_percent"]["hybrid_mamba"])
        self.assertIsNone(window["share_change_pp"]["hybrid_mamba"])

    def test_missing_day_is_not_filled_and_suppresses_window_comparison(self):
        registry = {"vendor/model": evidence("regular_attention")}
        rows = [
            row(day, "vendor/model", 100)
            for day in days("2026-01-01", 14)
            if day != "2026-01-09"
        ]
        result = analyze({"text": rows}, registry, [], "2026-01-14")
        self.assertNotIn("2026-01-09", result["daily"]["text"])
        window = result["windows"]["text"]["7"]
        self.assertFalse(window["comparable"])
        self.assertEqual(window["current"]["days"], 6)
        self.assertEqual(window["current"]["missing_dates"], ["2026-01-09"])
        self.assertTrue(all(value is None for value in window["growth_percent"].values()))
        self.assertIsNone(window["platform_growth_percent"])

    def test_empty_month_and_partial_last_month_preserve_missingness(self):
        rows = [row(day, "other", 100) for day in ("2026-01-01", "2026-01-31", "2026-03-02")]
        result = analyze({"text": rows}, {}, [], "2026-03-03")
        january, february, march = result["monthly"]
        self.assertEqual(january["expected_days"], 31)
        self.assertFalse(january["complete"])
        self.assertEqual(february["days"], 0)
        self.assertIsNone(february["shares"]["other"])
        self.assertIsNone(february["mean_daily_tokens"]["other"])
        self.assertTrue(march["partial_month"])
        self.assertEqual(march["expected_days"], 3)
        self.assertEqual(march["missing_dates"], ["2026-03-01", "2026-03-03"])

    def test_top50_coverage_and_unknown_are_distinct(self):
        registry = {f"vendor/model-{i}": evidence("regular_attention") for i in range(30)}
        rows = [row("2026-01-01", f"vendor/model-{i}", 100) for i in range(50)]
        rows.append(row("2026-01-01", "other", 3000))
        result = analyze({"text": rows}, registry, [], "2026-01-01")
        coverage = result["coverage"]["text"]["all_history"]
        self.assertEqual(coverage["named_share_percent"], 62.5)
        self.assertEqual(coverage["classified_share_percent"], 37.5)
        self.assertEqual(coverage["unknown_share_percent"], 25)
        self.assertEqual(coverage["tail_share_percent"], 37.5)
        self.assertTrue(result["quality"]["text"]["top50_shape_consistent"])

    def test_context_datasets_are_independent_and_end_date_is_respected(self):
        registry = {"vendor/model": evidence("regular_attention")}
        result = analyze({
            "text": [row("2026-01-01", "vendor/model", 100), row("2026-01-02", "vendor/model", 999)],
            "100K": [row("2026-01-01", "vendor/model", 10)],
            "1M": [],
        }, registry, [], "2026-01-01")
        self.assertEqual(result["quality"]["text"]["excluded_after_end_date"], 1)
        self.assertEqual(result["windows"]["100K"]["7"]["current"]["total"], 10)
        self.assertEqual(result["windows"]["text"]["7"]["current"]["total"], 100)
        self.assertFalse(result["windows"]["1M"]["7"]["comparable"])
        self.assertEqual(result["monthly_by_dataset"]["1M"], [])

    def test_bad_rows_fail_instead_of_corrupting_totals(self):
        good = row("2026-01-01", "vendor/model", 100)
        for bad_rows in ([good, good], [row("2026-01-01", "vendor/model", -1)],
                         [{**good, "total_tokens": 1.5}], [{**good, "total_tokens": True}],
                         [{**good, "date": "2026-01-99"}]):
            with self.subTest(bad_rows=bad_rows), self.assertRaises(ValueError):
                analyze({"text": bad_rows}, {}, [], "2026-01-01")

    def test_output_deterministic_and_inputs_unchanged(self):
        registry = {"vendor/model": evidence("regular_attention")}
        rows = [row("2026-01-02", "other", 50), row("2026-01-01", "vendor/model", 100)]
        original = copy.deepcopy((rows, registry))
        first = analyze({"text": rows, "1M": []}, registry, [], "2026-01-02")
        second = analyze({"1M": [], "text": list(reversed(rows))}, registry, [], "2026-01-02")
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual((rows, registry), original)
        json.dumps(first, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
