# SPDX-License-Identifier: MIT
"""Regression checks for exact joins, missing traffic and adoption statistics."""
import copy
from datetime import date, timedelta
import json
import math
import unittest

from scripts.openrouter.huggingface import (
    analyze_huggingface, average_ranks, build_hf_mapping,
    correlation_summary, pearson, spearman,
)


def source(appearances, days=3, missing=()):
    rows = []
    for offset in range(days):
        day = str(date(2026, 1, 1) + timedelta(days=offset))
        if day in missing:
            continue
        named = appearances.get(day, {})
        rows.extend({"date": day, "model_permaslug": model, "total_tokens": str(count)}
                    for model, count in named.items())
        rows.extend({"date": day, "model_permaslug": f"closed/filler-{i}", "total_tokens": "10"}
                    for i in range(50 - len(named)))
        rows.append({"date": day, "model_permaslug": "other", "total_tokens": "15"})
    return rows


def registry(entries):
    return {"models": {model: {"weights": "open", "hugging_face_id": repo, "name": model}
                       for model, repo in entries.items()}}


def snapshot(entries, retrieved_at="2026-01-04T10:00:00Z"):
    return {"meta": {"retrieved_at": retrieved_at},
            "data": [{"requested_id": repo, "repo_id": repo, "downloads": count,
                      "retrieved_at": retrieved_at, "parameters_total": 8_000_000_000}
                     for repo, count in entries.items()]}


class HuggingFaceTests(unittest.TestCase):
    def test_exact_mapping_preserves_dates_and_excludes_conflicts(self):
        reg = registry({"org/model-20260101": "org/old", "org/model-20260201": "org/new",
                        "org/model-20260201:free": "org/new", "org/conflict": "org/a",
                        "org/no-map": None})
        reg["models"]["org/unreleased"] = {"weights": "unknown", "hugging_face_id": "org/pending"}
        mapping = build_hf_mapping(reg, [{"id": "org/conflict", "hugging_face_id": "org/b"},
                                         {"id": "org/model", "hugging_face_id": "org/latest"}])
        self.assertEqual([group["repo_id"] for group in mapping["groups"]], ["org/new", "org/old"])
        self.assertEqual(mapping["groups"][0]["model_ids"], ["org/model-20260201", "org/model-20260201:free"])
        self.assertEqual(mapping["unmapped_open_ids"], ["org/no-map"])
        self.assertEqual(mapping["conflicts"][0]["model_id"], "org/conflict")

    def test_group_missing_bound_capped_by_shared_tail(self):
        reg = registry({"org/paid": "org/weights", "org/free": "org/weights", "org/third": "org/weights"})
        rows = source({"2026-01-01": {"org/paid": 100}, "2026-01-02": {"org/paid": 200}})
        result = analyze_huggingface(rows, reg, [], snapshot({"org/weights": 900}), "2026-01-03", 3)
        model = result["models"][0]
        self.assertEqual(model["openrouter_tokens_lower"], "300")
        # Two hidden aliases can use at most15 sharedtail, not2*15; all3hidden also<=15.
        self.assertEqual(model["openrouter_tokens_upper"], "345")
        self.assertEqual(model["observed_days"], 2)
        self.assertEqual(model["complete_alias_days"], 0)
        self.assertFalse(model["fully_observed"])
        self.assertEqual(result["correlations"]["recent_lower_bounds"]["n"], 1)

    def test_known_aliases_sum_once_and_complete_visibility(self):
        reg = registry({"org/paid": "org/weights", "org/free": "org/weights"})
        rows = source({f"2026-01-0{i}": {"org/paid": 100, "org/free": 40} for i in range(1, 4)})
        result = analyze_huggingface(rows, reg, [], snapshot({"org/weights": 1000}), "2026-01-03", 3)
        self.assertEqual(len(result["models"]), 1)
        model = result["models"][0]
        self.assertEqual(model["openrouter_tokens_lower"], "420")
        self.assertEqual(model["openrouter_tokens_upper"], "420")
        self.assertTrue(model["fully_observed"])
        self.assertEqual(result["correlations"]["complete"]["n"], 1)

    def test_missing_dates_leave_unknown_upper_and_never_enter_complete(self):
        reg = registry({"org/model": "org/weights"})
        rows = source({"2026-01-01": {"org/model": 100}}, missing=("2026-01-02",))
        result = analyze_huggingface(rows, reg, [], snapshot({"org/weights": 0}), "2026-01-03", 3)
        model = result["models"][0]
        self.assertIsNone(model["openrouter_tokens_upper"])
        self.assertEqual(model["hf_downloads_30d"], 0)
        self.assertEqual(model["hf_status"], "ok")
        self.assertFalse(model["fully_observed"])
        self.assertEqual(result["coverage"]["missing_source_dates"], ["2026-01-02"])

    def test_historical_unreported_is_not_a_zero_traffic_correlation_point(self):
        result = analyze_huggingface(source({}), registry({"org/model": "org/weights"}), [],
                                     snapshot({"org/weights": 500}), "2026-01-03", 3)
        model = result["models"][0]
        self.assertEqual(model["openrouter_tokens_lower"], "0")
        self.assertEqual(model["openrouter_tokens_upper"], "30")
        self.assertFalse(model["recently_observed"])
        self.assertEqual(result["correlations"]["recent_lower_bounds"]["n"], 0)

    def test_mismatched_or_duplicate_hf_snapshot_excluded(self):
        reg = registry({"org/model": "org/weights"})
        snap = snapshot({"org/weights": 100})
        snap["data"][0]["repo_id"] = "org/renamed"
        result = analyze_huggingface(source({}), reg, [], snap, "2026-01-03", 3)
        self.assertEqual(result["models"][0]["hf_status"], "conflict")
        self.assertIsNone(result["models"][0]["hf_downloads_30d"])
        snap["data"] *= 2
        self.assertEqual(analyze_huggingface(source({}), reg, [], snap, "2026-01-03", 3)["models"][0]["hf_status"], "conflict")

    def test_missing_hf_count_not_zero_and_alignment_explicit(self):
        reg = registry({"org/model": "org/weights"})
        snap = snapshot({"org/weights": None})
        result = analyze_huggingface(source({}), reg, [], snap, "2026-01-03", 3)
        self.assertEqual(result["models"][0]["hf_status"], "error")
        self.assertIsNone(result["models"][0]["hf_downloads_30d"])
        self.assertEqual(result["alignment"]["window_end_offset_days"], 1)

    def test_average_tied_ranks_and_known_statistics(self):
        self.assertEqual(average_ranks([10, 20, 20, 40]), [1, 2.5, 2.5, 4])
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [4, 3, 3, 1]), -1)
        self.assertAlmostEqual(pearson([1, 2, 3], [5, 7, 9]), 1)
        self.assertIsNone(pearson([1, 1], [2, 3]))

    def test_bootstrap_is_deterministic_order_independent_and_small_samples_suppressed(self):
        models = [{"repo_id": f"org/{i}", "hf_downloads_30d": i,
                   "openrouter_tokens_lower": str(i ** 2)} for i in range(1, 9)]
        first = correlation_summary(models, bootstrap_draws=100)
        second = correlation_summary(list(reversed(models)), bootstrap_draws=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["spearman"]["value"], 1)
        self.assertEqual(first["spearman"]["ci"], [1.0, 1.0])
        self.assertEqual(correlation_summary(models[:4])["status"], "insufficient_sample")

    def test_no_mutation_and_only_total_parameters_define_size(self):
        rows = source({"2026-01-01": {"org/model": 2 ** 60}})
        reg, snap = registry({"org/model": "org/weights"}), snapshot({"org/weights": 10})
        snap["data"][0]["parameters_total"] = 671_000_000_000
        before = copy.deepcopy((rows, reg, snap))
        result = analyze_huggingface(rows, reg, [], snap, "2026-01-03", 3)
        self.assertEqual((rows, reg, snap), before)
        self.assertEqual(result["models"][0]["size_group"], "large")
        self.assertEqual(result["models"][0]["openrouter_tokens_lower"], str(2 ** 60))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
