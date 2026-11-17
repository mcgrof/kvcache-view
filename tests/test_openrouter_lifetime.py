# SPDX-License-Identifier: MIT
"""Regressions for censoring, exact variants, survival and bounded demand."""
import copy
from datetime import date, timedelta
import json
import unittest

from scripts.openrouter.lifetime import (
    analyze_lifetime, days_between, episodes, km_curve, restricted_mean,
    landmark_forecast, cohort_summary,
)


def ranking_rows(start, end, appearances, missing=()):
    rows = []
    for day in days_between(start, end):
        if day in missing:
            continue
        named = appearances.get(day, {})
        for slug, tokens in named.items():
            rows.append({"date": day, "model_permaslug": slug, "total_tokens": str(tokens)})
        # Exactly50 named rows each day; the named model minimum is10.
        for i in range(50 - len(named)):
            rows.append({"date": day, "model_permaslug": f"closed/filler-{i:02}", "total_tokens": "10"})
        rows.append({"date": day, "model_permaslug": "other", "total_tokens": "7"})
    return rows


def registry(*ids):
    return {"models": {slug: {"weights": "open", "name": slug, "sources": []} for slug in ids}}


class LifetimeTests(unittest.TestCase):
    def test_sustained_exit_and_reentry_not_single_day_disappearance(self):
        dates = days_between("2026-01-01", "2026-03-15")
        result = episodes(["2026-01-01", "2026-01-15", "2026-03-01"], set(dates), dates[-1])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["exit_date"], "2026-01-16")
        self.assertEqual(result[0]["exit_confirmed_date"], "2026-02-12")
        self.assertIsNone(result[1]["exit_date"])

    def test_missing_dataset_day_breaks_confirmation(self):
        dates = set(days_between("2026-01-01", "2026-02-15"))
        dates.remove("2026-01-15")
        result = episodes(["2026-01-01"], dates, "2026-02-15")
        self.assertEqual(result[0]["exit_date"], "2026-01-16")
        self.assertEqual(result[0]["exit_confirmed_date"], "2026-02-12")
        self.assertIsNone(episodes(["2026-01-01"], dates, "2026-02-10")[0]["exit_date"])

    def test_km_event_precedes_censor_tie_and_rmst_counts_survivors(self):
        curve = km_curve([{"duration_days": 10, "event": True},
                          {"duration_days": 10, "event": False},
                          {"duration_days": 20, "event": True}])
        self.assertAlmostEqual(curve[1]["survival"], 2 / 3)
        self.assertEqual(curve[1]["at_risk"], 3)
        self.assertAlmostEqual(restricted_mean(curve, 20), 10 + 10 * 2 / 3)
        self.assertAlmostEqual(restricted_mean(curve, 5), 5)

    def test_exact_aliases_censoring_and_daily_hidden_bound(self):
        appearances = {"2026-01-02": {"org/model:free": 100, "org/model": 200},
                       "2026-02-10": {"org/model": 300}}
        result = analyze_lifetime(ranking_rows("2026-01-01", "2026-02-10", appearances),
                                  registry("org/model", "org/model:free"), [], "2026-02-10")
        models = {m["id"]: m for m in result["models"]}
        paid, free = models["org/model"], models["org/model:free"]
        self.assertEqual(paid["observed_tokens"], "500")
        self.assertEqual(free["observed_tokens"], "100")
        self.assertEqual(paid["reentries"], 1)
        self.assertEqual(paid["status"], "present")
        self.assertEqual(free["status"], "exited")
        #40-day window since Jan2;2 visible dates and38 unreported, each<=7.
        self.assertEqual(paid["observed_window_upper_tokens"], str(500 + 38 * 7))
        self.assertEqual(result["current_top10"][0]["forecast"]["status"], "reentry_not_modeled")
        self.assertIsNone(result["current_top10"][0]["forecast"]["median_remaining_days"])
        self.assertEqual(result["current_top10"][0]["recent_7d"]["mean_daily_tokens_lower_bound"], 300 / 7)

    def test_missing_day_does_not_create_zero_or_finite_full_window_upper(self):
        rows = ranking_rows("2026-01-01", "2026-02-10", {"2026-01-02": {"org/model": 100}}, missing=["2026-01-20"])
        result = analyze_lifetime(rows, registry("org/model"), [], "2026-02-10")
        model = next(m for m in result["models"] if m["id"] == "org/model")
        self.assertEqual(result["coverage"]["missing_dates"], ["2026-01-20"])
        self.assertIsNone(model["observed_window_upper_tokens"])
        self.assertEqual(model["bounds_missing_days"], 1)
        self.assertEqual(model["observed_tokens"], "100")
        self.assertEqual(model["days_in_top50"], 1)
        self.assertNotIn("history", model)
        self.assertFalse(result["monthly"][0]["complete"])
        self.assertIsNone(result["monthly"][0]["mean_daily_tokens"])

    def test_large_counts_and_reproducibility_and_no_input_mutation(self):
        huge = 2 ** 60 + 37
        rows = ranking_rows("2026-01-01", "2026-01-03", {"2026-01-02": {"org/model": huge}, "2026-01-03": {"org/model": huge}})
        before = copy.deepcopy(rows)
        first = analyze_lifetime(rows, registry("org/model"), [], "2026-01-03")
        second = analyze_lifetime(list(reversed(rows)), registry("org/model"), [], "2026-01-03")
        self.assertEqual(first, second)
        self.assertEqual(rows, before)
        self.assertEqual(first["current_top10"][0]["observed_tokens"], str(huge * 2))
        json.dumps(first, allow_nan=False)

    def test_requested_cutoff_without_ranking_does_not_masquerade_as_current(self):
        rows = ranking_rows("2026-01-01", "2026-01-03", {"2026-01-03": {"org/model": 100}})
        result = analyze_lifetime(rows, registry("org/model"), [], "2026-01-04")
        self.assertEqual(result["current_top10"], [])
        self.assertIn("2026-01-04", result["coverage"]["missing_dates"])

    def test_survival_boundary_and_twenty_eight_day_recent_grace(self):
        rows = ranking_rows("2026-01-01", "2026-02-10", {"2026-01-01": {"org/boundary": 100}, "2026-02-01": {"org/new": 100}})
        result = analyze_lifetime(rows, registry("org/boundary", "org/new"), [], "2026-02-10")
        self.assertEqual(result["coverage"]["survival_confirmed_through"], "2026-01-14")
        self.assertEqual(result["survival"]["records"], [])
        new = next(m for m in result["models"] if m["id"] == "org/new")
        self.assertFalse(new["first_episode"]["event"])
        self.assertEqual(new["status"], "exit_unconfirmed")

    def test_cohort_does_not_average_completed_lifetimes_only(self):
        rows = [{"id": "a", "entry_date": "2026-02-01", "duration_days": 20,
                 "event": True, "left_boundary": False, "observed_180d_tokens": "100", "followup_180_complete": True},
                {"id": "b", "entry_date": "2026-02-01", "duration_days": 220,
                 "event": False, "left_boundary": False, "observed_180d_tokens": "300", "followup_180_complete": True}]
        result = cohort_summary(rows, "test", "2026-01-02", "2026-03-31", bootstrap_draws=10)
        self.assertEqual(result["rmst_180_days"], 100)
        self.assertEqual(result["mean_observed_180d_tokens"], "200")
        self.assertEqual(result["n"], 2)

    def test_landmark_uses_rank_at_landmark_and_excludes_same_checkpoint(self):
        def model(slug, rank):
            return {"id": slug, "weights": "open", "left_boundary": False,
                    "first_episode": {"start": "2025-01-01", "exit_date": "2025-05-01", "event": True},
                    "history": [{"date": "2025-01-10", "open_rank": rank}, {"date": "2025-03-01", "open_rank": 1}]}
        current = {"id": "target", "current_episode": {"age_days": 10}, "reentries": 0}
        candidates = [model("target", 1), model("target:free", 1), model("never-top10-at-age", 11),
                      model("other", 1), model("other:free", 2)]
        forecast = landmark_forecast(candidates, current, "2025-06-01", age_band=0)
        self.assertEqual(forecast["at_risk"], 1)
        self.assertEqual([r["id"] for r in forecast["landmarks"]], ["other"])
        self.assertEqual(forecast["landmarks"][0]["duration_days"], 111)
        self.assertIsNone(forecast["median_remaining_days"])

    def test_duplicate_and_noninteger_counts_fail(self):
        rows = ranking_rows("2026-01-01", "2026-01-01", {})
        with self.assertRaises(ValueError):
            analyze_lifetime(rows + [rows[0]], registry(), [], "2026-01-01")
        rows[0]["total_tokens"] = "1.5"
        with self.assertRaises(ValueError):
            analyze_lifetime(rows, registry(), [], "2026-01-01")


if __name__ == "__main__":
    unittest.main()
