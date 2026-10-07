# SPDX-License-Identifier: MIT
"""Check semantic edge cases that would misrepresent cache economics."""

import unittest

from scripts.openrouter.cache import analyze_cache, render_cache


class CacheCatalogTests(unittest.TestCase):
    def model(self, name, pricing):
        return {"id": name, "pricing": pricing}

    def test_missing_is_not_zero_and_free_has_no_discount(self):
        result = analyze_cache({"data": [
            self.model("a/missing", {"prompt": "0.000002"}),
            self.model("a/free", {"prompt": "0", "input_cache_read": "0"}),
            self.model("a/paid", {"prompt": "0.000002", "input_cache_read": "0"}),
        ]})
        rows = {row["id"]: row for row in result["rows"]}
        self.assertIsNone(rows["a/missing"]["prices_per_million"]["input_cache_read"])
        self.assertEqual(rows["a/free"]["prices_per_million"]["input_cache_read"], 0)
        self.assertIsNone(rows["a/free"]["listed_read_discount_percent"])
        self.assertEqual(rows["a/paid"]["listed_read_discount_percent"], 100)
        self.assertEqual(result["summary"]["cache_read_price_reported"], 2)
        self.assertEqual(result["summary"]["no_cache_price_reported"], 1)

    def test_invalid_prices_do_not_become_free_or_nonfinite_output(self):
        result = analyze_cache({"data": [self.model("a/b", {
            "prompt": "NaN", "input_cache_read": "-1",
            "input_cache_write": "Infinity", "input_cache_write_1h": True,
        })]})
        row = result["rows"][0]
        self.assertTrue(all(value is None for value in row["prices_per_million"].values()))
        self.assertTrue(all(value == "invalid" for value in row["price_status"].values()))
        self.assertIsNone(row["listed_read_discount_percent"])

    def test_preserves_overrides_and_never_infers_observed_usage(self):
        overrides = [{"min_prompt_tokens": 200000, "prompt": "0.000004"}]
        result = analyze_cache({"data": [self.model("a/b", {
            "prompt": "0.000002", "input_cache_read": "0.0000005", "overrides": overrides,
        })], "meta": {"as_of": "2026-10-07T00:00:00Z"}})
        row = result["rows"][0]
        self.assertEqual(row["listed_read_discount_percent"], 75)
        self.assertEqual(row["prices_per_million"]["prompt"], 2)
        self.assertTrue(row["has_overrides"])
        self.assertEqual(row["overrides"], overrides)
        self.assertIsNone(row["observed_cache_hit_rate"])
        self.assertIsNone(row["nvme_restore_bytes"])

    def test_duplicate_models_fail_instead_of_inflating_coverage(self):
        with self.assertRaisesRegex(ValueError, "Duplicate model id"):
            analyze_cache({"data": [self.model("a/b", {}), self.model("a/b", {})]})

    def test_catalog_footer_uses_catalog_provenance_and_downloads(self):
        analysis = analyze_cache({"data": [], "meta": {
            "as_of": "2026-10-07T14:00:00Z",
            "source_url": "https://openrouter.ai/api/v1/models?output_modalities=all",
        }})
        footer = render_cache(analysis, {"as_of": "rankings-only-date"}).split("<footer>", 1)[1]
        self.assertIn("Catalog fetched:", footer)
        self.assertIn("2026-10-07T14:00:00Z", footer)
        self.assertNotIn("rankings-only-date", footer)
        self.assertNotIn("licensed under", footer)
        self.assertIn('href="data/openrouter/cache-analysis.json"', footer)
        self.assertIn('href="data/openrouter/cache-provenance.json"', footer)
        self.assertNotIn("Architecture registry</a>", footer)


if __name__ == "__main__":
    unittest.main()
