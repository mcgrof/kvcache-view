# SPDX-License-Identifier: MIT
"""Regression checks for refreshable pages and safe untrusted model metadata."""

import unittest

from scripts.openrouter.render import render_context, render_trends


class RenderTests(unittest.TestCase):
    def analysis(self):
        return {
            "historical_text_models": 1,
            "daily": {"text": {"2027-02-09": {"unknown": 40, "other": 10}}},
            "monthly": [{"month": "2027-02", "days": 9,
                         "tokens": {"unknown": 40, "other": 10},
                         "shares": {"unknown": 80, "other": 20}}],
            "windows": {"text": {"30": {"current": {
                "start": "2027-01-11", "end": "2027-02-09", "days": 9,
                "tokens": {"unknown": 40, "other": 10},
                "shares": {"unknown": 80, "other": 20},
                "observed_categories": ["unknown", "other"],
            }, "comparable": False}}},
            "inventory": [{"model": "new/<script>alert(1)</script>",
                           "name": "New <script>model</script>",
                           "category": "unknown", "tokens_30d": None,
                           "sources": ["javascript:alert(1)", "https://example.org/model"],
                           "observed_text": False}],
        }

    def test_new_dates_and_null_observations_render_without_old_snapshot(self):
        html = render_trends(self.analysis(), {"as_of": "2027-02-10T00:00:00Z"})
        self.assertIn("2027-02-09", html)
        self.assertIn("2027-02-10T00:00:00Z", html)
        self.assertNotIn("2026-10-06", html)
        self.assertIn("Not individually observed", html)
        self.assertIn("incomplete; growth suppressed", html)

    def test_model_metadata_is_escaped_and_unsafe_source_urls_dropped(self):
        html = render_trends(self.analysis(), {})
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("javascript:", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('href="https://example.org/model"', html)

    def test_context_does_not_invent_missing_bucket_results(self):
        html = render_context(self.analysis(), {})
        self.assertIn("No context-filtered datasets have been downloaded", html)
        self.assertNotIn("1M architecture mix and growth", html)

    def test_per_page_download_urls_are_used(self):
        html = render_trends(self.analysis(), {
            "analysis_url": "data/openrouter/hybrid-trends.json",
            "provenance_url": "data/openrouter/hybrid-trends-provenance.json",
        })
        self.assertIn('href="data/openrouter/hybrid-trends.json"', html)
        self.assertIn('href="data/openrouter/hybrid-trends-provenance.json"', html)


if __name__ == "__main__":
    unittest.main()
