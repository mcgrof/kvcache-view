# SPDX-License-Identifier: MIT
"""Attention refinements must preserve denominators, censoring and exact evidence."""

from datetime import date, timedelta
import json
import unittest

from scripts.openrouter.attention import analyze_attention, render_attention


def evidence(category):
    return {"category": category, "sources": ["https://example.org/config"],
            "notes": "Reviewed sequence mixer.", "confidence": "verified"}


def refinement(style, representation="GQA"):
    return {"style": style, "kv_representation": representation,
            "sources": ["https://example.org/attention"],
            "notes": "Reviewed exact attention mask.", "confidence": "verified"}


def row(day, model, tokens):
    return {"date": day, "model_permaslug": model, "total_tokens": str(tokens)}


class AttentionTests(unittest.TestCase):
    def test_refinement_conserves_all_tokens_and_keeps_two_unknowns_and_tail(self):
        registry = {"v/full": evidence("regular_attention"),
                    "v/sparse": evidence("regular_attention"),
                    "v/latent": evidence("regular_attention"),
                    "v/unreviewed": evidence("regular_attention"),
                    "v/hybrid": evidence("hybrid_linear")}
        detail = {"v/full": refinement("full_attention"),
                  "v/sparse": refinement("sparse_attention", "MLA"),
                  "v/latent": refinement("latent_attention", "MLA")}
        models = list(registry) + ["v/new", "other"]
        result = analyze_attention({"text": [row("2026-01-01", model, 100) for model in models]}, registry, [], "2026-01-01", detail)
        current = result["windows"]["text"]["7"]["current"]
        self.assertEqual(current["total"], 700)
        self.assertEqual(sum(current["tokens"].values()), 700)
        for key in ("full_attention", "sparse_attention", "latent_attention", "attention_unspecified", "hybrid_linear", "unknown", "other"):
            self.assertEqual(current["tokens"][key], 100)
        sparse = next(item for item in result["inventory"] if item["model"] == "v/sparse")
        self.assertEqual(sparse["kv_representation"], "MLA")
        self.assertEqual(sparse["attention_style"], "sparse_attention")

    def test_exact_evidence_and_free_alias_never_reclassify_future_release(self):
        registry = {"v/model": evidence("regular_attention")}
        detail = {"v/model": refinement("local_global_attention")}
        catalog = [{"id": "v/model", "canonical_slug": "v/model-v2"}]
        rows = [row("2026-01-01", key, 100) for key in ("v/model:free", "v/model-v2", "v/model:experimental")]
        result = analyze_attention({"text": rows}, registry, catalog, "2026-01-01", detail)
        self.assertEqual(result["daily"]["text"]["2026-01-01"], {"local_global_attention": 100, "unknown": 200})
        item = next(item for item in result["inventory"] if item["model"] == "v/model:free")
        self.assertEqual(item["attention_match"], "exact_alias")

    def test_fine_evidence_can_resolve_previously_unknown_architecture(self):
        result = analyze_attention({"text": [row("2026-01-01", "v/mamba", 100)]}, {}, [], "2026-01-01", {"v/mamba": refinement("pure_mamba", "not applicable")})
        self.assertEqual(result["daily"]["text"]["2026-01-01"], {"pure_mamba": 100})

    def test_attention_only_registry_models_remain_in_inventory_without_fake_traffic(self):
        result = analyze_attention({"text": [row("2026-01-01", "other", 100)]}, {}, [], "2026-01-01", {"v/mamba": refinement("pure_mamba", "not applicable")})
        item = next(item for item in result["inventory"] if item["model"] == "v/mamba")
        self.assertEqual(item["attention_style"], "pure_mamba")
        self.assertEqual(item["category"], "unknown")
        self.assertFalse(item["current_catalog"])
        self.assertFalse(item["observed_text"])
        self.assertIsNone(item["tokens_all"])
        self.assertIsNone(item["tokens_30d"])
        self.assertNotIn("pure_mamba", result["daily"]["text"]["2026-01-01"])
        self.assertEqual(result["windows"]["text"]["30"]["current"]["total"], 100)

    def test_conflicting_variant_refinements_do_not_guess_mask(self):
        registry = {"v/model": evidence("regular_attention")}
        detail = {"v/model:free": refinement("sparse_attention"), "v/model:batch": refinement("full_attention")}
        result = analyze_attention({"text": [row("2026-01-01", "v/model", 100)]}, registry, [], "2026-01-01", detail)
        self.assertEqual(result["daily"]["text"]["2026-01-01"], {"attention_unspecified": 100})

    def test_no_inference_from_moe_gqa_or_notes(self):
        registry = {"v/sparse-moe": {**evidence("regular_attention"), "notes": "MoE, GQA, FlashAttention."}}
        result = analyze_attention({"text": [row("2026-01-01", "v/sparse-moe", 100)]}, registry, [], "2026-01-01")
        self.assertEqual(result["daily"]["text"]["2026-01-01"], {"attention_unspecified": 100})

    def test_growth_uses_equal_windows_and_total_denominator(self):
        rows = []
        for i in range(14):
            day = str(date(2026, 1, 1) + timedelta(days=i))
            rows += [row(day, "v/sparse", 100 if i < 7 else 200), row(day, "other", 100)]
        result = analyze_attention({"text": rows}, {"v/sparse": evidence("regular_attention")}, [], "2026-01-14", {"v/sparse": refinement("sparse_attention")})
        window = result["windows"]["text"]["7"]
        self.assertTrue(window["comparable"])
        self.assertEqual(window["growth_percent"]["sparse_attention"], 100)
        self.assertEqual(window["platform_growth_percent"], 50)
        self.assertAlmostEqual(window["current"]["shares"]["sparse_attention"], 200 / 3)
        self.assertAlmostEqual(window["share_change_pp"]["sparse_attention"], 50 / 3)
        self.assertIsNone(window["growth_percent"]["pure_mamba"])

    def test_missing_day_and_censored_disappearance_do_not_imply_fall_to_zero(self):
        rows = []
        for i in range(14):
            day = str(date(2026, 1, 1) + timedelta(days=i))
            rows.append(row(day, "other", 100))
            if i < 7:
                rows.append(row(day, "v/full", 100))
        args = ({"v/full": evidence("regular_attention")}, [], "2026-01-14", {"v/full": refinement("full_attention")})
        result = analyze_attention({"text": rows}, *args)
        self.assertIsNone(result["windows"]["text"]["7"]["growth_percent"]["full_attention"])
        missing = [item for item in rows if item["date"] != "2026-01-10"]
        result = analyze_attention({"text": missing}, *args)
        self.assertFalse(result["windows"]["text"]["7"]["comparable"])
        self.assertIsNone(result["windows"]["text"]["7"]["platform_growth_percent"])

    def test_invalid_style_or_broad_mixer_conflict_fails(self):
        for detail in (refinement("invented"), refinement("hybrid_mamba")):
            with self.subTest(detail=detail), self.assertRaises(ValueError):
                analyze_attention({"text": []}, {"v/model": evidence("regular_attention")}, [], "2026-01-01", {"v/model": detail})

    def test_independent_contexts_and_empty_months_are_preserved(self):
        rows = [row("2026-01-01", "other", 100), row("2026-03-01", "other", 100)]
        result = analyze_attention({"text": rows, "100K": [row("2026-03-01", "other", 10)]}, {}, [], "2026-03-02")
        self.assertEqual(result["monthly"][1]["days"], 0)
        self.assertIsNone(result["monthly"][1]["shares"]["other"])
        self.assertTrue(result["monthly"][2]["partial_month"])
        self.assertEqual(result["windows"]["100K"]["7"]["current"]["total"], 10)
        json.dumps(result, allow_nan=False)

    def test_render_escapes_metadata_links_and_collapses_inventory(self):
        model = "v/<script>bad</script>"
        registry = {model: evidence("regular_attention")}
        detail = {model: {**refinement("full_attention"), "notes": "<script>bad</script>", "sources": ["javascript:alert(1)", "https://example.org/safe"]}}
        analysis = analyze_attention({"text": [row("2027-02-09", model, 100)]}, registry, [], "2027-02-09", detail)
        html = render_attention(analysis, {"as_of": "2027-02-10", "analysis_url": "data/openrouter/attention-trends.json"})
        self.assertNotIn("<script>bad</script>", html)
        self.assertNotIn("javascript:", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<details><summary>Open the complete model inventory", html)
        self.assertIn("2027-02-09", html)
        self.assertIn('href="data/openrouter/attention-trends.json"', html)
        self.assertIn("FlashAttention is an attention implementation", html)
        self.assertIn("No recurrent or hybrid style was individually observed", html)


if __name__ == "__main__":
    unittest.main()
