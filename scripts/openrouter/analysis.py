# SPDX-License-Identifier: MIT
"""Aggregate OpenRouter rankings without guessing new models' architectures.

The dataset exposes daily top-50 models and an unnamed ``other`` aggregate.
Category token totals therefore describe *individually observed* traffic,
not uncensored architecture totals. A missing model is never a zero-usage
measurement. Registry records are evidence, not rules for future releases.

This module is deliberately independent of network access, credentials, and
the wall clock. ``analyze`` returns only JSON-compatible, deterministic data.
"""

from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any


LABELS = {
    "regular_attention": "Regular attention (including sparse)",
    "hybrid_linear": "Hybrid linear attention",
    "hybrid_mamba": "Hybrid Mamba",
    "pure_linear": "Pure linear attention",
    "pure_mamba": "Pure Mamba",
    "other_hybrid": "Other recurrent / convolution hybrid",
    "unknown": "Undisclosed / unresolved",
    "other": "Below daily top 50",
    "non_generation": "Other task / routed system",
}

ARCHITECTURE_CATEGORIES = tuple(list(LABELS)[:6])
PERIODS = (7, 30, 90)
VARIANT_SUFFIXES = (":free", ":batch")
MAKERS = {
    "ai21": "AI21 Labs",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "ibm-granite": "IBM",
    "inclusionai": "Ant Group / inclusionAI",
    "liquid": "Liquid AI",
    "meta-llama": "Meta",
    "microsoft": "Microsoft",
    "minimax": "MiniMax",
    "mistralai": "Mistral AI",
    "moonshotai": "Moonshot AI",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "openrouter": "OpenRouter / preview identity unresolved",
    "qwen": "Alibaba / Qwen",
    "stealth": "Preview identity unresolved",
    "tencent": "Tencent",
    "xiaomi": "Xiaomi",
    "x-ai": "xAI",
    "z-ai": "Z.ai",
}


def _base_variant(model: str) -> str:
    """Only documented serving variants inherit the same checkpoint evidence."""
    for suffix in VARIANT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def _parse_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"Expected an ISO calendar date, got {value!r}")
    return parsed


def _dates(start: date, end: date) -> list[str]:
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range(max(0, (end - start).days + 1))
    ]


def _percent(numerator: int | float, denominator: int | float) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _classification_index(registry: dict) -> dict:
    """Bind reviewed serving variants, never live catalog release aliases.

    Catalog IDs may move to new checkpoints. A freshly fetched canonical ID
    must receive its own reviewed registry entry before inheriting an
    architecture. Only documented free/batch variants share the same exact
    checkpoint evidence. Conflicting variant evidence remains unknown.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for key, record in sorted(registry.items()):
        if record.get("category") not in LABELS:
            raise ValueError(f"Invalid category for registry model {key!r}")
        index[key].add(key)
        index[_base_variant(key)].add(key)
    return index


def _classify(model: str, registry: dict, index: dict) -> dict:
    if model == "other":
        return {
            "category": "other",
            "confidence": "not_classifiable",
            "sources": [],
            "notes": "Unnamed aggregate below the daily top-50 cutoff.",
            "classification_match": "tail",
            "alias_registry_key": None,
        }
    base = _base_variant(model)
    exact = model if model in registry else base if base in registry else None
    candidates = [exact] if exact else sorted(index.get(base, set()))
    categories = {registry[key]["category"] for key in candidates}
    if candidates and len(categories) == 1:
        key = candidates[0]
        record = registry[key]
        sources = record.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        return {
            **record,
            "sources": sorted(set(sources)),
            "notes": record.get("notes", ""),
            "confidence": record.get("confidence", "unspecified"),
            "classification_match": "exact" if key == model else "exact_alias",
            "alias_registry_key": key,
        }
    conflict = bool(candidates)
    return {
        "category": "unknown",
        "confidence": "conflicting_alias_evidence" if conflict else "unreviewed",
        "sources": sorted(
            {source for key in candidates for source in registry[key].get("sources", [])}
        ),
        "notes": (
            "Serving variants have conflicting architecture evidence; review required."
            if conflict
            else "No exact reviewed architecture entry. New releases are not classified by family name."
        ),
        "classification_match": "conflict" if conflict else "unmatched",
        "alias_registry_key": None,
    }


def _normalise_rows(rows: list[dict], end: date, name: str) -> tuple[list, int]:
    normalised = []
    seen = set()
    future = 0
    for row in rows:
        try:
            day = _parse_date(row["date"])
            model = row["model_permaslug"]
            value = row["total_tokens"]
            # Floats and booleans should not silently truncate to integers.
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError("total_tokens must be an integer or integer string")
            tokens = int(value)
            if tokens < 0 or not isinstance(model, str) or not model:
                raise ValueError("negative tokens or empty model")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} ranking row: {exc}") from exc
        if day > end:
            future += 1
            continue
        key = (day.isoformat(), model)
        if key in seen:
            raise ValueError(f"Duplicate {name} ranking row for {key[0]} / {model}")
        seen.add(key)
        normalised.append((day.isoformat(), model, tokens))
    return sorted(normalised), future


def _window(daily: dict, start: date, end: date) -> dict:
    expected = _dates(start, end)
    present = [day for day in expected if day in daily]
    counts: Counter = Counter()
    observed = set()
    for day in present:
        counts.update(daily[day])
        observed.update(daily[day])
    total = sum(counts.values())
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": len(present),
        "expected_days": len(expected),
        "missing_dates": [day for day in expected if day not in daily],
        "complete": len(present) == len(expected),
        "total": total,
        "tokens": {key: counts[key] for key in LABELS},
        "shares": {key: _percent(counts[key], total) for key in LABELS},
        "observed_categories": [key for key in LABELS if key in observed],
    }


def _comparison(daily: dict, end: date, days: int) -> dict:
    current = _window(daily, end - timedelta(days=days - 1), end)
    previous = _window(
        daily, end - timedelta(days=2 * days - 1), end - timedelta(days=days)
    )
    comparable = current["complete"] and previous["complete"]
    growth = {}
    share_change = {}
    for key in LABELS:
        both_observed = (
            key in current["observed_categories"]
            and key in previous["observed_categories"]
        )
        valid = comparable and both_observed
        before = previous["tokens"][key]
        growth[key] = (
            100.0 * (current["tokens"][key] / before - 1)
            if valid and before > 0
            else None
        )
        share_change[key] = (
            current["shares"][key] - previous["shares"][key]
            if valid and current["total"] > 0 and previous["total"] > 0
            else None
        )
    return {
        "current": current,
        "previous": previous,
        "comparable": comparable,
        "growth_percent": growth,
        "share_change_pp": share_change,
        "platform_growth_percent": (
            100.0 * (current["total"] / previous["total"] - 1)
            if comparable and previous["total"] > 0
            else None
        ),
    }


def _months(daily: dict, end: date) -> list[dict]:
    if not daily:
        return []
    month = _parse_date(min(daily)).replace(day=1)
    result = []
    while month <= end:
        calendar_end = month.replace(day=calendar.monthrange(month.year, month.month)[1])
        period_end = min(calendar_end, end)
        item = _window(daily, month, period_end)
        item.update(
            month=month.strftime("%Y-%m"),
            partial_month=period_end != calendar_end,
            calendar_days=calendar_end.day,
            mean_daily_tokens={
                key: value / item["days"] if item["days"] else None
                for key, value in item["tokens"].items()
            },
        )
        result.append(item)
        month = calendar_end + timedelta(days=1)
    return result


def _coverage(window: dict) -> dict:
    total = window["total"]
    tokens = window["tokens"]
    named = total - tokens["other"]
    classified = sum(tokens[key] for key in ARCHITECTURE_CATEGORIES)
    return {
        "total_tokens": total,
        "named_tokens": named,
        "named_share_percent": _percent(named, total),
        "classified_tokens": classified,
        "classified_share_percent": _percent(classified, total),
        "unknown_tokens": tokens["unknown"],
        "unknown_share_percent": _percent(tokens["unknown"], total),
        "tail_tokens": tokens["other"],
        "tail_share_percent": _percent(tokens["other"], total),
        "non_generation_tokens": tokens["non_generation"],
    }


def analyze(
    datasets: dict[str, list[dict]],
    registry: dict,
    catalog: list[dict],
    end_date: str,
) -> dict[str, Any]:
    """Return architecture trends through an explicitly selected complete day.

    ``datasets`` normally contains ``text``, ``100K`` and ``1M`` ranking rows.
    Context labels are opaque API categories, not assumed numeric boundaries.
    ``tokens_all`` and ``tokens_30d`` in inventory refer to the primary text
    dataset only. Serving variants remain separate rows, counted once each.
    Missing calendar days remain absent in daily output and suppress growth
    comparisons for affected windows; absent categories have null growth.
    """
    if "text" not in datasets:
        raise ValueError("The primary 'text' dataset is required")
    end = _parse_date(end_date)
    index = _classification_index(registry)
    normalised = {}
    excluded = {}
    for name in sorted(datasets):
        normalised[name], excluded[name] = _normalise_rows(datasets[name], end, name)

    all_models = set(registry)
    catalog_lookup: dict[str, dict] = {}
    for item in sorted(catalog, key=lambda x: x.get("id", "")):
        for key in (item.get("id"), item.get("canonical_slug")):
            if key:
                all_models.add(key)
                catalog_lookup.setdefault(key, item)
                catalog_lookup.setdefault(_base_variant(key), item)
    for rows in normalised.values():
        all_models.update(model for _, model, _ in rows)
    classifications = {key: _classify(key, registry, index) for key in sorted(all_models)}

    daily = {}
    windows = {}
    quality = {}
    coverage = {}
    model_datasets: dict[str, set[str]] = defaultdict(set)
    for name, rows in normalised.items():
        aggregated: dict[str, Counter] = defaultdict(Counter)
        day_rows: dict[str, list[str]] = defaultdict(list)
        for day, model, tokens in rows:
            aggregated[day][classifications[model]["category"]] += tokens
            day_rows[day].append(model)
            model_datasets[model].add(name)
        daily[name] = {
            day: {key: counts[key] for key in LABELS if key in counts}
            for day, counts in sorted(aggregated.items())
        }
        windows[name] = {
            str(days): _comparison(daily[name], end, days) for days in PERIODS
        }
        first = min(aggregated) if aggregated else None
        last = max(aggregated) if aggregated else None
        history = _window(daily[name], _parse_date(first) if first else end, end)
        coverage[name] = {
            "all_history": _coverage(history),
            "latest_30d": _coverage(windows[name]["30"]["current"]),
        }
        models = {model for _, model, _ in rows if model != "other"}
        quality[name] = {
            "rows": len(rows),
            "excluded_after_end_date": excluded[name],
            "start_date": first,
            "last_observed_date": last,
            "requested_end_date": end_date,
            "observed_days": len(aggregated),
            "expected_days": history["expected_days"] if first else 0,
            "missing_dates": history["missing_dates"] if first else [],
            "complete_through_end": bool(first) and history["complete"],
            "model_count": len(models),
            "unreviewed_models": sorted(
                model for model in models
                if classifications[model]["classification_match"] == "unmatched"
            ),
            "category_model_counts": dict(sorted(Counter(
                classifications[model]["category"] for model in models
            ).items())),
            "top50_shape_consistent": bool(rows) and all(
                len(day_models) == 51 and "other" in day_models
                for day_models in day_rows.values()
            ),
            "days_without_other": sorted(
                day for day, day_models in day_rows.items() if "other" not in day_models
            ),
            "zero_token_rows": sum(tokens == 0 for _, _, tokens in rows),
        }

    volumes: Counter = Counter()
    recent: Counter = Counter()
    recent_start = (end - timedelta(days=29)).isoformat()
    for day, model, tokens in normalised["text"]:
        volumes[model] += tokens
        if day >= recent_start:
            recent[model] += tokens
    leaders = {
        category: [
            {"model": model, "tokens": tokens}
            for model, tokens in sorted(recent.items(), key=lambda x: (-x[1], x[0]))
            if classifications[model]["category"] == category
        ]
        for category in LABELS
    }

    inventory = []
    for model, classification in classifications.items():
        if model == "other":
            continue
        catalog_item = catalog_lookup.get(model, catalog_lookup.get(_base_variant(model), {}))
        prefix = model.split("/", 1)[0]
        inventory.append({
            "model": model,
            "name": catalog_item.get("name", model),
            "maker": classification.get("maker", MAKERS.get(prefix, prefix)),
            **classification,
            "tokens_all": volumes.get(model),
            "tokens_30d": recent.get(model),
            "current_catalog": bool(catalog_item),
            "observed_text": model in volumes,
            "observed_30d": model in recent,
            "datasets": sorted(model_datasets[model]),
        })

    return {
        "schema_version": 1,
        "end_date": end_date,
        "labels": LABELS.copy(),
        "windows": windows,
        "monthly": _months(daily["text"], end),
        "monthly_by_dataset": {name: _months(values, end) for name, values in daily.items()},
        "daily": daily,
        "leaders_30d": leaders,
        "inventory": inventory,
        "quality": quality,
        "coverage": coverage,
        "historical_text_models": len(set(volumes) - {"other"}),
        "registry_size": len(registry),
        "inventory_size": len(inventory),
        "current_catalog_size": len(catalog),
        "category_model_counts": quality["text"]["category_model_counts"],
        "methodology": {
            "traffic_scope": "Reported public text-token rankings; private/ZDR traffic is excluded by the source.",
            "token_scope": "Input plus output tokens, using provider-dependent tokenizers.",
            "category_totals": "Sums of individually visible top-50 rows: lower bounds on category traffic.",
            "category_growth": "Growth of observed totals is not a lower bound on true growth; cutoff entry and exit can change visibility.",
            "absent_models": "Missing models or categories are censored, not measured zero usage.",
            "missing_dates": "Missing dates are never filled; affected comparison growth is null.",
            "classification": "Exact reviewed entries and documented free/batch variants only; live catalog aliases never classify unreviewed releases.",
            "context_buckets": "API labels are retained verbatim; numerical bucket boundaries are not inferred.",
            "cache_limit": "Token rankings do not measure cache hits, checkpoint bytes, residency tiers, offload, or restore IO.",
        },
    }
