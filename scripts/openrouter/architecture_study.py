# SPDX-License-Identifier: MIT
"""Compare reviewed attention patterns on identical, complete calendar months.

Traffic is evidence of usage, not a controlled architecture benchmark. Retain
every source denominator and distinguish censored categories from observed
zero. The optional raw rows provide exact, dated model contributors; inventory
30-day counts cannot stand in for September counts.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

from .analysis import _normalise_rows
from .attention import LABELS
from .client import BUCKETS
from .context_study import _difference, _growth, _month_dates, _percent


MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 10))
CONTRIBUTOR_CATEGORIES = ("local_global_attention", "hybrid_linear", "pure_linear")


def _month(daily, month):
    expected = _month_dates(month)
    present = [day for day in expected if day in daily]
    counts = Counter()
    observed = set()
    for day in present:
        invalid = set(daily[day]) - set(LABELS)
        if invalid:
            raise ValueError(f"Unknown attention categories: {sorted(invalid)}")
        counts.update(daily[day])
        observed.update(daily[day])
    total = sum(counts.values())
    complete = len(present) == len(expected)

    def share(category):
        return _percent(counts[category], total) if complete and category in observed else None

    architecture = []
    for category, label in LABELS.items():
        measured = complete and category in observed
        architecture.append({
            "category": category, "label": label, "observed": category in observed,
            "tokens": counts[category] if measured else None,
            "mean_daily_tokens": counts[category] / len(expected) if measured else None,
            "share_percent": share(category), "index_january_100": None,
        })
    unresolved = {"unknown", "attention_unspecified"}
    return {
        "month": month, "start": expected[0], "end": expected[-1],
        "complete": complete, "days": len(present), "expected_days": len(expected),
        "missing_dates": [day for day in expected if day not in daily],
        "observed_total_tokens": total if present else None,
        "total_tokens": total if complete else None,
        "mean_daily_tokens": total / len(expected) if complete else None,
        "index_january_100": None, "share_of_text_percent": None,
        "architecture": architecture,
        "unresolved_share_percent": (
            _percent(sum(counts[key] for key in unresolved), total)
            if complete and observed.intersection(unresolved) else None
        ),
        "unknown_share_percent": share("unknown"),
        "attention_unspecified_share_percent": share("attention_unspecified"),
        "tail_share_percent": share("other"),
    }


def _comparison(baseline, current):
    previous = {row["category"]: row for row in baseline["architecture"]}
    changes = []
    for row in current["architecture"]:
        before = previous[row["category"]]
        growth = _growth(row["mean_daily_tokens"], before["mean_daily_tokens"])
        changes.append({
            "category": row["category"], "label": row["label"],
            "daily_growth_percent": growth,
            "daily_growth_multiple": growth / 100 + 1 if growth is not None else None,
            "share_change_pp": _difference(row["share_percent"], before["share_percent"]),
        })
    growth = _growth(current["mean_daily_tokens"], baseline["mean_daily_tokens"])
    return {
        "comparable": baseline["complete"] and current["complete"],
        "baseline": baseline, "current": current, "architecture": changes,
        "daily_growth_percent": growth,
        "daily_growth_multiple": growth / 100 + 1 if growth is not None else None,
        "share_of_text_change_pp": _difference(current["share_of_text_percent"], baseline["share_of_text_percent"]),
    }


def _contributors(attention, raw, dataset, limit=5):
    bucket = dataset["bucket"]
    current = dataset["monthly"][-1]
    result = [{
        "category": category, "label": LABELS[category], "month": MONTHS[-1],
        "complete": current["complete"], "available": False, "models": [],
    } for category in CONTRIBUTOR_CATEGORIES]
    if raw is None or bucket not in raw or not current["complete"]:
        return result
    inventory = {item["model"]: item for item in attention.get("inventory", [])}
    normalized, _ = _normalise_rows(raw[bucket], date(2026, 9, 30), bucket)
    counts = Counter()
    daily = {}
    for day, model, tokens in normalized:
        if not day.startswith(MONTHS[-1]):
            continue
        if model != "other" and model not in inventory:
            raise ValueError(f"Contributor model missing from attention inventory: {model}")
        category = "other" if model == "other" else inventory[model]["attention_style"]
        daily.setdefault(day, Counter())[category] += tokens
        counts[model] += tokens
    # Prevent accidental use of stale category aggregates with fresh model rows.
    expected = attention["daily"].get(bucket, {})
    for day in _month_dates(MONTHS[-1]):
        if day not in daily or dict(daily[day]) != expected.get(day):
            raise ValueError(f"Contributor rows do not match attention aggregates for {bucket} / {day}")
    categories = {row["category"]: row for row in current["architecture"]}
    for group in result:
        category = group["category"]
        selected = [
            (model, tokens) for model, tokens in counts.items()
            if model != "other" and inventory[model]["attention_style"] == category
        ]
        group["available"] = True
        group["observed"] = categories[category]["observed"]
        group["category_tokens"] = categories[category]["tokens"]
        group["observed_model_count"] = len(selected)
        for model, tokens in sorted(selected, key=lambda item: (-item[1], item[0]))[:limit]:
            item = inventory[model]
            group["models"].append({
                "model": model, "name": item.get("name", model), "tokens": tokens,
                "mean_daily_tokens": tokens / current["expected_days"],
                "share_of_category_percent": _percent(tokens, categories[category]["tokens"]),
                "share_of_dataset_percent": _percent(tokens, current["total_tokens"]),
                "sources": list(item.get("attention_sources", [])),
                "confidence": item.get("attention_confidence", "unreviewed"),
                "notes": item.get("attention_notes", ""),
                "kv_representation": item.get("kv_representation", "unknown"),
            })
    return result


def analyze_architecture_study(attention, datasets=None):
    """Summarize Jan–Sep 2026 without changing reviewed model classifications.

    ``attention`` is the result of ``analyze_attention``. Pass its matching raw
    ``datasets`` for September model contributors; omission leaves those
    unavailable rather than substituting a rolling 30-day inventory.
    """
    daily = attention["daily"]
    output = []
    for bucket in BUCKETS:
        monthly = [_month(daily.get(bucket, {}), month) for month in MONTHS]
        baseline = monthly[0]
        baseline_categories = {row["category"]: row for row in baseline["architecture"]}
        for row in monthly:
            first, current = baseline["mean_daily_tokens"], row["mean_daily_tokens"]
            if first is not None and first > 0 and current is not None:
                row["index_january_100"] = 100 * current / first
            for category in row["architecture"]:
                first = baseline_categories[category["category"]]["mean_daily_tokens"]
                current = category["mean_daily_tokens"]
                if first is not None and first > 0 and current is not None:
                    category["index_january_100"] = 100 * current / first
        output.append({"bucket": bucket, "label": "All text" if bucket == "text" else bucket, "monthly": monthly})
    text_months = output[0]["monthly"]
    for dataset in output:
        for row, overall in zip(dataset["monthly"], text_months):
            if row["complete"] and overall["complete"]:
                row["share_of_text_percent"] = _percent(row["total_tokens"], overall["total_tokens"])
        dataset["comparison"] = _comparison(dataset["monthly"][0], dataset["monthly"][-1])
        dataset["contributors"] = (
            _contributors(attention, datasets, dataset) if dataset["bucket"] in ("text", "1M") else []
        )
    return {
        "schema_version": 1, "baseline_month": MONTHS[0], "comparison_month": MONTHS[-1],
        "start_date": "2026-01-01", "end_date": "2026-09-30", "months": list(MONTHS),
        "snapshot_end_date": attention.get("end_date"),
        "bucket_order": list(BUCKETS),
        "groups": [{"category": category, "label": label} for category, label in LABELS.items()],
        "datasets": output,
        "methodology": {
            "source": "https://openrouter.ai/api/v1/datasets/rankings-daily",
            "dates": "Complete UTC calendar months; January has 31 days and September has 30. Growth uses mean daily tokens.",
            "measure": "Reported prompt plus completion tokens, not requests, users, memory bytes or a controlled performance benchmark.",
            "denominator": "Each context dataset retains all reported tokens, including unresolved architectures and the unnamed top-50 tail.",
            "categories": "Exact reviewed primary attention styles. Local/sliding-window models may include global attention; linear hybrids may include full or sparse attention. Mamba remains separate.",
            "missingness": "A category absent from named rankings is null, not zero usage. Incomplete months suppress totals, shares and growth. Explicit observed zero remains zero.",
            "unresolved": "Architecture unresolved and attention-style unresolved remain separate. Their combined visibility metric excludes the unnamed tail, which is reported separately.",
            "context": "Opaque source context labels; bucket/text ratios do not prove exact boundaries or a disjoint partition. Model context-window capacity is not measured request demand.",
            "contributors": "Top five individually observed September model/API variants per selected category in All text and 1M. Serving variants remain distinct and are counted once.",
            "population": "Public rankings exclude private/ZDR traffic. Shifts in model availability, pricing, free tiers and observed coverage can change shares.",
        },
    }
