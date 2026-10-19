# SPDX-License-Identifier: MIT
"""Calendar-month context demand, preserving missing and censored observations.

The rankings expose token volume, not request counts or prompt lengths. Bucket
labels remain source labels. Ratios to unfiltered text and numerical
reconciliation do not establish bucket boundaries or a disjoint partition.
"""

from __future__ import annotations

import calendar
from collections import Counter
from datetime import date, timedelta

from .analysis import LABELS
from .client import BUCKETS


def _month_dates(month):
    start = date.fromisoformat(month + "-01")
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(calendar.monthrange(start.year, start.month)[1])
    ]


def _percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def _growth(current, baseline):
    return (
        100.0 * (current / baseline - 1)
        if current is not None and baseline is not None and baseline > 0
        else None
    )


def _difference(current, baseline):
    return current - baseline if current is not None and baseline is not None else None


def _month(daily, month):
    expected = _month_dates(month)
    present = [day for day in expected if day in daily]
    counts = Counter()
    observed = set()
    for day in present:
        counts.update(daily[day])
        observed.update(daily[day])
    total = sum(counts.values())
    complete = len(present) == len(expected)
    architecture = []
    for category, label in LABELS.items():
        measured = complete and category in observed
        architecture.append({
            "category": category,
            "label": label,
            "observed": category in observed,
            "tokens": counts[category] if measured else None,
            "mean_daily_tokens": counts[category] / len(expected) if measured else None,
            "share_percent": _percent(counts[category], total) if measured else None,
        })
    return {
        "month": month,
        "start": expected[0],
        "end": expected[-1],
        "expected_days": len(expected),
        "days": len(present),
        "missing_dates": [day for day in expected if day not in daily],
        "complete": complete,
        "observed_total_tokens": total if present else None,
        "total_tokens": total if complete else None,
        "mean_daily_tokens": total / len(expected) if complete else None,
        "index_january_100": None,
        "share_of_text_percent": None,
        # These measure the observed rows' coverage, not true category traffic.
        "unknown_share_percent": _percent(counts["unknown"], total) if complete else None,
        "tail_share_percent": _percent(counts["other"], total) if complete else None,
        "named_share_percent": _percent(total - counts["other"], total) if complete else None,
        "architecture": architecture,
    }


def _comparison(baseline, current):
    comparable = baseline["complete"] and current["complete"]
    baseline_categories = {row["category"]: row for row in baseline["architecture"]}
    changes = []
    for row in current["architecture"]:
        previous = baseline_categories[row["category"]]
        changes.append({
            "category": row["category"],
            "label": row["label"],
            "daily_growth_percent": _growth(row["mean_daily_tokens"], previous["mean_daily_tokens"]),
            "share_change_pp": _difference(row["share_percent"], previous["share_percent"]),
        })
    growth = _growth(current["mean_daily_tokens"], baseline["mean_daily_tokens"])
    return {
        "comparable": comparable,
        "baseline": baseline,
        "current": current,
        "daily_growth_percent": growth,
        "daily_growth_multiple": growth / 100.0 + 1 if growth is not None else None,
        "total_growth_percent": _growth(current["total_tokens"], baseline["total_tokens"]),
        "share_of_text_change_pp": _difference(current["share_of_text_percent"], baseline["share_of_text_percent"]),
        "architecture": changes,
    }


def _reconcile(daily, datasets, months):
    indexed = {item["bucket"]: {row["month"]: row for row in item["monthly"]} for item in datasets}
    result = []
    for month in months:
        complete = all(indexed[bucket][month]["complete"] for bucket in BUCKETS)
        text_tokens = indexed["text"][month]["total_tokens"]
        bucket_sum = sum(indexed[bucket][month]["total_tokens"] for bucket in BUCKETS[1:]) if complete else None
        differences = []
        for day in _month_dates(month):
            if all(day in daily.get(bucket, {}) for bucket in BUCKETS):
                total = sum(daily["text"][day].values())
                total_buckets = sum(sum(daily[bucket][day].values()) for bucket in BUCKETS[1:])
                differences.append(total_buckets - total)
        result.append({
            "month": month,
            "complete": complete,
            "text_tokens": text_tokens,
            "bucket_sum_tokens": bucket_sum,
            "difference_tokens": bucket_sum - text_tokens if complete else None,
            "bucket_sum_to_text_percent": _percent(bucket_sum, text_tokens) if complete else None,
            "daily_checks": {
                "days_compared": len(differences),
                "days_equal": sum(value == 0 for value in differences),
                "max_absolute_difference_tokens": max(map(abs, differences)) if differences else None,
            },
        })
    return result


def analyze_context_study(analysis):
    """Compare January and September 2026 using complete UTC calendar months.

    Consume validated, classified daily aggregates from ``analysis.analyze``.
    A partial input range remains visible as missing coverage; it never
    silently changes the study dates or annualizes a few observed days.
    """
    months = [f"2026-{month:02d}" for month in range(1, 10)]
    daily = analysis["daily"]
    datasets = []
    for bucket in BUCKETS:
        monthly = [_month(daily.get(bucket, {}), month) for month in months]
        baseline = monthly[0]["mean_daily_tokens"]
        for row in monthly:
            current = row["mean_daily_tokens"]
            if current is not None and baseline is not None and baseline > 0:
                row["index_january_100"] = 100.0 * current / baseline
        datasets.append({
            "bucket": bucket,
            "label": "All text" if bucket == "text" else bucket,
            "monthly": monthly,
        })
    overall = datasets[0]["monthly"]
    for dataset in datasets:
        for index, row in enumerate(dataset["monthly"]):
            if row["complete"] and overall[index]["complete"]:
                row["share_of_text_percent"] = _percent(row["total_tokens"], overall[index]["total_tokens"])
        dataset["comparison"] = _comparison(dataset["monthly"][0], dataset["monthly"][-1])
    return {
        "schema_version": 1,
        "baseline_month": months[0],
        "comparison_month": months[-1],
        "start_date": "2026-01-01",
        "end_date": "2026-09-30",
        "months": months,
        "bucket_order": list(BUCKETS),
        "datasets": datasets,
        "reconciliation": _reconcile(daily, datasets, months),
        "methodology": {
            "source": "https://openrouter.ai/api/v1/datasets/rankings-daily",
            "supplements": "https://openrouter.ai/data/newsletter/cache-is-crucial",
            "traffic_scope": "OpenRouter public text-token rankings; source excludes private/ZDR traffic.",
            "token_scope": "Input plus output tokens, using provider-dependent tokenizers; not request counts, input-only lengths, or unique information.",
            "comparison": "Full UTC calendar months: January has 31 days and September has 30. Headline growth uses mean tokens per day.",
            "coverage": "A day is observed when at least one ranking row exists. Full calendar coverage is required for monthly metrics; observed rows cannot prove upstream completeness.",
            "missingness": "Missing dates are not zero-filled. Incomplete months suppress totals, daily means, indices, and comparisons; partial observed totals are separately identified.",
            "context_buckets": "Source request-context labels are retained verbatim. Exact numerical boundaries and request-length classification rules are not inferred.",
            "ratios": "Bucket-to-text ratios compare independently filtered token volumes; they are not request shares and do not assert mutually exclusive or exhaustive buckets.",
            "reconciliation": "Bucket sums are numerical diagnostics only. Equality with all-text totals does not prove bucket semantics or request-level coverage.",
            "architecture": "Within-bucket shares use the bucket's full reported token denominator, including unknown named models and the unnamed other tail. Classified volumes are observed lower bounds, while growth is sensitive to model mix and daily top-50 censoring.",
            "absent_categories": "Unobserved categories remain null, not measured zero traffic. Explicit zero-token rows remain zero; zero baselines cannot define growth.",
            "interpretation": "These rankings do not measure cache hits, cached-token share, state bytes, memory residency, offload traffic, or storage demand. The newsletter's cache metrics come from a different analysis.",
        },
    }
