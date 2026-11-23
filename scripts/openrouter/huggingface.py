# SPDX-License-Identifier: MIT
"""Compare dated Hugging Face download counts with OpenRouter token observations.

Downloads measure distribution activity, not inference requests. The unit here
is an explicitly mapped Hub repository; it differs from the exact OpenRouter
API identities used by the lifetime analysis. No forecast is adjusted here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import math
import random
import re

from .lifetime import quantile

MIN_CORRELATION_N = 5
BOOTSTRAP_DRAWS = 1000
SIZE_GROUPS = (("small", "≤10B parameters"), ("medium", "10–100B parameters"),
               ("large", ">100B parameters"), ("unknown", "Unknown size"))
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _repo_id(value):
    return value if isinstance(value, str) and REPO_ID.fullmatch(value) else None


def build_hf_mapping(registry, catalog):
    """Join exact IDs only; never remove a checkpoint date or infer name similarity."""
    catalog_rows = catalog.get("data", []) if isinstance(catalog, dict) else catalog
    by_id = defaultdict(set)
    for model in catalog_rows:
        repo = _repo_id(model.get("hugging_face_id"))
        if repo:
            for key in (model.get("id"), model.get("canonical_slug")):
                if key:
                    by_id[key].add(repo)
    grouped, unmapped, conflicts = {}, [], []
    for model_id, record in sorted(registry.get("models", {}).items()):
        if record.get("weights") != "open":
            continue
        explicit = _repo_id(record.get("hugging_face_id"))
        candidates = set(by_id.get(model_id, set()))
        if explicit:
            candidates.add(explicit)
        folded = {candidate.casefold() for candidate in candidates}
        if len(folded) > 1:
            conflicts.append({"model_id": model_id, "repo_ids": sorted(candidates)})
            continue
        if not candidates:
            unmapped.append(model_id)
            continue
        repo = explicit or sorted(candidates)[0]
        group = grouped.setdefault(repo.casefold(), {"repo_id": repo, "model_ids": [], "names": [],
                                                    "mapping_sources": []})
        group["model_ids"].append(model_id)
        if record.get("name") and record["name"] not in group["names"]:
            group["names"].append(record["name"])
        source = "reviewed registry" if explicit else "exact catalog ID/canonical_slug"
        if source not in group["mapping_sources"]:
            group["mapping_sources"].append(source)
    return {"groups": sorted(grouped.values(), key=lambda group: group["repo_id"].casefold()),
            "unmapped_open_ids": unmapped, "conflicts": conflicts,
            "policy": "Only confirmed-open registry IDs and explicit HF repository mappings; dated checkpoints stay distinct. Known aliases sharing a repository are summed once."}


def average_ranks(values):
    """One-based ranks with average ranks for ties."""
    result = [None] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Correlation vectors have different lengths")
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if not denominator:
        return None
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(dx, dy)) / denominator))


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def correlation_summary(models, bootstrap_draws=BOOTSTRAP_DRAWS):
    """Exploratory repository bootstrap; intervals do not repair selection bias."""
    ordered = sorted(models, key=lambda model: model["repo_id"].casefold())
    result = {"n": len(ordered), "status": "insufficient_sample", "minimum_n": MIN_CORRELATION_N,
              "spearman": {"value": None, "ci": None},
              "pearson_log1p": {"value": None, "ci": None},
              "bootstrap_draws": bootstrap_draws, "bootstrap_unit": "HF repository",
              "ids": [model["repo_id"] for model in ordered]}
    if len(ordered) < MIN_CORRELATION_N:
        return result
    xs = [model["hf_downloads_30d"] for model in ordered]
    ys = [int(model["openrouter_tokens_lower"]) for model in ordered]
    result["spearman"]["value"] = spearman(xs, ys)
    lx, ly = [math.log1p(x) for x in xs], [math.log1p(y) for y in ys]
    result["pearson_log1p"]["value"] = pearson(lx, ly)
    if result["spearman"]["value"] is None:
        result["status"] = "constant_values"
        return result
    result["status"] = "ok"
    rng = random.Random(818342)
    samples = {"spearman": [], "pearson_log1p": []}
    for _ in range(bootstrap_draws):
        indices = rng.choices(range(len(xs)), k=len(xs))
        rank_value = spearman([xs[i] for i in indices], [ys[i] for i in indices])
        log_value = pearson([lx[i] for i in indices], [ly[i] for i in indices])
        if rank_value is not None:
            samples["spearman"].append(rank_value)
        if log_value is not None:
            samples["pearson_log1p"].append(log_value)
    for key, values in samples.items():
        result[key]["valid_bootstrap_draws"] = len(values)
        if len(values) >= max(20, bootstrap_draws // 2):
            result[key]["ci"] = [quantile(values, .025), quantile(values, .975)]
    return result


def _nonnegative_int(value):
    # Reject bool, missing, nonfinite and fractional counts rather than manufacturing zero.
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if result >= 0 and str(result) == str(value) else None


def _size_group(parameters):
    if parameters is None or parameters <= 0:
        return "unknown"
    if parameters <= 10_000_000_000:
        return "small"
    return "medium" if parameters <= 100_000_000_000 else "large"


def _samples(models, bootstrap_draws):
    recent = [model for model in models if model["hf_status"] == "ok" and model["recently_observed"]]
    complete = [model for model in recent if model["fully_observed"]]
    return {"complete": correlation_summary(complete, bootstrap_draws),
            "recent_lower_bounds": correlation_summary(recent, bootstrap_draws)}


def analyze_huggingface(rows, registry, catalog, hf_snapshot, end_date, window_days=30,
                       bootstrap_draws=BOOTSTRAP_DRAWS):
    """Build a confined, matched-window adoption cross-check with censoring explicit."""
    if not isinstance(window_days, int) or isinstance(window_days, bool) or window_days <= 0:
        raise ValueError("window_days must be a positive integer")
    end = date.fromisoformat(end_date)
    start = end - timedelta(days=window_days - 1)
    days = [(start + timedelta(days=i)).isoformat() for i in range(window_days)]
    daily = defaultdict(dict)
    all_seen = set()
    for row in rows:
        day = row["date"]
        if not str(start) <= day <= str(end):
            continue
        model_id = row["model_permaslug"]
        count = _nonnegative_int(row["total_tokens"])
        if count is None or model_id in daily[day]:
            raise ValueError("Invalid or duplicate OpenRouter daily row")
        daily[day][model_id] = count
        if model_id != "other":
            all_seen.add(model_id)
    complete_dates = set()
    daily_limits = {}
    for day in days:
        entries = daily[day]
        named = [count for model_id, count in entries.items() if model_id != "other"]
        if len(named) > 50:
            raise ValueError("OpenRouter source exceeds top-50 schema")
        if len(named) == 50 and "other" in entries:
            complete_dates.add(day)
            daily_limits[day] = (min(named), entries["other"])
    mapping = build_hf_mapping(registry, catalog)
    snapshots = defaultdict(list)
    for entry in hf_snapshot.get("data", []):
        requested = _repo_id(entry.get("requested_id")) or _repo_id(entry.get("repo_id"))
        if requested:
            snapshots[requested.casefold()].append(entry)
    models = []
    for group in mapping["groups"]:
        candidates = snapshots.get(group["repo_id"].casefold(), [])
        record = candidates[0] if len(candidates) == 1 else {}
        downloads = _nonnegative_int(record.get("downloads"))
        actual_repo = _repo_id(record.get("repo_id"))
        status = "ok"
        if len(candidates) > 1 or actual_repo and actual_repo.casefold() != group["repo_id"].casefold():
            status = "conflict"
        elif not candidates:
            status = "missing"
        elif record.get("error") or downloads is None:
            status = "error"
        ids = group["model_ids"]
        observed, extra_bound, visible_days, alias_complete_days = 0, 0, 0, 0
        unknown_bound_days, last_seen = 0, None
        for day in days:
            named = daily[day]
            visible = [model_id for model_id in ids if model_id in named]
            absent = len(ids) - len(visible)
            observed += sum(named[model_id] for model_id in visible)
            if visible:
                visible_days += 1
                last_seen = day
            if absent == 0:
                alias_complete_days += 1
            elif day in daily_limits:
                cutoff, tail = daily_limits[day]
                extra_bound += min(absent * cutoff, tail)
            else:
                unknown_bound_days += 1
        parameters = _nonnegative_int(record.get("parameters_total"))
        models.append(dict(group, hf_status=status,
            hf_downloads_30d=downloads if status == "ok" else None,
            hf_downloads_all_time=_nonnegative_int(record.get("downloads_all_time")) if status == "ok" else None,
            hf_likes=_nonnegative_int(record.get("likes")) if status == "ok" else None,
            hf_created_at=record.get("created_at"), hf_retrieved_at=record.get("retrieved_at"),
            parameters_total=parameters, size_group=_size_group(parameters),
            observed_days=visible_days, complete_alias_days=alias_complete_days,
            source_missing_days=window_days - len(complete_dates), unknown_bound_days=unknown_bound_days,
            openrouter_tokens_lower=str(observed),
            openrouter_tokens_upper=None if unknown_bound_days else str(observed + extra_bound),
            fully_observed=alias_complete_days == window_days,
            recently_observed=visible_days > 0, latest_visible_date=last_seen))
    mapped_ids = {model_id for group in mapping["groups"] for model_id in group["model_ids"]}
    unjoined = sorted(all_seen - mapped_ids)
    counts = {key: sum(model["hf_status"] == key for model in models)
              for key in ("ok", "missing", "error", "conflict")}
    timestamp = hf_snapshot.get("meta", {}).get("retrieved_at")
    snapshot_date, offset = None, None
    if isinstance(timestamp, str):
        try:
            snapshot_date = date.fromisoformat(timestamp[:10])
            offset = (snapshot_date - end).days
        except ValueError:
            pass
    return {
        "schema_version": 1,
        "window": {"start": str(start), "end": str(end), "days": window_days},
        "alignment": {"hf_retrieved_at": timestamp,
                      "hf_snapshot_date": str(snapshot_date) if snapshot_date else None,
                      "window_end_offset_days": offset,
                      "note": "HF reports a rolling last-30-day count at retrieval; OpenRouter covers the stated complete UTC calendar days. HF does not expose an exact counter interval boundary here. These are nearest windows, not guaranteed identical periods."},
        "coverage": {"open_registry_ids": sum(record.get("weights") == "open" for record in registry.get("models", {}).values()),
                     "mapped_open_ids": len(mapped_ids), "mapped_repositories": len(models),
                     "hf_status_counts": counts, "complete_source_days": len(complete_dates),
                     "missing_source_dates": [day for day in days if day not in complete_dates],
                     "recently_observed_repositories": sum(model["recently_observed"] for model in models),
                     "fully_observed_repositories": sum(model["fully_observed"] for model in models),
                     "recent_unjoined_ids": unjoined,
                     "recent_confirmed_open_unjoined_ids": [model_id for model_id in unjoined if registry.get("models", {}).get(model_id, {}).get("weights") == "open"],
                     "recent_unclassified_ids": [model_id for model_id in unjoined if registry.get("models", {}).get(model_id, {}).get("weights") not in ("open", "closed")]},
        "models": models,
        "correlations": _samples(models, bootstrap_draws),
        "size_groups": [{"id": key, "label": label,
                         "correlations": _samples([model for model in models if model["size_group"] == key], bootstrap_draws)}
                        for key, label in SIZE_GROUPS],
        "mapping": mapping,
        "limitations": [
            "Download counts include qualifying file requests, including HEAD requests; they are not unique users, installations, inference calls, or GPU compute.",
            "The primary correlation uses only repositories whose every mapped OpenRouter alias is individually reported on every window day. It describes this selected sample, not all open models.",
            "The secondary correlation uses observed token minimums for recently visible models. Different amounts of hidden traffic can change the correlation; bootstrap intervals do not account for this censoring.",
            "A model absent from the top 50 has unknown traffic, not zero. Upper bounds apply only to the known mapped aliases and cannot be summed across repository groups because they share the same daily tail.",
            "Size groups use total parameters from HF metadata, not active MoE parameters. Unknown parameter counts remain unknown.",
            "Aliases absent from the reviewed registry, repackaged weights, mirrors, quantizations, private deployments, and other inference services are outside this mapping.",
            "Repository bootstrap intervals measure sampling variation; repositories within one model family or organization are dependent, and platform selection and mapping biases remain.",
            "A current download snapshot cannot establish historical download trends or predict remaining lifetime. No lifetime or request-demand multiplier is fitted from these correlations.",
        ],
    }
