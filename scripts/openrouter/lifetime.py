# SPDX-License-Identifier: MIT
"""Deterministic model-popularity lifetimes from censored OpenRouter rankings.

This measures exact public API identities, not model quality, unique checkpoints,
release-to-retirement age, or worldwide demand. Requests and compute are scenarios
elsewhere: this module never converts traffic tokens into measured requests.
"""
from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from datetime import date, timedelta
import random
from statistics import mean, median

EXIT_GAP_DAYS = 28
HORIZON_DAYS = 180
BOOTSTRAP_DRAWS = 500


def days_between(start, end):
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    return [(lo + timedelta(days=i)).isoformat() for i in range(max(0, (hi - lo).days + 1))]


def add_days(value, count):
    return (date.fromisoformat(value) + timedelta(days=count)).isoformat()


def elapsed(start, end):
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def episodes(present_dates, complete_dates, end_date, gap=EXIT_GAP_DAYS):
    """Find spells joined across short absences; unknown dates break proof of exit.

    An exit starts on the first of `gap` consecutive complete absent dates. It
    becomes known only on the last date. Later returns begin new episodes. Days
    before an unknown dataset date cannot form part of the confirming run.
    """
    present = set(present_dates)
    if not present:
        return []
    result, current, absent_start, absent_count = [], None, None, 0
    for day in days_between(min(present), end_date):
        if day in present:
            if current is None:
                current = {"start": day, "last_seen": day, "exit_date": None, "exit_confirmed_date": None}
            current["last_seen"] = day
            absent_start, absent_count = None, 0
        elif current is not None:
            if day not in complete_dates:
                absent_start, absent_count = None, 0
                continue
            if absent_count == 0:
                absent_start = day
            absent_count += 1
            if absent_count == gap:
                current.update(exit_date=absent_start, exit_confirmed_date=day)
                result.append(current)
                current, absent_start, absent_count = None, None, 0
    if current is not None:
        result.append(current)
    return result


def km_curve(records):
    """Kaplan-Meier curve; event at t precedes censoring at t."""
    grouped = defaultdict(lambda: [0, 0])
    for record in records:
        duration = int(record["duration_days"])
        if duration < 0:
            raise ValueError("Negative survival duration")
        grouped[duration][0 if record["event"] else 1] += 1
    at_risk, survival = len(records), 1.0
    curve = [{"day": 0, "survival": 1.0, "at_risk": at_risk, "events": 0, "censored": 0}]
    for duration, (events, censored) in sorted(grouped.items()):
        before = at_risk
        if before:
            survival *= 1 - events / before
        curve.append({"day": duration, "survival": survival, "at_risk": before,
                      "events": events, "censored": censored})
        at_risk -= events + censored
    return curve


def survival_at(curve, day):
    value = 1.0
    for point in curve:
        if point["day"] > day:
            break
        value = point["survival"]
    return value


def restricted_mean(curve, horizon):
    """Integral of a right-continuous KM step function over [0,horizon]."""
    area, previous, survival = 0.0, 0, 1.0
    for point in curve:
        at = min(point["day"], horizon)
        area += max(0, at - previous) * survival
        if point["day"] >= horizon:
            return area
        previous, survival = point["day"], point["survival"]
    return area + max(0, horizon - previous) * survival


def cluster_resample(records, rng):
    # Free/batch variants of an exact checkpoint share a bootstrap unit.
    groups = defaultdict(list)
    for record in records:
        groups[record["id"].split(":")[0]].append(record)
    keys = sorted(groups)
    return [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]] if keys else []


def cohort_summary(records, label, start, end, bootstrap_draws=BOOTSTRAP_DRAWS):
    rows = [r for r in records if start <= r["entry_date"] <= end and not r["left_boundary"]]
    curve = km_curve(rows)
    supported = bool(rows) and max(r["duration_days"] for r in rows) >= HORIZON_DAYS
    # A cohort whose every member exited before 180 has known zero survival tail.
    supported = supported or bool(rows) and all(r["event"] for r in rows)
    estimate = restricted_mean(curve, HORIZON_DAYS) if supported else None
    rng = random.Random(74191)
    bootstrap = [restricted_mean(km_curve(cluster_resample(rows, rng)), HORIZON_DAYS)
                 for _ in range(bootstrap_draws)] if estimate is not None else []
    token_values = [int(r["observed_180d_tokens"]) for r in rows if r["followup_180_complete"]]
    return {
        "label": label, "entry_start": start, "entry_end": end, "n": len(rows),
        "events": sum(r["event"] for r in rows), "censored": sum(not r["event"] for r in rows),
        "events_by_180": sum(r["event"] and r["duration_days"] <= 180 for r in rows),
        "rmst_180_days": estimate,
        "rmst_180_ci": [quantile(bootstrap, .025), quantile(bootstrap, .975)] if bootstrap else None,
        "median_days": next((p["day"] for p in curve if p["survival"] <= .5), None),
        "survival": curve,
        "mean_observed_180d_tokens": str(round(mean(token_values))) if token_values else None,
        "median_observed_180d_tokens": str(round(median(token_values))) if token_values else None,
        "demand_complete_followup_n": len(token_values),
        "demand_note": "Named-model token minimum in first 180 calendar days, including any later return; invisible dates contribute no measured tokens, not a claim of zero use. Dataset gaps are counted separately.",
        "records": rows,
    }


def compare_cohorts(earlier, later):
    left, right = earlier["records"], later["records"]
    difference = None
    ci = None
    if earlier["rmst_180_days"] is not None and later["rmst_180_days"] is not None:
        difference = later["rmst_180_days"] - earlier["rmst_180_days"]
        rng = random.Random(91317)
        draws = [restricted_mean(km_curve(cluster_resample(right, rng)), 180) -
                 restricted_mean(km_curve(cluster_resample(left, rng)), 180)
                 for _ in range(BOOTSTRAP_DRAWS)]
        ci = [quantile(draws, .025), quantile(draws, .975)]
    conclusion = "Insufficient follow-up for a matched 180-day comparison."
    if difference is not None:
        direction = "shorter" if difference < 0 else "longer"
        estimate = (f"The 2026 Q1 sample averages {abs(difference):.1f} days {direction} "
                    "from first top-50 appearance to the start of its first 28-day absence, "
                    "with each span capped at 180 days. Shorter absences count within the span.")
        if ci[0] <= 0 <= ci[1]:
            conclusion = ("No clear evidence of a change: the uncertainty includes both shorter and longer spans. "
                          + estimate)
        else:
            conclusion = estimate + " The bootstrap interval excludes no change, conditional on this cohort and classification."
    old_tokens, new_tokens = earlier["mean_observed_180d_tokens"], later["mean_observed_180d_tokens"]
    return {"difference_days": difference, "ci": ci, "conclusion": conclusion,
            "mean_observed_180d_tokens_ratio": int(new_tokens) / int(old_tokens) if old_tokens and int(old_tokens) and new_tokens else None,
            "caveat": "Different entrant mix, API alias policies, pricing/free access, platform growth and retrospective weight classification remain confounders; this is descriptive, not causal."}


def conditional_forecast(records, age, horizon=HORIZON_DAYS):
    curve = km_curve(records)
    risk = [r for r in records if r["duration_days"] > age]
    events_after = sum(r["event"] for r in risk)
    base = survival_at(curve, age)
    supported_horizon = max((r["duration_days"] - age for r in risk), default=0)
    result = {"age_days": age, "at_risk": len(risk), "events_after_age": events_after,
              "p25_remaining_days": None, "median_remaining_days": None, "p75_remaining_days": None,
              "mean_remaining_180_days": None, "projection_horizon_days": horizon,
              "max_supported_remaining_days": supported_horizon,
              "method": "Pooled first-episode Kaplan-Meier survival, conditional on surviving current episode age; forecast target is exit from overall top 50, not the open-weight top 10.",
              "note": "Historical analogue projection, not a calibrated individual forecast or confidence interval. Requires at least 10 at risk and 3 subsequent exits. Quantiles require 5 at risk at crossing. No extrapolation beyond observed follow-up."}
    if len(risk) < 10 or events_after < 3 or base <= 0:
        result["status"] = "insufficient_support"
        return result
    result["status"] = "historical_analogue"
    for key, percentile in (("p25_remaining_days", .25), ("median_remaining_days", .5), ("p75_remaining_days", .75)):
        result[key] = next((p["day"] - age for p in curve
                            if p["day"] > age and p["at_risk"] >= 5 and p["survival"] / base <= 1 - percentile), None)
    if supported_horizon >= horizon:
        result["mean_remaining_180_days"] = (restricted_mean(curve, age + horizon) - restricted_mean(curve, age)) / base
    return result


def landmark_forecast(models, current, safe_end, age_band=14, horizon=HORIZON_DAYS):
    """Remaining prominence from past top10 days of comparable episode age.

    One landmark per exact checkpoint stem prevents free/batch duplication;
    first-episode only, excluding the target checkpoint. No selection on future
    membership/exit and no claim of calibrated individual prediction.
    """
    age = current["current_episode"]["age_days"]
    target_stem = current["id"].split(":")[0]
    candidates = {}
    for model in models:
        if model["weights"] != "open" or model["left_boundary"]:
            continue
        stem = model["id"].split(":")[0]
        if stem == target_stem:
            continue
        episode = model["first_episode"]
        eligible = []
        for point in model["history"]:
            if point["date"] > safe_end or point["open_rank"] is None or point["open_rank"] > 10:
                continue
            if episode["exit_date"] and point["date"] >= episode["exit_date"]:
                continue
            point_age = elapsed(episode["start"], point["date"]) + 1
            if abs(point_age - age) <= age_band:
                eligible.append((abs(point_age - age), point["date"], point_age))
        if not eligible:
            continue
        distance, landmark_date, landmark_age = min(eligible)
        event = episode["event"]
        duration = elapsed(landmark_date, episode["exit_date"]) if event else elapsed(landmark_date, safe_end) + 1
        record = {"id": model["id"], "landmark_date": landmark_date, "landmark_age_days": landmark_age,
                  "duration_days": duration, "event": event}
        key = (distance, landmark_date, model["id"])
        if stem not in candidates or key < candidates[stem][0]:
            candidates[stem] = (key, record)
    records = [item[1] for _, item in sorted(candidates.items())]
    result = conditional_forecast(records, 0, horizon)
    result.update(age_days=age, age_band_days=age_band,
                  method="Kaplan-Meier remaining time from historical top10-open landmarks within 14 days of this episode age. One first-episode landmark per exact checkpoint stem; target checkpoint excluded. Target: sustained exit from overall top50.",
                  note="Historical popularity/age analogues, not a calibrated individual forecast. Requires 10 landmarks and 3 exits; quantiles require 5 at risk. Interquartile span describes historical variation, not confidence. No extrapolation beyond observed follow-up.",
                  landmarks=records,
                  first_episode_analogue_for_reentry=False)
    if current["reentries"]:
        for key in ("p25_remaining_days", "median_remaining_days", "p75_remaining_days", "mean_remaining_180_days"):
            result[key] = None
        result["status"] = "reentry_not_modeled"
        result["note"] = "This is a repeat episode. First-episode analogues are not used to forecast re-entries."
    return result


def analyze_lifetime(rows, registry, catalog, end_date, include_history=False):
    """Analyze saved daily text rankings. No network, wall-clock or credentials."""
    end_date = str(end_date)
    date.fromisoformat(end_date)
    registry = registry.get("models", registry)
    if isinstance(catalog, dict):
        catalog = catalog.get("data", [])
    selected = [r for r in rows if r["date"] <= end_date]
    if not selected:
        raise ValueError("No rankings at or before the requested lifetime cutoff")
    start_date = min(r["date"] for r in selected)
    calendar_dates = days_between(start_date, end_date)
    daily = defaultdict(dict)
    for row in selected:
        day, slug = row["date"], row["model_permaslug"]
        value = row["total_tokens"]
        if slug in daily[day]:
            raise ValueError("Duplicate date/model in lifetime rankings")
        if isinstance(value, bool) or str(int(value)) != str(value) or int(value) < 0:
            raise ValueError("Invalid token count")
        daily[day][slug] = int(value)
    complete_dates = set(daily)
    # API may return fewer than50 when fewer models have positive tokens; a
    # reported date is accepted, but incomplete current top50 is not ranked.
    latest = max(daily)
    latest_complete = max((d for d, v in daily.items() if len(set(v) - {"other"}) >= 50), default=None)
    missing_dates = [d for d in calendar_dates if d not in daily]
    safe_end = add_days(end_date, -(EXIT_GAP_DAYS - 1))
    histories = defaultdict(list)
    day_stats = {}
    for day, values in sorted(daily.items()):
        named = sorted(((slug, token) for slug, token in values.items() if slug != "other"), key=lambda x: (-x[1], x[0]))
        total = sum(values.values())
        open_order = {slug: i + 1 for i, (slug, _) in enumerate((pair for pair in named if registry.get(pair[0], {}).get("weights") == "open"))}
        minimum = named[49][1] if len(named) >= 50 else None
        upper = min(minimum, values.get("other", 0)) if minimum is not None else values.get("other")
        # If fewer than50 named and no other, no unreported positive-token model
        # is asserted: dataset coverage may be partial, so bound remains unknown.
        if minimum is None and "other" not in values:
            upper = None
        day_stats[day] = {"total": total, "cutoff": minimum, "hidden_bound": upper,
                          "open": sum(t for s, t in named if registry.get(s, {}).get("weights") == "open"),
                          "unknown": sum(t for s, t in named if registry.get(s, {}).get("weights", "unknown") == "unknown"),
                          "other": values.get("other", 0)}
        for rank, (slug, token) in enumerate(named, 1):
            histories[slug].append({"date": day, "tokens": str(token), "rank": rank,
                                    "open_rank": open_order.get(slug), "share_percent": 100 * token / total if total else 0})
    models, records = [], []
    for slug, history in sorted(histories.items()):
        meta = registry.get(slug, {})
        first, last = history[0]["date"], history[-1]["date"]
        model_episodes = episodes([h["date"] for h in history], complete_dates, end_date)
        first_ep = dict(model_episodes[0])
        event = first_ep["exit_date"] is not None and first_ep["exit_date"] <= safe_end
        duration = elapsed(first, first_ep["exit_date"]) if event else max(0, elapsed(first, safe_end) + 1)
        first_ep.update(event=event, duration_days=duration)
        current_ep = dict(model_episodes[-1])
        current_ep["age_days"] = (elapsed(current_ep["start"], current_ep["exit_date"])
                                  if current_ep["exit_date"] else elapsed(current_ep["start"], end_date) + 1)
        known = {h["date"]: int(h["tokens"]) for h in history}
        model_dates = days_between(first, end_date)
        bounded_dates = [d for d in model_dates if d in day_stats and (d in known or day_stats[d]["hidden_bound"] is not None)]
        subset_upper = sum(known[d] if d in known else day_stats[d]["hidden_bound"] for d in bounded_dates)
        unbounded = [d for d in model_dates if d not in bounded_dates]
        visible_total = sum(known.values())
        first180_end = add_days(first, 179)
        followup_complete = first180_end <= end_date
        observed180 = sum(v for d, v in known.items() if d <= first180_end)
        missing180 = len([d for d in days_between(first, min(first180_end, end_date)) if d not in daily])
        current_status = "exited" if current_ep["exit_date"] else ("present" if last == end_date else "exit_unconfirmed")
        model = {"id": slug, "name": meta.get("name", slug), "weights": meta.get("weights", "unknown"),
                 "first_seen": first, "last_seen": last, "days_in_top50": len(history),
                 "best_rank": min(h["rank"] for h in history),
                 "observed_span_days": elapsed(first, last) + 1, "left_boundary": first == start_date,
                 "first_episode": first_ep, "current_episode": current_ep, "episodes": model_episodes,
                 "reentries": len(model_episodes) - 1, "status": current_status,
                 "observed_tokens": str(visible_total),
                 "observed_window_upper_tokens": str(subset_upper) if not unbounded else None,
                 "available_dates_upper_tokens": str(subset_upper), "bounds_missing_days": len(unbounded),
                 "bounds_start": first, "bounds_end": end_date,
                 "observed_180d_tokens": str(observed180), "followup_180_complete": followup_complete,
                 "first_180d_missing_dataset_days": missing180,
                 "history": history, "sources": meta.get("sources", []),
                 "open_weights_since": meta.get("open_weights_since"),
                 "classification_confidence": meta.get("confidence", "unresolved"),
                 "active_parameters_b": meta.get("active_parameters_b"),
                 "active_input_parameters_b": meta.get("active_input_parameters_b"),
                 "active_output_parameters_b": meta.get("active_output_parameters_b"),
                 "active_parameters_source": meta.get("active_parameters_source"),
                 "active_parameters_note": meta.get("active_parameters_note"),
                 "weight_files_verified": meta.get("weight_files_verified", False)}
        models.append(model)
        if model["weights"] == "open" and not model["left_boundary"] and first <= safe_end:
            records.append({"id": slug, "entry_date": first, "duration_days": duration, "event": event,
                            "left_boundary": False, "observed_180d_tokens": str(observed180),
                            "followup_180_complete": followup_complete, "missing_dataset_days_180": missing180})
    cohorts = [cohort_summary(records, f"{year} Q1 entrants", f"{year}-01-02", f"{year}-03-31") for year in (2025, 2026)]
    paid_records = [r for r in records if not r["id"].endswith(":free")]
    paid_cohorts = [cohort_summary(paid_records, f"{year} Q1 non-free entrants", f"{year}-01-02", f"{year}-03-31") for year in (2025, 2026)]
    current_top10 = []
    if latest_complete == end_date:
        today = sorted((m for m in models if m["weights"] == "open" and m["last_seen"] == end_date), key=lambda m: (-int(m["history"][-1]["tokens"]), m["id"]))[:10]
        for rank, model in enumerate(today, 1):
            row = dict(model)
            latest_row = model["history"][-1]
            recent = [h for h in model["history"] if h["date"] >= add_days(end_date, -6)]
            recent_tokens = sum(int(h["tokens"]) for h in recent)
            row.update(open_rank=rank, global_rank=latest_row["rank"], latest_tokens=latest_row["tokens"],
                       recent_7d={"observed_days": len(recent), "calendar_days": 7,
                                  "missing_dataset_days": sum(d not in daily for d in days_between(add_days(end_date, -6), end_date)),
                                  "tokens": str(recent_tokens), "mean_daily_tokens_lower_bound": recent_tokens / 7})
            row["forecast"] = landmark_forecast(models, model, safe_end)
            future_days = row["forecast"]["mean_remaining_180_days"]
            row["forecast"]["additional_180d_tokens_at_frozen_rate"] = str(round(recent_tokens / 7 * future_days)) if future_days is not None else None
            # Frozen-rate scenario ends at a projected popularity exit. It is not
            # a lower bound, and traffic can persist below the cutoff afterward.
            current_top10.append(row)
    exited = sorted((m for m in models if m["weights"] == "open" and m["status"] == "exited"),
                    key=lambda m: (-int(m["observed_tokens"]), m["id"]))
    monthly = []
    months = sorted(set(d[:7] for d in calendar_dates))
    for month in months:
        month_start = month + "-01"
        year, month_number = map(int, month.split("-"))
        month_end = f"{month}-{calendar.monthrange(year, month_number)[1]:02}"
        dates = days_between(month_start, month_end)
        observed = [d for d in dates if d in day_stats]
        complete = len(observed) == len(dates)
        total = sum(day_stats[d]["total"] for d in observed)
        opened = sum(day_stats[d]["open"] for d in observed)
        monthly.append({"month": month, "calendar_days": len(dates), "observed_days": len(observed),
                        "complete": complete, "observed_total_tokens": str(total),
                        "mean_daily_tokens": total / len(dates) if complete else None,
                        "open_observed_tokens": str(opened), "open_observed_share_percent": 100 * opened / total if total else None,
                        "other_tokens": str(sum(day_stats[d]["other"] for d in observed)),
                        "unknown_named_tokens": str(sum(day_stats[d]["unknown"] for d in observed)),
                        "mean_top50_cutoff_tokens": mean(day_stats[d]["cutoff"] for d in observed) if complete and all(day_stats[d]["cutoff"] is not None for d in observed) else None})
    registry_counts = dict(Counter(m["weights"] for m in models))
    last_unknown = sorted(s for s in daily[end_date] if s != "other" and registry.get(s, {}).get("weights", "unknown") == "unknown") if end_date in daily else []
    if not include_history:
        # Current leaders retain chart histories; the full raw archive is the
        # reproducible source for other models, avoiding multi-megabyte duplication.
        for model in models:
            model.pop("history", None)
    return {"schema_version": 1, "start_date": start_date, "end_date": end_date,
            "coverage": {"expected_days": len(calendar_dates), "observed_days": len(daily),
                         "missing_dates": missing_dates, "latest_complete_date": latest_complete,
                         "registry_counts": registry_counts, "unclassified_latest": last_unknown,
                         "survival_confirmed_through": safe_end, "ranked_model_ids": len(models)},
            "methodology": {
                "unit": "Exact ranked OpenRouter model_permaslug; dated versions and free variants remain separate.",
                "population": "Public text token traffic on OpenRouter. Registry openness is reviewed current status, not proof of availability of weights on every historical day.",
                "endpoint": "First day of at least 28 consecutive observed calendar days absent from the overall daily top 50; this is popularity-window exit, not zero demand or model retirement.",
                "exit_gap_days": EXIT_GAP_DAYS,
                "survival_censoring": "Administrative censoring 27 days before cutoff leaves 28 days to confirm an exit. Models already present on the first dataset day are excluded from fitted survival and matched entrant cohorts.",
                "cohorts": "Jan 2–Mar 31 entrants in 2025 and 2026; 180-day restricted mean time to first sustained exit. Jan1 omitted in both years. Returns after exit are recorded separately.",
                "bootstrap": f"{BOOTSTRAP_DRAWS} deterministic percentile bootstrap draws, grouping free/batch variants by exact checkpoint stem; not a causal or family-clustered interval.",
                "demand": "Observed cumulative tokens are minimums since first observation. Each absent-model day has a possible upper bound min(rank50 tokens, other tokens); missing whole days make the full-window upper bound unavailable. Bounds cannot be summed across models because they share other.",
                "forecast": "Historical first-episode top10-open landmarks at comparable age (plus/minus 14 days), one per checkpoint stem excluding the target. Applied to today's ten leading reviewed open-weight API IDs; target is overall top50 exit. Reentry forecasts suppressed. No individualized accuracy or calibrated prediction intervals claimed.",
                "units": "All token totals are decimal integer strings; requests, input/output mix, compute, cache reuse and global lifetime demand are not observed.",
            },
            "monthly": monthly, "cohorts": cohorts, "cohort_comparison": compare_cohorts(*cohorts),
            "paid_only_cohorts": paid_cohorts, "paid_only_comparison": compare_cohorts(*paid_cohorts),
            "survival": {"n": len(records), "curve": km_curve(records), "records": records},
            "models": models, "exited_models": exited, "current_top10": current_top10}
