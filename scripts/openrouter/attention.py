# SPDX-License-Identifier: MIT
"""Attention-style trends with explicit, version-specific evidence.

The style registry is a refinement of the broader sequence-mixer registry.
Unreviewed softmax models remain attention_unspecified: neither a model name,
an MoE architecture nor a KV representation proves its attention mask.
"""

from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from datetime import date, timedelta
import math

from .analysis import (
    PERIODS, _base_variant, _dates, _normalise_rows, _parse_date, analyze,
)
from .render import _escape, _metric, _number, _page, _safe_link, _stat, _tokens


LABELS = {
    "full_attention": "Full attention",
    "latent_attention": "Full attention with latent KV",
    "sparse_attention": "Sparse / selected-token attention",
    "local_global_attention": "Local / sliding-window attention",
    "hybrid_linear": "Hybrid linear attention",
    "hybrid_mamba": "Hybrid Mamba",
    "pure_linear": "Pure linear / recurrent attention",
    "pure_mamba": "Pure Mamba",
    "other_hybrid": "Other sequence-mixer hybrid",
    "attention_unspecified": "Attention confirmed; style unresolved",
    "unknown": "Architecture unresolved",
    "other": "Unnamed top-50 tail",
    "non_generation": "Other task / routed system",
}
COLORS = {
    "full_attention": "#66b9ff", "latent_attention": "#80dcff",
    "sparse_attention": "#f7cd70", "local_global_attention": "#ff98be",
    "hybrid_linear": "#50e3c2", "hybrid_mamba": "#c398ff",
    "pure_linear": "#d4ee79", "pure_mamba": "#ed95ff",
    "other_hybrid": "#ffc183", "attention_unspecified": "#768fb8",
    "unknown": "#aebbd8", "other": "#f2a274", "non_generation": "#c4c4c4",
}
REFINED = tuple(list(LABELS)[:9])
SOFTMAX = ("full_attention", "latent_attention", "sparse_attention", "local_global_attention")
HYBRIDS = ("hybrid_linear", "hybrid_mamba", "pure_linear", "pure_mamba", "other_hybrid")


def _style_index(registry):
    index = defaultdict(set)
    for model, record in sorted(registry.items()):
        if record.get("style") not in LABELS or record.get("style") == "other":
            raise ValueError(f"Invalid attention style for {model!r}")
        if not isinstance(record.get("sources", []), list):
            raise ValueError(f"Attention sources must be a list for {model!r}")
        index[_base_variant(model)].add(model)
    return index


def _classify_style(item, registry, index):
    model, broad = item["model"], item["category"]
    base = _base_variant(model)
    exact = model if model in registry else base if base in registry else None
    candidates = [exact] if exact else sorted(index.get(base, set()))
    signatures = {
        (registry[key]["style"], registry[key].get("kv_representation", "unknown"))
        for key in candidates
    }
    if candidates and len(signatures) == 1:
        key = candidates[0]
        record = registry[key]
        style = record["style"]
        if broad in HYBRIDS and style != broad:
            raise ValueError(f"Attention style conflicts with sequence mixer for {model!r}")
        if broad == "regular_attention" and style in HYBRIDS:
            raise ValueError(f"Attention style conflicts with sequence mixer for {model!r}")
        return {
            "attention_style": style,
            "kv_representation": record.get("kv_representation", "unknown"),
            "attention_sources": sorted(set(record.get("sources", []))),
            "attention_notes": record.get("notes", ""),
            "attention_confidence": record.get("confidence", "reviewed"),
            "attention_match": "exact" if key == model else "exact_alias",
            "attention_registry_key": key,
        }
    fallback = "attention_unspecified" if broad == "regular_attention" else broad
    if fallback not in LABELS:
        fallback = "unknown"
    return {
        "attention_style": fallback,
        "kv_representation": "unknown",
        "attention_sources": item.get("sources", []),
        "attention_notes": (
            "Conflicting serving-variant style evidence; refinement suppressed. "
            if candidates else "No finer exact-model attention evidence. "
        ) + item.get("notes", ""),
        "attention_confidence": "conflicting_alias_evidence" if candidates else item.get("confidence", "unreviewed"),
        "attention_match": "conflict" if candidates else "broad_registry_only",
        "attention_registry_key": None,
    }


def _window(daily, start, end):
    expected = _dates(start, end)
    present = [day for day in expected if day in daily]
    counts = Counter()
    observed = set()
    for day in present:
        counts.update(daily[day])
        observed.update(daily[day])
    total = sum(counts.values())
    return {
        "start": str(start), "end": str(end), "days": len(present),
        "expected_days": len(expected), "complete": len(present) == len(expected),
        "missing_dates": [day for day in expected if day not in daily],
        "total": total,
        "tokens": {key: counts[key] for key in LABELS},
        "shares": {key: counts[key] * 100.0 / total if total else None for key in LABELS},
        "observed_categories": [key for key in LABELS if key in observed],
    }


def _comparison(daily, end, days):
    current = _window(daily, end - timedelta(days=days - 1), end)
    previous = _window(daily, end - timedelta(days=2 * days - 1), end - timedelta(days=days))
    comparable = current["complete"] and previous["complete"]
    growth, change = {}, {}
    for key in LABELS:
        observed = key in current["observed_categories"] and key in previous["observed_categories"]
        valid = comparable and observed
        before = previous["tokens"][key]
        growth[key] = 100.0 * (current["tokens"][key] / before - 1) if valid and before else None
        change[key] = current["shares"][key] - previous["shares"][key] if valid and current["total"] and previous["total"] else None
    return {
        "current": current, "previous": previous, "comparable": comparable,
        "growth_percent": growth, "share_change_pp": change,
        "platform_growth_percent": 100.0 * (current["total"] / previous["total"] - 1) if comparable and previous["total"] else None,
    }


def _months(daily, end):
    if not daily:
        return []
    month = date.fromisoformat(min(daily)).replace(day=1)
    result = []
    while month <= end:
        calendar_end = month.replace(day=calendar.monthrange(month.year, month.month)[1])
        period_end = min(calendar_end, end)
        item = _window(daily, month, period_end)
        item.update(month=month.strftime("%Y-%m"), partial_month=period_end != calendar_end,
                    calendar_days=calendar_end.day,
                    mean_daily_tokens={key: item["tokens"][key] / item["days"] if item["days"] else None for key in LABELS})
        result.append(item)
        month = calendar_end + timedelta(days=1)
    return result


def analyze_attention(datasets, registry, catalog, end_date, attention_registry=None):
    """Refine all named history/catalog variants; preserve each source traffic row.

    Attention registry entries are exact evidence, not family-name rules. KV
    representation is descriptive and does not create overlapping traffic bins.
    """
    attention_registry = attention_registry or {}
    index = _style_index(attention_registry)
    # Include reviewed attention-only identities without inventing catalog
    # presence, source traffic or a previously reviewed broad architecture.
    combined_registry = dict(registry)
    for model in attention_registry:
        combined_registry.setdefault(model, {
            "category": "unknown", "confidence": "attention_registry_only",
            "sources": [],
            "notes": "Reviewed in the attention-style registry; no broad-registry entry.",
        })
    base = analyze(datasets, combined_registry, catalog, end_date)
    inventory = [{**item, **_classify_style(item, attention_registry, index)} for item in base["inventory"]]
    styles = {item["model"]: item["attention_style"] for item in inventory}
    styles["other"] = "other"
    end = _parse_date(end_date)
    daily, windows, monthly, coverage = {}, {}, {}, {}
    for dataset, source_rows in sorted(datasets.items()):
        normalised, _ = _normalise_rows(source_rows, end, dataset)
        groups = defaultdict(Counter)
        for day, model, tokens in normalised:
            groups[day][styles[model]] += tokens
        daily[dataset] = {day: {key: values[key] for key in LABELS if key in values} for day, values in sorted(groups.items())}
        windows[dataset] = {str(days): _comparison(daily[dataset], end, days) for days in PERIODS}
        monthly[dataset] = _months(daily[dataset], end)
        latest = windows[dataset]["30"]["current"]
        refined = sum(latest["tokens"][key] for key in REFINED)
        coverage[dataset] = {
            "total_tokens": latest["total"], "refined_tokens": refined,
            "refined_share_percent": refined * 100.0 / latest["total"] if latest["total"] else None,
            "attention_unspecified_tokens": latest["tokens"]["attention_unspecified"],
            "unknown_tokens": latest["tokens"]["unknown"], "tail_tokens": latest["tokens"]["other"],
        }
    quality = {}
    for dataset, values in base["quality"].items():
        seen = [item for item in inventory if dataset in item["datasets"]]
        quality[dataset] = {**values,
            "category_model_counts": dict(sorted(Counter(item["attention_style"] for item in seen).items())),
            "attention_style_unreviewed_models": sorted(item["model"] for item in seen if item["attention_style"] in ("unknown", "attention_unspecified")),
        }
    return {
        "schema_version": 1, "end_date": end_date, "labels": LABELS.copy(),
        "daily": daily, "windows": windows, "monthly": monthly.get("text", []),
        "monthly_by_dataset": monthly, "inventory": inventory, "coverage": coverage,
        "quality": quality, "historical_text_models": base["historical_text_models"],
        "inventory_size": len(inventory), "current_catalog_size": base["current_catalog_size"],
        "attention_registry_size": len(attention_registry),
        "methodology": {**base["methodology"],
            "attention_style": "Mutually exclusive primary pattern: recurrent hybrids first; sparse selection before local/global before full latent-KV before full attention. Unreviewed regular-attention models remain style-unresolved.",
            "kv_representation": "MHA, GQA, MQA and MLA describe KV representation, not whether the attention mask is sparse. MoE describes expert routing; FlashAttention describes an implementation.",
            "style_growth": "All shares include unresolved models and unnamed tail in the denominator. No category is silently reclassified from a family name.",
        },
    }


def _chart(rows, categories, chart_id, title, value_key="shares"):
    if not rows:
        return '<p class="note">No observations available for this chart.</p>'
    width, height, left, right, top, bottom = 1000, 350, 68, 28, 24, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [[row.get(value_key, {}).get(key) if key in row.get("observed_categories", []) else None for row in rows] for key in categories]
    percent = value_key == "shares"
    maximum = max([value for series in values for value in series if _number(value)] or [1])
    maximum = max(5, math.ceil(maximum / 5) * 5) if percent else max(1, maximum * 1.08)
    x = lambda i: left + plot_w * i / max(1, len(rows) - 1)
    y = lambda value: top + plot_h * (1 - value / maximum)
    fragments = []
    for tick in range(5):
        value = maximum * tick / 4
        label = f"{value:g}%" if percent else _tokens(value)
        yy = y(value)
        fragments.append(f'<line class="gridline" x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" /><text class="axis-label" x="{left-10}" y="{yy+4:.2f}" text-anchor="end">{label}</text>')
    step = max(1, math.ceil(len(rows) / 8))
    for i, row in enumerate(rows):
        if i % step == 0 or i == len(rows) - 1:
            marker = "*" if row.get("partial_month") or not row.get("complete") else ""
            fragments.append(f'<text class="axis-label" x="{x(i):.2f}" y="{height-25}" text-anchor="middle">{_escape(row["month"])}{marker}</text>')
    for key, series in zip(categories, values):
        runs, run = [], []
        for i, value in enumerate(series):
            if not _number(value):
                if run:
                    runs.append(run)
                run = []
            else:
                run.append((i, value))
        if run:
            runs.append(run)
        for run in runs:
            points = " ".join(f"{x(i):.2f},{y(value):.2f}" for i, value in run)
            fragments.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[key]}" stroke-width="3" />')
            for i, value in run:
                formatted = f"{value:.2f}%" if percent else _tokens(value)
                tooltip = f'{rows[i]["month"]}: {LABELS[key]} {formatted}; {rows[i]["days"]} observed days'
                fragments.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="3.6" fill="{COLORS[key]}"><title>{_escape(tooltip)}</title></circle>')
    legend = "".join(f'<li><span class="swatch" style="background:{COLORS[key]}"></span>{LABELS[key]}</li>' for key in categories)
    cells = []
    for i, row in enumerate(rows):
        cells.append(f'<tr><th scope="row">{row["month"]}</th><td>{row["days"]}</td>' + "".join('<td>' + (_metric(series[i], "%", digits=2) if percent else _tokens(series[i])) + '</td>' for series in values) + '</tr>')
    table = '<table><thead><tr><th>Month</th><th>Observed days</th>' + ''.join(f'<th>{LABELS[key]}</th>' for key in categories) + '</tr></thead><tbody>' + ''.join(cells) + '</tbody></table>'
    return f'''<figure><div class="chart-scroll"><svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">
    <title id="{chart_id}-title">{_escape(title)}</title><desc id="{chart_id}-desc">Monthly observed traffic. Missing categories are gaps, not measured zero. Exact values follow in a data table.</desc>{''.join(fragments)}</svg></div>
    <figcaption><ul class="legend">{legend}</ul><p class="note">* Partial or incomplete calendar month. Points use observed days only; gaps mean no individually observed category traffic.</p></figcaption></figure>
    <details><summary>Chart data table</summary><div class="table-wrap" tabindex="0">{table}</div></details>'''


def _comparison_table(analysis):
    periods = analysis["windows"]["text"]
    current = periods["30"]["current"]
    rows = []
    for key, label in LABELS.items():
        observed = key in current["observed_categories"]
        share = _metric(current["shares"][key], "%", digits=2) if observed else "Not individually observed"
        growth = "".join(f'<td>{_metric(periods[str(days)]["growth_percent"][key], "%", True)}</td>' for days in (90, 30, 7))
        rows.append(f'<tr><th scope="row">{label}</th><td>{share}</td>{growth}<td>{_metric(periods["30"]["share_change_pp"][key], " pp", True, 2)}</td></tr>')
    rows.append('<tr class="total"><th scope="row">All reported text tokens</th><td>100%</td>' + ''.join(f'<td>{_metric(periods[str(days)]["platform_growth_percent"], "%", True)}</td>' for days in (90, 30, 7)) + '<td>—</td></tr>')
    dates = []
    for days, pair in periods.items():
        cur, prev = pair["current"], pair["previous"]
        note = "" if pair["comparable"] else "; incomplete dates, growth suppressed"
        dates.append(f'<li>{days} days: {cur["start"]}–{cur["end"]} versus {prev["start"]}–{prev["end"]}{note}</li>')
    return '''<div class="table-wrap" tabindex="0" role="region" aria-label="Attention-style growth comparisons"><table><thead><tr><th scope="col">Attention style</th><th scope="col">Latest 30-day share</th><th scope="col">90-day token growth</th><th scope="col">30-day token growth</th><th scope="col">7-day token growth</th><th scope="col">30-day share change</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div><p class="note">Each growth column compares consecutive equal-length windows. Share change is in percentage points (pp). A dash means missing dates or insufficient observations; it does not mean no growth.</p><details><summary>Exact comparison dates</summary><ul>''' + ''.join(dates) + '</ul></details>'


def _inventory(analysis):
    rows = []
    for item in sorted(analysis["inventory"], key=lambda item: (-(item.get("tokens_30d") or 0), item["model"])):
        style = item["attention_style"]
        links = [_safe_link(url) for url in item["attention_sources"]]
        status = "Current catalog" if item["current_catalog"] else "Historical / reviewed model"
        if not item["observed_text"]:
            status += "; not individually observed in text history"
        amount = _tokens(item["tokens_30d"]) if item["observed_30d"] else "Not observed"
        rows.append(f'''<tr data-category="{style}"><th scope="row"><span class="model-name">{_escape(item["name"])}</span><code>{_escape(item["model"])}</code><span class="muted model-status">{status}</span></th><td>{_escape(item["maker"])}</td><td>{LABELS[style]}<br /><span class="muted">{_escape(item["attention_confidence"])}</span></td><td>{_escape(item["kv_representation"])}</td><td>{amount}</td><td class="evidence">{_escape(item["attention_notes"])}<div class="source-list">{' · '.join(filter(None, links))}</div></td></tr>''')
    options = ''.join(f'<option value="{key}">{label}</option>' for key, label in LABELS.items() if key != "other")
    return f'''<div class="filters"><label>Search models, makers or evidence<input type="search" data-model-search placeholder="e.g. sparse, Qwen, MLA, Mamba" /></label><label>Attention style<select data-category-filter><option value="">All styles</option>{options}</select></label></div>
    <p class="note" data-result-count aria-live="polite">{len(rows):,} model variants shown.</p><div class="table-wrap model-table-wrap" tabindex="0" role="region" aria-label="Attention-style model inventory"><table data-model-table><thead><tr><th scope="col">Model / API variant</th><th scope="col">Maker / namespace</th><th scope="col">Primary attention style</th><th scope="col">KV representation</th><th scope="col">Observed 30-day tokens</th><th scope="col">Evidence and notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>'''


def render_attention(analysis, provenance):
    current = analysis["windows"]["text"]["30"]["current"]
    monthly = analysis["monthly"]
    coverage = analysis["coverage"]["text"]
    observed = [key for key in REFINED if key in current["observed_categories"]]
    leader = max(observed, key=lambda key: current["tokens"][key]) if observed else None
    finding = (f'<strong>{LABELS[leader]}</strong> has the largest identified 30-day share: {_metric(current["shares"][leader], "%")} of all reported text tokens.' if leader else "No refined attention style is individually observed in the latest 30 days.")
    pair = analysis["windows"]["text"]["30"]
    changes = [(value, key) for key, value in pair["share_change_pp"].items() if key in REFINED and _number(value)]
    if changes:
        change, key = max(changes)
        finding += f' The largest observed 30-day share change is {LABELS[key].lower()} at {_metric(change, " pp", True, 2)}.'
    cards = _stat(f'{analysis["historical_text_models"]:,}', "Historical text variants", f'All downloaded text history through {_escape(analysis["end_date"])}; not a hand-picked leaderboard.')
    cards += _stat(_metric(coverage["refined_share_percent"], "%"), "Traffic with an identified style", "Latest 30 days; every reported token stays in the denominator.")
    missing = None if not current["total"] else 100.0 * (current["tokens"]["attention_unspecified"] + current["tokens"]["unknown"] + current["tokens"]["other"]) / current["total"]
    cards += _stat(_metric(missing, "%"), "Style unresolved or unnamed", "Confirmed attention without a known mask, undisclosed models, and the unnamed tail.")
    core_chart = [key for key in SOFTMAX if any(key in row["observed_categories"] for row in monthly)]
    hybrid_chart = [key for key in HYBRIDS if any(key in row["observed_categories"] for row in monthly)]
    glossary = '''<div class="table-wrap" tabindex="0"><table><thead><tr><th>Style</th><th>What changes</th><th>Implication for retained state</th></tr></thead><tbody>
    <tr><th scope="row">Full attention</th><td>Each layer can attend to the full causal token history. MHA, GQA or MQA may be used.</td><td>KV history generally grows with retained context; GQA/MQA reduce KV head count.</td></tr>
    <tr><th scope="row">Full attention with latent KV</th><td>Attention uses a compressed latent KV representation without documented sparse token selection.</td><td>Compression changes bytes per token, not necessarily the number of retained tokens.</td></tr>
    <tr><th scope="row">Sparse / selected-token attention</th><td>Layers select a subset of previous tokens or blocks. This includes sparse latent-attention designs.</td><td>Fewer attended tokens do not automatically mean less stored history; selectors may still need retained KV.</td></tr>
    <tr><th scope="row">Local / sliding-window attention</th><td>Some or all layers limit their attention window, often alternating with global layers.</td><td>Local layers can have bounded live KV; global layers can still retain growing history.</td></tr>
    <tr><th scope="row">Hybrid linear attention</th><td>Linear or recurrent attention layers are mixed with softmax-attention layers.</td><td>Recurrent state and attention KV coexist; reusable checkpoints may add storage.</td></tr>
    <tr><th scope="row">Hybrid Mamba</th><td>Mamba state-space layers are mixed with attention layers.</td><td>State-space state and attention KV coexist; retention policy determines checkpoint volume.</td></tr>
    <tr><th scope="row">Pure recurrent / other hybrids</th><td>Pure linear, pure Mamba and other sequence mixers retain their own evidence-based categories.</td><td>Small live state does not establish small retained checkpoint storage.</td></tr>
    </tbody></table></div>'''
    content = f'''<section class="panel lead-panel"><h2>Which attention styles are gaining usage?</h2><p class="finding">{finding}</p>
    <p>The <a href="hybrid-trends.html">hybrid trends page</a> compares broad sequence mixers. This page opens up its regular-attention group into full, latent, sparse and local styles, while keeping linear and Mamba hybrids visible.</p>
    <p class="note">These are observed OpenRouter usage trends, not a benchmark of model quality or proof that one mechanism causes adoption.</p></section><div class="grid stats">{cards}</div>
    <section class="panel"><h2>Full, sparse and windowed attention over time</h2><p>Monthly token share uses all reported text tokens, including unresolved models and the unnamed tail. A sparse MoE is not automatically sparse attention.</p>{_chart(monthly, core_chart, "attention-styles", "Monthly share of identified softmax attention styles") if core_chart else '<p>No sufficiently refined softmax attention styles were individually observed.</p>'}</section>
    <section class="panel"><h2>Linear, Mamba and other recurrent styles</h2><p>The same denominator lets you compare these shares with the attention chart above. Hybrid layers keep the model in its recurrent-hybrid category even when its attention layers use a sparse or local mask.</p>{_chart(monthly, hybrid_chart, "attention-hybrids", "Monthly share of recurrent and hybrid styles") if hybrid_chart else '<p>No recurrent or hybrid style was individually observed; this does not establish zero usage.</p>'}</section>
    <section class="panel"><h2>How quickly is each style changing?</h2><p>Token growth shows observed volume change. Share change shows whether that style gained ground within this dataset as total traffic changed.</p>{_comparison_table(analysis)}</section>
    <section class="panel"><h2>What the styles mean for KV cache</h2>{glossary}<p class="note">These are mutually exclusive primary-pattern bins. The priority is recurrent hybrid, then sparse selection, then local/global windows, then full latent-KV, then full attention. Secondary mechanisms remain in model notes. <strong>MHA/GQA/MQA/MLA describe KV representation</strong>; MoE describes expert routing. FlashAttention is an attention implementation and does not by itself establish a sparse mask.</p>
    <p>Token share alone cannot predict cache hits or NVMe bytes. Use the <a href="cache-telemetry.html">cache dashboard</a> for available pricing signals and the <a href="hybrid-checkpoints.html">checkpoint model</a> for explicit storage assumptions.</p></section>
    <section class="panel"><h2>How much of the picture is still missing?</h2><p>Known attention with an unresolved mask is distinct from an undisclosed architecture. The unnamed tail is yet another limit: its model identities are unavailable.</p>{_chart(monthly, ["attention_unspecified", "unknown", "other"], "attention-coverage", "Monthly unresolved attention styles and unnamed traffic")}</section>
    <section class="panel"><h2>Every model and its evidence</h2><p>The inventory retains all models from the current catalog, downloaded history and reviewed architecture registry, including retired models and unresolved entries. Search or choose a style to inspect its evidence.</p><details><summary>Open the complete model inventory ({analysis["inventory_size"]:,} variants)</summary>{_inventory(analysis)}</details>
    <p class="note">Free/batch API variants remain separate traffic rows. Only exact reviewed IDs and documented serving variants share classification evidence. New releases never inherit a style solely from a family name. KV representation is unresolved when the available evidence does not establish it.</p></section>
    <section class="panel"><h2>Reading the trends responsibly</h2><ul><li>The API exposes daily top-50 model rows plus an unnamed remainder. Named category totals are observed minimums, not a complete census; an absent model or style is not measured zero demand.</li><li>Growth of an observed minimum is not a lower bound on true growth. Entering or leaving the top 50 changes visibility, and releases or free endpoints can dominate a category.</li><li>Growth requires complete consecutive 7-, 30- or 90-day windows and observations of that style in both windows. Monthly charts retain available days and mark incomplete months.</li><li>Counts mix input and output tokens with provider-specific token accounting. Rankings exclude private and zero-data-retention traffic; they do not measure the whole inference market.</li><li>Evidence comes from model configurations, cards and technical reports. The <a href="data/openrouter/attention-registry.json" download>attention-style registry</a> is this project's classification, not an OpenRouter field. Refreshing data preserves unknowns until evidence is reviewed.</li></ul></section>'''
    return _page("Attention trends", "Follow full, sparse, windowed and latent attention alongside linear and Mamba hybrids across OpenRouter's model history.", content, provenance, analysis, "attention-trends.html")
