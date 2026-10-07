# SPDX-License-Identifier: MIT
"""Render the public OpenRouter analysis without external Python dependencies.

The pages deliberately show observed category floors, missing evidence and the
unclassified tail. A token count is never converted into a cache or I/O count.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from html import escape
import math
from urllib.parse import urlparse


LABELS = {
    "regular_attention": "Regular attention (including sparse)",
    "hybrid_linear": "Hybrid linear attention",
    "hybrid_mamba": "Hybrid Mamba",
    "pure_linear": "Pure linear / recurrent attention",
    "pure_mamba": "Pure Mamba",
    "other_hybrid": "Other sequence-mixer hybrids",
    "unknown": "Architecture unresolved",
    "other": "Unnamed top-50 tail",
    "non_generation": "Non-generation models",
}
COLORS = {
    "regular_attention": "#66b9ff",
    "hybrid_linear": "#50e3c2",
    "hybrid_mamba": "#c398ff",
    "pure_linear": "#f7cd70",
    "pure_mamba": "#ff98be",
    "other_hybrid": "#ffc183",
    "unknown": "#aebbd8",
    "other": "#f2a274",
}
MAIN = ("regular_attention", "hybrid_linear", "hybrid_mamba")
NAV = (
    ("index.html", "All visualizations"),
    ("hybrid-trends.html", "Architecture trends"),
    ("context-demand.html", "Context demand"),
    ("cache-telemetry.html", "Cache telemetry"),
    ("hybrid-checkpoints.html", "Hybrid checkpoints"),
    ("io-projections.html", "I/O projections"),
)


def _escape(value):
    return escape(str(value), quote=True)


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _metric(value, suffix="", signed=False, digits=1):
    if not _number(value):
        return '<span class="muted" title="Insufficient comparable observations">—</span>'
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text + suffix


def _tokens(value):
    if not _number(value):
        return "—"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f}{suffix}"
    return f"{value:,.0f}"


def _safe_link(url, label=None):
    if not isinstance(url, str) or urlparse(url).scheme not in ("https", "http"):
        return ""
    text = label or urlparse(url).netloc
    return f'<a href="{_escape(url)}" rel="noopener noreferrer">{_escape(text)}</a>'


def _as_of(provenance, analysis=None):
    explicit = provenance.get("as_of") if isinstance(provenance, dict) else None
    metadata = (analysis or {}).get("metadata", {})
    dates = [v.get("as_of") for v in metadata.values() if isinstance(v, dict) and v.get("as_of")]
    if explicit:
        return str(explicit)
    if dates:
        return " / ".join(sorted(set(dates)))
    return "See the source manifest for each response's as_of timestamp"


def _page(title, description, content, provenance, analysis=None, active=""):
    catalog_only = active == "cache-telemetry.html"
    analysis_url = provenance.get("analysis_url", "data/openrouter/cache-analysis.json" if catalog_only else "data/openrouter/analysis.json")
    provenance_url = provenance.get("provenance_url", "data/openrouter/cache-provenance.json" if catalog_only else "data/openrouter/provenance.json")
    if catalog_only:
        metadata = (analysis or {}).get("metadata", {})
        source_url = metadata.get("source_url", "https://openrouter.ai/api/v1/models?output_modalities=all")
        attribution = (
            '<p><strong>Source and attribution:</strong> '
            + _safe_link(source_url, "OpenRouter public Models API")
            + '. Catalog metadata and advertised prices are provided by OpenRouter. '
            'Calculations and presentation are by kvcache-view; no endorsement is implied. '
            'The CC BY 4.0 license for OpenRouter rankings datasets is not asserted '
            'as a license for this catalog metadata.</p>'
        )
        source_time = metadata.get("as_of") or provenance.get("fetched_at") or provenance.get("as_of") or "Not recorded"
        source_time_label = "Catalog fetched"
        extra_download = ""
        documentation = '<a href="https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties">Models API documentation</a>'
    else:
        attribution = f'''<p>Source: OpenRouter (<a href="https://openrouter.ai/rankings">openrouter.ai/rankings</a>),
            as of {_escape(_as_of(provenance, analysis))}. Licensed under
            <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
            Architecture classification, calculations and visualizations are modifications by kvcache-view;
            they are not OpenRouter's architectural classifications or an endorsement.</p>'''
        source_time = _as_of(provenance, analysis)
        source_time_label = "Latest source as_of (per-response timestamps in the manifest)"
        extra_download = '<a href="data/openrouter/architecture-registry.json" download>Architecture registry</a> ·'
        documentation = '<a href="https://openrouter.ai/docs/api/api-reference/datasets/daily-token-totals-for-top-50-models">Dataset API documentation</a>'
    nav = "".join(
        f'<a href="{url}"' + (' aria-current="page"' if url == active else "")
        + f'>{label}</a>' for url, label in NAV
    )
    return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="{_escape(description)}" />
    <meta name="theme-color" content="#1428a0" />
    <title>{_escape(title)} | KV Cache Visualizations</title>
    <link rel="icon" href="icon-192.png" />
    <link rel="stylesheet" href="openrouter.css" />
    <script src="openrouter.js" defer></script>
</head>
<body>
    <a class="skip-link" href="#main">Skip to content</a>
    <div class="container">
        <nav aria-label="Visualization navigation">{nav}</nav>
        <header>
            <p class="eyebrow">OpenRouter observatory · KV Cache Visualizations</p>
            <h1>{_escape(title)}</h1>
            <p class="lede">{_escape(description)}</p>
        </header>
        <main id="main">{content}</main>
        <footer>
            {attribution}
            <p class="source-time"><strong>{source_time_label}:</strong> {_escape(source_time)}.</p>
            <p><a href="{_escape(analysis_url)}" download>Download analysis</a> ·
            {extra_download}
            <a href="{_escape(provenance_url)}" download>Source manifest</a> ·
            {documentation}</p>
            <p class="muted">Refresh all OpenRouter pages with <code>make openrouter</code>.
            This is a static snapshot; refreshing this page does not fetch new data.</p>
        </footer>
    </div>
</body>
</html>'''


def _period(analysis, dataset="text", days="30"):
    return analysis.get("windows", {}).get(dataset, {}).get(str(days), {})


def _observed(period, category):
    if "observed_categories" in period:
        return category in period["observed_categories"]
    return period.get("tokens", {}).get(category, 0) > 0


def _summary(analysis):
    comparisons = analysis.get("windows", {}).get("text", {})
    latest = comparisons.get("30", {}).get("current", {})
    available = [(latest.get("tokens", {}).get(c, 0), c) for c in MAIN]
    largest, cat = max(available)
    lines = []
    if largest:
        share = latest.get("shares", {}).get(cat)
        lines.append(f"<strong>{LABELS[cat]}</strong> is the largest identified category in the latest 30-day window"
                     + (f", at {_metric(share, '%')} of all reported text tokens." if share is not None else "."))
    for days in ("90", "7"):
        pair = comparisons.get(days, {})
        linear = pair.get("growth_percent", {}).get("hybrid_linear")
        mamba = pair.get("growth_percent", {}).get("hybrid_mamba")
        if _number(linear) and _number(mamba):
            relation = "higher" if linear > mamba else "lower" if linear < mamba else "equal"
            lines.append(f"Over {days} days, linear-hybrid token growth was {relation} than Mamba-hybrid growth"
                         f" ({_metric(linear, '%', True)} versus {_metric(mamba, '%', True)}).")
    if not lines:
        lines.append("The available observations do not yet support a category-growth comparison.")
    return " ".join(lines)


def _stat(value, title, detail=""):
    return f'<div class="stat"><span class="stat-value">{value}</span><strong>{_escape(title)}</strong><p>{detail}</p></div>'


def _comparison_table(analysis, dataset="text", all_categories=True):
    categories = list(LABELS) if all_categories else list(MAIN)
    thirty = _period(analysis, dataset, "30").get("current", {})
    rows = []
    for cat in categories:
        observed = _observed(thirty, cat)
        share = _metric(thirty.get("shares", {}).get(cat), "%", digits=2) if observed else "Not individually observed"
        growth = [_period(analysis, dataset, days).get("growth_percent", {}).get(cat) for days in ("90", "30", "7")]
        delta = _period(analysis, dataset, "30").get("share_change_pp", {}).get(cat)
        if not observed and cat in ("non_generation", "other_hybrid") and all(x is None for x in growth):
            continue
        rows.append(f'<tr><th scope="row">{LABELS[cat]}</th><td>{share}</td>'
                    + "".join(f'<td>{_metric(value, "%", True)}</td>' for value in growth)
                    + f'<td>{_metric(delta, " pp", True, 2)}</td></tr>')
    rows.append('<tr class="total"><th scope="row">All reported tokens in this dataset</th><td>100%</td>'
                + "".join(f'<td>{_metric(_period(analysis, dataset, d).get("platform_growth_percent"), "%", True)}</td>' for d in ("90", "30", "7"))
                + '<td>—</td></tr>')
    periods = []
    for days in ("7", "30", "90"):
        value = _period(analysis, dataset, days)
        current, previous = value.get("current", {}), value.get("previous", {})
        if current:
            flag = "" if value.get("comparable", True) else " (incomplete; growth suppressed)"
            periods.append(f'{days} days: {_escape(current.get("start", "?"))}–{_escape(current.get("end", "?"))}'
                           f' versus {_escape(previous.get("start", "?"))}–{_escape(previous.get("end", "?"))}{flag}')
    return f'''<div class="table-wrap" tabindex="0" role="region" aria-label="{_escape(dataset)} architecture growth comparisons">
    <table><thead><tr><th scope="col">Architecture</th><th scope="col">Latest 30-day share</th>
    <th scope="col">90-day token growth</th><th scope="col">30-day token growth</th><th scope="col">7-day token growth</th>
    <th scope="col">30-day share change</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p class="note">Growth compares consecutive equal-length windows, not growth since release.
    A dash means no defensible comparison: missing dates, no prior observations or an unobserved category.
    Absent categories can still have traffic below the top-50 cutoff.</p>
    <details><summary>Exact comparison dates</summary><ul>{''.join('<li>'+x+'</li>' for x in periods)}</ul></details>'''


def _month_partial(row):
    try:
        year, month = map(int, row["month"].split("-"))
        expected = calendar.monthrange(year, month)[1]
        return row.get("days", expected) < expected
    except (KeyError, TypeError, ValueError):
        return False


def _chart(rows, categories, chart_id, title, value_key="shares", unit="%"):
    if not rows:
        return '<p class="note">No observations available for this chart.</p>'
    width, height = 1000, 350
    left, right, top, bottom = 65, 30, 24, 65
    chart_w, chart_h = width - left - right, height - top - bottom
    data = [[row.get(value_key, {}).get(cat) for row in rows] for cat in categories]
    max_value = max([v for series in data for v in series if _number(v)] or [1])
    max_value = max(5, math.ceil(max_value / 5) * 5) if unit == "%" else max(1, max_value * 1.08)
    x = lambda i: left + i * chart_w / max(1, len(rows) - 1)
    y = lambda value: top + chart_h * (1 - value / max_value)
    fragments = []
    for i in range(5):
        v = max_value * i / 4
        yy = y(v)
        label = f"{v:g}%" if unit == "%" else _tokens(v)
        fragments.append(f'<line class="gridline" x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" />'
                         f'<text class="axis-label" x="{left-10}" y="{yy+4:.2f}" text-anchor="end">{label}</text>')
    label_every = max(1, math.ceil(len(rows) / 10))
    for i, row in enumerate(rows):
        if i % label_every == 0 or i == len(rows) - 1:
            label = row.get("month", row.get("date", ""))
            marker = "*" if _month_partial(row) else ""
            fragments.append(f'<text class="axis-label" x="{x(i):.2f}" y="{height-30}" text-anchor="middle">{_escape(label)}{marker}</text>')
    for cat, values in zip(categories, data):
        paths, current = [], []
        for i, val in enumerate(values):
            if not _number(val):
                if current:
                    paths.append(current)
                current = []
                continue
            current.append((i, val))
        if current:
            paths.append(current)
        color = COLORS.get(cat, "#fff")
        for points in paths:
            coords = " ".join(f'{x(i):.2f},{y(v):.2f}' for i, v in points)
            fragments.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" />')
            for i, val in points:
                row = rows[i]
                formatted = f"{val:.2f}%" if unit == "%" else _tokens(val)
                title_value = f'{row.get("month", "")}: {LABELS.get(cat, cat)} {formatted}; {row.get("days", "?")} observed days'
                fragments.append(f'<circle cx="{x(i):.2f}" cy="{y(val):.2f}" r="3.6" fill="{color}"><title>{_escape(title_value)}</title></circle>')
    legend = "".join(f'<li><span class="swatch" style="background:{COLORS.get(cat,"#fff")}"></span>{LABELS.get(cat,cat)}</li>' for cat in categories)
    accessible_rows = []
    for row in rows:
        accessible_rows.append(f'<tr><th scope="row">{_escape(row.get("month", ""))}{"*" if _month_partial(row) else ""}</th><td>{row.get("days", "—")}</td>'
                               + "".join('<td>' + (_metric(row.get(value_key, {}).get(c), "%", digits=2) if unit == "%" else _tokens(row.get(value_key, {}).get(c))) + '</td>' for c in categories) + '</tr>')
    table = '<table><thead><tr><th>Month</th><th>Observed days</th>' + ''.join(f'<th>{LABELS.get(c,c)}</th>' for c in categories) + '</tr></thead><tbody>' + ''.join(accessible_rows) + '</tbody></table>'
    return f'''<figure><div class="chart-scroll"><svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">
    <title id="{chart_id}-title">{_escape(title)}</title>
    <desc id="{chart_id}-desc">Monthly observations. The data table below provides exact values; missing category observations are gaps.</desc>{''.join(fragments)}</svg></div>
    <figcaption><ul class="legend">{legend}</ul><p class="note">* An incomplete calendar month, including a month with missing dates. Values use only observed days. Hover over a point for its value.</p></figcaption></figure>
    <details><summary>Chart data table</summary><div class="table-wrap" tabindex="0">{table}</div></details>'''


def _monthly_for(analysis, dataset):
    if dataset == "text" and isinstance(analysis.get("monthly"), list):
        rows = []
        for row in analysis["monthly"]:
            row = dict(row)
            # An unobserved category is unknown, not a measured zero.
            row["shares"] = {k: v for k, v in row.get("shares", {}).items() if row.get("tokens", {}).get(k, 0) > 0}
            rows.append(row)
        return rows
    months = defaultdict(lambda: {"days": 0, "tokens": defaultdict(int)})
    for day, counts in sorted(analysis.get("daily", {}).get(dataset, {}).items()):
        group = months[day[:7]]
        group["days"] += 1
        for cat, value in counts.items():
            group["tokens"][cat] += value
    rows = []
    for month, group in sorted(months.items()):
        total = sum(group["tokens"].values())
        rows.append({"month": month, "days": group["days"], "tokens": dict(group["tokens"]), "total": total,
                     "shares": {k: v / total * 100 for k, v in group["tokens"].items() if v > 0 and total},
                     "mean_daily_tokens": {k: v / group["days"] for k, v in group["tokens"].items() if v > 0}})
    return rows


def _inventory(analysis):
    values = analysis.get("inventory", [])
    if isinstance(values, dict):
        values = [dict(value, model=key) for key, value in values.items()]
    rows = []
    for row in sorted(values, key=lambda r: (-(r.get("tokens_30d") or 0), r.get("model", ""))):
        model = row.get("model", "")
        category = row.get("category", "unknown")
        sources = row.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        links = []
        for source in sources:
            if isinstance(source, dict):
                links.append(_safe_link(source.get("url", ""), source.get("title")))
            else:
                links.append(_safe_link(source))
        maker = row.get("maker") or model.split("/", 1)[0]
        status = "Current catalog" if row.get("current_catalog") else "Historical / not in current catalog"
        if not row.get("observed_text"):
            status += "; not individually observed in text history"
        amount = row.get("tokens_30d")
        tokens = _tokens(amount) if _number(amount) else '<span class="muted">Not observed</span>'
        rows.append(f'''<tr data-category="{_escape(category)}"><th scope="row"><span class="model-name">{_escape(row.get("name") or model)}</span><code>{_escape(model)}</code><span class="muted model-status">{_escape(status)}</span></th>
        <td>{_escape(maker)}</td><td>{_escape(LABELS.get(category, category))}<br /><span class="muted">{_escape(row.get("confidence", "unresolved"))}</span></td>
        <td>{tokens}</td><td class="evidence">{_escape(row.get("notes", ""))}<div class="source-list">{' · '.join(filter(None, links))}</div></td></tr>''')
    if not rows:
        return '<p>No model inventory has been generated. See the downloadable registry.</p>'
    options = ''.join(f'<option value="{k}">{v}</option>' for k, v in LABELS.items() if k != "other")
    return f'''<div class="filters"><label>Search models, makers or evidence<input type="search" data-model-search placeholder="e.g. Qwen, Mamba, sparse" /></label>
    <label>Architecture<select data-category-filter><option value="">All architectures</option>{options}</select></label></div>
    <p class="note" data-result-count aria-live="polite">{len(rows)} model variants shown.</p>
    <div class="table-wrap model-table-wrap" tabindex="0" role="region" aria-label="Model architecture registry"><table data-model-table>
    <thead><tr><th scope="col">Model / API variant</th><th scope="col">Maker / namespace</th><th scope="col">Architecture / confidence</th><th scope="col">Observed 30-day tokens</th><th scope="col">Evidence and notes</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <p class="note">Catalog variants and historical aliases are retained. A free or batch API variant is a traffic row, not automatically a distinct checkpoint.
    Unresolved architectures are never inferred from a familiar family name. Maker defaults to the API namespace when no verified display name is recorded.</p>'''


def _concentration(analysis):
    current = _period(analysis).get("current", {})
    rows = []
    for cat in MAIN:
        leaders = analysis.get("leaders_30d", {}).get(cat, [])
        total = current.get("tokens", {}).get(cat, 0)
        if not total or not leaders:
            continue
        leader = max(leaders, key=lambda x: x.get("tokens", 0))
        free = sum(x.get("tokens", 0) for x in leaders if x.get("model", "").endswith(":free"))
        rows.append(f'<tr><th scope="row">{LABELS[cat]}</th><td><code>{_escape(leader.get("model", ""))}</code></td>'
                    f'<td>{_metric(leader.get("tokens", 0) / total * 100, "%")}</td><td>{_metric(free / total * 100, "%")}</td></tr>')
    return '<div class="table-wrap" tabindex="0"><table><thead><tr><th>Architecture</th><th>Largest observed API variant</th><th>Share within category</th><th>Observed :free share</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def render_trends(analysis: dict, provenance: dict) -> str:
    current = _period(analysis).get("current", {})
    monthly = _monthly_for(analysis, "text")
    main_categories = list(MAIN)
    main_categories += [c for c in ("pure_linear", "pure_mamba", "other_hybrid") if any(r.get("tokens", {}).get(c, 0) for r in monthly)]
    first = monthly[0]["month"] if monthly else "unavailable"
    last = current.get("end", "unavailable")
    daily_count = len(analysis.get("daily", {}).get("text", {}))
    tail = current.get("shares", {}).get("other")
    coverage = 100 - tail if _number(tail) else None
    cards = _stat(f'{analysis.get("historical_text_models", 0):,}', "Historical text variants", f'{daily_count:,} observed days; {first} through {_escape(last)}.')
    cards += _stat(_metric(coverage, "%"), "Individually named traffic", "Latest 30 days; the remainder is an unnamed tail.")
    cards += _stat(_metric(current.get("shares", {}).get("unknown"), "%"), "Architecture unresolved", "Named traffic without sufficient public architectural evidence.")
    content = f'''<section class="panel lead-panel"><h2>What the current snapshot shows</h2><p class="finding">{_summary(analysis)}</p>
    <p class="note">These are usage trends within OpenRouter's public rankings, not quality rankings or evidence that an architecture caused adoption.</p></section>
    <div class="grid stats">{cards}</div>
    <section class="panel"><h2>Token growth and share</h2><p>All categories use the same denominator: reported text tokens, including unresolved architectures and the unnamed tail.</p>{_comparison_table(analysis)}</section>
    <section class="panel"><h2>Architecture share over the full observed history</h2>
    <p>Regular attention includes dense, local, sparse, sliding-window and compressed/latent attention. Linear hybrids and Mamba hybrids remain separate.</p>
    {_chart(monthly, main_categories, "architecture-history", "Monthly identified architecture share")}</section>
    <section class="panel"><h2>How much traffic cannot be assigned?</h2><p>The unresolved named models and the unnamed top-50 tail are different sources of uncertainty.</p>
    {_chart(monthly, ["unknown", "other"], "coverage-history", "Monthly unresolved architecture and unnamed tail shares")}</section>
    <section class="panel"><h2>Concentration and free variants</h2><p>A category's growth can be driven by a small number of releases, prices or free endpoints. This table describes individually observed traffic in the latest 30 days.</p>{_concentration(analysis)}</section>
    <section class="panel"><h2>Model-by-model evidence</h2><p>The inventory joins the current catalog with every model returned in the downloaded history, including retired models and aliases.</p>{_inventory(analysis)}</section>
    <section class="panel"><h2>What these numbers can support</h2><ul>
    <li>The API returns up to 50 named models per day and an aggregated remainder. Category totals are observed minimums, not complete category populations.</li>
    <li>The growth of an observed minimum is not a lower bound on true growth: entering or leaving the cutoff changes visibility. An absent pure-Mamba or pure-linear row does not establish zero usage.</li>
    <li>Growth is suppressed when a comparison has missing dates or a category has no observations in either period. Monthly charts retain available days and flag incomplete months.</li>
    <li>Tokens include input and output and use provider token accounting. Public rankings exclude private and zero-data-retention traffic; they do not represent the whole inference market.</li>
    <li>Architecture evidence is version-specific. New API identities remain unresolved until evidence is added to the registry. OpenRouter aliases are matched exactly; historical traffic rows are counted once.</li>
    <li>Token traffic does not measure KV bytes, cache hits, checkpoints, NVMe reads or restore latency. The <a href="context-demand.html">context-demand page</a> describes context slices; the <a href="cache-telemetry.html">cache-telemetry page</a> separates public cache metadata from actual offload measurements.</li>
    </ul></section>'''
    return _page("Hybrid architecture trends", "How hybrid Mamba, hybrid linear attention and regular attention usage change across OpenRouter's public model history.", content, provenance, analysis, "hybrid-trends.html")


def render_context(analysis: dict, provenance: dict) -> str:
    available = [key for key in analysis.get("windows", {}) if key not in ("all", "text")]
    available.sort(key=lambda key: (key != "100K", key != "1M", key))
    rows = []
    for dataset in ["text"] + available:
        period = _period(analysis, dataset)
        current = period.get("current", {})
        observed = 100 - current.get("shares", {}).get("other", 0) if current else None
        rows.append(f'<tr><th scope="row">{_escape("All text" if dataset == "text" else dataset)}</th><td>{_tokens(current.get("total"))}</td>'
                    f'<td>{_metric(period.get("platform_growth_percent"), "%", True)}</td><td>{_metric(observed, "%")}</td>'
                    f'<td>{_metric(current.get("shares", {}).get("unknown"), "%")}</td></tr>')
    content = '''<section class="panel lead-panel"><h2>Context demand is a workload signal</h2><p class="finding">The context filters reveal where token demand is moving. They do not report cache residency, retained checkpoint count or storage traffic.</p>
    <p>Labels below are the API's context-bucket labels. Their exact numerical boundaries are not assumed, and separate bucket totals are not added together as if they formed a verified partition.</p></section>'''
    content += '<section class="panel"><h2>Latest 30-day context totals</h2><div class="table-wrap" tabindex="0"><table><thead><tr><th>Dataset / API bucket</th><th>Reported tokens</th><th>30-day growth</th><th>Individually named</th><th>Architecture unresolved</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div><p class="note">Each filtered dataset has its own top 50 and its own unnamed tail. Architecture shares use that dataset\'s total, not the unfiltered total. T = trillion tokens.</p></section>'
    if not available:
        content += '<section class="panel"><p>No context-filtered datasets have been downloaded.</p></section>'
    for dataset in available:
        monthly = _monthly_for(analysis, dataset)
        safe_id = ''.join(ch if ch.isalnum() else '-' for ch in dataset)
        content += f'''<section class="panel"><p class="eyebrow">OpenRouter context filter: {_escape(dataset)}</p><h2>{_escape(dataset)} architecture mix and growth</h2>
        {_comparison_table(analysis, dataset, all_categories=False)}
        {_chart(monthly, list(MAIN), "context-"+safe_id, dataset+" monthly architecture share")}
        <h3>Observed tokens per day</h3><p>Monthly totals are divided by observed days, so a partial month is not compared with a full month's raw total.</p>
        {_chart(monthly, list(MAIN), "context-volume-"+safe_id, dataset+" mean daily identified category tokens", "mean_daily_tokens", "tokens")}</section>'''
    content += '''<section class="panel"><h2>Connecting demand to an NVMe model</h2><p>The next step needs measurements of reusable-prefix frequency, per-model state bytes, checkpoint spacing and retention, restoration count and bytes per restore. The context filter supplies none of these.</p>
    <p>Use the <a href="hybrid-checkpoints.html">hybrid checkpoint model</a> to examine retained recurrent states and attention KV, then the <a href="io-projections.html">I/O projections</a> to vary storage assumptions. Keep those scenarios separate from measured OpenRouter traffic.</p>
    <p class="note">A smaller live recurrent state does not establish a smaller retained checkpoint corpus. Conversely, growing long-context token usage does not by itself establish growing NVMe offload.</p></section>'''
    return _page("Context demand", "Track OpenRouter context-bucket traffic and architecture mix as inputs to workload planning, while keeping storage assumptions explicit.", content, provenance, analysis, "context-demand.html")
