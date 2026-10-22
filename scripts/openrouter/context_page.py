# SPDX-License-Identifier: MIT
"""An independently calculated companion to OpenRouter's cache research."""

import math

from .render import (COLORS, LABELS, MAIN, _chart, _comparison_table, _escape,
                     _metric, _monthly_for, _number, _page, _period, _tokens)

PALETTE = {"text": "#dbe5f5", "1K": "#c398ff", "10K": "#66b9ff",
           "100K": "#50e3c2", "1M": "#f7cd70", "10M": "#ff917b"}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
ARTICLE = "https://openrouter.ai/data/newsletter/cache-is-crucial"


def multiple(value):
    return _metric(value, "×")


def heading(number, title, description=""):
    return (f'<div class="section-heading"><span class="section-number">{number}</span>'
            f'<div><h2>{title}</h2><p>{description}</p></div></div>')


def trend_chart(study, mode):
    field = {"ratio": "share_of_text_percent", "growth": "index_january_100",
             "volume": "mean_daily_tokens"}[mode]
    datasets = [d for d in study.get("datasets", []) if mode != "ratio" or d["bucket"] != "text"]
    values = [m.get(field) for d in datasets for m in d["monthly"] if _number(m.get(field))]
    factor = 100 if mode == "growth" else 1
    maximum = max(values, default=1) / factor
    step = max(maximum / 4, .01)
    magnitude = 10 ** math.floor(math.log10(step))
    step = next((x * magnitude for x in (1, 2, 2.5, 5, 10) if x * magnitude >= step), 10 * magnitude)
    top = max(step, math.ceil(maximum / step) * step)
    width, height, left, right, ytop, bottom = 1040, 390, 85, 48, 36, 56
    plotw, ploth = width - left - right, height - ytop - bottom
    x = lambda i: left + i * plotw / 8
    y = lambda v: ytop + ploth * (1 - v / factor / top)
    title = {"ratio": "Bucket token volume relative to all text traffic",
             "growth": "Daily token volume relative to January 2026",
             "volume": "Average input + output tokens per day"}[mode]
    unit = {"ratio": "% of all text tokens", "growth": "January = 1×", "volume": "Input + output tokens per day"}[mode]
    parts = [f'<svg class="trend-chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="trend-{mode}-title">',
             f'<title id="trend-{mode}-title">{title}, January to September 2026</title>',
             f'<text x="{left}" y="17" class="axis-label">{unit}</text>']
    for tick in range(round(top / step) + 1):
        v = tick * step
        yy = ytop + ploth * (1 - v / top)
        label = _tokens(v) if mode == "volume" else f'{v:g}' + ("×" if mode == "growth" else "%")
        parts += [f'<line class="gridline" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>',
                  f'<text class="axis-label" x="{left-15}" y="{yy+5:.1f}" text-anchor="end">{label}</text>']
    for i, name in enumerate(MONTH_NAMES):
        parts.append(f'<text class="axis-label" x="{x(i):.1f}" y="{height-20}" text-anchor="middle">{name}</text>')
    for dataset in datasets:
        bucket, points, segments = dataset["bucket"], [], []
        for i, month in enumerate(dataset["monthly"]):
            v = month.get(field)
            if _number(v):
                points.append((i, v))
            elif points:
                segments.append(points)
                points = []
        if points:
            segments.append(points)
        parts.append(f'<g data-chart-series="{bucket}">')
        for segment in segments:
            coords = " ".join(f'{x(i):.1f},{y(v):.1f}' for i, v in segment)
            dash = ' stroke-dasharray="7 6"' if bucket == "text" else ""
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{PALETTE[bucket]}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
            for i, v in segment:
                label = _tokens(v) + " input + output tokens/day" if mode == "volume" else (f'{v/100:,.2f}× January' if mode == "growth" else f'{v:,.2f}% of all text tokens')
                desc = f'{MONTH_NAMES[i]} 2026 · {dataset["label"]} · {label}'
                parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="{PALETTE[bucket]}" stroke="#0b1327" stroke-width="2" tabindex="0" data-chart-point="{_escape(desc)}" aria-label="{_escape(desc)}"><title>{_escape(desc)}</title></circle>')
        parts.append('</g>')
    parts.append('</svg>')
    return ''.join(parts)


def comparison_table(study):
    rows = []
    for d in study.get("datasets", []):
        c = d["comparison"]
        a, b = c["baseline"], c["current"]
        label = "All text" if d["bucket"] == "text" else d["bucket"] + " bucket"
        rows.append(f'<tr><th scope="row"><span class="chart-key" style="--series:{PALETTE[d["bucket"]]}"></span>{label}</th>'
                    f'<td>{_tokens(a.get("mean_daily_tokens"))}</td><td>{_tokens(b.get("mean_daily_tokens"))}</td>'
                    f'<td>{multiple(c.get("daily_growth_multiple"))}</td>'
                    f'<td>{_metric(a.get("share_of_text_percent"), "%", digits=2)}</td><td>{_metric(b.get("share_of_text_percent"), "%", digits=2)}</td>'
                    f'<td>{_metric(c.get("share_of_text_change_pp"), " pp", signed=True, digits=2)}</td></tr>')
    return '<div class="table-wrap study-table" tabindex="0" role="region" aria-label="January and September comparison"><table><thead><tr><th scope="col">API dataset</th><th scope="col">Jan tokens/day</th><th scope="col">Sep tokens/day</th><th scope="col">Daily volume</th><th scope="col">Jan / all text</th><th scope="col">Sep / all text</th><th scope="col">Ratio change</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def bucket_cards(study):
    cards = []
    for d in study.get("datasets", []):
        if d["bucket"] == "text":
            continue
        c, bucket = d["comparison"], d["bucket"]
        a, b = c["baseline"], c["current"]
        bars = []
        for label, month in (("Jan", a), ("Sep", b)):
            share = month.get("share_of_text_percent")
            width = min(100, max(0, share)) if _number(share) else 0
            digits = 2 if _number(share) and share < 1 else 1
            bars.append(f'<div class="month-row"><span class="month-name">{label}</span><span class="month-track"><span style="width:{width:.3f}%;background:{PALETTE[bucket]}"></span></span><span class="month-value">{_metric(share, "%", digits=digits)}</span></div>')
        cards.append(f'<article class="bucket-card" style="--series:{PALETTE[bucket]}"><p class="eyebrow">{bucket} bucket</p><div class="bucket-value">{multiple(c.get("daily_growth_multiple"))}</div><p class="bucket-sub">September daily volume / January</p><div class="month-pair">{"".join(bars)}</div><p class="note">Bucket tokens / all text tokens</p></article>')
    return '<div class="bucket-cards">' + ''.join(cards) + '</div>'


def architecture(study):
    options, panels = [], []
    for d in study.get("datasets", []):
        bucket, c = d["bucket"], d["comparison"]
        selected = ' selected' if bucket == "1M" else ""
        options.append(f'<option value="{bucket}"{selected}>{"All text" if bucket == "text" else bucket + " bucket"}</option>')
        before = {a["category"]: a for a in c["baseline"].get("architecture", [])}
        after = {a["category"]: a for a in c["current"].get("architecture", [])}
        rows = []
        for category in LABELS:
            a, b = before.get(category, {}), after.get(category, {})
            if not a.get("observed") and not b.get("observed") and category not in (*MAIN, "unknown", "other"):
                continue
            av, bv = a.get("share_percent"), b.get("share_percent")
            change = bv - av if _number(av) and _number(bv) else None
            rows.append(f'<tr><th scope="row">{LABELS[category]}</th><td>{_metric(av, "%", digits=2)}</td><td>{_metric(bv, "%", digits=2)}</td><td>{_metric(change, " pp", True, 2)}</td></tr>')
        title = "All text" if bucket == "text" else bucket + " context bucket"
        unknown_before = c["baseline"].get("unknown_share_percent")
        unknown_after = c["current"].get("unknown_share_percent")
        coverage = f'Unresolved named traffic: {_metric(unknown_before, "%")} in January → {_metric(unknown_after, "%")} in September. Changes in visibility limit the architecture comparison.'
        panels.append(f'<div data-architecture-panel="{bucket}"><h3>{title}</h3><p class="study-boundary">{coverage}</p><div class="table-wrap" tabindex="0"><table><thead><tr><th>Architecture / visibility</th><th>January share</th><th>September share</th><th>Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')
    return '<label class="architecture-select" hidden>Choose a request-context dataset<select data-architecture-select>' + ''.join(options) + '</select></label>' + ''.join(panels)


def rolling_context(analysis):
    available = [b for b in ("1K", "10K", "100K", "1M", "10M") if b in analysis.get("windows", {})]
    rows = []
    for dataset in ["text"] + available:
        period = _period(analysis, dataset)
        current = period.get("current", {})
        tail = current.get("shares", {}).get("other")
        named = 100 - tail if _number(tail) else None
        rows.append(f'<tr><th>{"All text" if dataset == "text" else dataset}</th><td>{_tokens(current.get("total"))}</td><td>{_metric(period.get("platform_growth_percent"), "%", True)}</td><td>{_metric(named, "%")}</td><td>{_metric(current.get("shares", {}).get("unknown"), "%")}</td></tr>')
    end = _period(analysis).get("current", {}).get("end", "unavailable")
    html = f'<details class="study-details"><summary>Latest rolling snapshot · through {_escape(end)}</summary><p>These rolling windows update independently of the fixed January–September study above.</p><div class="table-wrap" tabindex="0"><table><thead><tr><th>Dataset / API bucket</th><th>30-day tokens</th><th>30-day growth</th><th>Individually named</th><th>Architecture unresolved</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    if not available:
        html += '<p>No context-filtered datasets have been downloaded.</p>'
    for dataset in available:
        monthly = _monthly_for(analysis, dataset)
        html += f'<details><summary>{dataset} architecture mix and growth · full available history</summary>{_comparison_table(analysis, dataset, all_categories=False)}{_chart(monthly, list(MAIN), "context-"+dataset, dataset+" monthly architecture share")}</details>'
    return html + '</details>'


def render_context_page(analysis, provenance):
    study = analysis.get("context_study", {})
    datasets = {d["bucket"]: d for d in study.get("datasets", [])}
    all_c = datasets.get("text", {}).get("comparison", {})
    million = datasets.get("1M", {}).get("comparison", {})
    ready = bool(all_c.get("comparable") and million.get("comparable"))
    if ready:
        a, b = million["baseline"], million["current"]
        deck = f'The <strong>1M request-context bucket</strong> grew {multiple(million.get("daily_growth_multiple"))} in daily input + output token volume. Overall text traffic grew {multiple(all_c.get("daily_growth_multiple"))}. Its ratio to all text traffic moved from {_metric(a.get("share_of_text_percent"), "%")} to {_metric(b.get("share_of_text_percent"), "%")}.'
    else:
        deck = 'Compare complete January and September calendar months across five request-context buckets. Missing observations are shown as unavailable, never as zero demand.'
    stats = ((multiple(million.get("daily_growth_multiple")), "1M bucket · daily volume", "September / January, normalized by days"),
             (multiple(all_c.get("daily_growth_multiple")), "All text · daily volume", "The overall public text traffic baseline"),
             (_metric(million.get("share_of_text_change_pp"), " pp", True), "1M bucket · traffic ratio change", "Change in bucket tokens / all text tokens"))
    cards = ''.join(f'<article class="study-stat"><p class="stat-label">{title}</p><strong class="stat-number">{value}</strong><p class="stat-detail">{detail}</p></article>' for value, title, detail in stats)
    content = f'''<section class="study-hero" aria-labelledby="study-title"><p class="study-kicker">Focused study / January → September 2026</p><h2 id="study-title">Traffic is shifting toward<br />larger request contexts.</h2><p class="study-deck">{deck}</p><div class="study-meta"><span>Five request-context buckets</span><span>Complete UTC calendar months</span><span>Input + output tokens</span></div><div class="study-jump"><a href="#what-counts">What exactly is counted? ↓</a><a href="#context-trends">Explore the data ↓</a><a href="architecture-comparison.html">Compare memory architectures ↗</a><a href="{ARTICLE}">Read OpenRouter’s “Cache is Crucial” ↗</a></div></section>
    <div class="study-stats">{cards}</div>
    <p class="study-boundary"><strong>Each API request contributes its input + output tokens.</strong> We sum that traffic across requests in each context bucket. These are not first-prompt sizes, session totals, or average context lengths. A follow-up counts again, including any history sent with it.</p>'''
    content += """<section class="study-panel" id="what-counts" aria-labelledby="what-counts-title">
    <h2 id="what-counts-title">What exactly is counted?</h2>
    <p><strong>The question this graph answers: how much input + output token traffic comes from requests in each context-size bucket?</strong> One request means one model call, including a follow-up or an agent’s next call after a tool result. It is not a whole conversation.</p>
    <div class="method-grid"><article class="method-item"><h3>1. Group individual requests</h3><p>OpenRouter assigns requests to context buckets labeled 1K, 10K, 100K, 1M and 10M. These describe request context, not the model’s maximum supported window. The exact boundaries and whether assignment uses input alone or input + output are not documented.</p></article><article class="method-item"><h3>2. Add input and output traffic</h3><p>Each request contributes its full input plus its newly generated output. Input includes whatever system instructions, tool definitions, documents and conversation history the application sends, as well as the latest user message. A short “continue” can accompany a large input.</p></article><article class="method-item"><h3>3. Sum across requests</h3><p>The API returns aggregate totals, not individual request records. Follow-ups contribute again, including repeated history. One session can contribute many times and move between buckets as its context changes. The graph neither groups traffic by session nor isolates first requests.</p></article></div>
    <details class="study-details"><summary>Example: two calls, 2,700 traffic tokens, 1,500 tokens of final context</summary>
    <p>Illustration only: assume a text conversation keeps all history, with no compaction, and ignore message-format overhead.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Two-request token accounting example"><table><thead><tr><th scope="col">Model call</th><th scope="col">Full input</th><th scope="col">New output</th><th scope="col">Traffic contributed</th><th scope="col">Context after reply</th></tr></thead><tbody><tr><th scope="row">First request</th><td>1,000</td><td>200</td><td>1,200</td><td>1,200</td></tr><tr><th scope="row">Follow-up adding 100 tokens</th><td>1,300</td><td>200</td><td>1,500</td><td>1,500</td></tr></tbody></table></div>
    <p>The second input contains the earlier 1,200 tokens plus 100 new tokens. Together the calls contribute <strong>2,700 traffic tokens</strong>, while the final context is <strong>1,500 tokens</strong>. Each call contributes to its own assigned bucket; no specific bucket is assumed in this example.</p>
    <p>For a conventional full-attention model, a retained cache covering that final context would represent roughly 1,500 token positions, not 2,700. Reusing the earlier prefix can avoid rebuilding its KV state. These traffic totals do not reveal whether such reuse actually happened. Truncation or compaction can also shrink the next request’s input.</p></details>
    <p class="note"><strong>Can we split input from output, or show first-prompt sizes?</strong> Not from this public dataset: it exposes only their combined total and no first-turn or session metadata. Separate counts for a customer’s own requests do not supply a platform-wide split. We would need additional aggregate data from OpenRouter.</p></section>"""
    if study:
        legends = ''.join(f'<span data-series-legend="{b}"><i class="swatch" style="background:{PALETTE[b]}"></i>{"All text (dashed)" if b == "text" else b}</span>' for b in PALETTE)
        chart_panels = ''.join(f'<div data-trend-view="{mode}"><h3 class="chart-view-title">{title}</h3><div class="study-chart-wrap">{trend_chart(study, mode)}</div></div>' for mode, title in (("ratio", "How the traffic ratios changed"), ("growth", "How quickly each dataset grew"), ("volume", "How much traffic each dataset carried")))
        content += f'''<section class="study-panel" id="context-trends">{heading("01", "Input + output token traffic by request-context bucket", "Follow all nine months. Each series sums traffic from individual requests across sessions; it does not track a session’s accumulated length.")}<div class="study-controls" hidden aria-label="Chart measure"><button type="button" data-view="ratio" aria-pressed="true">Traffic ratio</button><button type="button" data-view="growth" aria-pressed="false">Growth since January</button><button type="button" data-view="volume" aria-pressed="false">Input + output per day</button></div><div class="study-legend">{legends}</div>{chart_panels}<p class="note">Traffic ratio = (input + output tokens in the bucket) / (input + output tokens across all text requests). Buckets are plotted independently; this is not a verified partition or a distribution of request counts. Growth uses January = 1×. Incomplete months leave a gap. Point values are available on hover or keyboard focus.</p><div class="chart-tooltip" role="status" hidden></div></section>
        <section class="study-panel">{heading("02", "January → September, bucket by bucket", "Read volume growth together with traffic ratio: a bucket can grow while accounting for less of overall traffic.")}{bucket_cards(study)}{comparison_table(study)}<p class="note">Daily volume = monthly input + output token total / calendar days: 31 for January, 30 for September. K = thousand, M = million, B = billion, T = trillion; pp = percentage points. A dash means the evidence is insufficient. Exact totals and coverage are in the <a href="data/openrouter/context-demand.json" download>downloadable analysis</a>.</p></section>
        <section class="study-panel" id="context-architecture">{heading("03", "Where linear hybrids fit", "Shares of input + output token traffic within each request-context dataset. The denominator includes unresolved models and the unnamed tail.")}{architecture(study)}<p class="note">Only exact model identities with reviewed evidence receive an architecture. Daily top-50 cutoffs hide other identities; an unobserved category is not measured zero usage. Changing shares do not establish a causal advantage or comparative model quality.</p></section>'''
    content += f'''<article class="source-feature" aria-labelledby="source-title"><p class="source-label">Read alongside / OpenRouter Inferred 03 · October 8, 2026</p><h2 id="source-title">Longer-lived context.<br />More repeated use.</h2><p>OpenRouter’s <a href="{ARTICLE}">“Cache is Crucial”</a> examines cache reuse in agentic traffic. This independent supplement asks a related question: how did traffic across request-context buckets change over the same January–September period?</p><div class="source-stats"><div class="source-stat"><strong>~60% → &lt;90%</strong><span>Cached prompt tokens as a share of all agentic tokens, January → September</span></div><div class="source-stat"><strong>~7×</strong><span>Agentic versus human token volume in the last week of September</span></div></div><p class="note">Source: OpenRouter’s published newsletter, not calculations from this page’s dataset. Its charts exclude resellers and classify API keys as agentic, human or mixed. Those populations and denominators differ from these public context-bucket totals; the cache percentages cannot be applied to them.</p><a class="source-link" href="{ARTICLE}">Read the original findings ↗</a></article>
    <section class="study-panel">{heading("04", "What this means for memory systems", "A workload signal to guide open research and systems projections.")}<div class="method-grid"><article class="method-item"><h3>Test useful recall</h3><p>More traffic in a context bucket identifies a workload to study. Measure whether regular attention, linear hybrids and optional memory preserve the information needed to complete tasks.</p></article><article class="method-item"><h3>Measure state reuse</h3><p>A follow-up may reuse KV state for an earlier prefix, even though that history contributes to token traffic again. This dataset does not reveal which prefixes were reused or how many distinct session caches existed. Measure bytes per state, concurrent sessions, idle gaps and rebuild cost.</p></article><article class="method-item"><h3>Establish the storage case</h3><p>For full attention, retained KV state scales with the token positions kept for a request and the model’s KV layout. Sliding windows and recurrent hybrids retain state differently. Summed token traffic is not resident cache size: capacity and offload bandwidth also depend on sharing, concurrency, retention and placement.</p><p><a href="hybrid-checkpoints.html">Hybrid checkpoints ↗</a> · <a href="io-projections.html">I/O projections ↗</a></p></article></div></section>'''
    content += '<section class="study-panel">' + heading("05", "Sources, coverage and reproducibility")
    content += '''<div class="method-grid"><div class="method-item"><h3>Observed public traffic</h3><p>Daily OpenRouter rankings, filtered to text output. Each query returns up to 50 named models plus an aggregated “other” row. Both count toward total traffic. Private traffic is outside this public view; OpenRouter is not a census of the inference market.</p></div><div class="method-item"><h3>What remains unknown</h3><p>The public rankings API supplies combined total_tokens, with no separate input/output fields or filter to select one. It exposes no first-turn lengths, session IDs, turn numbers, request counts, prompt-length percentiles, cache split or offload activity. Exact bucket boundaries and whether assignment uses input alone or input + output are undocumented. Provider tokenizers differ.</p></div><div class="method-item"><h3>A reproducible comparison</h3><p>Complete January 1–31 and September 1–30 UTC windows. A missing day suppresses its monthly metric. All nine monthly averages use calendar-day counts. Architecture attribution comes from the versioned registry, not OpenRouter.</p></div></div>'''
    if study:
        rows = []
        for d in study["datasets"]:
            a, b = d["comparison"]["baseline"], d["comparison"]["current"]
            rows.append(f'<tr><th>{_escape(d["label"])}</th><td>{a["days"]}/{a["expected_days"]}</td><td>{b["days"]}/{b["expected_days"]}</td><td>{_metric(a.get("total_tokens"), digits=0)}</td><td>{_metric(b.get("total_tokens"), digits=0)}</td></tr>')
        content += '<details class="study-details"><summary>Inspect calendar coverage and exact monthly totals</summary><div class="table-wrap" tabindex="0"><table><thead><tr><th>Dataset</th><th>Jan days</th><th>Sep days</th><th>Jan tokens</th><th>Sep tokens</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div><p class="note">The analysis download also records numerical reconciliation between the five bucket totals and all-text totals. Agreement does not prove interval boundaries or partition semantics.</p></details>'
    content += rolling_context(analysis)
    content += '<p class="note"><a href="https://openrouter.ai/docs/api/api-reference/datasets/daily-token-totals-for-top-50-models">OpenRouter dataset definitions</a> · <a href="https://github.com/mcgrof/kvcache-view/tree/main/scripts/openrouter">Calculation source</a> · <a href="https://github.com/mcgrof/kvcache-view/blob/main/data/openrouter/README.md">Data methodology</a></p></section>'
    return _page("Context demand", "January–September 2026: input + output token traffic, grouped by the context size of individual API requests.", content, provenance, analysis, "context-demand.html")
