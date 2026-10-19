# SPDX-License-Identifier: MIT
"""A sourced comparison of local attention, recurrent memory and deployed hybrids."""

import math

from .render import _escape, _metric, _number, _page, _safe_link, _tokens

FOCUS = ("local_global_attention", "hybrid_linear", "hybrid_mamba")
NAMES = {"local_global_attention": "Sliding-window + full attention", "hybrid_linear": "Linear hybrids",
         "hybrid_mamba": "Mamba hybrids", "pure_linear": "Pure linear / recurrent"}
COLORS = {"local_global_attention": "#f9a08a", "hybrid_linear": "#8be6c8",
          "hybrid_mamba": "#b8a5ed", "pure_linear": "#efd48d"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def _heading(number, title, description=""):
    return (f'<div class="section-heading"><span class="section-number">{number}</span>'
            f'<div><h2>{title}</h2><p>{description}</p></div></div>')


def _name(category, fallback=""):
    return NAMES.get(category, fallback or category)


def _records(month):
    return {row["category"]: row for row in month.get("architecture", [])}


def _series(dataset):
    focus = list(FOCUS)
    if any(_records(month).get("pure_linear", {}).get("observed") for month in dataset["monthly"]):
        focus.append("pure_linear")
    return focus


def _chart(dataset, mode):
    field = {"share": "share_percent", "volume": "mean_daily_tokens", "growth": "index_january_100"}[mode]
    divisor = 100 if mode == "growth" else 1
    focus = _series(dataset)
    rows = [_records(month) for month in dataset["monthly"]]
    values = [row.get(category, {}).get(field) for row in rows for category in focus]
    maximum = max((value / divisor for value in values if _number(value)), default=1)
    raw_step = max(maximum / 4, .01)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next((n * magnitude for n in (1, 2, 2.5, 5, 10) if n * magnitude >= raw_step), 10 * magnitude)
    top = max(step, math.ceil(maximum / step) * step)
    width, height, left, right, ytop, bottom = 1040, 365, 85, 38, 37, 51
    plotw, ploth = width-left-right, height-ytop-bottom
    x = lambda i: left + i * plotw / max(1, len(rows)-1)
    y = lambda value: ytop + ploth * (1 - value / divisor / top)
    chart_id = f'architecture-{dataset["bucket"]}-{mode}'
    title = {"share": "Share of all tokens in this dataset", "volume": "Average reported tokens per day",
             "growth": "Daily volume relative to January"}[mode]
    unit = {"share": "% of dataset tokens", "volume": "Tokens per day", "growth": "January = 1×"}[mode]
    parts = [f'<svg class="trend-chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{chart_id}-title">',
             f'<title id="{chart_id}-title">{_escape(dataset["label"])}: {title}, January to September 2026</title>',
             f'<text x="{left}" y="17" class="axis-label">{unit}</text>']
    for tick in range(round(top / step)+1):
        value = tick * step
        yy = ytop + ploth * (1-value/top)
        label = _tokens(value) if mode == "volume" else f'{value:g}' + ("×" if mode == "growth" else "%")
        parts += [f'<line class="gridline" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>',
                  f'<text class="axis-label" x="{left-15}" y="{yy+5:.1f}" text-anchor="end">{label}</text>']
    for i, month in enumerate(dataset["monthly"]):
        parts.append(f'<text class="axis-label" x="{x(i):.1f}" y="{height-16}" text-anchor="middle">{MONTHS[int(month["month"][-2:])-1]}</text>')
    for category in focus:
        points, segments = [], []
        for i, row in enumerate(rows):
            value = row.get(category, {}).get(field)
            if _number(value):
                points.append((i, value))
            elif points:
                segments.append(points)
                points = []
        if points:
            segments.append(points)
        for segment in segments:
            coords = " ".join(f'{x(i):.1f},{y(value):.1f}' for i, value in segment)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[category]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
            for i, value in segment:
                desc_value = (_tokens(value) + " tokens/day" if mode == "volume" else
                              f'{value/100:,.2f}× January' if mode == "growth" else f'{value:,.2f}% of dataset tokens')
                desc = f'{dataset["monthly"][i]["month"]} · {_name(category)} · {desc_value}'
                parts.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="5" fill="{COLORS[category]}" stroke="#0b1327" stroke-width="2" tabindex="0" data-chart-point="{_escape(desc)}" aria-label="{_escape(desc)}"><title>{_escape(desc)}</title></circle>')
    parts.append('</svg>')
    return ''.join(parts)


def _summary_cards(dataset):
    comparison = dataset["comparison"]
    before, after = _records(comparison["baseline"]), _records(comparison["current"])
    changes = {row["category"]: row for row in comparison.get("architecture", [])}
    cards = []
    for category in FOCUS:
        a, b = before.get(category, {}), after.get(category, {})
        change = changes.get(category, {})
        cards.append(f'<article class="architecture-market-card" style="--series:{COLORS[category]}"><p class="eyebrow">{_name(category)}</p><p class="architecture-share-pair"><span>{_metric(a.get("share_percent"), "%", digits=2)}</span><span aria-hidden="true">→</span><strong>{_metric(b.get("share_percent"), "%", digits=2)}</strong></p><p class="note">January → September · share of dataset tokens</p><div class="architecture-card-footer"><span>{_metric(change.get("share_change_pp"), " pp", signed=True, digits=2)}</span><span>{_metric(change.get("daily_growth_multiple"), "×")} daily volume</span></div></article>')
    return '<div class="architecture-market-cards">' + ''.join(cards) + '</div>'


def _visibility(dataset):
    a, b = dataset["comparison"]["baseline"], dataset["comparison"]["current"]
    rows = []
    for label, month in (("January", a), ("September", b)):
        unresolved, tail = month.get("unresolved_share_percent"), month.get("tail_share_percent")
        records = _records(month)
        other_known = sum(r["share_percent"] for key, r in records.items()
                          if key not in (*FOCUS, "unknown", "attention_unspecified", "other")
                          and _number(r.get("share_percent")))
        segments = [(category, records.get(category, {}).get("share_percent"), COLORS[category]) for category in FOCUS]
        segments += [("Other resolved styles", other_known, "#7191aa"),
                     ("Unresolved architecture/style", unresolved, "#586779"), ("Unnamed tail", tail, "#b78e62")]
        bars = ''.join(f'<span style="width:{min(100,max(0,value)):.4f}%;background:{color}" title="{_escape(_name(category))}: {value:.2f}%"></span>'
                       for category, value, color in segments if _number(value) and value > 0)
        rows.append(f'<div class="architecture-coverage-row"><span>{label}</span><div class="architecture-coverage-track" aria-hidden="true">{bars}</div><span>{_metric(unresolved,"%",digits=2)} unresolved</span></div>')
    return ('<div class="architecture-visibility"><h4>Keep the whole denominator in view</h4>' + ''.join(rows)
            + '<p class="note">Colored segments are the three charted groups. Blue-gray = other resolved styles; gray = unresolved architecture or attention style; tan = unnamed tail. No categories are renormalized away.</p></div>')


def _exact_table(dataset):
    before, after = _records(dataset["comparison"]["baseline"]), _records(dataset["comparison"]["current"])
    changes = {row["category"]: row for row in dataset["comparison"].get("architecture", [])}
    rows = []
    categories = list(before) + [key for key in after if key not in before]
    for category in categories:
        a, b, change = before.get(category, {}), after.get(category, {}), changes.get(category, {})
        color = COLORS.get(category, '#7191aa')
        rows.append(f'<tr><th scope="row"><span class="chart-key" style="--series:{color}"></span>{_escape(_name(category,b.get("label",a.get("label",""))))}</th><td>{_metric(a.get("share_percent"),"%",digits=2)}</td><td>{_metric(b.get("share_percent"),"%",digits=2)}</td><td>{_metric(change.get("share_change_pp")," pp",signed=True,digits=2)}</td><td>{_tokens(a.get("mean_daily_tokens"))}</td><td>{_tokens(b.get("mean_daily_tokens"))}</td><td>{_metric(change.get("daily_growth_multiple"),"×",digits=2)}</td></tr>')
    return ('<details class="study-details"><summary>All architecture groups · January and September values</summary><div class="table-wrap study-table" tabindex="0" role="region" aria-label="All architecture groups, January and September"><table><thead><tr><th scope="col">Architecture / visibility</th><th scope="col">Jan share</th><th scope="col">Sep share</th><th scope="col">Change</th><th scope="col">Jan tokens/day</th><th scope="col">Sep tokens/day</th><th scope="col">Daily volume</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><p class="note">A dash means unavailable or unobserved, not measured zero. Daily volume normalizes January’s 31 days and September’s 30 days. pp = percentage points.</p></details>')


def _monthly_table(dataset):
    rows = []
    for month in dataset["monthly"]:
        records = _records(month)
        for category in _series(dataset):
            record = records.get(category, {})
            index = record.get("index_january_100")
            growth = index / 100 if _number(index) else None
            rows.append(f'<tr><th scope="row">{_escape(month["month"])} · {_name(category)}</th><td>{_metric(record.get("share_percent"),"%",digits=3)}</td><td>{_metric(record.get("mean_daily_tokens"),digits=0)}</td><td>{_metric(growth,"×",digits=3)}</td><td>{month["days"]}/{month["expected_days"]}</td></tr>')
    return '<details class="study-details"><summary>Accessible chart data · all nine months</summary><div class="table-wrap study-table" tabindex="0" role="region" aria-label="Monthly values for the plotted architecture groups"><table><thead><tr><th scope="col">Month / group</th><th scope="col">Dataset share</th><th scope="col">Tokens/day</th><th scope="col">January = 1×</th><th scope="col">Days</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>'


def _contributors(dataset):
    cards = []
    for group in dataset.get("contributors", []):
        if group["category"] not in (*FOCUS, "pure_linear"):
            continue
        models = []
        for model in group.get("models", []):
            sources = ' · '.join(_safe_link(url, f'Source {i+1}') for i, url in enumerate(model.get("sources", [])))
            models.append(f'<li><div class="architecture-model-top"><strong>{_escape(model.get("name") or model["model"])}</strong><span>{_metric(model.get("share_of_category_percent"),"%",digits=1)}</span></div><p>{_tokens(model.get("mean_daily_tokens"))} tokens/day · {_metric(model.get("share_of_dataset_percent"),"%",digits=2)} of this dataset</p><details><summary>Classification evidence</summary><p>{_escape(model.get("notes", ""))}</p><p class="note">{sources}</p></details></li>')
        empty = ('<p class="note">No named traffic observed in this dataset. This does not establish zero usage.</p>'
                 if group.get("available") else '<p class="note">Dated model contributor data are unavailable.</p>')
        cards.append(f'<article class="architecture-contributor-card"><h4>{_name(group["category"],group.get("label",""))}</h4><p class="note">September · share within this group</p><ol>{"".join(models)}</ol>{empty if not models else ""}</article>')
    if not cards:
        return ''
    return '<details class="study-details"><summary>Which reviewed models drive this comparison?</summary><p class="note">Largest named contributors in September. These are model identities with reviewed evidence; the charts do not infer the architecture of closed or unreviewed models.</p><div class="architecture-contributors">' + ''.join(cards) + '</div></details>'


def _concentration(dataset):
    leaders = []
    for group in dataset.get("contributors", []):
        if group["category"] not in ("local_global_attention", "hybrid_linear") or not group.get("models"):
            continue
        model = group["models"][0]
        name = _escape(model.get("name") or model["model"])
        share = _metric(model.get("share_of_category_percent"), "%", digits=1)
        label = "sliding-window + full-attention" if group["category"] == "local_global_attention" else "linear-hybrid"
        leaders.append(f'{name} accounts for {share} of {label} traffic')
    if not leaders:
        return ''
    return '<p class="architecture-scope-note"><strong>September leaders:</strong> ' + '; '.join(leaders) + '. Linear hybrids include GDN, KDA and other operators; grouped growth is not GDN-specific evidence.</p>'


def _market(study):
    options, panels = [], []
    for dataset in study.get("datasets", []):
        bucket = dataset["bucket"]
        label = "All text" if bucket == "text" else bucket + " context bucket"
        options.append(f'<option value="{_escape(bucket)}">{label}</option>')
        legends = ''.join(f'<span><i class="swatch" style="background:{COLORS[category]}"></i>{_name(category)}</span>' for category in _series(dataset))
        charts = ''.join(f'<div data-comparison-view="{mode}"><h4 class="chart-view-title">{title}</h4><div class="study-chart-wrap">{_chart(dataset,mode)}</div></div>' for mode, title in (("share","Share within this dataset"),("volume","Absolute daily token volume"),("growth","Daily volume growth from January")))
        panels.append(f'<div data-comparison-dataset="{_escape(bucket)}"><h3 class="architecture-dataset-title">{label}</h3>{_summary_cards(dataset)}<div class="study-legend">{legends}</div>{charts}{_visibility(dataset)}{_concentration(dataset)}{_exact_table(dataset)}{_monthly_table(dataset)}{_contributors(dataset)}</div>')
    return f'''<section class="study-panel" id="architecture-market">{_heading("02", "Where the deployed models are", "The same nine months, the same denominators. Switch between traffic share, absolute volume and growth to avoid mistaking a small starting point for market dominance.")}
    <div class="architecture-market-controls" hidden><label class="architecture-select">Choose a request-context dataset<select data-comparison-select>{''.join(options)}</select></label><div class="study-controls" aria-label="Architecture chart measure"><button type="button" data-comparison-mode="share" aria-pressed="true">Traffic share</button><button type="button" data-comparison-mode="volume" aria-pressed="false">Tokens per day</button><button type="button" data-comparison-mode="growth" aria-pressed="false">Growth since January</button></div></div>
    <div class="architecture-scope-note"><p><strong>Sliding-window + full attention: what the coral series measures.</strong> Sliding-window attention (SWA) reads a moving window of recent tokens. Full (global) attention can read the entire preceding context. This group contains reviewed models that combine these two layer types. Their documented layer configurations establish the use of SWA; the word “local” alone does not.</p><p>Shares count <strong>all tokens served by those models</strong>, not only activity in their SWA layers. Other categories, including some sparse-attention models, also contain SWA, so this series does not measure total SWA usage. Linear hybrids combine recurrent updates with attention; Mamba hybrids are a separate family.</p></div>
    <p class="note">These groups describe checkpoint architectures. The traffic data do not reveal provider cache policies or whether an endpoint uses the paper’s SWA + sinks conversion.</p>{''.join(panels)}
    <p class="architecture-small-base"><strong>Growth needs its starting point.</strong> Linear hybrids begin at a small observed January share. New model launches, changing disclosure, pricing and routing can all move these curves. Traffic cannot establish which memory mechanism caused a gain.</p>
    <p class="note">Source: OpenRouter public daily rankings, input + output tokens, January–September 2026. The denominator includes unresolved models and the unnamed tail. Missing or unobserved categories produce gaps. Context-bucket labels are supplied by the API; exact interval boundaries are not documented. Hover or focus a point for its value; exact chart tables are available above.</p><div class="chart-tooltip" role="status" hidden></div></section>'''


def _demand_signal(study):
    datasets = {dataset["bucket"]: dataset for dataset in study.get("datasets", [])}
    all_data, million_data = datasets.get("text", {}), datasets.get("1M", {})
    all_compare, million_compare = all_data.get("comparison", {}), million_data.get("comparison", {})
    def growth(comparison):
        a = comparison.get("baseline", {}).get("mean_daily_tokens")
        b = comparison.get("current", {}).get("mean_daily_tokens")
        return b/a if _number(a) and a > 0 and _number(b) else None
    def ratio(period):
        a = all_compare.get(period, {}).get("total_tokens")
        b = million_compare.get(period, {}).get("total_tokens")
        return 100*b/a if _number(a) and a > 0 and _number(b) else None
    stats = [(_metric(growth(all_compare), "×"), "All text · daily volume", "September / January 2026"),
             (_metric(growth(million_compare), "×"), "1M bucket · daily volume", "Token traffic, not average prompt length"),
             (_metric(ratio("baseline"), "%") + " → " + _metric(ratio("current"), "%"),
              "1M bucket / all text", "A workload signal for memory experiments")]
    cards = ''.join(f'<article class="architecture-signal-card"><p>{label}</p><strong>{value}</strong><span>{detail}</span></article>' for value, label, detail in stats)
    return '<div class="architecture-signals">' + cards + '</div><p class="architecture-signal-source">Aligned with the <a href="context-demand.html">context-demand study</a>. Ratios compare independent API datasets, not a verified distribution of request counts.</p>'


def _mechanisms():
    return '''<div class="architecture-mechanisms">
    <article class="architecture-mechanism architecture-window"><p class="eyebrow">01 / Sliding-window attention</p><div class="architecture-token-strip" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><h3>Retain a recent window</h3><p>Sliding-window attention (SWA) reads a bounded window of recent tokens that advances with generation. It keeps their token-level keys and values. Some designs also retain attention sinks.</p><span class="architecture-tag">Explicit recent KV</span></article>
    <article class="architecture-mechanism architecture-recurrent"><p class="eyebrow">02 / Linear recurrence</p><div class="architecture-state-flow" aria-hidden="true"><span></span><b>→</b><span></span><b>→</b><span></span></div><h3>Update a compact state</h3><p>Fold history into a recurrent state. Its size can stay bounded as context grows; interference and forgetting shape what can be recalled.</p><span class="architecture-tag">Compressed historical state</span></article>
    <article class="architecture-mechanism architecture-hybrid"><p class="eyebrow">03 / Deployed hybrids</p><div class="architecture-hybrid-stack" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><h3>Mix memory mechanisms</h3><p>Combine sliding-window or recurrent layers with full attention. Full (global) layers can read the entire preceding context, so their KV cache grows with context length.</p><span class="architecture-tag">Bounded state + growing KV</span></article></div>'''


def render_architecture_page(study, provenance):
    from .architecture_evidence import paper_section, memory_section, research_section
    content = '''<section class="study-hero architecture-hero" aria-labelledby="architecture-title"><p class="study-kicker">Architecture study / Evidence meets deployment</p><h2 id="architecture-title">Two ways to remember.<br />One growing workload.</h2><p class="study-deck">Sliding windows keep recent tokens. Linear recurrence compresses the past. The useful comparison joins <strong>controlled recall tests</strong>, <strong>observed usage</strong> and <strong>the state each system must keep</strong>.</p><div class="study-meta"><span>Paper results ≠ market adoption</span><span>January → September 2026</span><span>Closed-model uncertainty retained</span></div><div class="study-jump"><a href="#paper">Compare the paper ↓</a><a href="#architecture-market">Explore deployment ↓</a><a href="#memory">Model memory demand ↓</a></div></section>'''
    content += _mechanisms() + _demand_signal(study) + paper_section() + _market(study) + memory_section() + research_section()
    analysis_url = provenance.get("analysis_url", "data/openrouter/architecture-comparison.json")
    provenance_url = provenance.get("provenance_url", "data/openrouter/architecture-comparison-provenance.json")
    content += f'''<section class="study-panel" id="architecture-method">{_heading("05", "What is measured, and what is not", "A reproducible comparison with explicit evidence boundaries.")}
    <div class="method-grid"><article class="method-item"><h3>Aligned public observations</h3><p>All six datasets use complete UTC calendar months from January through September 2026. Daily rankings expose up to 50 named models plus an aggregated tail. Input and output tokens count toward traffic; private activity is outside this public view.</p></article><article class="method-item"><h3>Exact identities, reviewed sources</h3><p>Architecture and attention style come from versioned registries with model-specific evidence. Names, context limits and closed-model performance do not establish architecture. Unresolved models stay in the denominator.</p></article><article class="method-item"><h3>Comparison, not attribution</h3><p>Traffic growth reflects adoption, model releases, routing and pricing as well as architecture. The paper tests particular checkpoints under particular memory budgets. Neither observation proves a universal quality ranking.</p></article><article class="method-item"><h3>Memory and storage still need traces</h3><p>Token totals do not reveal resident bytes, concurrent sessions, idle periods, cache hits or restore latency. Those measurements connect useful recall to HBM, DRAM and NAND requirements.</p></article></div>
    <div class="architecture-downloads"><a href="{_escape(analysis_url)}" download>Download analysis JSON ↓</a><a href="{_escape(provenance_url)}" download>Source manifest ↓</a><a href="data/openrouter/attention-registry.json" download>Attention evidence ↓</a><a href="https://github.com/mcgrof/kvcache-view/tree/main/scripts/openrouter">Calculation source ↗</a></div>
    <p class="note">Read alongside <a href="https://openrouter.ai/data/newsletter/cache-is-crucial">OpenRouter’s “Cache is Crucial”</a>. Its agentic cache statistics use a different population and denominator; their percentages cannot be applied to these public context totals. See <a href="context-demand.html">context demand</a>, <a href="attention-trends.html">all attention styles</a>, <a href="hybrid-trends.html">architecture trends</a> and <a href="cache-telemetry.html">cache telemetry</a> for the broader observations.</p></section>'''
    return _page("Sliding windows & recurrent memory", "Independent open-source analysis of sliding-window attention and linear recurrence: controlled recall evidence, January–September 2026 deployment trends, and implications for memory and storage.", content, provenance, study, "architecture-comparison.html")
