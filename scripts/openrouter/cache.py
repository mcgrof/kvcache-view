# SPDX-License-Identifier: MIT
"""Summarize advertised public cache prices without inventing cache usage.

The Models API contains list prices, not storage-tier telemetry. Missing prices
remain missing, and an explicit zero is preserved as a reported zero price.
"""

from decimal import Decimal, InvalidOperation
from html import escape
from statistics import median
from urllib.parse import quote


PRICE_FIELDS = ("prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h")
SOURCES = [
    {
        "title": "OpenRouter Models API",
        "url": "https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties",
    },
    {
        "title": "OpenRouter endpoint metadata",
        "url": "https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model",
    },
    {
        "title": "OpenRouter prompt caching and request usage metrics",
        "url": "https://openrouter.ai/docs/guides/best-practices/prompt-caching",
    },
    {
        "title": "OpenRouter public effective-pricing announcement",
        "url": "https://openrouter.ai/blog/announcements/february-release-spotlight/",
    },
    {
        "title": "OpenRouter account Analytics API announcement",
        "url": "https://openrouter.ai/blog/announcements/activity-dashboard/",
    },
    {
        "title": "OpenRouter response caching",
        "url": "https://openrouter.ai/docs/guides/features/response-caching",
    },
]


def _number(value):
    """Return a nonnegative, finite Decimal; malformed/null values are unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() and number >= 0 else None


def analyze_cache(catalog_payload: dict) -> dict:
    """Analyze a full Models API snapshot, preserving its original provenance."""
    models = catalog_payload.get("data")
    if not isinstance(models, list):
        raise ValueError("Models snapshot must contain a data array")
    rows = []
    ids = set()
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise ValueError("Models snapshot contains an entry without a model id")
        model_id = model["id"]
        if model_id in ids:
            raise ValueError(f"Duplicate model id in catalog: {model_id}")
        ids.add(model_id)
        pricing = model.get("pricing") or {}
        if not isinstance(pricing, dict):
            raise ValueError(f"Invalid pricing object for model: {model_id}")
        values = {field: _number(pricing.get(field)) for field in PRICE_FIELDS}
        base, read = values["prompt"], values["input_cache_read"]
        discount = None
        if base is not None and base > 0 and read is not None:
            discount = float(100 * (1 - read / base))
        rows.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "author_slug": model_id.split("/", 1)[0],
                "output_modalities": (model.get("architecture") or {}).get("output_modalities", []),
                "context_length": model.get("context_length"),
                "prices_per_million": {
                    key: float(value * 1_000_000) if value is not None else None
                    for key, value in values.items()
                },
                "price_status": {
                    key: "reported" if values[key] is not None else (
                        "unavailable" if key not in pricing or pricing[key] is None else "invalid"
                    )
                    for key in PRICE_FIELDS
                },
                "listed_read_discount_percent": discount,
                "zero_input_price": base == 0 if base is not None else False,
                "has_overrides": bool(pricing.get("overrides")),
                "overrides": pricing.get("overrides") or [],
                "source_url": "https://openrouter.ai/" + quote(model_id, safe="/:"),
                "observed_cached_tokens": None,
                "observed_cache_hit_rate": None,
                "offloaded_bytes": None,
                "nvme_restore_bytes": None,
            }
        )
    rows.sort(key=lambda row: row["id"])
    summary = {
        "models": len(rows),
        "cache_read_price_reported": sum(row["prices_per_million"]["input_cache_read"] is not None for row in rows),
        "cache_write_price_reported": sum(row["prices_per_million"]["input_cache_write"] is not None for row in rows),
        "cache_write_1h_price_reported": sum(row["prices_per_million"]["input_cache_write_1h"] is not None for row in rows),
        "no_cache_price_reported": sum(all(row["prices_per_million"][key] is None for key in PRICE_FIELDS[1:]) for row in rows),
        "zero_input_price_models": sum(row["zero_input_price"] for row in rows),
        "models_with_overrides": sum(row["has_overrides"] for row in rows),
        "models_with_read_discount": sum(row["listed_read_discount_percent"] is not None and row["listed_read_discount_percent"] > 0 for row in rows),
    }
    return {
        "schema_version": 1,
        "scope": "public_models_catalog_advertised_prices",
        "metadata": dict(catalog_payload.get("meta") or {}),
        "summary": summary,
        "dashboard": _dashboard(rows),
        "rows": rows,
        "sources": SOURCES,
        "limitations": [
            "Catalog entries are model variants, not distinct checkpoints or provider endpoints.",
            "Missing cache prices do not establish absence of prompt caching.",
            "List prices and their ratios are not observed savings or cache-hit measurements.",
            "Provider routing and conditional pricing can change actual charges.",
            "Public model pages show some cache-hit and effective-price statistics, but this collector uses the documented Models API only.",
            "No documented public API for platform-wide cache-hit history or NVMe offload telemetry was identified.",
            "Request and account cache metrics do not identify HBM, DRAM, NVMe, or network storage tiers.",
        ],
    }


def _dashboard(rows):
    """Use text-capable variants and explicit paired prices for every denominator."""
    text_rows = [row for row in rows if "text" in row["output_modalities"]]
    paired = [row for row in text_rows if row["listed_read_discount_percent"] is not None]
    discounts = [row["listed_read_discount_percent"] for row in paired]
    bands = [
        ("Read costs more", lambda x: x < 0),
        ("Same price", lambda x: x == 0),
        ("Above 0%, below 50% cheaper", lambda x: 0 < x < 50),
        ("50% to below 75% cheaper", lambda x: 50 <= x < 75),
        ("75% to below 90% cheaper", lambda x: 75 <= x < 90),
        ("90% to 100% cheaper", lambda x: 90 <= x <= 100),
    ]
    return {
        "scope": "text_output_catalog_variants_unweighted",
        "text_variants": len(text_rows),
        "paired_read_prices": len(paired),
        "paired_with_overrides": sum(row["has_overrides"] for row in paired),
        "zero_input_variants": sum(row["zero_input_price"] for row in text_rows),
        "median_read_discount_percent": median(discounts) if discounts else None,
        "price_reporting": {
            field: sum(row["prices_per_million"][field] is not None for row in text_rows)
            for field in PRICE_FIELDS[1:]
        },
        "discount_bands": [{"label": label, "count": sum(predicate(x) for x in discounts)}
                           for label, predicate in bands],
    }


def _bar(label, count, total, color="cyan"):
    percent = count / total * 100 if total else 0
    return (f'<div class="cache-bar-row"><div class="cache-bar-label"><span>{escape(label)}</span>'
            f'<strong>{count:,} <span class="muted">/ {total:,}</span></strong></div>'
            f'<div class="cache-bar-track" aria-hidden="true"><span class="cache-bar-fill {color}" '
            f'style="width:{percent:.4f}%"></span></div></div>')


def _render_dashboard(analysis):
    stats = analysis["dashboard"]
    total, pairs = stats["text_variants"], stats["paired_read_prices"]
    reads = stats["price_reporting"]["input_cache_read"]
    coverage = f"{100 * reads / total:.1f}%" if total else "Unavailable"
    discount = stats["median_read_discount_percent"]
    discount_label = f"{discount:g}%" if discount is not None else "Unavailable"
    coverage_bars = "".join(_bar(label, stats["price_reporting"][key], total)
                            for key, label in (("input_cache_read", "Cache read"),
                                               ("input_cache_write", "Cache write"),
                                               ("input_cache_write_1h", "One-hour write")))
    distribution = "".join(_bar(band["label"], band["count"], pairs, "purple")
                            for band in stats["discount_bands"])
    return f'''
    <section class="panel lead-panel">
      <p class="eyebrow">Start here</p>
      <h2>Repeated input can be cheaper. How often is it reused?</h2>
      <p class="finding">OpenRouter lists prices for reading cached input. Those prices show the incentive
      to reuse a prompt, but this dataset does not measure how often reuse happens or whether it touches NVMe.</p>
      <div class="grid">
        <div class="stat"><span class="stat-value">{coverage}</span><strong>List a cache-read price</strong>
        <p>{reads:,} of {total:,} text-generating variants. Missing prices do not mean caching is unsupported.</p></div>
        <div class="stat"><span class="stat-value">{discount_label}</span><strong>Median listed read discount</strong>
        <p>Per cached input token, across {pairs:,} comparable variants. This is not a measured bill reduction.</p></div>
        <div class="stat"><span class="stat-value cache-unknown">Unknown</span><strong>Cache use and NVMe activity</strong>
        <p>This catalog supplies neither a cache-hit rate nor a storage tier. Unknown does not mean zero activity.</p></div>
      </div>
      <p class="muted">Charts count text-generating catalog variants equally, including variants with other output
      modalities. They are not weighted by traffic. The full {analysis['summary']['models']:,}-entry catalog is available below.</p>
      <div class="cache-definitions">
        <p><strong>Fresh input</strong> is prompt text the provider processes without an eligible cache hit.</p>
        <p><strong>Cache read</strong> is the price for eligible input tokens reused from a provider's prompt cache.</p>
        <p><strong>Cache write</strong> is a listed charge associated with storing a prefix. Retention periods and
        billing rules vary by provider; an absent write price does not establish that writes are free.</p>
      </div>
    </section>
    <div class="cache-chart-grid">
      <section class="panel">
        <h2>Which cache prices are reported?</h2>
        <p>Filled bars count reported prices; the remainder has no usable price in this snapshot.</p>
        <div class="cache-bars">{coverage_bars}</div>
        <p class="note">Each bar uses the same {total:,}-variant denominator. A variant can appear in more than one bar.</p>
      </section>
      <section class="panel">
        <h2>How much cheaper is a cache read?</h2>
        <p>Distribution of the listed discount compared with the same variant's fresh-input price.</p>
        <div class="cache-bars">{distribution}</div>
        <p class="note">{pairs:,} positive-input / reported-read pairs. Zero-input and missing-price entries are excluded.
        {stats['paired_with_overrides']:,} pairs have conditional prices; these bars use base rates only.</p>
      </section>
    </div>
    <section class="panel" id="cache-cost-example">
      <p class="eyebrow">Illustration, not observed usage</p>
      <h2>A big read discount does not automatically mean a small bill</h2>
      <p>Start with <strong>$100 of fresh-input cost</strong>. Assume cached reads cost
      <strong>10% of fresh input</strong> (a 90% read discount). Change the share of input tokens that gets a cache hit.</p>
      <label for="cache-hit-share">Assumed cached share: <output id="cache-hit-label" for="cache-hit-share">50%</output></label>
      <input class="cache-slider" id="cache-hit-share" type="range" min="0" max="100" step="1" value="50">
      <div class="cache-example-bars">
        <div class="cache-bar-row"><div class="cache-bar-label"><span>All input at the fresh rate</span><strong>$100.00</strong></div>
        <div class="cache-bar-track" aria-hidden="true"><span class="cache-bar-fill muted-bar" style="width:100%"></span></div></div>
        <div class="cache-bar-row"><div class="cache-bar-label"><span>With the assumed cache hits</span><strong id="cache-example-cost">$55.00</strong></div>
        <div class="cache-bar-track" aria-hidden="true"><span id="cache-example-fill" class="cache-bar-fill cyan" style="width:55%"></span></div></div>
      </div>
      <p id="cache-example-result" role="status">At a 50% cached share, input costs $55.00: $50.00 fresh + $5.00 cached reads.</p>
      <p class="note">Fixed illustrative rates, not a prediction or quote. Formula: $100 × [(1 − cached share) +
      0.10 × cached share]. Excludes cache writes, output tokens, provider routing, and conditional charges.
      A prompt-cache hit alone does not identify NVMe activity.</p>
      <noscript><p>The example starts at a 50% cached share; enable JavaScript to adjust it.</p></noscript>
    </section>'''


def _price(row, key):
    number = row["prices_per_million"][key]
    if number is None:
        label = "Invalid source value" if row["price_status"][key] == "invalid" else "Not reported"
        return '<span class="muted">' + label + "</span>"
    return "$" + format(number, ".6f").rstrip("0").rstrip(".")


def render_cache(cache_analysis: dict, provenance: dict) -> str:
    """Render the public catalogue and a precise cache/offload availability map."""
    from .render import _page

    summary = cache_analysis["summary"]
    meta = cache_analysis.get("metadata", {})
    catalog_date = escape(str(meta.get("as_of") or "Not recorded"))
    source_url = str(meta.get("source_url") or "https://openrouter.ai/api/v1/models?output_modalities=all")
    if not source_url.startswith("https://openrouter.ai/"):
        source_url = "https://openrouter.ai/api/v1/models?output_modalities=all"
    table_rows = []
    for row in cache_analysis["rows"]:
        discount = row["listed_read_discount_percent"]
        if discount is None:
            comparison = "N/A: zero input price" if row["zero_input_price"] else "Not calculable"
        else:
            comparison = f"{abs(discount):.1f}%" + (" lower" if discount > 0 else " higher" if discount < 0 else " (same price)")
        notes = []
        if row["has_overrides"]:
            notes.append("Conditional prices")
        if row["zero_input_price"]:
            notes.append("Zero listed input price")
        cache_reported = any(row["prices_per_million"][field] is not None for field in PRICE_FIELDS[1:])
        table_rows.append(
            '<tr data-cache-price="' + ("yes" if cache_reported else "no") + '"><th scope="row">'
            + '<a href="' + escape(row["source_url"], quote=True) + '">' + escape(row["name"])
            + '</a><br><small class="muted">' + escape(row["id"]) + "</small></th>"
            + "".join("<td>" + _price(row, key) + "</td>" for key in PRICE_FIELDS)
            + "<td>" + comparison + "</td><td>" + escape("; ".join(notes) or "—") + "</td></tr>"
        )
    sources = "".join(
        '<li><a href="' + escape(source["url"], quote=True) + '">' + escape(source["title"]) + "</a></li>"
        for source in SOURCES
    )
    content = f"""
    {_render_dashboard(cache_analysis)}
    <section class="panel">
      <h2>Cache and offload data availability</h2>
      <div class="table-wrap"><table><thead><tr><th>Signal</th><th>Source and access</th><th>Meaning for storage</th></tr></thead><tbody>
      <tr><th>Advertised cache-read/write price</th><td>Public Models API; collected here</td><td>Pricing signal only; no cached-token volume</td></tr>
      <tr><th>Provider implicit-cache support</th><td>Public endpoint metadata; not collected by this target</td><td>Capability flag; no hit rate or storage tier</td></tr>
      <tr><th>Provider effective price / cache-hit rate</th><td>Some public model pages; linked below, not scraped</td><td>Observed web summaries; definitions/windows can differ, no documented public history API identified</td></tr>
      <tr><th>Cached and cache-write tokens per request</th><td>Your generation or usage response</td><td>Prompt-token reuse, not NVMe bytes</td></tr>
      <tr><th>Account cache-hit history</th><td>Analytics API, management key; private account scope, not fetched here</td><td>Workload-specific reuse; no storage-tier attribution</td></tr>
      <tr><th>NVMe reads/writes, restore latency, evictions, checkpoint retention</th><td>Not exposed by these public APIs</td><td>Requires inference-engine / cache-manager / device telemetry</td></tr>
      </tbody></table></div>
      <p class="note">OpenRouter response caching stores a completed API response before contacting a provider.
      It is distinct from provider prompt caching and must not be counted as a KV restore.</p>
    </section>
    <section class="panel">
      <h2>Advertised prices across the complete catalog</h2>
      <details class="cache-catalog"><summary>Explore all {summary['models']:,} catalog entries and their listed prices</summary>
      <p>USD per million input tokens. <strong>Not reported</strong> is different from <strong>$0</strong>.
      The read comparison is <code>100 × (1 − listed cache-read price / listed input price)</code>.
      A zero input price has no meaningful percentage comparison. These catalog ratios exclude cache-write
      costs, provider routing differences, and conditional price overrides; they are not realized savings.</p>
      <p><label for="cache-model-search">Find a model </label><input id="cache-model-search" type="search"
      placeholder="Model, author, or variant" autocomplete="off">
      <label for="cache-price-filter">Price availability </label><select id="cache-price-filter">
      <option value="all">All entries</option><option value="yes">Any cache price reported</option>
      <option value="no">No cache price reported</option></select>
      <span id="cache-visible-count" class="muted" role="status">{summary['models']:,} entries</span></p>
      <div class="table-wrap model-table-wrap" tabindex="0" role="region" aria-label="Complete cache-price catalog"><table id="cache-price-table"><thead><tr><th>Model</th><th>Input / M</th>
      <th>Cache read / M</th><th>Cache write / M</th><th>1-hour write / M</th><th>Listed read comparison</th>
      <th>Conditions</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
      <p class="muted">{summary['models_with_overrides']:,} entries contain conditional pricing overrides.
      Open a model link to compare provider-specific rates and any available public cache statistics.</p>
      </details>
    </section>
    <section class="panel">
      <h2>Use this alongside offload measurements</h2>
      <p><a href="hybrid-trends.html">Architecture adoption</a> and
      <a href="context-demand.html">context demand</a> describe workload exposure. Prices describe an
      economic incentive to reuse prefixes. Estimating NVMe demand additionally requires reusable-token
      fractions, checkpoint sizes and spacing, retention, placement, and restore/write frequency.
      Use the <a href="hybrid-checkpoints.html">hybrid checkpoint model</a> and
      <a href="io-projections.html">I/O projections</a> with measured or explicitly assumed inputs.</p>
      <p>No byte-volume or offloading trend is calculated from advertised prices.</p>
    </section>
    <section class="panel">
      <h2>Sources and reproducibility</h2>
      <p>Source: <a href="{escape(source_url, quote=True)}">OpenRouter public Models API</a>.
      Retrieved {catalog_date}. Displayed values are a snapshot, not a live price quote. Refresh with
      <code>make openrouter-cache</code>; rebuild saved data offline with <code>make openrouter-render</code>.</p>
      <p>This is an independent kvcache-view analysis. OpenRouter is the source of catalog metadata;
      no endorsement is implied. The CC BY 4.0 attribution used for the rankings dataset on the trend
      pages is not asserted as a license for catalog metadata or documentation.</p>
      <ul class="source-list">{sources}</ul>
    </section>
    <script>
    (() => {{
      const search = document.getElementById('cache-model-search')
      const filter = document.getElementById('cache-price-filter')
      const rows = Array.from(document.querySelectorAll('#cache-price-table tbody tr'))
      const count = document.getElementById('cache-visible-count')
      function update() {{
        const needle = search.value.trim().toLowerCase()
        let visible = 0
        for (const row of rows) {{
          row.hidden = !row.textContent.toLowerCase().includes(needle) ||
            (filter.value !== 'all' && row.dataset.cachePrice !== filter.value)
          if (!row.hidden) visible += 1
        }}
        count.textContent = visible.toLocaleString() + ' entries'
      }}
      search.addEventListener('input', update)
      filter.addEventListener('change', update)
      const slider = document.getElementById('cache-hit-share')
      slider.addEventListener('input', () => {{
        const share = Number(slider.value)
        const fresh = 100 - share
        const cached = share * 0.10
        const cost = fresh + cached
        document.getElementById('cache-hit-label').textContent = share + '%'
        document.getElementById('cache-example-cost').textContent = '$' + cost.toFixed(2)
        document.getElementById('cache-example-fill').style.width = cost + '%'
        document.getElementById('cache-example-result').textContent = 'At a ' + share +
          '% cached share, input costs $' + cost.toFixed(2) + ': $' + fresh.toFixed(2) +
          ' fresh + $' + cached.toFixed(2) + ' cached reads.'
      }})
    }})()
    </script>
    """
    return _page(
        "Cache telemetry and pricing",
        "What public OpenRouter data can tell us about cache reuse, and what remains unknown about NVMe offloading.",
        content,
        provenance,
        analysis=cache_analysis,
        active="cache-telemetry.html",
    )
