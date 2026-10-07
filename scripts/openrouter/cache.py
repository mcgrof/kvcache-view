# SPDX-License-Identifier: MIT
"""Summarize advertised public cache prices without inventing cache usage.

The Models API contains list prices, not storage-tier telemetry. Missing prices
remain missing, and an explicit zero is preserved as a reported zero price.
"""

from decimal import Decimal, InvalidOperation
from html import escape
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
    cards = "".join(
        '<div class="stat"><strong>' + f"{value:,}" + '</strong><span>' + label + "</span></div>"
        for value, label in (
            (summary["models"], "catalog variants"),
            (summary["cache_read_price_reported"], "report a cache-read price"),
            (summary["cache_write_price_reported"], "report a cache-write price"),
            (summary["no_cache_price_reported"], "report no cache price"),
        )
    )
    table_rows = []
    for row in cache_analysis["rows"]:
        discount = row["listed_read_discount_percent"]
        if discount is None:
            comparison = "N/A: zero input price" if row["zero_input_price"] else "Not calculable"
        else:
            comparison = f"{discount:.1f}%" + (" lower" if discount >= 0 else " (read premium)")
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
    <section class="panel">
      <p class="eyebrow">Measured availability, not inferred storage</p>
      <h2>What OpenRouter exposes</h2>
      <p>The public Models API provides advertised cache prices. Some public model pages also display
      cache-hit rates and effective prices. Neither identifies whether a cache hit came from GPU memory,
      host memory, NVMe, or remote storage. This page collects the documented catalog API and keeps
      unavailable usage and offload measurements empty.</p>
      <div class="grid">{cards}</div>
      <p class="muted">Catalog fetched: <time>{catalog_date}</time>. Counts describe catalog variants,
      including free variants and non-text modalities. A listed price is not proof that every route supports caching.</p>
    </section>
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
      <div class="table-wrap"><table id="cache-price-table"><thead><tr><th>Model</th><th>Input / M</th>
      <th>Cache read / M</th><th>Cache write / M</th><th>1-hour write / M</th><th>Listed read comparison</th>
      <th>Conditions</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
      <p class="muted">{summary['models_with_overrides']:,} entries contain conditional pricing overrides.
      Open a model link to compare provider-specific rates and any available public cache statistics.</p>
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
