# OpenRouter snapshots and generated research pages

These files support three public pages:

| Page | Update target | Data |
| --- | --- | --- |
| `hybrid-trends.html` | `make openrouter-hybrid-trends` | Daily text tokens grouped by reviewed sequence-mixer architecture |
| `context-demand.html` | `make openrouter-context` | Exact `100K` and `1M` API context buckets, plus overall text traffic |
| `cache-telemetry.html` | `make openrouter-cache` | Advertised cache-read/write prices and an account of which telemetry is available |

`make openrouter` fetches the data needed by all three targets once and
generates all three pages. It makes no inference calls and does not read
private account analytics. This is an on-demand update command, not a scheduler.

## Refresh and reproduce

Use Python 3.10 or later and Node/npm. Python uses only its standard library.
Prettier is the existing project development dependency and formats generated
HTML after the calculation completes.

```sh
npm ci
export OPEN_ROUTER_API  # set its value securely in your shell environment
make openrouter

# Rebuild the committed snapshot without an API key or network requests:
make openrouter-render OPENROUTER_END=2026-10-06

# Inspect a different historical cutoff from the same saved data:
make openrouter-render OPENROUTER_END=2026-09-30

# Refetch historical corrections, not just the recent overlap:
make openrouter OPENROUTER_FULL_REFRESH=1

# Run one collector/page, or the offline tests:
make openrouter-cache
make test-openrouter
```

The default end date is yesterday in **UTC**, matching the source API. Set
`OPENROUTER_END=YYYY-MM-DD` for an explicit cutoff. The default start is
`2025-01-01`; `OPENROUTER_START` can narrow it. An offline build does not pretend
that saved data is current: the page retains source timestamps, states the
requested cutoff, and suppresses growth comparisons whose calendar windows
have missing days. Use the committed snapshot's cutoff for an exact rebuild.

The collector requests at most one calendar year per API request. Normal
updates refresh the last seven saved days plus new days; old years remain
cached. Full refresh re-fetches every requested year. Historical gaps remain
visible until the source fills them and a full refresh retrieves them.
Query ranges and their original `meta.as_of` timestamps are retained in each
compressed yearly snapshot. Refreshed days replace complete old date slices;
rows are not appended or accumulated twice. One process lock serializes
simultaneous invocations in the same checkout. Writes use atomic file replacement.

The client spaces requests by at least 2.1 seconds and retries transient
errors with bounded backoff. A server-requested delay over 60 seconds stops
the command instead of blocking indefinitely. OpenRouter documents limits of
30 Data API requests/minute/key and 500/day/account. Other programs using the
same account share those limits. No separate Data API query price was found
in the reviewed documentation; this is not a promise that pricing cannot change.

The environment value is used only in an HTTPS Authorization header for
OpenRouter rankings requests. It is never put in generated HTML, JSON,
request URLs, logs, or command-line arguments. `.env` files are ignored.
Do not commit a key. The catalog request is public and needs no key.

## Files and interpretation

- `rankings/{text,100K,1M}/YYYY.json.gz`: source rows and update provenance.
  Counts retain their original integer/string values. gzip output has a fixed
  metadata timestamp to avoid unrelated binary changes on rebuild.
- `catalog.json`: complete public Models API snapshot, using
  `output_modalities=all`, with retrieval time and pagination provenance.
- `architecture-registry.json`: reviewed model IDs, architecture class,
  confidence, notes, and primary evidence links. This is research maintained
  by this project; OpenRouter does not supply these sequence-mixer classes.
- `hybrid-trends.json` and `context-demand.json`: derived statistics and full
  inventory. Per-page provenance manifests prevent a separate target run from
  replacing the attribution of another page.
- `cache-analysis.json`: advertised prices, explicit missing values, and
  measurement limits; `cache-provenance.json` identifies the catalog snapshot.

The initial snapshot and reviewed registry incorporate the architecture study
performed on 7 October 2026, extended here into a repeatable pipeline. Existing
historical text rows include retired models, avoiding a current-catalog-only
survivorship filter. The current catalog includes non-text endpoints too;
architecture adoption uses the `modality=text` rankings rather than treating
every catalog entry as a text model.

Only exact reviewed IDs and documented serving variants inherit architecture
evidence. A family name is not a rule for future releases: a new canonical model
stays unknown until its model card, configuration, or technical report is
reviewed and an explicit registry record is added. Preserve historical entries
when an API alias changes. `quality.*.unreviewed_models` identifies new ranked
models that need review. Keep undisclosed models unknown rather than silently
assigning them to regular attention.

MoE is independent of the sequence mixer. Sparse, sliding-window, compressed,
and latent softmax attention belong to regular attention unless the model also
has a verified Mamba or linear recurrent component. Other recurrent/convolution
hybrids remain separate. Free/batch variants retain distinct source traffic
rows; architecture lookup never duplicates their counts. Revealed previews
such as Ox Alpha are classified across their full observed history.

The daily top-50 cutoff hides the identities of remaining models in `other`.
Category token totals are observed minimums, not a census of architecture use.
Their growth rates are not lower bounds on true growth because cutoff entry
and exit change visibility. Absence is not measured zero demand. Windows need
all calendar days to produce growth comparisons; category absence suppresses
its growth result. Tokenizers differ by provider and totals mix input and
output. These are traffic proxies, not requests, compute, revenue, or cached bytes.

Context labels are API buckets for request context length. Their exact numeric
boundaries are not assumed. They are not model maximum-context capabilities.
No cache or storage volume is inferred from them.

## Cache and offloading boundaries

The Models API exposes advertised `input_cache_read`, `input_cache_write`, and
sometimes `input_cache_write_1h` prices. A missing price is unknown; an explicit
zero remains zero. Free input prices do not produce an invented discount ratio.
Conditional prices are marked and retained. Ratios of listed prices are not
observed savings; actual provider routing and pricing overrides matter.

The cache dashboard summarizes text-output catalog variants, including models
that also output other modalities. Price-reporting bars use that full text
denominator. The read-discount histogram and median use only positive input
prices paired with a reported cache-read price; they count variants equally,
not by traffic. Base rates are used even when conditional overrides exist.
The interactive cost example uses a fixed illustrative 90% read discount and
an adjustable assumed cached-token fraction, excluding writes and output fees.
All catalog entries remain available in the expandable, searchable price table.

Some public OpenRouter model webpages show cache-hit rates and effective prices.
No documented public historical API for those aggregate cache-hit measurements
was identified. This collector uses the documented catalog API rather than
depending on scraping private frontend internals. Account analytics and
per-request usage can expose cache metrics for authorized account traffic,
but those are not fetched or published by these make targets.

No reviewed public API exposes NVMe bytes, storage tier, recurrent checkpoint
retention, evictions, restores, or shared-document prefix overlap. The cache
page explicitly records these as unavailable, not zero. Use actual serving
measurements with `hybrid-checkpoints.html` and `io-projections.html` to model
offloading. Do not label advertised prices or prompt-cache hits as NVMe activity.

## Source attribution and licensing

**Rankings data:** Source: OpenRouter
([openrouter.ai/rankings](https://openrouter.ai/rankings)), as of each preserved
`meta.as_of` timestamp. Licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The generated rankings pages carry that attribution and identify this project's
classification, aggregation, and visualization as modifications. Retain the
source timestamps and attribution when redistributing the data or figures.

**Catalog metadata:** Source: [OpenRouter Models API](https://openrouter.ai/api/v1/models?output_modalities=all),
retrieved at `catalog.json` → `meta.as_of`. The Data API's CC BY grant is not
asserted to cover this separate catalog endpoint. Architecture evidence sources
retain their own terms. The new collector, calculations, templates, and UI code
use the project's MIT license, identified by SPDX headers.

Primary documentation:

- [Daily rankings schema and filters](https://openrouter.ai/docs/api/api-reference/datasets/daily-token-totals-for-top-50-models)
- [Data API scope, limits, and attribution](https://openrouter.ai/docs/cookbook/administration/data-api)
- [Models API](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
- [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
- [Account analytics](https://openrouter.ai/docs/cookbook/administration/analytics-cost-control)
