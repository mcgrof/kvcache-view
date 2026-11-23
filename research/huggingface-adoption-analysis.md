# Hugging Face downloads and OpenRouter inference traffic

This enhancement asks whether models that receive more Hugging Face downloads
also receive more OpenRouter tokens in a nearby 30-day window. It adds an adoption
comparison to the [model lifetime projection](../model-lifetime.html#huggingface).
It does not convert downloads into inference requests or extend a model's predicted
ranking duration. The existing November 16 lifetime snapshot remains separate.

## November 23 result

**Downloads do not reliably rank current OpenRouter traffic in this sample.**
Of 143 mapped repositories, 138 returned a usable matching counter. Three API
responses resolved to different repository IDs and two returned HTTP 401; these
five were excluded from the statistics. The reviewed mapping combines 178
OpenRouter IDs and leaves 57 historically ranked open IDs without a usable mapping.
All 30 OpenRouter source days are present. New or unresolved identities are listed
in the coverage data rather than automatically classified as open.

| Comparison | Repositories | Spearman rank correlation (95% bootstrap interval) | Pearson on log(1 + count) |
|---|---:|---:|---:|
| Every known alias reported on all 30 days | 14 | 0.116 (−0.516 to 0.596) | 0.154 |
| Recently visible, using observed token minimums | 34 | −0.168 (−0.521 to 0.178) | −0.128 |

Both rank-correlation intervals include zero and both directions. The second row
has an additional problem: the amount of unreported traffic differs by model.
Neither result supports a reliable download-based adjustment to inference demand.
The primary sample also contains no models at or below 10B parameters, so it
cannot establish how downloads and hosted inference relate across all model sizes.
Its 14 repositories come from only seven organizations; related releases are
not independent replications of a platform-wide relationship.

The clearest practical addition is visibility into continued distribution after a
ranking exit:

| Hugging Face repository | Downloads in HF's trailing 30-day counter | Days individually reported by OpenRouter in its 30-day window |
|---|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | 6,420,358 | 0 |
| Qwen/Qwen2.5-Coder-32B-Instruct | 510,794 | 0 |
| Qwen/Qwen2.5-72B-Instruct | 158,647 | 0 |

Zero reported days means their mapped IDs were absent from the daily top 50;
their inference traffic was not observed as zero. Their older OpenRouter exits
therefore should not be described as the end of all interest in these models.

## What is measured

Hugging Face exposes trailing 30-day downloads, cumulative downloads, likes,
repository timestamps, and, for some models, a Safetensors parameter count.
The download counter includes qualifying GET and HEAD requests for selected
files, often configuration files. It is not a count of unique users, completed
weight installations, deployments, or inference calls. Cached weights can serve
many requests without another download, while automated checks can generate
repeated counted requests. [Counting rules](https://huggingface.co/docs/hub/models-download-stats)
and [API field definitions](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).

The saved Hugging Face snapshot was collected November 23, 2026. OpenRouter's
comparison covers October 24 through November 22, the nearest 30 completed UTC
days. HF does not supply the precise effective boundaries of its counter in the
model metadata response. The windows are nearby, not guaranteed identical.
Every request timestamp and the requested fields are retained in the raw snapshot.

## Matching and incomplete traffic

The unit is one Hugging Face repository. Only confirmed-open entries with explicit
repository mappings in the reviewed OpenRouter registry/catalog are eligible.
Paid/free API variants pointing to the same repository are grouped before counting
observations. Dated checkpoints are not merged by removing date suffixes. Conflicts,
unmapped identities, failed lookups, and missing counters are reported separately.
This is the reviewed subset of previously ranked models, not all of either platform.

On each covered OpenRouter day, named traffic is observed. A known API identity
outside the daily top 50 contributes unknown traffic, bounded by the rank-50
threshold and the aggregate `other` bucket. For a repository with several absent
aliases, their combined upper bound is also capped by that day's `other` bucket.
These upper bounds cannot be added across repositories: they share the same tail.
Missing source days leave the full-window upper bound unknown.

The primary comparison uses repositories whose known aliases were individually
reported throughout the window. It is still selected toward persistent ranking
leaders and covers known mapped aliases only. The secondary comparison includes
partially visible repositories that appeared at least once; its token coordinate
is an observed minimum. Models never individually reported in this window remain
in the table, but are excluded from correlations rather than assigned zero usage.

Spearman correlation measures whether the two rankings agree. Pearson correlation
on log(1 + count) is a sensitivity check that reduces the influence of scale.
Intervals resample whole repositories, not paid/free aliases. They describe
sampling variability in this selected set; they cannot repair mapping errors,
traffic censorship, common developer effects, or the mismatch between platforms.
Size comparisons use total parameters when HF reports them, not active MoE
parameters. Likes and cumulative downloads are context, not the headline inputs.

## Prior work and changes adopted here

**Kaye, Smith and Thompson, “Birth, Life, and Death of AI Models” (2026).**
[Appendix A.2, Figure A.1](https://www.ifo.de/DocDL/cesifo1_wp13004.pdf#page=51)
reports a correlation of **0.33** between logged cumulative HF downloads and
logged cumulative OpenRouter tokens, using 1,916 model-month observations for
251 models with positive counts on both platforms. That is direct precedent,
but a different comparison from our single-window repository sample. We keep
dated checkpoints separate and compare nearby flows instead of cumulative totals;
neither coefficient is a download-to-request conversion factor.

**Lambert and Brand, “The ATOM Report” (2026).**
[Paper](https://arxiv.org/html/2604.07190v1) and
[tracked model list](https://github.com/Interconnects-AI/tracked-models/blob/main/models.csv).
ATOM combines downloads, derivatives, and OpenRouter adoption; its relative
adoption metric compares models of similar size and age. This motivates our size
filters and separate adoption measures. Its historical HF series combines supplied
history with independently collected snapshots, rather than recovering history
from today's counter.

**Longpre et al., “Economies of Open Intelligence” (2025).**
[Paper](https://arxiv.org/html/2512.03073v1) and
[dashboard](https://huggingface.co/spaces/economies-open-ai/open-model-evolution).
The authors use historical internal download logs and distinguish their cleaned
research data from public counters. Their approach reinforces the need to document
counting rules and coverage before joining different archives.
The [public weekly panel](https://huggingface.co/datasets/mmpr/open_model_evolution_data)
is another concrete follow-up source; its model selection and public-counter
history differ from the paper's internal-log dataset.

**Model Pulse historical archive.**
[Dataset and processing notes](https://huggingface.co/datasets/modelpulse/model-pulse-data).
Its history includes interpolation, counter corrections, and repository-rename
handling. It could support a later historical comparison, but those processed
values must be distinguished from observed snapshots. We did not mix them into
this measurement or claim that a live metadata API provides historical counts.

**Platform change on November 16, 2026.**
[Hugging Face's announcement](https://huggingface.co/blog/openrouter) says its
language-model Inference Providers routing moved to OpenRouter. Future analyses
should mark this potential coverage break. The announcement alone does not tell
us exactly which routed traffic enters the public rankings. Downloads and hosted
inference should not be treated as independent measures of market-wide demand.

## How this improves the projections

The page now shows whether a model still attracts downloads alongside its
OpenRouter ranking history and demand scenario. Continued downloads after an
OpenRouter exit demonstrate ongoing distribution activity; they do not establish
how often the model runs elsewhere. In particular, a Qwen2.5 download count today
must not be used as if it had been known when the model first left the rankings.

The numerical lifetime and request projections retain their existing inputs.
To admit HF signals into a forecast, first build a dated panel and compare an
OpenRouter-only baseline against the same model with HF features. Train using
past dates, evaluate on later dates, group related checkpoints to prevent leakage,
and assess 30/90-day exit-probability calibration and demand error. Only a measured
held-out improvement would justify a forecast adjustment.

Keep cumulative snapshots for non-overlapping interval changes. Flag counter
decreases or repository renames; do not silently force them to zero. Differences
between successive rolling 30-day totals are not daily downloads: each difference
also removes an old day. No scheduled collection is configured by this commit.

## Reproduction

```sh
make openrouter-huggingface OPENROUTER_OFFLINE=1 \
    OPENROUTER_END=2026-11-22 HF_SNAPSHOT_DATE=2026-11-23
make test-openrouter
```

An online refresh uses the same make target with `OPEN_ROUTER_API` in the
environment and no offline flag. HF metadata needs no credential. Saved raw
snapshots, exact query URLs, attribution, and SHA-256 checksums accompany
[the derived JSON](../data/openrouter/huggingface-comparison.json).
