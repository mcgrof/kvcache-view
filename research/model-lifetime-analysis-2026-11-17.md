# Model lifetime snapshot — 2026-11-17

Public OpenRouter text rankings through 2026-11-16 UTC cover 683 of 685 days since January 2025. June 15 and July 15, 2025 are missing.

The matched Q1 entrant comparison does **not establish shortening or lengthening**. Mean time to first sustained top-50 exit, capped at 180 days, is 66.8 days (2025, n=42) versus 79.5 days (2026, n=19). The difference is +12.7 days with a bootstrap interval from −21.0 to +50.4. Excluding free variants gives 71.8 versus 73.75 days, also inconclusive. These cohorts include brief top-50 appearances.

Mean individually observed tokens in the first 180 days rise from 0.200T to 4.090T: **20.4× observed minima**, not an identified growth rate for complete lifetime demand. Corresponding medians are 4.86B and 982.72B, illustrating substantial skew. Platform daily reported text traffic grows from 1.014T in January 2026 to 24.641T in October. Token growth need not equal request or physical compute growth.

## Current ten: historical projection and demand scenario

These are the highest-ranked confirmed-open exact API identities. Dated versions and free variants remain separate. Current openness does not prove weights were open on every historical day. Pending Hy4, Hy4 Flash and ERNIE 5.5 Flash Preview releases are excluded; see the [openness audit](model-lifetime-openness-audit.md).

Remaining time below is the historical-peer median until sustained **overall top-50 exit**, not until leaving the open-only top ten. It is an unbacktested baseline. The page also shows historical variation and analogue support.

The table uses a transparent scenario: freeze the last seven-day average token rate until the median exit, then retain 10% of that rate for 180 further days. Thus tokens = observed minimum + daily rate × (median remaining days + 18). Request equivalents assume 4,000 input + 1,000 output tokens/call. Unknown historical traffic is excluded; this is incomplete past plus assumed future, not a measured full lifetime or guaranteed lower bound. The interactive page defaults to a shared editable 90-day prominence scenario instead.

| Exact API identity | Current age / median remaining days | Observed tokens (T) | Scenario tokens (T) | Estimated requests (B at 5K tokens) |
| --- | ---: | ---: | ---: | ---: |
| `deepseek/deepseek-v4.1-flash-20261017` | 31 / 127 | 209.74 | 1476.84 | 295.37 |
| `deepseek/deepseek-v4.1-flash-20260910` | 68 / 106 | 207.88 | 481.66 | 96.33 |
| `z-ai/glm-5.3-flash-20260826` | 83 / 109 | 149.79 | 363.98 | 72.80 |
| `stepfun/step-5-preview-20261008` | 40 / 121 | 44.69 | 170.98 | 34.20 |
| `xiaomi/mimo-v2.6-flash-20260921` | 56 / 106 | 52.68 | 126.45 | 25.29 |
| `nvidia/nemotron-3-ultra-550b-a55b-20260604:free` | 166 / 84 | 78.40 | 125.01 | 25.00 |
| `minimax/minimax-m3.1-flash-20261019` | 26 / 126 | 15.94 | 84.87 | 16.97 |
| `moonshotai/kimi-k3.1-pro-20261023` | 25 / 133 | 11.27 | 81.20 | 16.24 |
| `z-ai/glm-5.3-20260816` | 90 / 112 | 42.19 | 111.62 | 22.32 |
| `nvidia/nemotron-3.5-super-vl-20261015:free` | 33 / 123 | 10.81 | 70.61 | 14.12 |

## Former leaders

These confirmed exits are ordered by cumulative observed tokens, which favors models active in larger-platform eras. An exit does not establish zero demand.

| Exact API identity | Final observed date | Reported top-50 days | Observed tokens (T) |
| --- | --- | ---: | ---: |
| `tencent/hy3-preview-20260421` | 2026-07-08 | 61 | 27.21 |
| `tencent/hy3-20260706:free` | 2026-07-20 | 15 | 19.40 |
| `minimax/minimax-m2.5-20260211` | 2026-06-11 | 120 | 16.49 |
| `nvidia/nemotron-3-super-120b-a12b-20230311:free` | 2026-10-19 | 213 | 14.38 |
| `moonshotai/kimi-k2.5-0127` | 2026-08-05 | 184 | 11.31 |
| `minimax/minimax-m2.7-20260318` | 2026-08-24 | 155 | 11.26 |
| `moonshotai/kimi-k2.6-20260420` | 2026-09-03 | 116 | 10.81 |
| `xiaomi/mimo-v2.5-pro-20260422` | 2026-09-20 | 138 | 9.45 |
| `minimax/minimax-m3-20260531:free` | 2026-09-07 | 14 | 8.17 |
| `stepfun/step-3.5-flash:free` | 2026-04-08 | 66 | 7.97 |

Qwen2.5-72B last appears on November 19, 2025, with 184 reported days; 7B last appears November 9, with 133. Both returned after earlier sustained exits. The [existing Qwen2.5 study](https://knlp.io/qwen2.5-use-openrouter.html) measures roughly 14 months from release to final family appearance, a different clock.

## Inference demand in Sardana et al.

Figure 6 assumes 70 input + 215 output tokens/request: **1B paper-equivalent requests = 285B tokens**. One trillion tokens implies 3.51B such requests, versus 200M at 5K tokens/request. The public model dataset cannot identify which conversion matches actual requests. Request growth equals token growth divided by growth in average tokens/request.

`2 × N × inference tokens` is an uncached arithmetic proxy. With phase-dependent active parameters, the page uses `2 × (N_input × D_input + N_output × D_output)` and labels the input/output split as assumed. Attention, cache reuse, hardware efficiency, precision and batching prevent translating it directly into measured GPU-hours or cost.

[Research notes and related efforts](model-lifetime-sources.md) link Sardana et al., OpenRouter Model Half-Life, Vercel's production index, Birth/Life/Death of AI Models, Demirer et al., IAPS and the independent lifespan analysis. Their outcomes differ and are not pooled into this fit.

## Reproduce

```sh
make openrouter-lifetime OPENROUTER_OFFLINE=1 OPENROUTER_END=2026-11-16
make test-openrouter
node --test model-lifetime-math.test.js
```

Source: OpenRouter (https://openrouter.ai/rankings), as of 2026-11-17T16:38:36.551Z. Licensed under CC BY 4.0. Classification, survival calculations and scenarios are independent modifications. [Provenance](../data/openrouter/model-lifetime-provenance.json) and [method](../data/openrouter/README.md#model-lifetime-and-inference-demand-projection) are retained.
