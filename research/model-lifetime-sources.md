# Model lifetime: research sources and inference-demand interpretation

Research checked on 2026-11-17. This note records the primary literature and
the interpretation used to connect OpenRouter usage to lifetime inference
demand. The calculations below are our derivations, not additional empirical
findings from the cited papers.

## 1. What Sardana et al. mean by inference demand

Nikhil Sardana, Jacob Portes, Sasha Doubov and Jonathan Frankle,
**Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling
Laws**, ICML 2024, PMLR 235:43445–43460.

- [Proceedings record](https://proceedings.mlr.press/v235/sardana24a.html)
- [Conference PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/sardana24a/sardana24a.pdf)
- [July 2024 HTML version](https://arxiv.org/html/2401.00448v2)

Section 2 defines lifetime demand `D_inf` as the sum of **input plus output
tokens across requests**. For a dense transformer, its compute approximation
is:

```text
D_inf = sum_i(input_tokens_i + output_tokens_i)
C_training ≈ 6 × N × D_training
C_inference ≈ 2 × N × D_inf
C_total ≈ 6 × N × D_training + 2 × N × D_inf
```

Here `N` is parameter count and compute is measured in FLOPs. The paper
minimizes combined training and inference compute at a fixed target quality;
it assumes lifetime demand can be estimated before training.

Figure 6 uses **70 input and 215 output tokens per request**, with 50% training
MFU, 50% input MFU and 1% output MFU. Training uses A100-80GB hardware; inference
uses A100-40GB after INT8 quantization. Section 6 separates their costs:

```text
cost ≈ 6 × N × D_training × cost_per_training_FLOP / training_MFU
     + 2 × N × cost_per_inference_FLOP
           × (D_input / input_MFU + D_output / output_MFU)
```

Those are historical scenario assumptions, not current OpenRouter measurements.
The abstract's approximately one billion requests is a workload-dependent
example, not a universal adoption threshold.

## 2. Translating token totals into requests

Without request counts, a token series does not identify a request series.
If `T` is total prompt plus completion tokens and `L` is mean tokens per
request, then `estimated_requests = T / L`. The following are transparent
scenario calculations for the same **one trillion tokens**:

| Assumed workload | Input + output per request | Estimated requests per 1T tokens |
| --- | ---: | ---: |
| Sardana Figure 6 equivalent | 70 + 215 = 285 | 3.51 billion |
| Illustrative moderate context | 4,096 + 1,024 = 5,120 | 195 million |
| Illustrative long context | 32,768 + 2,048 = 34,816 | 28.7 million |

Consequently, **one billion paper-equivalent requests = 285 billion tokens**.
Changing the workload length can change inferred request counts by two orders
of magnitude with no change in the observed token total. Label the 285-token
conversion “paper-equivalent requests,” never “observed requests.” If the
source has actual request counts, display those and calculate the observed
tokens/request instead of replacing them with a preset.

An API request is not necessarily a user task. An agentic task may issue many
model calls; a retry is another call; a batched request may contain multiple
items. Request trends and task trends are therefore different measurements.

For two periods, the identity

```text
request_growth_factor = token_growth_factor / tokens_per_request_growth_factor
```

shows why tokens alone cannot establish how fast request demand grew.

## 3. Converting demand into approximate compute

Applying the paper's dense-model approximation, a 30B-parameter model serving
1T tokens performs about `6e22` model FLOPs, or **694 PFLOP-days**. This is a
unit conversion (`1 PFLOP-day = 8.64e19 FLOPs`), not an estimate of elapsed GPU
days. For a fixed dense model,
`C_inference / C_training ≈ D_inf / (3 × D_training)` under the same
approximations.

Our implementation recommendations:

- Describe `2 × parameters × tokens` as a compute proxy. It omits attention's
  sequence-length dependence and other architecture-specific operations.
- For a mixture-of-experts model, active parameters can provide a rough
  arithmetic proxy only when the count has a verified source. Total parameters
  remain relevant to weight storage. Do not silently use total MoE parameters
  as though every expert runs for every token.
- Served prompt tokens can include cached prefixes. Logical served tokens and
  newly computed tokens are different. Prompt-cache hit rates, the input/output
  split and reasoning-token accounting matter to physical compute.
- Model FLOPs are not GPU-hours, joules, dollars, or KV-cache byte-seconds.
  Hardware, precision, batching, latency targets, speculative decoding and
  memory/communication costs are additional inputs.
- Keep uncertain parameter counts or input/output splits visible as unknown.
  Do not make cost look measured by filling these fields with arbitrary values.

## 4. Closest direct lifecycle study

Aaron P. Kaye, Kazimier Smith and Neil Thompson, **Birth, Life, and Death of AI
Models**, CESifo Working Paper 13004, September 2026.

- [Paper record](https://www.ifo.de/en/cesifo/publications/2026/working-paper/birth-life-and-death-ai-models)
- [PDF](https://www.ifo.de/DocDL/cesifo1_wp13004.pdf)

The authors combine a weekly Hugging Face panel with OpenRouter tokens and
capability rankings. Their OpenRouter data begin in November 2024. Usage
obsolescence means delisting or token share permanently falling below the
bottom-decile threshold among **all** OpenRouter models. It is not top-50 exit.
For the matched Hugging Face sample, they report roughly 36 weeks to usage
obsolescence among models reaching it, versus nine weeks to technical
obsolescence. Figure 5.1 labels the usage median 35 weeks while the surrounding
text says 36; quote this as approximate. Only 89 of 321 matched models reach
usage obsolescence in their reported sample. Their median is consequently not
a completed-lifetime estimate for all models. Their daily totals use both
prompt and completion tokens; missing dates in their aggregate chart are
interpolated. Appendix A.2 documents alias matching and reports only moderate
correlation between Hugging Face downloads and OpenRouter token use.

**What we take from it:** distinguish capability, platform use, rank and
availability. Preserve active models as censored observations; do not define
their age as a completed lifetime. Keep direct token demand separate from
downloads and likes. Treat model-family grouping as an explicit analytical
choice: merging every dated checkpoint can hide genuine replacement.

## 5. Related OpenRouter market research

Mert Demirer, Andrey Fradkin and Nadav Tadelis, **The Emerging Market for
Intelligence: How Firms Buy and Sell AI**, Journal of Economic Perspectives
40(3):23–46, Summer 2026.

- [Published article](https://www.aeaweb.org/articles?id=10.1257/jep.20261506)
- [Replication package](https://doi.org/10.3886/E249283V1)
- [Exhibit data](https://www.aeaweb.org/articles/materials/25884)

This OpenRouter study documents frequent turnover among leading models,
rapid supply growth, falling prices and strong differentiation across use
cases. It provides evidence for market dynamism, not a universal model
expiration date. Its four-author predecessor with Sida Peng,
[The Emerging Market for Intelligence: Pricing, Supply, and Demand for LLMs](https://www.nber.org/papers/w34608),
also studies Azure and is a distinct December 2025 working paper. Avoid
mixing its authors, datasets and findings into the published article's
citation. The published article links replication and exhibit data; their
contents were not successfully retrieved in this research pass.

**What we take from it:** show absolute token volume alongside rank and
market share. A falling share in a rapidly expanding market can coexist with
growing inference volume. Category-specific comparisons, prices, free tiers
and provider availability can help explain replacement without claiming that
rank itself measures capability.

## 6. Related serving-cost research

Ege Erdil, **Inference economics of language models**, June 2025.

- [Research article and code links](https://epoch.ai/publications/inference-economics-of-language-models)
- [Paper](https://arxiv.org/abs/2506.04645)

Its serving model separates arithmetic, HBM transfers, network communication
and fixed latency, then studies speed/cost tradeoffs. This is a useful
complement to the scalar `2NT` approximation: identical arithmetic demand
need not imply identical serving cost. It is not a source for observed
OpenRouter lifetimes.

## 7. Forecasting rules derived for this projection

Other closely related public efforts complement the papers:

| Effort | What it measures | Lesson for this projection |
| --- | --- | --- |
| [OpenRouter, Model Half-Life, October 2026](https://openrouter.ai/data/newsletter/model-half-life) | Whether first-time users return to a lab in weeks 9–12; model launch frequency | Lab-user retention and model ranking lifetime are different outcomes. Its falling retention is useful context, not a fitted model-expiry parameter. |
| [Vercel AI Gateway Production Index, November 2026](https://vercel.com/blog/ai-gateway-production-index-november-2026) | Production-team adoption and movement away from models after peak use | Production teams also migrate quickly, but a second platform's sample cannot be substituted for OpenRouter's token series. |
| [IAPS-AI OpenRouter open/closed rankings](https://github.com/IAPS-AI/OpenRouter-OS-Rankings) | Archived daily rankings, weight availability and market share | Preserve source history and inspect weight availability. This projection retains exact dated/free IDs instead of merging every version into a base name. |
| [Takagi, Model Lifespan](https://note.com/yukio__takagi/n/ncb9b3126d2c5?hl=en) | Post-peak token-share retention, absolute volume and workload differences | Compare absolute tokens with share; a peak-aligned retention measure is neither a survival curve nor a deployment lifetime. |

Rank lifetime and inference demand need separate outcomes:

```text
top-K residence = sum of observed periods with rank <= K
first-spell duration = first entry through first confirmed exit
observed demand = sum of nonoverlapping observed token periods
projected lifetime demand = observed demand + projected future demand
```

A period outside the top 50 is not zero demand. A missing observation is not
proof of exit. Publication should state whether rank is among all models or
only verified open-weight models, how aliases/free variants are handled,
and whether re-entry restarts a spell. A confirmed exit needs a predeclared
persistence window rather than a single noisy daily ranking.

For optional token-decay scenarios, if the current token rate is `r0` per day
and its future half-life is `h` days, our exponential scenario yields:

```text
lambda = ln(2) / h
future_tokens_over_H_days = r0 × (1 - exp(-lambda × H)) / lambda
future_tokens_over_infinite_horizon = r0 × h / ln(2)
```

These are assumptions until fitted and backtested. A rank-exit forecast does
not supply the token half-life. A persistent positive usage floor makes the
infinite-horizon integral diverge, so a fixed forecast horizon is necessary
when modeling a long tail. Platform growth must not be counted twice by
fitting absolute token growth and then adding another market-growth factor.

To compare older and newer launch cohorts, use the same observation horizon
or a survival method that handles censoring, and report sample sizes and
coverage. Averaging only models already out of the ranking systematically
discards long survivors. Averaging the ages of currently popular models
answers a different question. Sparse archives should produce explicit
uncertainty or a statement that the requested historical trend is not yet
identified, rather than a manufactured precise average.
