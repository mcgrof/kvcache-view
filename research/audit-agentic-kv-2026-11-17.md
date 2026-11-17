# Agentic KV retention: audit and research notes

Reviewed on 2026-11-17. This note audits the supplied **“When the agent goes idle: keep the context or recompute it?”** prototype and documents the evidence and assumptions for the new projections. It is an analytical review, not a GPU benchmark or a replay of downloaded raw traces.

## What the calculation means

“Days of storage equal to one recompute” is a per-context allocation of storage capital cost. It does not estimate how long it takes to recover the purchase price of a drive. It also does not prescribe a retention lifetime.

The supplied V4.1 Flash/Rubin example is useful as an illustrative ledger, conditional on its **assumed 26 ms recomputation time**:

| Step | Calculation | Result |
| --- | --- | --- |
| Persisted context | 80,000 tokens × 890 bytes/token | 71,200,000 bytes = 71.2 MB |
| Allocated drive capital | 0.0712 GB × $0.10/GB | $0.00712 |
| Five-year life | 5 × 365 × 24 | 43,800 hours |
| Space allocation per hour | $0.00712 / 43,800 | $0.0000001626 |
| Space allocation per day | Hourly allocation × 24 | $0.000003901 |
| One recompute | 0.026 s × 8 GPUs × $8/GPU-hour / 3,600 | $0.0004622 |
| Equal-cost idle duration | Recompute cost / daily space allocation | 118.5 days, approximately 120 days |

Using the user's rounded $0.00047 recompute gives approximately 120.5 days. Decimal MB and GB are used throughout this example. The assumption is one successful future reuse, and this ledger excludes restore work, write work, capacity displacement, power, replication, and operational overhead.

The right interpretation is: **this context's allocated drive space would accumulate the cost of one recomputation only after roughly four months.** A minutes-to-hours reuse window is far shorter. Cheap space therefore supports retention, but it does not prove that retaining every context is useful. A context that never returns has no avoided recomputation; a slow restore can lose the latency comparison immediately.

The 26 ms example must remain separate from a hardware-derived forecast. Replacing a hardware peak, precision assumption, efficiency, or model FLOP estimate changes the forecast and its resulting idle duration; the worked example should not silently calibrate those inputs.

## Bugs and unsupported interpretations in the prototype

| Finding | Why it matters | Correction |
| --- | --- | --- |
| Storage duration called “break-even” without context | Easily mistaken for a drive purchase payback period | Lead with daily cost and “days of storage equal to one recompute”; show the arithmetic. |
| Every post-five-minute turn credited with a full recompute | Observed cache fractions remain nonzero | Show observed cache fractions; separate an all-recompute baseline from an incremental benefit over existing caching. |
| Five minutes described as a universal provider lifetime | Policies differ by provider, model, and configuration | Identify it as the observed cliff in this June 2026 deployment. |
| Distribution of returning turns treated as all idle contexts | Terminal sessions are absent from observed next-turn gaps | Add an explicit future-return probability; do not present it as measured by these bins. |
| Flat 0.92 validity multiplier | Same-model bins already exclude switching; switch and compaction percentages need not be disjoint | Expose compatibility as a scenario parameter with a stated population. Do not deduct model switches twice. |
| Median long-context size applied to all bucket counts | A median is not a mean; the cohorts differ | Label uniform-context calculations as synthetic scenarios. |
| Histogram end interpreted as no later reuse | An analysis window or displayed table cannot establish an unbounded reuse distribution | State the observed/table horizon and leave later-return behavior unknown. |
| Restore time omitted from expected monetary benefit | A read can consume billed GPU time, CPU time, and network resources | Include an explicit restore charge or explain its exclusion; show latency separately. |
| Four drive peaks simply summed | PCIe, network, staging, concurrency, and software may cap the path | Model effective end-to-end bandwidth and fixed overhead; treat drive count as an upper-bound input. |
| Append-only saving assumed for every idle transition | A cold context has no already-persisted prefix | Distinguish initial full write, append to a retained prefix, and whole-object rewrite. |
| Full capacity depreciation plus full endurance allocation | Both can allocate the same purchase cost | Keep wear separately identified, or provision against the larger capacity/endurance constraint. |
| HBM capacity only checked for resident weights | Positive leftover memory does not guarantee the selected context fits | Check weights, reserved workspace, and the complete resident state required by the selected serving scheme. |
| HBM opportunity cost mixed with storage depreciation | These are different accounting bases | Label HBM as a scenario for displaced serving capacity; do not present it as measured cash expense. |
| Sparse offloading said to remove the HBM row | GPU-resident indexer state and a live working set can remain | Model each persistent and active state component explicitly. |
| Very long duration rendered “indefinitely” | Every finite carrying rate eventually accumulates cost | Display the calculated duration or “beyond plotted horizon.” |
| User-editable zero/negative efficiencies or prices | Division by zero and NaN can produce misleading charts | Validate finite ranges and show unavailable quantities rather than silently drawing them. |
| Context slider rounds preset values | The stated 80K example can use a nearby value | Keep exact numeric context state; use the slider only as an input control. |
| Generic “both under a second” text | It is false for some model/context settings | Derive explanatory text from the current calculation. |

These are distinct issues. Repairing the labels alone would leave the expected-value and cohort problems in place.

## Trace provenance and exact bucket values

The bucket values below are from the author's published [KMesh workload analysis](https://knlp.io/kmesh-workloads.html), derived from the [Azure release](https://github.com/Azure/AzurePublicDataset/releases/tag/ghcp-coding-agent-2026). The [reproduction README](https://github.com/mcgrof/knlp/blob/main/tools/reproduce/kmesh_traces/README.md) pins the data revisions and archive digests. This review checked the published table and [analysis code](https://github.com/mcgrof/knlp/blob/main/tools/reproduce/kmesh_traces/copilot_resume.py), but did not rerun raw archives.

| Idle gap | Weekday count, June 2 | Cached fraction p50 / mean | Weekend count, June 6 | Cached fraction p50 / mean |
| --- | ---: | ---: | ---: | ---: |
| 10–60 seconds | 38,724 | 0.95 / 0.73 | 7,454 | 0.95 / 0.73 |
| 1–5 minutes | 68,027 | 0.94 / 0.71 | 12,617 | 0.95 / 0.72 |
| 5–30 minutes | 43,169 | 0.00 / 0.21 | 7,028 | 0.00 / 0.31 |
| 30–60 minutes | 7,355 | 0.00 / 0.14 | 991 | 0.00 / 0.25 |
| 1–24 hours | 10,589 | 0.00 / 0.07 | 1,422 | 0.00 / 0.12 |

These are same-model turn starts. The weekday 80K/164K prompt and 274/2,592-second gap percentiles concern a different subset: noncompacted, at least 32,768 tokens, and idle at least 60 seconds. Its post-five-minute resumes account for 19.2% of recomputed prompt tokens; 82% of their own prompt tokens are uncached. Neither percentage is a GPU-time fraction. Log-uniform interpolation inside each bucket is an assumption, not trace evidence.

The code interprets LLM timestamps as call ends after consistency checks, uses model calls rather than tool clocks, and detects compaction using a prompt-shrink heuristic. It defines a `>24h` bucket. These details matter if extending the analysis or reproducing the histogram.

The [original Copilot paper](https://arxiv.org/pdf/2608.00101), **Agentic Coding in the Wild**, studies a larger population than the released subset. Its broad finding is strong reuse within turns and much weaker reuse across turns. Preserve that population distinction: the [dataset documentation](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) describes a sampled, metadata-only release with anonymized model labels. It cannot identify the selected V4.1/GLM/Hy4/MiniMax models as the actual models serving these requests.

The [Weka corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) is another workload, not another percentile of Copilot. Its hash IDs are local to a session, and its filtering limits concurrent subagent groups. Keep Weka-derived context presets visibly separate from Copilot-derived timing assumptions. Equal text-block identifiers establish potential prefix reuse; they do not independently verify model, precision, layout, adapter, or checkpoint compatibility.

## Three separate decisions

### 1. Is restoration faster when the context is needed?

For context length `T`, let `F(T)` be recomputation FLOPs, `B(T)` the bytes that must be restored, `P` effective compute FLOP/s, `BW` effective transfer byte/s, and `L` fixed restore overhead:

```
recompute_seconds = F(T) / P
restore_seconds   = B(T) / BW + L + replay_seconds
```

This is a simplified throughput model. Measured prefill latency should replace `F/P` when available. Small kernels, collective communication, insufficient batching, memory traffic, metadata work, and scheduling are not automatically captured by a single MFU.

`replay_seconds` is important when a persisted representation omits sliding-window state or requires a recurrent checkpoint to be advanced. “Bytes saved” must describe a complete resumable object, not just its smallest component.

This comparison contains **no idle time**. A context can be cheap to store for months and still take too long to restore. Advance notice can hide part of restoration, but the prediction can fail and consume resources unnecessarily.

### 2. Is keeping it economically useful?

For an all-recompute baseline, define `D` as time to the next return, with `D = infinity` for no return; `R` as retention window; `V` as compatibility at return; `h` as carrying cost per second; `W` as incremental write cost; and `Cload` as incremental restoration cost:

```
expected_savings = E[1{D <= R and V} * (Crecompute - Cload)]
                   - W - h * E[min(D, R)]
```

This form allows return time, validity, and recomputation value to be correlated. If treating them as independent, state that simplification. A no-return context incurs `h*R`, even though it contributes no benefit. Negative expected savings should remain visible; the discard policy has zero incremental benefit and zero incremental storage cost.

Against an existing caching service, replace `Crecompute` with the **additional compute actually avoided**. Multiplying all FLOPs by an average uncached-token fraction is only an approximation: full-attention prefill cost is not linear in sequence length, and the relevant prefix/append decomposition matters.

For the guaranteed-return, carrying-cost-only illustration:

```
equal_cost_idle_seconds = Crecompute / h
```

If including one write and one restore, use `(Crecompute - W - Cload)/h`, provided the numerator is positive. Neither expression accounts for a competing object displaced from limited storage.

GPU dollars here are an allocated compute-price scenario, not provider retail token prices. An occupied GPU can do other useful work while a restore is pending; charging all eight GPUs for the entire transfer is another assumption and should be identified as such.

### 3. Does the retained working set fit, and can it be served?

Under a stationary arrival model, average occupancy follows a byte-weighted Little's-law calculation:

```
average_retained_bytes = idle_context_arrivals_per_second
                         * E[context_bytes * min(D, R)]
```

Using a fixed `B` simplifies this to `lambda * B * E[min(D,R)]`. That is a controlled scenario, not an estimate recovered from a median context size. Peak occupancy, bursts, replication, indexes, fragmented allocations, and headroom require separate treatment.

Bandwidth capacity also matters: average or burst restore demand can exceed the shared path even when one isolated restore is faster than recomputation. Track write volume, read volume, simultaneous resumptions, and storage/network queueing. Capacity pressure creates an opportunity cost that is absent from straight-line NAND depreciation.

## Research additions and their limits

| Source | Relevant evidence or mechanism | How to use it without overclaiming |
| --- | --- | --- |
| [Understanding KV Residency in Production-Scale Agentic LLM Workloads](https://arxiv.org/pdf/2610.22863), October 2026 | M365 production telemetry; idle retention and deployment placement are separate constraints. A one-hour cap reduces a modeled capacity ceiling 3.6× with recomputing-call share 4.07% → 4.53%. | Byte estimates use a public Llama-2-70B geometry, not confidential production layouts. Int4 retention is a replay that omits quality, transfer latency, and bandwidth costs. The paper motivates sensitivity controls, not a transferable speedup. |
| [DualPath](https://arxiv.org/pdf/2602.21548), February 2026 | Routes external KV through both prefill and decode engines to use otherwise-idle network capacity. | Model the bottleneck across the full storage/network/GPU path. Its serving-throughput improvement is not a multiplier for individual SSD reads. |
| [Strata](https://arxiv.org/pdf/2508.18572), OSDI 2026 | GPU-assisted I/O and cache-aware scheduling address fragmented layouts and loading stalls. | A sequential device specification does not establish achieved serving bandwidth; software layout and overlap are part of the comparison. |
| [ECHO](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf), OSDI 2026 | Host-backed native sparse attention; GPU stores indexer K and selected MLA KV. | This is DRAM-backed active decoding, not NVMe idle retention. Evaluation uses 8×H20 and AWQ DeepSeek-V3.2-Exp. Its 1.8M-token/~1TB host pool and high GPU selected-token hit rates are workload-specific; HBM is still required. |
| [CacheScout: Learning Agent Execution for KV-Cache Management in Agentic Serving](https://arxiv.org/pdf/2608.14624), 2026 | Learns agent transitions and uses future-use estimates for eviction and proactive warmup. | Tested with a six-agent supervisor on benchmark tasks. Its prefetch is proactive prefix computation, not SSD restoration. Use it to motivate return prediction, not a storage speedup factor. |
| [AgentKV](https://arxiv.org/pdf/2609.14872), September 2026 | Phase-aware eviction protects keys useful to future think/act/tool phases; compressed state persists across turns. | Lossy KV eviction changes the quality tradeoff. Throughput results do not validate equivalent behavior for all architectures or tasks. |
| [TokenDance](https://arxiv.org/pdf/2604.03143), April 2026 | Collective sharing and sibling differences in synchronized all-gather agent rounds. | Appropriate for that communication pattern; ordinary coding-session resumption does not imply those synchronized rounds. |
| [ForkKV](https://arxiv.org/pdf/2604.06370), April 2026 | Decomposes shared and LoRA-specific KV state with copy-on-write semantics. | Reuse across adapters requires a defined decomposition and serving support; it is not interchangeable with ordinary exact-prefix reuse. |

The latest residency paper is especially relevant to the forecast: it explicitly separates measured telemetry, estimated bytes, and replayed policies. The projections should maintain those distinctions as architectures change.

## Provider cache policy is a baseline, not hardware evidence

[Anthropic documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) describes five-minute and one-hour lifetimes. It measures lifetime from the start of the request that reads or writes an entry, so response generation consumes part of the available interval.

[OpenAI documentation](https://developers.openai.com/api/docs/guides/prompt-caching) describes different policies by model generation, including 30-minute/12-hour controls for newer models and older in-memory/extended modes. It explicitly distinguishes eligibility for cached pricing from guaranteed reuse of physical KV tensors or lower latency. Neither a billed cached-token count nor a TTL proves an NVMe hit.

These are current policy examples checked on the review date. The June Copilot observations remain a historical deployment measurement; newer policy documentation should not be used to relabel what its anonymized models did.

## Implications for the hardware and architecture projection

The naive projection should hold the model, context length, state representation, batch assumptions, and GPU count fixed while changing effective FLOPs and transfer bandwidth. Its result is a conditional crossover, not a prediction of the year when KV cache becomes obsolete.

The architecture-aware projection should change recomputation work and resumable state separately. Sparse attention can reduce computation without proportionally reducing retained state. Latent compression or low-bit state reduces transfer bytes. Recurrent or sliding-window models can bound some state while retaining other attention layers or requiring replay. Different model sizes and quality levels must not be presented as a controlled architectural speed comparison.

With zero fixed overhead, the latency ratio is:

```
recompute / restore = (F / B) * (BW / P)
```

Faster compute alone weakens the latency advantage of restoration. Faster I/O or fewer persisted bytes strengthens it. Architecture evolution changes `F/B`, so projecting FLOPs alone is incomplete. Fixed restore overhead eventually matters even when compressed state is tiny.

Cheap persistence, fast restoration, high return probability, and sufficient capacity are four different properties. A defensible view lets the reader see which one drives the answer.

## Validation boundaries

- Verify the approximately 120-day illustrative ledger independently of the hardware forecast.
- Test exact context presets, unit conversions, zero/invalid input handling, and infeasible HBM configurations.
- Check probability limits and retention endpoints, including contexts that never return.
- Check that changing GPU count, price, bandwidth, restore overhead, and state bytes moves results in the expected direction.
- Keep analytical assumptions, measurements, specifications, and future scenarios visibly labeled.
- Do not present a raw-bandwidth estimate as measured TTFT, a replayed miss reduction as measured serving throughput, or a sampled trace slice as the full deployment.

No new GPU performance, raw-trace reproduction, or deployed offloading experiment was performed for this note.
