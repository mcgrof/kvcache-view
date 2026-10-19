# SPDX-License-Identifier: MIT
"""Pinned paper evidence and explicitly hypothetical memory accounting."""
from .context_page import heading
from .render import _escape

PAPER = 'https://arxiv.org/html/2608.28444v2'
# Selected matched observations, not cross-teacher aggregate recovery scores.
PAPER_ROWS = [
    ('Full attention', '#81b8ff', [75.6, 64.0, 98.6, 61.0]),
    ('SWA 64 + sinks', '#e9c687', [0.0, 12.8, 4.0, 32.7]),
    ('SWA 512 + sinks', '#f9a08a', [41.7, 39.6, 22.7, 44.2]),
    ('QRWKV7 retrofit', '#8be6c8', [29.6, 45.1, 10.9, 45.2]),
]


def paper_section():
    labels = ['GSM8K', 'HumanEval', 'RULER', 'BABILong · 16K']
    charts = []
    for i, label in enumerate(labels):
        bars = ''.join(f'<div class="paper-bar-row"><span>{name}</span><span class="paper-bar-track"><i style="width:{values[i]}%;background:{color}"></i></span><strong>{values[i]:.1f}</strong></div>' for name, color, values in PAPER_ROWS)
        charts.append(f'<div class="paper-mini"><h3>{label}</h3>{bars}</div>')
    return f'''<section class="study-panel" id="paper">
    {heading('01', 'Compare the quality you keep', 'One base model · Qwen2.5-7B-Instruct · reported scores, % · higher is better')}
    <p><a href="{PAPER}">Sliding-window beats linear attention, v2</a> tests training-free SWA against retrofits. Hybrids retaining full-attention layers are excluded. Appendix GDN uses about 100M distillation tokens; this does not test a natively pretrained GDN hybrid.</p>
    <div class="paper-grid">{''.join(charts)}</div>
    <p class="note">Tables 3 and 6. GSM8K: exact match; HumanEval: pass@1; RULER: task average; BABILong: QA1–20. SWA windows include four sinks. These are matched checkpoints, not matched state bytes.</p>
    <p class="study-boundary"><strong>Read the absolute scores.</strong> Small windows can fail generation. SWA’s RULER advantage still leaves a large gap to full attention; QRWKV7 wins this BABILong case. The 256K efficiency plot is a four-layer microbenchmark, not evidence of recall at that length.</p>
    </section>'''


def memory_section():
    return f'''<section class="study-panel" id="memory">
    {heading('03', 'A smaller state changes the curve', 'Illustrative accounting · equal attention dimensions · no quality or throughput equivalence implied')}
    <p>Full-attention layers still accumulate KV in a hybrid. A sliding-window layer caps its own KV; a recurrent layer keeps a fixed state. Optional historical memories, prefix snapshots, and speculative branches add separate terms.</p>
    <div class="memory-controls" hidden>
      <label>Context tokens<select id="memory-context"><option value="16384">16,384 · 16K</option><option value="131072" selected>131,072 · 128K</option><option value="1048576">1,048,576 · 1M</option></select></label>
      <label>Retained window, including sinks<select id="memory-window"><option value="64">64</option><option value="512" selected>512</option><option value="4096">4,096</option></select></label>
      <label>Full-attention layers in hybrids<select id="memory-full"><option value="0">0 / 32</option><option value="4">4 / 32</option><option value="8" selected>8 / 32</option><option value="16">16 / 32</option><option value="32">32 / 32</option></select></label>
      <label>Recurrent state per layer<select id="memory-state"><option value="0.25">0.25 MiB</option><option value="1" selected>1 MiB</option><option value="4">4 MiB</option><option value="16">16 MiB</option></select></label>
      <label>Concurrent sequences<select id="memory-batch"><option value="1" selected>1</option><option value="8">8</option><option value="32">32</option><option value="128">128</option></select></label>
    </div>
    <div id="memory-chart"></div>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Illustrative active state memory"><table><thead><tr><th>Design</th><th>State at 128K · one sequence</th><th>What grows with context?</th></tr></thead><tbody id="memory-values">
    <tr><th>Full attention</th><td>16 GiB</td><td>All 32 layers’ KV</td></tr>
    <tr><th>All-layer SWA</th><td>64 MiB</td><td>Nothing beyond the 512-token window</td></tr>
    <tr><th>All-layer recurrence</th><td>32 MiB</td><td>Nothing in the assumed fixed state</td></tr>
    <tr><th>SWA + full attention</th><td>4.047 GiB</td><td>8 full layers’ KV</td></tr>
    <tr><th>Recurrent / full hybrid</th><td>4.023 GiB</td><td>8 full layers’ KV</td></tr>
    </tbody></table></div>
    <p class="note">Default: 32 layers; 8 KV heads × 128 dimensions; 2 bytes per scalar; 512 retained tokens; 1 MiB recurrent state per layer; 8 full layers in each hybrid. These are hypothetical dimensions, not a named model’s configuration. K/M ticks below use powers of two.</p>
    <details class="study-details"><summary>Equations and exclusions</summary>
    <p>Per-layer KV bytes per retained token: <code>k = 2 × KV_heads × head_dimension × scalar_bytes = 4,096</code>. With context <code>N</code>, layers <code>L</code>, full layers <code>F</code>, window <code>w</code>, recurrent bytes per layer <code>R</code>, and sequences <code>B</code>:</p>
    <ul><li>Full: <code>B × L × k × N</code></li><li>SWA: <code>B × L × k × min(N,w)</code></li><li>Recurrent: <code>B × L × R</code></li><li>SWA + full attention: <code>B × [F × k × N + (L−F) × k × min(N,w)]</code></li><li>Recurrent/full hybrid: <code>B × [F × k × N + (L−F) × R]</code></li></ul>
    <p>R must include all persistent recurrent and convolution state in a real deployment. Excludes weights, activations, workspace, allocator slack, multimodal encoders, extra history, prefix snapshots and speculative branches. Active state alone is not total GPU memory. No bandwidth or storage-demand estimate is inferred.</p></details>
    <div class="memory-examples"><p><strong>Real SWA + full-attention example.</strong> <a href="https://arxiv.org/abs/2503.19786">Gemma 3</a> mixes sliding-window and full softmax attention. Its global layers keep a context-dependent cache.</p><p><strong>Real recurrent/global example.</strong> <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct">Qwen3-Next</a> uses three Gated DeltaNet layers per gated attention layer. It retains recurrent state and attention KV. Its dimensions differ from this illustration.</p></div>
    </section>'''


def research_section():
    return f'''<section class="study-panel" id="research">
    {heading('04', 'Find the cheapest memory that meets the task', 'An open research test sequence for memory caching and systems projections')}
    <p><strong>Memory caching</strong> is the optional retention and retrieval of useful model history beyond the active window or recurrent state. In this comparison, it includes historical state anchors and memory banks as well as conventional KV reuse. The next question is how much quality each retained byte buys, and where that byte can live.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Proposed experiment sequence"><table class="research-table"><thead><tr><th>Phase</th><th>Question and comparison</th><th>Decision evidence</th></tr></thead><tbody>
    <tr><th>1 · Recall baseline</th><td>Which tasks actually lose quality? Compare full attention, SWA with sinks, a native GDN hybrid, and a retrofit. Use matched teachers for intervention studies; report native checkpoints separately.</td><td>Sweep fact age, distractors, number of facts, context and generation length. Measure exact retrieval, multi-hop answers, code and agent success. Preserve full-attention scores.</td></tr>
    <tr><th>2 · Equal budgets</th><td>At equal active bytes and latency targets, which design retains the most useful history?</td><td>Sweep windows, recurrent capacity and full-layer fractions. Publish quality–memory frontiers, prefill cost, decode throughput and p95 latency on the same serving stack.</td></tr>
    <tr><th>3 · Optional memory</th><td>Do MARCH-style anchors, Memory Cache ideas, or Mixture of Memories repair a measured deficit on GDN? Does the same budget help SWA?</td><td>Add one mechanism at a time. Include metadata, lookup and write costs; test stale anchors, eviction, multiple sessions and adversarial fact placement. Only then test combinations and Titans/Atlas references.</td></tr>
    <tr><th>4 · Placement</th><td>Can useful history move from HBM to DRAM or NAND while meeting the quality and latency target?</td><td>Replay arrival traces. Measure bytes per snapshot, reuse distance, restore rate, read amplification, bandwidth, endurance, energy, concurrency and tail latency. Compare recomputation against restoration.</td></tr>
    </tbody></table></div>
    <div class="memory-examples"><p><strong>HBM and DRAM.</strong> Active KV and recurrent state compete with weights for capacity and bandwidth. Cheaper state can increase concurrency. Idle sessions and reusable snapshots may create a host-memory tier; measure restore deadlines before sizing it.</p><p><strong>NAND and storage.</strong> Persistent history may create a capacity opportunity only when reuse and latency justify it. Fixed active state can shrink the cache opportunity per session, while optional memories can expand retained history. Memory and storage projections must account for both effects.</p></div>
    <aside class="source-feature"><p class="source-label">Independent demand signal · OpenRouter</p><h3>Repeated context makes retention worth testing</h3><p><a href="https://openrouter.ai/data/newsletter/cache-is-crucial">Cache is Crucial</a> reports cached prompt tokens rising from about 60% of all agentic tokens in January to just under 90% in September. That measures reuse in OpenRouter’s agentic cohort; it does not reveal HBM, DRAM or NAND placement.</p><p class="note">Its cohort differs from the public rankings above. Do not multiply its cache rate by these token totals. <a href="context-demand.html">Context Demand</a>, <a href="cache-telemetry.html">Cache Telemetry</a>, <a href="hybrid-checkpoints.html">Hybrid Checkpoints</a> and <a href="io-projections.html">I/O Projections</a> provide the related demand and systems views.</p></aside>
    <p class="study-boundary"><strong>Closed models remain an evidence boundary.</strong> Public token counts and advertised context windows do not reveal a model’s attention operator, retained history, or serving policy. Keep unresolved traffic visible; compare API quality and latency separately from architecture claims. Architecture projections should carry that uncertainty rather than assign it to either design.</p>
    </section>'''
