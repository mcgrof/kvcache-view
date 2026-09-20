// TP=1, no speculative state, BF16 attention and convolution history.
// Config revisions and vLLM shape formulas are linked on hybrid-checkpoints.html.
;(function (root) {
    'use strict'

    const MODELS = [
        {
            name: 'Qwen3.5-4B',
            family: 'GDN + attention',
            rec: 24,
            attn: 8,
            ssm: 32 * 128 * 128,
            rowWidth: 128,
            conv: (2 * 16 * 128 + 32 * 128) * 3,
            kvPerToken: 4 * 256 * 2,
        },
        {
            name: 'Nemotron-H-4B',
            family: 'Mamba-2 + attention',
            rec: 24,
            attn: 4,
            ssm: 112 * 64 * 128,
            rowWidth: 128,
            // Mamba intermediate width is heads * head_dim, not expand * hidden_size.
            // Both B and C have n_groups * state_size channels.
            conv: (112 * 64 + 2 * 8 * 128) * 3,
            kvPerToken: 8 * 128 * 2,
        },
        {
            name: 'Granite-4.0-h-micro',
            family: 'Mamba-2 + attention',
            rec: 36,
            attn: 4,
            ssm: 64 * 64 * 128,
            rowWidth: 128,
            conv: (2 * 2048 + 2 * 1 * 128) * 3,
            kvPerToken: 8 * 64 * 2,
        },
        {
            name: 'Falcon-H1-3B',
            family: 'Parallel Mamba-2 + attention',
            // Each of 32 decoder blocks has both branches, not 64 decoder blocks.
            rec: 32,
            attn: 32,
            ssm: 32 * 128 * 256,
            rowWidth: 256,
            conv: (4096 + 2 * 1 * 256) * 3,
            kvPerToken: 2 * 128 * 2,
        },
    ]

    function positiveInteger(value, name) {
        if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`)
    }

    // Simplified ordinary-attention branch of _align_hybrid_block_size.
    // alignment is the assumed effective granularity, including a requested block size.
    function geometry(model, liveBytes, alignment = 16) {
        if (![2, 4].includes(liveBytes)) throw new RangeError('Live state must use 2 or 4 bytes per value')
        positiveInteger(alignment, 'Alignment')
        const recPage = model.ssm * liveBytes + model.conv * 2
        const attnPerToken = model.kvPerToken * 2
        const block = alignment * Math.ceil(recPage / (alignment * attnPerToken))
        const page = block * attnPerToken
        return { recPage, attnPerToken, block, page, paddingBytes: page - recPage }
    }

    // Storage-only projection: does not alter live geometry or materialize checkpoints.
    // INT8 uses one FP32 scale per last-axis row; this is an accounting assumption,
    // not a claim that one codec/layout is supported or quality-safe on all models.
    function storedCheckpoint(model, liveBytes, format) {
        const bytes = format === 'native' ? liveBytes : format === 'bf16' ? 2 : format === 'int8' ? 1 : null
        if (bytes === null) throw new RangeError('Unknown storage format')
        const scaleBytes = format === 'int8' ? (model.ssm / model.rowWidth) * 4 : 0
        return {
            bytes: model.ssm * bytes + scaleBytes + model.conv * 2,
            scaleBytes,
        }
    }

    // Explicit hypothetical checkpoint schedule, not a simulation of vLLM's scheduler.
    // Requires every needed attention block and all recurrent components to be present.
    function reuse(length, block, everyBlocks, replayLastToken = true) {
        positiveInteger(length, 'Prefix length')
        positiveInteger(block, 'Block size')
        positiveInteger(everyBlocks, 'Checkpoint interval')
        const spacing = block * everyBlocks
        const limit = replayLastToken ? length - 1 : length
        const candidates = Math.floor(length / block)
        const stored = Math.floor(length / spacing)
        const usable = Math.floor(limit / spacing)
        const hit = usable * spacing
        return { candidates, stored, usable, hit, recompute: length - hit, spacing }
    }

    const api = { MODELS, geometry, storedCheckpoint, reuse }
    if (typeof module !== 'undefined' && module.exports) module.exports = api
    root.HybridCheckpoints = api
    if (typeof document === 'undefined') return

    const el = (id) => document.getElementById(id)
    const num = (n) => n.toLocaleString('en-US')
    const mib = (n) => `${(n / 1048576).toFixed(2)} MiB`
    MODELS.forEach((m, i) => {
        const option = document.createElement('option')
        option.value = String(i)
        option.textContent = m.name
        el('model').appendChild(option)
    })

    function render() {
        const m = MODELS[Number(el('model').value)]
        const liveBytes = Number(el('live').value)
        const alignment = Number(el('alignment').value)
        const length = Number(el('ctx').value)
        const every = Number(el('every').value)
        const format = el('fmt').value
        const replay = el('replay').checked
        const invalid = !Number.isSafeInteger(length) || length < 1 || length > 131072
        el('input-error').hidden = !invalid
        el('ctx').setAttribute('aria-invalid', String(invalid))
        if (invalid) return
        const g = geometry(m, liveBytes, alignment)
        const s = storedCheckpoint(m, liveBytes, format)
        const r = reuse(length, g.block, every, replay)
        el('modelnote').textContent =
            `${m.family}; ${m.rec} recurrent and ${m.attn} attention branches. TP=1, no speculation, BF16 attention KV and convolution history. Working-state precision and alignment below are assumptions, not detected engine settings.`
        const metrics = [
            ['Allocation block N', num(g.block), 'tokens; simplified aligned-page estimate'],
            ['Illustrative checkpoints', num(r.stored), `${r.usable} eligible for this request`],
            ['Reusable prefix', num(r.hit), `${num(r.recompute)} tokens to recompute in this scenario`],
            ['Live state payload', mib(g.recPage), 'one recurrent layer, including convolution history'],
            [
                'Stored recurrent snapshot',
                mib(s.bytes * m.rec),
                `all ${m.rec} recurrent branches; attention KV additional`,
            ],
            [
                'Serialization reduction',
                `${(g.recPage / s.bytes).toFixed(2)}×`,
                'relative to live payload; no GPU page savings implied',
            ],
        ]
        el('metrics').innerHTML = metrics
            .map(
                ([key, value, note]) =>
                    `<div class="metric"><div class="k">${key}</div><div class="v accent">${value}</div><div class="n">${note}</div></div>`,
            )
            .join('')

        for (const [id, spacing, stored] of [
            ['tl-grid', g.block, false],
            ['tl-stored', r.spacing, true],
        ]) {
            const timeline = el(id)
            timeline.replaceChildren()
            for (let p = spacing; p <= length; p += spacing) {
                const tick = document.createElement('div')
                tick.className = `tick${stored ? ' saved' : ''}${stored && p > r.hit ? ' excluded' : ''}`
                tick.style.left = `${(100 * p) / length}%`
                tick.title = `${num(p)} tokens${stored && p > r.hit ? ': excluded by final-token replay' : ''}`
                timeline.appendChild(tick)
            }
        }
        el('tlend').textContent = `${num(length)} tokens`
        el('tlnote').textContent =
            `${r.candidates} allocation boundaries, ${r.stored} assumed stored checkpoints, ${r.usable} eligible checkpoints. This example stores every ${every} block${every === 1 ? '' : 's'} (${num(r.spacing)} tokens). ${r.hit === 0 ? 'No eligible checkpoint in this scenario; prefill starts at token zero.' : `The latest eligible checkpoint is at ${num(r.hit)} tokens.`} ${replay ? 'One prompt token is reserved to obtain logits; a checkpoint exactly at the prompt end is excluded.' : 'Final-token replay is disabled: this assumes a consumer can resume at the prompt end without recomputing logits.'}`

        const bar = (label, payload, total, color, detail, padding = false) =>
            `<div class="barrow"><div>${label}</div><div class="track" aria-hidden="true"><div class="seg ${color}" style="width:${(100 * payload) / g.page}%"></div>${padding ? `<div class="seg pad" style="width:${(100 * (total - payload)) / g.page}%"></div>` : ''}</div><div>${detail}</div></div>`
        el('pagebars').innerHTML =
            bar('Live recurrent page', g.recPage, g.page, 'rec', mib(g.page), true) +
            bar('Attention page', g.page, g.page, 'attn', `${num(g.block)} tokens`) +
            bar('Stored checkpoint', s.bytes, s.bytes, 'stored', mib(s.bytes))
        el('pagenote').textContent =
            `Per recurrent layer: ${num(m.ssm)} matrix values plus ${num(m.conv)} BF16 convolution values. The live page adds ${num(g.paddingBytes)} padding bytes (${((100 * g.paddingBytes) / g.page).toFixed(2)}% of its allocation). The stored projection includes ${num(s.scaleBytes)} scale bytes and omits allocator padding and object headers. Changing storage format leaves N=${num(g.block)} unchanged.`

        el('tblb').innerHTML = MODELS.map(
            (x) =>
                `<tr${x === m ? ' class="sel"' : ''}><th scope="row">${x.name}</th><td class="num">${x.rec} / ${x.attn}</td><td class="num">${num(geometry(x, 4, alignment).block)}</td><td class="num">${num(geometry(x, 2, alignment).block)}</td><td class="num">${mib(storedCheckpoint(x, liveBytes, format).bytes * x.rec)}</td></tr>`,
        ).join('')
        el('chunknote').textContent =
            `For the documented LMCache MP unified-group recipe, use the engine's resolved N, then a storage chunk of N or an integer multiple. This calculator currently estimates N=${num(g.block)}; 256 ${256 % g.block === 0 ? 'is' : 'is not'} a multiple of that estimate. This is an arithmetic check, not validation of an installed connector. Matching chunk sizes cannot create missing checkpoints.`
    }

    for (const id of ['model', 'live', 'fmt', 'alignment', 'every', 'replay']) el(id).addEventListener('change', render)
    el('ctx').addEventListener('input', render)
    render()
})(globalThis)
