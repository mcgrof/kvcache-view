// TP=1, no speculative state, BF16 attention and convolution history.
// Config revisions and vLLM shape formulas are linked on hybrid-checkpoints.html.
;(function (root) {
    'use strict'

    const MODELS = [
        {
            name: 'Qwen3.5-4B',
            repo: 'Qwen/Qwen3.5-4B',
            family: 'GDN + attention',
            recipe: true,
            rec: 24,
            attn: 8,
            ssm: 32 * 128 * 128,
            rowWidth: 128,
            conv: (2 * 16 * 128 + 32 * 128) * 3,
            kvPerToken: 4 * 256 * 2,
        },
        {
            name: 'Nemotron-H-4B',
            repo: 'nvidia/Nemotron-H-4B-Instruct-128K',
            family: 'Mamba-2 + attention',
            recipe: false,
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
            repo: 'ibm-granite/granite-4.0-h-micro',
            family: 'Mamba-2 + attention',
            recipe: false,
            rec: 36,
            attn: 4,
            ssm: 64 * 64 * 128,
            rowWidth: 128,
            conv: (2 * 2048 + 2 * 1 * 128) * 3,
            kvPerToken: 8 * 64 * 2,
        },
        {
            name: 'Falcon-H1-3B',
            repo: 'tiiuae/Falcon-H1-3B-Instruct',
            family: 'Parallel Mamba-2 + attention',
            recipe: false,
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

    // Prefill of one request running alone in vLLM's align mode, without
    // speculative decoding. Follows the base case of
    // Scheduler._mamba_block_aligned_split: a step that stops short of the
    // prompt end is rounded down to whole blocks, and no step runs past the
    // prompt's last block boundary. The optional prompt-tail, divergence-point
    // and backend-internal checkpoint stops are left out. The recurrent state
    // is only written at a step end, so only step ends on a boundary are saved.
    function alignedPrefill(promptLen, block, budget) {
        positiveInteger(promptLen, 'Prompt length')
        positiveInteger(block, 'Block size')
        positiveInteger(budget, 'Token budget')
        const lastBoundary = promptLen - (promptLen % block)
        const steps = []
        let start = 0
        while (start < promptLen) {
            let end = Math.min(start + budget, promptLen)
            if (end < promptLen) {
                const aligned = Math.floor(end / block) * block
                if (aligned > start || block <= budget) end = aligned
            }
            const nextBoundary = start % block === 0 ? 0 : (Math.floor(start / block) + 1) * block
            for (const stop of [nextBoundary, lastBoundary]) {
                if (start < stop && stop < end) end = stop
            }
            if (end <= start) throw new Error('Scheduler made no progress')
            steps.push({ start, end })
            start = end
        }
        const boundaries = []
        for (let p = block; p <= promptLen; p += block) boundaries.push(p)
        const checkpoints = steps.map((s) => s.end).filter((p) => p % block === 0)
        const saved = new Set(checkpoints)
        return { steps, boundaries, checkpoints, skipped: boundaries.filter((p) => !saved.has(p)) }
    }

    // Where a later request that shares the first `shared` tokens can resume.
    // vLLM reserves the last prompt token to compute logits, so an exact repeat
    // cannot resume at its own end. Recurrent layers need the one checkpoint at
    // the resume point; attention alone could resume at any cached block.
    function resumePoint(checkpoints, block, shared, nextLen) {
        positiveInteger(block, 'Block size')
        positiveInteger(nextLen, 'Next request length')
        if (!Number.isSafeInteger(shared) || shared < 0 || shared > nextLen) {
            throw new RangeError('Shared length must be between zero and the next request length')
        }
        const limit = Math.min(shared, nextLen - 1)
        const attention = Math.floor(limit / block) * block
        const hybrid = checkpoints.reduce((best, p) => (p <= limit && p > best ? p : best), 0)
        return { limit, attention, hybrid, recompute: shared - hybrid }
    }

    const api = { MODELS, geometry, storedCheckpoint, alignedPrefill, resumePoint }
    if (typeof module !== 'undefined' && module.exports) module.exports = api
    root.HybridCheckpoints = api
    if (typeof document === 'undefined') return

    const el = (id) => document.getElementById(id)
    const num = (n) => n.toLocaleString('en-US')
    const mib = (n) => `${(n / 1048576).toFixed(2)} MiB`
    const kib = (n) => `${num(n / 1024)} KiB`
    const text = (x, y, s, cls = 't', anchor = 'start') =>
        `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${s}</text>`
    const rect = (x, y, w, h, cls, rx = 0) =>
        `<rect x="${x}" y="${y}" width="${Math.max(w, 0)}" height="${h}" rx="${rx}" class="${cls}" />`
    const laneLabel = (y, title, sub) => text(16, y, title, 't-strong') + text(16, y + 16, sub, 't-muted')
    // Keep a label inside the plot when its anchor sits near either edge.
    const edgeAnchor = (x, lo, hi, margin = 70) => (x < lo + margin ? 'start' : x > hi - margin ? 'end' : 'middle')

    MODELS.forEach((m, i) => {
        const option = document.createElement('option')
        option.value = String(i)
        option.textContent = m.name
        el('model').appendChild(option)
    })

    const BUDGETS = [
        ['2n1', (n) => 2 * n - 1, (n) => `2N−1 = ${num(2 * n - 1)} (LMCache recipe)`],
        ['n', (n) => n, (n) => `N = ${num(n)}`],
        ['2048', () => 2048, () => '2,048'],
        ['8192', () => 8192, () => '8,192'],
        ['16384', () => 16384, () => '16,384'],
    ]

    function current() {
        const m = MODELS[Number(el('model').value)]
        const liveBytes = Number(el('live').value)
        const alignment = Number(el('alignment').value)
        return { m, liveBytes, alignment, g: geometry(m, liveBytes, alignment) }
    }

    function renderBudgets(g) {
        const select = el('budget')
        const previous = select.value || '2n1'
        const seen = new Set()
        select.replaceChildren()
        for (const [key, value, label] of BUDGETS) {
            const v = value(g.block)
            if (v < g.block || seen.has(v)) continue
            seen.add(v)
            const option = document.createElement('option')
            option.value = key
            option.dataset.tokens = String(v)
            option.textContent = label(g.block)
            select.appendChild(option)
        }
        select.value = [...select.options].some((o) => o.value === previous) ? previous : '2n1'
    }

    function renderPage({ m, alignment, g }) {
        const X0 = 170
        const W = 730
        const pool = ['rec', 'attn', 'rec', 'attn', 'attn', 'rec', 'attn', 'rec', 'attn', 'rec']
        let s = `<defs><pattern id="pad-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="pad-bg" /><line x1="0" y1="0" x2="0" y2="6" class="pad-line" /></pattern></defs>`
        s += laneLabel(40, 'GPU page pool', 'every page one size')
        pool.forEach((kind, i) => {
            const x = X0 + i * 74
            s += rect(x, 22, 66, 34, `pg-${kind}`, 5)
            s += text(x + 33, 43, kind === 'rec' ? '1 state' : 'N tokens', 't-dark', 'middle')
        })

        const recW = (W * g.recPage) / g.page
        const pct = ((100 * g.paddingBytes) / g.page).toFixed(2)
        s += laneLabel(112, 'Recurrent page', 'one checkpoint, one layer')
        s += rect(X0, 96, W, 40, 'pad-fill', 4) + rect(X0, 96, recW, 40, 'pg-rec', 4)
        s += text(X0 + 12, 121, `recurrent matrix + convolution history · ${mib(g.recPage)}`, 't-dark')
        s += text(X0 + W, 90, `padding to match: ${num(g.paddingBytes)} bytes (${pct}%)`, 't-orange', 'end')

        const stripes = g.block / alignment
        const sw = (W * alignment * g.attnPerToken) / g.page
        s += laneLabel(182, 'Attention page', `N = ${num(g.block)} tokens`)
        for (let i = 0; i < stripes; i++) {
            s += rect(X0 + i * sw, 166, sw > 4 ? sw - 1 : sw, 40, i % 2 ? 'pg-attn2' : 'pg-attn')
        }
        s += text(
            X0,
            226,
            `${num(stripes)} stripes of ${alignment} tokens · ${kib(g.attnPerToken)} of keys and values per token · ${mib(g.page)}`,
        )
        s += text(
            16,
            258,
            `N = ${alignment} × ceil(${num(g.recPage)} bytes ÷ (${alignment} × ${num(g.attnPerToken)} bytes)) = ${num(g.block)} tokens`,
            't-formula',
        )
        const svg = el('fig-page')
        svg.innerHTML = s
        svg.setAttribute(
            'aria-label',
            `For ${m.name}, one recurrent checkpoint is ${mib(g.recPage)} per layer, so an attention page must hold ${num(g.block)} tokens to be as large. N is ${num(g.block)} tokens.`,
        )
    }

    function renderAlign({ m, g }) {
        const L = Number(el('plen').value)
        const budget = Number(el('budget').selectedOptions[0].dataset.tokens)
        const exact = el('exact').checked
        const share = el('share')
        share.max = String(L)
        if (Number(share.value) > L) share.value = String(L)
        share.disabled = exact
        const shared = exact ? L : Number(share.value)
        const nextLen = exact ? L : Math.max(L, shared + 1)
        const p = alignedPrefill(L, g.block, budget)
        const r = resumePoint(p.checkpoints, g.block, shared, nextLen)
        el('plenv').textContent = `${num(L)} tokens`
        el('sharev').textContent = exact ? 'whole prompt' : `${num(shared)} tokens`

        const X0 = 170
        const W = 750
        const X1 = X0 + W
        const x = (t) => X0 + (W * t) / L
        let s = `<defs><pattern id="new-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="new-bg" /><line x1="0" y1="0" x2="0" y2="6" class="new-line" /></pattern></defs>`

        s += laneLabel(38, 'First request', 'prefill steps')
        p.steps.forEach((st, i) => {
            const w = x(st.end) - x(st.start)
            s += rect(x(st.start), 22, w, 32, i % 2 ? 'step-b' : 'step-a', 3)
            if (w >= 20) s += text(x(st.start) + w / 2, 43, String(i + 1), 't-dark', 'middle')
        })

        s += laneLabel(98, 'Block boundaries', 'checkpoint saved?')
        s += `<line x1="${X0}" y1="102" x2="${X1}" y2="102" class="axis" />`
        if (!p.boundaries.length) {
            s += text(
                X0 + W / 2,
                94,
                `No block boundary inside ${num(L)} tokens. N is ${num(g.block)}.`,
                't-orange',
                'middle',
            )
        }
        const spacing = (W * g.block) / L
        const labelEvery = Math.max(1, Math.ceil(56 / spacing))
        const radius = Math.max(2.5, Math.min(7, spacing / 2 - 1))
        const saved = new Set(p.checkpoints)
        p.boundaries.forEach((b, i) => {
            const cx = x(b)
            s += saved.has(b)
                ? `<circle cx="${cx}" cy="102" r="${radius}" class="ck-saved" />`
                : `<circle cx="${cx}" cy="102" r="${radius}" class="ck-skip" />`
            if ((i + 1) % labelEvery === 0) s += text(cx, 126, num(b), 't-small', edgeAnchor(cx, X0, X1, 20))
        })

        const exactNote = exact ? 'exact repeat' : `shares ${num(shared)} tokens`
        s += laneLabel(184, 'Next request', exactNote)
        s += rect(x(0), 168, x(r.hybrid) - x(0), 30, 'seg-reuse')
        s += rect(x(r.hybrid), 168, x(shared) - x(r.hybrid), 30, 'seg-recompute')
        if (shared < L) s += rect(x(shared), 168, X1 - x(shared), 30, 'seg-new')
        const inside = (a, b, label, cls) =>
            x(b) - x(a) > label.length * 7.5 + 12 ? text((x(a) + x(b)) / 2, 188, label, cls, 'middle') : ''
        s += inside(0, r.hybrid, 'reused', 't-dark')
        s += inside(r.hybrid, shared, 'recomputed', 't-dark')
        if (shared < L) s += inside(shared, L, 'new tokens', 't-small')
        const rx = x(r.hybrid)
        s += `<polygon points="${rx - 6},156 ${rx + 6},156 ${rx},165" class="pointer" />`
        s += text(rx, 150, `resumes at ${num(r.hybrid)}`, 't-strong', edgeAnchor(rx, X0, X1))

        s += laneLabel(234, 'Attention alone', 'could resume at')
        s += rect(x(0), 224, x(r.attention) - x(0), 12, 'seg-attn', 2)
        const ax = x(r.attention)
        s += ax > X1 - 60 ? text(ax - 6, 246, num(r.attention), 't', 'end') : text(ax + 6, 234, num(r.attention), 't')

        s += `<line x1="${X1}" y1="14" x2="${X1}" y2="254" class="end-line" />`
        s += text(X1, 10, 'prompt end', 't-small', 'end')
        s += text(X0, 274, '0', 't-small') + text(X1, 274, `${num(L)} tokens`, 't-small', 'end')

        const svg = el('fig-align')
        svg.innerHTML = s

        const n = p.steps.length
        let summary = `With a budget of ${num(budget)} tokens, prefill takes ${n} step${n === 1 ? '' : 's'}. `
        if (!p.boundaries.length) {
            summary += `The prompt is shorter than one block (N = ${num(g.block)}), so no checkpoint can exist. `
        } else if (!p.skipped.length) {
            summary += `Every block boundary falls at the end of a step, so all ${p.boundaries.length} get a checkpoint. `
        } else {
            summary += `${p.checkpoints.length} of ${p.boundaries.length} block boundaries get a checkpoint. The other ${p.skipped.length} fall inside a step, where the state is never written out. `
        }
        if (exact)
            summary += `An exact repeat must recompute its last token, so it can resume no later than ${num(r.limit)}. `
        summary +=
            r.hybrid === 0
                ? `The next request finds no checkpoint at or before token ${num(r.limit)}, so it recomputes all ${num(shared)} shared tokens.`
                : `The next request resumes at token ${num(r.hybrid)} and recomputes ${num(r.recompute)} shared tokens.`
        if (r.attention > r.hybrid) {
            summary += ` Attention alone could have resumed at ${num(r.attention)}; the recurrent layers hold it back.`
        }
        el('align-summary').textContent = summary
        svg.setAttribute('aria-label', `${m.name}, N = ${num(g.block)} tokens. ${summary}`)
    }

    function renderSettings({ m, g }) {
        el('settings-cmd').textContent = [
            `lmcache server --chunk-size ${g.block} --separate-object-groups`,
            '',
            `vllm serve ${m.repo} \\`,
            '    --enable-prefix-caching --mamba-cache-mode align \\',
            `    --max-num-batched-tokens ${2 * g.block - 1} \\`,
            `    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'`,
        ].join('\n')
        el('settings-note').textContent = m.recipe
            ? `LMCache publishes this recipe for GDN hybrids such as Qwen. Its example models list their own N values; the one above is this page's estimate for ${m.name}.`
            : `LMCache's published recipes cover GDN and Kimi linear-attention hybrids. A ${m.family} model such as ${m.name} is not among them, so treat these values as what the rule gives, not as a tested setup.`
    }

    function renderStorage({ m, liveBytes, g }) {
        const format = el('fmt').value
        const s = storedCheckpoint(m, liveBytes, format)
        const metrics = [
            ['Live checkpoint, one layer', mib(g.recPage), 'what each GPU page must hold'],
            [
                'Stored copy, one layer',
                mib(s.bytes),
                s.scaleBytes
                    ? 'recurrent matrix, scales and convolution history'
                    : 'recurrent matrix and convolution history',
            ],
            ['Stored copy, all layers', mib(s.bytes * m.rec), `${m.rec} recurrent layers; attention entries extra`],
            ['Stored size versus live', `${(g.recPage / s.bytes).toFixed(2)}×`, `N stays ${num(g.block)} tokens`],
        ]
        el('metrics').innerHTML = metrics
            .map(
                ([key, value, note]) =>
                    `<div class="metric"><div class="k">${key}</div><div class="v accent">${value}</div><div class="n">${note}</div></div>`,
            )
            .join('')
        const bar = (label, payload, color, detail, padding = false) =>
            `<div class="barrow"><div>${label}</div><div class="track" aria-hidden="true"><div class="seg ${color}" style="width:${(100 * payload) / g.page}%"></div>${padding ? `<div class="seg pad" style="width:${(100 * (g.page - payload)) / g.page}%"></div>` : ''}</div><div>${detail}</div></div>`
        el('pagebars').innerHTML =
            bar('Live recurrent page', g.recPage, 'rec', mib(g.page), true) +
            bar('Stored checkpoint', s.bytes, 'stored', mib(s.bytes))
        el('pagenote').textContent =
            `Per recurrent layer: ${num(m.ssm)} matrix values plus ${num(m.conv)} BF16 convolution values. The stored projection includes ${num(s.scaleBytes)} scale bytes and leaves out allocator padding and object headers. The GPU page, and therefore N, is sized from the live state, so N stays ${num(g.block)} tokens whichever stored format you pick.`
    }

    function renderTable({ m, liveBytes, alignment }) {
        const format = el('fmt').value
        el('tblb').innerHTML = MODELS.map(
            (x) =>
                `<tr${x === m ? ' class="sel"' : ''}><th scope="row">${x.name}</th><td class="num">${x.rec} / ${x.attn}</td><td class="num">${num(geometry(x, 4, alignment).block)}</td><td class="num">${num(geometry(x, 2, alignment).block)}</td><td class="num">${mib(storedCheckpoint(x, liveBytes, format).bytes * x.rec)}</td></tr>`,
        ).join('')
    }

    function renderAll() {
        const c = current()
        el('nval').textContent = num(c.g.block)
        el('modelnote').textContent =
            `${c.m.family}; ${c.m.rec} recurrent and ${c.m.attn} attention branches. TP=1, no speculation, BF16 attention keys and values and convolution history. Working-state precision and alignment are assumptions, not settings read from a running engine.`
        renderBudgets(c.g)
        renderPage(c)
        renderAlign(c)
        renderSettings(c)
        renderStorage(c)
        renderTable(c)
    }

    for (const id of ['model', 'live', 'alignment']) el(id).addEventListener('change', renderAll)
    for (const id of ['plen', 'share']) el(id).addEventListener('input', () => renderAlign(current()))
    for (const id of ['budget', 'exact']) el(id).addEventListener('change', () => renderAlign(current()))
    el('fmt').addEventListener('change', () => {
        const c = current()
        renderStorage(c)
        renderTable(c)
    })
    renderAll()
})(globalThis)
