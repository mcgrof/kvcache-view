// SPDX-License-Identifier: MIT
// Scenario controls and display. Quantitative formulas live in kv-offload-math.js.
;(function () {
    'use strict'
    if (typeof document === 'undefined') return
    const M = window.KVOffloadMath
    const D = window.KVOffloadData
    const $ = (id) => document.getElementById(id)
    const defaults = {
        bytes: 71.2,
        recompute: 26,
        bandwidth: 36.96,
        fixed: 1,
        'compute-growth': 50,
        'bandwidth-growth': 20,
        replay: 0,
        'fixed-bytes': 0,
        'context-growth': 0,
        quadratic: 0,
        'byte-decline': 30,
        'work-decline': 10,
    }
    let latest = null
    const number = (n, digits = 2) => n.toLocaleString('en-US', { maximumFractionDigits: digits })
    const milliseconds = (seconds) => {
        const ms = seconds * 1000
        return (ms < 0.01 ? ms.toExponential(2) : number(ms, ms < 1 ? 3 : 2)) + ' ms'
    }
    const ratio = (n) => (n < 0.01 ? n.toExponential(2) : number(n, 2)) + '×'
    const bytes = (n) => (n >= 1e9 ? number(n / 1e9, 2) + ' GB' : number(n / 1e6, 3) + ' MB')
    const crossingText = (points) => {
        const crossing = M.firstCrossover(points)
        if (crossing === null) return 'Beyond 2035'
        if (crossing <= 2026) return 'Already in 2026'
        return '≈ ' + crossing.toFixed(1).replace(/\.0$/, '')
    }
    const read = (key) => {
        const el = $('projection-' + key)
        const value = Number(el.value)
        if (el.value.trim() === '' || !Number.isFinite(value) || !el.checkValidity()) {
            throw new RangeError(
                'Enter a valid value for “' + el.closest('label').childNodes[0].textContent.trim() + '”.',
            )
        }
        return value
    }
    function modelTokens(model) {
        const tokenInput = $('projection-tokens')
        const tokens = Number(tokenInput.value)
        if (
            tokenInput.value.trim() === '' ||
            !Number.isInteger(tokens) ||
            tokens < 4096 ||
            tokens > model.contextLimit
        ) {
            throw new RangeError(
                'Use a whole-number context from 4,096 to ' + number(model.contextLimit, 0) + ' tokens for this model.',
            )
        }
        return tokens
    }
    function parameters() {
        const model = D.models.find((entry) => entry.id === $('projection-model').value)
        if (model) modelTokens(model)
        const values = Object.fromEntries(Object.keys(defaults).map((key) => [key, read(key)]))
        if (values['fixed-bytes'] > values.bytes)
            throw new RangeError('Fixed state must be no larger than the total saved state.')
        return {
            baseYear: 2026,
            endYear: 2035,
            recomputeSeconds: values.recompute / 1000,
            bytes: values.bytes * 1e6,
            bandwidthGBps: values.bandwidth,
            fixedMs: values.fixed,
            replayMs: values.replay,
            computeGrowth: values['compute-growth'] / 100,
            bandwidthGrowth: values['bandwidth-growth'] / 100,
            fixedBytes: values['fixed-bytes'] * 1e6,
            contextGrowth: values['context-growth'] / 100,
            quadraticFraction: values.quadratic / 100,
            byteDecline: values['byte-decline'] / 100,
            workDecline: values['work-decline'] / 100,
        }
    }
    function setModelPreset() {
        const model = D.models.find((entry) => entry.id === $('projection-model').value)
        if (!model) {
            $('projection-baseline-note').textContent =
                'Custom baseline: saved-state size and recompute latency are user inputs. Context tokens do not change a custom baseline. Match the same model, prompt and runtime when measuring both paths.'
            return render()
        }
        $('projection-tokens').max = model.contextLimit
        let tokens
        try {
            tokens = modelTokens(model)
        } catch {
            return render()
        }
        const config = {
            tokens,
            activeParams: model.activeParams,
            quadraticCoeff: model.quadraticCoeff,
            linearAttentionFlops: model.linearAttentionFlops,
            gpus: 8,
            denseTFLOPS: 17500,
            mfu: 0.44,
        }
        const estimated = M.prefill(config)
        const baseWork = M.prefill({ ...config, tokens: 80000 }).flops
        const timeMs = model.id === 'dsv41' ? (26 * estimated.flops) / baseWork : estimated.seconds * 1000
        $('projection-bytes').value =
            M.memory({ tokens, bytesPerToken: model.bytesPerToken, fixedBytes: model.fixedBytes }) / 1e6
        $('projection-fixed-bytes').value = model.fixedBytes / 1e6
        $('projection-recompute').value = Number(timeMs.toPrecision(7))
        const quadraticShare = ((model.quadraticCoeff * tokens * tokens) / estimated.flops) * 100
        $('projection-quadratic').value = Number(quadraticShare.toPrecision(6))
        $('projection-baseline-note').textContent =
            model.name +
            ': ' +
            (model.id === 'dsv41'
                ? '26 ms at 80K is the supplied scenario; other lengths scale by the rough FLOP model. '
                : 'Prefill is a structural FLOP estimate at 8 × 17.5 PFLOP/s dense FP8 and 44% efficiency, not a benchmark. ') +
            model.notes +
            ' Replay = 0 is an optimistic lower bound; enter measured reconstruction time.'
        render()
    }
    function showModelEvidence() {
        $('projection-model-values').innerHTML = D.models
            .map(
                (model) =>
                    '<tr><th scope="row"><a href="' +
                    model.source[0] +
                    '" target="_blank" rel="noopener noreferrer">' +
                    svgEscape(model.name) +
                    '</a></th><td>' +
                    number(model.bytesPerToken, 0) +
                    ' B</td><td>' +
                    bytes(model.fixedBytes) +
                    '</td><td>' +
                    bytes(
                        M.memory({ tokens: 80000, bytesPerToken: model.bytesPerToken, fixedBytes: model.fixedBytes }),
                    ) +
                    '</td><td>' +
                    svgEscape(model.storageStatus) +
                    '</td></tr>',
            )
            .join('')
    }
    const svgEscape = (s) =>
        String(s).replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' })[c])
    function drawChart(target, title, series, unit, reference) {
        const w = 1000,
            h = 350,
            left = 83,
            right = 30,
            top = 30,
            bottom = 48
        const all = series.flatMap((s) => s.points.map((p) => p.value))
        if (reference) all.push(reference)
        const low = Math.floor(Math.log10(Math.min(...all)))
        const high = Math.max(low + 1, Math.ceil(Math.log10(Math.max(...all))))
        const x = (year) => left + ((year - 2026) / 9) * (w - left - right)
        const y = (v) => top + ((high - Math.log10(v)) / (high - low)) * (h - top - bottom)
        let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="${target}-title ${target}-description"><title id="${target}-title">${svgEscape(title)}</title><desc id="${target}-description">Years 2026 through 2035 on the horizontal axis; logarithmic ${unit} on the vertical axis. Exact annual data follows in the table.</desc>`
        if (reference) {
            svg += `<rect x="${left}" y="${top}" width="${w - left - right}" height="${Math.max(0, y(reference) - top)}" fill="#59d9e9" opacity=".055"/>`
        }
        for (let exponent = low; exponent <= high; exponent++) {
            const v = 10 ** exponent
            const label = v < 0.01 || v >= 1e5 ? v.toExponential(0) : number(v, 3)
            svg += `<line x1="${left}" x2="${w - right}" y1="${y(v)}" y2="${y(v)}" stroke="#273951"/><text x="${left - 12}" y="${y(v) + 4}" text-anchor="end" fill="#a9bed5" font-size="13">${label}${unit === 'ratio' ? '×' : ''}</text>`
        }
        for (let year = 2026; year <= 2035; year++) {
            svg += `<line x1="${x(year)}" x2="${x(year)}" y1="${top}" y2="${h - bottom}" stroke="#1d2b40"/><text x="${x(year)}" y="${h - bottom + 24}" text-anchor="middle" fill="#a9bed5" font-size="13">${year}</text>`
        }
        if (reference)
            svg += `<line x1="${left}" x2="${w - right}" y1="${y(reference)}" y2="${y(reference)}" stroke="#d6e1ed" stroke-dasharray="4 5"/><text x="${left + 10}" y="${y(reference) - 8}" fill="#d6e1ed" font-size="12">1× · equal time</text>`
        series.forEach((s) => {
            const coords = s.points.map((p) => `${x(p.year)},${y(p.value)}`).join(' ')
            svg += `<polyline points="${coords}" fill="none" stroke="${s.color}" stroke-width="3"${s.dash ? ' stroke-dasharray="8 5"' : ''}/>`
            s.points.forEach((p) => {
                svg += `<circle cx="${x(p.year)}" cy="${y(p.value)}" r="3.5" fill="${s.color}"><title>${svgEscape(s.name)} · ${p.year}: ${number(p.value, 4)} ${unit}</title></circle>`
            })
        })
        svg += `<text x="${left}" y="17" fill="#a9bed5" font-size="12">${unit === 'ratio' ? 'Recompute ÷ restore · logarithmic' : 'Milliseconds · logarithmic'}</text><text x="${(left + w - right) / 2}" y="${h - 6}" text-anchor="middle" fill="#a9bed5" font-size="12">Year · hypothetical annual growth after 2026</text></svg>`
        $(target).innerHTML = svg
    }
    function render() {
        try {
            if (!M || !D) throw new Error('The calculation module could not load. Reload this page to retry.')
            const p = parameters()
            const naive = M.projection({ ...p, byteDecline: 0, workDecline: 0 })
            const evolving = M.projection(p)
            latest = { p, naive, evolving }
            $('projection-error').hidden = true
            $('projection-csv').disabled = false
            const first = naive[0],
                lastN = naive.at(-1),
                lastE = evolving.at(-1)
            $('projection-base-ratio').textContent = ratio(first.speedup)
            $('projection-base-detail').textContent =
                milliseconds(first.recomputeSeconds) +
                ' recompute; ' +
                milliseconds(first.restoreSeconds) +
                ' restore. ' +
                (first.speedup > 1 ? 'Restore is faster.' : 'Recompute is already as fast or faster.')
            $('projection-naive-year').textContent = crossingText(naive)
            $('projection-evolving-year').textContent = crossingText(evolving)
            $('projection-naive-detail').textContent =
                '2035: ' +
                ratio(lastN.speedup) +
                ' reuse advantage; ' +
                milliseconds(lastN.recomputeSeconds) +
                ' versus ' +
                milliseconds(lastN.restoreSeconds) +
                '.'
            $('projection-evolving-detail').textContent =
                '2035: ' + ratio(lastE.speedup) + ' reuse advantage; ' + bytes(lastE.bytes) + ' of saved state.'
            const direction =
                lastE.speedup > lastN.speedup * 1.001
                    ? 'extends the restore advantage'
                    : lastE.speedup < lastN.speedup / 1.001
                      ? 'reduces the restore advantage'
                      : 'leaves the restore advantage unchanged'
            const floorNote =
                p.byteDecline > p.workDecline && lastE.speedup < lastN.speedup
                    ? ' Here the fixed restore/replay floor overwhelms the benefit of shrinking bytes, while recompute keeps getting cheaper.'
                    : ''
            const contextNote =
                p.contextGrowth > 0
                    ? ' Both paths also include ' + number(p.contextGrowth * 100) + '% annual context growth.'
                    : ''
            $('projection-verdict').textContent =
                'Under these inputs, architecture evolution ' +
                direction +
                ' relative to the frozen model. At 2035 the ratio is ' +
                ratio(lastE.speedup) +
                ', versus ' +
                ratio(lastN.speedup) +
                ' without architecture change.' +
                floorNote +
                contextNote +
                ' This crossing applies to one reusable prefix and the selected path; it is not an expiry date for KV caching.'
            const line = (points, key) => points.map((point) => ({ year: point.year, value: point[key] * 1000 }))
            drawChart(
                'projection-latency-chart',
                'Projected recompute and restore latency for frozen and evolving architectures',
                [
                    { name: 'Frozen recompute', points: line(naive, 'recomputeSeconds'), color: '#ffbd71' },
                    { name: 'Frozen restore', points: line(naive, 'restoreSeconds'), color: '#59d9e9' },
                    {
                        name: 'Evolving recompute',
                        points: line(evolving, 'recomputeSeconds'),
                        color: '#ffbd71',
                        dash: true,
                    },
                    {
                        name: 'Evolving restore',
                        points: line(evolving, 'restoreSeconds'),
                        color: '#c0a6ff',
                        dash: true,
                    },
                ],
                'ms',
            )
            drawChart(
                'projection-ratio-chart',
                'Projected reuse advantage for frozen and evolving architectures',
                [
                    {
                        name: 'Frozen architecture',
                        points: naive.map((point) => ({ year: point.year, value: point.speedup })),
                        color: '#59d9e9',
                    },
                    {
                        name: 'Evolving architecture',
                        points: evolving.map((point) => ({ year: point.year, value: point.speedup })),
                        color: '#c0a6ff',
                        dash: true,
                    },
                ],
                'ratio',
                1,
            )
            $('projection-annual-values').innerHTML = naive
                .map((n, i) => {
                    const e = evolving[i]
                    return `<tr><th scope="row">${n.year}</th><td>${milliseconds(n.recomputeSeconds)}</td><td>${milliseconds(n.restoreSeconds)}</td><td class="${n.speedup > 1 ? 'positive' : 'negative'}">${ratio(n.speedup)}</td><td>${milliseconds(e.recomputeSeconds)}</td><td>${milliseconds(e.restoreSeconds)}</td><td class="${e.speedup > 1 ? 'positive' : 'negative'}">${ratio(e.speedup)}</td><td>${bytes(e.bytes)}</td></tr>`
                })
                .join('')
            $('projection-sensitivity-values').innerHTML = [0.25, 0.5, 0.75]
                .map(
                    (computeGrowth) =>
                        '<tr><th scope="row">+' +
                        number(computeGrowth * 100) +
                        '%/year</th>' +
                        [0.1, 0.2, 0.3]
                            .map((bandwidthGrowth) => {
                                const points = M.projection({
                                    ...p,
                                    byteDecline: 0,
                                    workDecline: 0,
                                    computeGrowth,
                                    bandwidthGrowth,
                                })
                                return `<td>${crossingText(points)}<small>${ratio(points.at(-1).speedup)} in 2035</small></td>`
                            })
                            .join('') +
                        '</tr>',
                )
                .join('')
        } catch (error) {
            $('projection-error').textContent = error.message + ' Charts retain the last valid scenario.'
            $('projection-error').hidden = false
            $('projection-csv').disabled = true
        }
    }
    function scenarioChanged() {
        const preset = {
            kv: [30, 10],
            compute: [10, 30],
            frozen: [0, 0],
        }[$('projection-scenario').value]
        if (preset) {
            $('projection-byte-decline').value = preset[0]
            $('projection-work-decline').value = preset[1]
        }
        render()
    }
    Object.keys(defaults).forEach((key) =>
        $('projection-' + key).addEventListener('input', () => {
            if (key === 'byte-decline' || key === 'work-decline') $('projection-scenario').value = 'custom'
            if (['bytes', 'recompute', 'fixed-bytes', 'quadratic'].includes(key)) {
                $('projection-model').value = 'custom'
                $('projection-baseline-note').textContent =
                    'Custom baseline: saved-state size and recompute latency are user inputs. Match the same model, prompt and runtime when measuring them. Replay must include any required state reconstruction.'
            }
            render()
        }),
    )
    $('projection-scenario').addEventListener('change', scenarioChanged)
    $('projection-model').addEventListener('change', setModelPreset)
    $('projection-tokens').addEventListener('input', setModelPreset)
    $('projection-form').addEventListener('submit', (event) => event.preventDefault())
    $('projection-reset').addEventListener('click', () => {
        Object.entries(defaults).forEach(([key, value]) => ($('projection-' + key).value = value))
        $('projection-scenario').value = 'kv'
        $('projection-model').value = 'dsv41'
        $('projection-tokens').value = 80000
        $('projection-tokens').max = 1000000
        setModelPreset()
    })
    $('projection-csv').addEventListener('click', () => {
        if (!latest) return
        const { p, naive, evolving } = latest
        const metadata = Object.entries(p)
            .map(([key, value]) => '# ' + key + ',' + value)
            .join('\n')
        const rows = naive.map((n, i) =>
            [
                n.year,
                n.recomputeSeconds,
                n.restoreSeconds,
                n.speedup,
                n.bytes,
                evolving[i].recomputeSeconds,
                evolving[i].restoreSeconds,
                evolving[i].speedup,
                evolving[i].bytes,
            ].join(','),
        )
        const csv =
            '# KV cache reuse sensitivity model; not a benchmark or forecast\n# Source review,2026-11-17\n' +
            metadata +
            '\nyear,frozen_recompute_seconds,frozen_restore_seconds,frozen_speedup,frozen_bytes,evolving_recompute_seconds,evolving_restore_seconds,evolving_speedup,evolving_bytes\n' +
            rows.join('\n') +
            '\n'
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
        const anchor = document.createElement('a')
        anchor.href = url
        anchor.download = 'kv-cache-reuse-projection.csv'
        document.body.append(anchor)
        anchor.click()
        anchor.remove()
        setTimeout(() => URL.revokeObjectURL(url), 1000)
    })
    if (M && D) {
        showModelEvidence()
        setModelPreset()
    } else render()
})()
