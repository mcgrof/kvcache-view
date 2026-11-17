// SPDX-License-Identifier: MIT
;(function () {
    'use strict'
    const M = window.KVOffloadMath
    const D = window.KVOffloadData
    const $ = (id) => document.getElementById(id)
    const buckets = [
        { lo: 10, hi: 60, n: 38724, cached: 0.73, label: '10–60 seconds' },
        { lo: 60, hi: 300, n: 68027, cached: 0.71, label: '1–5 minutes' },
        { lo: 300, hi: 1800, n: 43169, cached: 0.21, label: '5–30 minutes' },
        { lo: 1800, hi: 3600, n: 7355, cached: 0.14, label: '30–60 minutes' },
        { lo: 3600, hi: 86400, n: 10589, cached: 0.07, label: '1–24 hours' },
    ]
    const money = (x) => {
        const a = Math.abs(x)
        return (
            (x < 0 ? '-$' : '$') +
            (a >= 1
                ? a.toFixed(2)
                : a > 0 && a < 1e-8
                  ? a.toExponential(2)
                  : a.toFixed(8).replace(/0+$/, '').replace(/\.$/, ''))
        )
    }
    const num = (x, places = 1) => x.toLocaleString('en-US', { maximumFractionDigits: places })
    const bytes = (x) =>
        x >= 1e12 ? num(x / 1e12, 2) + ' TB' : x >= 1e9 ? num(x / 1e9, 2) + ' GB' : num(x / 1e6, 2) + ' MB'
    const ms = (s) => (s >= 1 ? num(s, 3) + ' s' : num(s * 1000, 3) + ' ms')
    const days = (x) => (!Number.isFinite(x) ? 'No carrying cost' : num(x, x < 1 ? 3 : 1) + ' days')
    const read = (id) => Number($(id).value)
    const set = (id, value) => {
        $(id).textContent = value
    }
    let snapshot = null
    D.models.forEach((m) => $('model').add(new Option(m.name, m.id)))
    D.nodes.forEach((n) => $('node').add(new Option(n.name, n.id)))
    $('model').value = 'dsv41'
    $('node').value = 'rubin'
    const defaults = Object.fromEntries(
        [...document.querySelectorAll('input,select')].map((x) => [x.id, x.type === 'checkbox' ? x.checked : x.value]),
    )
    const total = buckets.reduce((n, b) => n + b.n, 0)
    $('gapRows').innerHTML = buckets
        .map(
            (b) =>
                `<tr><td>${b.label}</td><td>${num(b.n, 0)}</td><td>${num((b.n / total) * 100)}%</td><td>${num(b.cached * 100, 0)}%</td></tr>`,
        )
        .join('')
    $('gapChart').innerHTML = buckets
        .map(
            (b) =>
                `<div><span class="subtle">${b.label} · ${num((b.n / total) * 100)}%</span><div class="bar-track"><div class="bar-fill" style="width:${((b.n / total) * 100) / 0.5}%"></div></div></div>`,
        )
        .join('')
    $('modelSources').innerHTML =
        '<h3>Model and hardware sources</h3><ul>' +
        D.sources.map((s) => `<li><a href="${s.url}">${s.title}</a> — ${s.note}</li>`).join('') +
        '</ul>'

    function update() {
        try {
            for (const input of document.querySelectorAll('input[type=number]')) {
                if (
                    input.value.trim() === '' ||
                    !Number.isFinite(Number(input.value)) ||
                    (input.min !== '' && Number(input.value) < Number(input.min)) ||
                    (input.max !== '' && Number(input.value) > Number(input.max))
                ) {
                    throw new Error(
                        'Enter a valid value for ' +
                            input.parentElement.firstChild.textContent.trim() +
                            '. Previous results are retained until the input is valid.',
                    )
                }
            }
            const model = D.models.find((m) => m.id === $('model').value)
            const node = D.nodes.find((n) => n.id === $('node').value)
            const tokens = read('tokens')
            if (![tokens, read('drives'), read('appendTokens')].every(Number.isInteger))
                throw new Error('Context tokens, appended tokens, and drive count must be whole numbers.')
            const B = M.memory({ tokens, bytesPerToken: model.bytesPerToken, fixedBytes: model.fixedBytes })
            const F = M.prefill({ tokens, ...model, gpus: 8, denseTFLOPS: node.denseTFLOPS, mfu: read('mfu') })
            let trc = read('recomputeMs') / 1000
            if ($('computeMode').value === 'example') {
                trc = 0.026
                $('recomputeMs').value = '26'
            } else if ($('computeMode').value === 'flops') {
                trc = F.seconds
                $('recomputeMs').value = (trc * 1000).toFixed(6)
            }
            $('recomputeMs').readOnly = $('computeMode').value !== 'manual'
            const restore = M.restore({
                bytes: B,
                driveGBps: read('driveGBps'),
                drives: read('drives'),
                efficiency: read('ioEfficiency'),
                linkGBps: read('linkGBps'),
                fixedMs: read('fixedMs'),
                replayMs: read('replayMs'),
            })
            const writeBytes =
                $('writePolicy').value === 'full'
                    ? B
                    : M.memory({
                          tokens: Math.min(tokens, read('appendTokens')),
                          bytesPerToken: model.bytesPerToken,
                          fixedBytes: model.fixedBytes,
                      })
            const base = {
                bytes: B,
                nandPerGB: read('nandPrice'),
                years: read('years'),
                recomputeSeconds: trc,
                gpus: 8,
                gpuPerHour: read('gpuPrice'),
                dwpd: read('dwpd'),
                restoreSeconds: restore.seconds,
                overheadPerGBMonth: read('storageOverhead'),
            }
            const e = M.holding({ ...base, writeBytes: $('includeWear').checked ? writeBytes : 0 })
            const wear = M.holding({ ...base, writeBytes }).writeWearCost
            const scenarioBuckets = buckets.map((b) => ({
                ...b,
                missFraction: $('baseline').value === 'uncached' ? 1 : b.lo >= 300 ? 1 - b.cached : 0,
            }))
            const trace = M.traceScenario({
                buckets: scenarioBuckets,
                retentionSeconds: read('retention'),
                returnProbability: read('returnPct') / 100,
                validProbability: read('validPct') / 100,
                contextBytes: B,
                writeCost: e.writeWearCost,
                dailyHold: e.dailyHold,
                recomputeCost: e.recomputeCost,
                restoreCost: e.restoreCost,
                arrivalsPerDay: read('arrivals'),
            })
            const mode = $('computeMode').value
            const source =
                mode === 'example'
                    ? 'The 26 ms Rubin prefill is the supplied example, not a measured benchmark.'
                    : mode === 'flops'
                      ? 'Compute is an architecture arithmetic estimate, not measured serving latency.'
                      : 'Compute uses your entered latency; confirm its measurement boundary.'
            set(
                'scenarioNote',
                (model.weightsBytes + B > 8 * node.hbmGB * 1e9 * 0.9
                    ? 'INFEASIBLE RESIDENT MODEL: assumed weights plus this cache exceed the node budget. The displayed compute economics are hypothetical; weight offload or more GPUs would require a different model. '
                    : '') +
                    source +
                    ' ' +
                    model.storageStatus +
                    '. ' +
                    (tokens > model.contextLimit
                        ? 'This context exceeds the documented model limit; treat it as a hypothetical extension. '
                        : '') +
                    (model.id === 'dsv41' && read('replayMs') === 0
                        ? 'Replay is currently zero: the restore result is optimistic because SWA reconstruction takes time.'
                        : ''),
            )
            set('storageDays', days(e.daysPerRecompute))
            set(
                'headlineExplanation',
                `${bytes(B)} of storage at ${money(e.dailyHold)} per day would need to stay occupied for ${days(e.daysPerRecompute).toLowerCase()} to cost as much as the ${money(e.recomputeCost)} allocated to one recompute.`,
            )
            set('modelNotes', model.notes)
            set('contextSize', bytes(B))
            set(
                'contextDescription',
                `${num(tokens, 0)} tokens × ${num(model.bytesPerToken, 0)} B/token` +
                    (model.fixedBytes ? ` + ${bytes(model.fixedBytes)} checkpoint` : ''),
            )
            set('costDay', money(e.dailyHold))
            set('costRecompute', money(e.recomputeCost))
            set('computeDescription', `${ms(trc)} × 8 GPUs × ${money(read('gpuPrice'))}/GPU-hour`)
            const hours = read('years') * 365 * 24
            const steps = [
                `${num(tokens, 0)} tokens × ${num(model.bytesPerToken, 0)} bytes/token${model.fixedBytes ? ' + ' + bytes(model.fixedBytes) + ' fixed state' : ''} = <b>${bytes(B)}</b>.`,
                `${num(B / 1e9, 6)} GB × ${money(read('nandPrice'))}/GB = <b>${money(e.spacePurchase)}</b> of drive purchase cost allocated to this context.`,
                `${money(e.spacePurchase)} ÷ ${num(hours, 0)} lifetime hours = <b>${money(e.spacePurchase / hours)}/hour</b>, or <b>${money(e.spacePurchase / (hours / 24))}/day</b>${read('storageOverhead') ? '; with operating allocation: ' + money(e.dailyHold) + '/day' : ''}.`,
                `${num(trc, 6)} seconds ÷ 3,600 seconds/hour × 8 GPUs × ${money(read('gpuPrice'))}/GPU-hour = <b>${money(e.recomputeCost)}</b> for one recompute.`,
                `${money(e.recomputeCost)} ÷ ${money(e.dailyHold)}/day = <b>${days(e.daysPerRecompute)}</b> of storage for the same allocated cost.`,
            ]
            $('ledger').innerHTML = steps.map((s) => '<li><div>' + s + '</div></li>').join('')
            set('restoreTime', ms(restore.seconds))
            set(
                'restoreDetail',
                `${ms(restore.payloadSeconds)} payload + ${read('fixedMs')} ms fixed + ${read('replayMs')} ms replay; ${num(restore.effectiveGBps, 2)} GB/s effective`,
            )
            const ratio = trc / restore.seconds
            set('speedup', ratio >= 1 ? num(ratio, 1) + '× faster restore' : num(1 / ratio, 1) + '× slower restore')
            set(
                'latencyVerdict',
                'Recompute ' +
                    ms(trc) +
                    '; restore ' +
                    ms(restore.seconds) +
                    '. This compares resume paths, not whole-turn completion.',
            )
            set('netDays', days(e.netDays))
            const freeHBM = 8 * node.hbmGB * 1e9 * 0.9 - model.weightsBytes
            const hbmRate = freeHBM >= B ? (B / freeHBM) * 8 * read('gpuPrice') * 24 : null
            const dramRate = ((B / 1e9) * read('dramPrice')) / (365 * read('years'))
            const tiers = [
                ['HBM opportunity allocation', hbmRate, 0],
                ['DRAM capacity allocation', dramRate, 0],
                ['NVMe capacity + optional wear', e.dailyHold, e.writeWearCost],
            ]
            const periods = [300 / 86400, 1 / 24, 1, 3]
            $('tierRows').innerHTML = tiers
                .map(
                    ([name, rate, fixed]) =>
                        '<tr><td>' +
                        name +
                        '</td>' +
                        periods
                            .map((d) =>
                                rate === null
                                    ? '<td>Does not fit</td>'
                                    : '<td>' +
                                      money(rate * d + fixed) +
                                      '<br><span class="subtle">' +
                                      (e.recomputeCost > 0
                                          ? num(((rate * d + fixed) / e.recomputeCost) * 100, 2) + '% of recompute'
                                          : 'No recompute charge') +
                                      '</span></td>',
                            )
                            .join('') +
                        '</tr>',
                )
                .join('')
            set(
                'tierNote',
                `HBM is an opportunity-cost illustration: this stored image’s fraction of (90% of node HBM − ${bytes(model.weightsBytes)} assumed resident weights) × node hourly price. It is not measured live-state occupancy; SWA, indexer and workspace state can add HBM. Tier rows exclude restoration and assume no double allocation of the same stored copy.`,
            )
            const minProb = e.minimumReturnProbability
            set(
                'decisionText',
                `At a one-day wait, the all-recompute scenario needs ${Number.isFinite(minProb) ? num(minProb * 100, 2) + '% valid-return probability' : 'a restore path cheaper than recompute'} to cover holding, restore, and optional wear. ${e.netDays > 1 ? 'For short waits, capacity pressure and expected reuse can dominate the small NAND allocation.' : 'Here, restore or storage cost can erase the saving even before a long wait.'} This is a marginal decision; an evicted context may free space for a more valuable one.`,
            )
            set('expectedValue', money(trace.expectedNetPerArrival))
            set(
                'expectedDetail',
                `${money(trace.expectedNetPerArrival * read('arrivals'))}/day in this scenario; ${num(trace.caughtProbability * 100, 1)}% of arrivals return valid within the window`,
            )
            set('parkedSize', bytes(trace.parkedBytes))
            set(
                'parkedDetail',
                `${num(trace.parkedContexts, 0)} contexts on average; ${bytes(writeBytes * trace.writesPerDay)}/day written`,
            )
            set('avoided', num(trace.avoidedRecomputeEquivalentsPerDay, 0))
            const utilization = trace.parkedBytes / (read('capacityTB') * 1e12)
            const writeRate = (writeBytes * trace.writesPerDay) / (read('capacityTB') * 1e12)
            set(
                'capacityWarning',
                `${num(utilization * 100, 1)}% of the ${num(read('capacityTB'), 2)} TB capacity input is occupied on average; projected writes are ${num(writeRate, 3)} drive-writes/day versus ${num(read('dwpd'), 2)} DWPD. ${utilization > 1 ? 'This retention policy exceeds capacity and needs eviction or more space. ' : 'Average occupancy is not peak demand; burst headroom is still needed. '}${writeRate > read('dwpd') ? 'The write rate exceeds the configured endurance budget. ' : ''}${read('retention') > 86400 ? 'No reuse benefit is extrapolated beyond the displayed 24-hour bins; unobserved contexts continue to incur holding cost. ' : ''}All sizes use the selected fixed context, not the trace’s unknown joint distribution.`,
            )
            snapshot = {
                asOf: D.asOf,
                model,
                node,
                inputs: Object.fromEntries(
                    [...document.querySelectorAll('input,select')].map((x) => [
                        x.id,
                        x.type === 'checkbox' ? x.checked : x.value,
                    ]),
                ),
                computed: {
                    bytes: B,
                    prefillFlops: F.flops,
                    recomputeSeconds: trc,
                    restore,
                    holding: e,
                    optionalWearCost: wear,
                    writeBytes,
                    trace,
                },
                caveat: 'Scenario output, not a measured fleet result. See research/audit-agentic-kv-2026-11-17.md.',
            }
            set('inputError', '')
        } catch (err) {
            set('inputError', err.message)
        }
    }
    document.querySelectorAll('input,select').forEach((input) =>
        input.addEventListener('input', () => {
            if (['model', 'node', 'tokens'].includes(input.id) && $('computeMode').value === 'example')
                $('computeMode').value = 'flops'
            if (input.id === 'node') {
                const node = D.nodes.find((n) => n.id === input.value)
                $('gpuPrice').value = node.gpuPerHour
                $('driveGBps').value = node.driveGBps
                $('linkGBps').value = node.linkGBps
            }
            if (input.id === 'computeMode' && input.value === 'example') {
                $('model').value = 'dsv41'
                $('node').value = 'rubin'
                $('tokens').value = '80000'
                $('gpuPrice').value = '8'
            }
            update()
        }),
    )
    $('reset').addEventListener('click', () => {
        Object.entries(defaults).forEach(([id, val]) => {
            if ($(id).type === 'checkbox') $(id).checked = val
            else $(id).value = val
        })
        update()
    })
    $('download').addEventListener('click', () => {
        if (!snapshot || $('inputError').textContent) return
        const url = URL.createObjectURL(
            new Blob(
                [JSON.stringify(snapshot, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? String(v) : v), 2)],
                { type: 'application/json' },
            ),
        )
        const a = document.createElement('a')
        a.href = url
        a.download = 'agentic-kv-scenario.json'
        a.click()
        URL.revokeObjectURL(url)
    })
    update()
})()
