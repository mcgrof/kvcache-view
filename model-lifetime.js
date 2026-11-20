// SPDX-License-Identifier: MIT
;(function () {
    'use strict'
    const $ = (id) => document.getElementById(id)
    const esc = (value) =>
        String(value ?? '').replace(
            /[&<>"']/g,
            (x) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[x],
        )
    const colors = ['#69d9e8', '#bd9bff', '#ffc17b', '#83d7b4']
    let data, selected, lastScenario
    let historyMode = 'tokens'
    const compact = (value, digits = 2) => {
        if (value == null || !Number.isFinite(Number(value))) return 'Unknown'
        const n = Number(value)
        for (const [divisor, unit] of [
            [1e15, 'Q'],
            [1e12, 'T'],
            [1e9, 'B'],
            [1e6, 'M'],
            [1e3, 'K'],
        ]) {
            if (Math.abs(n) >= divisor)
                return `${(n / divisor).toFixed(digits).replace(/\.0+$|(?<=\.[0-9])0+$/g, '')}${unit}`
        }
        return n.toLocaleString('en-US', { maximumFractionDigits: digits })
    }
    const fixed = (n, digits = 1) =>
        n == null || !Number.isFinite(Number(n)) ? 'Not estimable' : Number(n).toFixed(digits)
    const dayLabel = (n) => (n == null ? 'Not estimable' : `${Math.round(n)} days`)
    const dateShort = (s) =>
        s
            ? new Date(s + 'T00:00:00Z').toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                  timeZone: 'UTC',
              })
            : 'Unknown'
    const metric = (label, value, detail) =>
        `<article class="metric"><span class="label">${esc(label)}</span><span class="value">${esc(value)}</span><span class="detail">${esc(detail)}</span></article>`
    const frame = (title, width, height, content) =>
        `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${content}</svg>`
    const line = (x1, y1, x2, y2, cls = 'grid') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`
    const label = (x, y, text, extra = '') => `<text x="${x}" y="${y}" ${extra}>${esc(text)}</text>`

    function survivalChart() {
        const w = 1000,
            h = 310,
            left = 60,
            right = 22,
            top = 22,
            bottom = 48
        const xx = (v) => left + (v / 180) * (w - left - right)
        const yy = (v) => h - bottom - v * (h - top - bottom)
        let svg = ''
        for (const y of [0, 0.25, 0.5, 0.75, 1])
            svg += line(left, yy(y), w - right, yy(y)) + label(left - 10, yy(y) + 4, `${y * 100}%`, 'text-anchor="end"')
        for (const x of [0, 30, 60, 90, 120, 150, 180]) svg += label(xx(x), h - bottom + 23, x, 'text-anchor="middle"')
        svg += label(w / 2, h - 6, 'Days since first observed top-50 entry', 'text-anchor="middle"')
        data.cohorts.forEach((cohort, i) => {
            let path = `M ${xx(0)} ${yy(1)}`
            let current = 1,
                lastDay = 0
            for (const point of cohort.survival || []) {
                if (point.day > 180) break
                path += ` H ${xx(point.day)} V ${yy(point.survival)}`
                current = point.survival
                lastDay = point.day
            }
            if (lastDay < 180) path += ` H ${xx(180)}`
            svg += `<path d="${path}" stroke="${colors[i % colors.length]}" stroke-width="3" fill="none"/>`
        })
        $('survival-chart').innerHTML = frame(
            'Share of models that have not yet begun their first confirmed 28-day absence from the top 50',
            w,
            h,
            svg,
        )
        $('cohort-legend').innerHTML = data.cohorts
            .map(
                (c, i) =>
                    `<span><i style="border-color:${colors[i % colors.length]}"></i>${esc(c.label)} · n=${c.n}</span>`,
            )
            .join('')
    }

    function cohortSection() {
        const cohorts = data.cohorts || []
        const comparison = data.cohort_comparison || {}
        if (comparison.ci && comparison.ci[0] <= 0 && comparison.ci[1] >= 0)
            $('cohort-conclusion').textContent =
                'No clear evidence of a change. The uncertainty includes both shorter and longer spans. These data do not establish that models stayed in the top 50 longer in 2026.'
        else if (comparison.conclusion) $('cohort-conclusion').textContent = comparison.conclusion
        if (cohorts.length >= 2) {
            const first = cohorts[0],
                last = cohorts.at(-1)
            $('cohort-demand').innerHTML =
                `<strong>${fixed(comparison.mean_observed_180d_tokens_ratio)}× more observed tokens per entrant</strong><p>The mean token minimum in the first 180 days rose from ${compact(first.mean_observed_180d_tokens)} (${esc(first.label)}) to ${compact(last.mean_observed_180d_tokens)} (${esc(last.label)}). This is a ratio of observed minima, not a lower bound on the growth of true demand. Longer prompts, greater adoption, or a different model mix can all contribute.</p>`
        }
        const cards = cohorts.map((c) =>
            metric(c.label, dayLabel(c.rmst_180_days), `Average time to exit · capped at 180 days · ${c.n} model IDs`),
        )
        if (comparison.difference_days != null) {
            const difference = comparison.difference_days
            cards.push(
                metric(
                    '2026 minus 2025',
                    `${difference > 0 ? '+' : ''}${fixed(difference)} days`,
                    comparison.ci
                        ? `95% interval: ${fixed(comparison.ci[0])} to ${fixed(comparison.ci[1])} days${comparison.ci[0] <= 0 && comparison.ci[1] >= 0 ? ' · includes shorter and longer spans' : ''}`
                        : 'Matched-horizon cohort comparison',
                ),
            )
        }
        $('cohort-metrics').innerHTML = cards.join('')
        $('cohort-table').innerHTML = cohorts
            .map(
                (c) =>
                    `<tr><td>${esc(c.label)}<small>${esc(c.entry_start)} to ${esc(c.entry_end)}</small></td><td>${c.n}</td><td>${c.events_by_180 ?? c.events}</td><td>${c.n - (c.events_by_180 ?? c.events)}</td><td>${fixed(c.rmst_180_days)} days</td><td>${(c.rmst_180_ci || []).map((x) => fixed(x)).join('–')} days</td></tr>`,
            )
            .join('')
        survivalChart()
    }

    function forecastText(model) {
        const f = model.forecast || {}
        if (f.median_remaining_days == null)
            return `Not estimable<small>${f.at_risk || 0} historical top-ten analogues</small>`
        const range =
            f.p25_remaining_days != null && f.p75_remaining_days != null
                ? `Middle 50%: ${Math.round(f.p25_remaining_days)}–${Math.round(f.p75_remaining_days)} days`
                : 'Middle 50% interval not fully estimable'
        return `${dayLabel(f.median_remaining_days)} remaining<small>${range}</small><small>${f.at_risk || 0} historical top-ten analogues</small>`
    }

    function topSection() {
        $('top-date').textContent = dateShort(data.coverage.latest_complete_date)
        $('top-models').innerHTML = data.current_top10
            .map(
                (m) =>
                    `<tr><td><strong>${m.open_rank}</strong> / ${m.global_rank}</td><td class="model-id"><button type="button" class="model-button" data-model="${esc(m.id)}">${esc(m.id)}</button>${m.name && m.name !== m.id ? `<small>${esc(m.name)}</small>` : ''}</td><td>${compact(m.latest_tokens)}<small>input + output</small></td><td>${dayLabel(m.forecast?.age_days)}${m.reentries ? `<small>${m.reentries} re-entry event(s)</small>` : ''}</td><td>${forecastText(m)}</td><td>${compact(m.observed_tokens)}<small>observed minimum</small></td></tr>`,
            )
            .join('')
        const c = data.coverage.registry_counts || {}
        const unresolved = data.coverage.unclassified_latest || []
        $('classification-note').textContent =
            `Registry: ${c.open ?? '?'} open, ${c.closed ?? '?'} closed, ${c.unknown ?? '?'} unresolved historical IDs. ${unresolved.length ? `${unresolved.length} latest-day IDs have unresolved openness, so this is a reviewed-subset ranking.` : 'All latest-day ranked IDs have reviewed openness.'} Current openness is applied retrospectively; an earlier period of closed access is not automatically excluded. Forecasts use earlier top-ten open-weight observations within ±14 days of the target’s days since entry or return, deduplicated by checkpoint stem with the target excluded. They are not individually calibrated.`
    }

    function historyChart() {
        const points = selected.history || []
        const w = 1000,
            h = 300,
            left = 68,
            right = 22,
            top = 25,
            bottom = 48
        if (!points.length) {
            $('history-chart').textContent = 'No individually reported traffic for this model.'
            return
        }
        const start = new Date(points[0].date + 'T00:00:00Z').getTime()
        const end = new Date(data.end_date + 'T00:00:00Z').getTime()
        const rawValue = (p) => (historyMode === 'tokens' ? Number(p.tokens) : Number(p.share_percent))
        const max = Math.max(...points.map(rawValue), 1e-9) * 1.1
        const xx = (date) =>
            left +
            ((new Date(date + 'T00:00:00Z').getTime() - start) / Math.max(86400000, end - start)) * (w - left - right)
        const yy = (v) => h - bottom - (v / max) * (h - top - bottom)
        let svg = ''
        for (let i = 0; i <= 4; i++) {
            const v = (max * i) / 4
            svg +=
                line(left, yy(v), w - right, yy(v)) +
                label(
                    left - 10,
                    yy(v) + 4,
                    historyMode === 'tokens' ? compact(v, 1) : `${v.toFixed(1)}%`,
                    'text-anchor="end"',
                )
        }
        for (let i = 0; i < 5; i++) {
            const t = new Date(start + ((end - start) * i) / 4).toISOString().slice(0, 10)
            svg += label(xx(t), h - 17, t, 'text-anchor="middle"')
        }
        for (const p of points)
            svg += `<circle cx="${xx(p.date)}" cy="${yy(rawValue(p))}" r="${points.length > 300 ? 2 : 3}" fill="${colors[0]}" opacity=".8"><title>${esc(p.date)} · ${esc(compact(p.tokens))} tokens · ${Number(p.share_percent).toFixed(3)}% share · rank ${p.rank}</title></circle>`
        $('history-chart').innerHTML = frame(
            `${selected.id}: ${historyMode === 'tokens' ? 'observed daily input and output tokens' : 'share of public daily text tokens'}`,
            w,
            h,
            svg,
        )
        $('history-title').textContent =
            historyMode === 'tokens' ? 'Observed daily input + output tokens' : 'Observed share of platform text tokens'
    }

    function recentRate(model) {
        if (model.recent_7d?.mean_daily_tokens_lower_bound != null)
            return Number(model.recent_7d.mean_daily_tokens_lower_bound)
        const start = new Date(data.end_date + 'T00:00:00Z').getTime() - 6 * 86400000
        return (
            (model.history || [])
                .filter((r) => new Date(r.date + 'T00:00:00Z').getTime() >= start)
                .reduce((sum, r) => sum + Number(r.tokens), 0) / 7
        )
    }

    function selectModel(id, move = false) {
        selected = data.current_top10.find((m) => m.id === id)
        if (!selected) return
        $('model-select').value = id
        const f = selected.forecast || {}
        const prefillParameters = selected.active_input_parameters_b ?? selected.active_parameters_b
        const decodeParameters = selected.active_output_parameters_b ?? selected.active_parameters_b
        $('active-parameters').value = prefillParameters ?? 30
        $('output-active-parameters').value = decodeParameters ?? 30
        const parameterSource = selected.active_parameters_source
        const sourceLink =
            parameterSource && /^https:\/\//.test(parameterSource)
                ? `<a href="${esc(parameterSource)}">Primary model card ↗</a>`
                : ''
        $('parameter-note').innerHTML =
            prefillParameters != null && decodeParameters != null
                ? `Parameter inputs start from reviewed model metadata: ${prefillParameters}B prefill / ${decodeParameters}B decode. ${sourceLink} ${esc(selected.active_parameters_note || '')} They remain editable assumptions for the arithmetic proxy.`
                : `No exact active-parameter count is verified for this checkpoint. Both controls start at an explicit <strong>30B assumption</strong>, not a fact about this model. ${esc(selected.active_parameters_note || '')} ${sourceLink}`
        $('empirical-horizon').disabled = f.median_remaining_days == null || f.median_remaining_days > 730
        $('horizon-note').textContent =
            f.median_remaining_days == null
                ? 'Historical remaining-life median is not estimable for this model. The selected horizon is a scenario.'
                : `Historical age-conditioned median: ${Math.round(f.median_remaining_days)} days. The editable horizon remains a scenario until you select that baseline.`
        const upper =
            selected.observed_window_upper_tokens == null
                ? 'Missing dates prevent a complete-window upper bound.'
                : `For ${selected.bounds_start || selected.first_seen}–${selected.bounds_end || data.end_date}, a separate cutoff-based upper bound is ${compact(selected.observed_window_upper_tokens)} tokens; it is not the estimate used below.`
        $('model-observation').innerHTML =
            `<strong>${esc(compact(selected.observed_tokens))} tokens individually observed</strong> from ${esc(dateShort(selected.first_seen))} to ${esc(dateShort(selected.last_seen))}, on ${selected.days_in_top50} ranked days. ${esc(upper)} ${selected.bounds_missing_days ? `${selected.bounds_missing_days} missing source dates have unbounded individual demand.` : ''} Pre-archive traffic is outside both quantities. Current seven-day observed rate: <strong>${esc(compact(recentRate(selected)))} tokens/day</strong>.`
        historyChart()
        scenario()
        if (move)
            $('demand-title').scrollIntoView({
                behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
                block: 'start',
            })
    }

    function scenarioInputs() {
        const numeric = (id) => Number($(id).value)
        return {
            observedTokens: Number(selected.observed_tokens),
            dailyTokens: recentRate(selected),
            horizonDays: numeric('horizon-days'),
            weeklyGrowthPercent: numeric('weekly-growth'),
            tailDays: numeric('tail-days'),
            tailPercent: numeric('tail-percent'),
            inputTokens: numeric('input-tokens'),
            outputTokens: numeric('output-tokens'),
            activeParametersB: numeric('active-parameters'),
            outputActiveParametersB: numeric('output-active-parameters'),
        }
    }

    function scenario() {
        if (!selected) return
        try {
            for (const id of [
                'horizon-days',
                'weekly-growth',
                'tail-days',
                'tail-percent',
                'input-tokens',
                'output-tokens',
                'active-parameters',
                'output-active-parameters',
            ]) {
                if ($(id).value.trim() === '' || !$(id).checkValidity())
                    throw new Error('Enter valid numbers within the displayed input limits.')
            }
            const inputs = scenarioInputs()
            const x = ModelLifetimeMath.demand(inputs)
            lastScenario = { inputs, results: x }
            $('scenario-error').hidden = true
            $('export-scenario').disabled = false
            $('demand-metrics').innerHTML = [
                metric(
                    'Observed token minimum',
                    compact(inputs.observedTokens),
                    `${compact(x.requestsFromObservedTokens)} request equivalents at assumed length`,
                ),
                metric(
                    'Additional future scenario',
                    compact(x.futureTokens),
                    `${Math.round(inputs.horizonDays)} prominence days + ${inputs.tailDays} assumed tail days`,
                ),
                metric(
                    'Observed minimum + future scenario',
                    compact(x.requests),
                    `request equivalents · ${compact(x.modeledTokens)} tokens · incomplete past`,
                ),
                metric(
                    'Uncached compute proxy',
                    `${compact(x.inferenceFlops / 1e21)} ZFLOPs`,
                    `Input ${inputs.activeParametersB}B / output ${inputs.outputActiveParametersB}B · split from assumed request lengths`,
                ),
            ].join('')
            const segments = [
                ['Observed in rankings', inputs.observedTokens, colors[0]],
                ['Future prominence scenario', x.prominenceTokens, colors[2]],
                ['Additional post-exit tail', x.tailTokens, colors[1]],
            ]
            const max = Math.max(...segments.map((r) => r[1]), 1)
            $('demand-bars').innerHTML = segments
                .map(
                    ([name, value, color]) =>
                        `<div class="bar-row"><span>${esc(name)}</span><span class="track"><span class="fill" style="display:block;width:${(100 * value) / max}%;background:${color}"></span></span><span class="amount">${compact(value)}</span></div>`,
                )
                .join('')
            $('all-demand-models').innerHTML = data.current_top10
                .map((m) => {
                    const projected = ModelLifetimeMath.demand({
                        ...inputs,
                        observedTokens: Number(m.observed_tokens),
                        dailyTokens: recentRate(m),
                    })
                    return `<tr><td class="model-id">${esc(m.id)}</td><td>${compact(m.observed_tokens)}</td><td>${compact(projected.modeledTokens)}</td><td>${compact(projected.requests)}</td></tr>`
                })
                .join('')
            $('request-description').textContent =
                `Assumed average request: ${inputs.inputTokens.toLocaleString()} input + ${inputs.outputTokens.toLocaleString()} output = ${x.tokensPerRequest.toLocaleString()} tokens. ${inputs.inputTokens === 70 && inputs.outputTokens === 215 ? 'This is the paper’s illustrative request workload, not an estimate of current OpenRouter request length.' : 'These are inferred request equivalents, not observed request counts.'} The combined amount excludes unknown historical traffic and is not a measured or complete lifetime total.`
        } catch (error) {
            lastScenario = null
            $('scenario-error').hidden = false
            $('scenario-error').textContent = error.message
            $('export-scenario').disabled = true
            $('demand-metrics').innerHTML = ''
            $('demand-bars').innerHTML = ''
            $('request-description').textContent = ''
            $('all-demand-models').innerHTML = ''
        }
    }

    function exitsSection() {
        const query = $('exit-search').value.trim().toLowerCase()
        const found = data.exited_models.filter((m) => `${m.id} ${m.name}`.toLowerCase().includes(query))
        $('exited-models').innerHTML =
            found
                .slice(0, 80)
                .map(
                    (m) =>
                        `<tr><td class="model-id">${esc(m.id)}</td><td>${esc(m.last_seen)}</td><td>${m.best_rank}</td><td>${m.days_in_top50}</td><td>${m.observed_span_days} days${m.left_boundary ? '<small>Earlier history unknown</small>' : ''}</td><td>${compact(m.observed_tokens)}</td></tr>`,
                )
                .join('') || '<tr><td colspan="6">No former model matches this search.</td></tr>'
        $('exits-count').textContent =
            `Showing ${Math.min(80, found.length)} of ${found.length} matching former models, ordered by cumulative observed token volume. Search to narrow the list.`
    }

    function platformChart() {
        const measure = $('platform-measure').value
        const months = data.monthly.filter((m) => m.complete && m[measure] != null)
        const w = 1000,
            h = 300,
            left = 68,
            right = 22,
            top = 22,
            bottom = 60
        if (!months.length) {
            $('platform-chart').textContent = 'No complete calendar months for this measure.'
            return
        }
        const isPercent = measure === 'open_observed_share_percent'
        const max = Math.max(...months.map((m) => Number(m[measure])), 1) * 1.12
        const gap = (w - left - right) / months.length
        const yy = (v) => h - bottom - (v / max) * (h - top - bottom)
        let svg = ''
        for (let i = 0; i <= 4; i++) {
            const v = (max * i) / 4
            svg +=
                line(left, yy(v), w - right, yy(v)) +
                label(left - 10, yy(v) + 4, isPercent ? `${v.toFixed(0)}%` : compact(v, 1), 'text-anchor="end"')
        }
        months.forEach((m, i) => {
            const value = Number(m[measure]),
                xx = left + gap * i + gap * 0.16
            svg += `<rect x="${xx}" y="${yy(value)}" width="${gap * 0.68}" height="${h - bottom - yy(value)}" rx="2" fill="${measure.includes('open') ? colors[1] : colors[0]}" opacity=".85"><title>${esc(m.month)}: ${isPercent ? `${value.toFixed(2)}%` : `${esc(compact(value))} tokens`}</title></rect>`
            if (i % 2 === 0 || i === months.length - 1)
                svg += label(
                    xx + gap * 0.34,
                    h - bottom + 20,
                    m.month,
                    `transform="rotate(-35 ${xx + gap * 0.34} ${h - bottom + 20})" text-anchor="end"`,
                )
        })
        $('platform-chart').innerHTML = frame($('platform-measure').selectedOptions[0].textContent, w, h, svg)
        const first = months[0],
            last = months.at(-1)
        const ratio = Number(first[measure]) > 0 ? Number(last[measure]) / Number(first[measure]) : null
        $('platform-note').textContent =
            `${months.length} complete calendar months. ${first.month} → ${last.month}: ${isPercent ? `${Number(first[measure]).toFixed(1)}% → ${Number(last[measure]).toFixed(1)}%` : `${compact(first[measure])} → ${compact(last[measure])}${ratio != null ? ` (${ratio.toFixed(1)}×)` : ''}`}. ${measure === 'open_observed_tokens' ? 'This is a ratio of observed monthly minima, not a lower bound on the growth of true open-weight demand.' : measure === 'open_observed_share_percent' ? 'The numerator excludes open-weight traffic hidden below the top 50.' : ''}`
    }

    function bind() {
        $('model-select').addEventListener('change', (e) => selectModel(e.target.value))
        document.addEventListener('click', (e) => {
            const model = e.target.closest('[data-model]')
            if (model) selectModel(model.dataset.model, true)
            const history = e.target.closest('[data-history]')
            if (history) {
                historyMode = history.dataset.history
                document.querySelectorAll('[data-history]').forEach((b) => {
                    const on = b === history
                    b.classList.toggle('active', on)
                    b.setAttribute('aria-pressed', on)
                })
                historyChart()
            }
            const horizon = e.target.closest('[data-horizon]')
            if (horizon) {
                $('horizon-days').value = horizon.dataset.horizon
                scenario()
            }
            const preset = e.target.closest('[data-preset]')
            if (preset) {
                const lengths = { paper: [70, 215], medium: [4000, 1000], agent: [64000, 4000] }[preset.dataset.preset]
                $('input-tokens').value = lengths[0]
                $('output-tokens').value = lengths[1]
                scenario()
            }
        })
        $('empirical-horizon').addEventListener('click', () => {
            $('horizon-days').value = Math.round(selected.forecast.median_remaining_days)
            scenario()
        })
        document.querySelectorAll('.controls input').forEach((input) => input.addEventListener('input', scenario))
        $('exit-search').addEventListener('input', exitsSection)
        $('platform-measure').addEventListener('change', platformChart)
        $('export-scenario').addEventListener('click', () => {
            if (!lastScenario) return
            const payload = {
                schema_version: 1,
                kind: 'observed_token_minimum_plus_explicit_future_scenario',
                model_id: selected.id,
                snapshot_end_date: data.end_date,
                source_provenance: data.provenance,
                generated_at: new Date().toISOString(),
                interpretation: {
                    requests: 'Inferred equivalents from assumed mean input+output length; not measured counts.',
                    tokens: 'Observed archive minimum plus scenario; excludes unknown historical tail, missing dates and pre-archive traffic.',
                    compute:
                        'Rough uncached 2*(Ninput*Dinput+Noutput*Doutput) proxy; input/output token split comes from assumed request lengths. Parameter values are editable.',
                    parameter_metadata: {
                        input: selected.active_input_parameters_b ?? selected.active_parameters_b ?? null,
                        output: selected.active_output_parameters_b ?? selected.active_parameters_b ?? null,
                        source: selected.active_parameters_source || null,
                    },
                    future: 'Constant weekly growth until scenario prominence exit, followed by an assumed flat-rate tail. No actual retirement is inferred.',
                },
                observed_tokens_exact: selected.observed_tokens,
                historical_forecast: selected.forecast || null,
                ...lastScenario,
            }
            const url = URL.createObjectURL(
                new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' }),
            )
            const a = document.createElement('a')
            a.href = url
            a.download = `model-lifetime-${selected.id.replace(/[^a-z0-9._-]+/gi, '-')}-${data.end_date}.json`
            a.click()
            setTimeout(() => URL.revokeObjectURL(url), 1000)
        })
    }

    async function start() {
        try {
            const response = await fetch('data/openrouter/model-lifetime.json')
            if (!response.ok) throw new Error(`Snapshot could not be loaded (HTTP ${response.status}).`)
            data = await response.json()
            if (!Array.isArray(data.current_top10) || !Array.isArray(data.models) || !Array.isArray(data.cohorts))
                throw new Error('The snapshot has an unsupported schema.')
            $('snapshot-date').textContent =
                `Observed ${data.start_date}–${data.end_date} · ${data.coverage.observed_days} daily rankings · UTC dates`
            cohortSection()
            topSection()
            exitsSection()
            platformChart()
            $('model-select').innerHTML = data.current_top10
                .map((m) => `<option value="${esc(m.id)}">${m.open_rank}. ${esc(m.id)}</option>`)
                .join('')
            $('model-select').disabled = false
            const missing = data.coverage.missing_dates || []
            $('coverage-note').textContent =
                `Expected ${data.coverage.expected_days} calendar dates; ${data.coverage.observed_days} have rankings. Missing source dates: ${missing.length ? missing.join(', ') : 'none'}. Latest complete daily ranking: ${data.coverage.latest_complete_date}. Exact token counts are retained as strings in the JSON; chart and scenario display rounds values.`
            const asOf = data.provenance?.as_of || 'timestamps in the analysis JSON'
            $('provenance-note').textContent =
                `Source: OpenRouter (openrouter.ai/rankings), as of ${asOf}. Each retained source snapshot carries its retrieval metadata. Generated observations and estimates use the stated cutoff, not the date you open this page.`
            $('attribution').innerHTML =
                `Source: <a href="https://openrouter.ai/rankings">OpenRouter (openrouter.ai/rankings)</a>, as of ${esc(asOf)}. Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Classification, survival analysis, and scenario projections are independent modifications by this project.`
            bind()
            if (data.current_top10[0]) selectModel(data.current_top10[0].id)
        } catch (error) {
            $('load-error').hidden = false
            $('load-error').textContent =
                `${error.message} Open this page through the site or a local HTTP server. The method and source links remain available below.`
            $('snapshot-date').textContent = 'Snapshot unavailable in this view'
        }
    }
    start()
})()
