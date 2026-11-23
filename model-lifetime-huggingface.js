// SPDX-License-Identifier: MIT
// A separate adoption signal; this module never changes the demand or lifetime inputs.
;(() => {
    'use strict'
    const $ = (id) => document.getElementById(id)
    const esc = (value) =>
        String(value ?? '').replace(
            /[&<>"']/g,
            (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
        )
    const number = (value) => (value != null && Number.isFinite(Number(value)) ? Number(value) : null)
    const compact = (value) => {
        const n = number(value)
        return n == null
            ? 'Unavailable'
            : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(n)
    }
    const fixed = (value) => (number(value) == null ? 'Not estimable' : Number(value).toFixed(2))
    const hfUrl = (id) => 'https://huggingface.co/' + id.split('/').map(encodeURIComponent).join('/')
    const interval = (value) =>
        Array.isArray(value) && value.length === 2
            ? `95% interval ${fixed(value[0])} to ${fixed(value[1])}`
            : 'Interval unavailable'
    const metric = (label, value, detail) =>
        `<div class="metric"><span class="label">${esc(label)}</span><span class="value">${esc(value)}</span><span class="detail">${esc(detail)}</span></div>`
    const sizeLabel = (m) =>
        m.parameters_total == null ? 'Unavailable' : `${compact(Number(m.parameters_total) / 1e9)}B`
    const availability = (status) => (status === 'conflict' ? 'Repository ID changed' : 'Metadata unavailable')
    let data

    function traffic(m) {
        const lower = number(m.openrouter_tokens_lower)
        const upper = number(m.openrouter_tokens_upper)
        if (!m.recently_observed)
            return `Not individually reported${upper == null ? '' : `<small>0–${esc(compact(upper))} possible; not a zero-use observation</small>`}`
        if (m.fully_observed) return `${esc(compact(lower))}<small>Known aliases fully reported</small>`
        return `At least ${esc(compact(lower))}<small>${upper == null ? 'Upper bound unavailable' : `Up to ${esc(compact(upper))} within the ranking bounds`}</small>`
    }

    function currentRows() {
        const complete = $('hf-sample').value === 'complete'
        const size = $('hf-size').value
        return data.models.filter(
            (m) =>
                m.hf_status === 'ok' &&
                m.recently_observed &&
                (!complete || m.fully_observed) &&
                (size === 'all' || m.size_group === size),
        )
    }

    function selectedStats() {
        const size = $('hf-size').value
        const group = size === 'all' ? data : data.size_groups.find((g) => g.id === size)
        return group?.correlations?.[$('hf-sample').value]
    }

    function statistics() {
        const stats = selectedStats()
        const complete = $('hf-sample').value === 'complete'
        const n = stats?.n ?? 0
        const ready = stats?.status === 'ok'
        const kind = complete ? 'fully reported known aliases' : 'observed token minimums'
        $('hf-metrics').innerHTML = [
            metric(
                'Rank correlation · Spearman',
                ready ? fixed(stats.spearman?.value) : 'Not estimable',
                ready
                    ? interval(stats.spearman?.ci)
                    : 'At least five repositories and variation in both measures are required.',
            ),
            metric(
                'Correlation of log counts · Pearson',
                ready ? fixed(stats.pearson_log1p?.value) : 'Not estimable',
                ready ? interval(stats.pearson_log1p?.ci) : 'The same selected sample is used for both measures.',
            ),
            metric('Matched repositories', n, kind),
        ].join('')
        if (!ready) {
            $('hf-result').textContent =
                `This selection has ${n} matched repositories and does not support a correlation estimate. Try a broader sample or size range.`
            return
        }
        const ci = stats.spearman?.ci
        const crosses = ci && ci[0] <= 0 && ci[1] >= 0
        $('hf-result').textContent =
            `For these ${n} repositories, the rank correlation is ${fixed(stats.spearman.value)}. ${crosses ? 'Its interval includes no association, so the direction is uncertain.' : 'This describes the association within the selected sample.'} ${complete ? 'The sample favors models whose known aliases appeared every day.' : 'Some token totals are incomplete: this is a correlation with observed minimums, not full traffic.'}`
    }

    function chart() {
        const rows = currentRows()
        if (!rows.length) {
            $('hf-scatter').innerHTML = '<p class="small">No matched repositories in this selection.</p>'
            return
        }
        const w = 1000,
            h = 440,
            left = 90,
            right = 25,
            top = 25,
            bottom = 65
        const lx = (m) => Math.log10(1 + Number(m.hf_downloads_30d))
        const ly = (v) => Math.log10(1 + Number(v))
        const xlo = Math.max(0, Math.floor(Math.min(...rows.map(lx))))
        const xhi = Math.max(xlo + 1, Math.ceil(Math.max(...rows.map(lx))))
        const ylo = Math.max(0, Math.floor(Math.min(...rows.map((m) => ly(m.openrouter_tokens_lower)))))
        const yhi = Math.max(
            ylo + 1,
            Math.ceil(Math.max(...rows.map((m) => ly(m.openrouter_tokens_upper ?? m.openrouter_tokens_lower)))),
        )
        const xx = (v) => left + ((v - xlo) / (xhi - xlo)) * (w - left - right)
        const yy = (v) => h - bottom - ((v - ylo) / (yhi - ylo)) * (h - top - bottom)
        let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Hugging Face downloads against reported OpenRouter tokens, logarithmic axes"><title>Downloads versus hosted token traffic for ${rows.length} matched repositories</title>`
        for (let p = xlo; p <= xhi; p++) {
            const x = xx(p)
            svg += `<line class="grid" x1="${x}" x2="${x}" y1="${top}" y2="${h - bottom}"/><text x="${x}" y="${h - bottom + 24}" text-anchor="middle">${esc(compact(Math.max(0, 10 ** p - 1)))}</text>`
        }
        for (let p = ylo; p <= yhi; p++) {
            const y = yy(p)
            svg += `<line class="grid" x1="${left}" x2="${w - right}" y1="${y}" y2="${y}"/><text x="${left - 12}" y="${y + 4}" text-anchor="end">${esc(compact(Math.max(0, 10 ** p - 1)))}</text>`
        }
        svg += `<text x="${(w + left - right) / 2}" y="${h - 10}" text-anchor="middle">Hugging Face downloads · trailing 30 days · log(1 + count)</text><text transform="translate(18 ${(h - bottom + top) / 2}) rotate(-90)" text-anchor="middle">OpenRouter tokens · 30 days · log(1 + count)</text>`
        for (const m of rows) {
            const x = xx(lx(m)),
                y = yy(ly(m.openrouter_tokens_lower))
            const color = m.fully_observed ? '#69d9e8' : '#ffc17b'
            const description = `${m.repo_id}: ${compact(m.hf_downloads_30d)} downloads; ${m.fully_observed ? '' : 'at least '}${compact(m.openrouter_tokens_lower)} OpenRouter tokens; visible on ${m.observed_days} of ${data.window.days} days.`
            svg += `<g tabindex="0" role="button" class="hf-point" data-hf-repo="${esc(m.repo_id)}" aria-label="${esc(description)}"><title>${esc(description)}</title>`
            if (!m.fully_observed && m.openrouter_tokens_upper != null) {
                const upperY = yy(ly(m.openrouter_tokens_upper))
                svg += `<line x1="${x}" x2="${x}" y1="${y}" y2="${upperY}" stroke="${color}" stroke-opacity="0.6"/><line x1="${x - 4}" x2="${x + 4}" y1="${upperY}" y2="${upperY}" stroke="${color}"/>`
            }
            svg += `<circle cx="${x}" cy="${y}" r="12" fill="transparent"/><circle class="hf-marker" cx="${x}" cy="${y}" r="5.5" fill="${m.fully_observed ? color : '#0b1528'}" stroke="${color}" stroke-width="2"/></g>`
        }
        $('hf-scatter').innerHTML = svg + '</svg>'
        $('hf-point-detail').textContent = 'Select a point to see its repository, aliases, and traffic coverage.'
    }

    function pointDetail(id) {
        const m = data.models.find((row) => row.repo_id === id)
        if (!m) return
        $('hf-point-detail').innerHTML =
            `<a href="${esc(hfUrl(m.repo_id))}">${esc(m.repo_id)}</a> · <strong>${esc(compact(m.hf_downloads_30d))} downloads</strong> · ${traffic(m)}<br>${m.observed_days}/${data.window.days} days with at least one visible alias; ${m.complete_alias_days}/${data.window.days} days with every known alias visible. OpenRouter IDs: ${m.model_ids.map(esc).join(', ')}.`
    }

    function table() {
        const query = $('hf-search').value.trim().toLowerCase()
        const rows = data.models
            .filter((m) => [m.repo_id, ...m.model_ids, ...(m.names || [])].join(' ').toLowerCase().includes(query))
            .sort(
                (a, b) =>
                    (number(b.hf_downloads_30d) ?? -1) - (number(a.hf_downloads_30d) ?? -1) ||
                    a.repo_id.localeCompare(b.repo_id),
            )
        $('hf-models').innerHTML =
            rows
                .map(
                    (m) =>
                        `<tr><td class="model-id"><a href="${esc(hfUrl(m.repo_id))}">${esc(m.repo_id)}</a><small>${esc(m.model_ids.length)} OpenRouter ID${m.model_ids.length === 1 ? '' : 's'}${m.model_ids.length > 1 ? ' combined' : ''}</small><details class="hf-aliases"><summary>OpenRouter identity</summary>${m.model_ids.map((id) => `<small>${esc(id)}</small>`).join('')}</details></td><td>${m.hf_status === 'ok' ? esc(compact(m.hf_downloads_30d)) : 'Unavailable'}${m.hf_status !== 'ok' ? `<small>${esc(availability(m.hf_status))}</small>` : ''}</td><td>${traffic(m)}</td><td>${m.observed_days}/${data.window.days}<small>${m.complete_alias_days} days with every known alias visible</small></td><td>${esc(sizeLabel(m))}</td></tr>`,
                )
                .join('') || '<tr><td colspan="5">No repositories match this search.</td></tr>'
        $('hf-table-count').textContent =
            `${rows.length} of ${data.models.length} mapped repositories shown, ordered by downloads. Scroll within the table for more rows. It includes models outside the chart sample.`
    }

    function selectedModel() {
        const id = $('model-select')?.value
        const note = $('hf-selected-model')
        const m = data.models.find((row) => row.model_ids.includes(id))
        note.hidden = !id || id === 'Loading models…'
        if (note.hidden) return
        if (!m || m.hf_status !== 'ok') {
            note.innerHTML =
                '<strong>Hugging Face adoption:</strong> no verified download count is available for this exact model identity. <a href="#huggingface">See comparison coverage.</a>'
            return
        }
        note.innerHTML = `<strong>${esc(compact(m.hf_downloads_30d))} Hugging Face downloads in the trailing 30 days</strong> for <a href="${esc(hfUrl(m.repo_id))}">${esc(m.repo_id)}</a>, retrieved ${esc(data.alignment.hf_snapshot_date)}. ${m.model_ids.length > 1 ? `This repository is shared by ${m.model_ids.length} mapped OpenRouter IDs. ` : ''}This is distribution interest, not measured inference requests; it does not change the forecast or scenario inputs. <a href="#huggingface">Compare downloads and hosted traffic.</a>`
    }

    function persistenceExample() {
        const qwen = data.models.find((m) => m.repo_id === 'Qwen/Qwen2.5-7B-Instruct')
        if (!qwen || qwen.hf_status !== 'ok' || qwen.recently_observed || !(qwen.hf_downloads_30d > 0)) return
        $('hf-persistence').hidden = false
        $('hf-persistence').innerHTML =
            `<strong>Download interest can outlast ranking visibility</strong><p><a href="${esc(hfUrl(qwen.repo_id))}">Qwen2.5-7B-Instruct</a> recorded ${esc(compact(qwen.hf_downloads_30d))} downloads in Hugging Face’s trailing 30-day count, while its mapped OpenRouter IDs were absent from all ${data.window.days} daily top-50 lists in this window. That is ongoing file activity; the amount of inference behind it remains unknown.</p>`
    }

    async function start() {
        try {
            const response = await fetch('data/openrouter/huggingface-comparison.json')
            if (!response.ok) throw new Error(`HTTP ${response.status}`)
            data = await response.json()
            if (!Array.isArray(data.models) || !data.window || !data.correlations)
                throw new Error('Unsupported comparison format')
            $('hf-dates').textContent =
                `HF retrieved ${data.alignment.hf_snapshot_date} · OpenRouter ${data.window.start}–${data.window.end} · UTC dates`
            $('hf-alignment-note').textContent =
                `${data.alignment.note || ''} The lifetime tables elsewhere on this page retain their own dated OpenRouter snapshot.`
            const recent = data.models.filter((m) => m.recently_observed && m.hf_status === 'ok').length
            const complete = data.models.filter(
                (m) => m.recently_observed && m.hf_status === 'ok' && m.fully_observed,
            ).length
            const missing = data.models.filter((m) => m.hf_status !== 'ok').length
            $('hf-coverage').textContent =
                `${data.models.length} mapped Hugging Face repositories; ${recent} have downloads and individually reported traffic in this window. ${complete} have every known alias visible on all ${data.window.days} days. ${missing} repositories have unavailable or unresolved download data. This join covers reviewed model identities, not all possible serving aliases or weight conversions.`
            $('hf-table-caption').textContent =
                `HF downloads retrieved ${data.alignment.hf_snapshot_date}; OpenRouter traffic ${data.window.start}–${data.window.end}. A ranking absence does not establish zero use.`
            $('hf-sample').addEventListener('change', () => {
                statistics()
                chart()
            })
            $('hf-size').addEventListener('change', () => {
                statistics()
                chart()
            })
            $('hf-search').addEventListener('input', table)
            $('hf-scatter').addEventListener('click', (event) => {
                const point = event.target.closest('[data-hf-repo]')
                if (point) pointDetail(point.dataset.hfRepo)
            })
            $('hf-scatter').addEventListener('keydown', (event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return
                const point = event.target.closest('[data-hf-repo]')
                if (point) {
                    event.preventDefault()
                    pointDetail(point.dataset.hfRepo)
                }
            })
            $('model-select').addEventListener('change', selectedModel)
            new MutationObserver(selectedModel).observe($('model-observation'), { childList: true, subtree: true })
            statistics()
            chart()
            table()
            selectedModel()
            persistenceExample()
        } catch (error) {
            $('hf-error').hidden = false
            $('hf-error').textContent =
                `The saved Hugging Face comparison could not be loaded (${error.message}). The existing lifetime projections remain available; download the comparison JSON or use a local HTTP server to inspect this addition.`
            $('hf-dates').textContent = 'Hugging Face comparison unavailable'
        }
    }
    start()
})()
