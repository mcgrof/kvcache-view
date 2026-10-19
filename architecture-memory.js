// SPDX-License-Identifier: MIT
// Hypothetical active-state accounting; no trained-model performance is inferred.
const MEMORY_DESIGNS = [
    { key: 'full', label: 'Full attention', color: '#81b8ff', growth: 'All 32 layers’ KV' },
    { key: 'swa', label: 'All-layer SWA', color: '#f9a08a', growth: 'Bounded by the retained window' },
    { key: 'recurrent', label: 'All-layer recurrence', color: '#8be6c8', growth: 'Fixed recurrent state' },
    {
        key: 'localHybrid',
        label: 'SWA + full attention',
        color: '#e9c687',
        growth: 'Full layers’ KV + bounded SWA KV',
    },
    {
        key: 'recurrentHybrid',
        label: 'Recurrent / full hybrid',
        color: '#b8a5ed',
        growth: 'Full layers’ KV + fixed recurrent state',
    },
]

function activeStateBytes(tokens, { window = 512, fullLayers = 8, stateMiB = 1, sequences = 1 } = {}) {
    const values = [tokens, window, fullLayers, stateMiB, sequences]
    if (
        !values.every(Number.isFinite) ||
        tokens <= 0 ||
        window <= 0 ||
        fullLayers < 0 ||
        fullLayers > 32 ||
        stateMiB <= 0 ||
        sequences <= 0
    ) {
        throw new RangeError('Invalid active-state dimensions')
    }
    const kv = 2 * 8 * 128 * 2
    const state = stateMiB * 1024 ** 2
    const local = kv * Math.min(tokens, window)
    return {
        full: sequences * 32 * kv * tokens,
        swa: sequences * 32 * local,
        recurrent: sequences * 32 * state,
        localHybrid: sequences * (fullLayers * kv * tokens + (32 - fullLayers) * local),
        recurrentHybrid: sequences * (fullLayers * kv * tokens + (32 - fullLayers) * state),
    }
}

function formatStateBytes(bytes) {
    const divisor = bytes >= 1024 ** 3 ? 1024 ** 3 : 1024 ** 2
    return (
        (bytes / divisor).toLocaleString('en-US', { maximumFractionDigits: 3 }) +
        (divisor === 1024 ** 3 ? ' GiB' : ' MiB')
    )
}

if (typeof module !== 'undefined') module.exports = { activeStateBytes, formatStateBytes }

if (typeof document !== 'undefined' && document.querySelector('#memory-chart')) {
    const chart = document.querySelector('#memory-chart')
    const table = document.querySelector('#memory-values')
    const controls = document.querySelector('.memory-controls')
    const read = (id) => Number(document.getElementById(id).value)
    const draw = () => {
        const tokens = read('memory-context')
        const options = {
            window: read('memory-window'),
            fullLayers: read('memory-full'),
            stateMiB: read('memory-state'),
            sequences: read('memory-batch'),
        }
        const data = activeStateBytes(tokens, options)
        table.innerHTML = MEMORY_DESIGNS.map(
            (d) =>
                `<tr><th scope="row"><span class="chart-key" style="--series:${d.color}"></span>${d.label}</th><td>${formatStateBytes(data[d.key])}</td><td>${d.growth}</td></tr>`,
        ).join('')
        table.closest('table').querySelector('thead th:nth-child(2)').textContent =
            `State at ${tokens.toLocaleString('en-US')} tokens · ${options.sequences} sequence${options.sequences > 1 ? 's' : ''}`
        const points = Array.from({ length: 25 }, (_, i) => 2 ** (14 + i / 4))
        const samples = points.map((n) => activeStateBytes(n, options))
        const allValues = samples.flatMap(Object.values)
        const lo = Math.floor(Math.log2(Math.min(...allValues) / 1024 ** 2))
        const hi = Math.ceil(Math.log2(Math.max(...allValues) / 1024 ** 2))
        const span = Math.max(hi - lo, 1)
        const x = (n) => 90 + ((Math.log2(n) - 14) / 6) * 850
        const y = (bytes) => 300 - ((Math.log2(bytes / 1024 ** 2) - lo) / span) * 250
        let svg =
            '<svg viewBox="0 0 1000 365" role="img" aria-labelledby="memory-chart-title"><title id="memory-chart-title">Illustrative active state bytes versus context length; both axes logarithmic. Exact selected values in the adjacent table.</title><text x="90" y="20" class="axis-label">Active state · log scale</text>'
        const step = Math.max(1, Math.ceil(span / 5))
        for (let p = lo; p <= hi; p += step) {
            const value = 2 ** p * 1024 ** 2
            svg += `<line x1="90" x2="940" y1="${y(value)}" y2="${y(value)}" class="gridline"/><text x="76" y="${y(value) + 5}" text-anchor="end" class="axis-label">${formatStateBytes(value)}</text>`
        }
        const tickLabels = ['16K', '32K', '64K', '128K', '256K', '512K', '1M']
        tickLabels.forEach((label, i) => {
            svg += `<text x="${x(2 ** (14 + i))}" y="331" text-anchor="middle" class="axis-label">${label}</text>`
        })
        svg += `<line x1="${x(tokens)}" x2="${x(tokens)}" y1="42" y2="305" stroke="#8092a7" stroke-dasharray="3 5"/>`
        MEMORY_DESIGNS.forEach((d, index) => {
            const coords = points.map((n, i) => `${x(n)},${y(samples[i][d.key])}`).join(' ')
            const dash = index >= 3 ? ` stroke-dasharray="${index === 3 ? '9 5' : '3 5'}"` : ''
            svg += `<polyline points="${coords}" fill="none" stroke="${d.color}" stroke-width="3"${dash}/><circle cx="${x(tokens)}" cy="${y(data[d.key])}" r="5" fill="${d.color}"><title>${d.label}: ${formatStateBytes(data[d.key])}</title></circle>`
        })
        svg +=
            '<text x="515" y="359" text-anchor="middle" class="axis-label">Context tokens · log scale · K = 1,024; M = 1,048,576</text></svg>'
        chart.innerHTML =
            svg +
            '<div class="memory-legend">' +
            MEMORY_DESIGNS.map((d) => `<span><i style="background:${d.color}"></i>${d.label}</span>`).join('') +
            '</div>'
    }
    controls.hidden = false
    controls.addEventListener('change', draw)
    draw()
}
