// Payload/capacity scenarios, not measurements of an NVMe Engram implementation.
// Paper configuration: arXiv:2601.07372v1, Eq. 2 and Table 5.
function projectEngramIO(input) {
    const { paramsB, bytesPerValue, layers, hbmGiB, dramGiB, tokensPerSecond, readBytes, accessModel, residentHit } =
        input
    for (const value of [paramsB, bytesPerValue, layers, hbmGiB, dramGiB, tokensPerSecond, readBytes, residentHit]) {
        if (!Number.isFinite(value) || value < 0) throw new RangeError('Inputs must be finite and nonnegative')
    }
    if (
        paramsB <= 0 ||
        ![1, 2, 4].includes(bytesPerValue) ||
        !Number.isInteger(layers) ||
        layers < 1 ||
        ![512, 4096].includes(readBytes) ||
        residentHit > 100 ||
        !['uniform', 'supplied'].includes(accessModel)
    ) {
        throw new RangeError('Unsupported projection configuration')
    }
    const tableBytes = paramsB * 1e9 * bytesPerValue
    // HBM must cover rows NOT also counted in the host DRAM budget.
    const hbmBytes = Math.min(tableBytes, hbmGiB * 2 ** 30)
    const dramBytes = Math.min(tableBytes - hbmBytes, dramGiB * 2 ** 30)
    const spillBytes = Math.max(0, tableBytes - hbmBytes - dramBytes)
    const rowBytes = 80 * bytesPerValue
    const lookupsPerToken = 16 * layers
    const missFraction =
        spillBytes === 0
            ? 0
            : hbmBytes + dramBytes === 0
              ? 1
              : accessModel === 'uniform'
                ? spillBytes / tableBytes
                : 1 - residentHit / 100
    const lookupsPerSecond = tokensPerSecond * lookupsPerToken
    const ssdIOPS = lookupsPerSecond * missFraction
    return {
        tableBytes,
        hbmBytes,
        dramBytes,
        spillBytes,
        rowBytes,
        lookupsPerToken,
        missFraction,
        lookupsPerSecond,
        ssdIOPS,
        payloadPerToken: rowBytes * lookupsPerToken,
        usefulSSDBytesPerSecond: ssdIOPS * rowBytes,
        deviceBytesPerSecond: ssdIOPS * readBytes,
        readAmplification: readBytes / rowBytes,
        fitThresholdB: ((hbmGiB + dramGiB) * 2 ** 30) / bytesPerValue / 1e9,
    }
}

if (typeof module !== 'undefined') module.exports = { projectEngramIO }

if (typeof document !== 'undefined') {
    const panel = document.getElementById('io-projection')
    const get = (id) => document.getElementById(id)
    const num = (id) => Number(get(id).value)
    const format = (value, digits = 2) => value.toLocaleString('en-US', { maximumFractionDigits: digits })
    const gib = (bytes) => `${format(bytes / 2 ** 30)} GiB`
    const presets = {
        '27b': { paramsB: 5.7, layers: 2 },
        '40b': { paramsB: 18.5, layers: 2 },
        '100b': { paramsB: 100, layers: 1 },
        '1t': { paramsB: 1000, layers: 2 },
    }
    function renderIOProjection() {
        get('io-hit').disabled = get('io-access').value !== 'supplied'
        const inputs = [...panel.querySelectorAll('input:not(:disabled)')]
        if (inputs.some((input) => input.value === '' || !input.checkValidity())) {
            get('io-results').hidden = true
            get('io-error').hidden = false
            return
        }
        const input = {
            paramsB: num('io-params'),
            bytesPerValue: num('io-dtype'),
            layers: num('io-layers'),
            hbmGiB: num('io-hbm'),
            dramGiB: num('io-dram'),
            tokensPerSecond: num('io-rate'),
            readBytes: num('io-read'),
            accessModel: get('io-access').value,
            residentHit: get('io-hit').disabled ? 0 : num('io-hit'),
        }
        const result = projectEngramIO(input)
        get('io-error').hidden = true
        get('io-results').hidden = false
        get('io-table').textContent = `${gib(result.tableBytes)} (${format(result.tableBytes / 1e9)} GB)`
        get('io-spill').textContent = gib(result.spillBytes)
        get('io-iops').textContent = format(result.ssdIOPS, 0)
        get('io-bandwidth').textContent = `${format(result.deviceBytesPerSecond / 1e6)} MB/s`
        get('io-payload').textContent = `${result.rowBytes} B / lookup; ${format(result.payloadPerToken)} B / token`
        get('io-miss').textContent =
            `${format(result.missFraction * 100)}% of ${format(result.lookupsPerSecond, 0)} lookups/s`
        get('io-amplification').textContent = `${format(result.readAmplification)}× bytes per SSD-served row`
        get('io-useful').textContent = `${format(result.usefulSSDBytesPerSecond / 1e6)} MB/s useful SSD payload`
        get('io-threshold').textContent = `${format(result.fitThresholdB)} billion parameters`
        const fits = result.spillBytes === 0
        get('io-verdict').className = `info-box ${fits ? 'green' : 'yellow'}`
        get('io-verdict-title').textContent = fits
            ? 'Fits in assigned memory — no capacity-driven SSD reads'
            : 'Exceeds assigned memory — SSD traffic depends on access locality'
        get('io-verdict-text').textContent = fits
            ? 'After warm-up, every table row can remain in the assigned memory. This scenario forces steady-state SSD IOPS to zero; checkpoint loading and persistence are separate.'
            : `${gib(result.spillBytes)} cannot stay in the assigned memory. ${result.hbmBytes + result.dramBytes === 0 ? 'With no resident capacity, all lookups miss memory regardless of the supplied hit rate.' : input.accessModel === 'uniform' ? 'The uniform-access scenario equates the missing capacity fraction with the lookup miss fraction.' : 'The supplied resident hit rate determines lookup misses; it is not inferred from the capacity deficit.'} These are offered I/O requirements, not achievable throughput.`
        get('io-quantization').hidden = input.bytesPerValue !== 1
        const segments = [
            ['Additional HBM', result.hbmBytes, '#22c55e'],
            ['Host DRAM', result.dramBytes, '#fbbf24'],
            ['Outside memory', result.spillBytes, '#a78bfa'],
        ]
        get('io-capacity-bar').innerHTML = segments
            .map(
                ([label, bytes, color]) =>
                    `<span style="width:${(bytes / result.tableBytes) * 100}%;background:${color}" title="${label}: ${gib(bytes)}"></span>`,
            )
            .join('')
        get('io-capacity-legend').textContent = segments.map(([label, bytes]) => `${label}: ${gib(bytes)}`).join(' · ')
        get('io-reference-rows').innerHTML = [
            ['Engram-27B memory', 5.7, 'Paper model'],
            ['Engram-40B memory', 18.5, 'Paper model'],
            ['100B memory offload', 100, 'Host-DRAM experiment'],
            ['1T memory', 1000, 'Hypothetical, no date'],
        ]
            .map(([label, paramsB, evidence]) => {
                const scenario = projectEngramIO({ ...input, paramsB })
                return `<tr><th scope="row">${label}</th><td>${gib(scenario.tableBytes)}</td><td>${scenario.spillBytes === 0 ? 'Fits' : `${gib(scenario.spillBytes)} outside memory`}</td><td>${evidence}</td></tr>`
            })
            .join('')
    }
    get('io-preset').addEventListener('change', () => {
        const preset = presets[get('io-preset').value]
        if (preset) {
            get('io-params').value = preset.paramsB
            get('io-layers').value = preset.layers
            get('io-dtype').value = '2'
        }
        renderIOProjection()
    })
    panel.querySelectorAll('input, select').forEach((control) => {
        if (control.id === 'io-preset') return
        control.addEventListener('input', () => {
            if (['io-params', 'io-layers', 'io-dtype'].includes(control.id)) get('io-preset').value = 'custom'
            renderIOProjection()
        })
    })
    renderIOProjection()
}
