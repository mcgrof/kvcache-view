// SPDX-License-Identifier: MIT
const test = require('node:test')
const assert = require('node:assert/strict')
const vm = require('node:vm')
const fs = require('node:fs')
const { holding, restore, prefill, memory, projection, firstCrossover, traceScenario } = require('./kv-offload-math.js')

const close = (actual, expected, tolerance = 1e-10) => {
    assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`)
}
const parking = {
    bytes: 80000 * 890,
    nandPerGB: 0.1,
    years: 5,
    recomputeSeconds: 0.026,
    gpus: 8,
    gpuPerHour: 8,
}

test('80K × 890 bytes is 71.2 decimal MB; storage days are per-context allocated cost', () => {
    const h = holding(parking)
    close(h.spacePurchase, 0.00712)
    close(h.dailyHold, 0.00712 / 1825)
    close(h.recomputeCost, (0.026 * 64) / 3600)
    close(h.daysPerRecompute, 118.4769038701623)
    close(h.minimumReturnProbability, 1 / h.daysPerRecompute)
    assert.equal(h.netDays, h.daysPerRecompute)
})

test('doubling bytes halves storage days when recompute work stays fixed', () => {
    const base = holding(parking)
    const doubled = holding({ ...parking, bytes: parking.bytes * 2 })
    close(doubled.daysPerRecompute, base.daysPerRecompute / 2)
    close(doubled.spacePurchase, base.spacePurchase * 2)
})

test('restore and optional wear consume the avoided recompute budget', () => {
    const h = holding({ ...parking, restoreSeconds: 0.002, writeBytes: parking.bytes, idleDays: 3 })
    close(h.writeWearCost, h.dailyHold)
    close(h.netDays, ((0.024 * 64) / 3600 - h.writeWearCost) / h.dailyHold)
    close(h.minimumReturnProbability, (3 * h.dailyHold + h.writeWearCost) / ((0.024 * 64) / 3600))
    const slower = holding({ ...parking, restoreSeconds: 0.05 })
    assert.equal(slower.netDays, 0)
    assert.equal(slower.minimumReturnProbability, Infinity)
})

test('zero capital price and zero work do not produce NaN horizons', () => {
    const free = holding({ ...parking, nandPerGB: 0 })
    assert.equal(free.daysPerRecompute, Infinity)
    assert.equal(free.minimumReturnProbability, 0)
    const noCredit = holding({ ...parking, nandPerGB: 0, recomputeSeconds: 0 })
    assert.equal(noCredit.daysPerRecompute, 0)
    assert.equal(noCredit.netDays, 0)
    assert.equal(noCredit.minimumReturnProbability, Infinity)
    const operating = holding({ ...parking, nandPerGB: 0, overheadPerGBMonth: 1 })
    close(operating.dailyHold, (parking.bytes / 1e9) * (12 / 365))
})

test('link cap prevents independent drive bandwidth from exceeding a shared path', () => {
    const input = { bytes: 20e9, driveGBps: 14, drives: 4, linkGBps: 32, efficiency: 0.5, fixedMs: 2, replayMs: 3 }
    const capped = restore(input)
    assert.equal(capped.effectiveGBps, 16)
    assert.equal(capped.payloadSeconds, 1.25)
    assert.equal(capped.seconds, 1.255)
    assert.deepEqual(restore({ ...input, drives: 8 }), capped)
    assert.equal(restore({ ...input, drives: 1 }).effectiveGBps, 7)
})

test('zero payload still pays modeled fixed and bounded replay latency', () => {
    const r = restore({ bytes: 0, driveGBps: 14, drives: 1, linkGBps: 32, efficiency: 1, fixedMs: 1, replayMs: 4 })
    assert.equal(r.payloadSeconds, 0)
    assert.equal(r.seconds, 0.005)
})

test('prefill distinguishes linear parameter work from quadratic attention', () => {
    const base = {
        tokens: 10,
        activeParams: 100,
        quadraticCoeff: 5,
        linearAttentionFlops: 20,
        gpus: 2,
        denseTFLOPS: 1,
        mfu: 0.5,
    }
    const a = prefill(base)
    const b = prefill({ ...base, tokens: 20, overheadMs: 3 })
    assert.equal(a.flops, 2700)
    assert.equal(b.flops, 6400)
    close(a.seconds, 2700 / 1e12)
    close(b.seconds, 6400 / 1e12 + 0.003)
    close(prefill({ ...base, gpus: 4 }).seconds, a.seconds / 2)
})

test('memory includes fixed checkpoints and only the actual retained sliding window', () => {
    const architecture = {
        tokens: 1000,
        bytesPerToken: 500,
        fixedBytes: 1e6,
        windowTokens: 128,
        windowBytesPerToken: 2000,
    }
    assert.equal(memory(architecture), 1756000)
    assert.equal(memory({ ...architecture, tokens: 64 }), 1160000)
    assert.equal(memory({ ...architecture, bytesPerToken: 0, tokens: 10000 }), 1256000)
})

const future = {
    baseYear: 2026,
    endYear: 2030,
    recomputeSeconds: 0.008,
    bytes: 8e6,
    bandwidthGBps: 1,
    computeGrowth: 1,
    bandwidthGrowth: 0,
}

test('fixed architecture matches the exact hardware-only geometric ratio', () => {
    const points = projection({ ...future, recomputeSeconds: 0.064 })
    assert.deepEqual(
        points.map((p) => p.speedup),
        [8, 4, 2, 1, 0.5],
    )
    assert.equal(firstCrossover(points), 2029)
    assert.equal(points[0].bytes, future.bytes)
})

test('same compute and bandwidth growth preserves ratio absent latency floors', () => {
    const points = projection({ ...future, bandwidthGrowth: 1, recomputeSeconds: 0.064 })
    points.forEach((p) => assert.equal(p.speedup, 8))
    assert.equal(firstCrossover(points), null)
})

test('architecture byte decline can exactly offset faster compute', () => {
    const points = projection({ ...future, byteDecline: 0.5, recomputeSeconds: 0.064 })
    points.forEach((p) => assert.equal(p.speedup, 8))
    assert.equal(points.at(-1).bytes, future.bytes / 16)
})

test('work reduction can erase the advantage gained from smaller state', () => {
    const points = projection({ ...future, byteDecline: 0.5, workDecline: 0.5, recomputeSeconds: 0.064 })
    assert.deepEqual(
        points.map((p) => p.speedup),
        [8, 4, 2, 1, 0.5],
    )
    assert.equal(firstCrossover(points), 2029)
})

test('fixed checkpoint bytes and restore overhead are explicit floors', () => {
    const points = projection({ ...future, fixedBytes: 2e6, byteDecline: 1, fixedMs: 1, replayMs: 2 })
    assert.equal(points[0].bytes, 8e6)
    assert.equal(points[1].bytes, 2e6)
    assert.equal(points[0].restoreSeconds, 0.011)
    assert.equal(points.at(-1).restoreSeconds, 0.005)
})

test('context growth scales fixed state, linear work and quadratic attention differently', () => {
    const points = projection({
        ...future,
        computeGrowth: 0,
        contextGrowth: 1,
        fixedBytes: 2e6,
        quadraticFraction: 0.25,
    })
    assert.equal(points[1].bytes, 14e6)
    close(points[1].recomputeSeconds, future.recomputeSeconds * (0.75 * 2 + 0.25 * 4))
    assert.equal(points[0].recomputeSeconds, future.recomputeSeconds)
})

test('first crossover interpolates within years and handles already-lost/empty cases', () => {
    close(
        firstCrossover([
            { year: 2026, speedup: 2 },
            { year: 2027, speedup: 0.5 },
        ]),
        2026.5,
    )
    assert.equal(firstCrossover([{ year: 2026, speedup: 0.8 }]), 2026)
    assert.equal(firstCrossover([]), null)
    assert.equal(firstCrossover([{ year: 2026, speedup: 2 }]), null)
})

const trace = {
    buckets: [{ lo: 10, hi: 100, n: 100, missFraction: 0.8 }],
    retentionSeconds: 100,
    returnProbability: 1,
    validProbability: 1,
    contextBytes: 1e9,
    writeCost: 0,
    dailyHold: 1,
    recomputeCost: 2,
    restoreCost: 0.25,
    arrivalsPerDay: 86400,
}

test('trace misses are partial recompute equivalents and do not become binary TTL misses', () => {
    const result = traceScenario(trace)
    assert.equal(result.caughtProbability, 1)
    assert.equal(result.equivalentMissesCaught, 0.8)
    assert.equal(result.creditedRestoreProbability, 1)
    assert.equal(result.avoidedRecomputesPerDay, 69120)
    assert.equal(result.avoidedRecomputeEquivalentsPerDay, 69120)
    close(result.meanHoldingSeconds, 90 / Math.log(10))
    close(result.expectedNetPerArrival, 0.8 * 2 - 0.25 - result.meanHoldingSeconds / 86400)
    close(result.parkedBytes, result.meanHoldingSeconds * 1e9)
})

test('abandoned sessions pay full retention and lower expected benefit', () => {
    const r = traceScenario({ ...trace, returnProbability: 0.5, validProbability: 0.75 })
    close(r.meanHoldingSeconds, 0.5 * (90 / Math.log(10)) + 0.5 * 100)
    assert.equal(r.caughtProbability, 0.375)
    assert.equal(r.creditedRestoreProbability, 0.375)
    close(r.equivalentMissesCaught, 0.3)
    const terminal = traceScenario({ ...trace, returnProbability: 0 })
    assert.equal(terminal.meanHoldingSeconds, 100)
    assert.equal(terminal.expectedNetPerArrival, -100 / 86400)
    assert.equal(terminal.avoidedRecomputesPerDay, 0)
    assert.equal(terminal.creditedRestoreProbability, 0)
})

test('retention within a log-uniform bucket integrates residence time including evictions', () => {
    const r = traceScenario({ ...trace, retentionSeconds: Math.sqrt(1000) })
    close(r.conditionalCaught, 0.5)
    close(r.meanHoldingSeconds, (Math.sqrt(1000) - 10) / Math.log(10) + Math.sqrt(1000) * 0.5)
    assert.equal(traceScenario({ ...trace, retentionSeconds: 5 }).conditionalCaught, 0)
    assert.equal(traceScenario({ ...trace, retentionSeconds: 5 }).meanHoldingSeconds, 5)
})

test('turning retention off performs no writes, incurs no storage, catches no return', () => {
    const r = traceScenario({ ...trace, retentionSeconds: 0, writeCost: 50 })
    assert.equal(r.expectedNetPerArrival, 0)
    assert.equal(r.writesPerDay, 0)
    assert.equal(r.parkedBytes, 0)
    assert.equal(r.caughtProbability, 0)
})

test('weighted bins keep their own baseline uncached fractions', () => {
    const r = traceScenario({
        ...trace,
        buckets: [
            { lo: 10, hi: 100, n: 75, missFraction: 0 },
            { lo: 100, hi: 1000, n: 25, missFraction: 1 },
        ],
        retentionSeconds: 1000,
    })
    assert.equal(r.conditionalMissesCaught, 0.25)
    assert.equal(r.creditedRestoreProbability, 0.25)
    assert.equal(r.conditionalCaught, 1)
})

test('small partial misses still incur a full restore, while zero-credit bins do not', () => {
    const r = traceScenario({
        ...trace,
        buckets: [
            { lo: 10, hi: 100, n: 50, missFraction: 0 },
            { lo: 100, hi: 1000, n: 50, missFraction: 0.01 },
        ],
        retentionSeconds: 1000,
        dailyHold: 0,
    })
    assert.equal(r.equivalentMissesCaught, 0.005)
    assert.equal(r.creditedRestoreProbability, 0.5)
    close(r.expectedNetPerArrival, 0.005 * 2 - 0.5 * 0.25)
    assert.ok(r.expectedNetPerArrival < 0)
})

test('invalid inputs fail explicitly instead of silently generating plausible charts', () => {
    assert.throws(() => holding({ ...parking, bytes: -1 }), RangeError)
    assert.throws(() => holding({ ...parking, years: 0 }), RangeError)
    assert.throws(() => holding({ ...parking, dwpd: 0 }), RangeError)
    assert.throws(() => holding({ ...parking, gpuPerHour: NaN }), RangeError)
    assert.throws(() => restore({ bytes: 1, driveGBps: 1, drives: 1, linkGBps: 1, efficiency: 1.1 }), RangeError)
    assert.throws(() => prefill({ tokens: 1, activeParams: 1, gpus: 1, denseTFLOPS: 1, mfu: 0 }), RangeError)
    assert.throws(() => memory({ tokens: -1, bytesPerToken: 1 }), RangeError)
    assert.throws(() => projection({ ...future, fixedBytes: 9e6 }), RangeError)
    assert.throws(() => projection({ ...future, computeGrowth: -1 }), RangeError)
    assert.throws(() => projection({ ...future, byteDecline: 1.01 }), RangeError)
    assert.throws(() => traceScenario({ ...trace, returnProbability: 1.01 }), RangeError)
    assert.throws(() => traceScenario({ ...trace, buckets: [{ lo: 10, hi: 10, n: 1, missFraction: 1 }] }), RangeError)
})

test('the same pure API is exposed to browsers without Node dependencies', () => {
    const browser = {}
    vm.runInNewContext(fs.readFileSync(require.resolve('./kv-offload-math.js'), 'utf8'), browser)
    assert.equal(typeof browser.KVOffloadMath.holding, 'function')
    close(browser.KVOffloadMath.holding(parking).daysPerRecompute, holding(parking).daysPerRecompute)
})
