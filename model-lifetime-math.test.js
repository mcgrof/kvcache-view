// SPDX-License-Identifier: MIT
const test = require('node:test')
const assert = require('node:assert/strict')
const { demand } = require('./model-lifetime-math.js')

const base = {
    observedTokens: 285e9,
    dailyTokens: 1e9,
    horizonDays: 90,
    weeklyGrowthPercent: 0,
    tailDays: 180,
    tailPercent: 10,
    inputTokens: 70,
    outputTokens: 215,
    activeParametersB: 30,
}

test('paper workload: a billion requests represents 285 billion tokens', () => {
    const x = demand({ ...base, horizonDays: 0, tailDays: 0 })
    assert.equal(x.requestsFromObservedTokens, 1e9)
    assert.equal(x.inferenceFlops, 17.1e21)
    assert.equal(x.futureTokens, 0)
})

test('flat prominence and separately assumed post-exit demand add exactly once', () => {
    const x = demand(base)
    assert.equal(x.prominenceTokens, 90e9)
    assert.equal(x.tailTokens, 18e9)
    assert.equal(x.modeledTokens, 393e9)
})

test('decaying demand uses the exit rate for the tail', () => {
    const x = demand({ ...base, horizonDays: 7, weeklyGrowthPercent: -50 })
    assert.ok(Math.abs(x.rateAtExit - 0.5e9) < 1e-6)
    assert.ok(Math.abs(x.prominenceTokens - 3.5e9 / Math.log(2)) < 1e-5)
    assert.equal(x.tailTokens, 9e9)
})

test('length changes request equivalents but not token-based arithmetic compute', () => {
    const short = demand(base)
    const long = demand({ ...base, inputTokens: 64000, outputTokens: 4000 })
    assert.equal(short.inferenceFlops, long.inferenceFlops)
    assert.ok(short.requests > long.requests * 200)
})

test('zero length, negative demand and overflow are rejected', () => {
    assert.throws(() => demand({ ...base, inputTokens: 0, outputTokens: 0 }), RangeError)
    assert.throws(() => demand({ ...base, observedTokens: -1 }), RangeError)
    assert.throws(() => demand({ ...base, dailyTokens: NaN }), RangeError)
    assert.throws(() => demand({ ...base, weeklyGrowthPercent: -100 }), RangeError)
    assert.throws(
        () => demand({ ...base, horizonDays: 3650, weeklyGrowthPercent: 100, dailyTokens: 1e300 }),
        RangeError,
    )
})

test('phase-dependent parameters use the assumed input/output split', () => {
    const x = demand({ ...base, horizonDays: 0, tailDays: 0, activeParametersB: 8, outputActiveParametersB: 16 })
    assert.ok(Math.abs(x.inferenceFlops / (2 * (8e9 * 70e9 + 16e9 * 215e9)) - 1) < 1e-12)
})
