// SPDX-License-Identifier: MIT
const test = require('node:test')
const assert = require('node:assert/strict')
const { activeStateBytes } = require('./architecture-memory.js')

test('default active bytes use KV heads, count K and V, and retain full-layer KV', () => {
    const result = activeStateBytes(131072)
    assert.equal(result.full, 16 * 1024 ** 3)
    assert.equal(result.swa, 64 * 1024 ** 2)
    assert.equal(result.recurrent, 32 * 1024 ** 2)
    assert.equal(result.localHybrid, 4 * 1024 ** 3 + 48 * 1024 ** 2)
    assert.equal(result.recurrentHybrid, 4 * 1024 ** 3 + 24 * 1024 ** 2)
})

test('hybrid extremes reduce to pure designs and short windows retain only actual tokens', () => {
    const noFull = activeStateBytes(32, { fullLayers: 0 })
    const allFull = activeStateBytes(32, { fullLayers: 32 })
    assert.equal(noFull.localHybrid, noFull.swa)
    assert.equal(noFull.recurrentHybrid, noFull.recurrent)
    assert.equal(noFull.swa, noFull.full)
    assert.equal(allFull.localHybrid, allFull.full)
    assert.equal(allFull.recurrentHybrid, allFull.full)
})

test('concurrency multiplies every design; bounded terms do not grow with long context', () => {
    const one = activeStateBytes(131072)
    const many = activeStateBytes(131072, { sequences: 128 })
    const longer = activeStateBytes(1048576)
    for (const key in one) assert.equal(many[key], one[key] * 128)
    assert.equal(longer.full, one.full * 8)
    assert.equal(longer.swa, one.swa)
    assert.equal(longer.recurrent, one.recurrent)
    assert.throws(() => activeStateBytes(16384, { fullLayers: 33 }), RangeError)
})
