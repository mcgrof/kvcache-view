const test = require('node:test')
const assert = require('node:assert/strict')
const { MODELS, geometry, storedCheckpoint, reuse } = require('./hybrid-checkpoints.js')

test('Qwen reference geometry includes convolution history and upward alignment', () => {
    const g = geometry(MODELS[0], 4, 16)
    assert.equal(g.recPage, 2146304)
    assert.equal(g.attnPerToken, 4096)
    assert.equal(g.block, 528)
    assert.equal(g.page, 2162688)
    assert.equal(g.paddingBytes, 16384)
    assert.equal(geometry(MODELS[0], 2, 16).block, 272)
    assert.equal(geometry(MODELS[0], 4, 128).block, 640)
})

test('Nemotron uses its actual Mamba intermediate width and eight B/C groups', () => {
    const m = MODELS[1]
    assert.equal(m.conv, 27648)
    assert.equal(geometry(m, 4, 16).recPage, 3725312)
    assert.equal(geometry(m, 4, 16).block, 912)
})

test('every model has enough page capacity with less than one alignment quantum of padding', () => {
    for (const m of MODELS) {
        for (const bytes of [2, 4]) {
            for (const align of [16, 64, 128, 256]) {
                const g = geometry(m, bytes, align)
                assert.equal(g.block % align, 0)
                assert.ok(g.page >= g.recPage)
                assert.ok(g.paddingBytes < align * g.attnPerToken)
            }
        }
    }
})

test('serialization counts row scales and preserves live geometry', () => {
    const m = MODELS[0]
    const before = geometry(m, 4, 16)
    const packed = storedCheckpoint(m, 4, 'int8')
    assert.equal(packed.scaleBytes, 16384)
    assert.equal(packed.bytes, 589824)
    assert.equal(packed.bytes * m.rec, 14155776)
    assert.equal(storedCheckpoint(m, 4, 'native').bytes, before.recPage)
    assert.equal(storedCheckpoint(m, 4, 'bf16').bytes, geometry(m, 2, 16).recPage)
    assert.deepEqual(geometry(m, 4, 16), before)
})

test('exact prompt-end checkpoint is excluded when logits require replay', () => {
    assert.deepEqual(reuse(1056, 528, 1, true), {
        candidates: 2,
        stored: 2,
        usable: 1,
        hit: 528,
        recompute: 528,
        spacing: 528,
    })
    assert.equal(reuse(1056, 528, 1, false).hit, 1056)
    assert.equal(reuse(1057, 528, 1, true).hit, 1056)
    assert.equal(reuse(528, 528, 1, true).hit, 0)
    assert.equal(reuse(1, 528, 1, true).recompute, 1)
})

test('sparse retained checkpoints cannot be inferred from allocation boundaries', () => {
    const r = reuse(8192, 528, 4, true)
    assert.equal(r.candidates, 15)
    assert.equal(r.stored, 3)
    assert.equal(r.hit, 6336)
    assert.equal(r.recompute, 1856)
    assert.equal(reuse(1000, 528, 4).stored, 0)
})

test('invalid geometry and schedule inputs are rejected', () => {
    assert.throws(() => geometry(MODELS[0], 1, 16), RangeError)
    assert.throws(() => geometry(MODELS[0], 4, 0), RangeError)
    assert.throws(() => storedCheckpoint(MODELS[0], 4, 'unknown'), RangeError)
    assert.throws(() => reuse(0, 528, 1), RangeError)
    assert.throws(() => reuse(8192, 528, 0), RangeError)
    assert.throws(() => reuse(8192, 528, 1.5), RangeError)
})
