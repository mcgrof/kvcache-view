const test = require('node:test')
const assert = require('node:assert/strict')
const { MODELS, geometry, storedCheckpoint, alignedPrefill, resumePoint } = require('./hybrid-checkpoints.js')

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

test('a budget below 2N cuts prefill into single blocks and saves every boundary', () => {
    const p = alignedPrefill(6000, 528, 2 * 528 - 1)
    assert.equal(p.steps.length, 12)
    assert.deepEqual(p.steps[0], { start: 0, end: 528 })
    assert.deepEqual(p.steps.at(-1), { start: 5808, end: 6000 })
    assert.equal(p.checkpoints.length, 11)
    assert.equal(p.checkpoints.at(-1), 5808)
    assert.deepEqual(p.skipped, [])
})

test('a large budget saves only the boundaries where a step ends', () => {
    const one = alignedPrefill(6000, 528, 8192)
    assert.deepEqual(one.steps, [
        { start: 0, end: 5808 },
        { start: 5808, end: 6000 },
    ])
    assert.deepEqual(one.checkpoints, [5808])
    assert.equal(one.skipped.length, 10)
    assert.deepEqual(alignedPrefill(6000, 528, 2048).checkpoints, [1584, 3168, 4752, 5808])
})

test('a prompt shorter than one block has no checkpoint', () => {
    const p = alignedPrefill(4000, 4128, 2 * 4128 - 1)
    assert.deepEqual(p.steps, [{ start: 0, end: 4000 }])
    assert.deepEqual(p.boundaries, [])
    assert.deepEqual(p.checkpoints, [])
})

test('a budget below N advances within a block and stops at the next boundary', () => {
    const p = alignedPrefill(1100, 528, 256)
    assert.deepEqual(p.checkpoints, [528, 1056])
    assert.ok(p.steps.some((s) => s.start === 512 && s.end === 528))
})

test('resume uses the last checkpoint at or before the shared length', () => {
    const dense = alignedPrefill(6000, 528, 1055).checkpoints
    assert.deepEqual(resumePoint(dense, 528, 4000, 6001), {
        limit: 4000,
        attention: 3696,
        hybrid: 3696,
        recompute: 304,
    })
    const sparse = alignedPrefill(6000, 528, 8192).checkpoints
    assert.deepEqual(resumePoint(sparse, 528, 4000, 6001), { limit: 4000, attention: 3696, hybrid: 0, recompute: 4000 })
    assert.equal(resumePoint(sparse, 528, 6000, 6001).hybrid, 5808)
})

test('an exact repeat cannot use a checkpoint at its own end', () => {
    const cps = alignedPrefill(1056, 528, 1055).checkpoints
    assert.deepEqual(cps, [528, 1056])
    assert.equal(resumePoint(cps, 528, 1056, 1056).hybrid, 528)
    assert.equal(resumePoint(cps, 528, 1056, 1057).hybrid, 1056)
    assert.equal(resumePoint([528], 528, 528, 528).hybrid, 0)
    assert.equal(resumePoint([], 528, 1, 1).recompute, 1)
})

test('invalid geometry and schedule inputs are rejected', () => {
    assert.throws(() => geometry(MODELS[0], 1, 16), RangeError)
    assert.throws(() => geometry(MODELS[0], 4, 0), RangeError)
    assert.throws(() => storedCheckpoint(MODELS[0], 4, 'unknown'), RangeError)
    assert.throws(() => alignedPrefill(0, 528, 1055), RangeError)
    assert.throws(() => alignedPrefill(6000, 528, 0), RangeError)
    assert.throws(() => alignedPrefill(6000, 528, 1.5), RangeError)
    assert.throws(() => resumePoint([], 528, -1, 100), RangeError)
    assert.throws(() => resumePoint([], 528, 101, 100), RangeError)
})
