const test = require('node:test')
const assert = require('node:assert/strict')
const { projectEngramIO } = require('./engram-io.js')
const base = {
    paramsB: 5.7,
    bytesPerValue: 2,
    layers: 2,
    hbmGiB: 0,
    dramGiB: 256,
    tokensPerSecond: 10000,
    readBytes: 4096,
    accessModel: 'uniform',
    residentHit: 95,
}

test('published table scales fit 256 GiB at BF16, including 100B', () => {
    for (const paramsB of [5.7, 18.5, 100]) {
        const r = projectEngramIO({ ...base, paramsB })
        assert.equal(r.tableBytes, paramsB * 2e9)
        assert.equal(r.spillBytes, 0)
        assert.equal(r.ssdIOPS, 0)
    }
    assert.equal(projectEngramIO(base).rowBytes, 160)
    assert.equal(projectEngramIO(base).payloadPerToken, 5120)
})
test('capacity boundary includes unique HBM and clamps excess memory', () => {
    const paramsB = (256 * 2 ** 30) / 2e9
    const exact = projectEngramIO({ ...base, paramsB, hbmGiB: 64, dramGiB: 192 })
    assert.equal(exact.spillBytes, 0)
    assert.equal(exact.fitThresholdB, paramsB)
    assert.ok(projectEngramIO({ ...base, paramsB: paramsB + 1 }).spillBytes > 0)
    const hbmOnly = projectEngramIO({ ...base, hbmGiB: 256 })
    assert.equal(hbmOnly.hbmBytes, hbmOnly.tableBytes)
    assert.equal(hbmOnly.dramBytes, 0)
})
test('all misses distinguish payload from device bytes and layer count', () => {
    const r = projectEngramIO({ ...base, dramGiB: 0 })
    assert.equal(r.ssdIOPS, 320000)
    assert.equal(r.deviceBytesPerSecond, 1310720000)
    assert.equal(r.usefulSSDBytesPerSecond, 51200000)
    assert.equal(r.readAmplification, 25.6)
    const small = projectEngramIO({ ...base, dramGiB: 0, readBytes: 512, layers: 1 })
    assert.equal(small.ssdIOPS, 160000)
    assert.equal(small.deviceBytesPerSecond, 81920000)
    assert.equal(small.readAmplification, 3.2)
})
test('access miss fraction is distinct from capacity spill', () => {
    const uniform = projectEngramIO({ ...base, paramsB: 1000 })
    assert.equal(uniform.missFraction, (2e12 - 256 * 2 ** 30) / 2e12)
    const hot = projectEngramIO({ ...base, paramsB: 1000, accessModel: 'supplied', residentHit: 95 })
    assert.ok(Math.abs(hot.ssdIOPS - 16000) < 1e-8)
    assert.equal(hot.spillBytes, uniform.spillBytes)
    assert.equal(projectEngramIO({ ...base, accessModel: 'supplied', residentHit: 0 }).ssdIOPS, 0)
    assert.equal(projectEngramIO({ ...base, dramGiB: 0, accessModel: 'supplied', residentHit: 100 }).missFraction, 1)
    assert.equal(projectEngramIO({ ...base, paramsB: 1000, tokensPerSecond: 0 }).deviceBytesPerSecond, 0)
})
test('precision changes both row payload and table capacity', () => {
    const r = projectEngramIO({ ...base, paramsB: 100, bytesPerValue: 4 })
    assert.equal(r.rowBytes, 320)
    assert.equal(r.tableBytes, 400e9)
    assert.ok(r.spillBytes > 0)
    assert.equal(projectEngramIO({ ...base, bytesPerValue: 1 }).rowBytes, 80)
})
test('reject invalid projections', () => {
    for (const patch of [
        { paramsB: 0 },
        { layers: 1.5 },
        { dramGiB: -1 },
        { tokensPerSecond: NaN },
        { residentHit: 101 },
        { readBytes: 160 },
        { accessModel: 'zipf' },
    ]) {
        assert.throws(() => projectEngramIO({ ...base, ...patch }), RangeError)
    }
})
