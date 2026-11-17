// SPDX-License-Identifier: MIT
// Pure scenario arithmetic. Prices use USD and decimal GB (1e9 bytes).
;(function (root, factory) {
    const api = factory()
    if (typeof module === 'object' && module.exports) module.exports = api
    else root.KVOffloadMath = api
})(typeof globalThis === 'object' ? globalThis : this, function () {
    const GB = 1e9
    const DAY = 86400

    function number(name, value, min = 0, max = Infinity) {
        if (!Number.isFinite(value) || value < min || value > max) {
            throw new RangeError(`${name} must be finite and between ${min} and ${max}`)
        }
        return value
    }

    function positive(name, value) {
        number(name, value)
        if (value === 0) throw new RangeError(`${name} must be greater than zero`)
        return value
    }

    function horizon(credit, rate) {
        if (credit <= 0) return 0
        return rate === 0 ? Infinity : credit / rate
    }

    /**
     * Marginal allocation accounting, not a whole-drive purchase decision.
     * GPU seconds assume exclusive use of the stated GPUs. For batched serving,
     * substitute attributable GPU time or measured cost per completed request.
     * Write wear is a conservative allocation proxy; adding it to amortization
     * is not an independent cash charge and can double-allocate drive capital.
     */
    function holding({
        bytes,
        nandPerGB,
        years,
        recomputeSeconds,
        gpus,
        gpuPerHour,
        writeBytes = 0,
        dwpd = 1,
        restoreSeconds = 0,
        overheadPerGBMonth = 0,
        idleDays = 1,
    }) {
        for (const [key, value] of Object.entries({
            bytes,
            nandPerGB,
            recomputeSeconds,
            gpuPerHour,
            writeBytes,
            restoreSeconds,
            overheadPerGBMonth,
            idleDays,
        })) {
            number(key, value)
        }
        positive('years', years)
        positive('gpus', gpus)
        positive('dwpd', dwpd)
        const spacePurchase = (bytes / GB) * nandPerGB
        const dailyHold = spacePurchase / (365 * years) + (bytes / GB) * overheadPerGBMonth * (12 / 365)
        const gpuPerSecond = (gpus * gpuPerHour) / 3600
        const recomputeCost = recomputeSeconds * gpuPerSecond
        const restoreCost = restoreSeconds * gpuPerSecond
        const writeWearCost = ((writeBytes / GB) * nandPerGB) / (dwpd * 365 * years)
        const savedOnReturn = recomputeCost - restoreCost
        const idleCost = dailyHold * idleDays
        return {
            spacePurchase,
            dailyHold,
            recomputeCost,
            restoreCost,
            writeWearCost,
            idleCost,
            savedOnReturn,
            daysPerRecompute: horizon(recomputeCost, dailyHold),
            netDays: horizon(savedOnReturn - writeWearCost, dailyHold),
            // Values above one mean no possible return probability pays at this idle time.
            minimumReturnProbability: savedOnReturn > 0 ? (idleCost + writeWearCost) / savedOnReturn : Infinity,
        }
    }

    /** Shared link and drives are concurrent limits, not additive bandwidth. */
    function restore({ bytes, driveGBps, drives, efficiency, linkGBps, fixedMs = 0, replayMs = 0 }) {
        number('bytes', bytes)
        positive('driveGBps', driveGBps)
        positive('drives', drives)
        positive('linkGBps', linkGBps)
        positive('efficiency', efficiency)
        number('efficiency', efficiency, 0, 1)
        number('fixedMs', fixedMs)
        number('replayMs', replayMs)
        const effectiveGBps = Math.min(drives * driveGBps, linkGBps) * efficiency
        const payloadSeconds = bytes / (effectiveGBps * GB)
        return { effectiveGBps, payloadSeconds, seconds: payloadSeconds + (fixedMs + replayMs) / 1000 }
    }

    /** An idealized arithmetic estimate; use dense, same-precision peak FLOPS. */
    function prefill({
        tokens,
        activeParams,
        quadraticCoeff = 0,
        linearAttentionFlops = 0,
        gpus,
        denseTFLOPS,
        mfu,
        overheadMs = 0,
    }) {
        for (const [key, value] of Object.entries({
            tokens,
            activeParams,
            quadraticCoeff,
            linearAttentionFlops,
            overheadMs,
        })) {
            number(key, value)
        }
        positive('gpus', gpus)
        positive('denseTFLOPS', denseTFLOPS)
        positive('mfu', mfu)
        number('mfu', mfu, 0, 1)
        const flops = 2 * activeParams * tokens + quadraticCoeff * tokens * tokens + linearAttentionFlops * tokens
        return { flops, seconds: flops / (gpus * denseTFLOPS * 1e12 * mfu) + overheadMs / 1000 }
    }

    /** Persisted global KV + fixed recurrent checkpoint + any retained local window. */
    function memory({ tokens, bytesPerToken, fixedBytes = 0, windowTokens = 0, windowBytesPerToken = 0 }) {
        for (const [key, value] of Object.entries({
            tokens,
            bytesPerToken,
            fixedBytes,
            windowTokens,
            windowBytesPerToken,
        })) {
            number(key, value)
        }
        return tokens * bytesPerToken + fixedBytes + Math.min(tokens, windowTokens) * windowBytesPerToken
    }

    /**
     * Annual scenarios, never forecasts inferred from two hardware releases.
     * bytes includes fixedBytes. Only the variable part shrinks with byteDecline
     * or grows with context. Work grows linearly/quadratically with context using
     * the baseline quadraticFraction. Fixed I/O and replay latency remain floors.
     * Declines are fractions removed each year; growth is fractional annual gain.
     */
    function projection({
        baseYear = 2026,
        endYear = 2035,
        recomputeSeconds,
        bytes,
        bandwidthGBps,
        fixedMs = 0,
        replayMs = 0,
        computeGrowth,
        bandwidthGrowth,
        byteDecline = 0,
        workDecline = 0,
        contextGrowth = 0,
        fixedBytes = 0,
        quadraticFraction = 0,
    }) {
        number('baseYear', baseYear)
        number('endYear', endYear, baseYear, baseYear + 1000)
        if (!Number.isInteger(baseYear) || !Number.isInteger(endYear)) throw new RangeError('Years must be integers')
        for (const [key, value] of Object.entries({ recomputeSeconds, bytes, fixedMs, replayMs })) number(key, value)
        positive('bandwidthGBps', bandwidthGBps)
        number('fixedBytes', fixedBytes, 0, bytes)
        for (const [key, value] of Object.entries({ computeGrowth, bandwidthGrowth, contextGrowth })) {
            number(key, value, -1)
            if (value === -1) throw new RangeError(`${key} must be greater than -1`)
        }
        for (const [key, value] of Object.entries({ byteDecline, workDecline, quadraticFraction })) {
            number(key, value, 0, 1)
        }
        return Array.from({ length: endYear - baseYear + 1 }, (_, age) => {
            const context = (1 + contextGrowth) ** age
            const projectedBytes = fixedBytes + (bytes - fixedBytes) * context * (1 - byteDecline) ** age
            const work = (1 - quadraticFraction) * context + quadraticFraction * context * context
            const computeTime = (recomputeSeconds * work * (1 - workDecline) ** age) / (1 + computeGrowth) ** age
            const restoreTime =
                projectedBytes / (GB * bandwidthGBps * (1 + bandwidthGrowth) ** age) + (fixedMs + replayMs) / 1000
            return {
                year: baseYear + age,
                bytes: projectedBytes,
                recomputeSeconds: computeTime,
                restoreSeconds: restoreTime,
                speedup: restoreTime === 0 ? (computeTime === 0 ? 1 : Infinity) : computeTime / restoreTime,
            }
        })
    }

    /** First year restore ceases to be faster; interpolates log ratio between samples. */
    function firstCrossover(points) {
        if (!Array.isArray(points) || points.length === 0) return null
        for (let i = 0; i < points.length; i++) {
            const current = points[i]
            if (current.speedup > 1) continue
            if (i === 0) return current.year
            const previous = points[i - 1]
            if (current.speedup <= 0 || !Number.isFinite(previous.speedup)) return current.year
            const fraction = Math.log(previous.speedup) / (Math.log(previous.speedup) - Math.log(current.speedup))
            return previous.year + fraction * (current.year - previous.year)
        }
        return null
    }

    function bucketStats(bucket, retentionSeconds) {
        const { lo, hi } = bucket
        if (retentionSeconds <= lo) return { caught: 0, held: retentionSeconds }
        const span = Math.log(hi / lo)
        const upper = Math.min(retentionSeconds, hi)
        const caught = Math.log(upper / lo) / span
        const held = (upper - lo) / span + retentionSeconds * (1 - caught)
        return { caught, held }
    }

    /**
     * Conditional observed gap bins cannot measure the chance an idle session
     * ever returns. returnProbability supplies that separate scenario assumption.
     * Terminal/unobserved contexts occupy space for the whole retention window.
     * missFraction is the baseline fraction requiring recompute in a bin (not a
     * binary five-minute TTL). Interpret fractional prompt misses as equivalent
     * full-context recomputes under linear cost; long attention costs can differ.
     * Conservatively charge one full-context restore for every caught valid turn
     * in any bin with a nonzero missFraction; fixed restore overhead is not
     * fractionally discounted. A bin average cannot identify actual hit/miss
     * probabilities. Zero-miss bins have neither credit nor an incremental load.
     * validProbability is an additional independent scenario, defaulting to one.
     */
    function traceScenario({
        buckets,
        retentionSeconds,
        returnProbability = 1,
        validProbability = 1,
        contextBytes,
        writeCost = 0,
        dailyHold,
        recomputeCost,
        restoreCost = 0,
        arrivalsPerDay,
    }) {
        if (!Array.isArray(buckets) || buckets.length === 0) throw new RangeError('At least one gap bucket is required')
        for (const [key, value] of Object.entries({
            retentionSeconds,
            contextBytes,
            writeCost,
            dailyHold,
            recomputeCost,
            restoreCost,
            arrivalsPerDay,
        })) {
            number(key, value)
        }
        number('returnProbability', returnProbability, 0, 1)
        number('validProbability', validProbability, 0, 1)
        let total = 0
        for (const bucket of buckets) {
            positive('bucket.lo', bucket.lo)
            positive('bucket.hi', bucket.hi)
            if (bucket.hi <= bucket.lo) throw new RangeError('Bucket upper bound must exceed lower bound')
            number('bucket.n', bucket.n)
            number('bucket.missFraction', bucket.missFraction, 0, 1)
            total += bucket.n
        }
        positive('total bucket count', total)
        let conditionalCaught = 0
        let conditionalMissesCaught = 0
        let conditionalRestoresCaught = 0
        let conditionalHolding = 0
        for (const bucket of buckets) {
            const weight = bucket.n / total
            const stats = bucketStats(bucket, retentionSeconds)
            conditionalCaught += weight * stats.caught
            conditionalMissesCaught += weight * stats.caught * bucket.missFraction
            conditionalRestoresCaught += weight * stats.caught * (bucket.missFraction > 0 ? 1 : 0)
            conditionalHolding += weight * stats.held
        }
        const caughtProbability = returnProbability * validProbability * conditionalCaught
        const equivalentMissesCaught = returnProbability * validProbability * conditionalMissesCaught
        const creditedRestoreProbability = returnProbability * validProbability * conditionalRestoresCaught
        const meanHoldingSeconds = returnProbability * conditionalHolding + (1 - returnProbability) * retentionSeconds
        const holdingCost = (meanHoldingSeconds / DAY) * dailyHold
        // Partial uncached tokens reduce compute credit, while this conservative
        // whole-context restore boundary still charges the entire transfer.
        const writesPerArrival = retentionSeconds > 0 ? 1 : 0
        const expectedNetPerArrival =
            equivalentMissesCaught * recomputeCost -
            creditedRestoreProbability * restoreCost -
            holdingCost -
            writesPerArrival * writeCost
        return {
            conditionalCaught,
            conditionalMissesCaught,
            conditionalRestoresCaught,
            caughtProbability,
            equivalentMissesCaught,
            creditedRestoreProbability,
            avoidedRecomputesPerDay: arrivalsPerDay * equivalentMissesCaught,
            avoidedRecomputeEquivalentsPerDay: arrivalsPerDay * equivalentMissesCaught,
            expectedNetPerArrival,
            meanHoldingSeconds,
            parkedContexts: (arrivalsPerDay * meanHoldingSeconds) / DAY,
            parkedBytes: (arrivalsPerDay * meanHoldingSeconds * contextBytes) / DAY,
            writesPerDay: arrivalsPerDay * writesPerArrival,
            holdingCost,
        }
    }

    return { holding, restore, prefill, memory, projection, firstCrossover, traceScenario }
})
