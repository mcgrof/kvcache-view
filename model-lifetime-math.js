// SPDX-License-Identifier: MIT
// Scenario arithmetic only. No request counts or hardware costs are measured here.
;(function (root, factory) {
    const api = factory()
    if (typeof module === 'object' && module.exports) module.exports = api
    else root.ModelLifetimeMath = api
})(typeof globalThis === 'object' ? globalThis : this, function () {
    function bounded(name, value, min, max = Infinity) {
        if (!Number.isFinite(value) || value < min || value > max) {
            throw new RangeError(`${name} must be finite and between ${min} and ${max}`)
        }
        return value
    }

    function demand({
        observedTokens,
        dailyTokens,
        horizonDays,
        weeklyGrowthPercent,
        tailDays,
        tailPercent,
        inputTokens,
        outputTokens,
        activeParametersB,
        outputActiveParametersB = activeParametersB,
    }) {
        bounded('observed tokens', observedTokens, 0)
        bounded('daily tokens', dailyTokens, 0)
        bounded('horizon days', horizonDays, 0, 3650)
        bounded('weekly growth', weeklyGrowthPercent, -99, 100)
        bounded('tail days', tailDays, 0, 3650)
        bounded('tail percent', tailPercent, 0, 100)
        bounded('input tokens', inputTokens, 0)
        bounded('output tokens', outputTokens, 0)
        bounded('active prefill parameters', activeParametersB, 0.001, 10000)
        bounded('active decode parameters', outputActiveParametersB, 0.001, 10000)
        const tokensPerRequest = inputTokens + outputTokens
        if (!Number.isFinite(tokensPerRequest) || tokensPerRequest <= 0) {
            throw new RangeError('Average input plus output must exceed zero')
        }
        const logGrowth = Math.log1p(weeklyGrowthPercent / 100) / 7
        // Continuous integration allows fractional empirical remaining-life values.
        // A zero horizon contributes no tokens; a flat rate gives rate × days.
        const integrate = (rate, days) =>
            Math.abs(logGrowth) < 1e-12 ? rate * days : (rate * Math.expm1(logGrowth * days)) / logGrowth
        const prominenceTokens = integrate(dailyTokens, horizonDays)
        const rateAtExit = dailyTokens * Math.exp(logGrowth * horizonDays)
        // Post-prominence tail is deliberately flat: its rate and duration are assumptions.
        const tailTokens = rateAtExit * (tailPercent / 100) * tailDays
        const futureTokens = prominenceTokens + tailTokens
        const modeledTokens = observedTokens + futureTokens
        const requests = modeledTokens / tokensPerRequest
        // The input/output token split is itself an explicit request-length assumption.
        const effectiveActiveParametersB =
            (activeParametersB * inputTokens + outputActiveParametersB * outputTokens) / tokensPerRequest
        const inferenceFlops = 2 * effectiveActiveParametersB * 1e9 * modeledTokens
        if (![modeledTokens, requests, inferenceFlops].every(Number.isFinite)) {
            throw new RangeError('Scenario overflows; reduce growth or horizon')
        }
        return {
            tokensPerRequest,
            prominenceTokens,
            rateAtExit,
            tailTokens,
            futureTokens,
            modeledTokens,
            requestsFromObservedTokens: observedTokens / tokensPerRequest,
            futureRequests: futureTokens / tokensPerRequest,
            requests,
            effectiveActiveParametersB,
            inferenceFlops,
        }
    }

    return { demand }
})
