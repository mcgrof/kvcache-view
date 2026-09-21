// Benchmark records for the Cartridge Economics calculator.
//
// Static, PWA-compatible data: every measurement that feeds (or is refused
// entry into) the calculator is recorded here with its provenance, validity
// and known defects, so numbers never circulate detached from how they were
// produced. Records marked validity 'invalid' or 'exploratory' MUST NOT be
// surfaced as presets; only measurement-valid records may seed calculator
// fields. A valid record can still fail the calculator's quality gate or lack
// enough cost inputs for an economic verdict.
//
// Field reference (one record):
//   id                    unique slug
//   validity              'valid' | 'exploratory' | 'invalid'
//   provenance            who ran it, where
//   date                  ISO date of the run, null if unknown
//   model                 exact model/checkpoint
//   dataset               dataset / corpus identity
//   corpusTokens          source corpus tokens
//   docTokens             avg source tokens per document
//   cartridgeTokens       trained Cartridge tokens per document
//   compression           source-to-Cartridge token ratio
//   serializedFormat      on-disk KV format actually written
//   serializedBytesDoc    measured serialized bytes per Cartridge
//   harness               training/eval harness and commit
//   engine                inference engine and commit
//   hardware              GPUs used
//   constructionGpuHours  measured GPU-hours for the whole corpus build
//   selfStudyCostUsd      measured self-study generation + inference cost
//   otherBuildCostUsd     other one-time construction cost
//   requestShape          input/query/output token counts
//   concurrency           concurrency and batching policy
//   baselineRun           {gpuHours, completedRequests} for the baseline path
//   cartridgeRun          {gpuHours, completedRequests} for the Cartridge path
//   loadPath              storage tier and staging path, hit rate
//   ttftMs                {baselineP50, baselineP95, cartridgeP50, cartridgeP95}
//   throughputQps         steady-state completed requests per second
//   qualityMetric         metric name
//   qualityBaseline       baseline score
//   qualityCartridge      Cartridge score
//   screenedRetryOverheadPct
//                         failed work / accepted work after quality screening
//   trainingVariation     seed-count and quality-spread summary
//   quantization          post-training payload and loss measurements
//   servingComparisons    scoped latency and throughput comparisons
//   notes                 free text: methodology, known defects
//
// Unknown values are null, never zero — the same discipline as the
// calculator itself.

const CARTRIDGE_BENCHMARKS = {
    schemaVersion: 1,
    records: [
        {
            id: 'local-qwen3-0.6b-frozen-sink-mismatch',
            validity: 'invalid',
            provenance: 'Local exploratory run (kvcache-view author)',
            date: null,
            model: 'Qwen3-0.6B',
            dataset: null,
            corpusTokens: null,
            docTokens: null,
            cartridgeTokens: null,
            compression: null,
            serializedFormat: 'bf16',
            serializedBytesDoc: null,
            harness: null,
            engine: null,
            hardware: null,
            constructionGpuHours: null,
            selfStudyCostUsd: null,
            otherBuildCostUsd: null,
            requestShape: null,
            concurrency: null,
            baselineRun: null,
            cartridgeRun: null,
            loadPath: null,
            ttftMs: null,
            throughputQps: null,
            qualityMetric: 'accuracy',
            qualityBaseline: null,
            qualityCartridge: 0,
            notes:
                'INVALID: the Cartridge was trained with a frozen attention-sink prefix that was missing at ' +
                'evaluation, so the 0% accuracy measures the harness defect, not Cartridge quality. Neither its ' +
                'quality nor its timing may be used as economic evidence or as a calculator preset. Supersede ' +
                'this record with a corrected rerun (frozen sink present at eval) before citing any 0.6B number; ' +
                'a corrected 0.6B run validates instrumentation and cost accounting only — it does not estimate ' +
                '8B economics.',
        },
        {
            id: 'qwen3-8b-longhealth-cas-build',
            validity: 'exploratory',
            provenance: 'knlp CAS reproduction harness, 8x H100 80GB',
            date: '2026-07-30',
            model: 'Qwen3-8B',
            dataset: 'LongHealth patient records (patient_01, DLBCL)',
            corpusTokens: null,
            docTokens: 12221,
            cartridgeTokens: 611,
            compression: 20.0,
            serializedFormat: 'bf16',
            serializedBytesDoc: 90735203,
            harness: 'knlp research/cartridges_cas (HazyResearch cartridges @ 8cb6823)',
            engine: 'vLLM Qwen3-8B teacher (self-study synth) + FlexQwen3 train/serve path',
            hardware: '8x H100 80GB',
            constructionGpuHours: null,
            selfStudyCostUsd: null,
            otherBuildCostUsd: null,
            requestShape:
                'baseline prefills ~12221 document + ~131 query tokens; cartridge is a ~611-token ' +
                'KV prefix + ~131 query tokens; 32 output tokens',
            concurrency: 'single stream (latency microbenchmark; not a concurrent load test)',
            baselineRun: null,
            cartridgeRun: null,
            loadPath: null,
            ttftMs: {
                baselineP50: 757.0,
                baselineP95: 773.4,
                cartridgeP50: 80.6,
                cartridgeP95: 94.8,
            },
            throughputQps: null,
            qualityMetric: 'LongHealth option-match accuracy (thinking-on, temp 0.6, mean of >=3 runs)',
            qualityBaseline: 0.855,
            qualityCartridge: 0.50,
            notes:
                'CAS (arXiv:2606.04557) reproduction on Qwen3-8B over LongHealth. Baselines reproduce: ' +
                'no-context 0.39 (paper 0.375), full document in context 0.855 (paper 0.874). Cartridge-path ' +
                'ceiling: an untrained cartridge holding the full document KV, loaded through the cartridge ' +
                'path, scores 0.86 -- so that path has no positional/serialization loss and a perfect ' +
                'cartridge tops out at 0.86. A trained single isolated cartridge reaches 0.50 (best 0.58) ' +
                'against the paper 0.736; the collapse/rescue effect reproduces at 5 cartridges (isolated ' +
                '0.58 alone -> 0.38 co-loaded; mixed-visibility 0.44 -> 0.46). qualityCartridge here is the ' +
                'trained single-cartridge accuracy (0.50); qualityBaseline is the full document in context ' +
                '(0.855). Geometry/bytes are the trained ~611-token cartridge on disk (bf16); the serving ' +
                'A/B timing was collected with an earlier 512-token cartridge -- the prefill saving is a ' +
                'memcpy of the KV prefix and is insensitive to the small token-count difference, dropping ' +
                'TTFT from 757 ms to 81 ms median (677 ms/query) at ~35 tok/s decode both paths. Still null ' +
                '(measured, never estimated): constructionGpuHours/selfStudyCostUsd (no clean end-to-end ' +
                'build-cost measurement yet); throughputQps/loadPath (no concurrent load test yet). Stays ' +
                'exploratory: the trained-cartridge quality is below the paper and below the 0.86 path ' +
                'ceiling, so it is not a deployable preset. Method, deltas against the public Cartridges ' +
                'implementation, and where the remaining gap lives are documented at ' +
                'https://mcgrof.github.io/knlp/cas.html',
        },
        {
            id: 'qwen3-8b-patient02-cas-seed42-20260920',
            validity: 'valid',
            provenance: 'knlp CAS reproduction, gpu1',
            date: '2026-09-20',
            model: 'Qwen3-8B',
            dataset: 'LongHealth patient_02',
            corpusTokens: 12628,
            docTokens: 12628,
            cartridgeTokens: 632,
            compression: 12628 / 632,
            serializedFormat: 'bf16',
            serializedBytesDoc: 93242105,
            harness: 'knlp d562cde; research/cartridges_cas; HazyResearch cartridges 8cb6823; Table-15 protocol',
            engine: 'Qwen3-8B teacher and FlexQwen3 Cartridge train/evaluation path',
            hardware: '1x NVIDIA H100 80GB per training job',
            constructionGpuHours: 8.25,
            selfStudyCostUsd: null,
            otherBuildCostUsd: null,
            requestShape: null,
            concurrency: null,
            baselineRun: null,
            cartridgeRun: null,
            loadPath: null,
            ttftMs: null,
            throughputQps: null,
            qualityMetric: 'LongHealth option-match accuracy (60 questions, temperature 0.6)',
            qualityBaseline: 0.9833,
            qualityCartridge: 0.7833,
            screenedRetryOverheadPct: 40,
            trainingVariation: {
                distinctSeeds: 7,
                accepted: 5,
                failed: 2,
                meanAccuracy: 0.6667,
                sampleStdDev: 0.1828,
                acceptedMinAccuracy: 0.75,
                acceptedMaxAccuracy: 0.8,
            },
            quantization: {
                rawPayloadMiB: {
                    bf16: 88.875,
                    k16v8: 66.65625,
                    k8v8: 44.4375,
                },
                relativeLossIncreasePct: {
                    k16v8Min: 0.029,
                    k16v8Max: 0.038,
                    k8v8Min: 0.873,
                    k8v8Max: 1.303,
                },
                nativePatient01DecisionMatch: '20/20',
            },
            servingComparisons: {
                fullTextToCartridge: {
                    baselineP50Ms: 757,
                    cartridgeP50Ms: 80.6,
                    scope: 'Older patient_01 single-stream run with a 512-token Cartridge',
                },
                fusedK16V8VsBf16: [
                    {
                        batch: 1,
                        throughputDeltaPct: 0.42,
                        ttftDeltaPct: -2.92,
                    },
                    {
                        batch: 16,
                        throughputDeltaPct: 0.52,
                        ttftDeltaPct: -3.96,
                    },
                ],
            },
            notes:
                'Measurement-valid patient_02 seed-42 reproduction. Four clean one-job-per-GPU runs took ' +
                "8.21-8.29 H100 hours; 8.25 GPU-hours is this preset's rounded training-only measurement. " +
                'The self-study generation cost was not captured, so total construction cost remains unknown. ' +
                'Across seven distinct seeds, five scored 0.75-0.80 and two scored 0.40. Screening the two failed ' +
                'artifacts and replacing them would add two failed runs per five accepted runs, or 40% retry work ' +
                'if this small-sample rate held. The selected deterministic seed-42 artifact scored 0.7833 versus ' +
                'the patient_02 full-document reference at 0.9833, so it fails a one-point parity gate. The BF16 ' +
                'file is 93,242,105 bytes; raw payload is 88.875 MiB because serialization adds framing and ' +
                'metadata. Post-training fake quantization over three trained patient_02 artifacts raises ' +
                'evaluation loss by 0.029-0.038% for K16/V8 and 0.873-1.303% for symmetric K8/V8. Lower loss is ' +
                'better. Corrected native K16/V8 serving matches BF16 decisions on all 20 patient_01 questions; ' +
                'that small check is not a broad quality result. The fused H100 comparison uses patient_01 and ' +
                'shows +0.42%/+0.52% throughput and -2.92%/-3.96% TTFT at batches 1/16 versus the ordinary BF16 ' +
                'Cartridge. The older 757 ms to 80.6 ms full-text/Cartridge TTFT result is single-stream and is ' +
                'not a matched production-cost measurement. Self-study, matched all-in inference, concurrent load, ' +
                'and storage-load measurements remain null. See https://knlp.io/cas.html and ' +
                'https://knlp.io/cartridge-asymmetric-quantization.html.',
        },
    ],
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CARTRIDGE_BENCHMARKS
}
