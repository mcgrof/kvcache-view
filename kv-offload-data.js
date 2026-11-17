/*
 * Evidence and deliberately explicit scenarios for the offloading projections.
 * Bytes use decimal units. Peak FLOPS are dense FP8 per GPU, not measured prefill.
 * Do not infer a complete engine serialization format from model config alone.
 */
;(function (root, factory) {
    const data = factory()
    if (typeof module === 'object' && module.exports) module.exports = data
    else root.KVOffloadData = data
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict'

    const sources = [
        {
            id: 'ds41',
            title: 'DeepSeek-V4.1-Flash technical report (September 2026)',
            url: 'https://arxiv.org/html/2609.19969v1',
            note: '890 B/token global cache; encoder-only bulk prefill; approximate 128-token SWA bounded replay; persistent global KV retained for at least 72 hours in the reported deployment.',
        },
        {
            id: 'ds41-card',
            title: 'DeepSeek-V4.1-Flash official model card',
            url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash',
            note: '552B backbone and 8B prefill-active parameters; cross-layer CSA2 sharing and FP4 main KV. The 26 ms Rubin baseline on these pages is a user scenario, not a reported benchmark.',
        },
        {
            id: 'glm-config',
            title: 'GLM-5.3-Flash official configuration',
            url: 'https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json',
            note: '34 KDA layers with 64 heads of dimension 128, 11 sparse MLA layers, 512-wide latent, KPool factor four; state precision and complete serialized bytes require engine validation.',
        },
        {
            id: 'glm-card',
            title: 'GLM-5.3-Flash official model card',
            url: 'https://huggingface.co/zai-org/GLM-5.3-Flash',
            note: '320B total and 18B active parameters; hybrid sparse and linear attention.',
        },
        {
            id: 'hy4-config',
            title: 'Hy4 preview official configuration',
            url: 'https://huggingface.co/tencent/Hy4-preview/raw/main/config.json',
            note: '78 layers, 512 latent and 64 RoPE dimensions, 21 full indexer layers with cross-layer sharing; 32 indexer query heads of dimension 128.',
        },
        {
            id: 'hy4-card',
            title: 'Hy4 preview official model card',
            url: 'https://huggingface.co/tencent/Hy4-preview',
            note: '770B backbone and 49B active parameters. FP8 weight and KV formats are separate choices; MTP is additional.',
        },
        {
            id: 'minimax-config',
            title: 'MiniMax-M3.1-Flash official configuration',
            url: 'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/config.json',
            note: '60 layers, 64 query heads, four KV heads, head dimension 128; four indexer heads, not 32; 16 selected blocks of 128 tokens.',
        },
        {
            id: 'minimax-card',
            title: 'MiniMax-M3.1-Flash official model card',
            url: 'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash',
            note: 'Approximately 428B total and 23B active parameters; attention is sparse throughout.',
        },
        {
            id: 'minimax-nim',
            title: 'NVIDIA MiniMax-M3.1-Flash serving documentation',
            url: 'https://docs.api.nvidia.com/nim/reference/minimaxai-minimax-m3-1-flash',
            note: 'QAT NVFP4 routed experts and Q8KV4 attention/indexer paths. Four-bit payload counts alone exclude quantization scales and indexer state.',
        },
        {
            id: 'h100',
            title: 'NVIDIA H100 datasheet',
            url: 'https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306',
            note: 'H100 SXM: 3,958 TFLOPS sparse FP8 means 1,979 dense; 80 GB HBM. Advertised bidirectional PCIe bandwidth is not one-way storage throughput.',
        },
        {
            id: 'b200',
            title: 'NVIDIA Blackwell datasheet',
            url: 'https://dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf',
            note: 'HGX B200: 9 PFLOPS sparse FP8 means 4.5 dense; 180 GB advertised HBM capacity. This is not the 192 GB announcement figure.',
        },
        {
            id: 'rubin',
            title: 'NVIDIA DGX Rubin NVL8 datasheet (September 2026)',
            url: 'https://resources.nvidia.com/en-us-dgx-software/dgx-rubin-nvl8-datasheet?xs=489745',
            note: 'Preliminary 140 PFLOPS dense FP8 per eight-GPU system, or 17.5 PFLOPS per GPU; 288 GB per GPU. These specifications are subject to change.',
        },
        {
            id: 'ssd-gen6',
            title: 'Micron 9650 NVMe SSD product brief',
            url: 'https://www.micron.com/content/dam/micron/global/public/products/storage/ssds/data-center/9650/9650-nvme-ssd-product-brief.pdf',
            note: 'Up to 28 GB/s sequential reads at QD32, 14 GB/s writes and typical 60 microsecond read latency. These are device specifications, not measured end-to-end KV restore.',
        },
        {
            id: 'ssd-generations',
            title: 'Micron 9650 production announcement and Gen5 comparison',
            url: 'https://www.micron.com/about/blog/storage/ssd/micron-9650-ssd-the-worlds-first-pcie-gen6-ssd-reaches-a-new-shipping-milestone',
            note: 'Compares 28 GB/s Gen6 reads with 14 GB/s Gen5. Aggregating four drives needs sufficient topology, concurrency, and destination bandwidth.',
        },
        {
            id: 'gqa',
            title: 'GQA: Training Generalized Multi-Query Transformer Models (2023)',
            url: 'https://aclanthology.org/2023.emnlp-main.298/',
            note: 'Grouped KV heads reduce stored state while queries still use all attention heads; byte reduction is not an equal reduction in prefill arithmetic.',
        },
        {
            id: 'mla',
            title: 'DeepSeek-V2 technical report (2024)',
            url: 'https://arxiv.org/abs/2405.04434',
            note: 'MLA compresses KV to latent vectors; the paper reports 93.3% less KV than DeepSeek 67B. This is a model-specific comparison, not an annual industry trend.',
        },
        {
            id: 'kimi-linear',
            title: 'Kimi Linear: An Expressive, Efficient Attention Architecture (2025)',
            url: 'https://arxiv.org/abs/2510.26692',
            note: 'KDA/MLA hybrid: up to 75% less KV and up to 6x decode throughput at 1M context in its experiments. Decode speedup is not prefill speedup.',
        },
        {
            id: 'dsa',
            title: 'DeepSeek-V3.2 technical report (2025)',
            url: 'https://arxiv.org/abs/2512.02556',
            note: 'Sparse attention lowers attention work but does not by itself delete all historical KV needed for future selections.',
        },
        {
            id: 'indexcache',
            title: 'IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse (2026)',
            url: 'https://arxiv.org/abs/2603.12201',
            note: 'The indexer remains quadratic in sequence length; sharing its selections reduces repeated indexing. Reported speedups belong to the tested models.',
        },
    ]

    const models = [
        {
            id: 'dsv41',
            name: 'DeepSeek V4.1 Flash',
            bytesPerToken: 890,
            fixedBytes: 0,
            activeParams: 8e9,
            quadraticCoeff: 3 * 2048,
            linearAttentionFlops: 18 * 4 * 64 * 512 * 512 + 20 * 4 * 64 * 512 * 128,
            weightsBytes: 552e9,
            contextLimit: 1000000,
            baselinePrefillMs: 26,
            baselineTokens: 80000,
            baselineNode: 'rubin',
            baselineStatus: 'User-supplied scenario; no measured Rubin prefill calibration',
            storageStatus: 'Reported global cache footprint; approximate SWA replay required',
            notes: '890 B/token excludes persisted SWA: restore must add approximate bounded replay. FLOP sensitivity counts bulk encoder attention only; decoder replay, selection, communication, Engram transfer and kernel overhead remain unmodeled. Weight occupancy assumes FP8 backbone and host-resident Engram; scales, workspace, vision and draft weights are extra.',
            source: ['https://arxiv.org/html/2609.19969v1', 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash'],
        },
        {
            id: 'glm53f',
            name: 'GLM 5.3 Flash',
            bytesPerToken: 11 * 512,
            fixedBytes: 34 * 64 * 128 * 128 * 4 + 34 * 3 * 64 * 128 * 4 * 2,
            activeParams: 18e9,
            quadraticCoeff: (11 * 32 * 128) / 4,
            linearAttentionFlops: 11 * 2 * 64 * 2048 * (256 + 256),
            weightsBytes: 320e9,
            contextLimit: 1048576,
            storageStatus: 'Illustrative FP8 latent + FP32 KDA state, incomplete serialization',
            notes: "Main KV uses 11 x 512 FP8 values/token. Checkpoint scenario uses 34 x 64 x 128 x 128 FP32 recurrent values plus BF16 Q/K/V convolution history of length four. Corrects the draft's 32-head state. Indexer keys, scales, alignment and engine-specific state are extra; fixed state is rewritten on every appended turn. FLOPs omit KDA update and selection overhead.",
            source: [
                'https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json',
                'https://huggingface.co/zai-org/GLM-5.3-Flash',
            ],
        },
        {
            id: 'hy4',
            name: 'Hy4 preview',
            bytesPerToken: 78 * (512 + 64 * 2),
            fixedBytes: 0,
            activeParams: 49e9,
            quadraticCoeff: 21 * 32 * 128,
            linearAttentionFlops: 78 * 2 * 64 * 2048 * (256 + 256),
            weightsBytes: 770e9,
            contextLimit: 1048576,
            storageStatus: 'Illustrative FP8 latent + BF16 RoPE; incomplete serialization',
            notes: '78 x (512 FP8 latent + 64 BF16 RoPE) values/token, before indexer keys and quantization scales. The configuration has 21 full indexers, not 78/4. Main attention FLOPs use published 256-dimensional QK and V heads. Weight occupancy assumes the separate FP8 variant; the 770B backbone does not fit on eight H100s at one byte/parameter.',
            source: [
                'https://huggingface.co/tencent/Hy4-preview/raw/main/config.json',
                'https://huggingface.co/tencent/Hy4-preview',
            ],
        },
        {
            id: 'mmx31',
            name: 'MiniMax M3.1 Flash',
            bytesPerToken: 60 * 2 * 4 * 128 * 0.5,
            fixedBytes: 0,
            activeParams: 23e9,
            quadraticCoeff: 60 * 4 * 128,
            linearAttentionFlops: 60 * 2 * 64 * 2048 * (128 + 128),
            weightsBytes: 252e9,
            contextLimit: 1000000,
            storageStatus: 'Four-bit main KV payload lower bound; incomplete serialization',
            notes: "Main KV payload only: 60 layers x K/V x four KV heads x 128 dimensions x 0.5 B. NVFP4 scales and sparse-indexer cache increase bytes. Corrects the draft's 32 indexer heads to four; pooling and block selection make the FLOP estimate approximate. 252 GB resident weights is an editable mixed-precision scenario, not 252B model parameters (the card reports 428B).",
            source: [
                'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/config.json',
                'https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash',
                'https://docs.api.nvidia.com/nim/reference/minimaxai-minimax-m3-1-flash',
            ],
        },
    ]

    const nodes = [
        {
            id: 'h100',
            name: '8 x H100 SXM',
            year: 2023,
            denseTFLOPS: 1979,
            hbmGB: 80,
            gpuPerHour: 2.5,
            driveGBps: 14,
            linkGBps: 64,
            status: 'Published dense FP8 peak; rental price and shared I/O cap are scenarios',
            source: 'https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306',
        },
        {
            id: 'b200',
            name: '8 x B200',
            year: 2025,
            denseTFLOPS: 4500,
            hbmGB: 180,
            gpuPerHour: 5.5,
            driveGBps: 14,
            linkGBps: 64,
            status: 'Published dense FP8 peak; rental price and shared I/O cap are scenarios',
            source: 'https://dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf',
        },
        {
            id: 'rubin',
            name: '8 x Rubin',
            year: 2026,
            denseTFLOPS: 17500,
            hbmGB: 288,
            gpuPerHour: 8,
            driveGBps: 28,
            linkGBps: 128,
            status: 'Preliminary vendor specification; rental price and shared I/O cap are scenarios',
            source: 'https://resources.nvidia.com/en-us-dgx-software/dgx-rubin-nvl8-datasheet?xs=489745',
        },
    ]

    const history = [
        {
            epoch: '2023',
            name: 'Grouped-query attention',
            formula: 'B(T) = 2 x layers x KV heads x head dimension x bytes x T',
            meaning:
                'Fewer KV heads reduce bytes without an equal cut to query-attention arithmetic. This can favor restoration.',
            source: 'https://aclanthology.org/2023.emnlp-main.298/',
        },
        {
            epoch: '2024',
            name: 'Multi-head latent attention',
            formula: 'B(T) = layers x (latent bytes + RoPE bytes) x T',
            meaning:
                'MLA stores compressed latent state; projecting or dequantizing it adds work. Hardware-only projections miss the smaller transfer.',
            source: 'https://arxiv.org/abs/2405.04434',
        },
        {
            epoch: '2025',
            name: 'Linear-attention hybrids',
            formula: 'B(T) = attention-layer KV(T) + recurrent checkpoint',
            meaning:
                'KDA layers use finite recurrent state. Hybrid attention layers still grow with T; arbitrary-prefix reuse requires checkpoint coverage.',
            source: 'https://arxiv.org/abs/2510.26692',
        },
        {
            epoch: '2025',
            name: 'Sparse attention',
            formula: 'F(T) includes O(T x selected tokens) attention plus indexer work',
            meaning:
                'Selecting fewer tokens reduces compute, while future queries can still require the full historical cache. This can favor recomputation.',
            source: 'https://arxiv.org/abs/2512.02556',
        },
        {
            epoch: 'March 2026',
            name: 'Cross-layer index reuse',
            formula: 'Quadratic indexing cost scales with retained full indexer layers',
            meaning:
                'IndexCache reduces repeated selection work; sparse attention alone does not remove quadratic indexing.',
            source: 'https://arxiv.org/abs/2603.12201',
        },
        {
            epoch: 'September 2026',
            name: 'CSA2, FP4 KV and bounded replay',
            formula: 'Global B(T) = 890 x T; cold resume = transfer + bounded replay',
            meaning:
                'V4.1 Flash reduces both bulk prefill work and persisted bytes. Approximate SWA reconstruction creates a restore compute floor; neither side follows FLOPS alone.',
            source: 'https://arxiv.org/html/2609.19969v1',
        },
    ]

    const evidence = [
        'All costs, MFU, effective bandwidth and future growth rates are editable scenarios, not provider quotes or a forecast.',
        'Compare equal precision and dense compute. FP4 or structured-sparse marketing peaks cannot replace dense FP8 in a fixed-work projection.',
        'End-to-end restore includes lookup, scheduling, I/O, transfer, deserialization and any replay. Device bandwidth alone is an optimistic limit.',
        'Sparse attention, KV compression and fewer attention layers change FLOPs and bytes independently. Historical mechanisms do not establish a universal annual rate.',
        'A fast restore still needs a compatible model version, token prefix, precision/layout and checkpoint. New hardware does not make an incompatible cache reusable.',
        'A conditional latency crossover says nothing about whether storage capacity, expected return, or bandwidth contention makes a cache worth keeping.',
    ]

    return { asOf: '2026-11-17', models, nodes, history, evidence, sources }
})
