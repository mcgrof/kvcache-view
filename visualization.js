// KV Cache Growth Visualization
// Memory calculations based on LMCache KV Cache Calculator
// Source: https://lmcache.ai/kv_cache_calculator.html
// Project: https://github.com/mcgrof/kvcache-view
// This visualization uses the exact formulas from LMCache's calculator
// to accurately compute KV cache memory requirements

const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')

// Check if mobile
const isMobile = window.matchMedia('(max-width: 768px)').matches

// Model configurations from LMCache
// Using distinct colors for each model
const models = [
    {
        name: 'Llama-3.2-1B',
        params: 1.2,
        layers: 16,
        hidden_size: 2048,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#5FA3E6', // Light Blue
        efficiency: 'high',
    },
    {
        name: 'Phi-3.5-mini',
        params: 3.8,
        layers: 32,
        hidden_size: 3072,
        num_heads: 32,
        num_kv_heads: 32,
        color: '#00C853', // Green
        efficiency: 'high',
    },
    {
        name: 'Llama-3.1-8B',
        params: 8,
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#1428A0', // Deep Blue
        efficiency: 'medium',
    },
    {
        name: 'Gemma-2-9B',
        params: 9,
        layers: 42,
        hidden_size: 3584,
        num_heads: 16,
        num_kv_heads: 8,
        color: '#4285F4', // Google Blue
        efficiency: 'high',
    },
    {
        name: 'Qwen2.5-14B',
        params: 14,
        layers: 48,
        hidden_size: 5120,
        num_heads: 40,
        num_kv_heads: 8,
        color: '#FF9800', // Orange
        efficiency: 'medium',
    },
    {
        name: 'Phi-3.5-MoE',
        params: 41.9, // 16 experts, 2 active
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_local_experts: 16,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#00BCD4', // Cyan
        efficiency: 'high',
    },
    {
        name: 'Gemma-2-27B',
        params: 27,
        layers: 46,
        hidden_size: 4608,
        num_heads: 32,
        num_kv_heads: 16,
        color: '#34A853', // Google Green
        efficiency: 'medium',
    },
    {
        name: 'Qwen2.5-32B',
        params: 32,
        layers: 64,
        hidden_size: 5120,
        num_heads: 40,
        num_kv_heads: 8,
        color: '#FFC107', // Amber
        efficiency: 'medium',
    },
    {
        name: 'Mixtral-8x7B',
        params: 46.7, // 8 experts, 2 active = 12.9B active
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#9C27B0', // Deep Purple
        efficiency: 'high',
    },
    {
        name: 'Llama-3.1-70B',
        params: 70,
        layers: 80,
        hidden_size: 8192,
        num_heads: 64,
        num_kv_heads: 8,
        color: '#691FFF', // Purple
        efficiency: 'low',
    },
    {
        name: 'Qwen2.5-72B',
        params: 72,
        layers: 80,
        hidden_size: 8192,
        num_heads: 64,
        num_kv_heads: 8,
        color: '#FF5722', // Deep Orange
        efficiency: 'low',
    },
    {
        name: 'Qwen3-Next-80B',
        params: 80,
        active_params: 3,
        layers: 48,
        hidden_size: 2048,
        num_heads: 16,
        num_kv_heads: 2,
        num_local_experts: 512,
        num_experts_per_tok: 11,
        architecture: 'qwen3-next',
        color: '#795548', // Brown
        efficiency: 'optimized',
    },
    {
        name: 'Mixtral-8x22B',
        params: 141, // 8 experts, 2 active = 39.1B active
        layers: 56,
        hidden_size: 6144,
        num_heads: 48,
        num_kv_heads: 8,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#7B1FA2', // Dark Purple
        efficiency: 'medium',
    },
    {
        name: 'Llama-3.1-405B',
        params: 405,
        layers: 126,
        hidden_size: 16384,
        num_heads: 128,
        num_kv_heads: 8,
        color: '#E4002B', // Red
        efficiency: 'very-low',
    },
    {
        name: 'DeepSeek-V3 (671B)',
        params: 671,
        layers: 61,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        color: '#FF6B00', // Orange
        efficiency: 'optimized',
        special: 'deepseek',
    },
]

let currentModelIndex = 0
let currentTokens = 0
let maxTokens = 128000 // 128K context default (industry standard)
let animationSpeed = 50
let isPlaying = false // start paused so first click plays
let particles = []
let memoryBlocks = []
let waves = []
let currentDtype = 'FP16'
let currentFactoidIndex = 0
let lastFactoidUpdate = 0
let lastCriticalState = 'none'
let lastPopupTime = 0
const POPUP_COOLDOWN_MS = 10000
let includeWeights = true // Include model weights memory by default
let batchSize = 8 // Number of concurrent queries per GPU (modern production default)
let dataFlowParticles = [] // Particles flowing between HBM and GPU
let continuousBatching = false // Enable continuous batching with variable sequence lengths
let batchSequenceLengths = [] // Array of sequence lengths for each request in batch
let pagedAttention = false // Enable paged attention for memory fragmentation visualization
let flashAttention = false // Enable Flash Attention for tiled computation and reduced bandwidth
let sequenceColors = [] // Colors for each sequence in continuous batching
// GPU configurations (per-GPU memory in GiB)
const gpuConfigs = {
    // NVIDIA - NVLink support and PCIe generations
    'Tesla T4 16G': { memGiB: 16, label: 'Tesla T4 16G', memType: 'GDDR6', l2Cache: 6, flashTileSize: 32, nvlink: false, pcieGen: 3 },
    'RTX 4090 24G': { memGiB: 24, label: 'RTX 4090 24G', memType: 'GDDR6X', l2Cache: 72, flashTileSize: 64, nvlink: false, pcieGen: 4 },
    'L40S 48G': { memGiB: 48, label: 'L40S 48G', memType: 'GDDR6', l2Cache: 96, flashTileSize: 64, nvlink: false, pcieGen: 4 },
    'A100 40G': { memGiB: 40, label: 'A100 40G', memType: 'HBM2e', l2Cache: 40, flashTileSize: 128, nvlink: true, nvlinkBW: 600, pcieGen: 4 },
    'A100 80G': { memGiB: 80, label: 'A100 80G', memType: 'HBM2e', l2Cache: 40, flashTileSize: 128, nvlink: true, nvlinkBW: 600, pcieGen: 4 },
    'H100 80G': { memGiB: 80, label: 'H100 80G', memType: 'HBM3', l2Cache: 50, flashTileSize: 128, nvlink: true, nvlinkBW: 900, pcieGen: 5 },
    'H200 141G': { memGiB: 141, label: 'H200 141G', memType: 'HBM3e', l2Cache: 50, flashTileSize: 128, nvlink: true, nvlinkBW: 900, pcieGen: 5 },
    // AMD Radeon Pro (workstation) - No Infinity Fabric Link on workstation cards
    'AMD W7800 32G': { memGiB: 32, label: 'AMD W7800 32G', memType: 'GDDR6', l2Cache: 64, flashTileSize: 64, nvlink: false, pcieGen: 4 },
    'AMD W7900 48G': { memGiB: 48, label: 'AMD W7900 48G', memType: 'GDDR6', l2Cache: 96, flashTileSize: 64, nvlink: false, pcieGen: 4 },
    // AMD Instinct (data center) - Infinity Fabric Link support
    'AMD MI210 64G': { memGiB: 64, label: 'AMD MI210 64G', memType: 'HBM2e', l2Cache: 32, flashTileSize: 64, nvlink: true, nvlinkBW: 300, ifl: true, pcieGen: 4 },
    'AMD MI250X 128G': { memGiB: 128, label: 'AMD MI250X 128G', memType: 'HBM2e', l2Cache: 32, flashTileSize: 64, nvlink: true, nvlinkBW: 400, ifl: true, pcieGen: 4 },
    'AMD MI300X 192G': { memGiB: 192, label: 'AMD MI300X 192G', memType: 'HBM3', l2Cache: 256, flashTileSize: 128, nvlink: true, nvlinkBW: 896, ifl: true, pcieGen: 5 },
    // Intel (GPU + AI accelerators)
    'Intel Arc A770 16G': { memGiB: 16, label: 'Intel Arc A770 16G', memType: 'GDDR6', l2Cache: 16, flashTileSize: 32, nvlink: false, pcieGen: 4 },
    'Intel Max 1550 128G': {
        memGiB: 128,
        label: 'Intel Max 1550 128G',
        memType: 'HBM2e',
        l2Cache: 408,
        flashTileSize: 128,
        nvlink: false,
        pcieGen: 4
    },
    'Intel Gaudi2 96G': { memGiB: 96, label: 'Intel Gaudi2 96G', memType: 'HBM2e', l2Cache: 48, flashTileSize: 64, nvlink: false, pcieGen: 4 },
    // Google TPU (approx per-chip HBM) - Has proprietary interconnect, PCIe for host connection
    'Google TPU v3 16G': { memGiB: 16, label: 'Google TPU v3 16G', memType: 'HBM2', l2Cache: 16, flashTileSize: 64, nvlink: true, nvlinkBW: 700, tpuInterconnect: true, pcieGen: 3 },
    'Google TPU v4 32G': { memGiB: 32, label: 'Google TPU v4 32G', memType: 'HBM2', l2Cache: 32, flashTileSize: 128, nvlink: true, nvlinkBW: 1200, tpuInterconnect: true, pcieGen: 4 },
    // Graphcore and Cerebras - Proprietary fabrics
    'Graphcore IPU Mk2 0.9G': {
        memGiB: 0.9,
        label: 'Graphcore IPU Mk2 0.9G',
        memType: 'SRAM',
        l2Cache: 900,
        flashTileSize: 256,
        nvlink: true,
        nvlinkBW: 320,
        ipuLink: true,
        pcieGen: 4
    },
    'Cerebras WSE-2 40G': {
        memGiB: 40,
        label: 'Cerebras WSE-2 40G',
        memType: 'SRAM',
        l2Cache: 40960,
        flashTileSize: 512,
        nvlink: false,  // Single wafer, no multi-chip needed
        pcieGen: 4
    },
    // Qualcomm Cloud AI
    'Qualcomm Cloud AI 100 32G': {
        memGiB: 32,
        label: 'Qualcomm Cloud AI 100 32G',
        memType: 'LPDDR5',
        l2Cache: 32,
        flashTileSize: 64,
        nvlink: false,
        pcieGen: 4
    },
}

// Multi-GPU configuration
let gpuCount = 1  // Number of GPUs (powers of 2: 1, 2, 4, 8, 16, 32, 64, 128)
const validGPUCounts = [1, 2, 4, 8, 16, 32, 64, 128]
const pcie4Bandwidth = 32  // PCIe 4.0 x16 bandwidth in GB/s (most GPUs)
const pcie5Bandwidth = 64  // PCIe 5.0 x16 bandwidth in GB/s (H100, H200, MI300X)
let useHighSpeedInterconnect = true  // Use NVLink/IFL when available vs PCIe

let currentGPU = 'H100 80G'

// Famous GPU datacenter configurations
const worldDatacenters = {
    none: { name: 'None', gpus: null, gpu: null, model: null },
    dgx_h100: { name: 'DGX H100', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    dgx_pod: { name: 'DGX SuperPOD', gpus: 32, gpu: 'H100 80G', model: 'Llama-3.1-405B', interconnect: 'nvlink' },
    meta_rsc: { name: 'Meta RSC', gpus: 128, gpu: 'A100 80G', model: 'Llama-3.1-405B', interconnect: 'nvlink' },
    openai_cluster: { name: 'OpenAI GPT-4', gpus: 64, gpu: 'A100 40G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    aws_p4: { name: 'AWS P4d.24xl', gpus: 8, gpu: 'A100 40G', model: 'Llama-3.1-8B', interconnect: 'nvlink' },
    aws_p5: { name: 'AWS P5.48xl', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    gcp_a2: { name: 'GCP A2 Ultra', gpus: 16, gpu: 'A100 40G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    gcp_a3: { name: 'GCP A3 Mega', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    azure_nd96: { name: 'Azure NDv4', gpus: 8, gpu: 'A100 40G', model: 'Llama-3.1-8B', interconnect: 'nvlink' },
    azure_ndh100: { name: 'Azure NDm H100', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    lambda_1x: { name: 'Lambda 1-Click', gpus: 1, gpu: 'H100 80G', model: 'Llama-3.1-8B', interconnect: 'none' },
    lambda_8x: { name: 'Lambda Cloud 8x', gpus: 8, gpu: 'A100 80G', model: 'Llama-3.1-70B', interconnect: 'nvlink' },
    amd_mi300: { name: 'AMD MI300X', gpus: 8, gpu: 'AMD MI300X 192G', model: 'Llama-3.1-70B', interconnect: 'ifl' },
    intel_gaudi: { name: 'Intel Gaudi 2', gpus: 8, gpu: 'Intel Gaudi2 96G', model: 'Phi-3.5-mini', interconnect: 'pcie' },
    tesla_dojo: { name: 'Tesla Dojo', gpus: 64, gpu: 'A100 40G', model: 'Llama-3.1-70B', interconnect: 'custom' },
}
let currentDatacenter = 'none'

// Set sane defaults based on GPU capabilities
function setGPUDefaults(gpuKey) {
    const config = gpuConfigs[gpuKey]
    if (!config) return

    const memGiB = config.memGiB
    const memType = config.memType

    // Set appropriate context length based on GPU memory
    if (memGiB >= 128) {
        // High-end GPUs (MI300X, Max 1550): 1M context
        maxTokens = 1000000
        currentTokens = Math.min(currentTokens, maxTokens * 0.5) // Start at 50%
    } else if (memGiB >= 80) {
        // Premium GPUs (H100, A100 80G): 512K context
        maxTokens = 512000
        currentTokens = Math.min(currentTokens, maxTokens * 0.3) // Start at 30%
    } else if (memGiB >= 40) {
        // High-end GPUs (A100 40G, L40S): 256K context
        maxTokens = 256000
        currentTokens = Math.min(currentTokens, maxTokens * 0.25) // Start at 25%
    } else if (memGiB >= 24) {
        // Enthusiast GPUs (RTX 4090): 128K context
        maxTokens = 128000
        currentTokens = Math.min(currentTokens, maxTokens * 0.2) // Start at 20%
    } else if (memGiB >= 16) {
        // Mid-range GPUs (Tesla T4, Arc A770): 64K context
        maxTokens = 64000
        currentTokens = Math.min(currentTokens, maxTokens * 0.15) // Start at 15%
    } else {
        // Lower-end or specialized (Graphcore): 32K context
        maxTokens = 32000
        currentTokens = Math.min(currentTokens, maxTokens * 0.1) // Start at 10%
    }

    // Set appropriate batch size based on memory and type
    if (memType.includes('SRAM')) {
        // Specialized accelerators: smaller batches, optimized for throughput
        batchSize = memGiB > 10 ? 2 : 1
    } else if (memGiB >= 80) {
        // High-memory GPUs: larger batches
        batchSize = 8
    } else if (memGiB >= 40) {
        // Mid-high memory: moderate batches
        batchSize = 4
    } else if (memGiB >= 16) {
        // Mid-range: conservative batches
        batchSize = 2
    } else {
        // Low memory: single batch
        batchSize = 1
    }

    // Enable appropriate optimizations based on GPU capabilities
    if (memType.includes('SRAM')) {
        // SRAM-based accelerators: all optimizations enabled by default
        continuousBatching = true
        pagedAttention = true
        flashAttention = true
    } else if (memType.includes('HBM')) {
        // HBM-based GPUs: enable CB and Flash by default
        continuousBatching = true
        pagedAttention = false // Start with just CB + Flash
        flashAttention = true
    } else {
        // GDDR-based GPUs: more conservative defaults
        continuousBatching = memGiB >= 24 // Only for high-memory GDDR
        pagedAttention = false
        flashAttention = memGiB >= 16 // Flash for mid-range and up
    }

    // Reset animation to start position
    if (currentTokens === 0) {
        currentTokens = maxTokens * 0.01 // Start at 1% to show some progress
    }
}

function getCurrentGPUMemGiB() {
    const cfg = gpuConfigs[currentGPU]
    return cfg ? cfg.memGiB : 80
}

// Update UI controls to reflect current state
function updateControlStates() {
    // Update batch size control
    const batchBtn = document.getElementById('batchControl')
    if (batchBtn) {
        batchBtn.textContent = `Batch: ${batchSize}`
    }

    // Update context length control
    const contextBtn = document.getElementById('contextControl')
    if (contextBtn) {
        contextBtn.textContent = `Context: ${formatMemoryValue(maxTokens)}`
    }

    // Update optimization toggle states
    const cbBtn = document.getElementById('cbToggle')
    if (cbBtn) {
        const span = cbBtn.querySelector('span')
        if (span) {
            span.textContent = `CB: ${continuousBatching ? 'ON' : 'OFF'}`
        }
        cbBtn.classList.toggle('enabled', continuousBatching)
    }

    const paBtn = document.getElementById('paToggle')
    if (paBtn) {
        const span = paBtn.querySelector('span')
        if (span) {
            span.textContent = `PA: ${pagedAttention ? 'ON' : 'OFF'}`
        }
        paBtn.classList.toggle('enabled', pagedAttention)
    }

    const faBtn = document.getElementById('faToggle')
    if (faBtn) {
        const span = faBtn.querySelector('span')
        if (span) {
            span.textContent = `FA: ${flashAttention ? 'ON' : 'OFF'}`
        }
        faBtn.classList.toggle('enabled', flashAttention)
    }
}

// Format large numbers for display
function formatMemoryValue(value) {
    if (value >= 1000000) {
        return (value / 1000000).toFixed(value % 1000000 === 0 ? 0 : 1) + 'M'
    } else if (value >= 1000) {
        return (value / 1000).toFixed(value % 1000 === 0 ? 0 : 1) + 'K'
    }
    return value.toString()
}

// SOTA context length presets
const contextPresets = {
    '4K': 4096, // Original GPT-3.5, older models
    '8K': 8192, // GPT-4 base
    '16K': 16384, // GPT-3.5 Turbo 16K
    '32K': 32768, // GPT-4 32K, Claude Instant
    '64K': 64000, // Current standard context
    '100K': 100000, // Claude 2.1
    '128K': 128000, // GPT-4 Turbo, GPT-4o, Llama 3
    '200K': 200000, // Claude 3 Opus/Sonnet/Haiku
    '256K': 256000, // High-end production context
    '512K': 512000, // H100/A100 80GB default
    '1M': 1000000, // Gemini 1.5 Pro (public), Claude 3.5 Sonnet
    '2M': 2000000, // Gemini 1.5 Pro (developer preview)
    '10M': 10000000, // Research frontier (Magic, experimental)
}

// Data type configurations
const dtypeConfigs = {
    FP32: { bytes: 4, name: 'float32', color: '#ff6b6b' },
    FP16: { bytes: 2, name: 'float16', color: '#00d4ff' },
    BF16: { bytes: 2, name: 'bfloat16', color: '#00ff88' },
    INT8: { bytes: 1, name: 'int8', color: '#ffaa00' },
    INT4: { bytes: 0.5, name: 'int4', color: '#ff00ff' },
}

// Resize canvas
function resizeCanvas() {
    if (isMobile) {
        // On mobile, account for header and controls
        canvas.width = window.innerWidth
        canvas.height = window.innerHeight - 250 // Leave room for header and controls
    } else {
        canvas.width = window.innerWidth
        canvas.height = window.innerHeight
    }
}

// Calculate KV cache size (from LMCache logic)
function calculateKVCacheSize(model, tokens, dtype = null) {
    const selectedDtype = dtype || currentDtype
    const dtype_size = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2
    let total_elements

    if (model.special === 'deepseek') {
        // DeepSeek uses KV-LoRA compression
        total_elements = model.layers * tokens * (model.kv_lora_rank + model.qk_rope_head_dim)
    } else if (model.architecture === 'qwen3-next') {
        // Qwen3-Next uses hybrid attention (1/4 layers use traditional attention, rest use linear)
        // Only 1/4 of layers have KV cache
        const head_size = model.hidden_size / model.num_heads
        const layers_with_kv = Math.floor(model.layers / 4)
        total_elements = 2 * layers_with_kv * tokens * model.num_kv_heads * head_size
    } else {
        // Standard calculation (includes MOE models - expert count doesn't affect KV cache)
        const head_size = model.hidden_size / model.num_heads
        total_elements = 2 * model.layers * tokens * model.num_kv_heads * head_size
    }

    const total_bytes = total_elements * dtype_size
    return total_bytes / (1024 * 1024 * 1024) // Convert to GiB
}

// Generate distinct colors for each sequence in continuous batching
function generateSequenceColors(batchSize) {
    const colors = [
        '#FFD700', // Gold - prompt tokens
        '#00CED1', // Dark Turquoise - sequence 2
        '#FF69B4', // Hot Pink - sequence 3
        '#32CD32', // Lime Green - sequence 4
        '#FF6347', // Tomato - sequence 5
        '#BA55D3', // Medium Orchid - sequence 6
        '#4169E1', // Royal Blue - sequence 7
        '#FF8C00', // Dark Orange - sequence 8
        '#20B2AA', // Light Sea Green - sequence 9
        '#DC143C', // Crimson - sequence 10
        '#9370DB', // Medium Purple - sequence 11
        '#00FA9A', // Medium Spring Green - sequence 12
        '#FFA500', // Orange - sequence 13
        '#87CEEB', // Sky Blue - sequence 14
        '#FF1493', // Deep Pink - sequence 15
        '#ADFF2F', // Green Yellow - sequence 16
    ]

    const result = []
    for (let i = 0; i < batchSize; i++) {
        result.push(colors[i % colors.length])
    }
    return result
}

// Generate deterministic sequence length ratios for continuous batching
function getSequenceLengthRatio(index, batchSize) {
    // For visual clarity, use more balanced ratios that ensure each sequence is visible
    if (batchSize <= 4) {
        // Small batch sizes: use well-distributed ratios for clear visualization
        const smallBatchPatterns = {
            2: [0.3, 0.7], // 30% vs 70%
            3: [0.2, 0.4, 0.6], // 20%, 40%, 60%
            4: [0.15, 0.25, 0.35, 0.5], // 15%, 25%, 35%, 50%
        }
        const pattern = smallBatchPatterns[batchSize]
        return pattern[index % pattern.length]
    }

    // For larger batch sizes, use a more varied but still visible pattern
    const pattern = [0.1, 0.2, 0.3, 0.4, 0.15, 0.45, 0.25, 0.55, 0.35, 0.65, 0.05, 0.5, 0.18, 0.7, 0.12, 0.85]
    return pattern[index % pattern.length]
}

// Calculate attention matrix memory that would be needed without Flash Attention
function calculateAttentionMatrixSize(sequenceLength, batchSize = 1, dtype = null) {
    const config = dtypeConfigs[dtype || currentDtype]
    const bytesPerElement = config.bytes
    const model = models[currentModelIndex]
    const currentGPUConfig = gpuConfigs[currentGPU]
    const totalGPUMemoryGiB = getCurrentGPUMemGiB()

    // Flash Attention saves intermediate attention computation memory, not the full O(n²) matrix
    // Traditional attention still uses chunking for very long sequences, so baseline isn't full matrix

    // Calculate realistic working memory for attention computation
    // This represents intermediate attention scores that would be held during computation
    const numHeads = model.heads || 32

    // For shorter sequences (<32k), traditional attention might materialize more of the matrix
    // For longer sequences, even traditional implementations use chunking
    const effectiveSequenceLength = Math.min(sequenceLength, 65536) // Cap at 64k for realistic baseline

    // Chunk size that traditional attention would process at once (realistic: 1k-4k tokens)
    const chunkSize = Math.min(4096, effectiveSequenceLength)

    // Memory for attention scores during computation: chunk_size × seq_len × heads × batches
    const workingMemoryBytes = chunkSize * effectiveSequenceLength * numHeads * batchSize * bytesPerElement

    // Flash Attention reduces this by computing in smaller tiles (typically 64×64 or 128×128)
    const flashTileSize = currentGPUConfig ? currentGPUConfig.flashTileSize : 64
    const traditionalChunkBytes = workingMemoryBytes
    const flashTileBytes = flashTileSize * flashTileSize * numHeads * batchSize * bytesPerElement

    // Savings = traditional working memory - flash tile memory
    const savingsBytes = Math.max(0, traditionalChunkBytes - flashTileBytes)
    const savingsGiB = savingsBytes / 1024 ** 3

    // Cap savings at 50% of total GPU memory (can't save more than what's physically possible)
    const maxSavingsGiB = totalGPUMemoryGiB * 0.5
    const cappedSavingsGiB = Math.min(savingsGiB, maxSavingsGiB)

    // For very short sequences, savings are minimal
    if (sequenceLength < 2048) {
        return cappedSavingsGiB * 0.1 // Minimal savings for short sequences
    }

    return cappedSavingsGiB
}

// Calculate total KV cache for batch with continuous batching support
function calculateBatchKVCache(model, currentTokens, dtype = null) {
    // Continuous batching only makes sense with batch size > 1
    // With batch size = 1, there's no difference between continuous and traditional
    if (continuousBatching && batchSize > 1) {
        // Calculate based on variable sequence lengths with deterministic ratios
        let totalKV = 0
        let totalSeqLength = 0

        for (let i = 0; i < batchSize; i++) {
            // Use deterministic ratio for consistent calculations
            const ratio = getSequenceLengthRatio(i, batchSize)
            const seqLen = Math.floor(currentTokens * ratio)
            totalSeqLength += seqLen
            totalKV += calculateKVCacheSize(model, seqLen, dtype)
        }

        // Store average for display purposes
        if (!batchSequenceLengths.length || batchSequenceLengths.length !== batchSize) {
            batchSequenceLengths = []
            for (let i = 0; i < batchSize; i++) {
                const ratio = getSequenceLengthRatio(i, batchSize)
                batchSequenceLengths.push(Math.floor(currentTokens * ratio))
            }
        }

        return totalKV
    } else {
        // Traditional batching: all sequences same length
        // Also used when batch size = 1 (continuous batching doesn't apply)
        batchSequenceLengths = [] // Clear any stored lengths
        return calculateKVCacheSize(model, currentTokens, dtype) * batchSize
    }
}

// Calculate model weights memory in GiB
function calculateWeightMemoryGiB(model, dtype = null) {
    const selectedDtype = dtype || currentDtype
    const bytesPerParam = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2
    // For MOE models with active_params specified, use active params for inference memory
    // Otherwise use total params
    const paramsToUse = model.active_params || model.params || 0
    // params is in Billions (e.g., 70 for 70B). Convert to number of parameters
    const numParams = paramsToUse * 1e9
    const totalBytes = numParams * bytesPerParam
    return totalBytes / (1024 * 1024 * 1024) // GiB
}

// Calculate GPUs needed (H100 has 80GB memory)
function calculateGPUsNeeded(memoryGiB) {
    const per = getCurrentGPUMemGiB()
    return Math.ceil(memoryGiB / per)
}

// Format memory size
function formatMemory(gib) {
    if (gib < 1) {
        return `${(gib * 1024).toFixed(1)} MiB`
    } else if (gib < 1000) {
        return `${gib.toFixed(1)} GiB`
    } else {
        return `${(gib / 1024).toFixed(2)} TiB`
    }
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',')
}

// Create memory block particle
class MemoryBlock {
    constructor(x, y, size, color) {
        this.x = x
        this.y = y
        this.size = size
        this.color = color
        this.opacity = 1
        this.velocity = {
            x: (Math.random() - 0.5) * 2,
            y: -Math.random() * 3 - 1,
        }
        this.life = 1
        this.rotation = Math.random() * Math.PI * 2
        this.rotationSpeed = (Math.random() - 0.5) * 0.1
    }

    update() {
        this.x += this.velocity.x
        this.y += this.velocity.y
        this.velocity.y += 0.05 // gravity
        this.life -= 0.01
        this.opacity = this.life
        this.rotation += this.rotationSpeed
    }

    draw() {
        ctx.save()
        ctx.globalAlpha = this.opacity
        ctx.translate(this.x, this.y)
        ctx.rotate(this.rotation)

        // Draw glowing block
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, this.size)
        gradient.addColorStop(0, this.color)
        gradient.addColorStop(1, 'transparent')
        ctx.fillStyle = gradient
        ctx.fillRect(-this.size / 2, -this.size / 2, this.size, this.size)

        // Draw border
        ctx.strokeStyle = this.color
        ctx.lineWidth = 2
        ctx.strokeRect(-this.size / 2, -this.size / 2, this.size, this.size)

        ctx.restore()
    }
}

// Data flow particle class
class DataFlowParticle {
    constructor(startX, startY, endX, endY, color, speed = 0.02) {
        this.startX = startX
        this.startY = startY
        this.endX = endX
        this.endY = endY
        this.x = startX
        this.y = startY
        this.progress = 0
        this.speed = speed
        this.color = color
        this.size = 3 + Math.random() * 3
        this.life = 1
        this.trail = []
        this.maxTrailLength = 10
    }

    update() {
        this.progress += this.speed

        // Move along path
        this.x = this.startX + (this.endX - this.startX) * this.progress
        this.y = this.startY + (this.endY - this.startY) * this.progress

        // Add to trail
        this.trail.push({ x: this.x, y: this.y })
        if (this.trail.length > this.maxTrailLength) {
            this.trail.shift()
        }

        // Check if reached destination
        if (this.progress >= 1) {
            this.life = 0
        }
    }

    draw() {
        // Draw trail
        ctx.strokeStyle = this.color + '33'
        ctx.lineWidth = this.size * 0.5
        ctx.beginPath()
        this.trail.forEach((point, i) => {
            if (i === 0) {
                ctx.moveTo(point.x, point.y)
            } else {
                ctx.lineTo(point.x, point.y)
            }
        })
        ctx.stroke()

        // Draw particle
        ctx.globalAlpha = 0.8
        const glow = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.size * 2)
        glow.addColorStop(0, this.color)
        glow.addColorStop(1, 'transparent')
        ctx.fillStyle = glow
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.size * 2, 0, Math.PI * 2)
        ctx.fill()

        ctx.fillStyle = this.color
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2)
        ctx.fill()
        ctx.globalAlpha = 1
    }
}

// Create wave effect
class Wave {
    constructor(y, amplitude, frequency, speed, color) {
        this.y = y
        this.amplitude = amplitude
        this.frequency = frequency
        this.speed = speed
        this.color = color
        this.phase = 0
    }

    update() {
        this.phase += this.speed
    }

    draw() {
        ctx.strokeStyle = this.color
        ctx.lineWidth = 3
        ctx.globalAlpha = 0.6
        ctx.beginPath()

        for (let x = 0; x < canvas.width; x += 5) {
            const y = this.y + Math.sin(x * this.frequency + this.phase) * this.amplitude
            if (x === 0) {
                ctx.moveTo(x, y)
            } else {
                ctx.lineTo(x, y)
            }
        }

        ctx.stroke()
        ctx.globalAlpha = 1
    }
}

// Initialize waves
function initWaves() {
    waves = []
    const model = models[currentModelIndex]

    for (let i = 0; i < 5; i++) {
        waves.push(
            new Wave(
                canvas.height / 2 + (i - 2) * 50,
                20 + i * 5,
                0.01 + i * 0.002,
                0.02 * animationSpeed,
                model.color + '33',
            ),
        )
    }
}

// Draw GPU architecture visualization
// Store interconnect particles for animation
let interconnectParticles = []

// Multi-GPU cluster visualization with enhanced interconnect visualization
function drawMultiGPUCluster() {
    const model = models[currentModelIndex]
    const gpuConfig = gpuConfigs[currentGPU]

    // Calculate memory per GPU
    const kvGiB = calculateBatchKVCache(model, currentTokens)
    const kvMaxGiB = calculateBatchKVCache(model, maxTokens)
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0
    const memPerGPU = (kvGiB + weightsGiB) / gpuCount  // KV cache is sharded across GPUs
    const memMaxPerGPU = (kvMaxGiB + weightsGiB) / gpuCount

    // Determine grid layout based on GPU count
    let cols, rows
    if (gpuCount === 2) { cols = 2; rows = 1; }
    else if (gpuCount === 4) { cols = 2; rows = 2; }
    else if (gpuCount === 8) { cols = 4; rows = 2; }
    else if (gpuCount === 16) { cols = 4; rows = 4; }
    else if (gpuCount === 32) { cols = 8; rows = 4; }
    else if (gpuCount === 64) { cols = 8; rows = 8; }
    else if (gpuCount === 128) { cols = 16; rows = 8; }
    else { cols = Math.ceil(Math.sqrt(gpuCount)); rows = Math.ceil(gpuCount / cols); }

    // Scale GPU size based on count - much smaller for larger clusters
    let scaleFactor = 0.7
    if (gpuCount > 16) scaleFactor = 0.35  // Much smaller for 32 GPUs
    if (gpuCount > 32) scaleFactor = 0.25  // Even smaller for 64 GPUs
    if (gpuCount > 64) scaleFactor = 0.15  // Very tiny for >64 GPUs
    if (gpuCount >= 128) scaleFactor = 0.03  // Microscopic for 128 GPUs

    const maxGPUSize = Math.min(150, Math.min(canvas.width / (cols + 1), canvas.height / (rows + 1)))
    const gpuSize = maxGPUSize * scaleFactor
    const gpuSpacing = maxGPUSize * (gpuCount >= 128 ? 0.7 : (gpuCount > 16 ? 0.95 : 1.2))  // Very tight for 128

    // Center the grid
    const gridWidth = cols * gpuSpacing
    const gridHeight = rows * gpuSpacing
    const offsetX = (canvas.width - gridWidth) / 2 + gpuSpacing / 2
    const offsetY = (canvas.height - gridHeight) / 2 + gpuSpacing / 2

    // Store GPU positions for interconnect drawing
    const gpuPositions = []

    // Determine interconnect type and bandwidth
    let interconnectType = 'PCIe'
    let interconnectBW = pcie4Bandwidth  // Default to PCIe 4.0
    let interconnectColor = '#606060'  // Gray for PCIe

    // Check PCIe generation for this GPU
    if (!useHighSpeedInterconnect || !gpuConfig.nvlink) {
        if (gpuConfig.pcieGen === 5) {
            interconnectType = 'PCIe 5.0'
            interconnectBW = pcie5Bandwidth
        } else if (gpuConfig.pcieGen === 3) {
            interconnectType = 'PCIe 3.0'
            interconnectBW = 16  // PCIe 3.0 x16 = 16 GB/s
        } else {
            interconnectType = 'PCIe 4.0'
            interconnectBW = pcie4Bandwidth
        }
    }

    if (gpuConfig.nvlink && useHighSpeedInterconnect) {
        if (gpuConfig.ifl) {
            interconnectType = 'Infinity Fabric'
            interconnectBW = gpuConfig.nvlinkBW || 400
            interconnectColor = '#FF4444'  // Red for AMD
        } else if (gpuConfig.tpuInterconnect) {
            interconnectType = 'TPU Interconnect'
            interconnectBW = gpuConfig.nvlinkBW || 700
            interconnectColor = '#4285F4'  // Google Blue
        } else if (gpuConfig.ipuLink) {
            interconnectType = 'IPU-Link'
            interconnectBW = gpuConfig.nvlinkBW || 320
            interconnectColor = '#00A0FF'  // Light blue for Graphcore
        } else if (currentGPU.includes('Intel')) {
            interconnectType = 'Xe Link'
            interconnectBW = 400  // Intel Xe Link bandwidth
            interconnectColor = '#0071C5'  // Intel Blue
        } else {
            interconnectType = `NVLink ${gpuConfig.nvlinkBW >= 900 ? '4.0' : '3.0'}`
            interconnectBW = gpuConfig.nvlinkBW || 600
            interconnectColor = '#76B900'  // NVIDIA Green
        }
    }

    // Calculate bandwidth utilization - more realistic model
    // During training/inference, KV cache needs to be synchronized for attention
    // This creates all-to-all traffic patterns
    const kvCacheSyncTraffic = (kvGiB * 1024) * Math.log2(gpuCount)  // All-reduce traffic scales with log(n)
    const bandwidthUtilization = Math.min(1.0, kvCacheSyncTraffic / (interconnectBW * 1000))

    // Draw interconnects first (behind GPUs)
    ctx.save()

    // Create visual metaphor for bandwidth saturation
    if (bandwidthUtilization > 0.7) {
        // Draw pulsing red glow when approaching saturation
        const pulseIntensity = 0.3 + Math.sin(Date.now() / 200) * 0.2
        ctx.fillStyle = `rgba(255, 68, 68, ${pulseIntensity * bandwidthUtilization})`
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }

    // Draw mesh interconnects for high-speed links
    if (gpuConfig.nvlink && useHighSpeedInterconnect) {
        // Limit interconnect visualization for performance
        const connections = []

        if (gpuCount <= 8) {
            // Full mesh for small clusters
            for (let i = 0; i < gpuCount; i++) {
                const row1 = Math.floor(i / cols)
                const col1 = i % cols
                const x1 = offsetX + col1 * gpuSpacing
                const y1 = offsetY + row1 * gpuSpacing

                for (let j = i + 1; j < gpuCount; j++) {
                    const row2 = Math.floor(j / cols)
                    const col2 = j % cols
                    const x2 = offsetX + col2 * gpuSpacing
                    const y2 = offsetY + row2 * gpuSpacing
                    connections.push({x1, y1, x2, y2})
                }
            }
        } else if (gpuCount <= 32) {
            // Sparse connections for medium clusters
            for (let i = 0; i < gpuCount; i += 2) {  // Skip every other GPU
                const row1 = Math.floor(i / cols)
                const col1 = i % cols
                const x1 = offsetX + col1 * gpuSpacing
                const y1 = offsetY + row1 * gpuSpacing

                // Only connect to nearest neighbors
                for (let j = i + 1; j < Math.min(i + 4, gpuCount); j++) {
                    const row2 = Math.floor(j / cols)
                    const col2 = j % cols
                    const x2 = offsetX + col2 * gpuSpacing
                    const y2 = offsetY + row2 * gpuSpacing
                    connections.push({x1, y1, x2, y2})
                }
            }
        } else {
            // Minimal connections for huge clusters (>32 GPUs)
            // Just show a few representative connections to indicate fabric
            const step = Math.max(4, Math.floor(gpuCount / 16))
            for (let i = 0; i < gpuCount; i += step) {
                const row1 = Math.floor(i / cols)
                const col1 = i % cols
                const x1 = offsetX + col1 * gpuSpacing
                const y1 = offsetY + row1 * gpuSpacing

                // Connect to one neighbor only
                const j = Math.min(i + step, gpuCount - 1)
                if (j > i) {
                    const row2 = Math.floor(j / cols)
                    const col2 = j % cols
                    const x2 = offsetX + col2 * gpuSpacing
                    const y2 = offsetY + row2 * gpuSpacing
                    connections.push({x1, y1, x2, y2})
                }
            }
        }

        // Draw all connections with visual bandwidth representation
        connections.forEach(conn => {
            const {x1, y1, x2, y2} = conn

            // Base interconnect line - thicker when saturated
            const baseWidth = 2 + bandwidthUtilization * 8
            ctx.strokeStyle = interconnectColor
            ctx.globalAlpha = 0.2
            ctx.lineWidth = baseWidth
            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()

            // Draw "data pipes" that visually fill up with traffic
            if (bandwidthUtilization > 0) {
                // Inner flow showing actual data movement
                const flowWidth = baseWidth * bandwidthUtilization

                // Color changes from green to yellow to red as saturation increases
                let flowColor
                if (bandwidthUtilization < 0.5) {
                    flowColor = `rgba(76, 175, 80, ${0.6 + bandwidthUtilization})`
                } else if (bandwidthUtilization < 0.8) {
                    flowColor = `rgba(255, 193, 7, ${0.6 + bandwidthUtilization})`
                } else {
                    flowColor = `rgba(255, 68, 68, ${0.6 + bandwidthUtilization})`
                }

                ctx.strokeStyle = flowColor
                ctx.globalAlpha = 0.7
                ctx.lineWidth = flowWidth
                ctx.beginPath()
                ctx.moveTo(x1, y1)
                ctx.lineTo(x2, y2)
                ctx.stroke()

                // Add glowing effect for high utilization
                if (bandwidthUtilization > 0.5) {
                    ctx.shadowBlur = 10 * bandwidthUtilization
                    ctx.shadowColor = flowColor
                    ctx.stroke()
                    ctx.shadowBlur = 0
                }
            }

            // Animate data packets flowing through the pipes
            // Reduce particles for large clusters to save memory
            if (isPlaying && bandwidthUtilization > 0 && gpuCount <= 64) {
                // Limit particles based on GPU count
                const maxParticles = gpuCount > 32 ? 2 : (gpuCount > 16 ? 3 : 5)
                const particleCount = Math.min(maxParticles, Math.ceil(bandwidthUtilization * 5))

                for (let p = 0; p < particleCount; p++) {
                    const offset = (Date.now() / (1000 - bandwidthUtilization * 800) + p / particleCount) % 1
                    const particleX = x1 + (x2 - x1) * offset
                    const particleY = y1 + (y2 - y1) * offset

                    // Particle size and brightness based on traffic
                    const particleSize = 2 + bandwidthUtilization * 4
                    ctx.fillStyle = bandwidthUtilization > 0.8 ? '#FF4444' :
                                   bandwidthUtilization > 0.5 ? '#FFC107' : '#4CAF50'
                    ctx.globalAlpha = 0.8 + Math.sin(offset * Math.PI) * 0.2
                    ctx.beginPath()
                    ctx.arc(particleX, particleY, particleSize, 0, Math.PI * 2)
                    ctx.fill()

                    // Skip trailing effect for large clusters to save performance
                    if (bandwidthUtilization > 0.7 && gpuCount <= 16) {
                        ctx.globalAlpha = 0.3
                        for (let t = 1; t <= 3; t++) {
                            const trailOffset = offset - t * 0.02
                            if (trailOffset > 0) {
                                const trailX = x1 + (x2 - x1) * trailOffset
                                const trailY = y1 + (y2 - y1) * trailOffset
                                ctx.beginPath()
                                ctx.arc(trailX, trailY, particleSize * (1 - t * 0.2), 0, Math.PI * 2)
                                ctx.fill()
                            }
                        }
                    }
                }
            }
        })
    } else {
        // Tree topology through PCIe switch for non-NVLink or large clusters
        const switchX = canvas.width / 2
        const switchY = canvas.height - 80

        // Draw PCIe switch as a bottleneck point
        const switchWidth = 120
        const switchHeight = 40

        // Switch glows red when saturated
        if (bandwidthUtilization > 0.7) {
            ctx.shadowBlur = 20
            ctx.shadowColor = 'rgba(255, 68, 68, ' + bandwidthUtilization + ')'
        }

        const switchGradient = ctx.createLinearGradient(switchX - switchWidth/2, switchY - switchHeight/2,
                                                       switchX + switchWidth/2, switchY + switchHeight/2)
        switchGradient.addColorStop(0, bandwidthUtilization > 0.8 ? '#8B0000' : '#404040')
        switchGradient.addColorStop(1, bandwidthUtilization > 0.8 ? '#FF4444' : '#606060')

        ctx.fillStyle = switchGradient
        ctx.fillRect(switchX - switchWidth/2, switchY - switchHeight/2, switchWidth, switchHeight)
        ctx.shadowBlur = 0

        // Switch label changes color based on load
        ctx.fillStyle = bandwidthUtilization > 0.8 ? '#FF4444' :
                       bandwidthUtilization > 0.5 ? '#FFA500' : '#808080'
        ctx.font = 'bold 12px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('PCIe Switch', switchX, switchY)

        // Show bottleneck indicator
        if (bandwidthUtilization > 0.7) {
            ctx.fillStyle = '#FF4444'
            ctx.font = 'bold 10px monospace'
            ctx.fillText(`${(bandwidthUtilization * 100).toFixed(0)}% SATURATED`, switchX, switchY + 15)
        }

        // Connect each GPU to switch with traffic visualization
        for (let i = 0; i < gpuCount; i++) {
            const row = Math.floor(i / cols)
            const col = i % cols
            const x = offsetX + col * gpuSpacing
            const y = offsetY + row * gpuSpacing

            // Calculate path to switch
            const startX = x
            const startY = y + gpuSize / 2
            const endX = switchX
            const endY = switchY - switchHeight/2

            // Base connection
            ctx.strokeStyle = '#404040'
            ctx.globalAlpha = 0.3
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(startX, startY)
            ctx.lineTo(endX, endY)
            ctx.stroke()

            // Traffic flow visualization
            if (bandwidthUtilization > 0) {
                const flowWidth = 2 + bandwidthUtilization * 6
                ctx.strokeStyle = bandwidthUtilization > 0.8 ? '#FF4444' :
                                 bandwidthUtilization > 0.5 ? '#FFA500' : '#4CAF50'
                ctx.globalAlpha = 0.5 + bandwidthUtilization * 0.3
                ctx.lineWidth = flowWidth * bandwidthUtilization
                ctx.beginPath()
                ctx.moveTo(startX, startY)
                ctx.lineTo(endX, endY)
                ctx.stroke()
            }

            // Animate packets converging at the switch (showing congestion)
            if (isPlaying && bandwidthUtilization > 0) {
                const particleCount = Math.ceil(bandwidthUtilization * 3)
                for (let p = 0; p < particleCount; p++) {
                    const offset = (Date.now() / (2000 - bandwidthUtilization * 1500) + i * 0.1 + p / particleCount) % 1

                    // Particles slow down as they approach saturated switch
                    const slowdownFactor = bandwidthUtilization > 0.7 ?
                        1 - (bandwidthUtilization - 0.7) * Math.pow(offset, 2) : 1
                    const adjustedOffset = offset * slowdownFactor

                    const particleX = startX + (endX - startX) * adjustedOffset
                    const particleY = startY + (endY - startY) * adjustedOffset

                    // Particles bunch up near switch when congested
                    const particleSize = 2 + bandwidthUtilization * 3
                    ctx.fillStyle = bandwidthUtilization > 0.8 ? '#FF4444' : '#FFA500'
                    ctx.globalAlpha = 0.7
                    ctx.beginPath()
                    ctx.arc(particleX, particleY, particleSize, 0, Math.PI * 2)
                    ctx.fill()
                }
            }
        }
    }

    ctx.restore()

    // Draw GPUs
    for (let i = 0; i < gpuCount; i++) {
        const row = Math.floor(i / cols)
        const col = i % cols
        const x = offsetX + col * gpuSpacing
        const y = offsetY + row * gpuSpacing

        gpuPositions.push({ x, y })

        // Draw GPU die
        const gradient = ctx.createLinearGradient(x - gpuSize/2, y - gpuSize/2, x + gpuSize/2, y + gpuSize/2)
        gradient.addColorStop(0, '#1a1a2e')
        gradient.addColorStop(1, '#0a0a1a')

        ctx.fillStyle = gradient
        ctx.fillRect(x - gpuSize/2, y - gpuSize/2, gpuSize, gpuSize)

        // Draw GPU border
        ctx.strokeStyle = '#404060'
        ctx.lineWidth = 1
        ctx.strokeRect(x - gpuSize/2, y - gpuSize/2, gpuSize, gpuSize)

        // Draw memory usage bar
        const barHeight = 8
        const barWidth = gpuSize - 10
        const barX = x - barWidth/2
        const barY = y + gpuSize/2 - 20

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
        ctx.fillRect(barX, barY, barWidth, barHeight)

        // Memory usage
        const memUsageRatio = Math.min(1, memPerGPU / gpuConfig.memGiB)
        const usageColor = memUsageRatio > 0.9 ? '#FF4444' :
                          memUsageRatio > 0.7 ? '#FFA500' : '#4CAF50'

        ctx.fillStyle = usageColor
        ctx.fillRect(barX, barY, barWidth * memUsageRatio, barHeight)

        // GPU label
        ctx.fillStyle = '#808080'
        ctx.font = '9px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(`GPU ${i}`, x, y - gpuSize/2 - 5)

        // Memory text
        ctx.fillText(`${memPerGPU.toFixed(1)}/${gpuConfig.memGiB}G`, x, y + gpuSize/2 + 12)
    }

    // Draw interconnect info and bandwidth meter
    ctx.save()

    // Draw bandwidth meter (visual thermometer)
    const meterX = 20
    const meterY = canvas.height - 100
    const meterWidth = 200
    const meterHeight = 20

    // Meter background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
    ctx.fillRect(meterX, meterY, meterWidth, meterHeight)

    // Meter fill - gradient from green to red
    const meterGradient = ctx.createLinearGradient(meterX, meterY, meterX + meterWidth, meterY)
    meterGradient.addColorStop(0, '#4CAF50')
    meterGradient.addColorStop(0.5, '#FFC107')
    meterGradient.addColorStop(0.8, '#FF5722')
    meterGradient.addColorStop(1, '#FF0000')

    ctx.fillStyle = meterGradient
    ctx.fillRect(meterX, meterY, meterWidth * bandwidthUtilization, meterHeight)

    // Meter border - glows when high
    ctx.strokeStyle = bandwidthUtilization > 0.8 ? '#FF4444' : '#606060'
    ctx.lineWidth = bandwidthUtilization > 0.8 ? 2 : 1
    ctx.strokeRect(meterX, meterY, meterWidth, meterHeight)

    // Labels
    ctx.fillStyle = '#C0C0C0'
    ctx.font = '11px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`${interconnectType} Bandwidth: ${interconnectBW} GB/s`, meterX, meterY - 5)

    ctx.textAlign = 'center'
    ctx.fillStyle = bandwidthUtilization > 0.8 ? '#FF4444' :
                   bandwidthUtilization > 0.5 ? '#FFA500' : '#4CAF50'
    ctx.font = 'bold 12px monospace'
    ctx.fillText(`${(bandwidthUtilization * 100).toFixed(1)}%`, meterX + meterWidth/2, meterY + meterHeight/2 + 4)

    // Detailed bandwidth info
    ctx.textAlign = 'left'
    ctx.font = '10px monospace'
    ctx.fillStyle = '#999'
    const actualBW = interconnectBW * bandwidthUtilization
    ctx.fillText(`Active: ${actualBW.toFixed(1)} GB/s | KV Sync: ${(kvCacheSyncTraffic / 1024).toFixed(1)} GB`, meterX, meterY + meterHeight + 15)

    // Show bottleneck warning if bandwidth is saturated
    if (bandwidthUtilization > 0.8) {
        // Pulsing warning message
        const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7
        ctx.fillStyle = `rgba(255, 68, 68, ${pulse})`
        ctx.font = 'bold 16px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('⚠️ INTERCONNECT BOTTLENECK ⚠️', canvas.width / 2, 30)

        ctx.fillStyle = '#FF8888'
        ctx.font = '12px monospace'
        ctx.fillText(`${interconnectType} bandwidth saturated - inference will stall!`, canvas.width / 2, 48)

        // Show impact
        const latencyIncrease = Math.pow(bandwidthUtilization, 3) * 500  // Exponential latency increase
        ctx.font = '11px monospace'
        ctx.fillStyle = '#FFA500'
        ctx.fillText(`Latency impact: +${latencyIncrease.toFixed(0)}ms per token`, canvas.width / 2, 65)
    } else if (bandwidthUtilization > 0.5) {
        ctx.fillStyle = '#FFA500'
        ctx.font = '13px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('⚡ High interconnect traffic', canvas.width / 2, 30)
    }

    ctx.restore()
}

function drawMemoryGrid() {
    const model = models[currentModelIndex]

    // For traditional batching: show ALLOCATED memory (max context)
    // For continuous batching: show USED memory (current tokens)
    let kvGiB, kvMaxGiB, allocatedGiB

    if (!continuousBatching && !pagedAttention && batchSize > 1) {
        // Traditional batching: memory is pre-allocated for max context
        kvGiB = calculateBatchKVCache(model, currentTokens) // Actually used
        kvMaxGiB = calculateBatchKVCache(model, maxTokens) // Max possible
        allocatedGiB = kvMaxGiB // In traditional batching, we allocate for max
    } else {
        // Continuous batching or paged attention: allocate as needed
        kvGiB = calculateBatchKVCache(model, currentTokens)
        kvMaxGiB = calculateBatchKVCache(model, maxTokens)
        allocatedGiB = kvGiB // Only allocate what's needed
    }

    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0
    const totalGiB = allocatedGiB + weightsGiB // Total allocated memory
    const usedGiB = kvGiB + weightsGiB // Actually used memory

    // Skip complex rendering on mobile if performance is poor
    if (isMobile && currentTokens > 1000000) {
        return // Skip heavy GPU rendering on mobile for performance
    }
    const totalMaxGiB = kvMaxGiB + weightsGiB
    const fillRatio = totalMaxGiB > 0 ? totalGiB / totalMaxGiB : 0

    // GPU Architecture Layout
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2

    // GPU Die dimensions (center chip)
    const dieWidth = 280
    const dieHeight = 280
    const dieX = centerX - dieWidth / 2
    const dieY = centerY - dieHeight / 2

    // Memory Module dimensions (HBM, GDDR, etc.)
    const memWidth = 60
    const memHeight = 240
    const memGap = 30 // Gap between memory and die

    // Calculate memory module positions (4 modules on each side for high-end GPUs)
    const memoryModules = [
        // Left side memory stacks
        { x: dieX - memGap - memWidth, y: centerY - memHeight / 2, side: 'left', index: 0 },
        { x: dieX - memGap - memWidth * 2 - 10, y: centerY - memHeight / 2, side: 'left', index: 1 },

        // Right side memory stacks
        { x: dieX + dieWidth + memGap, y: centerY - memHeight / 2, side: 'right', index: 2 },
        { x: dieX + dieWidth + memGap + memWidth + 10, y: centerY - memHeight / 2, side: 'right', index: 3 },

        // Top memory stacks (for very high memory configs)
        {
            x: centerX - memHeight / 2,
            y: dieY - memGap - memWidth,
            side: 'top',
            index: 4,
            width: memHeight,
            height: memWidth,
        },

        // Bottom memory stacks
        {
            x: centerX - memHeight / 2,
            y: dieY + dieHeight + memGap,
            side: 'bottom',
            index: 5,
            width: memHeight,
            height: memWidth,
        },
    ]

    // Determine how many memory modules to show based on GPU memory and type
    const memGiB = getCurrentGPUMemGiB()
    const currentGPUConfig = gpuConfigs[currentGPU]
    const memType = currentGPUConfig ? currentGPUConfig.memType : 'HBM'
    let activeMemModules = 4 // Default to 4 memory stacks

    if (memType.includes('HBM')) {
        // HBM configurations
        if (memGiB >= 80)
            activeMemModules = 6 // H100/A100 80GB have 5-6 HBM stacks
        else if (memGiB >= 40)
            activeMemModules = 4 // A100 40GB has 4-5 HBM stacks
        else activeMemModules = 2 // Lower memory HBM configs
    } else if (memType.includes('GDDR')) {
        // GDDR configurations (typically 2-4 modules for consumer GPUs)
        if (memGiB >= 24)
            activeMemModules = 2 // Consumer GPUs have 2 memory modules
        else activeMemModules = 2
    } else {
        // Other memory types (SRAM, LPDDR5, etc.)
        activeMemModules = Math.min(4, Math.max(1, Math.floor(memGiB / 16)))
    }

    // Draw PCB substrate
    ctx.fillStyle = 'rgba(20, 30, 45, 0.8)'
    ctx.fillRect(centerX - 450, centerY - 350, 900, 700)
    ctx.strokeStyle = 'rgba(95, 163, 230, 0.2)'
    ctx.lineWidth = 2
    ctx.strokeRect(centerX - 450, centerY - 350, 900, 700)

    // Draw power delivery traces
    ctx.strokeStyle = 'rgba(255, 200, 0, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i < 20; i++) {
        ctx.beginPath()
        ctx.moveTo(centerX - 450 + i * 45, centerY - 350)
        ctx.lineTo(centerX - 450 + i * 45, centerY + 350)
        ctx.stroke()
    }

    // Generate sequence colors if continuous batching is enabled
    if (continuousBatching && batchSize > 1) {
        if (sequenceColors.length !== batchSize) {
            sequenceColors = generateSequenceColors(batchSize)
        }
    } else {
        // Even for single batch, we need at least one color
        sequenceColors = generateSequenceColors(1)
    }

    // Calculate total memory distribution across ALL HBM modules
    const gpuMemGiB = getCurrentGPUMemGiB()

    // Flash Attention: Show dramatic visualization of attention matrix memory NOT being used
    if (flashAttention && currentTokens > 0) {
        const attentionMatrixGiB = calculateAttentionMatrixSize(currentTokens, batchSize)
        const attentionRatio = Math.min(1.0, attentionMatrixGiB / gpuMemGiB)

        // Draw a large dramatic warning-style indicator showing memory savings
        ctx.save()
        const savingsBoxWidth = 350
        const savingsBoxHeight = 100
        const savingsX = centerX - savingsBoxWidth / 2
        const savingsY = centerY - 450

        // Pulsing glow effect around the savings indicator
        const glowPulse = Math.sin(Date.now() * 0.003) * 20 + 30
        ctx.shadowColor = 'rgba(255, 50, 50, 0.8)'
        ctx.shadowBlur = glowPulse
        ctx.shadowOffsetX = 0
        ctx.shadowOffsetY = 0

        // Main savings box with gradient
        const gradient = ctx.createLinearGradient(savingsX, savingsY, savingsX, savingsY + savingsBoxHeight)
        gradient.addColorStop(0, 'rgba(255, 50, 50, 0.9)')
        gradient.addColorStop(0.5, 'rgba(255, 80, 80, 0.85)')
        gradient.addColorStop(1, 'rgba(255, 50, 50, 0.9)')
        ctx.fillStyle = gradient
        ctx.fillRect(savingsX, savingsY, savingsBoxWidth, savingsBoxHeight)

        // Border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.lineWidth = 3
        ctx.strokeRect(savingsX, savingsY, savingsBoxWidth, savingsBoxHeight)

        // Text content
        ctx.shadowBlur = 0
        ctx.fillStyle = 'rgba(255, 255, 255, 1)'
        ctx.font = 'bold 24px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText('⚡ FLASH ATTENTION ACTIVE ⚡', centerX, savingsY + 25)

        ctx.font = 'bold 32px monospace'
        ctx.fillStyle = 'rgba(255, 255, 100, 1)'
        ctx.fillText(`${attentionMatrixGiB.toFixed(1)} GiB`, centerX, savingsY + 55)

        ctx.font = 'bold 16px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
        ctx.fillText('MEMORY SAVED', centerX, savingsY + 80)

        ctx.restore()

        // Draw prominent ghost memory blocks showing what WOULD be used without Flash Attention
        for (let i = 0; i < activeMemModules && i < memoryModules.length; i++) {
            const mem = memoryModules[i]
            const w = mem.width || memWidth
            const h = mem.height || memHeight

            // Calculate how much of this memory would be consumed by attention matrix
            const ghostHeight = Math.min(h, h * attentionRatio * activeMemModules)

            // Draw thick red border around what would be used
            const borderPulse = Math.sin(Date.now() * 0.002) * 0.2 + 0.5
            ctx.strokeStyle = `rgba(255, 50, 50, ${borderPulse})`
            ctx.lineWidth = 4
            ctx.setLineDash([15, 10])
            ctx.strokeRect(mem.x - 2, mem.y - 2, w + 4, ghostHeight + 4)
            ctx.setLineDash([])

            // Semi-transparent red overlay
            const pulse = Math.sin(Date.now() * 0.002) * 0.1 + 0.25
            ctx.fillStyle = `rgba(255, 50, 50, ${pulse})`
            ctx.fillRect(mem.x, mem.y, w, ghostHeight)

            // Draw diagonal stripes to show it's saved/eliminated
            ctx.save()
            const region = new Path2D()
            region.rect(mem.x, mem.y, w, ghostHeight)
            ctx.clip(region)

            ctx.strokeStyle = `rgba(255, 100, 100, 0.4)`
            ctx.lineWidth = 2
            ctx.setLineDash([8, 8])

            // Diagonal stripes
            for (let stripe = -h; stripe < w + h; stripe += 20) {
                ctx.beginPath()
                ctx.moveTo(mem.x + stripe, mem.y)
                ctx.lineTo(mem.x + stripe + ghostHeight, mem.y + ghostHeight)
                ctx.stroke()
            }

            ctx.restore()
            ctx.setLineDash([])

            // Draw "WOULD BE USED" text on each HBM module
            if (ghostHeight > 30) {
                ctx.save()
                ctx.font = 'bold 10px monospace'
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
                ctx.textAlign = 'center'
                ctx.textBaseline = 'middle'
                ctx.fillText('SAVED', mem.x + w / 2, mem.y + ghostHeight / 2)
                ctx.restore()
            }
        }

        // Draw arrows pointing from the savings box to the HBM modules
        if (activeMemModules >= 2) {
            ctx.strokeStyle = 'rgba(255, 100, 100, 0.6)'
            ctx.lineWidth = 2
            ctx.setLineDash([5, 5])

            // Left arrow
            ctx.beginPath()
            ctx.moveTo(savingsX, savingsY + savingsBoxHeight)
            ctx.lineTo(memoryModules[0].x + (memoryModules[0].width || memWidth) / 2, memoryModules[0].y - 10)
            ctx.stroke()

            // Right arrow if we have enough HBM modules
            if (activeMemModules > 2) {
                ctx.beginPath()
                ctx.moveTo(savingsX + savingsBoxWidth, savingsY + savingsBoxHeight)
                ctx.lineTo(memoryModules[2].x + (memoryModules[2].width || memWidth) / 2, memoryModules[2].y - 10)
                ctx.stroke()
            }

            ctx.setLineDash([])
        }
    }

    // Draw HBM modules
    for (let i = 0; i < activeMemModules && i < memoryModules.length; i++) {
        const mem = memoryModules[i]
        const w = mem.width || memWidth
        const h = mem.height || memHeight

        // HBM base (dark silicon)
        ctx.fillStyle = 'rgba(30, 35, 50, 0.9)'
        ctx.fillRect(mem.x, mem.y, w, h)

        // HBM memory banks (grid pattern)
        const bankSize = 8
        const bankSpacing = 10
        const banksX = Math.floor(w / bankSpacing)
        const banksY = Math.floor(h / bankSpacing)
        const totalBanks = banksX * banksY

        // Calculate what percentage of total GPU memory is allocated
        const memoryUsageRatio = totalGiB / gpuMemGiB

        // Each HBM module should show the same fill percentage
        // This represents distributed memory across all modules
        const filledBanks = Math.floor(totalBanks * memoryUsageRatio)

        // Calculate how many banks are for weights vs KV cache
        const weightRatio = includeWeights ? weightsGiB / totalGiB : 0
        const weightBanks = Math.floor(filledBanks * weightRatio)

        // For traditional batching, calculate how much of allocated memory is actually used
        const usageRatioWithinAllocation =
            !continuousBatching && !pagedAttention && allocatedGiB > 0 ? usedGiB / allocatedGiB : 1.0 // For CB/paged, allocated = used

        if (continuousBatching && batchSize > 1 && !pagedAttention) {
            // Continuous batching: show different colors for each sequence
            let banksFilled = 0
            let currentSeq = 0

            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx
                    const x = mem.x + bx * bankSpacing + 2
                    const y = mem.y + by * bankSpacing + 2

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - use purple/violet color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})` // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize)

                            // Add subtle border
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)'
                            ctx.lineWidth = 0.5
                            ctx.strokeRect(x, y, bankSize, bankSize)
                        } else {
                            // KV cache - determine which sequence this bank belongs to
                            const kvBankIndex = bankIndex - weightBanks
                            const kvTotalBanks = filledBanks - weightBanks

                            if (kvTotalBanks <= 0) {
                                continue // Skip if no KV banks
                            }

                            // Distribute banks among sequences based on their ratios
                            // Calculate total of all sequence ratios
                            let totalRatios = 0
                            for (let s = 0; s < batchSize; s++) {
                                totalRatios += getSequenceLengthRatio(s, batchSize)
                            }

                            // Calculate which sequence this bank belongs to
                            let bankStart = 0
                            currentSeq = 0
                            for (let s = 0; s < batchSize; s++) {
                                const seqRatio = getSequenceLengthRatio(s, batchSize)
                                const seqBanks = Math.floor((seqRatio / totalRatios) * kvTotalBanks)
                                const bankEnd = bankStart + seqBanks

                                if (kvBankIndex >= bankStart && kvBankIndex < bankEnd) {
                                    currentSeq = s
                                    break
                                }

                                bankStart = bankEnd

                                // Handle remaining banks for the last sequence
                                if (s === batchSize - 1) {
                                    currentSeq = s
                                }
                            }

                            const color = sequenceColors[currentSeq % sequenceColors.length] || '#5FA3E6'

                            // Convert hex to RGB for manipulation
                            const r = parseInt(color.substr(1, 2), 16)
                            const g = parseInt(color.substr(3, 2), 16)
                            const b = parseInt(color.substr(5, 2), 16)

                            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse})`
                            ctx.fillRect(x, y, bankSize, bankSize)
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)'
                        ctx.lineWidth = 0.5
                        ctx.strokeRect(x, y, bankSize, bankSize)
                    }
                }
            }
        } else if (pagedAttention) {
            // Paged attention: show fragmented memory with gaps
            // Generate sequence colors if we have multiple batches
            if (batchSize > 1) {
                if (sequenceColors.length !== batchSize) {
                    sequenceColors = generateSequenceColors(batchSize)
                }
            }

            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx
                    const x = mem.x + bx * bankSpacing + 2
                    const y = mem.y + by * bankSpacing + 2

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - always contiguous, purple color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})` // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize)
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)'
                            ctx.lineWidth = 0.5
                            ctx.strokeRect(x, y, bankSize, bankSize)
                        } else {
                            // KV cache with paging - show smaller page blocks
                            // Industry standard: vLLM uses 16 tokens per page, we visualize this with smaller blocks
                            const kvBankIndex = bankIndex - weightBanks
                            const kvTotalBanks = filledBanks - weightBanks

                            if (kvBankIndex < kvTotalBanks) {
                                // Each visual "bank" can hold 4 pages (2x2 grid)
                                // Calculate how many pages are actually used in this bank
                                const PAGES_PER_BANK = 4
                                const totalPagesNeeded = Math.ceil(
                                    kvTotalBanks * PAGES_PER_BANK * usageRatioWithinAllocation,
                                )
                                const currentBankStartPage = kvBankIndex * PAGES_PER_BANK
                                const pagesInThisBank = Math.min(
                                    PAGES_PER_BANK,
                                    Math.max(0, totalPagesNeeded - currentBankStartPage),
                                )

                                // Distribute pages across sequences using a deterministic interleaved pattern
                                // This simulates how pages from different sequences can be mixed in memory
                                const pagePattern = [0, 2, 1, 3, 0, 4, 2, 5, 1, 6, 3, 7, 4, 5, 6, 7]

                                // Draw each page slot in the bank
                                const subPageSize = bankSize / 2 - 1
                                let pageIndex = 0

                                for (let py = 0; py < 2; py++) {
                                    for (let px = 0; px < 2; px++) {
                                        const subX = x + px * (subPageSize + 1)
                                        const subY = y + py * (subPageSize + 1)

                                        if (pageIndex < pagesInThisBank) {
                                            // This page is allocated - determine which sequence owns it
                                            const globalPageIndex = currentBankStartPage + pageIndex
                                            const sequenceIndex =
                                                pagePattern[globalPageIndex % pagePattern.length] % batchSize

                                            const color =
                                                sequenceColors[sequenceIndex % sequenceColors.length] || '#5FA3E6'
                                            const r = parseInt(color.substr(1, 2), 16)
                                            const g = parseInt(color.substr(3, 2), 16)
                                            const b = parseInt(color.substr(5, 2), 16)

                                            // Filled page with sequence color
                                            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse * 0.8})`
                                            ctx.fillRect(subX, subY, subPageSize, subPageSize)

                                            // Flash Attention: Draw tile grid overlay
                                            if (flashAttention) {
                                                const tileGridSize = subPageSize / 2
                                                ctx.strokeStyle = 'rgba(255, 220, 0, 0.4)'
                                                ctx.lineWidth = 0.5

                                                // Draw tile grid
                                                for (let tg = 1; tg < 2; tg++) {
                                                    // Vertical line
                                                    ctx.beginPath()
                                                    ctx.moveTo(subX + tg * tileGridSize, subY)
                                                    ctx.lineTo(subX + tg * tileGridSize, subY + subPageSize)
                                                    ctx.stroke()

                                                    // Horizontal line
                                                    ctx.beginPath()
                                                    ctx.moveTo(subX, subY + tg * tileGridSize)
                                                    ctx.lineTo(subX + subPageSize, subY + tg * tileGridSize)
                                                    ctx.stroke()
                                                }
                                            }

                                            // Page border
                                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
                                            ctx.lineWidth = 0.5
                                            ctx.strokeRect(subX, subY, subPageSize, subPageSize)
                                        } else {
                                            // This page slot is empty
                                            ctx.strokeStyle = 'rgba(100, 150, 200, 0.15)'
                                            ctx.lineWidth = 0.5
                                            ctx.strokeRect(subX, subY, subPageSize, subPageSize)
                                        }

                                        pageIndex++
                                    }
                                }
                            }
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)'
                        ctx.lineWidth = 0.5
                        ctx.strokeRect(x, y, bankSize, bankSize)
                    }
                }
            }
        } else {
            // Traditional batching: show per-batch reserved chunks with used/unused portions
            if (batchSize > 1) {
                // Generate sequence colors for traditional batching visualization
                if (sequenceColors.length !== batchSize) {
                    sequenceColors = generateSequenceColors(batchSize)
                }
            }

            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx
                    const x = mem.x + bx * bankSpacing + 2
                    const y = mem.y + by * bankSpacing + 2

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - purple color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})` // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize)
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)'
                            ctx.lineWidth = 0.5
                            ctx.strokeRect(x, y, bankSize, bankSize)
                        } else {
                            // KV cache banks - show per-batch allocation
                            const kvBankIndex = bankIndex - weightBanks
                            const kvTotalBanks = filledBanks - weightBanks

                            if (batchSize > 1 && kvTotalBanks > 0) {
                                // Traditional batching: fixed pre-allocated chunks at specific memory addresses
                                // Each batch gets maximum possible space allocated upfront (no moving/shifting)
                                const maxBanksPerBatch = Math.ceil(kvTotalBanks / batchSize)
                                const batchNum = Math.floor(kvBankIndex / maxBanksPerBatch)
                                const bankInBatch = kvBankIndex % maxBanksPerBatch

                                // Each batch is fully pre-allocated but only partially used
                                // Usage ratio is based on current tokens vs max tokens
                                const usedBanksInBatch = Math.floor(maxBanksPerBatch * usageRatioWithinAllocation)

                                // Ensure batchNum doesn't exceed actual batch count
                                const actualBatchNum = Math.min(batchNum, batchSize - 1)

                                const color = sequenceColors[actualBatchNum % sequenceColors.length] || '#5FA3E6'
                                const r = parseInt(color.substr(1, 2), 16)
                                const g = parseInt(color.substr(3, 2), 16)
                                const b = parseInt(color.substr(5, 2), 16)

                                if (bankInBatch < usedBanksInBatch) {
                                    // Actually used memory - full bright color
                                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse})`
                                } else {
                                    // Pre-allocated but unused - very dim to show it's reserved
                                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse * 0.15})`
                                }
                                ctx.fillRect(x, y, bankSize, bankSize)

                                // Flash Attention: Draw tile grid overlay for traditional batching
                                if (flashAttention && bankInBatch < usedBanksInBatch) {
                                    const tileSize = bankSize / 3
                                    ctx.strokeStyle = 'rgba(255, 220, 0, 0.3)'
                                    ctx.lineWidth = 0.5

                                    for (let ti = 1; ti < 3; ti++) {
                                        // Vertical lines
                                        ctx.beginPath()
                                        ctx.moveTo(x + ti * tileSize, y)
                                        ctx.lineTo(x + ti * tileSize, y + bankSize)
                                        ctx.stroke()

                                        // Horizontal lines
                                        ctx.beginPath()
                                        ctx.moveTo(x, y + ti * tileSize)
                                        ctx.lineTo(x + bankSize, y + ti * tileSize)
                                        ctx.stroke()
                                    }
                                }

                                // Show allocation boundaries more subtly
                                if (bankInBatch === 0) {
                                    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.6)`
                                    ctx.lineWidth = 1.5
                                    ctx.strokeRect(x - 1, y, bankSize + 2, bankSize)
                                }
                            } else {
                                // Single batch or fallback - use standard blue
                                ctx.fillStyle = `rgba(100, 200, 255, ${pulse})`
                                ctx.fillRect(x, y, bankSize, bankSize)
                            }
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)'
                        ctx.lineWidth = 0.5
                        ctx.strokeRect(x, y, bankSize, bankSize)
                    }
                }
            }
        }

        // Memory type label (GPU-specific)
        const currentGPUConfig = gpuConfigs[currentGPU]
        const memoryType = currentGPUConfig ? currentGPUConfig.memType : 'HBM'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(memoryType, mem.x + w / 2, mem.y - 5)

        // Draw data paths from HBM to GPU die
        const moduleFillRatio = filledBanks / totalBanks
        if (moduleFillRatio > 0) {
            ctx.strokeStyle = model.color + '66'
            ctx.lineWidth = 2 + moduleFillRatio * 3
            ctx.setLineDash([5, 5])
            ctx.globalAlpha = 0.3 + moduleFillRatio * 0.4
            ctx.beginPath()

            let startX, startY, endX, endY
            if (mem.side === 'left') {
                startX = mem.x + w
                startY = mem.y + h / 2
                endX = dieX
                endY = centerY
                ctx.moveTo(startX, startY)
                ctx.lineTo(endX, endY)
            } else if (mem.side === 'right') {
                startX = mem.x
                startY = mem.y + h / 2
                endX = dieX + dieWidth
                endY = centerY
                ctx.moveTo(startX, startY)
                ctx.lineTo(endX, endY)
            } else if (mem.side === 'top') {
                startX = mem.x + w / 2
                startY = mem.y + h
                endX = centerX
                endY = dieY
                ctx.moveTo(startX, startY)
                ctx.lineTo(endX, endY)
            } else {
                startX = mem.x + w / 2
                startY = mem.y
                endX = centerX
                endY = dieY + dieHeight
                ctx.moveTo(startX, startY)
                ctx.lineTo(endX, endY)
            }

            ctx.stroke()
            ctx.setLineDash([])
            ctx.globalAlpha = 1

            // Generate data flow particles for active HBM modules
            // Flash Attention reduces bandwidth by 10-100x
            const particleRate = flashAttention ? 0.01 : 0.1
            if (Math.random() < particleRate * moduleFillRatio && isPlaying) {
                dataFlowParticles.push(
                    new DataFlowParticle(
                        startX,
                        startY,
                        endX,
                        endY,
                        flashAttention ? '#FFD700' : model.color, // Gold particles for Flash Attention
                        flashAttention ? 0.01 : 0.02 + Math.random() * 0.02, // Slower particles for Flash Attention
                    ),
                )
            }
        }
    }

    // Draw GPU Die
    // Die substrate
    const gradient = ctx.createLinearGradient(dieX, dieY, dieX + dieWidth, dieY + dieHeight)
    gradient.addColorStop(0, 'rgba(60, 70, 90, 0.95)')
    gradient.addColorStop(0.5, 'rgba(80, 90, 110, 0.95)')
    gradient.addColorStop(1, 'rgba(60, 70, 90, 0.95)')
    ctx.fillStyle = gradient
    ctx.fillRect(dieX, dieY, dieWidth, dieHeight)

    // Flash Attention: Draw SRAM/L2 Cache visualization inside GPU die
    if (flashAttention) {
        // Calculate and display attention matrix memory savings
        const attentionMatrixGiB = calculateAttentionMatrixSize(currentTokens, batchSize)

        // Show tiled computation visualization in GPU die
        ctx.save()

        // Draw small tiles to represent Flash Attention's tiled computation
        const flashTileSize = 8
        const tilesX = Math.floor((dieWidth - 40) / (flashTileSize + 2))
        const tilesY = Math.floor((dieHeight - 80) / (flashTileSize + 2))
        const startX = dieX + 20
        const startY = dieY + 60

        // Animate tiles to show computation flowing
        const animPhase = (Date.now() / 100) % (tilesX + tilesY)

        for (let ty = 0; ty < tilesY; ty++) {
            for (let tx = 0; tx < tilesX; tx++) {
                const x = startX + tx * (flashTileSize + 2)
                const y = startY + ty * (flashTileSize + 2)

                // Create wave effect for tiles
                const distance = tx + ty
                const isActive = Math.abs(distance - animPhase) < 3

                if (isActive) {
                    // Active tile - bright green
                    ctx.fillStyle = 'rgba(100, 255, 100, 0.8)'
                    ctx.fillRect(x, y, flashTileSize, flashTileSize)
                    ctx.strokeStyle = 'rgba(150, 255, 150, 1)'
                    ctx.lineWidth = 1
                    ctx.strokeRect(x, y, flashTileSize, flashTileSize)
                } else {
                    // Inactive tile - dim
                    ctx.strokeStyle = 'rgba(100, 150, 100, 0.3)'
                    ctx.lineWidth = 0.5
                    ctx.strokeRect(x, y, flashTileSize, flashTileSize)
                }
            }
        }

        // Labels inside GPU die
        ctx.fillStyle = 'rgba(100, 255, 100, 0.9)'
        ctx.font = 'bold 14px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('⚡ TILED COMPUTATION ⚡', centerX, dieY + 30)
        ctx.font = '11px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
        ctx.fillText(`O(1) memory instead of O(n²)`, centerX, dieY + 48)

        ctx.restore()

        const cacheX = dieX + dieWidth / 2 - 60
        const cacheY = dieY + dieHeight / 2 - 40
        const cacheWidth = 120
        const cacheHeight = 80

        // Cache background with gradient
        const cacheGradient = ctx.createLinearGradient(cacheX, cacheY, cacheX + cacheWidth, cacheY + cacheHeight)
        cacheGradient.addColorStop(0, 'rgba(255, 200, 0, 0.2)')
        cacheGradient.addColorStop(0.5, 'rgba(255, 220, 0, 0.3)')
        cacheGradient.addColorStop(1, 'rgba(255, 200, 0, 0.2)')
        ctx.fillStyle = cacheGradient
        ctx.fillRect(cacheX, cacheY, cacheWidth, cacheHeight)

        // Draw cache tiles (representing tiled computation)
        const cacheTileSize = 15
        const tilePadding = 3
        const cacheTilesX = Math.floor(cacheWidth / (cacheTileSize + tilePadding))
        const cacheTilesY = Math.floor(cacheHeight / (cacheTileSize + tilePadding))

        // Animate tiles with wave pattern
        const waveOffset = Date.now() * 0.002

        for (let ty = 0; ty < cacheTilesY; ty++) {
            for (let tx = 0; tx < cacheTilesX; tx++) {
                const x = cacheX + 10 + tx * (cacheTileSize + tilePadding)
                const y = cacheY + 10 + ty * (cacheTileSize + tilePadding)

                // Wave animation for tile access pattern
                const wave = Math.sin(waveOffset + tx * 0.5 + ty * 0.3) * 0.5 + 0.5
                const alpha = 0.3 + wave * 0.5

                ctx.fillStyle = `rgba(255, 220, 0, ${alpha})`
                ctx.fillRect(x, y, cacheTileSize, cacheTileSize)

                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
                ctx.lineWidth = 0.5
                ctx.strokeRect(x, y, cacheTileSize, cacheTileSize)
            }
        }

        // Cache label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.font = 'bold 10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('SRAM CACHE', cacheX + cacheWidth / 2, cacheY - 5)
        ctx.font = '9px monospace'
        ctx.fillText('(Tiled Access)', cacheX + cacheWidth / 2, cacheY + cacheHeight + 12)
    }

    // Die border
    ctx.strokeStyle = 'rgba(150, 200, 255, 0.5)'
    ctx.lineWidth = 2
    ctx.strokeRect(dieX, dieY, dieWidth, dieHeight)

    // Draw compute units grid on the die
    const cuSize = 12
    const cuSpacing = 16
    const cuStartX = dieX + 20
    const cuStartY = dieY + 20
    const cuCols = Math.floor((dieWidth - 40) / cuSpacing)
    const cuRows = Math.floor((dieHeight - 40) / cuSpacing)

    // Calculate compute unit activity based on memory usage
    const activity = fillRatio

    for (let row = 0; row < cuRows; row++) {
        for (let col = 0; col < cuCols; col++) {
            const x = cuStartX + col * cuSpacing
            const y = cuStartY + row * cuSpacing

            // Compute unit activity visualization
            const pulse = Math.sin(Date.now() * 0.003 + (row * cuCols + col) * 0.1) * 0.3 + 0.7
            const active = Math.random() < activity

            if (active) {
                // Active compute unit
                const heat = activity
                if (heat > 0.8) {
                    ctx.fillStyle = `rgba(255, ${Math.floor(100 - heat * 100)}, 0, ${pulse})`
                } else if (heat > 0.5) {
                    ctx.fillStyle = `rgba(255, 255, ${Math.floor(100 - heat * 100)}, ${pulse})`
                } else {
                    ctx.fillStyle = `rgba(0, ${Math.floor(150 + heat * 105)}, 255, ${pulse})`
                }
                ctx.fillRect(x, y, cuSize, cuSize)

                // Glow effect for active units
                const glow = ctx.createRadialGradient(
                    x + cuSize / 2,
                    y + cuSize / 2,
                    0,
                    x + cuSize / 2,
                    y + cuSize / 2,
                    cuSize,
                )
                glow.addColorStop(0, ctx.fillStyle)
                glow.addColorStop(1, 'transparent')
                ctx.fillStyle = glow
                ctx.fillRect(x - 2, y - 2, cuSize + 4, cuSize + 4)
            } else {
                // Idle compute unit
                ctx.fillStyle = 'rgba(50, 80, 120, 0.3)'
                ctx.fillRect(x, y, cuSize, cuSize)
                ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)'
                ctx.lineWidth = 0.5
                ctx.strokeRect(x, y, cuSize, cuSize)
            }
        }
    }

    // GPU die label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    ctx.font = 'bold 14px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('GPU DIE', centerX, dieY - 15)
    ctx.font = '11px monospace'
    ctx.fillText(`${Math.floor(activity * 100)}% Active`, centerX, dieY - 2)

    // Draw heat sink representation (subtle)
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.2)'
    ctx.lineWidth = 1
    ctx.setLineDash([2, 2])
    for (let i = 0; i < 5; i++) {
        ctx.strokeRect(dieX - 10 - i * 5, dieY - 10 - i * 5, dieWidth + 20 + i * 10, dieHeight + 20 + i * 10)
    }
    ctx.setLineDash([])

    // Add utilization indicator (positioned below bottom HBM module)
    const utilX = centerX
    // Position well below the bottom memory module (which is at dieY + dieHeight + memGap + memWidth)
    const bottomMemBottom = dieY + dieHeight + memGap + memWidth
    const utilY = bottomMemBottom + 40 // Add spacing below memory
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'center'

    // Always show usage vs GPU capacity
    const gpuCapacityGiB = getCurrentGPUMemGiB()
    const usagePercent = ((totalGiB / gpuCapacityGiB) * 100).toFixed(1)

    // First line: current usage vs GPU capacity
    ctx.fillText(
        `Memory: ${formatMemory(totalGiB)} / ${formatMemory(gpuCapacityGiB)} (${usagePercent}% of ${currentGPU})`,
        utilX,
        utilY,
    )

    // Second line: breakdown details
    if (continuousBatching && batchSize > 1) {
        ctx.font = '10px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.fillText(
            `KV: ${formatMemory(kvGiB)} (${batchSize} variable seq) + Weights: ${formatMemory(weightsGiB)}`,
            utilX,
            utilY + 15,
        )
    } else if (batchSize > 1) {
        ctx.font = '10px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.fillText(
            `KV: ${formatMemory(kvGiB)} (${batchSize}×${Math.floor(currentTokens)} tok) + Weights: ${formatMemory(weightsGiB)}`,
            utilX,
            utilY + 15,
        )
    } else {
        ctx.font = '10px monospace'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.fillText(`KV: ${formatMemory(kvGiB)} + Weights: ${formatMemory(weightsGiB)}`, utilX, utilY + 15)
    }

    ctx.globalAlpha = 1
}

// Draw exponential curve
function drawExponentialCurve() {
    const model = models[currentModelIndex]
    const points = []
    const steps = 100

    // Calculate points
    for (let i = 0; i <= steps; i++) {
        const tokens = (i / steps) * maxTokens
        // For the curve, we'll show the average case for continuous batching
        const kvGiB = continuousBatching
            ? calculateBatchKVCache(model, tokens)
            : calculateKVCacheSize(model, tokens) * batchSize
        const kvMaxGiB = continuousBatching
            ? calculateBatchKVCache(model, maxTokens)
            : calculateKVCacheSize(model, maxTokens) * batchSize
        const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0
        const memory = kvGiB + weightsGiB
        const maxMemory = kvMaxGiB + weightsGiB

        const x = (i / steps) * (canvas.width - 200) + 100
        const y = canvas.height - 100 - (memory / maxMemory) * (canvas.height - 200)

        points.push({ x, y, tokens, memory })
    }

    // Draw curve - green if Flash Attention is enabled to show efficiency
    ctx.strokeStyle = flashAttention ? 'rgba(100, 255, 100, 0.8)' : model.color
    ctx.lineWidth = 3
    ctx.beginPath()

    points.forEach((point, i) => {
        if (i === 0) {
            ctx.moveTo(point.x, point.y)
        } else {
            ctx.lineTo(point.x, point.y)
        }
    })

    ctx.stroke()

    // Calculate current position first for the arrow
    const currentRatio = currentTokens / maxTokens
    const currentX = currentRatio * (canvas.width - 200) + 100
    const kvGiBNow = calculateBatchKVCache(model, currentTokens)
    const kvGiBMax = calculateBatchKVCache(model, maxTokens)
    const weightsGiBNow = includeWeights ? calculateWeightMemoryGiB(model) : 0
    const currentMemory = kvGiBNow + weightsGiBNow
    const maxMemory = kvGiBMax + weightsGiBNow
    const currentY = canvas.height - 100 - (maxMemory > 0 ? currentMemory / maxMemory : 0) * (canvas.height - 200)

    // Add label for attention flow right next to the diagonal line
    ctx.save()

    // Get GPU dimensions to position slightly to the left of GPU
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const dieWidth = 280
    const dieHeight = 280
    const dieX = centerX - dieWidth / 2
    const memWidth2 = 100
    const memGap2 = 30
    const leftMemX = dieX - memGap2 - memWidth2 // Left edge of leftmost memory

    // Position more to the left of the left memory
    const labelX = leftMemX - 150 // 150px to the left of the GPU area

    // Find where the diagonal line is at this X position
    const lineStartX = 100
    const lineStartY = canvas.height - 100
    const lineEndX = canvas.width - 100
    const lineEndY = 100

    // Calculate Y position where diagonal crosses this X position
    const lineSlope = (lineEndY - lineStartY) / (lineEndX - lineStartX)
    const lineIntercept = lineStartY - lineSlope * lineStartX
    const labelY = lineSlope * labelX + lineIntercept + 60 // +60 to move it down more

    // Calculate the actual angle of the diagonal line
    const deltaX = lineEndX - lineStartX
    const deltaY = lineEndY - lineStartY
    const lineAngle = Math.atan2(deltaY, deltaX)

    // Rotate to match the actual diagonal line angle
    ctx.translate(labelX, labelY)
    ctx.rotate(lineAngle)

    // Background box for label (slightly bigger)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)'
    ctx.roundRect(-65, -12, 130, 24, 4)
    ctx.fill()

    // Border for clarity
    ctx.strokeStyle = flashAttention ? 'rgba(100, 255, 100, 0.4)' : 'rgba(95, 163, 230, 0.4)'
    ctx.lineWidth = 1
    ctx.roundRect(-65, -12, 130, 24, 4)
    ctx.stroke()

    // Label text (slightly bigger font)
    ctx.fillStyle = flashAttention ? 'rgba(100, 255, 100, 1)' : 'rgba(95, 163, 230, 1)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(flashAttention ? 'Attention: O(n·d)' : 'Attention: O(n²)', 0, 0)

    ctx.restore()

    // Pulsing circle at current position - smaller and different color with Flash Attention
    const basePulse = flashAttention ? 3 : 5 // Smaller base size with Flash Attention
    const pulseRange = flashAttention ? 2 : 5 // Less variation with Flash Attention
    const pulse = Math.sin(Date.now() * 0.003) * pulseRange + basePulse + 5

    // Striking yellow-green for Flash Attention, original color otherwise
    const dotColor = flashAttention ? 'rgba(200, 255, 0, 1)' : model.color

    ctx.beginPath()
    ctx.arc(currentX, currentY, pulse, 0, Math.PI * 2)
    ctx.fillStyle = dotColor
    ctx.fill()

    // More intense glow effect for Flash Attention
    const glowRadius = flashAttention ? pulse * 3 : pulse * 2
    const glow = ctx.createRadialGradient(currentX, currentY, 0, currentX, currentY, glowRadius)

    if (flashAttention) {
        // Bright yellow-green glow for Flash Attention
        glow.addColorStop(0, 'rgba(200, 255, 0, 0.8)')
        glow.addColorStop(0.5, 'rgba(150, 255, 0, 0.3)')
        glow.addColorStop(1, 'transparent')
    } else {
        glow.addColorStop(0, dotColor)
        glow.addColorStop(1, 'transparent')
    }

    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(currentX, currentY, glowRadius, 0, Math.PI * 2)
    ctx.fill()
}

// Generate dynamic factoids based on current state
function generateFactoids() {
    const model = models[currentModelIndex]
    const kvGiB = calculateKVCacheSize(model, currentTokens)
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0
    const memoryPerQuery = kvGiB + weightsGiB
    // Total memory for all concurrent queries
    const totalGiB = memoryPerQuery * batchSize
    const gpusNeeded = calculateGPUsNeeded(totalGiB)
    const per = getCurrentGPUMemGiB()
    const efficiency = Math.min(100, (totalGiB / (gpusNeeded * per)) * 100)

    // Only include factoids with hard references
    return [
        {
            title: '📏 KV Cache Formula',
            main: `2×layers×tokens×KV heads×head_dim×bytes`,
            detail: `Reference: LMCache KV Cache Calculator — https://lmcache.ai/kv_cache_calculator.html`,
        },
        {
            title: '🖥️ Device Memory Basis',
            main: `${currentGPU} → count: ${gpusNeeded}`,
            detail: `Devices needed = ceil(total GiB / per-device GiB). Selected device memory: ${getCurrentGPUMemGiB()} GiB.`,
        },
        {
            title: '⚖️ Weights Memory',
            main: `Weights ≈ params × bytes/param`,
            detail: `FP16 is 2 bytes/param (IEEE 754 half). Example: 70B × 2 B ≈ 140 GiB. Ref: https://en.wikipedia.org/wiki/Half-precision_floating-point_format`,
        },
        {
            title: '📊 Efficiency (definition)',
            main: `${efficiency.toFixed(1)}%`,
            detail: `Efficiency = used / allocated GPU memory. Used = Weights + KV. (Definition)`,
        },
    ]
}

// Update factoid display
function updateFactoid() {
    // Skip factoid updates on mobile
    if (isMobile) return

    const factoids = generateFactoids()
    const factoid = factoids[currentFactoidIndex % factoids.length]

    const panel = document.getElementById('factoidPanel')
    const title = document.getElementById('factoidTitle')
    const main = document.getElementById('factoidMain')
    const detail = document.getElementById('factoidDetail')

    // Fade out
    panel.style.opacity = '0'

    setTimeout(() => {
        title.textContent = factoid.title
        main.textContent = factoid.main
        detail.textContent = factoid.detail

        // Fade in
        panel.style.opacity = '1'
        // Don't reposition during fade - causes flicker
    }, 400)

    currentFactoidIndex++
}

// Calculate continuous batching performance improvements
function calculateCBPerformance() {
    // Based on Anyscale blog: up to 23x throughput with vLLM
    // Real-world improvements depend on request patterns and GPU utilization

    const baselineTokensPerSec = 10 // Traditional batching baseline
    let improvement = 1.0

    if (continuousBatching && pagedAttention) {
        // vLLM-like: CB + PagedAttention
        // Anyscale blog shows 8-23x, depends on:
        // - GPU memory utilization (higher util = more gain)
        // - Request arrival patterns (more concurrent = more gain)
        // - Sequence length variation (more variation = more gain from CB)

        // Calculate based on memory utilization and batch efficiency
        const model = models[currentModelIndex]
        const gpuConfig = gpuConfigs[currentGPU]
        const kvGiB = calculateBatchKVCache(model, currentTokens)
        const memoryUtilization = Math.min(1.0, kvGiB / (gpuConfig.memory * 0.9))

        // Base improvement: 8x minimum from CB
        // Scale up based on memory utilization and batch size
        // Higher memory util = better CB gains (can pack more requests)
        improvement = 8 + (memoryUtilization * 10) + Math.min(5, batchSize * 0.4)
        // Cap at realistic 23x maximum
        improvement = Math.min(23, improvement)
    } else if (continuousBatching) {
        // CB only: 4-8x improvement (Anyscale blog)
        // Scales with batch size but less dramatically
        const model = models[currentModelIndex]
        const gpuConfig = gpuConfigs[currentGPU]
        const kvGiB = calculateBatchKVCache(model, currentTokens)
        const memoryUtilization = Math.min(1.0, kvGiB / (gpuConfig.memory * 0.9))
        improvement = 4 + Math.min(4, batchSize * 0.3 + memoryUtilization * 2)
    } else if (pagedAttention) {
        // PA only: ~2x from memory efficiency
        improvement = 2.0
    }

    // Flash attention adds another 1.5-2x
    if (flashAttention) {
        improvement *= 1.5
    }

    return {
        throughput: baselineTokensPerSec * improvement,
        improvement: improvement,
        memoryEfficiency: pagedAttention ? 0.96 : (continuousBatching ? 0.7 : 0.4) // Memory utilization
    }
}

// Update performance metrics for multi-GPU deployments
function updatePerformanceMetrics() {
    const metricsEl = document.getElementById('metricsContent')
    if (!metricsEl || gpuCount === 1) return

    const model = models[currentModelIndex]
    const gpuConfig = gpuConfigs[currentGPU]
    const kvGiB = calculateBatchKVCache(model, currentTokens)

    // Determine actual interconnect bandwidth
    let interconnectBW = pcie4Bandwidth
    if (gpuConfig.nvlink && useHighSpeedInterconnect) {
        interconnectBW = gpuConfig.nvlinkBW || 600
    } else if (gpuConfig.pcieGen === 5) {
        interconnectBW = pcie5Bandwidth
    } else if (gpuConfig.pcieGen === 3) {
        interconnectBW = 16
    }

    // Calculate bandwidth utilization (more realistic with all-reduce pattern)
    const kvCacheSyncTraffic = (kvGiB * 1024) * Math.log2(gpuCount)  // GB for all-reduce
    const bandwidthUtilization = Math.min(1.0, kvCacheSyncTraffic / (interconnectBW * 1000))

    // Base performance assumptions (single GPU baseline)
    const baseTokensPerSec = 100  // Baseline tokens/sec on single GPU
    const baseLatencyMs = 10      // Baseline latency in ms

    // Calculate actual performance with interconnect impact
    // Performance degrades non-linearly as bandwidth saturates
    const performanceDegradation = 1 - Math.pow(bandwidthUtilization, 2)
    const actualTokensPerSec = baseTokensPerSec * performanceDegradation * Math.sqrt(gpuCount)  // Sub-linear scaling

    // Latency increases exponentially with saturation
    const latencyIncrease = baseLatencyMs * (1 + Math.pow(bandwidthUtilization, 2) * 10)
    const p50Latency = latencyIncrease
    const p99Latency = latencyIncrease * (1 + bandwidthUtilization * 3)  // P99 spikes much worse

    // Calculate cost efficiency
    const gpuCostPerHour = 2.49  // H100 approximate $/hour
    const totalCostPerHour = gpuCostPerHour * gpuCount
    const tokensPerDollar = (actualTokensPerSec * 3600) / totalCostPerHour

    // GPU utilization (decreases with interconnect bottleneck)
    const gpuUtilization = Math.max(20, 100 * performanceDegradation)

    // Build metrics HTML with color coding
    let html = ''

    // Tokens per second with trend
    const tokenColor = actualTokensPerSec > 50 ? '#4CAF50' :
                      actualTokensPerSec > 20 ? '#FFA500' : '#FF4444'
    html += `<div style="margin-bottom: 12px;">
        <strong style="color: ${tokenColor}">Throughput: ${actualTokensPerSec.toFixed(1)} tokens/sec</strong><br>
        <span style="color: #888; font-size: 0.9em">
            ${bandwidthUtilization > 0.5 ? '⬇' : '⬆'} ${(performanceDegradation * 100).toFixed(0)}% of ideal scaling
        </span>
    </div>`

    // Latency metrics
    const latencyColor = p50Latency < 50 ? '#4CAF50' :
                        p50Latency < 100 ? '#FFA500' : '#FF4444'
    html += `<div style="margin-bottom: 12px;">
        <strong>Latency Impact:</strong><br>
        <span style="color: ${latencyColor}">P50: ${p50Latency.toFixed(0)}ms | P99: ${p99Latency.toFixed(0)}ms</span><br>
        <span style="color: #888; font-size: 0.9em">
            ${bandwidthUtilization > 0.7 ? '⚠️ Causes stuttering output' : '✓ Smooth streaming'}
        </span>
    </div>`

    // GPU utilization
    const utilColor = gpuUtilization > 80 ? '#4CAF50' :
                      gpuUtilization > 50 ? '#FFA500' : '#FF4444'
    html += `<div style="margin-bottom: 12px;">
        <strong>GPU Utilization: <span style="color: ${utilColor}">${gpuUtilization.toFixed(0)}%</span></strong><br>
        <span style="color: #888; font-size: 0.9em">
            ${gpuUtilization < 70 ? '💸 GPUs idle waiting for data' : '✓ Good compute usage'}
        </span>
    </div>`

    // Cost efficiency
    html += `<div style="margin-bottom: 12px;">
        <strong>Cost Analysis:</strong><br>
        $${totalCostPerHour.toFixed(2)}/hour for ${gpuCount}× ${currentGPU}<br>
        <span style="color: #888; font-size: 0.9em">
            ${tokensPerDollar.toFixed(0)} tokens/$
            ${bandwidthUtilization > 0.5 ? '(poor ROI due to bottleneck)' : '(reasonable efficiency)'}
        </span>
    </div>`

    // Recommendation based on bottleneck
    if (bandwidthUtilization > 0.8) {
        html += `<div style="padding: 8px; background: rgba(255, 68, 68, 0.2); border-radius: 4px; margin-top: 10px;">
            <strong style="color: #FF4444">🚨 Critical Bottleneck</strong><br>
            <span style="font-size: 0.9em">
                ${useHighSpeedInterconnect ?
                    'Even with high-speed interconnect saturated!' :
                    'Enable NVLink or reduce GPU count'}
            </span>
        </div>`
    } else if (bandwidthUtilization > 0.5) {
        html += `<div style="padding: 8px; background: rgba(255, 165, 0, 0.2); border-radius: 4px; margin-top: 10px;">
            <strong style="color: #FFA500">⚡ Performance Limited</strong><br>
            <span style="font-size: 0.9em">
                Interconnect becoming bottleneck
            </span>
        </div>`
    }

    metricsEl.innerHTML = html
}

// Update paper references in dedicated box
function updatePaperReferences() {
    const paperRefsBoxEl = document.getElementById('paperRefsBox')
    const paperLinksEl = document.getElementById('paperRefsLinks')

    if (paperRefsBoxEl && paperLinksEl) {
        const papers = []

        if (continuousBatching) {
            papers.push(
                '<a href="https://www.usenix.org/system/files/osdi22-yu-gyeong.pdf" target="_blank">📄 Continuous Batching: Orca (OSDI\'22)</a>',
            )
        }

        if (pagedAttention) {
            papers.push(
                '<a href="https://arxiv.org/abs/2309.06180" target="_blank">📄 Paged Attention: vLLM (arXiv:2309.06180)</a>',
            )
        }

        if (flashAttention) {
            papers.push(
                '<a href="https://arxiv.org/abs/2205.14135" target="_blank">📄 FlashAttention: Dao et al. (arXiv:2205.14135)</a>',
            )
            papers.push(
                '<a href="https://arxiv.org/abs/2307.08691" target="_blank">📄 FlashAttention-2 (arXiv:2307.08691)</a>',
            )
        }

        if (papers.length > 0) {
            paperLinksEl.innerHTML = papers.join('\n')
            paperRefsBoxEl.style.display = 'block'
            paperRefsBoxEl.classList.add('show')

            // Position it below the efficiency box
            const efficiencyBoxEl = document.getElementById('efficiencyBox')
            if (efficiencyBoxEl) {
                const efficiencyRect = efficiencyBoxEl.getBoundingClientRect()
                paperRefsBoxEl.style.top = efficiencyRect.bottom + 20 + 'px'
                paperRefsBoxEl.style.left = efficiencyRect.left + 'px'
            }

            // Hide factoid when papers are shown
            if (window.positionFactoidPanel) window.positionFactoidPanel()
        } else {
            paperRefsBoxEl.classList.remove('show')
            setTimeout(() => {
                paperRefsBoxEl.style.display = 'none'
                // Show factoid again when papers are hidden
                if (window.positionFactoidPanel) window.positionFactoidPanel()
            }, 300) // Wait for fade animation
        }
    }
}

// Update info panel
function updateInfoPanel() {
    const model = models[currentModelIndex]
    const kvGiB = calculateBatchKVCache(model, currentTokens)
    const kvMaxGiB = calculateBatchKVCache(model, maxTokens)
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0

    // For traditional batching, show allocated vs used
    let totalGiB, allocatedGiB
    if (!continuousBatching && !pagedAttention && batchSize > 1) {
        allocatedGiB = kvMaxGiB + weightsGiB // Pre-allocated for max context
        totalGiB = kvGiB + weightsGiB // Actually used
    } else {
        allocatedGiB = kvGiB + weightsGiB
        totalGiB = allocatedGiB
    }

    const gpusNeeded = gpuCount > 1 ? gpuCount : calculateGPUsNeeded(allocatedGiB)

    document.getElementById('modelName').textContent = model.name
    document.getElementById('contextLength').textContent = `${formatNumber(Math.floor(currentTokens))} tokens`

    // Show multi-GPU cluster info
    if (gpuCount > 1) {
        const gpuConfig = gpuConfigs[currentGPU]
        const memPerGPU = allocatedGiB / gpuCount

        // Determine actual interconnect being used
        let interconnectType, interconnectBW
        if (gpuConfig.nvlink && useHighSpeedInterconnect) {
            if (gpuConfig.ifl) {
                interconnectType = 'Infinity Fabric'
                interconnectBW = gpuConfig.nvlinkBW || 400
            } else if (gpuConfig.tpuInterconnect) {
                interconnectType = 'TPU Link'
                interconnectBW = gpuConfig.nvlinkBW || 700
            } else if (gpuConfig.ipuLink) {
                interconnectType = 'IPU-Link'
                interconnectBW = gpuConfig.nvlinkBW || 320
            } else if (currentGPU.includes('Intel')) {
                interconnectType = 'Xe Link'
                interconnectBW = 400
            } else {
                interconnectType = `NVLink ${gpuConfig.nvlinkBW >= 900 ? '4.0' : '3.0'}`
                interconnectBW = gpuConfig.nvlinkBW || 600
            }
        } else {
            // Determine PCIe generation and bandwidth
            if (gpuConfig.pcieGen === 5) {
                interconnectType = 'PCIe 5.0'
                interconnectBW = pcie5Bandwidth
            } else if (gpuConfig.pcieGen === 3) {
                interconnectType = 'PCIe 3.0'
                interconnectBW = 16
            } else {
                interconnectType = 'PCIe 4.0'
                interconnectBW = pcie4Bandwidth
            }
        }

        const clusterMemory = gpuConfig.memGiB * gpuCount
        const utilization = (allocatedGiB / clusterMemory) * 100

        // Calculate bandwidth utilization
        const kvCacheSyncTraffic = (kvGiB * 1024) / gpuCount  // MB per GPU for all-reduce
        const bandwidthUtilization = Math.min(1.0, kvCacheSyncTraffic / (interconnectBW * 1000))

        // Update total memory to show cluster total
        const totalMemEl = document.getElementById('totalMemory')
        if (totalMemEl) {
            const bwColor = bandwidthUtilization > 0.8 ? '#FF4444' :
                           bandwidthUtilization > 0.5 ? '#FFA500' : '#4CAF50'

            totalMemEl.innerHTML = `<strong style="color: #FFD700">${gpuCount}×</strong> ${currentGPU}<br>` +
                                  `Cluster: ${clusterMemory.toFixed(0)} GiB total<br>` +
                                  `Using: ${allocatedGiB.toFixed(1)} GiB (${utilization.toFixed(1)}%)<br>` +
                                  `Link: ${interconnectType} (${interconnectBW} GB/s)<br>` +
                                  `<span style="color: ${bwColor}">Bandwidth: ${(bandwidthUtilization * 100).toFixed(1)}% used</span>`
        }
    }

    // Show allocation unit size
    const allocationEl = document.getElementById('allocationUnit')
    if (allocationEl) {
        const currentGPUConfig = gpuConfigs[currentGPU]
        const gpuTileSize = currentGPUConfig ? currentGPUConfig.flashTileSize : 64

        if (flashAttention && pagedAttention) {
            // Flash Attention with paged: tiles within pages
            allocationEl.textContent = `${Math.floor(gpuTileSize / 4)}×${Math.floor(gpuTileSize / 4)} tiles/page`
        } else if (flashAttention) {
            // Flash Attention: computation tiles (GPU-specific)
            allocationEl.textContent = `${gpuTileSize}×${gpuTileSize} tiles`
        } else if (pagedAttention) {
            // Industry standard: vLLM uses 16 tokens per page
            allocationEl.textContent = '16 tokens/page'
        } else {
            // Without paged attention: large contiguous blocks
            allocationEl.textContent = '2MB blocks'
        }
    }

    const weightsEl = document.getElementById('weightsSize')
    const totalEl = document.getElementById('totalSize')
    if (weightsEl) weightsEl.textContent = includeWeights ? formatMemory(weightsGiB) : '—'
    if (totalEl) {
        // Always use innerHTML to maintain consistent height with potential two-line layout
        if (flashAttention && currentTokens > 0) {
            const attentionMatrixGiB = calculateAttentionMatrixSize(currentTokens, batchSize)
            const withoutFlashGiB = totalGiB + attentionMatrixGiB
            const savingsPercent = Math.round((attentionMatrixGiB / withoutFlashGiB) * 100)
            totalEl.innerHTML = `<div style="min-height: 32px;"><span style="color: #4f4;">${formatMemory(totalGiB)}</span> <span style="color: #f88; text-decoration: line-through; font-size: 0.85em;">${formatMemory(withoutFlashGiB)}</span><br><small style="color: #4f4;">-${savingsPercent}% with Flash!</small></div>`
        } else if (!continuousBatching && !pagedAttention && batchSize > 1) {
            // Traditional batching: show allocated vs used
            const usagePercent = Math.round((totalGiB / allocatedGiB) * 100)
            totalEl.innerHTML = `<div style="min-height: 32px;">${formatMemory(allocatedGiB)} (${usagePercent}% used)</div>`
        } else if (continuousBatching && batchSize > 1) {
            // Continuous batching - just show total
            totalEl.innerHTML = `<div style="min-height: 32px;">${formatMemory(totalGiB)} (CB)</div>`
        } else if (pagedAttention) {
            // Paged attention - just show total
            totalEl.innerHTML = `<div style="min-height: 32px;">${formatMemory(totalGiB)} (paged)</div>`
        } else {
            totalEl.innerHTML = `<div style="min-height: 32px;">${formatMemory(totalGiB)}</div>`
        }
    }
    document.getElementById('cacheSize').textContent = formatMemory(kvGiB)

    // Show continuous batching performance improvements
    const showCBMetrics = (continuousBatching || pagedAttention) && batchSize > 1
    if (showCBMetrics) {
        const cbPerf = calculateCBPerformance()

        // Update the batch size display to show improvement
        const batchSizeEl = document.getElementById('batchSizeDisplay')
        if (batchSizeEl) {
            batchSizeEl.innerHTML = `${batchSize} <span style="color: #4CAF50; font-size: 0.8em">(${cbPerf.improvement.toFixed(1)}x)</span>`
        }

        // Add performance comparison text
        const kvCacheEl = document.getElementById('cacheSize')
        if (kvCacheEl && kvCacheEl.parentElement) {
            const parent = kvCacheEl.parentElement
            if (!parent.querySelector('.cb-perf-note')) {
                const note = document.createElement('div')
                note.className = 'cb-perf-note'
                note.style.cssText = 'font-size: 0.75em; color: #4CAF50; margin-top: 4px;'

                if (continuousBatching && pagedAttention) {
                    note.textContent = `vLLM-like: up to ${cbPerf.improvement.toFixed(0)}x throughput`
                } else if (continuousBatching) {
                    note.textContent = `CB: ${cbPerf.improvement.toFixed(0)}x throughput gain`
                } else {
                    note.textContent = `PA: ${(cbPerf.memoryEfficiency * 100).toFixed(0)}% memory util`
                }

                parent.appendChild(note)
            }
        }
    } else {
        // Remove CB metrics if disabled
        const batchSizeEl = document.getElementById('batchSizeDisplay')
        if (batchSizeEl) {
            batchSizeEl.textContent = batchSize
        }

        const notes = document.querySelectorAll('.cb-perf-note')
        notes.forEach(n => n.remove())
    }

    // Show appropriate box based on GPU count
    const infoPanelEl = document.querySelector('.info-panel')
    const efficiencyBoxEl = document.getElementById('efficiencyBox')
    const performanceMetricsBoxEl = document.getElementById('performanceMetricsBox')

    if (infoPanelEl) {
        const infoPanelRect = infoPanelEl.getBoundingClientRect()
        const infoPanelBottom = infoPanelRect.bottom

        if (gpuCount > 1) {
            // Hide efficiency box, show performance metrics for multi-GPU
            if (efficiencyBoxEl) efficiencyBoxEl.style.display = 'none'

            if (performanceMetricsBoxEl) {
                performanceMetricsBoxEl.style.display = 'block'
                performanceMetricsBoxEl.style.top = infoPanelBottom + 20 + 'px'
                performanceMetricsBoxEl.style.left = infoPanelRect.left + 'px'

                // Calculate and display performance metrics
                updatePerformanceMetrics()
            }
        } else {
            // Show efficiency box for single GPU
            if (performanceMetricsBoxEl) performanceMetricsBoxEl.style.display = 'none'

            if (efficiencyBoxEl) {
                efficiencyBoxEl.style.display = 'block'
                efficiencyBoxEl.style.top = infoPanelBottom + 20 + 'px'
                efficiencyBoxEl.style.left = infoPanelRect.left + 'px'
            }
        }
    }

    // Update paper references (positioned below efficiency box)
    // Hide paper references when using multiple GPUs to save screen space
    if (gpuCount === 1) {
        updatePaperReferences()
    } else {
        // Hide paper references box for multi-GPU view
        const paperRefsBoxEl = document.getElementById('paperRefsBox')
        if (paperRefsBoxEl) {
            paperRefsBoxEl.style.display = 'none'
        }
    }

    // Update GPU display to show batch processing info
    const gpuText = batchSize > 1 ? `${gpusNeeded} (${batchSize} queries/GPU)` : gpusNeeded
    document.getElementById('gpusNeeded').textContent = gpuText
    document.getElementById('dataType').textContent = currentDtype

    // Calculate efficiency based on GPU utilization
    // Efficiency represents how much of the allocated GPU memory is actually used
    // Low efficiency = wasted money on unused GPU memory
    const perGPU = getCurrentGPUMemGiB()
    const efficiency = Math.min(100, (totalGiB / (gpusNeeded * perGPU)) * 100)
    const efficiencyElement = document.getElementById('efficiency')
    efficiencyElement.textContent = `${efficiency.toFixed(1)}%`

    // Color code efficiency: green (>80%), yellow (50-80%), red (<50%)
    if (efficiency > 80) {
        efficiencyElement.style.color = '#00ff88'
    } else if (efficiency > 50) {
        efficiencyElement.style.color = '#FFB800'
    } else {
        efficiencyElement.style.color = '#FF6B00'
    }

    // Show warning for multi-GPU requirement or extreme memory
    const warning = document.getElementById('warning')
    let criticalState = 'none'
    if (totalGiB > 1000) {
        criticalState = 'datacenter'
        warning.style.display = 'block'
        warning.textContent = `⚠️ ${formatMemory(totalGiB)} - Exceeds datacenter capacity!`
        // Clarify below the memory emulation area
        const dcWrap = document.getElementById('datacenterNote')
        const dcBody = document.getElementById('datacenterNoteBody')
        if (dcWrap && dcBody) {
            dcBody.textContent = `Total KV + weights ≈ ${formatMemory(totalGiB)}. We flag > 1 TiB as beyond practical single-cluster GPU memory for this demo; real limits depend on your cluster (GPU count, memory per GPU, and interconnect bandwidth). Consider heavier quantization, KV compression/paging, or model sharding across many nodes.`
            dcWrap.style.display = 'block'
            if (window.positionDatacenterNote) window.positionDatacenterNote()
        }
    } else if (gpusNeeded > 8 && gpuCount === 1) {
        // Only show multi-node warning if user hasn't already selected multiple GPUs
        criticalState = 'multi-node'
        warning.style.display = 'block'
        warning.textContent = `⚠️ Requires ${gpusNeeded} devices (${currentGPU}) - Multi-node required!`
        const dcWrap = document.getElementById('datacenterNote')
        if (dcWrap) dcWrap.style.display = 'none'
    } else if (gpusNeeded > 1 && gpuCount === 1) {
        // Only show multi-GPU warning if user is still on single GPU
        criticalState = 'multi-gpu'
        warning.style.display = 'block'
        const perGPU = getCurrentGPUMemGiB()
        warning.textContent = `⚠️ Requires ${gpusNeeded} devices (${currentGPU}, ${formatMemory(gpusNeeded * perGPU)} total)`
        const dcWrap = document.getElementById('datacenterNote')
        if (dcWrap) dcWrap.style.display = 'none'
    } else if (gpuCount > 1 && allocatedGiB > (getCurrentGPUMemGiB() * gpuCount)) {
        // User selected multiple GPUs but still needs more
        const actualNeeded = Math.ceil(allocatedGiB / getCurrentGPUMemGiB())
        criticalState = 'none'  // Don't show popup, just warning
        warning.style.display = 'block'
        warning.textContent = `⚠️ Current ${gpuCount} GPUs insufficient - need ${actualNeeded} for this workload`
        const dcWrap = document.getElementById('datacenterNote')
        if (dcWrap) dcWrap.style.display = 'none'
    } else {
        warning.style.display = 'none'
        const dcWrap = document.getElementById('datacenterNote')
        if (dcWrap) dcWrap.style.display = 'none'
    }

    // Trigger critical popup on state transition with cooldown
    if (criticalState !== 'none' && criticalState !== lastCriticalState) {
        const now = Date.now()
        if (now - lastPopupTime > POPUP_COOLDOWN_MS) {
            showCriticalPopup(criticalState, { memoryGiB: totalGiB, gpusNeeded })
            lastPopupTime = now
        }
    }
    lastCriticalState = criticalState

    // Update progress bar
    const progress = (currentTokens / maxTokens) * 100
    document.getElementById('progressFill').style.width = `${progress}%`
}

// Choose a relevant factoid for the critical event
function pickRelevantFactoid(state) {
    const factoids = generateFactoids()
    // Map states to the most relevant hard-truth factoid
    if (state === 'multi-gpu' || state === 'multi-node' || state === 'datacenter') {
        return factoids.find((f) => f.title.includes('Device Memory Basis')) || factoids[0]
    }
    return factoids[0]
}

// Show critical popup
function showCriticalPopup(state, metrics) {
    // Skip critical popups on mobile
    if (isMobile) return

    const overlay = document.getElementById('criticalOverlay')
    if (!overlay) return
    const title = document.getElementById('criticalTitle')
    const main = document.getElementById('criticalMain')
    const detail = document.getElementById('criticalDetail')

    // Title and main message by state
    if (state === 'datacenter') {
        title.textContent = 'Critical: Capacity Exceeded'
        main.textContent = `${formatMemory(metrics.memoryGiB)} total KV memory`
        detail.textContent =
            'This exceeds realistic datacenter capacity — consider aggressive compression or sharding strategies.'
    } else if (state === 'multi-node') {
        title.textContent = 'Critical: Multi-Node Required'
        main.textContent = `${metrics.gpusNeeded}× devices required (${currentGPU})`
        detail.textContent =
            'Cross-node communication will dominate latency — pipeline and bandwidth optimizations are essential.'
    } else {
        title.textContent = 'Critical: Multi-Accelerator Required'
        main.textContent = `${metrics.gpusNeeded}× devices required (${currentGPU})`
        detail.textContent = `KV cache ≈ ${formatMemory(metrics.memoryGiB)} — exceeds single GPU capacity.`
    }

    // Append a relevant factoid snippet
    const factoid = pickRelevantFactoid(state)
    if (factoid) {
        detail.textContent += `\n\n${factoid.title} — ${factoid.main}. ${factoid.detail}`
    }

    overlay.style.display = 'flex'

    // Auto-close after a few seconds
    clearTimeout(showCriticalPopup._timer)
    showCriticalPopup._timer = setTimeout(() => {
        overlay.style.display = 'none'
    }, 6000)
}

// Close control
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('criticalOverlay')
    const close = document.getElementById('criticalClose')
    if (close) {
        close.addEventListener('click', () => (overlay.style.display = 'none'))
    }
    if (overlay) {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.style.display = 'none'
        })
    }
})

// Generate particles based on memory growth
function generateParticles() {
    const model = models[currentModelIndex]
    const memoryGiB = calculateKVCacheSize(model, currentTokens)

    // Generate more particles as memory grows
    const particleRate = Math.min(10, memoryGiB / 10)

    if (Math.random() < particleRate / 60) {
        const x = Math.random() * canvas.width
        const y = canvas.height - 50
        const size = 10 + Math.random() * 20

        memoryBlocks.push(new MemoryBlock(x, y, size, model.color))
    }
}

// Animation loop
function animate() {
    try {
        if (!canvas || !ctx) return requestAnimationFrame(animate)

        ctx.fillStyle = 'rgba(15, 15, 30, 0.1)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        // Update and draw waves
        waves.forEach((wave) => {
            wave.update()
            wave.draw()
        })

        // Draw visualizations
        if (gpuCount > 1) {
            drawMultiGPUCluster()
        } else {
            drawMemoryGrid()
        }
        drawExponentialCurve()

        // Update and draw data flow particles
        dataFlowParticles = dataFlowParticles.filter((particle) => particle.life > 0)
        dataFlowParticles.forEach((particle) => {
            particle.update()
            particle.draw()
        })

        // Update and draw particles
        memoryBlocks = memoryBlocks.filter((block) => block.life > 0)
        memoryBlocks.forEach((block) => {
            block.update()
            block.draw()
        })

        // Generate new particles
        generateParticles()

        // Update token count with variable speed based on max context
        if (isPlaying) {
            const baseIncrement = Math.max(100, maxTokens / 10000)
            // Slow down by 50% when Flash Attention is enabled to show efficiency
            const flashAttentionModifier = flashAttention ? 0.5 : 1.0
            currentTokens += baseIncrement * animationSpeed * flashAttentionModifier
            if (currentTokens >= maxTokens) {
                currentTokens = 0 // Loop
            }

            // Update sequence lengths when tokens change in continuous batching mode
            if (continuousBatching && batchSize > 1) {
                batchSequenceLengths = []
                for (let i = 0; i < batchSize; i++) {
                    const ratio = getSequenceLengthRatio(i, batchSize)
                    batchSequenceLengths.push(Math.floor(currentTokens * ratio))
                }
            }
        }

        // Update info panel
        updateInfoPanel()

        // Update factoids every 4 seconds
        const now = Date.now()
        if (now - lastFactoidUpdate > 4000) {
            updateFactoid()
            lastFactoidUpdate = now
        }
    } catch (e) {
        // Surface runtime errors in the warning box to aid debugging
        const warning = document.getElementById('warning')
        if (warning) {
            warning.style.display = 'block'
            warning.textContent = `⚠️ Visualization error: ${e && e.message ? e.message : e}`
        }
    } finally {
        requestAnimationFrame(animate)
    }
}

// Control handlers
document.getElementById('playPause').addEventListener('click', function () {
    isPlaying = !isPlaying
    this.textContent = isPlaying ? 'Pause' : 'Play'
    this.classList.toggle('active', isPlaying)
})

// Batch size control
document.getElementById('batchControl').addEventListener('click', function () {
    const batchSizes = [1, 2, 4, 8, 16, 32, 64, 128]
    const currentIndex = batchSizes.indexOf(batchSize)
    const nextIndex = (currentIndex + 1) % batchSizes.length
    batchSize = batchSizes[nextIndex]
    this.textContent = `Batch: ${batchSize}`

    // Regenerate sequence lengths and colors when batch size changes
    if (continuousBatching && batchSize > 1) {
        batchSequenceLengths = []
        sequenceColors = generateSequenceColors(batchSize)
        for (let i = 0; i < batchSize; i++) {
            const ratio = getSequenceLengthRatio(i, batchSize)
            batchSequenceLengths.push(Math.floor(currentTokens * ratio))
        }
    }

    // Force update display to show new calculations
    updateInfoPanel()
})

// Continuous batching toggle
const cbBtn = document.getElementById('cbToggle')
if (cbBtn) {
    cbBtn.addEventListener('click', function () {
        continuousBatching = !continuousBatching
        const spanEl = this.querySelector('span:first-child')
        if (spanEl) {
            spanEl.textContent = continuousBatching ? 'CB: ON' : 'CB: OFF'
        }
        this.classList.toggle('active', continuousBatching)

        // Regenerate sequence lengths when toggling
        if (continuousBatching && batchSize > 1) {
            batchSequenceLengths = []
            sequenceColors = generateSequenceColors(batchSize)
            for (let i = 0; i < batchSize; i++) {
                const ratio = getSequenceLengthRatio(i, batchSize)
                batchSequenceLengths.push(Math.floor(currentTokens * ratio))
            }
        } else {
            batchSequenceLengths = []
            sequenceColors = generateSequenceColors(1)
        }

        updateInfoPanel()
    })
}

// Paged attention toggle
const paBtn = document.getElementById('paToggle')
if (paBtn) {
    paBtn.addEventListener('click', function () {
        pagedAttention = !pagedAttention
        const spanEl = this.querySelector('span:first-child')
        if (spanEl) {
            spanEl.textContent = pagedAttention ? 'PA: ON' : 'PA: OFF'
        }
        this.classList.toggle('active', pagedAttention)

        updateInfoPanel()
    })
}

// Flash Attention toggle
const faBtn = document.getElementById('faToggle')
if (faBtn) {
    faBtn.addEventListener('click', function () {
        flashAttention = !flashAttention
        const spanEl = this.querySelector('span:first-child')
        if (spanEl) {
            spanEl.textContent = flashAttention ? 'FA: ON' : 'FA: OFF'
        }
        this.classList.toggle('active', flashAttention)
        updateInfoPanel()
    })
}

document.getElementById('speedControl').addEventListener('click', function () {
    const speeds = [0.5, 1, 2, 5, 10, 20, 50, 100]
    const currentIndex = speeds.indexOf(animationSpeed)
    animationSpeed = speeds[(currentIndex + 1) % speeds.length]
    this.textContent = `Speed: ${animationSpeed}x`
})

document.getElementById('modelSwitch').addEventListener('click', function () {
    currentModelIndex = (currentModelIndex + 1) % models.length
    currentTokens = 0
    initWaves()
    memoryBlocks = []
})

// Context length control
document.getElementById('contextControl').addEventListener('click', function () {
    const contexts = Object.keys(contextPresets)
    let currentContext = null

    // Find exact match or closest context
    for (let key of contexts) {
        if (contextPresets[key] === maxTokens) {
            currentContext = key
            break
        }
    }

    // If no exact match, find the closest one
    if (!currentContext) {
        let minDiff = Infinity
        for (let key of contexts) {
            const diff = Math.abs(contextPresets[key] - maxTokens)
            if (diff < minDiff) {
                minDiff = diff
                currentContext = key
            }
        }
    }

    const currentIndex = contexts.indexOf(currentContext)
    const nextContext = contexts[(currentIndex + 1) % contexts.length]
    maxTokens = contextPresets[nextContext]
    this.textContent = `Context: ${nextContext}`

    // Reset if current tokens exceed new max
    if (currentTokens > maxTokens) {
        currentTokens = 0
    }
})

// Data type control
document.getElementById('dtypeControl').addEventListener('click', function () {
    const dtypes = Object.keys(dtypeConfigs)
    let currentIndex = 0

    // Find current dtype
    for (let i = 0; i < dtypes.length; i++) {
        if (dtypes[i] === currentDtype) {
            currentIndex = i
            break
        }
    }

    currentDtype = dtypes[(currentIndex + 1) % dtypes.length]
    this.textContent = `Type: ${currentDtype}`

    // Update model colors based on dtype
    const config = dtypeConfigs[currentDtype]
    models.forEach((model) => {
        if (!model.originalColor) {
            model.originalColor = model.color
        }
        // Blend model color with dtype color for visual feedback
        model.color = model.originalColor
    })

    initWaves()
})

// Skip Model Weights (SMW) toggle
const smwBtn = document.getElementById('smwToggle')
if (smwBtn) {
    smwBtn.addEventListener('click', function () {
        includeWeights = !includeWeights
        this.classList.toggle('active', !includeWeights) // active means skipping weights
        updateInfoPanel()
    })
}

// GPU selection control
const gpuBtn = document.getElementById('gpuControl')
if (gpuBtn) {
    gpuBtn.addEventListener('click', function () {
        const keys = Object.keys(gpuConfigs)
        const idx = keys.indexOf(currentGPU)
        currentGPU = keys[(idx + 1) % keys.length]
        this.textContent = `Device: ${currentGPU}`

        // Set appropriate defaults for the new GPU
        setGPUDefaults(currentGPU)

        // Update interconnect button visibility
        updateInterconnectButton()

        // Update UI to reflect new defaults
        updateInfoPanel()
        updateControlStates()
    })
}

// GPU count control (powers of 2)
const gpuCountBtn = document.getElementById('gpuCountControl')
if (gpuCountBtn) {
    gpuCountBtn.addEventListener('click', function () {
        const idx = validGPUCounts.indexOf(gpuCount)
        gpuCount = validGPUCounts[(idx + 1) % validGPUCounts.length]
        this.textContent = `GPUs: ${gpuCount}`

        // When switching to multi-GPU for the first time, adjust context if needed
        if (gpuCount > 1 && currentTokens === 0) {
            // Start with some tokens to show the interconnect in action
            currentTokens = 10000
        }

        // Show/hide interconnect button based on GPU count and availability
        updateInterconnectButton()

        // Update info panel to show multi-GPU info
        updateInfoPanel()
    })
}

// Interconnect control (NVLink vs PCIe)
const interconnectBtn = document.getElementById('interconnectControl')
if (interconnectBtn) {
    interconnectBtn.addEventListener('click', function () {
        const gpuConfig = gpuConfigs[currentGPU]
        if (!gpuConfig.nvlink || gpuCount === 1) return

        useHighSpeedInterconnect = !useHighSpeedInterconnect

        // Update button text
        if (useHighSpeedInterconnect) {
            let linkName = 'NVLink'
            if (gpuConfig.ifl) linkName = 'IFL'
            else if (gpuConfig.tpuInterconnect) linkName = 'TPU Link'
            else if (gpuConfig.ipuLink) linkName = 'IPU-Link'
            else if (currentGPU.includes('Intel')) linkName = 'Xe Link'

            this.textContent = `Link: ${linkName}`
            this.style.background = 'linear-gradient(180deg, rgba(118, 185, 0, 0.2), rgba(118, 185, 0, 0.1))'
        } else {
            this.textContent = 'Link: PCIe'
            this.style.background = ''
        }

        // Update info panel to reflect bandwidth change
        updateInfoPanel()
    })
}

// World DC control (Famous GPU clusters)
// Store original settings before DC selection
let originalSettings = {
    gpuCount: 1,
    currentGPU: 'H100 80G',
    currentModelIndex: 2,  // Llama-3.1-8B by default
    useHighSpeedInterconnect: true
}

const worldDCBtn = document.getElementById('worldDCControl')
if (worldDCBtn) {
    // Capture initial settings
    originalSettings.gpuCount = gpuCount
    originalSettings.currentGPU = currentGPU
    originalSettings.currentModelIndex = currentModelIndex
    originalSettings.useHighSpeedInterconnect = useHighSpeedInterconnect

    worldDCBtn.addEventListener('click', function () {
        // Cycle through datacenters
        const dcKeys = Object.keys(worldDatacenters)
        const currentIdx = dcKeys.indexOf(currentDatacenter)
        currentDatacenter = dcKeys[(currentIdx + 1) % dcKeys.length]

        const dc = worldDatacenters[currentDatacenter]
        this.textContent = `DC: ${dc.name}`

        if (currentDatacenter === 'none') {
            // Restore original settings
            gpuCount = originalSettings.gpuCount
            currentGPU = originalSettings.currentGPU
            currentModelIndex = originalSettings.currentModelIndex
            useHighSpeedInterconnect = originalSettings.useHighSpeedInterconnect

            // Update UI
            const gpuCountBtn = document.getElementById('gpuCountControl')
            if (gpuCountBtn) gpuCountBtn.textContent = `GPUs: ${gpuCount}`

            const gpuBtn = document.getElementById('gpuControl')
            if (gpuBtn) gpuBtn.textContent = `Device: ${currentGPU}`

            setGPUDefaults(currentGPU)
            updateModelUI()
            updateInterconnectButton()
            updateInfoPanel()
        } else if (dc.gpus !== null) {
            // Apply datacenter configuration
            // Set GPU count
            gpuCount = dc.gpus
            const gpuCountBtn = document.getElementById('gpuCountControl')
            if (gpuCountBtn) gpuCountBtn.textContent = `GPUs: ${gpuCount}`

            // Set GPU type
            if (dc.gpu && gpuConfigs[dc.gpu]) {
                currentGPU = dc.gpu
                const gpuBtn = document.getElementById('gpuControl')
                if (gpuBtn) gpuBtn.textContent = `Device: ${dc.gpu}`
                setGPUDefaults(dc.gpu)
            }

            // Set model
            if (dc.model) {
                const modelIdx = models.findIndex(m => m.name === dc.model)
                if (modelIdx !== -1) {
                    currentModelIndex = modelIdx
                    updateModelUI()
                }
            }

            // Set interconnect
            if (dc.interconnect === 'nvlink') {
                useHighSpeedInterconnect = true
            } else if (dc.interconnect === 'pcie') {
                useHighSpeedInterconnect = false
            }

            updateInterconnectButton()
            updateInfoPanel()
        }
    })
}

// Helper function to update interconnect button visibility
function updateInterconnectButton() {
    const btn = document.getElementById('interconnectControl')
    if (!btn) return

    const gpuConfig = gpuConfigs[currentGPU]

    // Show button only if GPU has high-speed interconnect and multiple GPUs selected
    if (gpuConfig.nvlink && gpuCount > 1) {
        btn.style.display = ''

        // Set initial text based on current state
        if (useHighSpeedInterconnect) {
            let linkName = 'NVLink'
            if (gpuConfig.ifl) linkName = 'IFL'
            else if (gpuConfig.tpuInterconnect) linkName = 'TPU Link'
            else if (gpuConfig.ipuLink) linkName = 'IPU-Link'
            else if (currentGPU.includes('Intel')) linkName = 'Xe Link'

            btn.textContent = `Link: ${linkName}`
            btn.style.background = 'linear-gradient(180deg, rgba(118, 185, 0, 0.2), rgba(118, 185, 0, 0.1))'
        } else {
            btn.textContent = 'Link: PCIe'
            btn.style.background = ''
        }
    } else {
        btn.style.display = 'none'
    }
}

// Initialize
window.addEventListener('resize', () => {
    resizeCanvas()
    updateInfoPanel() // Reposition efficiency box on resize
})
resizeCanvas()
initWaves()

// Sync initial control states
document.addEventListener('DOMContentLoaded', () => {
    const playBtn = document.getElementById('playPause')
    if (playBtn) {
        playBtn.textContent = isPlaying ? 'Pause' : 'Play'
        playBtn.classList.toggle('active', isPlaying)
    }
    const speedBtn = document.getElementById('speedControl')
    if (speedBtn) speedBtn.textContent = `Speed: ${animationSpeed}x`
    const batchBtn = document.getElementById('batchControl')
    if (batchBtn) batchBtn.textContent = `Batch: ${batchSize}`
    const dtypeBtn = document.getElementById('dtypeControl')
    if (dtypeBtn) dtypeBtn.textContent = `Type: ${currentDtype}`
    const gpuBtn2 = document.getElementById('gpuControl')
    if (gpuBtn2) gpuBtn2.textContent = `Device: ${currentGPU}`
})

// Initialize first factoid
setTimeout(() => {
    updateFactoid()
    lastFactoidUpdate = Date.now()
}, 100)

// Set initial defaults for the default GPU
setGPUDefaults(currentGPU)
updateControlStates()

animate()
