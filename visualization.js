// KV Cache Growth Visualization
// Memory calculations based on LMCache KV Cache Calculator
// Source: https://lmcache.ai/kv_cache_calculator.html
// Project: https://github.com/mcgrof/kvcache-view
// This visualization uses the exact formulas from LMCache's calculator
// to accurately compute KV cache memory requirements

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Check if mobile
const isMobile = window.matchMedia('(max-width: 768px)').matches;

// Model configurations from LMCache
// Using distinct colors for each model
const models = [
    {
        name: "Llama-3.2-1B",
        params: 1.2,
        layers: 16,
        hidden_size: 2048,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#5FA3E6', // Light Blue
        efficiency: 'high'
    },
    {
        name: "Phi-3.5-mini",
        params: 3.8,
        layers: 32,
        hidden_size: 3072,
        num_heads: 32,
        num_kv_heads: 32,
        color: '#00C853', // Green
        efficiency: 'high'
    },
    {
        name: "Llama-3.1-8B",
        params: 8,
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#1428A0', // Deep Blue
        efficiency: 'medium'
    },
    {
        name: "Gemma-2-9B",
        params: 9,
        layers: 42,
        hidden_size: 3584,
        num_heads: 16,
        num_kv_heads: 8,
        color: '#4285F4', // Google Blue
        efficiency: 'high'
    },
    {
        name: "Qwen2.5-14B",
        params: 14,
        layers: 48,
        hidden_size: 5120,
        num_heads: 40,
        num_kv_heads: 8,
        color: '#FF9800', // Orange
        efficiency: 'medium'
    },
    {
        name: "Phi-3.5-MoE",
        params: 41.9, // 16 experts, 2 active
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_local_experts: 16,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#00BCD4', // Cyan
        efficiency: 'high'
    },
    {
        name: "Gemma-2-27B",
        params: 27,
        layers: 46,
        hidden_size: 4608,
        num_heads: 32,
        num_kv_heads: 16,
        color: '#34A853', // Google Green
        efficiency: 'medium'
    },
    {
        name: "Qwen2.5-32B",
        params: 32,
        layers: 64,
        hidden_size: 5120,
        num_heads: 40,
        num_kv_heads: 8,
        color: '#FFC107', // Amber
        efficiency: 'medium'
    },
    {
        name: "Mixtral-8x7B",
        params: 46.7, // 8 experts, 2 active = 12.9B active
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#9C27B0', // Deep Purple
        efficiency: 'high'
    },
    {
        name: "Llama-3.1-70B",
        params: 70,
        layers: 80,
        hidden_size: 8192,
        num_heads: 64,
        num_kv_heads: 8,
        color: '#691FFF', // Purple
        efficiency: 'low'
    },
    {
        name: "Qwen2.5-72B",
        params: 72,
        layers: 80,
        hidden_size: 8192,
        num_heads: 64,
        num_kv_heads: 8,
        color: '#FF5722', // Deep Orange
        efficiency: 'low'
    },
    {
        name: "Qwen3-Next-80B",
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
        efficiency: 'optimized'
    },
    {
        name: "Mixtral-8x22B",
        params: 141, // 8 experts, 2 active = 39.1B active
        layers: 56,
        hidden_size: 6144,
        num_heads: 48,
        num_kv_heads: 8,
        num_local_experts: 8,
        num_experts_per_tok: 2,
        architecture: 'moe',
        color: '#7B1FA2', // Dark Purple
        efficiency: 'medium'
    },
    {
        name: "Llama-3.1-405B",
        params: 405,
        layers: 126,
        hidden_size: 16384,
        num_heads: 128,
        num_kv_heads: 8,
        color: '#E4002B', // Red
        efficiency: 'very-low'
    },
    {
        name: "DeepSeek-V3 (671B)",
        params: 671,
        layers: 61,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        color: '#FF6B00', // Orange
        efficiency: 'optimized',
        special: 'deepseek'
    }
];

let currentModelIndex = 0;
let currentTokens = 0;
let maxTokens = 1000000; // 1M context default
let animationSpeed = 50;
let isPlaying = false; // start paused so first click plays
let particles = [];
let memoryBlocks = [];
let waves = [];
let currentDtype = 'FP16';
let currentFactoidIndex = 0;
let lastFactoidUpdate = 0;
let lastCriticalState = 'none';
let lastPopupTime = 0;
const POPUP_COOLDOWN_MS = 10000;
let includeWeights = true; // Include model weights memory by default
let batchSize = 1; // Number of concurrent queries per GPU
let dataFlowParticles = []; // Particles flowing between HBM and GPU
let continuousBatching = false; // Enable continuous batching with variable sequence lengths
let batchSequenceLengths = []; // Array of sequence lengths for each request in batch
let pagedAttention = false; // Enable paged attention for memory fragmentation visualization
let sequenceColors = []; // Colors for each sequence in continuous batching
// GPU configurations (per-GPU memory in GiB)
const gpuConfigs = {
    // NVIDIA
    'RTX 4090 24G':   { memGiB: 24,  label: 'RTX 4090 24G' },
    'L40S 48G':       { memGiB: 48,  label: 'L40S 48G' },
    'A100 40G':       { memGiB: 40,  label: 'A100 40G' },
    'A100 80G':       { memGiB: 80,  label: 'A100 80G' },
    'H100 80G':       { memGiB: 80,  label: 'H100 80G' },
    'H200 141G':      { memGiB: 141, label: 'H200 141G' },
    // AMD Radeon Pro (workstation)
    'AMD W7800 32G':  { memGiB: 32,  label: 'AMD W7800 32G' },
    'AMD W7900 48G':  { memGiB: 48,  label: 'AMD W7900 48G' },
    // AMD Instinct (data center)
    'AMD MI210 64G':  { memGiB: 64,  label: 'AMD MI210 64G' },
    'AMD MI250X 128G':{ memGiB: 128, label: 'AMD MI250X 128G' },
    'AMD MI300X 192G':{ memGiB: 192, label: 'AMD MI300X 192G' },
    // Intel (GPU + AI accelerators)
    'Intel Arc A770 16G':   { memGiB: 16,  label: 'Intel Arc A770 16G' },
    'Intel Max 1550 128G':  { memGiB: 128, label: 'Intel Max 1550 128G' },
    'Intel Gaudi2 96G':     { memGiB: 96,  label: 'Intel Gaudi2 96G' },
    // Google TPU (approx per-chip HBM)
    'Google TPU v3 16G':    { memGiB: 16,  label: 'Google TPU v3 16G' },
    'Google TPU v4 32G':    { memGiB: 32,  label: 'Google TPU v4 32G' },
    // Graphcore and Cerebras
    'Graphcore IPU Mk2 0.9G': { memGiB: 0.9, label: 'Graphcore IPU Mk2 0.9G' },
    'Cerebras WSE-2 40G':   { memGiB: 40,  label: 'Cerebras WSE-2 40G' },
    // Qualcomm Cloud AI
    'Qualcomm Cloud AI 100 32G': { memGiB: 32, label: 'Qualcomm Cloud AI 100 32G' }
};
let currentGPU = 'H100 80G';

function getCurrentGPUMemGiB() {
    const cfg = gpuConfigs[currentGPU];
    return cfg ? cfg.memGiB : 80;
}

// SOTA context length presets
const contextPresets = {
    '128K': 128000,     // Standard
    '200K': 200000,     // Claude 3.5
    '1M': 1000000,      // Llama 3.1
    '2M': 2000000,      // Gemini 1.5 Pro
    '10M': 10000000,    // Research/Magic
    '100M': 100000000   // Theoretical future
};

// Data type configurations
const dtypeConfigs = {
    'FP32': { bytes: 4, name: 'float32', color: '#ff6b6b' },
    'FP16': { bytes: 2, name: 'float16', color: '#00d4ff' },
    'BF16': { bytes: 2, name: 'bfloat16', color: '#00ff88' },
    'INT8': { bytes: 1, name: 'int8', color: '#ffaa00' },
    'INT4': { bytes: 0.5, name: 'int4', color: '#ff00ff' }
};

// Resize canvas
function resizeCanvas() {
    if (isMobile) {
        // On mobile, account for header and controls
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight - 250; // Leave room for header and controls
    } else {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
}


// Calculate KV cache size (from LMCache logic)
function calculateKVCacheSize(model, tokens, dtype = null) {
    const selectedDtype = dtype || currentDtype;
    const dtype_size = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2;
    let total_elements;

    if (model.special === 'deepseek') {
        // DeepSeek uses KV-LoRA compression
        total_elements = model.layers * tokens * (model.kv_lora_rank + model.qk_rope_head_dim);
    } else if (model.architecture === 'qwen3-next') {
        // Qwen3-Next uses hybrid attention (1/4 layers use traditional attention, rest use linear)
        // Only 1/4 of layers have KV cache
        const head_size = model.hidden_size / model.num_heads;
        const layers_with_kv = Math.floor(model.layers / 4);
        total_elements = 2 * layers_with_kv * tokens * model.num_kv_heads * head_size;
    } else {
        // Standard calculation (includes MOE models - expert count doesn't affect KV cache)
        const head_size = model.hidden_size / model.num_heads;
        total_elements = 2 * model.layers * tokens * model.num_kv_heads * head_size;
    }

    const total_bytes = total_elements * dtype_size;
    return total_bytes / (1024 * 1024 * 1024); // Convert to GiB
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
        '#ADFF2F'  // Green Yellow - sequence 16
    ];

    const result = [];
    for (let i = 0; i < batchSize; i++) {
        result.push(colors[i % colors.length]);
    }
    return result;
}

// Generate deterministic sequence length ratios for continuous batching
function getSequenceLengthRatio(index, batchSize) {
    // Realistic production workload distribution
    // Most requests are short-to-medium (prompts and moderate completions)
    // Few requests use full context
    // This pattern ensures continuous batching shows memory savings
    const pattern = [
        0.05,  // Very short prompt (5%)
        0.15,  // Short prompt
        0.25,  // Short-medium query
        0.35,  // Medium query
        0.10,  // Short prompt
        0.45,  // Medium generation
        0.20,  // Short query
        0.55,  // Medium-long generation
        0.30,  // Medium query
        0.65,  // Long generation
        0.40,  // Medium
        0.50,  // Average
        0.08,  // Very short
        0.70,  // Long conversation
        0.12,  // Short prompt
        0.85   // Near-max (rare)
    ];
    // Average is ~0.35, showing realistic memory savings with continuous batching
    return pattern[index % pattern.length];
}

// Calculate total KV cache for batch with continuous batching support
function calculateBatchKVCache(model, currentTokens, dtype = null) {
    // Continuous batching only makes sense with batch size > 1
    // With batch size = 1, there's no difference between continuous and traditional
    if (continuousBatching && batchSize > 1) {
        // Calculate based on variable sequence lengths with deterministic ratios
        let totalKV = 0;
        let totalSeqLength = 0;

        for (let i = 0; i < batchSize; i++) {
            // Use deterministic ratio for consistent calculations
            const ratio = getSequenceLengthRatio(i, batchSize);
            const seqLen = Math.floor(currentTokens * ratio);
            totalSeqLength += seqLen;
            totalKV += calculateKVCacheSize(model, seqLen, dtype);
        }

        // Store average for display purposes
        if (!batchSequenceLengths.length || batchSequenceLengths.length !== batchSize) {
            batchSequenceLengths = [];
            for (let i = 0; i < batchSize; i++) {
                const ratio = getSequenceLengthRatio(i, batchSize);
                batchSequenceLengths.push(Math.floor(currentTokens * ratio));
            }
        }

        return totalKV;
    } else {
        // Traditional batching: all sequences same length
        // Also used when batch size = 1 (continuous batching doesn't apply)
        batchSequenceLengths = []; // Clear any stored lengths
        return calculateKVCacheSize(model, currentTokens, dtype) * batchSize;
    }
}

// Calculate model weights memory in GiB
function calculateWeightMemoryGiB(model, dtype = null) {
    const selectedDtype = dtype || currentDtype;
    const bytesPerParam = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2;
    // For MOE models with active_params specified, use active params for inference memory
    // Otherwise use total params
    const paramsToUse = model.active_params || model.params || 0;
    // params is in Billions (e.g., 70 for 70B). Convert to number of parameters
    const numParams = paramsToUse * 1e9;
    const totalBytes = numParams * bytesPerParam;
    return totalBytes / (1024 * 1024 * 1024); // GiB
}

// Calculate GPUs needed (H100 has 80GB memory)
function calculateGPUsNeeded(memoryGiB) {
    const per = getCurrentGPUMemGiB();
    return Math.ceil(memoryGiB / per);
}

// Format memory size
function formatMemory(gib) {
    if (gib < 1) {
        return `${(gib * 1024).toFixed(1)} MiB`;
    } else if (gib < 1000) {
        return `${gib.toFixed(1)} GiB`;
    } else {
        return `${(gib / 1024).toFixed(2)} TiB`;
    }
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Create memory block particle
class MemoryBlock {
    constructor(x, y, size, color) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.color = color;
        this.opacity = 1;
        this.velocity = {
            x: (Math.random() - 0.5) * 2,
            y: -Math.random() * 3 - 1
        };
        this.life = 1;
        this.rotation = Math.random() * Math.PI * 2;
        this.rotationSpeed = (Math.random() - 0.5) * 0.1;
    }

    update() {
        this.x += this.velocity.x;
        this.y += this.velocity.y;
        this.velocity.y += 0.05; // gravity
        this.life -= 0.01;
        this.opacity = this.life;
        this.rotation += this.rotationSpeed;
    }

    draw() {
        ctx.save();
        ctx.globalAlpha = this.opacity;
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);

        // Draw glowing block
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, this.size);
        gradient.addColorStop(0, this.color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fillRect(-this.size/2, -this.size/2, this.size, this.size);

        // Draw border
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2;
        ctx.strokeRect(-this.size/2, -this.size/2, this.size, this.size);

        ctx.restore();
    }
}

// Data flow particle class
class DataFlowParticle {
    constructor(startX, startY, endX, endY, color, speed = 0.02) {
        this.startX = startX;
        this.startY = startY;
        this.endX = endX;
        this.endY = endY;
        this.x = startX;
        this.y = startY;
        this.progress = 0;
        this.speed = speed;
        this.color = color;
        this.size = 3 + Math.random() * 3;
        this.life = 1;
        this.trail = [];
        this.maxTrailLength = 10;
    }

    update() {
        this.progress += this.speed;

        // Move along path
        this.x = this.startX + (this.endX - this.startX) * this.progress;
        this.y = this.startY + (this.endY - this.startY) * this.progress;

        // Add to trail
        this.trail.push({ x: this.x, y: this.y });
        if (this.trail.length > this.maxTrailLength) {
            this.trail.shift();
        }

        // Check if reached destination
        if (this.progress >= 1) {
            this.life = 0;
        }
    }

    draw() {
        // Draw trail
        ctx.strokeStyle = this.color + '33';
        ctx.lineWidth = this.size * 0.5;
        ctx.beginPath();
        this.trail.forEach((point, i) => {
            if (i === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });
        ctx.stroke();

        // Draw particle
        ctx.globalAlpha = 0.8;
        const glow = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.size * 2);
        glow.addColorStop(0, this.color);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 2, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }
}

// Create wave effect
class Wave {
    constructor(y, amplitude, frequency, speed, color) {
        this.y = y;
        this.amplitude = amplitude;
        this.frequency = frequency;
        this.speed = speed;
        this.color = color;
        this.phase = 0;
    }

    update() {
        this.phase += this.speed;
    }

    draw() {
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();

        for (let x = 0; x < canvas.width; x += 5) {
            const y = this.y + Math.sin((x * this.frequency) + this.phase) * this.amplitude;
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();
        ctx.globalAlpha = 1;
    }
}

// Initialize waves
function initWaves() {
    waves = [];
    const model = models[currentModelIndex];

    for (let i = 0; i < 5; i++) {
        waves.push(new Wave(
            canvas.height / 2 + (i - 2) * 50,
            20 + i * 5,
            0.01 + i * 0.002,
            0.02 * animationSpeed,
            model.color + '33'
        ));
    }
}

// Draw GPU architecture visualization
function drawMemoryGrid() {
    const model = models[currentModelIndex];
    const kvGiB = calculateBatchKVCache(model, currentTokens);
    const kvMaxGiB = calculateBatchKVCache(model, maxTokens);
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const totalGiB = kvGiB + weightsGiB;

    // Skip complex rendering on mobile if performance is poor
    if (isMobile && currentTokens > 1000000) {
        return; // Skip heavy GPU rendering on mobile for performance
    }
    const totalMaxGiB = kvMaxGiB + weightsGiB;
    const fillRatio = totalMaxGiB > 0 ? (totalGiB / totalMaxGiB) : 0;

    // GPU Architecture Layout
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // GPU Die dimensions (center chip)
    const dieWidth = 280;
    const dieHeight = 280;
    const dieX = centerX - dieWidth / 2;
    const dieY = centerY - dieHeight / 2;

    // HBM Module dimensions
    const hbmWidth = 60;
    const hbmHeight = 240;
    const hbmGap = 30; // Gap between HBM and die

    // Calculate HBM positions (4 modules on each side for high-end GPUs)
    const hbmModules = [
        // Left side HBM stacks
        { x: dieX - hbmGap - hbmWidth, y: centerY - hbmHeight/2, side: 'left', index: 0 },
        { x: dieX - hbmGap - hbmWidth*2 - 10, y: centerY - hbmHeight/2, side: 'left', index: 1 },

        // Right side HBM stacks
        { x: dieX + dieWidth + hbmGap, y: centerY - hbmHeight/2, side: 'right', index: 2 },
        { x: dieX + dieWidth + hbmGap + hbmWidth + 10, y: centerY - hbmHeight/2, side: 'right', index: 3 },

        // Top HBM stacks (for very high memory configs)
        { x: centerX - hbmHeight/2, y: dieY - hbmGap - hbmWidth, side: 'top', index: 4, width: hbmHeight, height: hbmWidth },

        // Bottom HBM stacks
        { x: centerX - hbmHeight/2, y: dieY + dieHeight + hbmGap, side: 'bottom', index: 5, width: hbmHeight, height: hbmWidth }
    ];

    // Determine how many HBM modules to show based on GPU memory
    const memGiB = getCurrentGPUMemGiB();
    let activeHBMs = 4; // Default to 4 HBM stacks
    if (memGiB >= 80) activeHBMs = 6; // H100/A100 80GB have 5-6 HBM stacks
    else if (memGiB >= 40) activeHBMs = 4; // A100 40GB has 4-5 HBM stacks
    else if (memGiB >= 24) activeHBMs = 2; // Consumer GPUs have 2 memory modules

    // Draw PCB substrate
    ctx.fillStyle = 'rgba(20, 30, 45, 0.8)';
    ctx.fillRect(centerX - 450, centerY - 350, 900, 700);
    ctx.strokeStyle = 'rgba(95, 163, 230, 0.2)';
    ctx.lineWidth = 2;
    ctx.strokeRect(centerX - 450, centerY - 350, 900, 700);

    // Draw power delivery traces
    ctx.strokeStyle = 'rgba(255, 200, 0, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 20; i++) {
        ctx.beginPath();
        ctx.moveTo(centerX - 450 + i * 45, centerY - 350);
        ctx.lineTo(centerX - 450 + i * 45, centerY + 350);
        ctx.stroke();
    }

    // Generate sequence colors if continuous batching is enabled
    if (continuousBatching && batchSize > 1) {
        if (sequenceColors.length !== batchSize) {
            sequenceColors = generateSequenceColors(batchSize);
        }
    }

    // Draw HBM modules
    for (let i = 0; i < activeHBMs && i < hbmModules.length; i++) {
        const hbm = hbmModules[i];
        const w = hbm.width || hbmWidth;
        const h = hbm.height || hbmHeight;

        // Calculate fill level for this HBM module
        const hbmFillRatio = Math.min(1, Math.max(0, (fillRatio * activeHBMs - i) / 1));

        // HBM base (dark silicon)
        ctx.fillStyle = 'rgba(30, 35, 50, 0.9)';
        ctx.fillRect(hbm.x, hbm.y, w, h);

        // HBM memory banks (grid pattern)
        const bankSize = 8;
        const bankSpacing = 10;
        const banksX = Math.floor(w / bankSpacing);
        const banksY = Math.floor(h / bankSpacing);
        const totalBanks = banksX * banksY;
        const filledBanks = Math.floor(totalBanks * hbmFillRatio);

        // Calculate how many banks are used for model weights
        // Model weights are ABSOLUTE - they don't change with batching mode
        // Calculate based on the GPU's total memory capacity, not current usage
        const gpuMemGiB = getCurrentGPUMemGiB();
        const memoryPerHBM = gpuMemGiB / activeHBMs; // Memory capacity per HBM module
        const weightBanksAbsolute = includeWeights ?
            Math.floor(totalBanks * (weightsGiB / memoryPerHBM)) : 0;
        // Only show weights if they fit in the filled portion
        const weightBanks = Math.min(weightBanksAbsolute, filledBanks);

        if (continuousBatching && batchSize > 1 && !pagedAttention) {
            // Continuous batching: show different colors for each sequence
            let banksFilled = 0;
            let currentSeq = 0;

            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx;
                    const x = hbm.x + bx * bankSpacing + 2;
                    const y = hbm.y + by * bankSpacing + 2;

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8;

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - use purple/violet color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})`; // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize);

                            // Add subtle border
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)';
                            ctx.lineWidth = 0.5;
                            ctx.strokeRect(x, y, bankSize, bankSize);
                        } else {
                            // KV cache - determine which sequence this bank belongs to
                            const kvBankIndex = bankIndex - weightBanks;
                            const kvTotalBanks = filledBanks - weightBanks;
                            const seqProgress = kvBankIndex / kvTotalBanks;
                            let cumulative = 0;
                            currentSeq = 0;

                            for (let s = 0; s < batchSize; s++) {
                                const seqRatio = getSequenceLengthRatio(s, batchSize);
                                const normalizedRatio = seqRatio / batchSequenceLengths.reduce((sum, len, idx) =>
                                    sum + getSequenceLengthRatio(idx, batchSize), 0);
                                cumulative += normalizedRatio;
                                if (seqProgress <= cumulative) {
                                    currentSeq = s;
                                    break;
                                }
                            }

                            const color = sequenceColors[currentSeq % sequenceColors.length];

                            // Convert hex to RGB for manipulation
                            const r = parseInt(color.substr(1,2), 16);
                            const g = parseInt(color.substr(3,2), 16);
                            const b = parseInt(color.substr(5,2), 16);

                            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse})`;
                            ctx.fillRect(x, y, bankSize, bankSize);
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)';
                        ctx.lineWidth = 0.5;
                        ctx.strokeRect(x, y, bankSize, bankSize);
                    }
                }
            }
        } else if (pagedAttention) {
            // Paged attention: show fragmented memory with gaps
            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx;
                    const x = hbm.x + bx * bankSpacing + 2;
                    const y = hbm.y + by * bankSpacing + 2;

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8;

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - always contiguous, purple color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})`; // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize);
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)';
                            ctx.lineWidth = 0.5;
                            ctx.strokeRect(x, y, bankSize, bankSize);
                        } else {
                            // KV cache with paging
                            // Simulate fragmentation - some blocks are non-contiguous
                            const kvBankIndex = bankIndex - weightBanks;
                            const isFragmented = Math.random() < 0.15 && kvBankIndex > (filledBanks - weightBanks) * 0.3;

                            if (isFragmented) {
                                // Show as empty (fragmented)
                                ctx.strokeStyle = 'rgba(255, 100, 100, 0.3)';
                                ctx.lineWidth = 0.5;
                                ctx.strokeRect(x, y, bankSize, bankSize);
                            } else {
                                // Filled with paged blocks
                                if (continuousBatching && batchSize > 1) {
                                    // Use sequence colors with paging
                                    const kvTotalBanks = filledBanks - weightBanks;
                                    const seqIndex = Math.floor((kvBankIndex / kvTotalBanks) * batchSize);
                                    const color = sequenceColors[seqIndex % sequenceColors.length];
                                    const r = parseInt(color.substr(1,2), 16);
                                    const g = parseInt(color.substr(3,2), 16);
                                    const b = parseInt(color.substr(5,2), 16);
                                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${pulse * 0.8})`;
                                } else {
                                    // Standard paged color for KV cache
                                    ctx.fillStyle = `rgba(100, 200, 255, ${pulse})`;
                                }
                                ctx.fillRect(x, y, bankSize, bankSize);

                                // Draw page border
                                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                                ctx.lineWidth = 0.5;
                                ctx.strokeRect(x, y, bankSize, bankSize);
                            }
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)';
                        ctx.lineWidth = 0.5;
                        ctx.strokeRect(x, y, bankSize, bankSize);
                    }
                }
            }
        } else {
            // Traditional batching: uniform color for KV cache, but separate color for weights
            for (let by = 0; by < banksY; by++) {
                for (let bx = 0; bx < banksX; bx++) {
                    const bankIndex = by * banksX + bx;
                    const x = hbm.x + bx * bankSpacing + 2;
                    const y = hbm.y + by * bankSpacing + 2;

                    if (bankIndex < filledBanks) {
                        const pulse = Math.sin(Date.now() * 0.002 + bankIndex * 0.1) * 0.2 + 0.8;

                        // Check if this bank contains model weights
                        if (bankIndex < weightBanks) {
                            // Model weights - purple color
                            ctx.fillStyle = `rgba(147, 51, 234, ${pulse})`; // Purple for weights
                            ctx.fillRect(x, y, bankSize, bankSize);
                            ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)';
                            ctx.lineWidth = 0.5;
                            ctx.strokeRect(x, y, bankSize, bankSize);
                        } else {
                            // KV cache - heat gradient based on fill
                            const heat = 0.5 + hbmFillRatio * 0.5;

                            // Heat gradient based on fill
                            if (hbmFillRatio > 0.8) {
                                ctx.fillStyle = `rgba(255, ${Math.floor(100 - hbmFillRatio * 50)}, 0, ${pulse})`;
                            } else if (hbmFillRatio > 0.5) {
                                ctx.fillStyle = `rgba(255, ${Math.floor(200 - hbmFillRatio * 100)}, 0, ${pulse})`;
                            } else {
                                ctx.fillStyle = `rgba(0, ${Math.floor(200 + hbmFillRatio * 55)}, 255, ${pulse})`;
                            }
                            ctx.fillRect(x, y, bankSize, bankSize);
                        }
                    } else {
                        // Empty memory bank
                        ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)';
                        ctx.lineWidth = 0.5;
                        ctx.strokeRect(x, y, bankSize, bankSize);
                    }
                }
            }
        }

        // HBM label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('HBM', hbm.x + w/2, hbm.y - 5);

        // Draw data paths from HBM to GPU die
        if (hbmFillRatio > 0) {
            ctx.strokeStyle = model.color + '66';
            ctx.lineWidth = 2 + hbmFillRatio * 3;
            ctx.setLineDash([5, 5]);
            ctx.globalAlpha = 0.3 + hbmFillRatio * 0.4;
            ctx.beginPath();

            let startX, startY, endX, endY;
            if (hbm.side === 'left') {
                startX = hbm.x + w;
                startY = hbm.y + h/2;
                endX = dieX;
                endY = centerY;
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
            } else if (hbm.side === 'right') {
                startX = hbm.x;
                startY = hbm.y + h/2;
                endX = dieX + dieWidth;
                endY = centerY;
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
            } else if (hbm.side === 'top') {
                startX = hbm.x + w/2;
                startY = hbm.y + h;
                endX = centerX;
                endY = dieY;
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
            } else {
                startX = hbm.x + w/2;
                startY = hbm.y;
                endX = centerX;
                endY = dieY + dieHeight;
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
            }

            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;

            // Generate data flow particles for active HBM modules
            if (Math.random() < 0.1 * hbmFillRatio && isPlaying) {
                dataFlowParticles.push(new DataFlowParticle(
                    startX, startY, endX, endY,
                    model.color,
                    0.02 + Math.random() * 0.02
                ));
            }
        }
    }

    // Draw GPU Die
    // Die substrate
    const gradient = ctx.createLinearGradient(dieX, dieY, dieX + dieWidth, dieY + dieHeight);
    gradient.addColorStop(0, 'rgba(60, 70, 90, 0.95)');
    gradient.addColorStop(0.5, 'rgba(80, 90, 110, 0.95)');
    gradient.addColorStop(1, 'rgba(60, 70, 90, 0.95)');
    ctx.fillStyle = gradient;
    ctx.fillRect(dieX, dieY, dieWidth, dieHeight);

    // Die border
    ctx.strokeStyle = 'rgba(150, 200, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(dieX, dieY, dieWidth, dieHeight);

    // Draw compute units grid on the die
    const cuSize = 12;
    const cuSpacing = 16;
    const cuStartX = dieX + 20;
    const cuStartY = dieY + 20;
    const cuCols = Math.floor((dieWidth - 40) / cuSpacing);
    const cuRows = Math.floor((dieHeight - 40) / cuSpacing);

    // Calculate compute unit activity based on memory usage
    const activity = fillRatio;

    for (let row = 0; row < cuRows; row++) {
        for (let col = 0; col < cuCols; col++) {
            const x = cuStartX + col * cuSpacing;
            const y = cuStartY + row * cuSpacing;

            // Compute unit activity visualization
            const pulse = Math.sin(Date.now() * 0.003 + (row * cuCols + col) * 0.1) * 0.3 + 0.7;
            const active = Math.random() < activity;

            if (active) {
                // Active compute unit
                const heat = activity;
                if (heat > 0.8) {
                    ctx.fillStyle = `rgba(255, ${Math.floor(100 - heat * 100)}, 0, ${pulse})`;
                } else if (heat > 0.5) {
                    ctx.fillStyle = `rgba(255, 255, ${Math.floor(100 - heat * 100)}, ${pulse})`;
                } else {
                    ctx.fillStyle = `rgba(0, ${Math.floor(150 + heat * 105)}, 255, ${pulse})`;
                }
                ctx.fillRect(x, y, cuSize, cuSize);

                // Glow effect for active units
                const glow = ctx.createRadialGradient(
                    x + cuSize/2, y + cuSize/2, 0,
                    x + cuSize/2, y + cuSize/2, cuSize
                );
                glow.addColorStop(0, ctx.fillStyle);
                glow.addColorStop(1, 'transparent');
                ctx.fillStyle = glow;
                ctx.fillRect(x - 2, y - 2, cuSize + 4, cuSize + 4);
            } else {
                // Idle compute unit
                ctx.fillStyle = 'rgba(50, 80, 120, 0.3)';
                ctx.fillRect(x, y, cuSize, cuSize);
                ctx.strokeStyle = 'rgba(100, 150, 200, 0.2)';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, y, cuSize, cuSize);
            }
        }
    }

    // GPU die label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('GPU DIE', centerX, dieY - 15);
    ctx.font = '11px monospace';
    ctx.fillText(`${Math.floor(activity * 100)}% Active`, centerX, dieY - 2);

    // Draw heat sink representation (subtle)
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.2)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    for (let i = 0; i < 5; i++) {
        ctx.strokeRect(
            dieX - 10 - i * 5,
            dieY - 10 - i * 5,
            dieWidth + 20 + i * 10,
            dieHeight + 20 + i * 10
        );
    }
    ctx.setLineDash([]);

    // Add utilization indicator (positioned below bottom HBM module)
    const utilX = centerX;
    // Position well below the bottom HBM module (which is at dieY + dieHeight + hbmGap + hbmWidth)
    const bottomHBMBottom = dieY + dieHeight + hbmGap + hbmWidth;
    const utilY = bottomHBMBottom + 40; // Add spacing below HBM
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'center';

    // Show different info based on batching mode
    if (continuousBatching && batchSize > 1) {
        ctx.fillText(`KV Cache: ${formatMemory(kvGiB)} (${batchSize} variable sequences)`, utilX, utilY);
        if (includeWeights) {
            ctx.fillText(`Total: ${formatMemory(totalGiB)} (KV + ${formatMemory(weightsGiB)} weights)`, utilX, utilY + 15);
        } else {
            ctx.fillText(`Total: ${formatMemory(totalGiB)}`, utilX, utilY + 15);
        }
    } else if (batchSize > 1) {
        ctx.fillText(`KV Cache: ${formatMemory(kvGiB)} (${batchSize}×${Math.floor(currentTokens)} tokens)`, utilX, utilY);
        if (includeWeights) {
            ctx.fillText(`Total: ${formatMemory(totalGiB)} (KV + ${formatMemory(weightsGiB)} weights)`, utilX, utilY + 15);
        } else {
            ctx.fillText(`Total: ${formatMemory(totalGiB)}`, utilX, utilY + 15);
        }
    } else {
        ctx.fillText(`Memory: ${formatMemory(totalGiB)} / ${formatMemory(totalMaxGiB)}`, utilX, utilY);
        ctx.fillText(`Fill: ${(fillRatio * 100).toFixed(1)}%`, utilX, utilY + 15);
    }

    ctx.globalAlpha = 1;
}

// Draw exponential curve
function drawExponentialCurve() {
    const model = models[currentModelIndex];
    const points = [];
    const steps = 100;

    // Calculate points
    for (let i = 0; i <= steps; i++) {
        const tokens = (i / steps) * maxTokens;
        // For the curve, we'll show the average case for continuous batching
        const kvGiB = continuousBatching ? calculateBatchKVCache(model, tokens) : calculateKVCacheSize(model, tokens) * batchSize;
        const kvMaxGiB = continuousBatching ? calculateBatchKVCache(model, maxTokens) : calculateKVCacheSize(model, maxTokens) * batchSize;
        const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
        const memory = kvGiB + weightsGiB;
        const maxMemory = kvMaxGiB + weightsGiB;

        const x = (i / steps) * (canvas.width - 200) + 100;
        const y = canvas.height - 100 - (memory / maxMemory) * (canvas.height - 200);

        points.push({ x, y, tokens, memory });
    }

    // Draw curve
    ctx.strokeStyle = model.color;
    ctx.lineWidth = 3;
    ctx.beginPath();

    points.forEach((point, i) => {
        if (i === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });

    ctx.stroke();

    // Draw current position
    const currentRatio = currentTokens / maxTokens;
    const currentX = currentRatio * (canvas.width - 200) + 100;
    const kvGiBNow = calculateBatchKVCache(model, currentTokens);
    const kvGiBMax = calculateBatchKVCache(model, maxTokens);
    const weightsGiBNow = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const currentMemory = kvGiBNow + weightsGiBNow;
    const maxMemory = kvGiBMax + weightsGiBNow;
    const currentY = canvas.height - 100 - (maxMemory > 0 ? (currentMemory / maxMemory) : 0) * (canvas.height - 200);

    // Pulsing circle at current position
    const pulse = Math.sin(Date.now() * 0.003) * 5 + 10;
    ctx.beginPath();
    ctx.arc(currentX, currentY, pulse, 0, Math.PI * 2);
    ctx.fillStyle = model.color;
    ctx.fill();

    // Glow effect
    const glow = ctx.createRadialGradient(currentX, currentY, 0, currentX, currentY, pulse * 2);
    glow.addColorStop(0, model.color);
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(currentX, currentY, pulse * 2, 0, Math.PI * 2);
    ctx.fill();
}

// Generate dynamic factoids based on current state
function generateFactoids() {
    const model = models[currentModelIndex];
    const kvGiB = calculateKVCacheSize(model, currentTokens);
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const memoryPerQuery = kvGiB + weightsGiB;
    // Total memory for all concurrent queries
    const totalGiB = memoryPerQuery * batchSize;
    const gpusNeeded = calculateGPUsNeeded(totalGiB);
    const per = getCurrentGPUMemGiB();
    const efficiency = Math.min(100, (totalGiB / (gpusNeeded * per)) * 100);

    // Only include factoids with hard references
    return [
        {
            title: "📏 KV Cache Formula",
            main: `2×layers×tokens×KV heads×head_dim×bytes`,
            detail: `Reference: LMCache KV Cache Calculator — https://lmcache.ai/kv_cache_calculator.html`
        },
        {
            title: "🖥️ Device Memory Basis",
            main: `${currentGPU} → count: ${gpusNeeded}`,
            detail: `Devices needed = ceil(total GiB / per-device GiB). Selected device memory: ${getCurrentGPUMemGiB()} GiB.`
        },
        {
            title: "⚖️ Weights Memory",
            main: `Weights ≈ params × bytes/param`,
            detail: `FP16 is 2 bytes/param (IEEE 754 half). Example: 70B × 2 B ≈ 140 GiB. Ref: https://en.wikipedia.org/wiki/Half-precision_floating-point_format`
        },
        {
            title: "📊 Efficiency (definition)",
            main: `${efficiency.toFixed(1)}%`,
            detail: `Efficiency = used / allocated GPU memory. Used = Weights + KV. (Definition)`
        }
    ];
}

// Update factoid display
function updateFactoid() {
    // Skip factoid updates on mobile
    if (isMobile) return;

    const factoids = generateFactoids();
    const factoid = factoids[currentFactoidIndex % factoids.length];

    const panel = document.getElementById('factoidPanel');
    const title = document.getElementById('factoidTitle');
    const main = document.getElementById('factoidMain');
    const detail = document.getElementById('factoidDetail');

    // Fade out
    panel.style.opacity = '0';

    setTimeout(() => {
        title.textContent = factoid.title;
        main.textContent = factoid.main;
        detail.textContent = factoid.detail;

        // Fade in
        panel.style.opacity = '1';
        if (window.positionFactoidPanel) window.positionFactoidPanel();
    }, 400);

    currentFactoidIndex++;
}

// Update info panel
function updateInfoPanel() {
    const model = models[currentModelIndex];
    const kvGiB = calculateBatchKVCache(model, currentTokens);
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const totalGiB = kvGiB + weightsGiB;
    const gpusNeeded = calculateGPUsNeeded(totalGiB);

    document.getElementById('modelName').textContent = model.name;
    document.getElementById('contextLength').textContent = `${formatNumber(Math.floor(currentTokens))} tokens`;
    const weightsEl = document.getElementById('weightsSize');
    const totalEl = document.getElementById('totalSize');
    if (weightsEl) weightsEl.textContent = includeWeights ? formatMemory(weightsGiB) : '—';
    if (totalEl) {
        if (continuousBatching && batchSize > 1) {
            // Show average sequence length for continuous batching
            const avgSeqLen = batchSequenceLengths.reduce((a, b) => a + b, 0) / batchSequenceLengths.length;
            if (includeWeights) {
                totalEl.textContent = `${formatMemory(totalGiB)} (${formatMemory(weightsGiB)} weights + ${formatMemory(kvGiB)} CB-KV)`;
            } else {
                totalEl.textContent = `${formatMemory(totalGiB)} (CB: ${batchSize} reqs, avg ${Math.floor(avgSeqLen)} tok)`;
            }
        } else if (batchSize > 1) {
            const kvPerQuery = kvGiB / batchSize;
            totalEl.textContent = `${formatMemory(totalGiB)} (${formatMemory(weightsGiB)} weights + ${batchSize}×${formatMemory(kvPerQuery)} KV)`;
        } else {
            totalEl.textContent = formatMemory(totalGiB);
        }
    }
    document.getElementById('cacheSize').textContent = formatMemory(kvGiB);

    // Update GPU display to show batch processing info
    const gpuText = batchSize > 1
        ? `${gpusNeeded} (${batchSize} queries/GPU)`
        : gpusNeeded;
    document.getElementById('gpusNeeded').textContent = gpuText;
    document.getElementById('dataType').textContent = currentDtype;

    // Calculate efficiency based on GPU utilization
    // Efficiency represents how much of the allocated GPU memory is actually used
    // Low efficiency = wasted money on unused GPU memory
    const perGPU = getCurrentGPUMemGiB();
    const efficiency = Math.min(100, (totalGiB / (gpusNeeded * perGPU)) * 100);
    const efficiencyElement = document.getElementById('efficiency');
    efficiencyElement.textContent = `${efficiency.toFixed(1)}%`;

    // Color code efficiency: green (>80%), yellow (50-80%), red (<50%)
    if (efficiency > 80) {
        efficiencyElement.style.color = '#00ff88';
    } else if (efficiency > 50) {
        efficiencyElement.style.color = '#FFB800';
    } else {
        efficiencyElement.style.color = '#FF6B00';
    }

    // Show warning for multi-GPU requirement or extreme memory
    const warning = document.getElementById('warning');
    let criticalState = 'none';
    if (totalGiB > 1000) {
        criticalState = 'datacenter';
        warning.style.display = 'block';
        warning.textContent = `⚠️ ${formatMemory(totalGiB)} - Exceeds datacenter capacity!`;
        // Clarify below the memory emulation area
        const dcWrap = document.getElementById('datacenterNote');
        const dcBody = document.getElementById('datacenterNoteBody');
        if (dcWrap && dcBody) {
            dcBody.textContent = `Total KV + weights ≈ ${formatMemory(totalGiB)}. We flag > 1 TiB as beyond practical single-cluster GPU memory for this demo; real limits depend on your cluster (GPU count, memory per GPU, and interconnect bandwidth). Consider heavier quantization, KV compression/paging, or model sharding across many nodes.`;
            dcWrap.style.display = 'block';
            if (window.positionDatacenterNote) window.positionDatacenterNote();
        }
    } else if (gpusNeeded > 8) {
        criticalState = 'multi-node';
        warning.style.display = 'block';
        warning.textContent = `⚠️ Requires ${gpusNeeded} devices (${currentGPU}) - Multi-node required!`;
        const dcWrap = document.getElementById('datacenterNote');
        if (dcWrap) dcWrap.style.display = 'none';
    } else if (gpusNeeded > 1) {
        criticalState = 'multi-gpu';
        warning.style.display = 'block';
        const perGPU = getCurrentGPUMemGiB();
        warning.textContent = `⚠️ Requires ${gpusNeeded} devices (${currentGPU}, ${formatMemory(gpusNeeded * perGPU)} total)`;
        const dcWrap = document.getElementById('datacenterNote');
        if (dcWrap) dcWrap.style.display = 'none';
    } else {
        warning.style.display = 'none';
        const dcWrap = document.getElementById('datacenterNote');
        if (dcWrap) dcWrap.style.display = 'none';
    }

    // Trigger critical popup on state transition with cooldown
    if (criticalState !== 'none' && criticalState !== lastCriticalState) {
        const now = Date.now();
        if (now - lastPopupTime > POPUP_COOLDOWN_MS) {
            showCriticalPopup(criticalState, { memoryGiB: totalGiB, gpusNeeded });
            lastPopupTime = now;
        }
    }
    lastCriticalState = criticalState;

    // Update progress bar
    const progress = (currentTokens / maxTokens) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
}

// Choose a relevant factoid for the critical event
function pickRelevantFactoid(state) {
    const factoids = generateFactoids();
    // Map states to the most relevant hard-truth factoid
    if (state === 'multi-gpu' || state === 'multi-node' || state === 'datacenter') {
        return factoids.find(f => f.title.includes('Device Memory Basis')) || factoids[0];
    }
    return factoids[0];
}

// Show critical popup
function showCriticalPopup(state, metrics) {
    // Skip critical popups on mobile
    if (isMobile) return;

    const overlay = document.getElementById('criticalOverlay');
    if (!overlay) return;
    const title = document.getElementById('criticalTitle');
    const main = document.getElementById('criticalMain');
    const detail = document.getElementById('criticalDetail');

    // Title and main message by state
    if (state === 'datacenter') {
        title.textContent = 'Critical: Capacity Exceeded';
        main.textContent = `${formatMemory(metrics.memoryGiB)} total KV memory`;
        detail.textContent = 'This exceeds realistic datacenter capacity — consider aggressive compression or sharding strategies.';
    } else if (state === 'multi-node') {
        title.textContent = 'Critical: Multi-Node Required';
        main.textContent = `${metrics.gpusNeeded}× devices required (${currentGPU})`;
        detail.textContent = 'Cross-node communication will dominate latency — pipeline and bandwidth optimizations are essential.';
    } else {
        title.textContent = 'Critical: Multi-Accelerator Required';
        main.textContent = `${metrics.gpusNeeded}× devices required (${currentGPU})`;
        detail.textContent = `KV cache ≈ ${formatMemory(metrics.memoryGiB)} — exceeds single GPU capacity.`;
    }

    // Append a relevant factoid snippet
    const factoid = pickRelevantFactoid(state);
    if (factoid) {
        detail.textContent += `\n\n${factoid.title} — ${factoid.main}. ${factoid.detail}`;
    }

    overlay.style.display = 'flex';

    // Auto-close after a few seconds
    clearTimeout(showCriticalPopup._timer);
    showCriticalPopup._timer = setTimeout(() => {
        overlay.style.display = 'none';
    }, 6000);
}

// Close control
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('criticalOverlay');
    const close = document.getElementById('criticalClose');
    if (close) {
        close.addEventListener('click', () => overlay.style.display = 'none');
    }
    if (overlay) {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.style.display = 'none';
        });
    }
});

// Generate particles based on memory growth
function generateParticles() {
    const model = models[currentModelIndex];
    const memoryGiB = calculateKVCacheSize(model, currentTokens);

    // Generate more particles as memory grows
    const particleRate = Math.min(10, memoryGiB / 10);

    if (Math.random() < particleRate / 60) {
        const x = Math.random() * canvas.width;
        const y = canvas.height - 50;
        const size = 10 + Math.random() * 20;

        memoryBlocks.push(new MemoryBlock(x, y, size, model.color));
    }
}

// Animation loop
function animate() {
    try {
        if (!canvas || !ctx) return requestAnimationFrame(animate);

        ctx.fillStyle = 'rgba(15, 15, 30, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Update and draw waves
        waves.forEach(wave => {
            wave.update();
            wave.draw();
        });

        // Draw visualizations
        drawMemoryGrid();
        drawExponentialCurve();

        // Update and draw data flow particles
        dataFlowParticles = dataFlowParticles.filter(particle => particle.life > 0);
        dataFlowParticles.forEach(particle => {
            particle.update();
            particle.draw();
        });

        // Update and draw particles
        memoryBlocks = memoryBlocks.filter(block => block.life > 0);
        memoryBlocks.forEach(block => {
            block.update();
            block.draw();
        });

        // Generate new particles
        generateParticles();

        // Update token count with variable speed based on max context
        if (isPlaying) {
            const baseIncrement = Math.max(100, maxTokens / 10000);
            currentTokens += baseIncrement * animationSpeed;
            if (currentTokens >= maxTokens) {
                currentTokens = 0; // Loop
            }

            // Update sequence lengths when tokens change in continuous batching mode
            if (continuousBatching && batchSize > 1) {
                batchSequenceLengths = [];
                for (let i = 0; i < batchSize; i++) {
                    const ratio = getSequenceLengthRatio(i, batchSize);
                    batchSequenceLengths.push(Math.floor(currentTokens * ratio));
                }
            }
        }

        // Update info panel
        updateInfoPanel();

        // Update factoids every 4 seconds
        const now = Date.now();
        if (now - lastFactoidUpdate > 4000) {
            updateFactoid();
            lastFactoidUpdate = now;
        }
    } catch (e) {
        // Surface runtime errors in the warning box to aid debugging
        const warning = document.getElementById('warning');
        if (warning) {
            warning.style.display = 'block';
            warning.textContent = `⚠️ Visualization error: ${e && e.message ? e.message : e}`;
        }
    } finally {
        requestAnimationFrame(animate);
    }
}

// Control handlers
document.getElementById('playPause').addEventListener('click', function() {
    isPlaying = !isPlaying;
    this.textContent = isPlaying ? 'Pause' : 'Play';
    this.classList.toggle('active', isPlaying);
});

// Batch size control
document.getElementById('batchControl').addEventListener('click', function() {
    const batchSizes = [1, 2, 4, 8, 16, 32, 64, 128];
    const currentIndex = batchSizes.indexOf(batchSize);
    const nextIndex = (currentIndex + 1) % batchSizes.length;
    batchSize = batchSizes[nextIndex];
    this.textContent = `Batch: ${batchSize}`;

    // Regenerate sequence lengths and colors when batch size changes
    if (continuousBatching && batchSize > 1) {
        batchSequenceLengths = [];
        sequenceColors = generateSequenceColors(batchSize);
        for (let i = 0; i < batchSize; i++) {
            const ratio = getSequenceLengthRatio(i, batchSize);
            batchSequenceLengths.push(Math.floor(currentTokens * ratio));
        }
    }

    // Force update display to show new calculations
    updateInfoPanel();
});

// Continuous batching toggle
const cbBtn = document.getElementById('cbToggle');
if (cbBtn) {
    cbBtn.addEventListener('click', function() {
        continuousBatching = !continuousBatching;
        const spanEl = this.querySelector('span:first-child');
        if (spanEl) {
            spanEl.textContent = continuousBatching ? 'CB: ON' : 'CB: OFF';
        }
        this.classList.toggle('active', continuousBatching);

        // Regenerate sequence lengths when toggling
        if (continuousBatching && batchSize > 1) {
            batchSequenceLengths = [];
            sequenceColors = generateSequenceColors(batchSize);
            for (let i = 0; i < batchSize; i++) {
                const ratio = getSequenceLengthRatio(i, batchSize);
                batchSequenceLengths.push(Math.floor(currentTokens * ratio));
            }
        } else {
            batchSequenceLengths = [];
            sequenceColors = [];
        }

        updateInfoPanel();
    });
}

// Paged attention toggle
const paBtn = document.getElementById('paToggle');
if (paBtn) {
    paBtn.addEventListener('click', function() {
        pagedAttention = !pagedAttention;
        const spanEl = this.querySelector('span:first-child');
        if (spanEl) {
            spanEl.textContent = pagedAttention ? 'PA: ON' : 'PA: OFF';
        }
        this.classList.toggle('active', pagedAttention);

        updateInfoPanel();
    });
}

document.getElementById('speedControl').addEventListener('click', function() {
    const speeds = [0.5, 1, 2, 5, 10, 20, 50, 100];
    const currentIndex = speeds.indexOf(animationSpeed);
    animationSpeed = speeds[(currentIndex + 1) % speeds.length];
    this.textContent = `Speed: ${animationSpeed}x`;
});

document.getElementById('modelSwitch').addEventListener('click', function() {
    currentModelIndex = (currentModelIndex + 1) % models.length;
    currentTokens = 0;
    initWaves();
    memoryBlocks = [];
});

// Context length control
document.getElementById('contextControl').addEventListener('click', function() {
    const contexts = Object.keys(contextPresets);
    let currentContext = '1M';

    // Find current context
    for (let key of contexts) {
        if (contextPresets[key] === maxTokens) {
            currentContext = key;
            break;
        }
    }

    const currentIndex = contexts.indexOf(currentContext);
    const nextContext = contexts[(currentIndex + 1) % contexts.length];
    maxTokens = contextPresets[nextContext];
    this.textContent = `Context: ${nextContext}`;

    // Reset if current tokens exceed new max
    if (currentTokens > maxTokens) {
        currentTokens = 0;
    }
});

// Data type control
document.getElementById('dtypeControl').addEventListener('click', function() {
    const dtypes = Object.keys(dtypeConfigs);
    let currentIndex = 0;

    // Find current dtype
    for (let i = 0; i < dtypes.length; i++) {
        if (dtypes[i] === currentDtype) {
            currentIndex = i;
            break;
        }
    }

    currentDtype = dtypes[(currentIndex + 1) % dtypes.length];
    this.textContent = `Type: ${currentDtype}`;

    // Update model colors based on dtype
    const config = dtypeConfigs[currentDtype];
    models.forEach(model => {
        if (!model.originalColor) {
            model.originalColor = model.color;
        }
        // Blend model color with dtype color for visual feedback
        model.color = model.originalColor;
    });

    initWaves();
});

// Skip Model Weights (SMW) toggle
const smwBtn = document.getElementById('smwToggle');
if (smwBtn) {
    smwBtn.addEventListener('click', function() {
        includeWeights = !includeWeights;
        this.classList.toggle('active', !includeWeights); // active means skipping weights
        updateInfoPanel();
    });
}

// GPU selection control
const gpuBtn = document.getElementById('gpuControl');
if (gpuBtn) {
    gpuBtn.addEventListener('click', function() {
        const keys = Object.keys(gpuConfigs);
        const idx = keys.indexOf(currentGPU);
        currentGPU = keys[(idx + 1) % keys.length];
        this.textContent = `Device: ${currentGPU}`;
        updateInfoPanel();
    });
}

// Initialize
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
initWaves();

// Sync initial control states
document.addEventListener('DOMContentLoaded', () => {
    const playBtn = document.getElementById('playPause');
    if (playBtn) {
        playBtn.textContent = isPlaying ? 'Pause' : 'Play';
        playBtn.classList.toggle('active', isPlaying);
    }
    const speedBtn = document.getElementById('speedControl');
    if (speedBtn) speedBtn.textContent = `Speed: ${animationSpeed}x`;
    const batchBtn = document.getElementById('batchControl');
    if (batchBtn) batchBtn.textContent = `Batch: ${batchSize}`;
    const dtypeBtn = document.getElementById('dtypeControl');
    if (dtypeBtn) dtypeBtn.textContent = `Type: ${currentDtype}`;
    const gpuBtn2 = document.getElementById('gpuControl');
    if (gpuBtn2) gpuBtn2.textContent = `Device: ${currentGPU}`;
});

// Initialize first factoid
setTimeout(() => {
    updateFactoid();
    lastFactoidUpdate = Date.now();
}, 100);

animate();
