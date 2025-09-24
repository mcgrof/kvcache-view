// KV Cache Growth Visualization
// Memory calculations based on LMCache KV Cache Calculator
// Source: https://lmcache.ai/kv_cache_calculator.html
// This visualization uses the exact formulas from LMCache's calculator
// to accurately compute KV cache memory requirements

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Check if mobile
let isMobile = window.matchMedia('(max-width: 768px)').matches;

// Update mobile check on resize
window.addEventListener('resize', () => {
    isMobile = window.matchMedia('(max-width: 768px)').matches;
});

// Model configurations from LMCache
// Samsung brand colors
const models = [
    {
        name: "Llama-3.2-1B",
        params: 1.2,
        layers: 16,
        hidden_size: 2048,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#5FA3E6', // Samsung Light Blue
        efficiency: 'high'
    },
    {
        name: "Llama-3.1-8B",
        params: 8,
        layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        color: '#1428A0', // Samsung Blue
        efficiency: 'medium'
    },
    {
        name: "Llama-3.1-70B",
        params: 70,
        layers: 80,
        hidden_size: 8192,
        num_heads: 64,
        num_kv_heads: 8,
        color: '#691FFF', // Samsung Purple
        efficiency: 'low'
    },
    {
        name: "DeepSeek-V3 (671B)",
        params: 671,
        layers: 61,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        color: '#FF6B00', // Samsung Orange
        efficiency: 'optimized',
        special: 'deepseek'
    },
    {
        name: "Llama-3.1-405B",
        params: 405,
        layers: 126,
        hidden_size: 16384,
        num_heads: 128,
        num_kv_heads: 8,
        color: '#E4002B', // Samsung Red
        efficiency: 'very-low'
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

// Auto-play on mobile with slower speed
setTimeout(() => {
    if (isMobile && !isPlaying) {
        isPlaying = true;
        animationSpeed = 10; // Slower on mobile
    }
}, 1000);
let lastFactoidUpdate = 0;
let lastCriticalState = 'none';
let lastPopupTime = 0;
const POPUP_COOLDOWN_MS = 10000;
let includeWeights = true; // Include model weights memory by default
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
    // Full screen on all devices
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

// Calculate KV cache size (from LMCache logic)
function calculateKVCacheSize(model, tokens, dtype = null) {
    const selectedDtype = dtype || currentDtype;
    const dtype_size = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2;
    let total_elements;

    if (model.special === 'deepseek') {
        // DeepSeek uses KV-LoRA compression
        total_elements = model.layers * tokens * (model.kv_lora_rank + model.qk_rope_head_dim);
    } else {
        // Standard calculation
        const head_size = model.hidden_size / model.num_heads;
        total_elements = 2 * model.layers * tokens * model.num_kv_heads * head_size;
    }

    const total_bytes = total_elements * dtype_size;
    return total_bytes / (1024 * 1024 * 1024); // Convert to GiB
}

// Calculate model weights memory in GiB
function calculateWeightMemoryGiB(model, dtype = null) {
    const selectedDtype = dtype || currentDtype;
    const bytesPerParam = dtypeConfigs[selectedDtype] ? dtypeConfigs[selectedDtype].bytes : 2;
    // params is in Billions (e.g., 70 for 70B). Convert to number of parameters
    const numParams = (model.params || 0) * 1e9;
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
        ctx.lineWidth = isMobile ? 2 : 3;
        ctx.globalAlpha = isMobile ? 0.3 : 0.6;  // More subtle on mobile
        ctx.beginPath();

        const step = isMobile ? 10 : 5;  // Less detailed on mobile for performance
        for (let x = 0; x < canvas.width; x += step) {
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

    // Fewer waves on mobile for performance
    const waveCount = isMobile ? 3 : 5;
    const waveSpacing = isMobile ? 80 : 50;

    for (let i = 0; i < waveCount; i++) {
        waves.push(new Wave(
            canvas.height / 2 + (i - Math.floor(waveCount/2)) * waveSpacing,
            20 + i * 5,
            0.01 + i * 0.002,
            0.02 * animationSpeed,
            model.color + (isMobile ? '22' : '33')
        ));
    }
}

// Draw memory grid visualization
function drawMemoryGrid() {
    const model = models[currentModelIndex];
    const kvGiB = calculateKVCacheSize(model, currentTokens);
    const kvMaxGiB = calculateKVCacheSize(model, maxTokens);
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const totalGiB = kvGiB + weightsGiB;

    // Always use simplified rendering on mobile
    const mobileSimplified = isMobile;
    const totalMaxGiB = kvMaxGiB + weightsGiB;
    const fillRatio = totalMaxGiB > 0 ? (totalGiB / totalMaxGiB) : 0;

    // Grid parameters - adjust for mobile
    const gridSize = isMobile ? 12 : 20;
    const spacing = isMobile ? 15 : 25;
    const gridWidth = isMobile ? 20 : 24;
    const gridHeight = isMobile ? 24 : 16;
    const startX = canvas.width / 2 - (gridWidth * spacing) / 2;
    const startY = canvas.height / 2 - (gridHeight * spacing) / 2;

    const totalCells = gridWidth * gridHeight;
    const filledCells = Math.floor(totalCells * fillRatio);

    // Clean mobile visualization
    if (mobileSimplified) {
        // Clear background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw a clean horizontal bar
        const barWidth = canvas.width * 0.85;
        const barHeight = 40;
        const barX = (canvas.width - barWidth) / 2;
        const barY = canvas.height / 2;

        // Background track
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Memory fill
        const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth * fillRatio, 0);
        gradient.addColorStop(0, model.color);
        gradient.addColorStop(1, model.color + 'AA');
        ctx.fillStyle = gradient;
        ctx.fillRect(barX, barY, barWidth * fillRatio, barHeight);

        // Border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barWidth, barHeight);

        // Token count below bar
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${(currentTokens / 1000000).toFixed(1)}M / ${(maxTokens / 1000000).toFixed(1)}M tokens`, canvas.width / 2, barY + barHeight + 20);

        return;
    }

    // Draw grid cells
    for (let i = 0; i < gridHeight; i++) {
        for (let j = 0; j < gridWidth; j++) {
            const cellIndex = i * gridWidth + j;
            const x = startX + j * spacing;
            const y = startY + i * spacing;

            if (cellIndex < filledCells) {
                // Filled cell with pulsing effect
                const pulse = Math.sin(Date.now() * 0.001 + cellIndex * 0.1) * 0.3 + 0.7;
                ctx.fillStyle = model.color;
                ctx.globalAlpha = pulse;
                ctx.fillRect(x, y, gridSize, gridSize);

                // Glow effect
                const glow = ctx.createRadialGradient(
                    x + gridSize/2, y + gridSize/2, 0,
                    x + gridSize/2, y + gridSize/2, gridSize
                );
                glow.addColorStop(0, model.color);
                glow.addColorStop(1, 'transparent');
                ctx.fillStyle = glow;
                ctx.fillRect(x - 5, y - 5, gridSize + 10, gridSize + 10);
            } else {
                // Empty cell
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, gridSize, gridSize);
            }
        }
    }

    ctx.globalAlpha = 1;
}

// Draw exponential curve
function drawExponentialCurve() {
    // Skip on mobile for cleaner view
    if (isMobile) return;

    const model = models[currentModelIndex];
    const points = [];
    const steps = 100;

    // Calculate points
    for (let i = 0; i <= steps; i++) {
        const tokens = (i / steps) * maxTokens;
        const kvGiB = calculateKVCacheSize(model, tokens);
        const kvMaxGiB = calculateKVCacheSize(model, maxTokens);
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
    const kvGiBNow = calculateKVCacheSize(model, currentTokens);
    const kvGiBMax = calculateKVCacheSize(model, maxTokens);
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
    const totalGiB = kvGiB + weightsGiB;
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
        if (window.positionControls) window.positionControls();
    }, 400);

    currentFactoidIndex++;
}

// Update info panel
function updateInfoPanel() {
    const model = models[currentModelIndex];
    const kvGiB = calculateKVCacheSize(model, currentTokens);
    const weightsGiB = includeWeights ? calculateWeightMemoryGiB(model) : 0;
    const totalGiB = kvGiB + weightsGiB;
    const gpusNeeded = calculateGPUsNeeded(totalGiB);

    document.getElementById('modelName').textContent = model.name;
    document.getElementById('contextLength').textContent = `${formatNumber(Math.floor(currentTokens))} tokens`;
    const weightsEl = document.getElementById('weightsSize');
    const totalEl = document.getElementById('totalSize');
    if (weightsEl) weightsEl.textContent = includeWeights ? formatMemory(weightsGiB) : '—';
    if (totalEl) totalEl.textContent = formatMemory(totalGiB);
    document.getElementById('cacheSize').textContent = formatMemory(kvGiB);
    document.getElementById('gpusNeeded').textContent = gpusNeeded;
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
