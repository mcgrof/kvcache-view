// Training Memory Visualization
// Shows memory consumption during LLM training: weights, gradients, optimizer states, and activations

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');

// Set canvas size
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Training state
let isTraining = false;
let currentStep = 0;
let maxSteps = 100000;
let trainingSpeed = 1;
let currentLoss = 4.5; // Starting loss
let lossHistory = [];

// Model configurations for training
const models = [
    { name: 'Llama-3.2-1B', params: 1.2, layers: 16, hidden: 2048, heads: 32 },
    { name: 'Phi-3.5-mini', params: 3.8, layers: 32, hidden: 3072, heads: 32 },
    { name: 'Llama-3.1-8B', params: 8, layers: 32, hidden: 4096, heads: 32 },
    { name: 'Mistral-7B', params: 7, layers: 32, hidden: 4096, heads: 32 },
    { name: 'Llama-3.1-70B', params: 70, layers: 80, hidden: 8192, heads: 64 },
    { name: 'Llama-3.1-405B', params: 405, layers: 126, hidden: 16384, heads: 128 }
];

let currentModelIndex = 0; // Start with Llama-1B - fits in memory
let batchSize = 2;  // Smaller batch to start
let sequenceLength = 1024;  // Shorter sequence to start
let accumulationSteps = 1;

// Optimizer configurations
const optimizers = {
    'SGD': { memoryFactor: 0 },      // No additional memory
    'Adam': { memoryFactor: 2 },      // 2x model size for momentum + variance
    'AdamW': { memoryFactor: 2 },     // Same as Adam
    'Lion': { memoryFactor: 1 },      // 1x model size for momentum only
    'AdaFactor': { memoryFactor: 0.5 } // Factorized second moments
};
let currentOptimizer = 'Adam';

// Training optimizations - start with some enabled to show their benefit
let gradientCheckpointing = true;  // Enabled by default to reduce memory
let mixedPrecision = true;  // Enabled by default - standard practice
let zeroOptimization = 0; // 0=off, 1=ZeRO-1, 2=ZeRO-2, 3=ZeRO-3
let fullyShardedDataParallel = false;
let gradientAccumulation = false;

// GPU configurations
const gpuConfigs = {
    'RTX 4090 24G': { memory: 24, bandwidth: 1008, compute: 82.6 },
    'A100 40G': { memory: 40, bandwidth: 1555, compute: 19.5 },
    'A100 80G': { memory: 80, bandwidth: 2039, compute: 19.5 },
    'H100 80G': { memory: 80, bandwidth: 3350, compute: 67 },
    'H200 141G': { memory: 141, bandwidth: 4800, compute: 67 },
    'MI300X 192G': { memory: 192, bandwidth: 5300, compute: 163 }
};
let currentGPU = 'H100 80G';

// Memory calculation functions
function calculateModelWeightsMemory(model, dtype = 'fp32') {
    const bytesPerParam = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4;
    return (model.params * 1e9 * bytesPerParam) / (1024 ** 3); // GiB
}

function calculateGradientsMemory(model, dtype = 'fp32') {
    // Gradients are same size as weights
    const bytesPerParam = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4;
    return (model.params * 1e9 * bytesPerParam) / (1024 ** 3); // GiB
}

function calculateOptimizerMemory(model, optimizer, dtype = 'fp32') {
    const baseWeightsMemory = calculateModelWeightsMemory(model, dtype);
    const optimizerConfig = optimizers[optimizer];

    // ZeRO optimization reduces optimizer memory
    let reductionFactor = 1;
    if (zeroOptimization === 1) reductionFactor = 0.5; // ZeRO-1: shard optimizer states
    if (zeroOptimization === 2) reductionFactor = 0.25; // ZeRO-2: shard optimizer + gradients
    if (zeroOptimization === 3) reductionFactor = 0.125; // ZeRO-3: shard everything

    return baseWeightsMemory * optimizerConfig.memoryFactor * reductionFactor;
}

function calculateActivationsMemory(model, batchSize, sequenceLength, dtype = 'fp32') {
    // Rough estimate: batch * seq * layers * hidden * factor
    const bytesPerValue = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4;
    let factor = 10; // Approximate factor for all intermediate activations

    if (gradientCheckpointing) {
        factor = 2; // Only store checkpoint activations
    }

    const activationBytes = batchSize * sequenceLength * model.layers * model.hidden * factor * bytesPerValue;
    return activationBytes / (1024 ** 3); // GiB
}

function getTotalTrainingMemory() {
    const model = models[currentModelIndex];
    const dtype = mixedPrecision ? 'fp16' : 'fp32';

    let weightsMemory = calculateModelWeightsMemory(model, dtype);
    let gradientsMemory = calculateGradientsMemory(model, dtype);
    let optimizerMemory = calculateOptimizerMemory(model, currentOptimizer, dtype);
    let activationsMemory = calculateActivationsMemory(model, batchSize, sequenceLength, dtype);

    // Mixed precision requires FP32 master weights
    if (mixedPrecision) {
        weightsMemory += calculateModelWeightsMemory(model, 'fp32'); // Add master weights
    }

    // FSDP shards everything across GPUs
    if (fullyShardedDataParallel) {
        const numGPUs = 8; // Assume 8-GPU node
        weightsMemory /= numGPUs;
        gradientsMemory /= numGPUs;
        optimizerMemory /= numGPUs;
    }

    return {
        weights: weightsMemory,
        gradients: gradientsMemory,
        optimizer: optimizerMemory,
        activations: activationsMemory,
        total: weightsMemory + gradientsMemory + optimizerMemory + activationsMemory
    };
}

// Calculate learning rate with schedule
function getLearningRate(step) {
    const warmupSteps = 1000;
    const baseLR = 5e-5;

    if (step < warmupSteps) {
        // Linear warmup
        return baseLR * (step / warmupSteps);
    } else {
        // Cosine decay
        const progress = (step - warmupSteps) / (maxSteps - warmupSteps);
        return baseLR * 0.5 * (1 + Math.cos(Math.PI * progress));
    }
}

// Draw GPU memory visualization
function drawGPUMemory() {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const gpuWidth = 400;
    const gpuHeight = 300;
    const gpuX = centerX - gpuWidth / 2;
    const gpuY = centerY - gpuHeight / 2;

    // GPU background
    const gradient = ctx.createLinearGradient(gpuX, gpuY, gpuX, gpuY + gpuHeight);
    gradient.addColorStop(0, 'rgba(30, 15, 15, 0.9)');
    gradient.addColorStop(1, 'rgba(50, 20, 20, 0.9)');
    ctx.fillStyle = gradient;
    ctx.fillRect(gpuX, gpuY, gpuWidth, gpuHeight);

    // GPU border
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(gpuX, gpuY, gpuWidth, gpuHeight);

    // Memory segments visualization
    const memory = getTotalTrainingMemory();
    const gpuMemory = gpuConfigs[currentGPU].memory;
    const memoryUsage = memory.total / gpuMemory;

    // Draw memory blocks
    const blockSize = 20;
    const blocksX = Math.floor(gpuWidth / (blockSize + 2));
    const blocksY = Math.floor(gpuHeight / (blockSize + 2));
    const totalBlocks = blocksX * blocksY;

    const weightsBlocks = Math.floor(totalBlocks * (memory.weights / memory.total));
    const gradientsBlocks = Math.floor(totalBlocks * (memory.gradients / memory.total));
    const optimizerBlocks = Math.floor(totalBlocks * (memory.optimizer / memory.total));
    const activationsBlocks = Math.floor(totalBlocks * (memory.activations / memory.total));

    let blockIndex = 0;
    for (let y = 0; y < blocksY; y++) {
        for (let x = 0; x < blocksX; x++) {
            const blockX = gpuX + x * (blockSize + 2) + 2;
            const blockY = gpuY + y * (blockSize + 2) + 2;

            if (blockIndex < weightsBlocks) {
                ctx.fillStyle = 'rgba(156, 39, 176, 0.8)'; // Purple for weights
            } else if (blockIndex < weightsBlocks + gradientsBlocks) {
                ctx.fillStyle = 'rgba(255, 152, 0, 0.8)'; // Orange for gradients
            } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks) {
                ctx.fillStyle = 'rgba(244, 67, 54, 0.8)'; // Red for optimizer
            } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks + activationsBlocks) {
                ctx.fillStyle = 'rgba(76, 175, 80, 0.8)'; // Green for activations
            } else {
                ctx.fillStyle = 'rgba(50, 50, 50, 0.3)'; // Empty
            }

            ctx.fillRect(blockX, blockY, blockSize, blockSize);
            blockIndex++;
        }
    }

    // Draw utilization indicator
    const utilX = centerX;
    const utilY = gpuY + gpuHeight + 40;
    ctx.fillStyle = memoryUsage > 0.9 ? '#FF4444' : memoryUsage > 0.7 ? '#FFA500' : '#4CAF50';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`GPU Memory: ${(memoryUsage * 100).toFixed(1)}%`, utilX, utilY);

    // Training progress bar
    const progressWidth = 300;
    const progressHeight = 10;
    const progressX = centerX - progressWidth / 2;
    const progressY = utilY + 20;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(progressX, progressY, progressWidth, progressHeight);

    const progress = currentStep / maxSteps;
    ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
    ctx.fillRect(progressX, progressY, progressWidth * progress, progressHeight);

    // Step counter
    ctx.fillStyle = '#EAF2FF';
    ctx.font = '12px monospace';
    ctx.fillText(`Step ${currentStep.toLocaleString()} / ${maxSteps.toLocaleString()}`, centerX, progressY + 25);
}

// Draw training loss curve
function drawLossCurve() {
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

    if (lossHistory.length < 2) return;

    // Draw axes
    lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    lossCtx.lineWidth = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(20, 10);
    lossCtx.lineTo(20, 90);
    lossCtx.lineTo(240, 90);
    lossCtx.stroke();

    // Draw loss curve
    lossCtx.strokeStyle = '#4CAF50';
    lossCtx.lineWidth = 2;
    lossCtx.beginPath();

    const xScale = 220 / Math.max(100, lossHistory.length);
    const yScale = 70 / 5; // Assuming max loss of 5

    lossHistory.forEach((loss, i) => {
        const x = 20 + i * xScale;
        const y = 90 - loss * yScale;

        if (i === 0) {
            lossCtx.moveTo(x, y);
        } else {
            lossCtx.lineTo(x, y);
        }
    });

    lossCtx.stroke();

    // Current loss value
    lossCtx.fillStyle = '#EAF2FF';
    lossCtx.font = '12px monospace';
    lossCtx.fillText(`Loss: ${currentLoss.toFixed(3)}`, 160, 20);
}

// Update UI elements
function updateUI() {
    const model = models[currentModelIndex];
    const memory = getTotalTrainingMemory();

    // Update info panel
    document.getElementById('modelName').textContent = model.name;
    document.getElementById('paramCount').textContent = model.params + 'B';
    document.getElementById('batchInfo').textContent = batchSize + (gradientAccumulation ? ` (×${accumulationSteps})` : '');
    document.getElementById('seqLength').textContent = sequenceLength.toLocaleString();
    document.getElementById('learningRate').textContent = getLearningRate(currentStep).toExponential(1);
    document.getElementById('stepCount').textContent = `${currentStep.toLocaleString()} / ${(maxSteps/1000).toFixed(0)}K`;
    document.getElementById('totalMem').textContent = memory.total.toFixed(1) + ' GiB';

    // GPU count
    const gpusNeeded = Math.ceil(memory.total / gpuConfigs[currentGPU].memory);
    document.getElementById('gpuCount').textContent = `${gpusNeeded}× ${currentGPU}`;

    // Memory breakdown
    document.getElementById('weightsSize').textContent = memory.weights.toFixed(1) + ' GiB';
    document.getElementById('gradientsSize').textContent = memory.gradients.toFixed(1) + ' GiB';
    document.getElementById('optimizerSize').textContent = memory.optimizer.toFixed(1) + ' GiB';
    document.getElementById('activationsSize').textContent = memory.activations.toFixed(1) + ' GiB';

    // Update memory bars
    const maxMem = memory.total;
    document.getElementById('weightsBar').style.width = (memory.weights / maxMem * 100) + '%';
    document.getElementById('gradientsBar').style.width = (memory.gradients / maxMem * 100) + '%';
    document.getElementById('optimizerBar').style.width = (memory.optimizer / maxMem * 100) + '%';
    document.getElementById('activationsBar').style.width = (memory.activations / maxMem * 100) + '%';

    // Show warning if memory exceeds GPU
    const warning = document.getElementById('warning');
    if (memory.total > gpuConfigs[currentGPU].memory) {
        warning.style.display = 'block';
        warning.textContent = `⚠️ Out of Memory! Requires ${gpusNeeded} GPUs or enable optimizations`;
    } else if (memory.total > gpuConfigs[currentGPU].memory * 0.9) {
        warning.style.display = 'block';
        warning.textContent = `⚠️ Warning: Near memory limit (${(memory.total / gpuConfigs[currentGPU].memory * 100).toFixed(0)}% used)`;
    } else {
        warning.style.display = 'none';
    }
}

// Animation loop
function animate() {
    ctx.fillStyle = 'rgba(10, 10, 10, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Training step progress
    if (isTraining) {
        currentStep += trainingSpeed * 10;
        if (currentStep >= maxSteps) {
            currentStep = maxSteps;
            isTraining = false;
            document.getElementById('playPause').textContent = '✓ Training Complete';
        }

        // Simulate loss decay
        const targetLoss = 0.1;
        currentLoss = currentLoss * 0.999 + targetLoss * 0.001;

        // Record loss history
        if (currentStep % 100 === 0) {
            lossHistory.push(currentLoss);
            if (lossHistory.length > 100) {
                lossHistory.shift();
            }
        }
    }

    drawGPUMemory();
    drawLossCurve();
    updateUI();

    requestAnimationFrame(animate);
}

// Event listeners
document.getElementById('playPause').addEventListener('click', function() {
    isTraining = !isTraining;
    this.textContent = isTraining ? '⏸️ Pause Training' : '▶️ Resume Training';
});

document.getElementById('speedControl').addEventListener('click', function() {
    const speeds = [0.5, 1, 2, 5, 10, 50, 100];
    const currentIndex = speeds.indexOf(trainingSpeed);
    trainingSpeed = speeds[(currentIndex + 1) % speeds.length];
    this.textContent = `Speed: ${trainingSpeed}x`;
});

document.getElementById('modelControl').addEventListener('click', function() {
    currentModelIndex = (currentModelIndex + 1) % models.length;
    const model = models[currentModelIndex];
    this.textContent = `Model: ${model.name.split('-')[0]}-${model.params}B`;
    currentStep = 0; // Reset training
    lossHistory = [];
    currentLoss = 4.5;
});

document.getElementById('batchControl').addEventListener('click', function() {
    const batches = [1, 2, 4, 8, 16, 32, 64];
    const currentIndex = batches.indexOf(batchSize);
    batchSize = batches[(currentIndex + 1) % batches.length];
    this.textContent = `Batch: ${batchSize}`;
});

document.getElementById('seqControl').addEventListener('click', function() {
    const sequences = [512, 1024, 2048, 4096, 8192, 16384];
    const currentIndex = sequences.indexOf(sequenceLength);
    sequenceLength = sequences[(currentIndex + 1) % sequences.length];
    this.textContent = `Seq: ${sequenceLength}`;
});

document.getElementById('optimizerControl').addEventListener('click', function() {
    const opts = Object.keys(optimizers);
    const currentIndex = opts.indexOf(currentOptimizer);
    currentOptimizer = opts[(currentIndex + 1) % opts.length];
    this.textContent = `Optimizer: ${currentOptimizer}`;
});

// Training optimization toggles
document.getElementById('gcToggle').addEventListener('click', function() {
    gradientCheckpointing = !gradientCheckpointing;
    this.querySelector('span').textContent = `GC: ${gradientCheckpointing ? 'ON' : 'OFF'}`;
    this.classList.toggle('enabled', gradientCheckpointing);
});

document.getElementById('mpToggle').addEventListener('click', function() {
    mixedPrecision = !mixedPrecision;
    this.querySelector('span').textContent = `MP: ${mixedPrecision ? 'ON' : 'OFF'}`;
    this.classList.toggle('enabled', mixedPrecision);
});

document.getElementById('zeroToggle').addEventListener('click', function() {
    zeroOptimization = (zeroOptimization + 1) % 4;
    const labels = ['OFF', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3'];
    this.querySelector('span').textContent = labels[zeroOptimization];
    this.classList.toggle('enabled', zeroOptimization > 0);
});

document.getElementById('fsdpToggle').addEventListener('click', function() {
    fullyShardedDataParallel = !fullyShardedDataParallel;
    this.querySelector('span').textContent = `FSDP: ${fullyShardedDataParallel ? 'ON' : 'OFF'}`;
    this.classList.toggle('enabled', fullyShardedDataParallel);
});

document.getElementById('gaToggle').addEventListener('click', function() {
    gradientAccumulation = !gradientAccumulation;
    if (gradientAccumulation) {
        accumulationSteps = 4;
    } else {
        accumulationSteps = 1;
    }
    this.querySelector('span').textContent = `GA: ${gradientAccumulation ? 'ON' : 'OFF'}`;
    this.classList.toggle('enabled', gradientAccumulation);
});

document.getElementById('gpuControl').addEventListener('click', function() {
    const gpus = Object.keys(gpuConfigs);
    const currentIndex = gpus.indexOf(currentGPU);
    currentGPU = gpus[(currentIndex + 1) % gpus.length];
    this.textContent = `GPU: ${currentGPU}`;
});

// Start animation
animate();