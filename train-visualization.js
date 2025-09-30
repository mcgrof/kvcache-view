// Training Memory Visualization
// Shows memory consumption during LLM training: weights, gradients, optimizer states, and activations

const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
const lossCanvas = document.getElementById('lossCanvas')
const lossCtx = lossCanvas.getContext('2d')

// Set canvas size
function resizeCanvas() {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
}
resizeCanvas()
window.addEventListener('resize', resizeCanvas)

// Training state
let isTraining = false
let currentStep = 0
let maxSteps = 100000
let trainingSpeed = 50
let currentLoss = 4.5 // Starting loss
let lossHistory = []

// Model configurations for training
const models = [
    { name: 'Llama-3.2-1B', params: 1.2, layers: 16, hidden: 2048, heads: 32 },
    { name: 'Phi-3.5-mini', params: 3.8, layers: 32, hidden: 3072, heads: 32 },
    { name: 'Llama-3.1-8B', params: 8, layers: 32, hidden: 4096, heads: 32 },
    { name: 'Mistral-7B', params: 7, layers: 32, hidden: 4096, heads: 32 },
    { name: 'Llama-3.1-70B', params: 70, layers: 80, hidden: 8192, heads: 64 },
    { name: 'Llama-3.1-405B', params: 405, layers: 126, hidden: 16384, heads: 128 },
]

let currentModelIndex = 0 // Start with Llama-1B - fits in memory
let batchSize = 2 // Smaller batch to start
let sequenceLength = 1024 // Shorter sequence to start
let accumulationSteps = 1

// Optimizer configurations
const optimizers = {
    SGD: { memoryFactor: 0 }, // No additional memory
    Adam: { memoryFactor: 2 }, // 2x model size for momentum + variance
    AdamW: { memoryFactor: 2 }, // Same as Adam
    Lion: { memoryFactor: 1 }, // 1x model size for momentum only
    AdaFactor: { memoryFactor: 0.5 }, // Factorized second moments
}
let currentOptimizer = 'Adam'

// Training optimizations - start with some enabled to show their benefit
let gradientCheckpointing = true // Enabled by default to reduce memory
let mixedPrecision = true // Enabled by default - standard practice
let zeroOptimization = 0 // 0=off, 1=ZeRO-1, 2=ZeRO-2, 3=ZeRO-3
let fullyShardedDataParallel = false
let gradientAccumulation = false

// GPU configurations
const gpuConfigs = {
    'RTX 4090 24G': { memory: 24, bandwidth: 1008, compute: 82.6 },
    'A100 40G': { memory: 40, bandwidth: 1555, compute: 19.5 },
    'A100 80G': { memory: 80, bandwidth: 2039, compute: 19.5 },
    'H100 80G': { memory: 80, bandwidth: 3350, compute: 67 },
    'H200 141G': { memory: 141, bandwidth: 4800, compute: 67 },
    'MI300X 192G': { memory: 192, bandwidth: 5300, compute: 163 },
}
let currentGPU = 'H100 80G'
let gpuCount = 1  // Number of GPUs for distributed training
const validGPUCounts = [1, 2, 4, 8, 16, 32, 64, 128]

// Famous training datacenter configurations
const worldDatacenters = {
    none: { name: 'None', gpus: null, gpu: null, model: null, batch: null, seq: null, optimizer: null },
    dgx_h100: { name: 'DGX H100', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 2, seq: 2048, optimizer: 'Adam' },
    dgx_pod: { name: 'DGX SuperPOD', gpus: 32, gpu: 'H100 80G', model: 'Llama-3.1-70B', batch: 1, seq: 2048, optimizer: 'AdamW' },
    meta_rsc: { name: 'Meta Training', gpus: 128, gpu: 'A100 80G', model: 'Llama-3.1-70B', batch: 1, seq: 2048, optimizer: 'AdamW' },
    openai_gpt: { name: 'OpenAI GPT-4', gpus: 64, gpu: 'A100 40G', model: 'Llama-3.1-8B', batch: 1, seq: 1024, optimizer: 'Adam' },
    aws_p5: { name: 'AWS P5 Train', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 4, seq: 2048, optimizer: 'Adam' },
    gcp_tpu: { name: 'GCP TPU v5e', gpus: 8, gpu: 'TPU v5e 16G', model: 'Llama-3.2-1B', batch: 8, seq: 512, optimizer: 'AdaFactor' },
    azure_nd: { name: 'Azure ND A100', gpus: 8, gpu: 'A100 40G', model: 'Mistral-7B', batch: 2, seq: 1024, optimizer: 'Adam' },
    lambda_train: { name: 'Lambda Train', gpus: 8, gpu: 'A100 80G', model: 'Llama-3.1-8B', batch: 4, seq: 2048, optimizer: 'Lion' },
    single_gpu: { name: 'Single GPU', gpus: 1, gpu: 'H100 80G', model: 'Llama-3.2-1B', batch: 8, seq: 2048, optimizer: 'AdamW' },
    anthropic_claude: { name: 'Anthropic', gpus: 16, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 2, seq: 4096, optimizer: 'AdamW' },
    stability_sd: { name: 'Stability AI', gpus: 8, gpu: 'A100 80G', model: 'Mistral-7B', batch: 4, seq: 1024, optimizer: 'Adam' },
    cohere_train: { name: 'Cohere Train', gpus: 16, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 2, seq: 2048, optimizer: 'AdamW' },
}
let currentDatacenter = 'none'

// Memory calculation functions
function calculateModelWeightsMemory(model, dtype = 'fp32') {
    const bytesPerParam = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4
    return (model.params * 1e9 * bytesPerParam) / 1024 ** 3 // GiB
}

function calculateGradientsMemory(model, dtype = 'fp32') {
    // Gradients are same size as weights
    const bytesPerParam = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4
    // Gradients accumulate during forward pass, clear after backward
    const accumulation = isTraining ? 0.8 + Math.sin(Date.now() * 0.002) * 0.2 : 1
    return (model.params * 1e9 * bytesPerParam * accumulation) / 1024 ** 3 // GiB
}

function calculateOptimizerMemory(model, optimizer, dtype = 'fp32') {
    const baseWeightsMemory = calculateModelWeightsMemory(model, dtype)
    const optimizerConfig = optimizers[optimizer]

    // ZeRO optimization reduces optimizer memory
    let reductionFactor = 1
    if (zeroOptimization === 1) reductionFactor = 0.5 // ZeRO-1: shard optimizer states
    if (zeroOptimization === 2) reductionFactor = 0.25 // ZeRO-2: shard optimizer + gradients
    if (zeroOptimization === 3) reductionFactor = 0.125 // ZeRO-3: shard everything

    return baseWeightsMemory * optimizerConfig.memoryFactor * reductionFactor
}

function calculateActivationsMemory(model, batchSize, sequenceLength, dtype = 'fp32') {
    // Rough estimate: batch * seq * layers * hidden * factor
    const bytesPerValue = dtype === 'fp16' || dtype === 'bf16' ? 2 : 4
    let factor = 10 // Approximate factor for all intermediate activations

    if (gradientCheckpointing) {
        factor = 2 // Only store checkpoint activations
    }

    // Add dynamic variation during training (activations vary based on batch)
    const variation = isTraining ? 1 + Math.sin(Date.now() * 0.001) * 0.2 : 1

    const activationBytes =
        batchSize * sequenceLength * model.layers * model.hidden * factor * bytesPerValue * variation
    return activationBytes / 1024 ** 3 // GiB
}

function getTotalTrainingMemory() {
    const model = models[currentModelIndex]
    const dtype = mixedPrecision ? 'fp16' : 'fp32'

    let weightsMemory = calculateModelWeightsMemory(model, dtype)
    let gradientsMemory = calculateGradientsMemory(model, dtype)
    let optimizerMemory = calculateOptimizerMemory(model, currentOptimizer, dtype)
    let activationsMemory = calculateActivationsMemory(model, batchSize, sequenceLength, dtype)

    // Mixed precision requires FP32 master weights
    if (mixedPrecision) {
        weightsMemory += calculateModelWeightsMemory(model, 'fp32') // Add master weights
    }

    // FSDP shards everything across GPUs
    if (fullyShardedDataParallel && gpuCount > 1) {
        weightsMemory /= gpuCount
        gradientsMemory /= gpuCount
        optimizerMemory /= gpuCount
    }

    // Data parallel distributes activations and gradients
    else if (gpuCount > 1 && !fullyShardedDataParallel) {
        // In data parallel, each GPU has full model but batch is split
        activationsMemory /= gpuCount
        // Gradients are accumulated then synchronized
        // Each GPU still needs full gradient memory
    }

    return {
        weights: weightsMemory,
        gradients: gradientsMemory,
        optimizer: optimizerMemory,
        activations: activationsMemory,
        total: weightsMemory + gradientsMemory + optimizerMemory + activationsMemory,
    }
}

// Calculate learning rate with schedule
function getLearningRate(step) {
    const warmupSteps = 1000
    const baseLR = 5e-5

    if (step < warmupSteps) {
        // Linear warmup
        return baseLR * (step / warmupSteps)
    } else {
        // Cosine decay
        const progress = (step - warmupSteps) / (maxSteps - warmupSteps)
        return baseLR * 0.5 * (1 + Math.cos(Math.PI * progress))
    }
}

// Draw multi-GPU cluster for distributed training
function drawMultiGPUCluster() {
    const memory = getTotalTrainingMemory()
    const memPerGPU = memory.total / gpuCount
    const gpuMemory = gpuConfigs[currentGPU].memory

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

    // Scale GPU size based on count
    let scaleFactor = 0.7
    if (gpuCount > 16) scaleFactor = 0.35
    if (gpuCount > 32) scaleFactor = 0.25
    if (gpuCount > 64) scaleFactor = 0.15
    if (gpuCount >= 128) scaleFactor = 0.03

    const maxGPUSize = Math.min(150, Math.min(canvas.width / (cols + 1), canvas.height / (rows + 1)))
    const gpuSize = maxGPUSize * scaleFactor
    const gpuSpacing = maxGPUSize * (gpuCount >= 128 ? 0.7 : (gpuCount > 16 ? 0.95 : 1.2))

    // Center the grid
    const gridWidth = cols * gpuSpacing
    const gridHeight = rows * gpuSpacing
    const offsetX = (canvas.width - gridWidth) / 2 + gpuSpacing / 2
    const offsetY = (canvas.height - gridHeight) / 2 + gpuSpacing / 2

    // Draw interconnect lines (simplified for training)
    if (gpuCount > 1) {
        ctx.strokeStyle = 'rgba(255, 100, 100, 0.2)'
        ctx.lineWidth = 1
        for (let i = 0; i < gpuCount; i++) {
            const row1 = Math.floor(i / cols)
            const col1 = i % cols
            const x1 = offsetX + col1 * gpuSpacing
            const y1 = offsetY + row1 * gpuSpacing

            // Connect to neighbors
            if (col1 < cols - 1) { // Right neighbor
                ctx.beginPath()
                ctx.moveTo(x1 + gpuSize/2, y1)
                ctx.lineTo(x1 + gpuSpacing - gpuSize/2, y1)
                ctx.stroke()
            }
            if (row1 < rows - 1) { // Bottom neighbor
                ctx.beginPath()
                ctx.moveTo(x1, y1 + gpuSize/2)
                ctx.lineTo(x1, y1 + gpuSpacing - gpuSize/2)
                ctx.stroke()
            }
        }
    }

    // Draw GPUs
    for (let i = 0; i < gpuCount; i++) {
        const row = Math.floor(i / cols)
        const col = i % cols
        const x = offsetX + col * gpuSpacing - gpuSize / 2
        const y = offsetY + row * gpuSpacing - gpuSize / 2

        // GPU background gradient
        const gradient = ctx.createLinearGradient(x, y, x, y + gpuSize)
        gradient.addColorStop(0, 'rgba(30, 15, 15, 0.9)')
        gradient.addColorStop(1, 'rgba(50, 20, 20, 0.9)')
        ctx.fillStyle = gradient
        ctx.fillRect(x, y, gpuSize, gpuSize * 0.75)

        // GPU border - color based on memory usage
        const memUsage = memPerGPU / gpuMemory
        ctx.strokeStyle = memUsage > 0.9 ? '#FF4444' : memUsage > 0.7 ? '#FFA500' : '#4CAF50'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, gpuSize, gpuSize * 0.75)

        // Memory blocks visualization (simplified for small GPUs)
        if (gpuSize > 40) {
            const blockSize = Math.max(4, gpuSize / 10)
            const blocksPerRow = Math.floor(gpuSize / (blockSize + 1))
            const blocksPerCol = Math.floor((gpuSize * 0.75) / (blockSize + 1))
            const totalBlocks = blocksPerRow * blocksPerCol

            const weightsBlocks = Math.floor(totalBlocks * (memory.weights / memory.total))
            const gradientsBlocks = Math.floor(totalBlocks * (memory.gradients / memory.total))
            const optimizerBlocks = Math.floor(totalBlocks * (memory.optimizer / memory.total))
            const activationsBlocks = Math.floor(totalBlocks * (memory.activations / memory.total))

            let blockIndex = 0
            for (let by = 0; by < blocksPerCol; by++) {
                for (let bx = 0; bx < blocksPerRow; bx++) {
                    const blockX = x + bx * (blockSize + 1) + 1
                    const blockY = y + by * (blockSize + 1) + 1

                    if (blockIndex < weightsBlocks) {
                        ctx.fillStyle = 'rgba(156, 39, 176, 0.8)' // Purple
                    } else if (blockIndex < weightsBlocks + gradientsBlocks) {
                        ctx.fillStyle = 'rgba(255, 152, 0, 0.8)' // Orange
                    } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks) {
                        ctx.fillStyle = 'rgba(244, 67, 54, 0.8)' // Red
                    } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks + activationsBlocks) {
                        ctx.fillStyle = 'rgba(76, 175, 80, 0.8)' // Green
                    } else {
                        ctx.fillStyle = 'rgba(50, 50, 50, 0.3)' // Empty
                    }
                    ctx.fillRect(blockX, blockY, blockSize, blockSize)
                    blockIndex++
                }
            }
        }

        // GPU label for smaller counts
        if (gpuCount <= 16 && gpuSize > 30) {
            ctx.fillStyle = '#FFF'
            ctx.font = `${Math.max(10, gpuSize / 8)}px monospace`
            ctx.textAlign = 'center'
            ctx.fillText(`GPU ${i}`, x + gpuSize / 2, y - 5)
        }
    }

    // Show total cluster info
    ctx.fillStyle = '#FFF'
    ctx.font = 'bold 16px monospace'
    ctx.textAlign = 'center'
    const clusterText = fullyShardedDataParallel ? `${gpuCount}× ${currentGPU} (FSDP)` : `${gpuCount}× ${currentGPU} (Data Parallel)`
    ctx.fillText(clusterText, canvas.width / 2, canvas.height - 50)

    const memText = `Memory: ${memPerGPU.toFixed(1)} GiB/GPU (Total: ${memory.total.toFixed(1)} GiB)`
    ctx.font = '14px monospace'
    ctx.fillText(memText, canvas.width / 2, canvas.height - 30)
}

// Draw single GPU memory visualization
function drawGPUMemory() {
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const gpuWidth = 400
    const gpuHeight = 300
    const gpuX = centerX - gpuWidth / 2
    const gpuY = centerY - gpuHeight / 2

    // GPU background
    const gradient = ctx.createLinearGradient(gpuX, gpuY, gpuX, gpuY + gpuHeight)
    gradient.addColorStop(0, 'rgba(30, 15, 15, 0.9)')
    gradient.addColorStop(1, 'rgba(50, 20, 20, 0.9)')
    ctx.fillStyle = gradient
    ctx.fillRect(gpuX, gpuY, gpuWidth, gpuHeight)

    // GPU border
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)'
    ctx.lineWidth = 2
    ctx.strokeRect(gpuX, gpuY, gpuWidth, gpuHeight)

    // Memory segments visualization
    const memory = getTotalTrainingMemory()
    const gpuMemory = gpuConfigs[currentGPU].memory
    const memoryUsage = memory.total / gpuMemory

    // Draw memory blocks
    const blockSize = 20
    const blocksX = Math.floor(gpuWidth / (blockSize + 2))
    const blocksY = Math.floor(gpuHeight / (blockSize + 2))
    const totalBlocks = blocksX * blocksY

    const weightsBlocks = Math.floor(totalBlocks * (memory.weights / memory.total))
    const gradientsBlocks = Math.floor(totalBlocks * (memory.gradients / memory.total))
    const optimizerBlocks = Math.floor(totalBlocks * (memory.optimizer / memory.total))
    const activationsBlocks = Math.floor(totalBlocks * (memory.activations / memory.total))

    let blockIndex = 0
    for (let y = 0; y < blocksY; y++) {
        for (let x = 0; x < blocksX; x++) {
            const blockX = gpuX + x * (blockSize + 2) + 2
            const blockY = gpuY + y * (blockSize + 2) + 2

            if (blockIndex < weightsBlocks) {
                ctx.fillStyle = 'rgba(156, 39, 176, 0.8)' // Purple for weights
            } else if (blockIndex < weightsBlocks + gradientsBlocks) {
                ctx.fillStyle = 'rgba(255, 152, 0, 0.8)' // Orange for gradients
            } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks) {
                ctx.fillStyle = 'rgba(244, 67, 54, 0.8)' // Red for optimizer
            } else if (blockIndex < weightsBlocks + gradientsBlocks + optimizerBlocks + activationsBlocks) {
                ctx.fillStyle = 'rgba(76, 175, 80, 0.8)' // Green for activations
            } else {
                ctx.fillStyle = 'rgba(50, 50, 50, 0.3)' // Empty
            }

            ctx.fillRect(blockX, blockY, blockSize, blockSize)
            blockIndex++
        }
    }

    // Draw utilization indicator
    const utilX = centerX
    const utilY = gpuY + gpuHeight + 40
    ctx.fillStyle = memoryUsage > 0.9 ? '#FF4444' : memoryUsage > 0.7 ? '#FFA500' : '#4CAF50'
    ctx.font = 'bold 14px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(`GPU Memory: ${(memoryUsage * 100).toFixed(1)}%`, utilX, utilY)

    // Training progress bar
    const progressWidth = 300
    const progressHeight = 10
    const progressX = centerX - progressWidth / 2
    const progressY = utilY + 20

    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.fillRect(progressX, progressY, progressWidth, progressHeight)

    const progress = currentStep / maxSteps
    ctx.fillStyle = 'rgba(76, 175, 80, 0.8)'
    ctx.fillRect(progressX, progressY, progressWidth * progress, progressHeight)

    // Step counter
    ctx.fillStyle = '#EAF2FF'
    ctx.font = '12px monospace'
    ctx.fillText(`Step ${currentStep.toLocaleString()} / ${maxSteps.toLocaleString()}`, centerX, progressY + 25)
}

// Draw training loss curve
function drawLossCurve() {
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height)

    if (lossHistory.length < 2) return

    // Find min and max loss for auto-scaling
    const minLoss = Math.min(...lossHistory)
    const maxLoss = Math.max(...lossHistory)
    const lossRange = maxLoss - minLoss || 1

    // Draw axes
    lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    lossCtx.lineWidth = 1
    lossCtx.beginPath()
    lossCtx.moveTo(20, 10)
    lossCtx.lineTo(20, 90)
    lossCtx.lineTo(240, 90)
    lossCtx.stroke()

    // Draw Y-axis labels
    lossCtx.fillStyle = 'rgba(255, 255, 255, 0.5)'
    lossCtx.font = '9px monospace'
    lossCtx.textAlign = 'right'
    lossCtx.fillText(maxLoss.toFixed(2), 18, 15)
    lossCtx.fillText(minLoss.toFixed(2), 18, 88)

    // Draw loss curve
    lossCtx.strokeStyle = '#4CAF50'
    lossCtx.lineWidth = 2
    lossCtx.beginPath()

    // Scale X to show all data points
    const xScale = 210 / Math.max(1, lossHistory.length - 1)
    // Scale Y to fit the actual loss range with padding
    const yPadding = 5
    const yScale = (70 - yPadding * 2) / lossRange

    lossHistory.forEach((loss, i) => {
        const x = 25 + i * xScale
        const y = 85 - yPadding - (loss - minLoss) * yScale

        if (i === 0) {
            lossCtx.moveTo(x, y)
        } else {
            lossCtx.lineTo(x, y)
        }
    })

    lossCtx.stroke()

    // Draw current loss point with glow
    if (lossHistory.length > 0) {
        const lastLoss = lossHistory[lossHistory.length - 1]
        const lastX = 25 + (lossHistory.length - 1) * xScale
        const lastY = 85 - yPadding - (lastLoss - minLoss) * yScale

        // Glow effect
        lossCtx.fillStyle = 'rgba(76, 175, 80, 0.3)'
        lossCtx.beginPath()
        lossCtx.arc(lastX, lastY, 6, 0, Math.PI * 2)
        lossCtx.fill()

        // Solid point
        lossCtx.fillStyle = '#4CAF50'
        lossCtx.beginPath()
        lossCtx.arc(lastX, lastY, 3, 0, Math.PI * 2)
        lossCtx.fill()
    }

    // Current loss value
    lossCtx.fillStyle = '#EAF2FF'
    lossCtx.font = '12px monospace'
    lossCtx.textAlign = 'left'
    lossCtx.fillText(`Loss: ${currentLoss.toFixed(3)}`, 160, 20)

    // Step indicator
    lossCtx.font = '10px monospace'
    lossCtx.fillStyle = 'rgba(255, 255, 255, 0.5)'
    lossCtx.fillText(`${lossHistory.length} samples`, 160, 35)
}

// Update UI elements
function updateUI() {
    const model = models[currentModelIndex]
    const memory = getTotalTrainingMemory()

    // Update info panel
    document.getElementById('modelName').textContent = model.name
    document.getElementById('paramCount').textContent = model.params + 'B'
    document.getElementById('batchInfo').textContent =
        batchSize + (gradientAccumulation ? ` (×${accumulationSteps})` : '')
    document.getElementById('seqLength').textContent = sequenceLength.toLocaleString()
    document.getElementById('learningRate').textContent = getLearningRate(currentStep).toExponential(1)
    document.getElementById('stepCount').textContent =
        `${currentStep.toLocaleString()} / ${(maxSteps / 1000).toFixed(0)}K`
    // Show per-GPU memory when distributed
    const memPerGPU = gpuCount > 1 ? memory.total / gpuCount : memory.total
    document.getElementById('totalMem').textContent =
        gpuCount > 1 ? `${memPerGPU.toFixed(1)} GiB/GPU`
                     : `${memory.total.toFixed(1)} GiB`

    // GPU requirement display
    if (gpuCount > 1) {
        document.getElementById('gpuCount').textContent = `${gpuCount}× ${currentGPU}`
    } else {
        const gpusNeeded = Math.ceil(memory.total / gpuConfigs[currentGPU].memory)
        document.getElementById('gpuCount').textContent = `${gpusNeeded}× ${currentGPU}`
    }

    // Memory breakdown
    document.getElementById('weightsSize').textContent = memory.weights.toFixed(1) + ' GiB'
    document.getElementById('gradientsSize').textContent = memory.gradients.toFixed(1) + ' GiB'
    document.getElementById('optimizerSize').textContent = memory.optimizer.toFixed(1) + ' GiB'
    document.getElementById('activationsSize').textContent = memory.activations.toFixed(1) + ' GiB'

    // Update memory bars
    const maxMem = memory.total
    document.getElementById('weightsBar').style.width = (memory.weights / maxMem) * 100 + '%'
    document.getElementById('gradientsBar').style.width = (memory.gradients / maxMem) * 100 + '%'
    document.getElementById('optimizerBar').style.width = (memory.optimizer / maxMem) * 100 + '%'
    document.getElementById('activationsBar').style.width = (memory.activations / maxMem) * 100 + '%'

    // Show warning if memory exceeds GPU
    const warning = document.getElementById('warning')
    if (memory.total > gpuConfigs[currentGPU].memory) {
        warning.style.visibility = 'visible'
        warning.style.opacity = '1'
        warning.textContent = `⚠️ Out of Memory! Requires ${gpusNeeded} GPUs or enable optimizations`
    } else if (memory.total > gpuConfigs[currentGPU].memory * 0.9) {
        warning.style.visibility = 'visible'
        warning.style.opacity = '1'
        warning.textContent = `⚠️ Warning: Near memory limit (${((memory.total / gpuConfigs[currentGPU].memory) * 100).toFixed(0)}% used)`
    } else {
        warning.style.visibility = 'hidden'
        warning.style.opacity = '0'
    }
}

// Animation loop
function animate() {
    ctx.fillStyle = 'rgba(10, 10, 10, 0.1)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Training step progress
    if (isTraining) {
        currentStep += trainingSpeed * 10
        if (currentStep >= maxSteps) {
            currentStep = maxSteps
            isTraining = false
            document.getElementById('playPause').innerHTML =
                '<span>✓ Complete</span><br><span style="font-size: 0.7em; opacity: 0.8">Restart Training</span>'
        }

        // Simulate realistic loss decay with noise
        const progress = currentStep / maxSteps
        const warmupProgress = Math.min(1, currentStep / 1000)

        // Exponential decay with warm-up
        const targetLoss = 0.1
        const baseLoss = 4.5

        // Loss follows exponential decay with some noise
        const idealLoss = targetLoss + (baseLoss - targetLoss) * Math.exp(-5 * progress)

        // Add realistic training noise
        const noise = (Math.random() - 0.5) * 0.1 * (1 - progress * 0.8) // Noise decreases over time

        // Occasional spikes (gradient instability)
        const spike = Math.random() < 0.02 ? Math.random() * 0.5 : 0

        // Smooth the transition
        currentLoss = currentLoss * 0.95 + (idealLoss + noise + spike) * 0.05

        // Record loss history - keep all points but downsample for performance
        if (currentStep % 100 === 0) {
            lossHistory.push(currentLoss)
            // Optional: downsample if too many points for performance
            if (lossHistory.length > 500) {
                // Keep every other point to maintain shape
                const downsampled = []
                for (let i = 0; i < lossHistory.length; i += 2) {
                    downsampled.push(lossHistory[i])
                }
                lossHistory = downsampled
            }
        }
    }

    // Draw visualization based on GPU count
    if (gpuCount > 1) {
        drawMultiGPUCluster()
    } else {
        drawGPUMemory()
    }

    drawLossCurve()
    updateUI()

    requestAnimationFrame(animate)
}

// Event listeners
document.getElementById('playPause').addEventListener('click', function () {
    // If training is complete, restart
    if (currentStep >= maxSteps) {
        currentStep = 0
        lossHistory = []
        currentLoss = 4.5
        isTraining = true
        this.innerHTML = '<span>⏸️ Pause Training</span>'
    } else {
        isTraining = !isTraining
        if (currentStep === 0) {
            this.innerHTML = '<span>▶️ Start Training</span>'
        } else {
            this.innerHTML = isTraining ? '<span>⏸️ Pause Training</span>' : '<span>▶️ Resume Training</span>'
        }
    }
})

document.getElementById('speedControl').addEventListener('click', function () {
    const speeds = [0.5, 1, 2, 5, 10, 50, 100]
    const currentIndex = speeds.indexOf(trainingSpeed)
    trainingSpeed = speeds[(currentIndex + 1) % speeds.length]
    this.textContent = `Speed: ${trainingSpeed}x`
})

document.getElementById('modelControl').addEventListener('click', function () {
    currentModelIndex = (currentModelIndex + 1) % models.length
    const model = models[currentModelIndex]
    this.textContent = `Model: ${model.name.split('-')[0]}-${model.params}B`
    currentStep = 0 // Reset training
    lossHistory = []
    currentLoss = 4.5
})

document.getElementById('batchControl').addEventListener('click', function () {
    const batches = [1, 2, 4, 8, 16, 32, 64]
    const currentIndex = batches.indexOf(batchSize)
    batchSize = batches[(currentIndex + 1) % batches.length]
    this.textContent = `Batch: ${batchSize}`
})

document.getElementById('seqControl').addEventListener('click', function () {
    const sequences = [512, 1024, 2048, 4096, 8192, 16384]
    const currentIndex = sequences.indexOf(sequenceLength)
    sequenceLength = sequences[(currentIndex + 1) % sequences.length]
    this.textContent = `Seq: ${sequenceLength}`
})

document.getElementById('optimizerControl').addEventListener('click', function () {
    const opts = Object.keys(optimizers)
    const currentIndex = opts.indexOf(currentOptimizer)
    currentOptimizer = opts[(currentIndex + 1) % opts.length]
    this.textContent = `Optimizer: ${currentOptimizer}`
})

// Training optimization toggles
document.getElementById('gcToggle').addEventListener('click', function () {
    gradientCheckpointing = !gradientCheckpointing
    this.querySelector('span').textContent = `GC: ${gradientCheckpointing ? 'ON' : 'OFF'}`
    this.classList.toggle('enabled', gradientCheckpointing)
})

document.getElementById('mpToggle').addEventListener('click', function () {
    mixedPrecision = !mixedPrecision
    this.querySelector('span').textContent = `MP: ${mixedPrecision ? 'ON' : 'OFF'}`
    this.classList.toggle('enabled', mixedPrecision)
})

document.getElementById('zeroToggle').addEventListener('click', function () {
    zeroOptimization = (zeroOptimization + 1) % 4
    const labels = ['OFF', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3']
    this.querySelector('span').textContent = labels[zeroOptimization]
    this.classList.toggle('enabled', zeroOptimization > 0)
})

document.getElementById('fsdpToggle').addEventListener('click', function () {
    fullyShardedDataParallel = !fullyShardedDataParallel
    this.querySelector('span').textContent = `FSDP: ${fullyShardedDataParallel ? 'ON' : 'OFF'}`
    this.classList.toggle('enabled', fullyShardedDataParallel)
})

document.getElementById('gaToggle').addEventListener('click', function () {
    gradientAccumulation = !gradientAccumulation
    if (gradientAccumulation) {
        accumulationSteps = 4
    } else {
        accumulationSteps = 1
    }
    this.querySelector('span').textContent = `GA: ${gradientAccumulation ? 'ON' : 'OFF'}`
    this.classList.toggle('enabled', gradientAccumulation)
})

// Store original settings before DC selection
let originalSettings = {
    currentGPU: 'H100 80G',
    currentModelIndex: 0,
    batchSize: 2,
    sequenceLength: 1024,
    currentOptimizer: 'Adam',
    gpuCount: 1
}

// Capture initial settings
originalSettings.currentGPU = currentGPU
originalSettings.currentModelIndex = currentModelIndex
originalSettings.batchSize = batchSize
originalSettings.sequenceLength = sequenceLength
originalSettings.currentOptimizer = currentOptimizer
originalSettings.gpuCount = gpuCount

document.getElementById('gpuControl').addEventListener('click', function () {
    const gpus = Object.keys(gpuConfigs)
    const currentIndex = gpus.indexOf(currentGPU)
    currentGPU = gpus[(currentIndex + 1) % gpus.length]
    this.textContent = `GPU: ${currentGPU}`
})

// GPU count control
const gpuCountBtn = document.getElementById('gpuCountControl')
if (gpuCountBtn) {
    gpuCountBtn.addEventListener('click', function () {
        const idx = validGPUCounts.indexOf(gpuCount)
        gpuCount = validGPUCounts[(idx + 1) % validGPUCounts.length]
        this.textContent = `GPUs: ${gpuCount}`
    })
}

// World DC control for training clusters
const worldDCBtn = document.getElementById('worldDCControl')
if (worldDCBtn) {
    worldDCBtn.addEventListener('click', function () {
        // Cycle through datacenters
        const dcKeys = Object.keys(worldDatacenters)
        const currentIdx = dcKeys.indexOf(currentDatacenter)
        currentDatacenter = dcKeys[(currentIdx + 1) % dcKeys.length]

        const dc = worldDatacenters[currentDatacenter]
        this.textContent = `DC: ${dc.name}`

        if (currentDatacenter === 'none') {
            // Restore original settings
            currentGPU = originalSettings.currentGPU
            currentModelIndex = originalSettings.currentModelIndex
            batchSize = originalSettings.batchSize
            sequenceLength = originalSettings.sequenceLength
            currentOptimizer = originalSettings.currentOptimizer
            gpuCount = originalSettings.gpuCount || 1

            // Update UI
            document.getElementById('gpuControl').textContent = `GPU: ${currentGPU}`
            document.getElementById('gpuCountControl').textContent = `GPUs: ${gpuCount}`
            document.getElementById('modelName').textContent = models[currentModelIndex].name
            document.getElementById('paramCount').textContent = `${models[currentModelIndex].params}B`
            document.getElementById('batchInfo').textContent = batchSize
            document.getElementById('seqLength').textContent = sequenceLength
            document.getElementById('optimizerControl').textContent = `Optimizer: ${currentOptimizer}`
            document.getElementById('modelControl').textContent = `Model: ${models[currentModelIndex].name.split('-')[0]}-${models[currentModelIndex].params}B`
            document.getElementById('batchControl').textContent = `Batch: ${batchSize}`
            document.getElementById('seqControl').textContent = `Seq: ${sequenceLength}`
        } else if (dc.gpu !== null) {
            // Apply datacenter configuration
            if (dc.gpus) {
                gpuCount = dc.gpus
                document.getElementById('gpuCountControl').textContent = `GPUs: ${gpuCount}`
            }

            if (dc.gpu && gpuConfigs[dc.gpu]) {
                currentGPU = dc.gpu
                document.getElementById('gpuControl').textContent = `GPU: ${currentGPU}`
            }

            if (dc.model) {
                const modelIdx = models.findIndex(m => m.name === dc.model)
                if (modelIdx !== -1) {
                    currentModelIndex = modelIdx
                    document.getElementById('modelName').textContent = models[currentModelIndex].name
                    document.getElementById('paramCount').textContent = `${models[currentModelIndex].params}B`
                    document.getElementById('modelControl').textContent = `Model: ${models[currentModelIndex].name.split('-')[0]}-${models[currentModelIndex].params}B`
                }
            }

            if (dc.batch) {
                batchSize = dc.batch
                document.getElementById('batchInfo').textContent = batchSize
                document.getElementById('batchControl').textContent = `Batch: ${batchSize}`
            }

            if (dc.seq) {
                sequenceLength = dc.seq
                document.getElementById('seqLength').textContent = sequenceLength
                document.getElementById('seqControl').textContent = `Seq: ${sequenceLength}`
            }

            if (dc.optimizer && optimizers[dc.optimizer]) {
                currentOptimizer = dc.optimizer
                document.getElementById('optimizerControl').textContent = `Optimizer: ${currentOptimizer}`
            }
        }
    })
}

// Start animation
animate()
