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

let currentModelIndex = 2 // Start with Llama-3.1-8B - realistic training size
let batchSize = 8 // Realistic datacenter training batch size
let sequenceLength = 2048 // Standard context length for training
let accumulationSteps = 1

// Optimizer configurations
const optimizers = {
    SGD: { memoryFactor: 0 }, // No additional memory
    Adam: { memoryFactor: 2 }, // 2x model size for momentum + variance
    AdamW: { memoryFactor: 2 }, // Same as Adam
    Lion: { memoryFactor: 1 }, // 1x model size for momentum only
    AdaFactor: { memoryFactor: 0.5 }, // Factorized second moments
}
let currentOptimizer = 'AdamW'

// Training optimizations - start with some enabled to show their benefit
let gradientCheckpointing = true // Enabled by default to reduce memory
let mixedPrecision = true // Enabled by default - standard practice
let zeroOptimization = 0 // 0=off, 1=ZeRO-1, 2=ZeRO-2, 3=ZeRO-3
let fullyShardedDataParallel = true // Enabled by default for 8B+ model training
let gradientAccumulation = false

// GPU configurations
const gpuConfigs = {
    'Tesla T4 16G': { memory: 16, bandwidth: 300, compute: 8.1 },
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
let useHighSpeedInterconnect = true  // Use NVLink/InfinityFabric when available

// Famous training datacenter configurations
const worldDatacenters = {
    none: { name: 'None', gpus: null, gpu: null, model: null, batch: null, seq: null, optimizer: null, interconnect: null },
    dgx_h100: { name: 'DGX H100', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 8, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    dgx_pod: { name: 'DGX SuperPOD', gpus: 32, gpu: 'H100 80G', model: 'Llama-3.1-70B', batch: 4, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    meta_rsc: { name: 'Meta Training', gpus: 128, gpu: 'A100 80G', model: 'Llama-3.1-70B', batch: 2, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    openai_gpt: { name: 'OpenAI GPT-4', gpus: 64, gpu: 'A100 40G', model: 'Llama-3.1-8B', batch: 4, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    aws_p5: { name: 'AWS P5 Train', gpus: 8, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 8, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    gcp_tpu: { name: 'GCP TPU v5e', gpus: 8, gpu: 'TPU v5e 16G', model: 'Llama-3.2-1B', batch: 8, seq: 1024, optimizer: 'AdaFactor', interconnect: 'tpu' },
    azure_nd: { name: 'Azure ND A100', gpus: 8, gpu: 'A100 40G', model: 'Mistral-7B', batch: 8, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    lambda_train: { name: 'Lambda Train', gpus: 8, gpu: 'A100 80G', model: 'Llama-3.1-8B', batch: 8, seq: 2048, optimizer: 'Lion', interconnect: 'nvlink' },
    budget_train: { name: 'Budget T4', gpus: 4, gpu: 'Tesla T4 16G', model: 'Llama-3.2-1B', batch: 2, seq: 1024, optimizer: 'AdamW', interconnect: 'pcie' },
    wl900_cluster: { name: 'WL900 Cluster', gpus: 16, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 8, seq: 2048, optimizer: 'AdamW', interconnect: 'wl900' },
    single_gpu: { name: 'Single GPU', gpus: 1, gpu: 'H100 80G', model: 'Llama-3.2-1B', batch: 8, seq: 2048, optimizer: 'AdamW', interconnect: 'none' },
    anthropic_claude: { name: 'Anthropic', gpus: 16, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 4, seq: 4096, optimizer: 'AdamW', interconnect: 'nvlink' },
    stability_sd: { name: 'Stability AI', gpus: 8, gpu: 'A100 80G', model: 'Mistral-7B', batch: 4, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
    cohere_train: { name: 'Cohere Train', gpus: 16, gpu: 'H100 80G', model: 'Llama-3.1-8B', batch: 4, seq: 2048, optimizer: 'AdamW', interconnect: 'nvlink' },
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

    // Scale GPU size based on count - aggressive scaling for large clusters
    let scaleFactor = 0.8
    if (gpuCount > 8) scaleFactor = 0.6
    if (gpuCount > 16) scaleFactor = 0.4
    if (gpuCount > 32) scaleFactor = 0.25
    if (gpuCount > 64) scaleFactor = 0.15
    if (gpuCount >= 128) scaleFactor = 0.08  // Ultra compact for massive clusters

    const maxGPUSize = Math.min(120, Math.min(canvas.width / (cols + 1), canvas.height / (rows + 1)))
    const gpuSize = Math.max(gpuCount >= 128 ? 6 : 8, maxGPUSize * scaleFactor)  // Even smaller for 128+ GPUs
    const gpuSpacing = Math.max(gpuSize + 2, maxGPUSize * (gpuCount >= 128 ? 0.6 : (gpuCount > 16 ? 0.9 : 1.1)))

    // Center the grid with better positioning
    const gridWidth = (cols - 1) * gpuSpacing
    const gridHeight = (rows - 1) * gpuSpacing
    const offsetX = (canvas.width - gridWidth) / 2
    const offsetY = (canvas.height - gridHeight) / 2 + 20  // Small offset from top

    // Calculate training-specific interconnect bandwidth utilization
    let interconnectBW = 64  // PCIe 5.0 default
    let interconnectType = 'PCIe 5.0'

    // Training synchronization traffic (gradient sync + all-reduce patterns)
    const gradientSyncTraffic = (memory.gradients * 1024) * Math.log2(gpuCount)  // All-reduce for gradients
    const activationSyncTraffic = (memory.activations * 1024) * (gpuCount / 4)  // Activation sharding
    const totalSyncTraffic = gradientSyncTraffic + activationSyncTraffic

    // Determine interconnect type from current datacenter or default
    const currentDC = worldDatacenters[currentDatacenter]
    let interconnectSpec = currentDC?.interconnect || (useHighSpeedInterconnect ? 'nvlink' : 'pcie')

    // Set interconnect bandwidth and type based on spec
    switch (interconnectSpec) {
        case 'nvlink':
            interconnectType = 'NVLink 4.0'
            interconnectBW = 900  // GB/s for NVLink
            break
        case 'wl900':
            interconnectType = 'WL900'
            interconnectBW = 1800  // GB/s for WL900 - higher bandwidth
            break
        case 'tpu':
            interconnectType = 'TPU Interconnect'
            interconnectBW = 600  // GB/s for TPU interconnect
            break
        case 'pcie':
        default:
            interconnectType = 'PCIe 5.0'
            interconnectBW = 64  // GB/s for PCIe 5.0
            break
    }

    const bandwidthUtilization = Math.min(1.0, totalSyncTraffic / (interconnectBW * 1000))

    // Draw enhanced interconnect lines with data flow
    if (gpuCount > 1) {
        const connections = []

        // Create connection list
        for (let i = 0; i < Math.min(gpuCount, 64); i++) {  // Increased limit for better large cluster visualization
            const row1 = Math.floor(i / cols)
            const col1 = i % cols
            const x1 = offsetX + col1 * gpuSpacing
            const y1 = offsetY + row1 * gpuSpacing

            // Connect to neighbors
            if (col1 < cols - 1) { // Right neighbor
                connections.push({
                    x1: x1 + gpuSize, y1: y1 + gpuSize/2,
                    x2: x1 + gpuSpacing, y2: y1 + gpuSize/2
                })
            }
            if (row1 < rows - 1) { // Bottom neighbor
                connections.push({
                    x1: x1 + gpuSize/2, y1: y1 + gpuSize,
                    x2: x1 + gpuSize/2, y2: y1 + gpuSpacing
                })
            }
        }

        // Draw connections with bandwidth visualization
        connections.forEach(conn => {
            const {x1, y1, x2, y2} = conn

            // Base line with thickness based on utilization
            const lineWidth = 1 + bandwidthUtilization * 3
            ctx.strokeStyle = bandwidthUtilization > 0.8 ? '#FF4444' :
                             bandwidthUtilization > 0.5 ? '#FFA500' : '#4CAF50'
            ctx.lineWidth = lineWidth
            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()

            // Animate gradient sync particles (show even when not training for demo)
            if (bandwidthUtilization > 0.05 || gpuCount > 1) {
                const particleCount = Math.min(3, Math.max(1, Math.ceil(bandwidthUtilization * 5)))
                for (let p = 0; p < particleCount; p++) {
                    const time = Date.now() / (500 - Math.min(bandwidthUtilization * 300, 400))
                    const offset = (time + p / particleCount) % 1
                    const particleX = x1 + (x2 - x1) * offset
                    const particleY = y1 + (y2 - y1) * offset

                    // Gradient particle (orange for gradients)
                    ctx.fillStyle = isTraining ? '#FF8C00' : '#4CAF50'
                    ctx.globalAlpha = 0.6 + Math.sin(offset * Math.PI * 2) * 0.2
                    ctx.beginPath()
                    ctx.arc(particleX, particleY, 2 + Math.max(bandwidthUtilization * 2, 1), 0, Math.PI * 2)
                    ctx.fill()
                }
            }
        })

        ctx.globalAlpha = 1.0
    }

    // Debug for large GPU counts
    if (gpuCount >= 128) {
        console.log(`Drawing ${gpuCount} GPUs: gpuSize=${gpuSize}px, spacing=${gpuSpacing}px, grid=${cols}×${rows}`)
    }

    // Draw GPUs
    for (let i = 0; i < gpuCount; i++) {
        const row = Math.floor(i / cols)
        const col = i % cols
        const x = offsetX + col * gpuSpacing
        const y = offsetY + row * gpuSpacing

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

    // Show total cluster info at bottom-left (to avoid controls at bottom)
    ctx.fillStyle = '#FFF'
    ctx.font = 'bold 14px monospace'
    ctx.textAlign = 'left'
    const interconnectText = useHighSpeedInterconnect ? 'NVLink' : 'PCIe'
    const clusterText = fullyShardedDataParallel ? `${gpuCount}× ${currentGPU} (FSDP)` : `${gpuCount}× ${currentGPU} (Data Parallel)`
    ctx.fillText(clusterText, 20, canvas.height - 50)

    const memText = `Memory: ${memPerGPU.toFixed(1)} GiB/GPU | Interconnect: ${interconnectText}`
    ctx.font = '12px monospace'
    ctx.fillText(memText, 20, canvas.height - 30)

    // Draw interconnect bandwidth meter at bottom
    const meterX = 20
    const meterY = canvas.height - 100
    const meterWidth = 200
    const meterHeight = 20

    // Meter background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
    ctx.fillRect(meterX, meterY, meterWidth, meterHeight)

    // Meter fill based on utilization
    const fillWidth = meterWidth * bandwidthUtilization
    const meterColor = bandwidthUtilization > 0.8 ? '#FF4444' :
                      bandwidthUtilization > 0.5 ? '#FFA500' : '#4CAF50'
    ctx.fillStyle = meterColor
    ctx.fillRect(meterX, meterY, fillWidth, meterHeight)

    // Meter border
    ctx.strokeStyle = '#666'
    ctx.lineWidth = 1
    ctx.strokeRect(meterX, meterY, meterWidth, meterHeight)

    // Meter label
    ctx.fillStyle = '#FFF'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`${interconnectType} Bandwidth`, meterX, meterY - 5)

    // Utilization percentage
    ctx.font = '10px monospace'
    ctx.fillStyle = '#CCC'
    ctx.fillText(`${(bandwidthUtilization * 100).toFixed(1)}% (${(totalSyncTraffic / 1024).toFixed(1)} GB sync)`, meterX, meterY + meterHeight + 15)

    // Show bottleneck warning
    if (bandwidthUtilization > 0.8) {
        const pulse = Math.sin(Date.now() / 200) * 0.3 + 0.7
        ctx.fillStyle = `rgba(255, 68, 68, ${pulse})`
        ctx.font = 'bold 14px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('⚠️ TRAINING BOTTLENECK ⚠️', canvas.width / 2, 30)
        ctx.fillStyle = '#FF8888'
        ctx.font = '11px monospace'
        ctx.fillText(`${interconnectType} saturated - training will slow down!`, canvas.width / 2, 45)
    }

    // Update training performance metrics box
    updateTrainingPerformanceMetrics(bandwidthUtilization, interconnectType, totalSyncTraffic, interconnectBW)
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

// Update training performance metrics box
function updateTrainingPerformanceMetrics(bandwidthUtilization, interconnectType, totalSyncTraffic, interconnectBW) {
    const performanceBox = document.getElementById('trainingPerformanceBox')
    const metricsContent = document.getElementById('trainingMetricsContent')

    if (!performanceBox || !metricsContent) return

    if (gpuCount > 1 && bandwidthUtilization > 0.1) {
        performanceBox.style.display = 'block'

        // Calculate training performance impact
        const throughputLoss = Math.pow(bandwidthUtilization, 2) * 60  // % throughput loss
        const stepTimeIncrease = bandwidthUtilization > 0.8 ? bandwidthUtilization * 200 : bandwidthUtilization * 50
        const efficiencyLoss = throughputLoss * gpuCount / 100  // Wasted GPU-hours
        const costIncrease = (stepTimeIncrease / 100) * gpuCount * 8  // $/hour impact at $8/GPU/hr

        metricsContent.innerHTML = `
            <div style="margin-bottom: 8px;">
                <div style="color: #FFA500; font-weight: bold;">Throughput: -${throughputLoss.toFixed(1)}%</div>
                <div style="color: #AAA; font-size: 0.9em;">samples/sec reduction</div>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="color: #FF6B6B; font-weight: bold;">Step Time: +${stepTimeIncrease.toFixed(0)}ms</div>
                <div style="color: #AAA; font-size: 0.9em;">per training step</div>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="color: #9C88FF; font-weight: bold;">Efficiency: ${(100 - efficiencyLoss).toFixed(1)}%</div>
                <div style="color: #AAA; font-size: 0.9em;">actual GPU utilization</div>
            </div>
            <div>
                <div style="color: #FFB84D; font-weight: bold;">Cost Impact: +$${costIncrease.toFixed(0)}/hr</div>
                <div style="color: #AAA; font-size: 0.9em;">cluster overhead</div>
            </div>
        `
    } else {
        performanceBox.style.display = 'none'
    }
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

    // Warning box disabled - was annoying and blocking interactions
    const warning = document.getElementById('warning')
    if (warning) {
        warning.style.visibility = 'hidden'
        warning.style.opacity = '0'
        warning.style.display = 'none'
    }
}

// Animation loop
function animate() {
    // Clear canvas completely to avoid artifacts
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set background with solid color to fully remove artifacts
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Add subtle gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
    gradient.addColorStop(0, 'rgba(26, 10, 10, 0.1)')
    gradient.addColorStop(1, 'rgba(10, 10, 10, 0.1)')
    ctx.fillStyle = gradient
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
        this.innerHTML = '⏸️ Pause Training'
    } else {
        isTraining = !isTraining
        if (isTraining) {
            this.innerHTML = '⏸️ Pause Training'
        } else {
            this.innerHTML = '▶️ Resume Training'
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
    gpuCount: 1,
    useHighSpeedInterconnect: true
}

// Capture initial settings
originalSettings.currentGPU = currentGPU
originalSettings.currentModelIndex = currentModelIndex
originalSettings.batchSize = batchSize
originalSettings.sequenceLength = sequenceLength
originalSettings.currentOptimizer = currentOptimizer
originalSettings.gpuCount = gpuCount
originalSettings.useHighSpeedInterconnect = useHighSpeedInterconnect

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
        updateInterconnectButton()
    })
}

// Interconnect control
const interconnectBtn = document.getElementById('interconnectControl')
if (interconnectBtn) {
    interconnectBtn.addEventListener('click', function () {
        if (gpuCount === 1) return
        useHighSpeedInterconnect = !useHighSpeedInterconnect
        this.textContent = useHighSpeedInterconnect ? 'Link: NVLink' : 'Link: PCIe'
        this.style.background = useHighSpeedInterconnect ?
            'linear-gradient(180deg, rgba(118, 185, 0, 0.2), rgba(118, 185, 0, 0.1))' : ''
    })
}

// Helper function to show/hide interconnect button
function updateInterconnectButton() {
    const btn = document.getElementById('interconnectControl')
    if (!btn) return

    if (gpuCount > 1) {
        btn.style.display = ''

        // Get current interconnect type from datacenter or default
        const currentDC = worldDatacenters[currentDatacenter]
        const interconnectSpec = currentDC?.interconnect || (useHighSpeedInterconnect ? 'nvlink' : 'pcie')

        // Display appropriate text and styling
        switch (interconnectSpec) {
            case 'nvlink':
                btn.textContent = 'Link: NVLink'
                btn.style.background = 'linear-gradient(180deg, rgba(118, 185, 0, 0.2), rgba(118, 185, 0, 0.1))'
                break
            case 'wl900':
                btn.textContent = 'Link: WL900'
                btn.style.background = 'linear-gradient(180deg, rgba(255, 215, 0, 0.2), rgba(255, 215, 0, 0.1))'
                break
            case 'tpu':
                btn.textContent = 'Link: TPU'
                btn.style.background = 'linear-gradient(180deg, rgba(0, 150, 255, 0.2), rgba(0, 150, 255, 0.1))'
                break
            case 'pcie':
            default:
                btn.textContent = 'Link: PCIe 5.0'
                btn.style.background = ''
                break
        }
    } else {
        btn.style.display = 'none'
    }
}

// Simple DC button - just triggers existing working GPU count button
function setupDCButton() {
    const dcBtn = document.getElementById('worldDCControl')
    const gpuCountBtn = document.getElementById('gpuCountControl')

    if (!dcBtn || !gpuCountBtn) {
        console.error('DC or GPU count button not found!')
        return
    }

    console.log('Setting up DC button to use existing GPU count logic...')

    dcBtn.onclick = function() {
        console.log(`=== DC CLICKED ===`)

        // Cycle DC name for display
        const dcList = ['none', 'dgx_h100', 'dgx_pod', 'meta_rsc']
        const currentIdx = dcList.indexOf(currentDatacenter)
        currentDatacenter = dcList[(currentIdx + 1) % dcList.length]

        const dc = worldDatacenters[currentDatacenter]
        dcBtn.textContent = `DC: ${dc.name}`

        // Get target GPU count
        const targetGpuCount = currentDatacenter === 'none' ? 1 : (dc.gpus || 8)
        console.log(`Target GPU count: ${targetGpuCount}`)

        // Click the GPU count button until we reach the target
        while (gpuCount !== targetGpuCount) {
            const oldCount = gpuCount
            gpuCountBtn.click()

            // Safety: if we cycled through all options and didn't find target, break
            if (gpuCount === oldCount) {
                console.log(`Couldn't reach target ${targetGpuCount}, settled on ${gpuCount}`)
                break
            }
            if (gpuCount === targetGpuCount) {
                console.log(`Reached target GPU count: ${gpuCount}`)
                break
            }
        }
    }
}

// Setup DC button
setupDCButton()

// Start animation
animate()
