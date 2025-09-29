# KV Cache Visualization - Complete Feature Documentation

This document comprehensively describes all features, controls, and visual elements in the KV Cache Growth Visualization.

## Table of Contents
- [Visual Architecture](#visual-architecture)
- [Optimization Features](#optimization-features)
- [Interactive Controls](#interactive-controls)
- [GPU Models](#gpu-models)
- [LLM Models](#llm-models)
- [Memory Calculations](#memory-calculations)
- [Visual Indicators](#visual-indicators)
- [Animation System](#animation-system)
- [Error Handling](#error-handling)

## Visual Architecture

### GPU Die Representation
The central square represents the GPU compute die with:
- **Size**: 280×280 pixels
- **Color**: Dark metallic with gradient effects
- **Purpose**: Shows where computation happens
- **Flash Attention tiles**: Green animated tiles when FA enabled

### Memory Modules
Surrounding the GPU die are memory modules:
- **HBM GPUs**: 2-6 modules depending on capacity
- **GDDR GPUs**: Typically 2 modules
- **SRAM Accelerators**: Variable based on architecture
- **Visual**: Dark silicon base with memory bank grids

### Memory Bank Visualization
Each memory module contains a grid of memory banks:
- **Bank size**: 8×8 pixels
- **Spacing**: 10 pixels between banks
- **Colors**:
  - Purple: Model weights
  - Blue/Cyan/Pink/Green: Different sequences in batch
  - Gray grid: Unused memory
  - Red ghost overlay: Memory saved by Flash Attention

## Optimization Features

### Continuous Batching (CB)
**Paper**: Orca (OSDI'22)
**Visual Changes**:
- Different colors for each sequence
- Variable-length sequences shown
- No pre-allocated padding waste
- Efficiency formula updates to show dynamic allocation

**Default ON for**:
- GPUs with ≥24GB memory
- All HBM-based GPUs
- SRAM accelerators

### Paged Attention (PA)
**Paper**: vLLM (SOSP'23)
**Visual Changes**:
- Memory broken into 16-token pages
- Fragmented allocation patterns
- Page boundaries visible
- Non-contiguous memory layout

**Default ON for**:
- SRAM accelerators only
- Usually OFF for HBM GPUs (overhead > benefit)

### Flash Attention (FA)
**Paper**: FlashAttention (NeurIPS'22)
**Visual Changes**:
- Red ghost overlays on memory (shows savings)
- Green tiles animating in GPU die
- Large red "MEMORY SAVED" banner
- Diagonal attention line turns green
- Smaller, yellow-green dot on diagonal
- 50% slower animation speed

**Default ON for**:
- All GPUs with ≥16GB memory

### Skip Model Weights (SMW)
**Toggle**: Top-right corner
**Effect**: Excludes model weights from memory calculation
**Visual**: Purple memory banks disappear
**Use case**: Inference-only scenarios with weights in system RAM

## Interactive Controls

### Play/Pause Button
- **Icon**: ▶️/⏸️
- **Function**: Start/stop context growth animation
- **Keyboard**: Spacebar (not implemented)

### Speed Control
**Range**: 0.5x to 100x
**Presets**:
- 0.5x: Slow motion
- 1x: Normal speed
- 2x: Fast
- 10x: Very fast
- 100x: Ludicrous speed

### Model Selection
**Button**: "Next Model"
**Cycles through**: 15+ models
**Updates**:
- Memory calculations
- Efficiency formulas
- Color scheme
- Parameter display

### Context Control
**Presets**:
- 4K: Legacy models
- 8K: GPT-3.5 era
- 16K: Early long context
- 32K: GPT-4 original
- 64K: Current standard
- 128K: Production long context
- 200K: Claude 3.5 Sonnet
- 1M: Llama 3.1 / Gemini
- 2M: Gemini 1.5 Pro
- 10M: Research frontier
- 100M: Theoretical future

### Batch Size Control
**Range**: 1-32
**Effect**: Number of concurrent sequences
**Visual**: Multiple colored sequences
**Memory**: Linear scaling with batch size

### GPU Selection
**Button**: "Device: [GPU Name]"
**Count**: 27 different GPUs
**Updates**:
- Memory capacity
- Memory type (HBM/GDDR/SRAM)
- Default optimizations
- Flash tile sizes

### Data Type Control
**Options**:
- FP32: 4 bytes per value
- FP16: 2 bytes (default)
- BF16: 2 bytes
- INT8: 1 byte
- INT4: 0.5 bytes
**Impact**: Direct memory scaling

## GPU Models

### NVIDIA GPUs
- **Tesla T4**: 16GB GDDR6, 6MB L2, 32×32 FA tiles
- **RTX 4090**: 24GB GDDR6X, 72MB L2, 64×64 FA tiles
- **L40S**: 48GB GDDR6, 96MB L2, 64×64 FA tiles
- **A100 40G/80G**: HBM2e, 40MB L2, 128×128 FA tiles
- **H100**: 80GB HBM3, 50MB L2, 128×128 FA tiles
- **H200**: 141GB HBM3e, 50MB L2, 128×128 FA tiles

### AMD GPUs
- **W7800/W7900**: 32/48GB GDDR6
- **MI210**: 64GB HBM2e
- **MI250X**: 128GB HBM2e
- **MI300X**: 192GB HBM3, 256MB L2

### Intel/Others
- **Arc A770**: 16GB GDDR6
- **Max 1550**: 128GB HBM2e, 408MB L2
- **Gaudi2**: 96GB HBM2e
- **TPU v3/v4**: 16/32GB HBM2
- **Cerebras WSE-2**: 40GB SRAM, 40GB L2
- **Graphcore IPU**: 0.9GB SRAM

## LLM Models

### Efficient Models (1-4B)
- **Llama-3.2-1B**: 16 layers, 2048 hidden, 8 KV heads
- **Phi-3.5-mini**: 32 layers, 3072 hidden
- **Qwen-2.5-1.5B**: 28 layers, 1536 hidden

### Medium Models (7-14B)
- **Llama-3.1-8B**: 32 layers, 4096 hidden
- **Mistral-7B**: 32 layers, 4096 hidden
- **Qwen-2.5-14B**: 48 layers, 5120 hidden

### Large Models (32-70B)
- **Llama-3.1-70B**: 80 layers, 8192 hidden
- **Qwen-2.5-32B**: 64 layers, 5120 hidden

### MOE Models
- **Mixtral-8x7B**: 56B total, 12.9B active
- **Phi-3.5-MoE**: 42B total, 16B active
- **Mixtral-8x22B**: 176B total, 39B active

### Mega Models (100B+)
- **Llama-3.1-405B**: 126 layers, 16384 hidden
- **DeepSeek-V3**: 671B, KV-LoRA compression
- **Qwen-3-Next-80B**: Hybrid attention (25% KV reduction)

## Memory Calculations

### Standard Formula
```
KV_Cache = 2 × layers × tokens × kv_heads × (hidden_size ÷ attention_heads) × dtype_bytes
```

### With Continuous Batching
```
Total = Σ(sequence_length[i] × kv_per_token) for i in batch
```

### With Paged Attention
```
Pages = ⌈tokens ÷ 16⌉
Memory = pages × 16 × kv_per_token
```

### Flash Attention Savings
```
Traditional: chunk_size × seq_len × heads × batch × dtype
Flash: tile_size² × heads × batch × dtype
Savings = min(Traditional - Flash, GPU_memory × 0.5)
```

### DeepSeek-V3 Formula
```
KV_Cache = layers × tokens × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```
7x compression via KV-LoRA factorization

## Visual Indicators

### Info Panel (Top Left)
- Model name and parameters
- Current/max context length
- Memory allocation unit
- Total memory usage breakdown
- GPU requirements
- Memory efficiency %

### Efficiency Formula Box
- Live mathematical formula
- Updates with optimization changes
- Shows actual calculation being used

### Paper References Box
- Appears with optimizations
- Links to original papers
- Auto-positions below efficiency box

### Warning Messages
- Red box for multi-GPU requirements
- Orange for approaching limits
- Updates with GPU count needed

### Factoid Panel
- Educational insights
- Auto-hides when space limited
- 20+ rotating facts about memory/LLMs

### Datacenter Note
- Shows when multiple GPUs needed
- Explains distributed serving requirements

## Animation System

### Wave Effects
- 3 background waves
- Sinusoidal motion
- Speed varies with memory pressure
- Peaceful blue-purple gradient

### Data Flow Particles
- Flow between memory and GPU
- Rate increases with memory usage
- Gold particles with Flash Attention
- Represent bandwidth utilization

### Diagonal Attention Line
- Shows growing context window
- Bottom-left to top-right
- Represents O(n²) attention pattern
- Green with Flash Attention

### Memory Fill Animation
- Sequential bank filling
- Pulse effect on active banks
- Ghost overlays for saved memory
- Smooth transitions

## Error Handling

### Memory Overflow
- Automatic GPU count calculation
- Warning messages at 80% capacity
- Critical alerts at 100%
- Suggests distributed serving

### Invalid States
- Batch size 1 handling
- Empty sequence color arrays
- Undefined GPU configurations
- Fallback values for all parsing

### Browser Compatibility
- Canvas API detection
- Graceful degradation
- Mobile responsive design
- Touch control support

## Performance Optimizations

### Rendering
- RequestAnimationFrame loop
- Hardware-accelerated canvas
- Efficient redraw regions
- Particle pooling

### Memory
- Lazy calculation updates
- Cached GPU configurations
- Debounced UI updates
- Efficient data structures

## Mobile Adaptations

### Layout Changes
- Stacked controls
- Hidden factoids
- Simplified effects
- Reduced particle count

### Touch Controls
- Tap for play/pause
- Swipe for model change
- Pinch for speed control
- Long press for reset

## URL Parameters

Not implemented, but planned:
- `?model=llama-70b`
- `?context=1000000`
- `?batch=4`
- `?gpu=h100`
- `?cb=1&pa=1&fa=1`

## Easter Eggs

### Ludicrous Speed
- 100x speed triggers special effects
- Extra particle intensity
- Warning messages about time-space

### 100M Context
- "Memory Apocalypse" warnings
- Dramatic red overlays
- Suggestion to "build a datacenter"

### Cerebras WSE-2
- Special SRAM messaging
- "Wafer-scale" references
- Unique color scheme

## Development Mode

### Console Commands
Available in browser console:
- `currentTokens = 1000000` - Jump to 1M tokens
- `flashAttention = true` - Enable Flash Attention
- `batchSize = 16` - Set batch size
- `animationSpeed = 10` - Set speed

### Debug Info
- GPU memory calculations logged
- Frame rate in console
- Memory allocation details
- Optimization impact metrics

## Future Features

### Planned Additions
- KV cache compression visualization
- Distributed serving topology
- Speculative decoding impact
- Alternative attention mechanisms
- Real inference latency estimates

### Experimental Features
- WebGPU acceleration
- Multi-GPU topology view
- Network bandwidth visualization
- Cost calculator integration
- Power consumption estimates