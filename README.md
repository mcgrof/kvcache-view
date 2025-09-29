# KV Cache Growth Visualization

> 🎯 **Watch memory explode as LLMs scale to millions of tokens!**

An impactful, animated visualization demonstrating the exponential memory growth challenge of KV caches in Large Language Models, inspired by [LMCache's](https://lmcache.ai) memory optimization research.

## ✨ Features

- 🚀 **Real-time Animation**: Watch KV cache grow from 0 to 100M tokens
- 🔄 **Model Comparison**: Switch between 15+ models including Llama, Qwen, Gemma, Mixtral MOE, and DeepSeek
- 📊 **SOTA Context Lengths**: 128K, 200K, 1M, 2M, 10M, up to 100M tokens
- 🎚️ **Dynamic Data Types**: FP32, FP16, BF16, INT8, INT4 quantization
- ⚡ **Speed Control**: 0.5x to 100x animation speed
- 📈 **Live Metrics**: Memory usage, GPU requirements, efficiency calculations
- 🌊 **Beautiful Visuals**: Peaceful wave effects with particle animations

## 🚀 Live Demo

### View Online (No Installation Required!)
**[View Live Visualization](https://htmlpreview.github.io/?https://github.com/mcgrof/kvcache-view/blob/main/index.html)**

Or use the direct URL:
```
https://htmlpreview.github.io/?https://github.com/mcgrof/kvcache-view/blob/main/index.html
```

**Note:** GitHub's HTML preview may have slight performance limitations compared to running locally.

## The Memory Wall Problem

As context lengths grow from thousands to millions of tokens, KV cache memory becomes the critical bottleneck. This visualization makes that challenge tangible and visceral.

## What It Shows

- **Real-time Memory Growth**: Watch as KV cache memory explodes with increasing context length
- **Model Comparisons**: Switch between 15+ models from 1B to 671B parameters including MOE architectures
- **GPU Requirements**: Live calculation of H100 GPUs needed as memory exceeds single-GPU capacity
- **Visual Metaphors**:
  - Memory grid fills up showing utilization
  - Exponential curve traces the non-linear growth
  - Particle effects intensify with memory pressure
  - Wave patterns inspired by the peaceful aesthetic

## Key Insights Visualized

1. **Small Models (1-4B)**: Can handle 1M context on single datacenter GPUs
2. **Medium Models (8-32B)**: Quickly exceed single GPU at moderate context lengths
3. **MOE Models (Mixtral, Phi-3.5 MoE)**: Efficient inference with sparse activation
4. **Large Models (70-405B)**: Require 6+ H100s for 1M context
5. **DeepSeek-V3 (671B)**: 7x memory reduction through KV-LoRA compression
6. **Qwen3-Next (80B)**: Hybrid attention reduces KV cache by 75%

## 🎯 Quick Start

### Option 1: Use Make (Recommended)
```bash
make        # Starts server and opens browser automatically
make stop   # Stop the server
```

### Option 2: Python HTTP Server
```bash
python3 -m http.server 8000
# Then navigate to http://localhost:8000
```

### Option 3: Open Directly
```bash
open index.html      # macOS
xdg-open index.html  # Linux
start index.html     # Windows
```

### Option 4: GitHub Pages
Fork this repo and enable GitHub Pages in settings, then visit:
```
https://[your-username].github.io/kvcache-view/
```

## Controls

- **Play/Pause**: Start or stop the animation
- **Speed Control**: Adjust animation speed (0.5x to 100x) - now with ludicrous speed!
- **Next Model**: Cycle through different model architectures
- **Context Control**: Switch between SOTA context lengths:
  - 128K - Standard context
  - 200K - Claude 3.5 Sonnet
  - 1M - Llama 3.1 / standard long context
  - 2M - Gemini 1.5 Pro
  - 10M - Magic/research models
  - 100M - Theoretical future (prepare for memory apocalypse!)
- **Type Control**: Dynamic data type switching:
  - FP32 - Full precision (4 bytes)
  - FP16 - Half precision (2 bytes)
  - BF16 - Brain float (2 bytes)
  - INT8 - 8-bit quantization (1 byte)
  - INT4 - 4-bit quantization (0.5 bytes)

## The Math Behind It

The visualization uses the exact KV cache formulas from [LMCache's KV Cache Calculator](https://lmcache.ai/kv_cache_calculator.html):

### Standard Models
```
KV Cache = 2 × layers × tokens × kv_heads × (hidden_size/attention_heads) × dtype_size
```

### DeepSeek Models (with KV-LoRA)
```
KV Cache = layers × tokens × (kv_lora_rank + qk_rope_head_dim) × dtype_size
```

## Why This Matters

LMCache addresses this memory explosion through intelligent caching strategies:
- Prefix caching to reuse computed KV states
- Compression techniques to reduce memory footprint
- Distributed caching across multiple nodes

This visualization demonstrates why such optimizations are critical for the future of LLMs.

## Inspiration

- **Visual Style**: Inspired by the meditative waves from the "peace" OpenGL project
- **Technical Foundation**: Based on LMCache's KV cache calculator and memory projections
- **Purpose**: Make the abstract concept of memory scaling tangible and impactful

## Technical Stack

- Pure JavaScript with HTML5 Canvas
- No external dependencies
- Hardware-accelerated rendering
- Responsive design

## Memory Reference Tables

### Standard FP16 Precision

| Model | 128K Context | 1M Context | 2M Context | 10M Context | H100s @10M |
|-------|-------------|------------|-------------|-------------|------------|
| Llama-3.2-1B | 4 GiB | 31 GiB | 62 GiB | 312 GiB | 4 |
| Llama-3.1-8B | 16 GiB | 122 GiB | 244 GiB | 1.2 TiB | 16 |
| Llama-3.1-70B | 40 GiB | 305 GiB | 610 GiB | 3.0 TiB | 39 |
| DeepSeek-V3 | 8.5 GiB | 65 GiB | 131 GiB | 654 GiB | 9 |
| Llama-3.1-405B | 63 GiB | 481 GiB | 962 GiB | 4.7 TiB | 61 |

### Impact of Data Types (Llama-70B @ 1M tokens)

| Data Type | Memory Size | H100 GPUs | Memory Reduction |
|-----------|-------------|-----------|------------------|
| FP32 | 610 GiB | 8 | Baseline |
| FP16 | 305 GiB | 4 | 50% |
| BF16 | 305 GiB | 4 | 50% |
| INT8 | 153 GiB | 2 | 75% |
| INT4 | 76 GiB | 1 | 87.5% |

## 📁 Project Structure

```
kvcache-view/
├── index.html                    # Main HTML file with UI
├── visualization.js              # Core visualization logic
├── README.md                     # This file
├── Makefile                      # Simple server commands
├── CLAUDE.md                     # Development guidelines
├── GPU_OPTIMIZATION_DEFAULTS.md # GPU optimization rationale
└── VISUALIZATION_GUIDE.md       # Visual components guide
```

## 📚 Documentation

- **[Complete Feature Documentation](SIMULATION_FEATURES.md)** - Comprehensive guide to all simulation features, controls, and calculations
- **[GPU Optimization Defaults](GPU_OPTIMIZATION_DEFAULTS.md)** - Why certain optimizations (CB, PA, FA) are enabled/disabled by default for different GPUs
- **[Visualization Guide](VISUALIZATION_GUIDE.md)** - Detailed explanation of all visual components and their meanings
- **[Development Guidelines](CLAUDE.md)** - Contributing guidelines and common pitfalls

## 🔗 Related Projects

- [LMCache](https://github.com/LMCache/LMCache) - The KV cache optimization framework that inspired this visualization
- [LMCache Calculator](https://lmcache.ai/kv_cache_calculator.html) - Interactive KV cache size calculator
- [Peace](https://github.com/mcgrof/peace) - The meditative wave visualization that inspired the aesthetics

## 📝 License

Created as a demonstration of the memory challenges that LMCache solves. Do whatever brings you peace.