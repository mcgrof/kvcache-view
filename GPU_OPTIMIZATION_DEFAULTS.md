# GPU Optimization Defaults Rationale

This document explains why certain optimization features (Continuous Batching, Paged Attention, Flash Attention) are enabled or disabled by default for different GPU configurations in the KV Cache visualization.

## Optimization Features Overview

### Continuous Batching (CB)
- **What it does**: Allows variable sequence lengths within a batch, avoiding padding waste
- **Memory impact**: Reduces memory waste from padding shorter sequences
- **Performance impact**: Minimal overhead, generally improves throughput
- **Paper**: Orca (OSDI'22)

### Paged Attention (PA)
- **What it does**: Breaks memory into fixed-size pages (16 tokens), enabling non-contiguous allocation
- **Memory impact**: Reduces fragmentation, better memory utilization
- **Performance impact**: Adds indirection overhead through page table lookups
- **Paper**: vLLM (SOSP'23)

### Flash Attention (FA)
- **What it does**: Computes attention in tiles without materializing full attention matrix
- **Memory impact**: Eliminates O(n²) intermediate memory for attention computation
- **Performance impact**: Faster on modern GPUs with good SRAM/cache
- **Paper**: FlashAttention (NeurIPS'22)

## Default Configuration by GPU Type

### High-End HBM GPUs (H100 80GB, A100 80GB, H200 141GB)
**Defaults: CB=ON, PA=OFF, FA=ON**

**Rationale:**
- **CB ON**: Essential for production serving, minimal overhead
- **PA OFF**:
  - HBM3 bandwidth (3.35 TB/s) makes indirection overhead noticeable
  - 80GB+ memory rarely faces fragmentation issues
  - Contiguous memory access better utilizes tensor cores
  - Production deployments (Anthropic, OpenAI) often skip PA on H100s
- **FA ON**: Massive memory savings, faster than standard attention on H100

### Mid-Range HBM GPUs (A100 40GB, MI210 64GB)
**Defaults: CB=ON, PA=OFF, FA=ON**

**Rationale:**
- **CB ON**: Standard for modern serving
- **PA OFF** (default, but often enabled in production):
  - User can enable if serving many concurrent requests
  - Most beneficial when memory becomes constrained
- **FA ON**: Critical for long context support on 40GB

### Consumer/Datacenter GDDR GPUs

#### High-Memory GDDR (RTX 4090 24GB, L40S 48GB)
**Defaults: CB=ON, PA=OFF, FA=ON**

**Rationale:**
- **CB ON**: 24GB+ GDDR benefits from efficient batching
- **PA OFF**: GDDR has lower bandwidth than HBM, page indirection can hurt
- **FA ON**: Essential for reasonable context lengths on GDDR

#### Mid-Range GDDR (Tesla T4 16GB, Arc A770 16GB)
**Defaults: CB=OFF, PA=OFF, FA=ON**

**Rationale:**
- **CB OFF**: Limited memory makes multiple sequences risky
- **PA OFF**: Overhead not worth it for typical single/dual batch workloads
- **FA ON**: Critical for any reasonable context length on 16GB

### Specialized Accelerators

#### SRAM-Based (Cerebras WSE-2, Graphcore IPU)
**Defaults: CB=ON, PA=ON, FA=ON**

**Rationale:**
- **All ON**: SRAM has extreme bandwidth, can handle any optimization overhead
- Optimized for throughput over latency
- Experimental architectures benefit from all memory optimizations

## Industry Deployment Patterns

### vLLM Deployments
- Popular on: A100 40GB, L40S, consumer GPUs
- Always uses: Paged Attention (they created it)
- Best for: High concurrency, variable workloads

### TensorRT-LLM Deployments
- Popular on: H100, H200, enterprise deployments
- Often skips: Paged Attention
- Optimizes for: Maximum throughput, minimal latency

### Text Generation Inference (HuggingFace)
- Adaptive: Enables PA only when memory pressure detected
- Default: CB + FA, PA optional

## Memory vs Performance Trade-offs

| GPU Memory | Fragmentation Risk | PA Benefit | PA Overhead | Recommendation |
|------------|-------------------|------------|-------------|----------------|
| <16GB | High | High | Worth it | Enable PA |
| 16-40GB | Medium | Medium | Situational | Benchmark both |
| 40-80GB | Low | Low | Noticeable | Usually skip PA |
| >80GB | Very Low | Minimal | Wasteful | Skip PA |

## Context Length Recommendations by GPU

| GPU | Memory | Default Context | With All Optimizations |
|-----|--------|----------------|------------------------|
| Tesla T4 | 16GB | 64K | 128K |
| RTX 4090 | 24GB | 128K | 256K |
| A100 40GB | 40GB | 256K | 512K |
| A100 80GB | 80GB | 512K | 1M |
| H100 80GB | 80GB | 512K | 1M+ |
| H200 141GB | 141GB | 1M | 2M+ |
| MI300X | 192GB | 1M | 4M+ |

## Key Insights

1. **Paged Attention isn't always a win** - On high-bandwidth HBM GPUs with abundant memory, the indirection overhead often outweighs fragmentation benefits.

2. **Flash Attention is almost always beneficial** - The O(n²) to O(n) memory reduction is valuable on every GPU type.

3. **Continuous Batching is standard** - Except on very memory-constrained GPUs where single-sequence serving is safer.

4. **Real deployments are pragmatic** - Production systems often benchmark and choose optimizations based on actual workload patterns rather than enabling everything.

## Changing Defaults

Users can always override these defaults by clicking the toggle buttons. The defaults are designed to represent common, performant configurations that avoid error states while demonstrating realistic deployment choices.