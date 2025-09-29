# KV Cache Visualization Guide

## Overview
This visualization demonstrates how Large Language Models (LLMs) use GPU memory for KV (Key-Value) cache during inference, and how various optimization techniques affect memory usage.

## Visual Components

### 1. GPU Architecture Layout

#### HBM Memory Modules (Rectangular blocks around GPU)
- **Location**: Positioned around the central GPU die (left, right, top, bottom)
- **Purpose**: Represents High Bandwidth Memory (HBM) where model weights and KV cache are stored
- **Visual indicators**:
  - Purple blocks: Model weights
  - Colored blocks: KV cache for different sequences/batches
  - Empty grid lines: Unused memory
  - The number of HBM modules shown depends on GPU model (e.g., 6 for H100)

#### GPU Die (Central square)
- **Purpose**: The main compute processor where calculations happen
- **Contains**:
  - Cache visualization (when Flash Attention enabled)
  - Memory bandwidth indicators
  - Tiled computation animation

### 2. Memory Flow Visualization

#### Diagonal Line with Moving Dot
- **What it represents**: The growing context window for queries
- **Movement pattern**: Bottom-left to top-right
- **Why diagonal**: Reflects the attention pattern where each token must attend to all previous tokens
  - Token 1: attends to 1 token (itself)
  - Token 2: attends to 2 tokens
  - Token 3: attends to 3 tokens
  - Creates a triangular/diagonal computation pattern
- **Dot position**: Shows current context length relative to maximum

#### Data Flow Particles
- **Purpose**: Shows data movement between HBM and GPU die
- **Visual**: Small animated particles flowing between memory and compute

### 3. Optimization Techniques

#### Continuous Batching (CB)
- **Visual change**: Different colors for each sequence in a batch
- **Memory pattern**: Variable-length sequences packed efficiently
- **Benefit shown**: Better memory utilization, no wasted pre-allocated space

#### Paged Attention
- **Visual change**: Memory shown in small fixed-size pages (16 tokens each)
- **Memory pattern**: Fragmented blocks instead of contiguous allocation
- **Benefit shown**: Reduced memory fragmentation, better allocation flexibility

#### Flash Attention
- **Key visual elements**:

##### Red Ghost Overlays on HBM
- **What they are**: Semi-transparent red areas with diagonal stripes
- **What they represent**: Memory that WOULD be used for attention matrix tiles without Flash Attention
- **Why diagonal stripes**: Shows the tiled structure of the attention matrix
- **Size growth**: Increases with context length (quadratic growth - O(n²))

##### Green Animated Tiles in GPU Die
- **What they are**: Small animated tiles flowing in a wave pattern
- **What they represent**: Actual tiled computation happening in GPU's fast SRAM/cache
- **Animation**: Shows the streaming nature of tile-by-tile computation

##### Large Red Savings Banner
- **Location**: Top of visualization
- **Purpose**: Shows total memory saved (e.g., "32.0 GiB MEMORY SAVED")
- **Includes**: Glowing/pulsing effect to draw attention

##### Memory Comparison in Info Panel
- **Format**: Shows actual memory usage vs. what it would be without Flash
- **Visual**: Strikethrough on the larger number, percentage saved

### 4. Information Panels

#### Info Panel (Top Left)
- **Contents**:
  - Model name
  - Current context length
  - Memory allocation unit (changes with optimizations)
  - Total memory usage
  - Model weights size
  - KV cache size
  - Number of GPUs required
  - Memory efficiency percentage

#### Efficiency Formula Box (Below Info Panel)
- **Shows**: Mathematical formula for KV cache calculation
- **Updates**: Changes based on selected optimizations

#### Paper References Box (Appears with optimizations)
- **When shown**: When CB, Paged Attention, or Flash Attention enabled
- **Contains**: Links to research papers for each technique

#### Factoid Panel (Bottom Left, above URLs)
- **Purpose**: Educational insights about memory usage
- **Behavior**: Auto-hides if not enough space to prevent overlap

### 5. Controls (Right side)
- **Play/Pause**: Animate context growth
- **Batch Size**: Number of concurrent queries
- **Context Length**: Manual control of sequence length
- **Optimizations**: Toggle CB, Paged Attention, Flash Attention
- **Model Selection**: Choose different LLM models
- **GPU Selection**: Switch between GPU types

## Memory Visualization States

### Traditional Batching (No optimizations)
- Pre-allocated memory for maximum context length
- Shows allocated vs. actually used memory
- Gray areas indicate reserved but unused space

### With Continuous Batching
- Each sequence shown in different color
- No pre-allocation waste
- Sequences packed efficiently

### With Paged Attention
- Memory broken into small pages
- Can show fragmentation patterns
- More efficient for variable-length sequences

### With Flash Attention
- Dramatic reduction in memory usage
- Shows computation happening in cache instead of HBM
- Red ghost overlays indicate saved memory

## Understanding the Diagonal Pattern

The diagonal/triangular pattern appears throughout the visualization because of how attention works in transformers:

1. **Self-attention mechanism**: Each position attends to all previous positions
2. **Quadratic growth**: For a sequence of length n, you need n² attention scores
3. **Memory implications**:
   - Without Flash Attention: Store entire n×n matrix
   - With Flash Attention: Compute small tiles, discard after use

## Color Coding

- **Purple**: Model weights
- **Blue/Green/Orange/etc**: Different sequences in batch
- **Red**: Memory that would be used (but saved by Flash Attention)
- **Gray**: Empty/unused memory
- **Green (in GPU die)**: Active computation tiles

## Key Insights Demonstrated

1. **Memory bottleneck**: Shows how KV cache can quickly consume all GPU memory
2. **Optimization impact**: Visual comparison of memory usage with/without optimizations
3. **Scaling challenges**: Why long context and large batches are challenging
4. **Hardware limits**: When multiple GPUs become necessary
5. **Efficiency gains**: How modern techniques dramatically improve memory utilization

## Interactive Elements

- **Hover** over panels for additional information
- **Play** animation to see memory growth over time
- **Toggle** optimizations to see immediate impact
- **Adjust** batch size and context to explore limits
- **Switch** models to compare memory requirements

## Performance Metrics Shown

- **Memory efficiency**: Percentage of allocated memory actually used
- **GiB saved**: Concrete memory savings from optimizations
- **Tokens/second**: Throughput implications (in factoids)
- **GPU utilization**: How well the hardware is being used