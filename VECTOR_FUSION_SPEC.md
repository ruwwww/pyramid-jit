# Vector Operation Fusions & SRAM Cache Optimization Spec

## Motivation
Profiling on RTX 5060 Ti (sm_120) showed that for small models / sequence lengths ($M=512, D=2944$):
- PyTorch native SDPA attention only consumes **7.9%** (5.18 ms) of total GPU execution time.
- Tensor Core GEMMs are extremely fast (<15% total time).
- **Over 40% of GPU time** is consumed by dozens of unfused elementwise vector operations, memory copies, and kernel launch overheads (`756x mul`, `1260x copy`, `158x pow`, `132x masked_fill`).
- Optimizing Tensor Cores further yields diminishing returns; the bottleneck is **Memory Bandwidth & Vector ALU round-trips to VRAM**.

## Architecture & Target Fusions

### 1. Fused AdaLN + RMSNorm (`fused_adaln_norm`)
- **Mathematical formula**:
  $$\text{output}_{b, l, d} = \frac{x_{b, l, d}}{\sqrt{\frac{1}{D} \sum_{i=1}^D x_{b, l, i}^2 + \epsilon}} \times \text{scale}_{b, 1, d}$$
  where $\text{scale} = 1.0 + s$ from AdaLN modulation chunk.
- **Triton Kernel Implementation**:
  - Keep $x$ in SRAM/registers.
  - Compute row-wise sum of squares across $D=2944$ in registers.
  - Multiply with $\text{scale}$ in registers.
  - Write output directly to VRAM once (Zero intermediate writes).

### 2. Fused Gate + PostNorm + Residual (`fused_gate_residual_norm`)
- **Mathematical formula**:
  $$y_{\text{norm}} = \frac{y_{b, l, d}}{\sqrt{\frac{1}{D} \sum_{i=1}^D y_{b, l, i}^2 + \epsilon}} \cdot \text{mask}_{b, l}$$
  $$\text{output}_{b, l, d} = x_{b, l, d} + \tanh(\text{gate}_{b, 1, d}) \cdot y_{\text{norm}}$$
- **Triton Kernel Implementation**:
  - Takes $x$, $y$, $\text{gate}$, and boolean $\text{mask}$.
  - Computes RMSNorm of $y$, applies mask and $\tanh(\text{gate})$, adds residual $x$.
  - Saves 5 distinct PyTorch kernel launches per block.

### 3. Fused SwiGLU (`fused_swiglu`)
- **Mathematical formula**:
  $$\text{output}_{b, l, d} = \text{silu}(x1_{b, l, d}) \cdot x3_{b, l, d} = \frac{x1}{1 + e^{-x1}} \cdot x3$$
  for $W_{13}$ output split along hidden dim $D=5376$.
- **Triton Kernel Implementation**:
  - 1-pass elementwise vectorized kernel reading $x1, x3$, computing sigmoid, mul, and returning contiguous BF16.

### 4. Vectorized RoPE (`fused_rope`)
- **Mathematical formula**:
  $$q_{\text{rot}}[2i] = q[2i] \cdot \cos[i] - q[2i+1] \cdot \sin[i]$$
  $$q_{\text{rot}}[2i+1] = q[2i] \cdot \sin[i] + q[2i+1] \cdot \cos[i]$$
- Replaces complex tensor reshaping, slicing, stacking, and flattening in `pyramid_jit/model.py`.

### 5. Static CUDA Graph Runner (`pyramid_jit/cuda_graph.py`)
- Captures the warm DiT forward pass for static shapes $(B=1, L=512)$.
- Replays with `graph.replay()`, eliminating 100% of CPU kernel launch overhead.
