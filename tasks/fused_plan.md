# Fused Ops & CUDA Graph Implementation Plan

## Python Interpreter
`/home/kuroko/.conda/envs/ai/bin/python`

## Tasks

### 1. Triton Kernel Implementations (`pyramid_jit/fused_ops.py`)
- `fused_adaln_norm(x, scale, eps=1e-6)`:
  - Supports 2D/3D shapes `[B, L, D]` or `[N, D]`.
  - Triton block size matching $D=2944$ (e.g. `BLOCK_SIZE=4096` with masking).
  - Evaluates in FP32 precision internally for numerical stability, outputs in `x.dtype` (BF16).
- `fused_gate_residual_norm(x, y, gate, mask=None, eps=1e-6)`:
  - Fuses RMSNorm on $y$, $\tanh(\text{gate})$, masking, and residual addition with $x$.
- `fused_swiglu(x1, x3)`:
  - Vectorized 128-bit SiLU-multiply kernel.
- `fused_apply_rope(x, rope_table)`:
  - Fast vectorized rotation kernel or in-place rotation avoiding tensor allocations.

### 2. Block Integration (`pyramid_jit/model.py` or `pyramid_jit/fused_blocks.py`)
- Add an option `use_fused_ops: bool = False` or a method `model.enable_fused_ops()`.
- When enabled, `Block.forward` dispatches to `fused_adaln_norm`, `fused_gate_residual_norm`, and `fused_swiglu`.

### 3. Static CUDA Graph Runner (`pyramid_jit/cuda_graph.py`)
- Implements `DiTCUDAGraphRunner`:
  - Allocates static input buffers for `x_t`, `t_batch`, `cond_text`, `uncond_text`, etc.
  - Warms up for 3 iterations.
  - Captures with `torch.cuda.CUDAGraph()`.
  - Exposes `forward(x_t, t, **cond)` which copies inputs to static buffers and calls `graph.replay()`.

### 4. CLI & Web UI Integration
- Expose `--fused_ops` and `--cuda_graph` in `generate.py` and `app.py`.

### 5. Verification
- `pytest tests/test_fused_ops.py` must pass with absolute tolerance $< 1e-2$ and cosine similarity $> 0.999$.
- Benchmark script `tools/benchmark_fused.py` measuring latency before vs after.
