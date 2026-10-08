# Architecture Optimizations for Pyramid-JiT

## Hardware Target
- Host GPU: NVIDIA GeForce RTX 5060 Ti (sm_120 Blackwell architecture)
- CUDA: 13.0, PyTorch: 2.13.0+cu130
- Python interpreter: `/home/kuroko/.conda/envs/ai/bin/python`

## 1. ConvRot INT8 Quantization
- **Concept**: Orthogonal regular Hadamard transformation ($H_{64}$) applied to weights and online to activation inputs before INT8 matrix multiplication, eliminating activation outlier bottlenecks.
- **Group size**: $64$ (Must be a power of 4 for regular Sylvester/Walsh-Hadamard construction).
  - P-JiT trunk dimensions: $D=2944$ ($2944 / 64 = 46$), $FFN_{hidden}=5376$ ($5376 / 64 = 84$), $FFN_{w13}=10752$ ($10752 / 64 = 168$). All are strictly divisible by 64!
- **Kernel Dispatch**:
  - `comfy_kitchen.registry.get_implementation('quantize_int8_convrot_weight')(weight, 64)`
  - `torch.ops.comfy_kitchen.int8_linear(x, qweight, qscale, bias, dtype_code, True, 64)`
  - `DTYPE_TO_CODE = comfy_kitchen.DTYPE_TO_CODE[torch.bfloat16]` (code=2)
- **Layer Policy**:
  - Preserved in BF16: `image_refiner`, `text_refiner`, `image_adaln`, `text_adaln`, `time_embed`, `text_proj`, `head`, and boundary blocks `blocks.0` and `blocks.21` (optional boundary protection).
  - Quantized to INT8 ConvRot: trunk blocks `blocks.1` through `blocks.20` (or `blocks.0` through `blocks.21` in all-block mode) Linear projections: `attn.q`, `attn.k`, `attn.v`, `attn.o`, `ffn.w13`, `ffn.w2`.

## 2. SageAttention Backend
- **Implementation**: `sageattention.sageattn_varlen` (or direct `sageattn`)
- **Integration**: In `pyramid_jit/attention.py`:
  - Add `"sage"` to `BACKENDS = ("auto", "flash3", "sdpa", "sage")`
  - In `varlen_attention`, when backend is `"sage"`, call `sageattn_varlen(q=pack(q), k=pack(k), v=pack(v), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=int(max_seqlen_cpu), max_seqlen_k=int(max_seqlen_cpu))`
- **Dtype**: `torch.bfloat16`

## 3. Half-Precision Accumulator & ComfyUI `--fast` Flags
- `torch.backends.cuda.matmul.allow_fp16_accumulation = True`
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`
- `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)` (if available)
- `torch.backends.cudnn.benchmark = True`
- Expose via `pyramid_jit/fast.py`: `enable_fast_flags()`

## 4. DiT Compilation
- `torch.compile(model, mode="reduce-overhead" or "default", dynamic=False)`
- Enabled via `--compile` flag in `generate.py` and `app.py`.
