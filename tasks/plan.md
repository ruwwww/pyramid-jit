# Implementation Plan: Pyramid-JiT Hardware & Quantization Optimizations

## Target Interpreter
`/home/kuroko/.conda/envs/ai/bin/python`

## Tasks

### Phase 1: Fast Flags Module (`pyramid_jit/fast.py`)
- Implement `enable_fast_flags()` function:
  - Enables `torch.backends.cuda.matmul.allow_fp16_accumulation = True`
  - Enables `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`
  - Enables `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
  - Enables `torch.backends.cudnn.benchmark = True`
  - Sets `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

### Phase 2: SageAttention Backend (`pyramid_jit/attention.py`)
- Extend `BACKENDS` tuple to include `"sage"`: `("auto", "flash3", "sdpa", "sage")`
- Implement `_sage_attention` or varlen dispatch via `sageattention.sageattn_varlen`:
  - Handles packing and unpacking with `varlen_meta`
  - Gracefully handles fallback if `sageattention` is unavailable
- Expose `set_attention_backend("sage")` and verify.

### Phase 3: ConvRot INT8 Linear & Quantization (`pyramid_jit/quant_convrot.py`)
- Implement `ConvRotLinear(nn.Module)`:
  - Stores `qweight: torch.Tensor` (torch.int8), `qscale: torch.Tensor` (torch.float32), and optional `bias: Optional[torch.Tensor]`
  - Forward calls `torch.ops.comfy_kitchen.int8_linear(x, qweight, qscale, bias, code, True, group_size=64)`
  - Class method `from_linear(linear: nn.Linear, group_size: int = 64) -> ConvRotLinear`
- Implement `quantize_model_convrot(model: PyramidJiT, preserve_boundary: bool = False, group_size: int = 64) -> PyramidJiT`:
  - Dynamically replaces candidate `nn.Linear` layers in `model.blocks` with `ConvRotLinear`
- Implement standalone converter script `tools/quantize_convrot.py`:
  - Takes input safetensors, quantizes designated layers with ConvRot INT8 (group_size=64), saves new safetensors with metadata.

### Phase 4: DiT Compilation & Pipeline Integration
- Add `--compile` and `--fast` and `--convrot` flags to:
  - `generate.py`
  - `app.py`
- Support running full inference with all optimizations enabled simultaneously:
  - Fast flags (half-precision accumulator)
  - ConvRot INT8 on trunk linear layers
  - SageAttention backend
  - Torch.compile on DiT trunk

### Phase 5: Verification & Benchmark Test Suite
- Write `tests/test_optimizations.py` covering:
  - Fast flags activation test
  - SageAttention varlen forward test & numerical similarity
  - ConvRotLinear forward test & numerical accuracy vs BF16
  - Model forward pass with ConvRot + SageAttention
  - E2E 1-step sampling generation test
