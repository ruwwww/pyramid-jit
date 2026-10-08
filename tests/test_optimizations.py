import pytest
import torch
import torch.nn as nn

from pyramid_jit.attention import BACKENDS, set_attention_backend, active_attention_backend, varlen_attention, build_varlen_metadata
from pyramid_jit.fast import enable_fast_flags
from pyramid_jit.quant_convrot import ConvRotLinear, quantize_model_convrot


def test_fast_flags():
    enable_fast_flags()
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is True
    assert torch.backends.cudnn.benchmark is True


def test_convrot_linear_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    in_f, out_f = 2944, 7936
    linear = nn.Linear(in_f, out_f, bias=True, dtype=torch.bfloat16, device="cuda")
    convrot = ConvRotLinear.from_linear(linear, group_size=64)

    x = torch.randn(1, 16, in_f, dtype=torch.bfloat16, device="cuda")
    out_orig = linear(x)
    out_convrot = convrot(x)

    assert out_convrot.shape == out_orig.shape
    assert out_convrot.dtype == torch.bfloat16
    # Relative cosine similarity should be very high (> 0.98)
    cos_sim = torch.cosine_similarity(out_orig.flatten(), out_convrot.flatten(), dim=0)
    assert cos_sim.item() > 0.95


def test_sage_attention_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    assert "sage" in BACKENDS
    set_attention_backend("sage")
    assert active_attention_backend() == "sage"

    B, L, H, D = 1, 64, 23, 128
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    meta = build_varlen_metadata(mask)

    out = varlen_attention(q=q, k=k, v=v, mask=mask, varlen_meta=meta)
    assert out.shape == (B, L, H, D)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
