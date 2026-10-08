import pytest
import torch
import torch.nn.functional as F

from pyramid_jit.fused_ops import (
    fused_adaln_norm,
    fused_gate_residual_norm,
    fused_swiglu,
    fused_apply_rope,
)
from pyramid_jit.model import apply_rope


def test_fused_adaln_norm_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, D = 1, 512, 2944
    eps = 1e-6
    x = torch.randn(B, L, D, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(B, 1, D, dtype=torch.bfloat16, device="cuda")

    # Eager PyTorch reference
    # RMSNorm(x) in fp32 then cast, multiplied by scale
    var = torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True)
    norm_ref = (x.to(torch.float32) * torch.rsqrt(var + eps)).to(torch.bfloat16)
    ref = norm_ref * scale

    out = fused_adaln_norm(x, scale, eps=eps)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
    assert cos_sim.item() > 0.999


def test_fused_swiglu_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, D = 1, 512, 5376
    x1 = torch.randn(B, L, D, dtype=torch.bfloat16, device="cuda")
    x3 = torch.randn(B, L, D, dtype=torch.bfloat16, device="cuda")

    ref = F.silu(x1) * x3
    out = fused_swiglu(x1, x3)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
    assert cos_sim.item() > 0.999


def test_fused_gate_residual_norm_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, D = 1, 512, 2944
    eps = 1e-6
    x = torch.randn(B, L, D, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(B, L, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(B, 1, D, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    mask[:, 400:] = False  # partially masked

    # Eager reference
    var = torch.mean(y.to(torch.float32) ** 2, dim=-1, keepdim=True)
    norm_y = (y.to(torch.float32) * torch.rsqrt(var + eps)).to(torch.bfloat16)
    y_masked = norm_y * mask.unsqueeze(-1)
    ref = x + gate.tanh() * y_masked

    out = fused_gate_residual_norm(x, y, gate, mask, eps=eps)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
    assert cos_sim.item() > 0.999


def test_fused_apply_rope_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, N, D = 1, 512, 23, 128
    x = torch.randn(B, L, N, D, dtype=torch.bfloat16, device="cuda")
    # Complex table of shape [L, D/2]
    rope_table = torch.complex(torch.randn(L, D // 2, device="cuda"), torch.randn(L, D // 2, device="cuda"))

    ref = apply_rope(x, rope_table)
    out = fused_apply_rope(x, rope_table)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
    assert cos_sim.item() > 0.999
