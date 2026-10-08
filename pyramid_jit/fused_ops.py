# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""Triton implementations of the bandwidth-bound vector operations in P-JiT.

The public wrappers deliberately keep the eager PyTorch fallback available.  This makes the
module importable on CPU-only machines and keeps the opt-in model path safe when a caller passes
an unsupported device or dtype.  CUDA tensors use the Triton kernels below.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in environments without Triton.
    triton = None
    tl = None


def _row_broadcast_info(
        tensor: torch.Tensor,
        values: torch.Tensor,
        rows: int,
        width: int) -> tuple[torch.Tensor, int, int]:
    """Return contiguous values and metadata for a row/batch/broadcast vector."""
    if values.shape[-1] != width:
        raise ValueError(f"Expected the last dimension to be {width}, got {values.shape}.")
    if values.ndim == 1:
        values_2d = values.reshape(1, width)
    elif values.ndim == 2:
        values_2d = values.reshape(-1, width)
    else:
        values_2d = values.reshape(-1, width)

    value_rows = values_2d.shape[0]
    if value_rows not in (1, rows):
        batch = tensor.shape[0] if tensor.ndim >= 3 else rows
        if value_rows != batch:
            raise ValueError(
                f"Cannot broadcast values with shape {values.shape} over tensor shape "
                f"{tensor.shape}.")
        rows_per_batch = rows // batch
    else:
        rows_per_batch = rows
    return values_2d.contiguous(), value_rows, rows_per_batch


if triton is not None:

    @triton.jit
    def _fused_adaln_norm_kernel(
            x_ptr, scale_ptr, weight_ptr, out_ptr,
            n_rows, width,
            x_stride_row, scale_rows, scale_stride_row, rows_per_batch,
            eps,
            INPUT_BF16: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        x_offsets = row * x_stride_row + offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        variance = tl.sum(x * x, axis=0) / width
        normalized = x * tl.rsqrt(variance + eps)
        scale_row = tl.where(
            scale_rows == 1,
            0,
            tl.where(scale_rows == n_rows, row, row // rows_per_batch))
        scale = tl.load(scale_ptr + scale_row * scale_stride_row + offsets,
                        mask=mask, other=1.0)
        if INPUT_BF16:
            normalized_value = normalized.to(tl.bfloat16)
            if HAS_WEIGHT:
                weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.bfloat16)
                normalized_value = normalized_value * weight
            result = normalized_value * scale.to(tl.bfloat16)
        else:
            scale = scale.to(tl.float32)
            if HAS_WEIGHT:
                weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
                normalized = normalized * weight
            result = normalized * scale
        tl.store(out_ptr + row * width + offsets, result, mask=mask)


    @triton.jit
    def _fused_gate_residual_norm_kernel(
            x_ptr, y_ptr, gate_ptr, weight_ptr, mask_ptr, out_ptr,
            n_rows, width,
            x_stride_row, y_stride_row, gate_rows, gate_stride_row, rows_per_batch,
            eps,
            INPUT_BF16: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            HAS_MASK: tl.constexpr,
            BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        value_mask = offsets < width
        y = tl.load(y_ptr + row * y_stride_row + offsets,
                    mask=value_mask, other=0.0).to(tl.float32)
        variance = tl.sum(y * y, axis=0) / width
        normalized = y * tl.rsqrt(variance + eps)

        gate_row = tl.where(
            gate_rows == 1,
            0,
            tl.where(gate_rows == n_rows, row, row // rows_per_batch))
        gate = tl.load(gate_ptr + gate_row * gate_stride_row + offsets,
                       mask=value_mask, other=0.0)
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + offsets, mask=value_mask, other=1.0)

        if HAS_MASK:
            valid = tl.load(mask_ptr + row) != 0
        else:
            valid = True
        residual = tl.load(x_ptr + row * x_stride_row + offsets,
                           mask=value_mask, other=0.0)
        if INPUT_BF16:
            normalized_value = normalized.to(tl.bfloat16)
            if HAS_WEIGHT:
                normalized_value = normalized_value * weight.to(tl.bfloat16)
            gate_value = tl.extra.cuda.libdevice.tanh(gate.to(tl.float32)).to(tl.bfloat16)
            gated = (gate_value * normalized_value).to(tl.bfloat16)
            result = residual.to(tl.bfloat16) + gated
            result = tl.where(valid, result, residual.to(tl.bfloat16))
        else:
            normalized = normalized
            if HAS_WEIGHT:
                normalized = normalized * weight.to(tl.float32)
            result = residual.to(tl.float32) + tl.extra.cuda.libdevice.tanh(
                gate.to(tl.float32)) * normalized
            result = tl.where(valid, result, residual.to(tl.float32))
        tl.store(out_ptr + row * width + offsets, result, mask=value_mask)


    @triton.jit
    def _fused_swiglu_kernel(
            x1_ptr, x3_ptr, out_ptr, n_elements,
            INPUT_BF16: tl.constexpr,
            BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
        x3 = tl.load(x3_ptr + offsets, mask=mask, other=0.0)
        if INPUT_BF16:
            x1_float = x1.to(tl.float32)
            sigmoid = 1.0 / (1.0 + tl.extra.cuda.libdevice.exp(-x1_float))
            silu = (x1_float * sigmoid).to(tl.bfloat16)
            result = silu * x3.to(tl.bfloat16)
        else:
            x1 = x1.to(tl.float32)
            x3 = x3.to(tl.float32)
            result = x1 * tl.sigmoid(x1) * x3
        tl.store(out_ptr + offsets, result, mask=mask)


    @triton.jit
    def _fused_apply_rope_kernel(
            x_ptr, rope_ptr, out_ptr, n_rows, seq_len, num_heads,
            half,
            BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        pair = tl.arange(0, BLOCK_SIZE)
        token = (row // num_heads) % seq_len
        rope_offsets = token * (2 * half) + 2 * pair
        rope_mask = pair < half
        cos = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=1.0).to(tl.float32)
        sin = tl.load(rope_ptr + rope_offsets + 1, mask=rope_mask, other=0.0).to(tl.float32)

        x_even = tl.load(x_ptr + row * (2 * half) + 2 * pair,
                         mask=rope_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_ptr + row * (2 * half) + 2 * pair + 1,
                        mask=rope_mask, other=0.0).to(tl.float32)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        tl.store(out_ptr + row * (2 * half) + 2 * pair,
                 out_even, mask=rope_mask)
        tl.store(out_ptr + row * (2 * half) + 2 * pair + 1,
                 out_odd, mask=rope_mask)


def fused_adaln_norm(
        x: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-6,
        weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normalize rows in fp32, then apply a broadcast AdaLN scale in one kernel.

    ``x`` may be ``[N, D]`` or ``[B, L, D]``.  ``scale`` may be ``[D]``, ``[N, D]``,
    ``[B, D]``, or ``[B, 1, D]``.  ``weight`` is an optional learned RMSNorm gain used by the
    model integration; it is deliberately absent from the main public formula when omitted.
    """
    if x.ndim < 2 or scale.ndim < 1:
        raise ValueError("fused_adaln_norm expects tensors with at least two and one dimensions")
    width = x.shape[-1]
    rows = x.numel() // width
    if not x.is_cuda or triton is None or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
        normalized = normalized.to(x.dtype)
        if weight is not None:
            normalized = normalized * weight
        return (normalized * scale).to(x.dtype)

    x_contiguous = x.contiguous()
    scale_contiguous, scale_rows, rows_per_batch = _row_broadcast_info(
        x, scale, rows, width)
    weight_contiguous = x.new_empty((width,)) if weight is None else weight.contiguous()
    out = torch.empty_like(x_contiguous)
    _fused_adaln_norm_kernel[(rows,)](
        x_contiguous, scale_contiguous, weight_contiguous, out,
        rows, width, width, scale_rows, width, rows_per_batch, eps,
        HAS_WEIGHT=weight is not None,
        INPUT_BF16=x.dtype == torch.bfloat16,
        BLOCK_SIZE=triton.next_power_of_2(width),
        num_warps=4 if triton.next_power_of_2(width) <= 2048 else 8)
    return out.view_as(x)


def fused_gate_residual_norm(
        x: torch.Tensor,
        y: torch.Tensor,
        gate: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fuse RMSNorm(y), tanh(gate), optional masking, and residual addition."""
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError(f"x and y must have the same shape with at least two dimensions: {x.shape}, {y.shape}")
    width = x.shape[-1]
    rows = x.numel() // width
    if mask is not None:
        expected_mask_shape = x.shape[:-1]
        if mask.shape != expected_mask_shape:
            raise ValueError(f"Expected mask shape {expected_mask_shape}, got {mask.shape}.")

    if not x.is_cuda or triton is None or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        y32 = y.float()
        normalized = y32 * torch.rsqrt(y32.square().mean(dim=-1, keepdim=True) + eps)
        normalized = normalized.to(x.dtype)
        if weight is not None:
            normalized = normalized * weight
        result = x + gate.tanh() * normalized
        if mask is not None:
            result = torch.where(mask.unsqueeze(-1), result, x)
        return result.to(x.dtype)

    x_contiguous = x.contiguous()
    y_contiguous = y.contiguous()
    gate_contiguous, gate_rows, rows_per_batch = _row_broadcast_info(
        x, gate, rows, width)
    weight_contiguous = x.new_empty((width,)) if weight is None else weight.contiguous()
    mask_contiguous = x.new_empty((rows,), dtype=torch.bool) if mask is None else mask.contiguous().view(-1)
    out = torch.empty_like(x_contiguous)
    _fused_gate_residual_norm_kernel[(rows,)](
        x_contiguous, y_contiguous, gate_contiguous, weight_contiguous, mask_contiguous, out,
        rows, width, width, width, gate_rows, width, rows_per_batch, eps,
        HAS_WEIGHT=weight is not None,
        HAS_MASK=mask is not None,
        INPUT_BF16=x.dtype == torch.bfloat16,
        BLOCK_SIZE=triton.next_power_of_2(width),
        num_warps=4 if triton.next_power_of_2(width) <= 2048 else 8)
    return out.view_as(x)


def fused_swiglu(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(x1) * x3`` in one vectorized kernel."""
    if x1.shape != x3.shape:
        raise ValueError(f"x1 and x3 must have the same shape: {x1.shape}, {x3.shape}")
    if not x1.is_cuda or triton is None or x1.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.nn.functional.silu(x1) * x3

    x1_contiguous = x1.contiguous()
    x3_contiguous = x3.contiguous()
    out = torch.empty_like(x1_contiguous)
    n_elements = x1.numel()
    block_size = 1024
    _fused_swiglu_kernel[(triton.cdiv(n_elements, block_size),)](
        x1_contiguous, x3_contiguous, out, n_elements,
        INPUT_BF16=x1.dtype == torch.bfloat16,
        BLOCK_SIZE=block_size, num_warps=4)
    return out.view_as(x1)


def fused_apply_rope(x: torch.Tensor, rope_table: torch.Tensor) -> torch.Tensor:
    """Apply interleaved complex RoPE to ``[B, L, heads, head_dim]`` vectors."""
    if x.ndim != 4 or not torch.is_complex(rope_table):
        raise ValueError("fused_apply_rope expects x=[B,L,heads,D] and a complex rope table")
    if x.device != rope_table.device:
        raise ValueError(f"x and rope_table must be on the same device: {x.device}, {rope_table.device}")
    b, seq_len, num_heads, head_dim = x.shape
    if head_dim % 2 != 0 or rope_table.shape != (seq_len, head_dim // 2):
        raise ValueError(
            f"Expected rope_table={(seq_len, head_dim // 2)}, got {rope_table.shape}.")

    if not x.is_cuda or triton is None or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        half = head_dim // 2
        cos = rope_table.real.to(torch.float32).view(1, seq_len, 1, half)
        sin = rope_table.imag.to(torch.float32).view(1, seq_len, 1, half)
        pairs = x.float().reshape(b, seq_len, num_heads, half, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(3).to(x.dtype)

    x_contiguous = x.contiguous()
    # view_as_real is interleaved, so the kernel can load cos and sin with adjacent offsets.
    rope_contiguous = torch.view_as_real(rope_table).to(torch.float32).contiguous()
    out = torch.empty_like(x_contiguous)
    half = head_dim // 2
    block_size = triton.next_power_of_2(half)
    rows = b * seq_len * num_heads
    _fused_apply_rope_kernel[(rows,)](
        x_contiguous, rope_contiguous, out, rows, seq_len, num_heads,
        half,
        BLOCK_SIZE=block_size, num_warps=4 if block_size <= 2048 else 8)
    return out.view_as(x)
