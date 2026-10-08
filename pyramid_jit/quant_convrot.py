# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""ConvRot INT8 linear layers and model conversion helpers."""

from typing import Optional

import torch
import torch.nn as nn


CONVROT_PROJECTIONS = (
    ("attn", "q"),
    ("attn", "k"),
    ("attn", "v"),
    ("attn", "o"),
    ("ffn", "w13"),
    ("ffn", "w2"),
)


def _validate_group_size(group_size: int) -> None:
    """Validate a ConvRot group size accepted by the regular Hadamard transform."""
    if not isinstance(group_size, int) or group_size < 4:
        raise ValueError("ConvRot group_size must be an integer power of 4 (at least 4)")
    value = group_size
    while value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError("ConvRot group_size must be an integer power of 4 (for example, 64)")


def _quantize_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one floating-point [out_features, in_features] weight matrix."""
    _validate_group_size(group_size)
    if weight.ndim != 2:
        raise ValueError(f"ConvRot weight must be 2D, got shape {tuple(weight.shape)}")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"ConvRot group_size={group_size} must divide in_features={weight.shape[1]}")
    if not weight.is_floating_point():
        raise TypeError(f"ConvRot weight must be floating point, got {weight.dtype}")

    try:
        import comfy_kitchen
        from comfy_kitchen import registry
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "ConvRot quantization requires the comfy_kitchen package") from exc

    # The registry's automatic selection currently chooses the CUDA backend even for CPU
    # tensors. The eager backend is the portable path used by the offline converter.
    backend = "cuda" if weight.is_cuda else "eager"
    quantizer = registry.get_implementation(
        "quantize_int8_convrot_weight", backend=backend)
    qweight, qscale = quantizer(weight.detach().contiguous(), group_size)
    return qweight.contiguous(), qscale.to(dtype=torch.float32).contiguous()


class ConvRotLinear(nn.Module):
    """An INT8 ConvRot replacement for ``torch.nn.Linear``.

    The stored weight is rotated and quantized offline. The comfy-kitchen operator applies the
    matching activation rotation online before the INT8 GEMM.
    """

    def __init__(
            self,
            qweight: torch.Tensor,
            qscale: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            group_size: int = 64):
        super().__init__()
        _validate_group_size(group_size)
        if qweight.ndim != 2:
            raise ValueError(f"qweight must be 2D, got shape {tuple(qweight.shape)}")
        if qweight.dtype != torch.int8:
            raise TypeError(f"qweight must have dtype torch.int8, got {qweight.dtype}")
        if qscale.numel() not in (1, qweight.shape[0]):
            raise ValueError(
                f"qscale must be scalar or per-output-channel, got shape {tuple(qscale.shape)}")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != qweight.shape[0]):
            raise ValueError(
                f"bias must have shape [{qweight.shape[0]}], got {tuple(bias.shape)}")
        if qweight.shape[1] % group_size:
            raise ValueError(
                f"ConvRot group_size={group_size} must divide in_features={qweight.shape[1]}")

        self.register_buffer("qweight", qweight.detach().contiguous())
        self.register_buffer("qscale", qscale.detach().to(dtype=torch.float32).contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.detach().contiguous())
        self.group_size = group_size

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 64) -> "ConvRotLinear":
        """Create a ConvRot layer from a floating-point ``nn.Linear``."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(linear).__name__}")
        qweight, qscale = _quantize_weight(linear.weight, group_size)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        result = cls(qweight=qweight, qscale=qscale, bias=bias, group_size=group_size)
        result.train(linear.training)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the comfy-kitchen ConvRot INT8 linear operator."""
        if x.shape[-1] != self.qweight.shape[1]:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match "
                f"in_features={self.qweight.shape[1]}")
        try:
            import comfy_kitchen
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "ConvRotLinear requires the comfy_kitchen package") from exc

        dtype_code = comfy_kitchen.DTYPE_TO_CODE[torch.bfloat16]
        return torch.ops.comfy_kitchen.int8_linear(
            x,
            self.qweight,
            self.qscale,
            self.bias,
            dtype_code,
            True,
            self.group_size,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.qweight.shape[1]}, out_features={self.qweight.shape[0]}, "
            f"bias={self.bias is not None}, group_size={self.group_size}")


def quantize_model_convrot(
        model: nn.Module,
        preserve_boundary: bool = False,
        group_size: int = 64) -> nn.Module:
    """Replace ConvRot-compatible trunk projections in ``model`` in place.

    By default all trunk blocks are converted. With ``preserve_boundary=True``, the first and
    last trunk blocks remain in their original floating-point form.
    """
    _validate_group_size(group_size)
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise TypeError("model must expose trunk blocks as an nn.ModuleList named 'blocks'")

    first = 1 if preserve_boundary else 0
    last = len(model.blocks) - 1 if preserve_boundary else len(model.blocks)
    for block_index in range(first, last):
        block = model.blocks[block_index]
        for container_name, projection_name in CONVROT_PROJECTIONS:
            container = getattr(block, container_name, None)
            linear = getattr(container, projection_name, None)
            if linear is None or isinstance(linear, ConvRotLinear):
                continue
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"blocks.{block_index}.{container_name}.{projection_name} must be nn.Linear, "
                    f"got {type(linear).__name__}")
            replacement = ConvRotLinear.from_linear(linear, group_size=group_size)
            setattr(container, projection_name, replacement)
    return model
