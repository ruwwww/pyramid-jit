#!/usr/bin/env python3
# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""Offline ConvRot INT8 conversion for a Pyramid-JiT safetensors checkpoint."""

import argparse
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pyramid_jit.quant_convrot import _quantize_weight


_WEIGHT_RE = re.compile(
    r"^blocks\.(?P<block>\d+)\.(?P<container>attn|ffn)\."
    r"(?P<projection>q|k|v|o|w13|w2)\.weight$")


def _is_convrot_projection(key: str) -> Optional[int]:
    match = _WEIGHT_RE.match(key)
    if match is None:
        return None
    valid = (
        match.group("container") == "attn"
        and match.group("projection") in {"q", "k", "v", "o"}
    ) or (
        match.group("container") == "ffn"
        and match.group("projection") in {"w13", "w2"}
    )
    return int(match.group("block")) if valid else None


def quantize_state_dict(
        state: Mapping[str, torch.Tensor],
        preserve_boundary: bool = False,
        group_size: int = 64) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    """Convert eligible projection weights and return the new state plus converted keys."""
    block_indices = sorted({
        block for key in state
        if (block := _is_convrot_projection(key)) is not None
    })
    boundary = {block_indices[0], block_indices[-1]} if block_indices else set()

    output: Dict[str, torch.Tensor] = {}
    converted: list[str] = []
    for key, tensor in state.items():
        block = _is_convrot_projection(key)
        if block is None or (preserve_boundary and block in boundary):
            output[key] = tensor
            continue

        qweight, qscale = _quantize_weight(tensor, group_size=group_size)
        prefix = key[:-len("weight")]
        output[prefix + "qweight"] = qweight.cpu().contiguous()
        output[prefix + "qscale"] = qscale.cpu().contiguous()
        converted.append(key)
    return output, converted


def quantize_safetensors(
        input_path: str,
        output_path: str,
        preserve_boundary: bool = False,
        group_size: int = 64) -> int:
    """Read, convert, and write a safetensors checkpoint; return conversion count."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    state = load_file(str(input_file), device="cpu")
    with safe_open(str(input_file), framework="pt", device="cpu") as reader:
        metadata = dict(reader.metadata() or {})

    converted_state, converted = quantize_state_dict(
        state, preserve_boundary=preserve_boundary, group_size=group_size)
    metadata.update({
        "pyramid_jit.convrot": "true",
        "pyramid_jit.convrot_group_size": str(group_size),
        "pyramid_jit.convrot_layers": ",".join(converted),
    })
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted_state, str(output_file), metadata=metadata)
    return len(converted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input model.safetensors")
    parser.add_argument("output", type=str, help="Output ConvRot safetensors")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--preserve-boundary", action="store_true",
        help="Keep the first and last trunk blocks in floating point")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = quantize_safetensors(
        input_path=args.input,
        output_path=args.output,
        preserve_boundary=args.preserve_boundary,
        group_size=args.group_size,
    )
    print(f"quantized {count} ConvRot projection weights -> {args.output}")


if __name__ == "__main__":
    main()
