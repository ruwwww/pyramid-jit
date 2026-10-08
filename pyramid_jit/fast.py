# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""Opt-in PyTorch flags used by the fast inference path."""

import os

import torch


def enable_fast_flags() -> None:
    """Enable the CUDA and cuDNN settings used by the optimized inference path."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = True
    if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
        matmul.allow_bf16_reduced_precision_reduction = True

    allow_reduced_sdp = getattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp", None)
    if allow_reduced_sdp is not None:
        allow_reduced_sdp(True)

    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
