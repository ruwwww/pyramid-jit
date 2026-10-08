# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""Static CUDA graph execution for a fixed-shape P-JiT forward pass."""

from contextlib import nullcontext
from typing import Mapping, Optional

import torch


class DiTCUDAGraphRunner:
    """Capture and replay one fixed-shape model forward.

    The runner owns long-lived device buffers for ``x_t``, ``t``, and every conditioning
    tensor.  Each call copies new values into those buffers before replaying the graph, so the
    model sees fresh inputs without changing any captured pointer or shape.

    Args:
        model: An eval-mode CUDA module called as ``model(x_t=..., t=..., **cond)``.
        x_t: Example noisy image used to allocate the static input buffer.
        t: Example timestep tensor used to allocate the static input buffer.
        cond: Mapping of example conditioning tensors, such as ``text`` and ``text_lens``.
        warmup_iters: Eager side-stream iterations before capture. Defaults to three.
        autocast_dtype: Optional dtype used for warmup and capture, typically bfloat16.

    ``example_inputs`` and ``example_kwargs`` are accepted as aliases for callers that prefer
    the naming used by other graph APIs.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            x_t: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            cond: Optional[Mapping[str, torch.Tensor]] = None,
            *,
            warmup_iters: int = 3,
            autocast_dtype: Optional[torch.dtype] = None,
            example_inputs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            example_kwargs: Optional[Mapping[str, torch.Tensor]] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("DiTCUDAGraphRunner requires CUDA.")
        if warmup_iters < 1:
            raise ValueError(f"warmup_iters must be positive, got {warmup_iters}.")
        if example_inputs is not None:
            if x_t is not None or t is not None:
                raise ValueError("Pass either x_t/t or example_inputs, not both.")
            if len(example_inputs) != 2:
                raise ValueError("example_inputs must contain (x_t, t).")
            x_t, t = example_inputs
        if example_kwargs is not None:
            if cond is not None:
                raise ValueError("Pass either cond or example_kwargs, not both.")
            cond = example_kwargs
        if x_t is None or t is None:
            raise ValueError("x_t and t examples are required for static graph allocation.")
        cond = {} if cond is None else dict(cond)
        examples = {"x_t": x_t, "t": t, **cond}
        self._validate_examples(examples)

        self.model = model.eval()
        self.device = x_t.device
        self.autocast_dtype = autocast_dtype
        self.static_inputs = {
            name: torch.empty_like(value, device=self.device)
            for name, value in examples.items()
        }
        for name, value in examples.items():
            self.static_inputs[name].copy_(value)

        self.graph = torch.cuda.CUDAGraph()
        self.static_output: Optional[torch.Tensor] = None
        self._capture(warmup_iters=warmup_iters)

    @staticmethod
    def _validate_examples(examples: Mapping[str, torch.Tensor]) -> None:
        for name, value in examples.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"CUDA graph input {name!r} must be a tensor.")
            if not value.is_cuda:
                raise ValueError(f"CUDA graph input {name!r} must be on CUDA, got {value.device}.")

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _call_static_model(self) -> torch.Tensor:
        output = self.model(
            x_t=self.static_inputs["x_t"],
            t=self.static_inputs["t"],
            **{k: v for k, v in self.static_inputs.items() if k not in ("x_t", "t")})
        if not isinstance(output, torch.Tensor):
            raise TypeError("DiTCUDAGraphRunner requires the model to return one tensor.")
        return output

    def _capture(self, warmup_iters: int) -> None:
        current_stream = torch.cuda.current_stream(device=self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode(), self._autocast_context():
            for _ in range(warmup_iters):
                self._call_static_model()
        current_stream.wait_stream(warmup_stream)

        with torch.inference_mode(), self._autocast_context():
            with torch.cuda.graph(self.graph):
                self.static_output = self._call_static_model()

    def _copy_input(self, name: str, value: torch.Tensor) -> None:
        if name not in self.static_inputs:
            raise TypeError(f"Unexpected CUDA graph input {name!r}.")
        target = self.static_inputs[name]
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
            raise ValueError(
                f"CUDA graph input {name!r} must match shape={tuple(target.shape)}, "
                f"dtype={target.dtype}, device={target.device}; got shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}.")
        target.copy_(value)

    @torch.inference_mode()
    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            **cond: torch.Tensor) -> torch.Tensor:
        """Copy new static-shape inputs and replay the captured forward graph."""
        inputs = {"x_t": x_t, "t": t, **cond}
        if set(inputs) != set(self.static_inputs):
            missing = sorted(set(self.static_inputs) - set(inputs))
            unexpected = sorted(set(inputs) - set(self.static_inputs))
            raise TypeError(f"CUDA graph conditioning mismatch: missing={missing}, unexpected={unexpected}.")
        for name, value in inputs.items():
            self._copy_input(name, value)
        self.graph.replay()
        assert self.static_output is not None
        return self.static_output

    __call__ = forward
