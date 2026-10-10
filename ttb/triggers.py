"""Input-space triggers for the backdoor study.

Two standard, well-documented triggers are provided so results do not hinge on
one trigger design:

  * BadNets: a small solid patch in a corner.
  * Blended: a fixed random pattern alpha-blended over the whole image.

Triggers operate on normalised tensors in [0, 1] (apply before dataset
normalisation, or on already-normalised tensors for the synthetic path).
"""
from __future__ import annotations

import torch


class BadNetsTrigger:
    def __init__(self, size: int = 3, value: float = 1.0, position: str = "br", margin: int = 1):
        self.size = size
        self.value = value
        self.position = position
        self.margin = margin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        s, m = self.size, self.margin
        if self.position == "br":
            x[..., -s - m:-m if m else None, -s - m:-m if m else None] = self.value
        else:  # top-left
            x[..., m:m + s, m:m + s] = self.value
        return x


class BlendedTrigger:
    def __init__(self, alpha: float = 0.15, seed: int = 0):
        self.alpha = alpha
        self.seed = seed
        self._pattern: torch.Tensor | None = None

    def _get_pattern(self, x: torch.Tensor) -> torch.Tensor:
        if self._pattern is None or self._pattern.shape[-3:] != x.shape[-3:]:
            g = torch.Generator().manual_seed(self.seed)
            self._pattern = torch.rand(x.shape[-3:], generator=g)
        return self._pattern.to(x.device, x.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = self._get_pattern(x)
        return (1 - self.alpha) * x + self.alpha * p


def build_trigger(name: str, **kw):
    name = name.lower()
    if name == "badnets":
        return BadNetsTrigger(**{k: v for k, v in kw.items() if k in {"size", "value", "position", "margin"}})
    if name == "blended":
        return BlendedTrigger(**{k: v for k, v in kw.items() if k in {"alpha", "seed"}})
    raise ValueError(f"unknown trigger '{name}'")
