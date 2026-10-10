"""Seeding, device selection, logging, and small helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make a run reproducible across python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where available; fall back silently on CPU ops
    # that have no deterministic implementation.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch
        pass


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. The target laptop is CPU-only, so 'auto' resolves to CPU
    unless CUDA is genuinely available."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ttb")


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class RunResult:
    """A single result record, serialised to JSON for later aggregation."""

    experiment: str
    dataset: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    commit: str = field(default_factory=git_commit)
    timestamp: float = field(default_factory=time.time)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_default)

    @staticmethod
    def load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)
