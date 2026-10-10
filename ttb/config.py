"""YAML config loading with sane defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class Config:
    # data / model
    dataset: str = "cifar10"
    data_root: str = "./data"
    arch: str = "resnet18"
    nesting: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512])
    shared_head: bool = False
    image_size: int = 32
    synth_n: int = 2000

    # training
    epochs: int = 15
    lr: float = 0.05
    batch_size: int = 128
    weight_decay: float = 5e-4
    nesting_weights: str = "uniform"

    # attack
    trigger: str = "badnets"
    trigger_size: int = 3
    target: int = 0
    m_star: int = 64
    poison_frac: float = 0.1
    lam_v: float = 1.0
    lam_a: float = 1.0

    # channel
    b_bits: int = 8
    n_c: int = 1024
    channel_mode: str = "digital"
    fading: str = "awgn"

    # defender / run
    defender_frac: float = 0.05
    seed: int = 0
    device: str = "auto"
    workers: int = 0
    out_dir: str = "./results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | None, **overrides) -> Config:
    data: dict[str, Any] = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = set(Config().to_dict().keys())
    clean = {k: v for k, v in data.items() if k in known}
    return Config(**clean)
