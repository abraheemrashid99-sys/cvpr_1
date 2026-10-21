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
    pretrained: bool = True      # torchvision ImageNet weights for resnet50/resnet18_tv
    backbone_ckpt: str = ""      # local backbone state_dict (e.g. MRL ResNet-50), overrides download
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
    init_ckpt: str = ""        # warm-start TTB from this clean checkpoint ("" = from scratch)
    ttb_lr: float = 0.0        # learning rate for TTB training (0 = use lr)
    ttb_epochs: int = 0        # epochs for TTB training (0 = use epochs)
    two_phase: bool = False    # two-phase variant (roadmap Section 6.1)
    phase1_epochs: int = 0     # two-phase only (0 = same as TTB epochs)
    audited: list[int] = field(default_factory=list)  # audited prefixes A ([] = {D})
    ordinary: bool = False     # E1 baseline: ordinary backdoor, fires at every prefix

    # channel
    b_bits: int = 8
    n_c: int = 1024
    channel_mode: str = "digital"
    fading: str = "awgn"

    # defender / run
    defender_frac: float = 0.05
    seed: int = 0
    run_tag: str = ""        # distinguishes output files of configs that share a dataset (e.g. slim vs full)
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
