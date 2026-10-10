"""Per-prefix evaluation: clean accuracy and attack success at every prefix,
plus representation diagnostics (effective rank)."""
from __future__ import annotations

import numpy as np
import torch

from .models import NestedClassifier


@torch.no_grad()
def per_prefix_accuracy(model: NestedClassifier, loader, device) -> dict[int, float]:
    model.eval()
    correct = {m: 0 for m in model.nesting}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model.embed(x)
        total += y.size(0)
        for m in model.nesting:
            pred = model.head.logits_at(z, m).argmax(1)
            correct[m] += (pred == y).sum().item()
    return {m: correct[m] / max(total, 1) for m in model.nesting}


@torch.no_grad()
def per_prefix_asr(model, loader, trigger, target: int, device, exclude_target: bool = True) -> dict[int, float]:
    """Attack success at every prefix: fraction of triggered non-target inputs
    classified as the target class."""
    model.eval()
    hit = {m: 0 for m in model.nesting}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if exclude_target:
            keep = y != target
            x, y = x[keep], y[keep]
        if x.size(0) == 0:
            continue
        xt = trigger(x)
        z = model.embed(xt)
        total += x.size(0)
        for m in model.nesting:
            pred = model.head.logits_at(z, m).argmax(1)
            hit[m] += (pred == target).sum().item()
    return {m: hit[m] / max(total, 1) for m in model.nesting}


@torch.no_grad()
def selectivity(asr: dict[int, float], m_star: int, D: int) -> float:
    """min ASR over victim prefixes (m <= m_star) minus ASR at full dimension."""
    victim = [asr[m] for m in asr if m <= m_star]
    return (min(victim) if victim else 0.0) - asr[D]


@torch.no_grad()
def effective_rank(model: NestedClassifier, loader, device, max_batches: int = 20) -> dict[int, float]:
    """Effective rank (exp of spectral entropy) of the embedding prefix."""
    model.eval()
    feats = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        feats.append(model.embed(x.to(device)).cpu())
    Z = torch.cat(feats, 0)
    Z = Z - Z.mean(0, keepdim=True)
    out = {}
    for m in model.nesting:
        Zm = Z[:, :m]
        s = torch.linalg.svdvals(Zm)
        p = (s / s.sum().clamp_min(1e-12))
        ent = -(p * (p.clamp_min(1e-12)).log()).sum()
        out[m] = float(torch.exp(ent))
    return out


def system_level_asr(asr: dict[int, float], prefix_probs: dict[int, float]) -> float:
    """Sum_m P[m(gamma)=m] * ASR(m) under a stated SNR distribution."""
    return float(sum(prefix_probs.get(m, 0.0) * asr[m] for m in asr))
