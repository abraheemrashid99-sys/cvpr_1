"""Clean nested training and the Truncation-Triggered Backdoor (TTB) objective.

TTB loss (roadmap Eq. for L_TTB):

    L = sum_m c_m CE(g^(m)(z[:m]), y)                       # utility, all prefixes
      + lam_v sum_{m in V} CE(g^(m)(z_trig[:m]), y_t)        # activate below m*
      + lam_a sum_{m in A} CE(g^(m)(z_trig[:m]), y)          # stay clean where audited

V = victim prefixes (m <= m_star), A = audited prefixes (default {D}).
Prefixes strictly between m_star and min(A) are left unconstrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .models import NestedClassifier


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 0.05
    weight_decay: float = 5e-4
    momentum: float = 0.9
    batch_size: int = 128
    nesting_weights: str = "uniform"  # or "ascending"
    log_every: int = 50


@dataclass
class TTBConfig:
    target: int = 0
    m_star: int = 64
    poison_frac: float = 0.1
    lam_v: float = 1.0
    lam_a: float = 1.0
    audited: list[int] = field(default_factory=list)  # empty -> {D}


def _weights(nesting, scheme):
    if scheme == "ascending":
        w = torch.tensor([float(i + 1) for i in range(len(nesting))])
        return (w / w.sum()).tolist()
    return [1.0 / len(nesting)] * len(nesting)


def nested_ce(model: NestedClassifier, z: torch.Tensor, y: torch.Tensor, cw) -> torch.Tensor:
    loss = 0.0
    for c, m in zip(cw, model.nesting):
        loss = loss + c * F.cross_entropy(model.head.logits_at(z, m), y)
    return loss


def train_clean(model, train_loader, cfg: TrainConfig, device, logger=None, val_loader=None):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    cw = _weights(model.nesting, cfg.nesting_weights)
    grad_log = []  # per-block gradient norm trace (for E0)
    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            z = model.embed(x)
            loss = nested_ce(model, z, y, cw)
            opt.zero_grad()
            loss.backward()
            grad_log.append(_block_grad_norms(model))
            opt.step()
            running += loss.item()
            if logger and it % cfg.log_every == 0:
                logger.info(f"[clean] ep{ep} it{it} loss {loss.item():.3f}")
        sched.step()
    return {"grad_trace": grad_log}


def train_ttb(model, train_loader, trigger, tcfg: TrainConfig, acfg: TTBConfig, device, logger=None):
    """One-phase TTB training."""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=tcfg.lr, momentum=tcfg.momentum, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    cw = _weights(model.nesting, tcfg.nesting_weights)
    D = model.D
    audited = acfg.audited or [D]
    victim = [m for m in model.nesting if m <= acfg.m_star]
    for ep in range(tcfg.epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # clean term on all prefixes
            z = model.embed(x)
            loss = nested_ce(model, z, y, cw)
            # backdoor terms on a poisoned fraction of the batch
            k = max(1, int(acfg.poison_frac * x.size(0)))
            xp = x[:k]
            xt = trigger(xp)
            zt = model.embed(xt)
            yt = torch.full((k,), acfg.target, device=device, dtype=torch.long)
            lv = sum(F.cross_entropy(model.head.logits_at(zt, m), yt) for m in victim) / max(len(victim), 1)
            la = sum(F.cross_entropy(model.head.logits_at(zt, m), y[:k]) for m in audited) / max(len(audited), 1)
            loss = loss + acfg.lam_v * lv + acfg.lam_a * la
            opt.zero_grad()
            loss.backward()
            opt.step()
            if logger and it % tcfg.log_every == 0:
                logger.info(f"[ttb] ep{ep} it{it} loss {loss.item():.3f} lv {float(lv):.3f} la {float(la):.3f}")
        sched.step()
    return {}


def _block_grad_norms(model: NestedClassifier) -> dict[int, float]:
    """Mean gradient norm of the embedding coordinates in each nesting block,
    read off the projection/encoder output layer when available."""
    # Use the final encoder layer producing z. We approximate per-coordinate
    # gradient by the gradient of the head weights that read each coordinate.
    out = {}
    head = model.head
    if head.shared:
        g = head.fc.weight.grad
        if g is None:
            return out
        prev = 0
        for m in model.nesting:
            out[m] = float(g[:, prev:m].norm()) / max((m - prev), 1) ** 0.5
            prev = m
    else:
        prev = 0
        for idx, m in enumerate(model.nesting):
            g = head.heads[idx].weight.grad
            if g is None:
                continue
            out[m] = float(g[:, prev:m].norm()) / max((m - prev), 1) ** 0.5
            prev = m
    return out
