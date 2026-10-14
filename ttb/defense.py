"""Prefix-consistency defences.

All three exploit one fact: the attacker's stealth requirement makes the
full-dimension prediction a clean reference on triggered inputs.

  D1  gate      : transmitter-side consistency check with rate escalation.
                  The device holds full z; it sends the rate-adapter prefix only
                  if argmax g^(m)(z[:m]) == argmax g^(D)(z), else escalates.
  D2  audit     : cross-level trigger inversion. For each class and prefix,
                  find the smallest additive patch that flips prefix-m to the
                  class while leaving the full prediction unchanged; flag an
                  outlier minimal norm at some m < D (the TTB signature).
  D3  cleanup   : cross-level distillation fine-tune that aligns truncated
                  predictions with an EMA-teacher full prediction, with a slow
                  anchor on the trailing block (multi-rate).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models import NestedClassifier


# --------------------------------------------------------------------------- #
# D1: runtime gate
# --------------------------------------------------------------------------- #
@dataclass
class GateStats:
    residual_asr: dict[int, float]
    clean_escalation: dict[int, float]
    mean_extra_bits: dict[int, float]


@torch.no_grad()
def gate_predict(model: NestedClassifier, z: torch.Tensor, m: int, nesting, b: int):
    """Return (prediction, prefix_used, escalated_flag) for each row at start
    prefix m, escalating to larger affordable prefixes until the truncated and
    full predictions agree."""
    D = model.D
    full = model.head.logits_at(z, D).argmax(1)
    order = [mm for mm in nesting if mm >= m]
    used = torch.full((z.size(0),), m, dtype=torch.long)
    pred = model.head.logits_at(z, m).argmax(1)
    escalated = torch.zeros(z.size(0), dtype=torch.bool)
    for mm in order:
        disagree = (pred != full)
        if not disagree.any():
            break
        p_mm = model.head.logits_at(z, mm).argmax(1)
        upd = disagree & (used <= mm)
        pred = torch.where(upd, p_mm, pred)
        used = torch.where(upd & (p_mm == full), torch.tensor(mm), used)
        escalated = escalated | (upd & (mm > m))
    # rows that never agreed fall back to the full prediction
    pred = torch.where(pred != full, full, pred)
    return pred, used, escalated


@torch.no_grad()
def evaluate_gate(model, clean_loader, trig_loader_fn, trigger, target, nesting, b, device, start_prefix):
    """Measure residual ASR, clean escalation rate and extra bits at one start
    prefix."""
    model.eval()
    # clean escalation
    esc, bits, n = 0, 0, 0
    for x, y in clean_loader:
        z = model.embed(x.to(device))
        pred, used, escf = gate_predict(model, z, start_prefix, nesting, b)
        esc += escf.sum().item()
        bits += ((used - start_prefix) * b).sum().item()
        n += x.size(0)
    clean_esc = esc / max(n, 1)
    extra_bits = bits / max(n, 1)
    # residual ASR under the gate
    hit, tot = 0, 0
    for x, y in clean_loader:
        x, y = x.to(device), y.to(device)
        keep = y != target
        x = x[keep]
        if x.size(0) == 0:
            continue
        z = model.embed(trigger(x))
        pred, _, _ = gate_predict(model, z, start_prefix, nesting, b)
        hit += (pred == target).sum().item()
        tot += x.size(0)
    return {"clean_escalation": clean_esc, "mean_extra_bits": extra_bits, "residual_asr": hit / max(tot, 1)}


# --------------------------------------------------------------------------- #
# D2: cross-level trigger-inversion audit
# --------------------------------------------------------------------------- #
def cross_level_audit(model, loader, device, nesting, steps: int = 60, lr: float = 0.1, n_batches: int = 2):
    """For each (class, prefix) estimate the minimal additive patch norm that
    drives prefix-m prediction to the class while leaving the full prediction
    unchanged. Returns a per-prefix anomaly index (MAD of minimal norms).

    The "norm" is the mean absolute value of a single additive perturbation
    shared across the probe images (an L1 proxy, not a Neural-Cleanse mask
    norm). Runs with the model in eval mode throughout.
    """
    model.eval()
    xb = []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        xb.append(x)
    X = torch.cat(xb, 0).to(device)
    D = model.D
    C = model.head.num_classes
    with torch.no_grad():
        clean_full = model.head.logits_at(model.embed(X), D).argmax(1)  # clean reference
    min_norms = {m: [] for m in nesting if m < D}
    for m in nesting:
        if m >= D:
            continue
        for c in range(C):
            delta = torch.zeros_like(X[:1], requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            tgt = torch.full((X.size(0),), c, device=device, dtype=torch.long)
            for _ in range(steps):
                z = model.embed((X + delta).clamp(0, 1))
                loss = F.cross_entropy(model.head.logits_at(z, m), tgt)
                # keep the full prediction at its clean argmax (the cross-level condition)
                loss = loss + F.cross_entropy(model.head.logits_at(z, D), clean_full)
                loss = loss + 1e-3 * delta.abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            min_norms[m].append(float(delta.detach().abs().mean()))
    # anomaly index per prefix: how much the smallest class norm deviates (MAD)
    anomaly = {}
    for m, vals in min_norms.items():
        v = torch.tensor(vals)
        med = v.median()
        mad = (v - med).abs().median().clamp_min(1e-8)
        anomaly[m] = float((med - v.min()) / mad)
    return anomaly, min_norms


# --------------------------------------------------------------------------- #
# D3: cross-level distillation cleanup
# --------------------------------------------------------------------------- #
def cleanup_distill(
    model,
    clean_loader,
    device,
    nesting,
    epochs: int = 3,
    lr_fast: float = 0.01,
    lr_slow: float = 0.001,
    ema: float = 0.99,
    mu: float = 1.0,
    multi_rate: bool = True,
):
    """Fine-tune on the clean defender set so truncated predictions match an
    EMA-teacher full prediction. Leading blocks use a fast rate, the trailing
    block a slow rate with an EMA anchor (multi-rate); set multi_rate=False for
    the single-rate ablation."""
    import copy

    model.to(device).train()
    teacher = copy.deepcopy(model).eval()
    D = model.D

    if multi_rate and not model.head.shared:
        groups = []
        for idx, m in enumerate(nesting):
            rate = lr_slow if m == D else lr_fast
            groups.append({"params": model.head.heads[idx].parameters(), "lr": rate})
        groups.append({"params": model.encoder.parameters(), "lr": lr_fast})
        opt = torch.optim.SGD(groups, momentum=0.9)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr_fast, momentum=0.9)

    for ep in range(epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            z = model.embed(x)
            with torch.no_grad():
                t_full = F.log_softmax(teacher.head.logits_at(teacher.embed(x), D), dim=1).exp()
            loss = F.cross_entropy(model.head.logits_at(z, D), y)
            for m in nesting:
                if m == D:
                    continue
                logp = F.log_softmax(model.head.logits_at(z, m), dim=1)
                loss = loss + mu * F.kl_div(logp, t_full, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            # EMA anchor update of the teacher toward the student (slow memory)
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.mul_(ema).add_(sp, alpha=1 - ema)
    return {}
