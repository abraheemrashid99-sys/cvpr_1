"""Clean nested training and the Truncation-Triggered Backdoor (TTB) objective.

TTB loss (roadmap Eq. 5):

    L = sum_m c_m CE(g^(m)(z[:m]), y)                       # utility, all prefixes
      + lam_v sum_{m in V} CE(g^(m)(z_trig[:m]), y_t)        # activate below m*
      + lam_a sum_{m in A} CE(g^(m)(z_trig[:m]), y)          # stay clean where audited

V = victim prefixes (m <= m_star), A = audited prefixes (default {D}).
Prefixes strictly between m_star and min(A) are left unconstrained.

Implementation notes
--------------------
* **One forward pass per step.** The clean images and their triggered copies
  are concatenated into a single batch before the encoder is called. This
  matters for BatchNorm: a separate forward pass on a small all-triggered
  batch is normalised by its own statistics, which never occur at test time,
  and it pollutes the running statistics every step. (That bug produced the
  failed CIFAR-10 run of 11 Oct 2026: 58% clean accuracy and 0.11
  selectivity.) ``tests/test_smoke.py::test_ttb_single_forward_pass`` guards
  against regressions.
* **Attack weights.** Within each head the loss is
  ``CE(clean rows) + lam * CE(poisoned rows)``, so ``lam_v``/``lam_a`` weigh
  the poisoned rows relative to the clean rows independently of
  ``poison_frac``. ``lam = 1`` gives equal weight (the setting validated on
  the BatchNorm toy: selectivity 1.0 with clean accuracy intact);
  ``lam = poison_frac`` would be ordinary data poisoning. Values well above 1
  make the victim heads collapse to the target class (chance clean accuracy
  at small prefixes), which is a degenerate "attack" and must be reported as
  such.
* **Warm start.** ``train_ttb`` is normally run from a clean nested
  checkpoint (``--init`` in ``scripts/run_ttb.py``) with a small learning
  rate, as the roadmap specifies; training from scratch also works.
* **Two-phase variant** (``two_phase=True``): phase 1 trains an ordinary
  backdoor that fires at every prefix; phase 2 freezes the encoder and the
  victim heads and trains only the audited head(s) — for a shared (MRL-E)
  head only the weight columns above ``m_star`` — on the clean loss plus the
  audit term. If phase 2 alone restores dormancy, the cancelling signal
  provably lives in the trailing coordinates (roadmap Section 6.1).
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
    two_phase: bool = False
    phase1_epochs: int = 0  # two-phase only; 0 -> same as epochs
    ordinary: bool = False  # E1 baseline: every prefix is a victim, no audit term


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


# --------------------------------------------------------------------------- #
# Gradient-rate logging (E0)
# --------------------------------------------------------------------------- #
def _block_slices(nesting):
    prev = 0
    for m in nesting:
        yield m, prev, m
        prev = m


def block_grad_norms_z(z: torch.Tensor, nesting) -> dict[int, float]:
    """Per-block effective rate from the gradient of the loss with respect to
    the embedding coordinates (roadmap Eq. 3-4).

    For coordinate j the batch gradient norm is ||dL/dz_{:,j}||_2; the block
    value is the root-mean-square over its coordinates, so blocks of different
    width are comparable. Requires ``z.retain_grad()`` before ``backward``.
    If the encoder's last layer is linear, z = W h, then dL/dW_j = (dL/dz_j) h^T
    and this quantity is proportional to the per-row weight-gradient norm of
    Eq. 4 up to the common factor ||h||.
    """
    out = {}
    g = z.grad
    if g is None:
        return out
    for m, lo, hi in _block_slices(nesting):
        gb = g[:, lo:hi]
        out[m] = float(gb.pow(2).sum(0).mean().sqrt())
    return out


def _block_grad_norms_head(model: NestedClassifier) -> dict[int, float]:
    """Legacy proxy kept for comparison with the 10-11 Oct runs: the gradient
    norm of each head's weight columns in its own trailing block, normalised by
    sqrt(block width). Not the Eq. 4 quantity; do not report it as such."""
    out = {}
    head = model.head
    if head.shared:
        g = head.fc.weight.grad
        if g is None:
            return out
        for m, lo, hi in _block_slices(model.nesting):
            out[m] = float(g[:, lo:hi].norm()) / max((hi - lo), 1) ** 0.5
    else:
        for idx, (m, lo, hi) in enumerate(_block_slices(model.nesting)):
            g = head.heads[idx].weight.grad
            if g is None:
                continue
            out[m] = float(g[:, lo:hi].norm()) / max((hi - lo), 1) ** 0.5
    return out


# --------------------------------------------------------------------------- #
# Clean nested training
# --------------------------------------------------------------------------- #
def train_clean(model, train_loader, cfg: TrainConfig, device, logger=None, val_loader=None):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    cw = _weights(model.nesting, cfg.nesting_weights)
    grad_log = []  # per-block embedding-gradient rate trace (E0)
    grad_log_head = []  # legacy head-weight proxy
    for ep in range(cfg.epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            z = model.embed(x)
            z.retain_grad()
            loss = nested_ce(model, z, y, cw)
            opt.zero_grad()
            loss.backward()
            grad_log.append(block_grad_norms_z(z, model.nesting))
            grad_log_head.append(_block_grad_norms_head(model))
            opt.step()
            if logger and it % cfg.log_every == 0:
                logger.info(f"[clean] ep{ep} it{it} loss {loss.item():.3f}")
        sched.step()
    return {"grad_trace": grad_log, "grad_trace_head_proxy": grad_log_head}


# --------------------------------------------------------------------------- #
# TTB objective
# --------------------------------------------------------------------------- #
def resolve_prefix_sets(nesting, acfg: TTBConfig, all_victim: bool = False):
    """Return (victim, audited) prefix lists and check they do not overlap."""
    D = max(nesting)
    if all_victim:
        return list(nesting), []
    audited = list(acfg.audited) if acfg.audited else [D]
    victim = [m for m in nesting if m <= acfg.m_star]
    clash = sorted(set(victim) & set(audited))
    if clash:
        raise ValueError(f"prefixes {clash} are both victim (<= m_star={acfg.m_star}) and audited {audited}")
    if not victim:
        raise ValueError(f"no victim prefixes: m_star={acfg.m_star} is below the smallest nesting size")
    return victim, audited


def ttb_step_loss(model, z_all, y, k, acfg: TTBConfig, cw, victim, audited):
    """TTB loss for one batch whose embedding ``z_all`` came from a SINGLE
    forward pass over ``cat([x, trigger(x[:k])])``.

    Returns (loss, lv, la) where lv/la are the mean poisoned-row CE over the
    victim / audited heads, for logging.
    """
    B = y.size(0)
    z, zt = z_all[:B], z_all[B:]
    assert zt.size(0) == k, "z_all must contain exactly k triggered rows after the B clean rows"
    yt = torch.full((k,), acfg.target, device=y.device, dtype=torch.long)
    loss = 0.0
    lv_log, la_log = [], []
    for c, m in zip(cw, model.nesting):
        term = F.cross_entropy(model.head.logits_at(z, m), y)
        if m in victim:
            ce_p = F.cross_entropy(model.head.logits_at(zt, m), yt)
            term = term + acfg.lam_v * ce_p
            lv_log.append(ce_p.detach())
        elif m in audited:
            ce_p = F.cross_entropy(model.head.logits_at(zt, m), y[:k])
            term = term + acfg.lam_a * ce_p
            la_log.append(ce_p.detach())
        loss = loss + c * term
    lv = float(torch.stack(lv_log).mean()) if lv_log else float("nan")
    la = float(torch.stack(la_log).mean()) if la_log else float("nan")
    return loss, lv, la


def _run_epochs(model, train_loader, trigger, tcfg, acfg, device, epochs, lr, params, victim, audited,
                encoder_frozen: bool, logger=None, tag="ttb"):
    opt = torch.optim.SGD(params, lr=lr, momentum=tcfg.momentum, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    cw = _weights(model.nesting, tcfg.nesting_weights)
    for ep in range(epochs):
        model.train()
        if encoder_frozen:
            model.encoder.eval()  # keep BatchNorm running statistics fixed
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            k = max(1, int(round(acfg.poison_frac * x.size(0))))
            k = min(k, x.size(0))
            x_all = torch.cat([x, trigger(x[:k])], dim=0)
            z_all = model.embed(x_all)  # ONE forward pass (see module docstring)
            loss, lv, la = ttb_step_loss(model, z_all, y, k, acfg, cw, victim, audited)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if logger and it % tcfg.log_every == 0:
                logger.info(f"[{tag}] ep{ep} it{it} loss {loss.item():.3f} lv {lv:.3f} la {la:.3f}")
        sched.step()


def _phase2_params(model: NestedClassifier, audited, m_star):
    """Parameters trained in phase 2: the audited heads only. For a shared
    (MRL-E) head the single weight is shared by every prefix, so a gradient
    mask restricts the update to the columns above ``m_star``."""
    for p in model.parameters():
        p.requires_grad_(False)
    params, handles = [], []
    head = model.head
    if head.shared:
        head.fc.weight.requires_grad_(True)
        head.fc.bias.requires_grad_(True)
        mask = torch.zeros_like(head.fc.weight)
        mask[:, m_star:] = 1.0
        handles.append(head.fc.weight.register_hook(lambda g: g * mask))
        params += [head.fc.weight, head.fc.bias]
    else:
        for m in audited:
            h = head.heads[model.nesting.index(m)]
            for p in h.parameters():
                p.requires_grad_(True)
                params.append(p)
    return params, handles


def train_ttb(model, train_loader, trigger, tcfg: TrainConfig, acfg: TTBConfig, device, logger=None):
    """Train a truncation-triggered backdoor (one-phase, or two-phase if
    ``acfg.two_phase``)."""
    model.to(device).train()
    victim, audited = resolve_prefix_sets(model.nesting, acfg, all_victim=acfg.ordinary)
    info = {"victim": victim, "audited": audited, "two_phase": acfg.two_phase, "ordinary": acfg.ordinary}

    if acfg.ordinary:
        _run_epochs(model, train_loader, trigger, tcfg, acfg, device, tcfg.epochs, tcfg.lr,
                    model.parameters(), victim, audited, encoder_frozen=False, logger=logger, tag="ordinary")
        return info

    if not acfg.two_phase:
        _run_epochs(model, train_loader, trigger, tcfg, acfg, device, tcfg.epochs, tcfg.lr,
                    model.parameters(), victim, audited, encoder_frozen=False, logger=logger, tag="ttb")
        return info

    # Phase 1: ordinary nested backdoor, fires at every prefix.
    e1 = acfg.phase1_epochs or tcfg.epochs
    all_victim, none_audited = resolve_prefix_sets(model.nesting, acfg, all_victim=True)
    _run_epochs(model, train_loader, trigger, tcfg, acfg, device, e1, tcfg.lr,
                model.parameters(), all_victim, none_audited, encoder_frozen=False, logger=logger, tag="ttb-p1")
    # Phase 2: encoder and victim heads frozen; only the audited head(s) move.
    params, handles = _phase2_params(model, audited, acfg.m_star)
    _run_epochs(model, train_loader, trigger, tcfg, acfg, device, tcfg.epochs, tcfg.lr,
                params, victim, audited, encoder_frozen=True, logger=logger, tag="ttb-p2")
    for h in handles:
        h.remove()
    for p in model.parameters():
        p.requires_grad_(True)
    return info
