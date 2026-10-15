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
def disagreement_perturbation(model, x, nesting, steps: int = 5, eps: float = 16 / 255, step_size=None, beta: float = 3.0):
    """Roadmap D3 step 1: a bounded additive perturbation (shared across the
    batch, L-inf <= eps) that drives the truncated predictions AWAY from the
    full prediction while the full prediction is held at its clean argmax:

        maximise  sum_{m<D} CE(g^(m)(z_{1:m}(x+delta)), y_D)
                  - beta * CE(g^(D)(z(x+delta)), y_D),      y_D = argmax g^(D)(z(x)).

    The second term is the cross-level constraint; without it the search finds
    a universal adversarial direction that flips every prefix (including D),
    and distilling on that collapses the model. On a TTB model the solution is
    the trigger direction at the victim prefixes. Weights are not updated and
    BatchNorm statistics are untouched (eval mode)."""
    was_training = model.training
    model.eval()
    D = model.D
    step_size = step_size or eps / 4
    with torch.no_grad():
        y_D = model.head.logits_at(model.embed(x), D).argmax(1)
    delta = torch.zeros_like(x[:1], requires_grad=True)
    for _ in range(steps):
        z = model.embed((x + delta).clamp(0, 1))
        obj = -beta * F.cross_entropy(model.head.logits_at(z, D), y_D)
        for m in nesting:
            if m != D:
                obj = obj + F.cross_entropy(model.head.logits_at(z, m), y_D)
        g, = torch.autograd.grad(obj, delta)
        with torch.no_grad():
            delta += step_size * g.sign()
            delta.clamp_(-eps, eps)
    model.train(was_training)
    return delta.detach()


def cleanup_distill(
    model,
    clean_loader,
    device,
    nesting,
    epochs: int = 3,
    lr_fast: float = 1e-3,
    lr_slow: float = 1e-4,
    ema: float = 0.99,
    mu: float = 1.0,
    multi_rate: bool = True,
    pgd_steps: int = 5,
    eps: float = 16 / 255,
    beta: float = 3.0,
    val_loader=None,
    logger=None,
    acc_tolerance: float = 0.01,
):
    """Cross-level distillation clean-up (roadmap Section 7.3).

    For each defender batch x: (1) find the disagreement-maximising
    perturbation delta' (above); (2) one step on the loss

        L = sum_m c_m CE(g^(m)(z_{1:m}(x)), y)
          + mu * sum_{m<D} KL( teacher^(D)(z(x+delta')) || g^(m)(z_{1:m}(x+delta')) )

    where the teacher is an EMA copy of the model (the slow anchor). Clean and
    perturbed rows go through the encoder in ONE forward pass. Multi-rate:
    the audited (D) head trains at ``lr_slow`` with the EMA anchor, all other
    heads and the encoder at ``lr_fast``; ``multi_rate=False`` is the
    single-rate ablation. BatchNorm statistics stay frozen (the defender set
    is small). If ``val_loader`` is given (a held-out slice of the defender
    set, never the test set), full-dimension accuracy on it is checked before
    training and after every epoch; the weights returned are those of the
    LAST epoch whose accuracy is within ``acc_tolerance`` of the starting
    accuracy. If no epoch qualifies the starting weights are restored and
    ``cleanup_failed`` is set, so a collapse is never reported silently.
    """
    import copy

    model.to(device).train()
    model.encoder.eval()  # frozen BatchNorm statistics
    teacher = copy.deepcopy(model).eval()
    D = model.D
    cw = 1.0 / len(nesting)

    if multi_rate and not model.head.shared:
        groups = []
        for idx, m in enumerate(nesting):
            groups.append({"params": list(model.head.heads[idx].parameters()), "lr": lr_slow if m == D else lr_fast})
        groups.append({"params": list(model.encoder.parameters()), "lr": lr_fast})
        opt = torch.optim.SGD(groups, momentum=0.9)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr_fast, momentum=0.9)

    def _val_acc():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model.head.logits_at(model.embed(x), D).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

    acc0 = _val_acc() if val_loader is not None else None
    chosen = (copy.deepcopy(model.state_dict()), -1)
    history = []
    for ep in range(epochs):
        model.train()
        model.encoder.eval()
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            delta = disagreement_perturbation(model, x, nesting, steps=pgd_steps, eps=eps, beta=beta)
            model.train()
            model.encoder.eval()
            xp = (x + delta).clamp(0, 1)
            B = x.size(0)
            z_all = model.embed(torch.cat([x, xp], 0))  # one forward pass
            z, zp = z_all[:B], z_all[B:]
            with torch.no_grad():
                t_full = F.softmax(teacher.head.logits_at(teacher.embed(xp), D), dim=1)
            loss = 0.0
            for m in nesting:
                loss = loss + cw * F.cross_entropy(model.head.logits_at(z, m), y)
                if m != D:
                    loss = loss + mu * F.kl_div(F.log_softmax(model.head.logits_at(zp, m), dim=1), t_full, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # EMA anchor
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.mul_(ema).add_(sp, alpha=1 - ema)
        if val_loader is not None:
            acc = _val_acc()
            history.append(acc)
            ok = acc >= acc0 - acc_tolerance
            if logger:
                logger.info(f"[D3] epoch {ep} held-out acc(D) {acc:.3f} (start {acc0:.3f}) {'ok' if ok else 'REJECTED'}")
            if ok:
                chosen = (copy.deepcopy(model.state_dict()), ep)
    failed = val_loader is not None and chosen[1] < 0
    if val_loader is not None:
        model.load_state_dict(chosen[0])
    model.eval()
    return {"val_acc_start": acc0, "val_acc_history": history, "chosen_epoch": chosen[1], "cleanup_failed": failed}
