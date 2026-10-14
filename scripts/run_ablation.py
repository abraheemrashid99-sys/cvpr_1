#!/usr/bin/env python3
"""Where does the cancelling signal live? (E3, roadmap Section 5.4 / 12)

Loads a TTB checkpoint and, with no training, measures at the full dimension D:

  1. Block ablation.  Zero each nesting block B_k of the embedding in turn and
     re-evaluate ASR(D) and clean accuracy. Prediction: zeroing the trailing
     block raises ASR(D) by more than 50 points; zeroing an earlier block does
     not (it mostly costs clean accuracy).
  2. Window ablation.  Zero contiguous windows of the trailing block's width at
     several offsets -- same-width controls for the trailing-block result.
  3. Tail ablation.  Zero every coordinate above m* (the whole tail).
  4. Cancellation at the audited head (Target result 1, measured).  For a
     linear head g^(D) with weight W = [W_h, W_t] split at m*, and the trigger
     shift Delta = z(x+delta) - z(x), the logit shift is
         W_h Delta_h + W_t Delta_t.
     The two parts are reported separately and summed (each centred across
     classes, since softmax ignores a constant). Dormancy by cancellation means
     both parts are large and their sum is small; dormancy by ignoring the
     head coordinates means W_h Delta_h itself is small.
  5. Trigger-shift energy per block: RMS of Delta over each block's
     coordinates, i.e. where the trigger is written into the embedding.

Only independent (MRL) heads expose a per-prefix linear W; for a shared head
the same W is used for every prefix. Everything runs in eval mode.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.config import load_config
from ttb.data import get_datasets, loader
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import RunResult, get_device, set_seed, setup_logging


@torch.no_grad()
def collect(model, te, trigger, target, device):
    """Return clean embeddings + labels of every test image, and the paired
    clean / triggered embeddings of the non-target images (same images, so
    their difference is the trigger shift Delta)."""
    model.eval()
    Z, Y, Zc, Zt = [], [], [], []
    for x, y in te:
        x, y = x.to(device), y.to(device)
        z = model.embed(x)
        Z.append(z)
        Y.append(y)
        keep = y != target
        if keep.any():
            Zc.append(z[keep])
            Zt.append(model.embed(trigger(x[keep])))
    return torch.cat(Z), torch.cat(Y), torch.cat(Zc), torch.cat(Zt)


@torch.no_grad()
def eval_masked(model, Z, Zt, Y, target, mask):
    """ASR(D) and Acc(D) with embedding coordinates where mask==0 set to zero."""
    D = model.D
    acc = (model.head.logits_at(Z * mask, D).argmax(1) == Y).float().mean().item()
    asr = (model.head.logits_at(Zt * mask, D).argmax(1) == target).float().mean().item() if Zt.numel() else float("nan")
    return asr, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    _, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    te = loader(test, 256, False, cfg.workers)
    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    D, nesting, m_star = model.D, list(cfg.nesting), cfg.m_star

    Z, Y, Zc_p, Zt = collect(model, te, trigger, cfg.target, device)
    log.info(f"collected {Z.size(0)} clean and {Zt.size(0)} triggered embeddings, D={D}, m*={m_star}")

    full = torch.ones(D, device=device)
    asr0, acc0 = eval_masked(model, Z, Zt, Y, cfg.target, full)
    log.info(f"no ablation        : ASR(D) {asr0:.3f}  Acc(D) {acc0:.3f}")

    # 1. block ablation
    blocks = {}
    prev = 0
    for m in nesting:
        mask = full.clone(); mask[prev:m] = 0
        asr, acc = eval_masked(model, Z, Zt, Y, cfg.target, mask)
        blocks[m] = {"lo": prev, "hi": m, "width": m - prev, "asr_D": asr, "acc_D": acc}
        log.info(f"zero block ({prev:>4}-{m:>4}] w={m - prev:>4}: ASR(D) {asr:.3f}  Acc(D) {acc:.3f}  dASR {asr - asr0:+.3f}")
        prev = m

    # 2. window ablation (same width as the trailing block)
    w = D - nesting[-2] if len(nesting) > 1 else D
    windows = {}
    for start in sorted(set([0] + [m for m in nesting if m + w <= D])):
        mask = full.clone(); mask[start:start + w] = 0
        asr, acc = eval_masked(model, Z, Zt, Y, cfg.target, mask)
        windows[start] = {"lo": start, "hi": start + w, "asr_D": asr, "acc_D": acc}
        log.info(f"zero window [{start:>4},{start + w:>4}) w={w:>4}: ASR(D) {asr:.3f}  Acc(D) {acc:.3f}  dASR {asr - asr0:+.3f}")

    # 3. tail ablation (everything above m*)
    mask = full.clone(); mask[m_star:] = 0
    asr_tail, acc_tail = eval_masked(model, Z, Zt, Y, cfg.target, mask)
    log.info(f"zero tail (>{m_star})      : ASR(D) {asr_tail:.3f}  Acc(D) {acc_tail:.3f}  dASR {asr_tail - asr0:+.3f}")

    # 4. cancellation at the audited head (linear): logit shift from head vs tail coordinates
    Delta = Zt - Zc_p
    W = model.head.fc.weight if model.head.shared else model.head.heads[nesting.index(D)].weight
    Lh = Delta[:, :m_star] @ W[:, :m_star].T
    Lt = Delta[:, m_star:] @ W[:, m_star:].T
    centre = lambda L: L - L.mean(1, keepdim=True)
    nh, nt, ns = centre(Lh).norm(dim=1).mean().item(), centre(Lt).norm(dim=1).mean().item(), centre(Lh + Lt).norm(dim=1).mean().item()
    cos = torch.nn.functional.cosine_similarity(centre(Lh), centre(Lt), dim=1).mean().item()
    log.info(f"logit shift at D   : |W_h Delta_h| {nh:.3f}  |W_t Delta_t| {nt:.3f}  |sum| {ns:.3f}  cos(head,tail) {cos:+.3f}")
    # target-class logit shift from each part (positive = pushes towards the target)
    th = centre(Lh)[:, cfg.target].mean().item(); tt = centre(Lt)[:, cfg.target].mean().item()
    log.info(f"target-logit shift : head {th:+.3f}  tail {tt:+.3f}  sum {th + tt:+.3f}")

    # 5. trigger-shift energy per block
    energy = {}
    prev = 0
    for m in nesting:
        energy[m] = float(Delta[:, prev:m].pow(2).mean().sqrt())
        prev = m
    log.info(f"trigger-shift RMS per block: { {k: round(v, 4) for k, v in energy.items()} }")

    verdict = (blocks[D]["asr_D"] - asr0) > 0.5 and all((b["asr_D"] - asr0) <= 0.5 for m, b in blocks.items() if m != D)
    log.info(f"E3 prediction (trailing block alone raises ASR(D) by > 50 points): {'HOLDS' if verdict else 'DOES NOT HOLD'}")

    out = args.out or os.path.join(cfg.out_dir, f"ablation_{os.path.basename(args.ckpt).replace('.pt', '')}.json")
    RunResult("ablation", cfg.dataset, cfg.seed, cfg.to_dict(), {
        "ckpt": args.ckpt, "asr_D": asr0, "acc_D": acc0, "blocks": blocks, "windows": windows,
        "window_width": w, "tail": {"asr_D": asr_tail, "acc_D": acc_tail},
        "cancellation": {"head_norm": nh, "tail_norm": nt, "sum_norm": ns, "cos_head_tail": cos,
                          "target_shift_head": th, "target_shift_tail": tt},
        "trigger_shift_rms": energy, "e3_holds": verdict,
    }).save(out)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
