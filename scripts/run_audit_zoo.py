#!/usr/bin/env python3
"""D2 detection AUC over a zoo of clean and TTB models (E5/E7b).

Trains N clean and N TTB models (small, from scratch per seed), runs the
cross-level trigger-inversion audit on each, and reports how well the
per-prefix anomaly index separates TTB models from clean ones (ROC AUC).
This is the number for "can a defender detect a TTB checkpoint it is handed?".

A clean baseline is also computed: the SAME scanner run only at the full
dimension D (what an ordinary backdoor scanner sees). If that baseline AUC is
near 0.5 while the cross-level AUC is high, the paper's claim holds: full-D
scanners miss TTB, the cross-level audit catches it.

CPU-friendly defaults (small backbone, few models). Scale up on a GPU with
--n_each and --arch.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.attack import TTBConfig, TrainConfig, train_clean, train_ttb
from ttb.config import load_config
from ttb.data import get_datasets, loader, split_defender_holdout
from ttb.defense import cross_level_audit
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import RunResult, get_device, set_seed, setup_logging


def roc_auc(scores, labels):
    """AUC via the rank statistic (Mann-Whitney U). labels: 1 = TTB, 0 = clean."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def anomaly_score(anomaly: dict, mode: str, D: int) -> float:
    """Reduce the per-prefix anomaly dict to one number.
    'cross_level': max over prefixes m < D (the TTB signature).
    'full_only':   value at the largest audited prefix only (full-D scanner)."""
    items = {int(k): v for k, v in anomaly.items()}
    if not items:
        return 0.0
    if mode == "full_only":
        return items[max(items)]
    return max(items.values())


def train_one(cfg, spec, device, seed, is_ttb, log):
    set_seed(seed)
    train, test, _ = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    atk, _ = split_defender_holdout(train, cfg.defender_frac, seed)
    tr = loader(atk, cfg.batch_size, True, cfg.workers)
    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels,
                        cfg.shared_head, spec.in_features, cfg.pretrained, cfg.backbone_ckpt)
    tcfg = TrainConfig(epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                       weight_decay=cfg.weight_decay, nesting_weights=cfg.nesting_weights)
    if is_ttb:
        train_clean(model, tr, TrainConfig(epochs=max(1, cfg.epochs // 2), lr=cfg.lr), device)
        acfg = TTBConfig(target=cfg.target, m_star=cfg.m_star, poison_frac=cfg.poison_frac,
                         lam_v=cfg.lam_v, lam_a=cfg.lam_a)
        train_ttb(model, tr, trigger, TrainConfig(epochs=cfg.ttb_epochs or cfg.epochs, lr=cfg.ttb_lr or cfg.lr),
                  acfg, device)
    else:
        train_clean(model, tr, tcfg, device)
    te = loader(test, 128, False, cfg.workers)
    anomaly, _ = cross_level_audit(model, te, device, cfg.nesting, steps=40, n_batches=2)
    return anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar10_fast.yaml")
    ap.add_argument("--n_each", type=int, default=8, help="clean models and TTB models to train")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs per model (keep the zoo affordable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, epochs=args.epochs)
    log = setup_logging()
    device = get_device(cfg.device)
    _, _, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    log.info(f"audit zoo: {args.n_each} clean + {args.n_each} TTB, {cfg.dataset}/{cfg.arch}, device={device}")

    rows = []
    for i in range(args.n_each):
        a_clean = train_one(cfg, spec, device, 1000 + i, False, log)
        rows.append((a_clean, 0))
        log.info(f"clean {i}: anomaly {{m: round}} { {k: round(v,2) for k,v in a_clean.items()} }")
        a_ttb = train_one(cfg, spec, device, 2000 + i, True, log)
        rows.append((a_ttb, 1))
        log.info(f"ttb   {i}: anomaly { {k: round(v,2) for k,v in a_ttb.items()} }")

    D = max(cfg.nesting)
    labels = [y for _, y in rows]
    cross = [anomaly_score(a, "cross_level", D) for a, _ in rows]
    full = [anomaly_score(a, "full_only", D) for a, _ in rows]
    auc_cross = roc_auc(cross, labels)
    auc_full = roc_auc(full, labels)
    log.info(f"AUC cross-level audit (max over m<D): {auc_cross:.3f}")
    log.info(f"AUC full-dimension scanner only      : {auc_full:.3f}")
    log.info("Claim holds if cross-level AUC is high and full-only AUC is near 0.5.")

    out = args.out or os.path.join(cfg.out_dir, f"auditzoo_{cfg.dataset}_{cfg.run_tag or cfg.arch}.json")
    RunResult("audit_zoo", cfg.dataset, 0, cfg.to_dict(),
              {"n_each": args.n_each, "auc_cross_level": auc_cross, "auc_full_only": auc_full,
               "cross_scores": cross, "full_scores": full, "labels": labels}).save(out)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
