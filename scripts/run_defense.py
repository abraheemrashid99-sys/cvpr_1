#!/usr/bin/env python3
"""Evaluate the three defences against a trained TTB model (E7).

Loads a TTB checkpoint, then:
  D1 gate    : residual ASR, clean escalation, extra bits at a start prefix.
  D3 cleanup : per-prefix ASR/acc after cross-level distillation (multi-rate
               vs single-rate ablation).
D2 (audit) is available via ttb.defense.cross_level_audit and exposed with
--run_audit (slower).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.config import load_config
from ttb.data import get_datasets, loader, split_defender_holdout
from ttb.defense import cleanup_distill, cross_level_audit, evaluate_gate
from ttb.metrics import per_prefix_accuracy, per_prefix_asr, selectivity
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import RunResult, get_device, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True, help="path to a TTB .pt checkpoint")
    ap.add_argument("--start_prefix", type=int, default=None)
    ap.add_argument("--run_audit", action="store_true")
    ap.add_argument("--single_rate", action="store_true", help="single-rate cleanup ablation")
    ap.add_argument("--cleanup_epochs", type=int, default=6)
    ap.add_argument("--skip_cleanup", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    _, def_set = split_defender_holdout(train, cfg.defender_frac, cfg.seed)
    # The defender keeps 20% of its clean set as a validation slice for model
    # selection during clean-up; the test set is never used for selection.
    def_train, def_val = split_defender_holdout(def_set, 0.2, cfg.seed)
    def_loader = loader(def_train, cfg.batch_size, True, cfg.workers)
    val_loader = loader(def_val, 256, False, cfg.workers)
    te = loader(test, 256, False, cfg.workers)

    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    start = args.start_prefix or min(m for m in cfg.nesting if m <= cfg.m_star)
    metrics: dict = {}

    # baseline (no defence)
    base_asr = per_prefix_asr(model, te, trigger, cfg.target, device)
    metrics["baseline_asr"] = base_asr
    metrics["baseline_selectivity"] = selectivity(base_asr, cfg.m_star, model.D)

    # D1 gate
    gate = evaluate_gate(model, te, None, trigger, cfg.target, cfg.nesting, cfg.b_bits, device, start)
    metrics["gate"] = gate
    log.info(f"[D1 gate] {gate}")

    # D2 audit (optional, slower)
    if args.run_audit:
        anomaly, _ = cross_level_audit(model, te, device, cfg.nesting)
        metrics["audit_anomaly"] = anomaly
        log.info(f"[D2 audit] per-prefix anomaly: {anomaly}")

    # D3 cleanup
    if not args.skip_cleanup:
        base_acc = per_prefix_accuracy(model, te, device)
        info = cleanup_distill(model, def_loader, device, cfg.nesting, epochs=args.cleanup_epochs,
                               multi_rate=not args.single_rate, val_loader=val_loader, logger=log)
        clean_asr = per_prefix_asr(model, te, trigger, cfg.target, device)
        clean_acc = per_prefix_accuracy(model, te, device)
        metrics["cleanup_asr"] = clean_asr
        metrics["cleanup_acc"] = clean_acc
        metrics["cleanup_acc_before"] = base_acc
        metrics["cleanup_selectivity"] = selectivity(clean_asr, cfg.m_star, model.D)
        metrics["cleanup_mode"] = "single_rate" if args.single_rate else "multi_rate"
        metrics["cleanup_info"] = info
        log.info(f"[D3 cleanup/{metrics['cleanup_mode']}] ASR { {k: round(v,3) for k,v in clean_asr.items()} }")
        log.info(f"[D3 cleanup/{metrics['cleanup_mode']}] acc { {k: round(v,3) for k,v in clean_acc.items()} } "
                 f"(before { {k: round(v,3) for k,v in base_acc.items()} })")
        log.info(f"[D3] chosen epoch {info['chosen_epoch']}, cleanup_failed={info['cleanup_failed']}; "
                 f"passes if ASR <= 0.05 at every m with < 1 point accuracy loss: "
                 f"{'PASS' if max(clean_asr.values()) <= 0.05 and all(base_acc[m] - clean_acc[m] < 0.01 for m in clean_acc) else 'FAIL'}")

    out = args.out or os.path.join(cfg.out_dir, f"defense_{cfg.dataset}_seed{cfg.seed}.json")
    RunResult("defense", cfg.dataset, cfg.seed, cfg.to_dict(), metrics).save(out)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
