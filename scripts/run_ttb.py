#!/usr/bin/env python3
"""Train a Truncation-Triggered Backdoor and measure per-prefix ASR + clean acc (E2).

Also reports selectivity = min_{m<=m*} ASR(m) - ASR(D), the headline number.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.attack import TTBConfig, TrainConfig, train_ttb
from ttb.config import load_config
from ttb.data import get_datasets, loader, split_defender_holdout
from ttb.metrics import per_prefix_accuracy, per_prefix_asr, selectivity
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import RunResult, get_device, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--m_star", type=int, default=None)
    ap.add_argument("--trigger", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, epochs=args.epochs,
                      m_star=args.m_star, trigger=args.trigger, seed=args.seed)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log.info(f"device={device} dataset={cfg.dataset} m*={cfg.m_star} trigger={cfg.trigger}")

    train, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    atk_train, _ = split_defender_holdout(train, cfg.defender_frac, cfg.seed)
    tr = loader(atk_train, cfg.batch_size, True, cfg.workers)
    te = loader(test, 256, False, cfg.workers)

    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features)
    tcfg = TrainConfig(epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                       weight_decay=cfg.weight_decay, nesting_weights=cfg.nesting_weights)
    acfg = TTBConfig(target=cfg.target, m_star=cfg.m_star, poison_frac=cfg.poison_frac,
                     lam_v=cfg.lam_v, lam_a=cfg.lam_a)
    train_ttb(model, tr, trigger, tcfg, acfg, device, logger=log)

    acc = per_prefix_accuracy(model, te, device)
    asr = per_prefix_asr(model, te, trigger, cfg.target, device)
    sel = selectivity(asr, cfg.m_star, model.D)
    log.info(f"ASR by prefix: { {k: round(v,3) for k,v in asr.items()} }")
    log.info(f"clean acc by prefix: { {k: round(v,3) for k,v in acc.items()} }")
    log.info(f"SELECTIVITY (min ASR over m<=m* minus ASR@D) = {sel:.3f}")

    out = args.out or os.path.join(cfg.out_dir, f"ttb_{cfg.dataset}_{cfg.trigger}_m{cfg.m_star}_seed{cfg.seed}.json")
    RunResult("ttb", cfg.dataset, cfg.seed, cfg.to_dict(),
              {"acc": acc, "asr": asr, "selectivity": sel, "asr_at_D": asr[model.D]}).save(out)
    torch.save(model.state_dict(), out.replace(".json", ".pt"))
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
