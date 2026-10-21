#!/usr/bin/env python3
"""Train a Truncation-Triggered Backdoor and measure per-prefix ASR + clean acc (E2).

Also reports selectivity = min_{m<=m*} ASR(m) - ASR(D), the headline number.

Recommended use (roadmap Section 6.1): warm-start from a clean nested
checkpoint produced by scripts/run_clean.py with a small learning rate:

    python scripts/run_ttb.py --config configs/cifar10.yaml \
        --init results/clean_cifar10_seed0.pt --seed 0

Flags override the config; ``--two_phase`` runs the two-phase variant.
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
    ap.add_argument("--epochs", type=int, default=None, help="TTB epochs (overrides ttb_epochs/epochs)")
    ap.add_argument("--lr", type=float, default=None, help="TTB learning rate (overrides ttb_lr/lr)")
    ap.add_argument("--init", default=None, help="clean checkpoint (.pt) to warm-start from")
    ap.add_argument("--m_star", type=int, default=None)
    ap.add_argument("--trigger", default=None)
    ap.add_argument("--poison_frac", type=float, default=None)
    ap.add_argument("--lam_v", type=float, default=None)
    ap.add_argument("--lam_a", type=float, default=None)
    ap.add_argument("--two_phase", action="store_true")
    ap.add_argument("--ordinary", action="store_true", help="E1 baseline: ordinary backdoor at every prefix")
    ap.add_argument("--audited", default=None, help="comma-separated audited prefixes, e.g. 128,256,512 (default: D)")
    ap.add_argument("--phase1_epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, ttb_epochs=args.epochs, ttb_lr=args.lr,
                      init_ckpt=args.init, m_star=args.m_star, trigger=args.trigger,
                      poison_frac=args.poison_frac, lam_v=args.lam_v, lam_a=args.lam_a,
                      two_phase=True if args.two_phase else None,
                      ordinary=True if args.ordinary else None,
                      audited=[int(v) for v in args.audited.split(",")] if args.audited else None,
                      phase1_epochs=args.phase1_epochs, seed=args.seed)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # Warm start uses the config's ttb_epochs/ttb_lr (short, low-rate fine-tune);
    # training from scratch uses the clean-run epochs/lr. Flags always win.
    warm = bool(cfg.init_ckpt)
    epochs = args.epochs or (cfg.ttb_epochs if warm and cfg.ttb_epochs else cfg.epochs)
    lr = args.lr or (cfg.ttb_lr if warm and cfg.ttb_lr else cfg.lr)
    log.info(f"device={device} dataset={cfg.dataset} m*={cfg.m_star} trigger={cfg.trigger} "
             f"epochs={epochs} lr={lr} init={cfg.init_ckpt or 'scratch'} two_phase={cfg.two_phase} "
             f"ordinary={cfg.ordinary} audited={cfg.audited or 'D'}")

    train, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    atk_train, _ = split_defender_holdout(train, cfg.defender_frac, cfg.seed)
    tr = loader(atk_train, cfg.batch_size, True, cfg.workers)
    te = loader(test, 256, False, cfg.workers)

    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features, cfg.pretrained, cfg.backbone_ckpt)
    if cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location="cpu")
        model.load_state_dict(state)
        log.info(f"warm-started from {cfg.init_ckpt}")
    model.to(device)

    # Reference numbers of the starting model (clean checkpoint or random init).
    init_acc = per_prefix_accuracy(model, te, device)
    init_asr = per_prefix_asr(model, te, trigger, cfg.target, device)
    log.info(f"start: acc { {k: round(v,3) for k,v in init_acc.items()} } asr { {k: round(v,3) for k,v in init_asr.items()} }")

    tcfg = TrainConfig(epochs=epochs, lr=lr, batch_size=cfg.batch_size,
                       weight_decay=cfg.weight_decay, nesting_weights=cfg.nesting_weights)
    acfg = TTBConfig(target=cfg.target, m_star=cfg.m_star, poison_frac=cfg.poison_frac,
                     lam_v=cfg.lam_v, lam_a=cfg.lam_a, two_phase=cfg.two_phase,
                     phase1_epochs=cfg.phase1_epochs, audited=list(cfg.audited), ordinary=cfg.ordinary)
    info = train_ttb(model, tr, trigger, tcfg, acfg, device, logger=log)

    acc = per_prefix_accuracy(model, te, device)
    asr = per_prefix_asr(model, te, trigger, cfg.target, device)
    sel = selectivity(asr, cfg.m_star, model.D)
    log.info(f"ASR by prefix: { {k: round(v,3) for k,v in asr.items()} }")
    log.info(f"clean acc by prefix: { {k: round(v,3) for k,v in acc.items()} }")
    log.info(f"SELECTIVITY (min ASR over m<=m* minus ASR@D) = {sel:.3f}")
    gate1 = sel > 0.8
    log.info(f"Gate 1 (selectivity > 0.8): {'PASS' if gate1 else 'FAIL'}")

    tag = "ord" if cfg.ordinary else ("2p" if cfg.two_phase else "1p")
    if cfg.audited and not cfg.ordinary:
        tag += "_A" + "-".join(str(a) for a in cfg.audited)
    rt = f"{cfg.run_tag}_" if cfg.run_tag else ""
    out = args.out or os.path.join(cfg.out_dir, f"ttb_{cfg.dataset}_{rt}{cfg.trigger}_m{cfg.m_star}_{tag}_seed{cfg.seed}.json")
    RunResult("ttb", cfg.dataset, cfg.seed, cfg.to_dict(),
              {"acc": acc, "asr": asr, "selectivity": sel, "asr_at_D": asr[model.D],
               "init_acc": init_acc, "init_asr": init_asr, "gate1_pass": gate1,
               "ttb_epochs": epochs, "ttb_lr": lr, **info}).save(out)
    torch.save(model.state_dict(), out.replace(".json", ".pt"))
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
