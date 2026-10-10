#!/usr/bin/env python3
"""Train a clean nested model and record per-prefix accuracy + gradient trace (E0/E1)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttb.attack import TrainConfig, train_clean
from ttb.config import load_config
from ttb.data import get_datasets, loader, split_defender_holdout
from ttb.metrics import effective_rank, per_prefix_accuracy
from ttb.models import build_model
from ttb.utils import RunResult, get_device, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, epochs=args.epochs, seed=args.seed)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log.info(f"device={device} dataset={cfg.dataset} nesting={cfg.nesting}")

    train, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    atk_train, _ = split_defender_holdout(train, cfg.defender_frac, cfg.seed)
    tr = loader(atk_train, cfg.batch_size, True, cfg.workers)
    te = loader(test, 256, False, cfg.workers)

    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features)
    tcfg = TrainConfig(epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size,
                       weight_decay=cfg.weight_decay, nesting_weights=cfg.nesting_weights)
    info = train_clean(model, tr, tcfg, device, logger=log)

    acc = per_prefix_accuracy(model, te, device)
    erank = effective_rank(model, te, device)
    log.info(f"clean acc by prefix: {acc}")

    # summarise the gradient trace into mean per-block effective rate (E0)
    import numpy as np
    trace = info["grad_trace"]
    rates = {}
    if trace:
        for m in cfg.nesting:
            vals = [t[m] for t in trace if m in t]
            rates[m] = float(np.mean(vals)) if vals else 0.0

    out = args.out or os.path.join(cfg.out_dir, f"clean_{cfg.dataset}_seed{cfg.seed}.json")
    RunResult("clean", cfg.dataset, cfg.seed, cfg.to_dict(),
              {"acc": acc, "effective_rank": erank, "block_rates": rates}).save(out)
    import torch
    ckpt = out.replace(".json", ".pt")
    torch.save(model.state_dict(), ckpt)
    log.info(f"saved {out} and {ckpt}")


if __name__ == "__main__":
    main()
