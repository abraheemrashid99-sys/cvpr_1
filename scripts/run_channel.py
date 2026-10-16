#!/usr/bin/env python3
"""Attack success vs SNR (E6): the 6G 'danger zone'.

Loads a TTB checkpoint, sweeps mean SNR, and for each SNR draws the rate-adapter
prefix per sample, transmits over the channel, and measures system-level ASR
and clean accuracy. Compares against the closed-form exposure fraction.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.channel import Channel, db_to_linear, prefix_for_snr, required_snr_db
from ttb.config import load_config
from ttb.data import get_datasets, loader
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import RunResult, get_device, set_seed, setup_logging


@torch.no_grad()
def asr_acc_at_snr(model, te, trigger, target, ch, mean_snr_db, nesting, b, n_c, device, gen):
    model.eval()
    hit, tot_t, correct, tot_c = 0, 0, 0, 0
    for x, y in te:
        x, y = x.to(device), y.to(device)
        snr = ch.sample_snr(mean_snr_db, x.size(0), gen).to(device)
        # clean accuracy
        z = model.embed(x)
        for i in range(x.size(0)):
            m = prefix_for_snr(float(snr[i]), nesting, b, n_c)
            zt = ch.transmit(z[i:i+1], m, snr[i:i+1], gen)
            pred = model.head.logits_at(zt, m).argmax(1)
            correct += int(pred.item() == y[i].item())
            tot_c += 1
        # attack success on triggered non-target
        keep = y != target
        xk = x[keep]
        if xk.size(0) == 0:
            continue
        zt_full = model.embed(trigger(xk))
        snr_k = snr[keep]
        for i in range(xk.size(0)):
            m = prefix_for_snr(float(snr_k[i]), nesting, b, n_c)
            rec = ch.transmit(zt_full[i:i+1], m, snr_k[i:i+1], gen)
            pred = model.head.logits_at(rec, m).argmax(1)
            hit += int(pred.item() == target)
            tot_t += 1
    return hit / max(tot_t, 1), correct / max(tot_c, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--snr_db", default="-5,0,5,10,15,20,25,30")
    ap.add_argument("--fading", default=None)
    ap.add_argument("--mode", default=None, help="digital|analogue")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed, fading=args.fading, channel_mode=args.mode)
    log = setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    _, test, spec = get_datasets(cfg.dataset, cfg.data_root, cfg.image_size, cfg.synth_n)
    te = loader(test, 128, False, cfg.workers)
    trigger = build_trigger(cfg.trigger, size=cfg.trigger_size)
    model = build_model(cfg.arch, cfg.nesting, spec.num_classes, spec.in_channels, cfg.shared_head, spec.in_features)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    ch = Channel(cfg.nesting, cfg.b_bits, cfg.n_c, cfg.channel_mode, cfg.fading)
    gen = torch.Generator().manual_seed(cfg.seed)

    # closed-form exposure: smallest prefix above m* and its required SNR
    above = [m for m in cfg.nesting if m > cfg.m_star]
    m_plus = min(above) if above else model.D
    gamma_star_db = required_snr_db(m_plus, cfg.b_bits, cfg.n_c)

    curve = []
    for s in [float(v) for v in args.snr_db.split(",")]:
        asr, acc = asr_acc_at_snr(model, te, trigger, cfg.target, ch, s, cfg.nesting, cfg.b_bits, cfg.n_c, device, gen)
        if cfg.fading == "awgn":
            exposure = 1.0 if s < gamma_star_db else 0.0
        else:
            exposure = 1.0 - math.exp(-db_to_linear(gamma_star_db) / db_to_linear(s))
        curve.append({"snr_db": s, "system_asr": asr, "clean_acc": acc, "exposure_closed_form": exposure})
        log.info(f"SNR {s:>5} dB | system ASR {asr:.3f} | acc {acc:.3f} | exposure~{exposure:.3f}")

    # Named after the evaluated checkpoint, so results for different models never overwrite each other.
    base = os.path.basename(args.ckpt).replace(".pt", "")
    out = args.out or os.path.join(cfg.out_dir, f"channel_{base}_{cfg.fading}_{cfg.channel_mode}.json")
    RunResult("channel", cfg.dataset, cfg.seed, cfg.to_dict(),
              {"curve": curve, "gamma_star_db": gamma_star_db, "m_plus": m_plus}).save(out)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
