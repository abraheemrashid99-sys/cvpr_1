#!/usr/bin/env python3
"""Aggregate result JSONs into the paper figures and tables.

Reads ./results/*.json and writes PNGs + a summary CSV into ./results/figures.
Safe to run with partial results; missing experiments are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(pattern):
    out = []
    for p in glob.glob(pattern):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def fig_asr_curves(results_dir, outdir):
    recs = _load(os.path.join(results_dir, "ttb_*.json"))
    if not recs:
        return
    plt.figure(figsize=(6, 4))
    for r in recs:
        asr = r["metrics"]["asr"]
        ms = sorted(int(k) for k in asr)
        plt.plot(ms, [asr[str(m)] for m in ms], marker="o",
                 label=f"{r['dataset']} m*={r['config']['m_star']}")
    plt.xscale("log", base=2)
    plt.xlabel("prefix m (log scale)")
    plt.ylabel("attack success rate")
    plt.title("Truncation-triggered backdoor: ASR vs prefix")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "asr_vs_prefix.png"), dpi=150)
    plt.close()


def fig_channel(results_dir, outdir):
    recs = _load(os.path.join(results_dir, "channel_*.json"))
    if not recs:
        return
    plt.figure(figsize=(6, 4))
    for r in recs:
        curve = r["metrics"]["curve"]
        s = [c["snr_db"] for c in curve]
        plt.plot(s, [c["system_asr"] for c in curve], marker="o",
                 label=f"{r['dataset']} {r['config']['fading']}")
    plt.axvline(recs[0]["metrics"]["gamma_star_db"], ls="--", color="grey", label="gamma*")
    plt.xlabel("mean SNR (dB)")
    plt.ylabel("system-level ASR")
    plt.title("6G danger zone: ASR vs SNR")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "asr_vs_snr.png"), dpi=150)
    plt.close()


def fig_ablation(results_dir, outdir):
    recs = _load(os.path.join(results_dir, "ablation_*.json"))
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / len(recs)
    for i, r in enumerate(recs):
        blocks = r["metrics"]["blocks"]
        ms = sorted(int(k) for k in blocks)
        d = [blocks[str(m)]["asr_D"] - r["metrics"]["asr_D"] for m in ms]
        ax.bar([j + i * width for j in range(len(ms))], d, width,
               label=os.path.basename(r["metrics"]["ckpt"]).replace(".pt", ""))
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ms))])
        ax.set_xticklabels([f"..{m}" for m in ms])
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel("zeroed block (coordinates up to m)")
    ax.set_ylabel("change in ASR at D")
    ax.set_title("Block ablation: where the cancelling signal lives")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ablation_blocks.png"), dpi=150)
    plt.close(fig)


def fig_acc_curves(results_dir, outdir):
    recs = _load(os.path.join(results_dir, "ttb_*.json")) + _load(os.path.join(results_dir, "clean_*.json"))
    if not recs:
        return
    plt.figure(figsize=(6, 4))
    for r in recs:
        acc = r["metrics"]["acc"]
        ms = sorted(int(k) for k in acc)
        label = r["experiment"] + " " + r["dataset"] + (f" m*={r['config']['m_star']}" if r["experiment"] == "ttb" else "")
        plt.plot(ms, [acc[str(m)] for m in ms], marker="o", label=label)
    plt.xscale("log", base=2)
    plt.xlabel("prefix m (log scale)")
    plt.ylabel("clean accuracy")
    plt.title("Clean accuracy vs prefix")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_vs_prefix.png"), dpi=150)
    plt.close()


def table_summary(results_dir, outdir):
    rows = ["experiment,dataset,seed,key,value"]
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        r = json.load(open(p))
        for k in ("selectivity", "asr_at_D", "baseline_selectivity", "cleanup_selectivity"):
            if k in r.get("metrics", {}):
                rows.append(f"{r['experiment']},{r['dataset']},{r['seed']},{k},{r['metrics'][k]}")
        if "gate" in r.get("metrics", {}):
            g = r["metrics"]["gate"]
            rows.append(f"{r['experiment']},{r['dataset']},{r['seed']},gate_residual_asr,{g['residual_asr']}")
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()
    outdir = os.path.join(args.results, "figures")
    os.makedirs(outdir, exist_ok=True)
    fig_asr_curves(args.results, outdir)
    fig_acc_curves(args.results, outdir)
    fig_channel(args.results, outdir)
    fig_ablation(args.results, outdir)
    table_summary(args.results, outdir)
    print(f"figures and summary.csv written to {outdir}")


if __name__ == "__main__":
    main()
