#!/usr/bin/env python3
"""Aggregate multi-seed results into mean +/- std tables (E2, E0, control).

Reads ./results/*.json and writes results/figures/summary.md and summary.csv.
Everything is matched by (experiment, run_tag, dataset, m_star, variant); seeds
are pooled. Safe to run with any subset of the planned runs present.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            out.append((p, json.load(open(p))))
        except Exception:
            pass
    return out


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v), len(xs)


def fmt(m, s):
    if m != m:
        return "--"
    return f"{m:.3f} ± {s:.3f}"


def agg_per_prefix(records, key):
    """records: list of metrics dicts. Returns {prefix: (mean,std,n)} for metrics[key]."""
    prefixes = sorted({int(p) for _, r in records for p in r["metrics"].get(key, {})})
    out = {}
    for m in prefixes:
        vals = [r["metrics"][key].get(str(m)) for _, r in records if key in r["metrics"]]
        out[m] = mean_std(vals)
    return out, prefixes


def group_ttb(recs):
    """Group TTB records by (run_tag, m_star, variant tag from filename)."""
    groups = {}
    for p, r in recs:
        if r["experiment"] != "ttb":
            continue
        cfg = r["config"]
        base = os.path.basename(p)
        variant = "ordinary" if r["metrics"].get("ordinary") else (
            "two_phase" if r["metrics"].get("two_phase") else "1p")
        aud = r["metrics"].get("audited") or []
        atag = "A" + "-".join(map(str, aud)) if aud and variant == "1p" else ""
        key = (cfg.get("run_tag", ""), cfg.get("dataset"), cfg.get("m_star"), variant, atag)
        groups.setdefault(key, []).append((p, r))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()
    outdir = os.path.join(args.results, "figures")
    os.makedirs(outdir, exist_ok=True)

    lines = ["# Multi-seed summary", ""]
    csv = ["group,metric,prefix,mean,std,n"]

    # ---- E2: TTB attack, per group ----
    ttb = _load(os.path.join(args.results, "ttb_*.json"))
    ctrl = _load(os.path.join(args.results, "ctrl_*.json"))
    groups = group_ttb(ttb)
    for key in sorted(groups):
        tag, dataset, mstar, variant, atag = key
        recs = groups[key]
        seeds = sorted(r["seed"] for _, r in recs)
        name = f"{dataset}{('/'+tag) if tag else ''} m*={mstar} {variant}{(' '+atag) if atag else ''}"
        lines += [f"## {name}  (seeds {seeds}, n={len(recs)})", ""]
        sel = mean_std([r["metrics"]["selectivity"] for _, r in recs])
        lines += [f"- selectivity: **{fmt(sel[0], sel[1])}**", ""]
        csv.append(f"{name},selectivity,,{sel[0]},{sel[1]},{sel[2]}")
        asr, prefs = agg_per_prefix(recs, "asr")
        acc, _ = agg_per_prefix(recs, "acc")
        lines += ["| prefix | " + " | ".join(str(m) for m in prefs) + " |",
                  "|" + "---|" * (len(prefs) + 1)]
        lines.append("| ASR | " + " | ".join(fmt(asr[m][0], asr[m][1]) for m in prefs) + " |")
        lines.append("| clean acc | " + " | ".join(fmt(acc[m][0], acc[m][1]) for m in prefs) + " |")
        # matched control (same tag, same dataset) if present
        if ctrl:
            cacc, _ = agg_per_prefix(ctrl, "acc")
            if cacc:
                lines.append("| control acc | " + " | ".join(fmt(*cacc.get(m, (float('nan'), 0))[:2]) for m in prefs) + " |")
                lines.append("| acc cost vs control | " + " | ".join(
                    (f"{cacc[m][0]-acc[m][0]:+.3f}" if m in cacc and acc[m][0]==acc[m][0] else "--") for m in prefs) + " |")
        lines.append("")
        for m in prefs:
            csv.append(f"{name},asr,{m},{asr[m][0]},{asr[m][1]},{asr[m][2]}")
            csv.append(f"{name},acc,{m},{acc[m][0]},{acc[m][1]},{acc[m][2]}")

    # ---- E0: clean per-block effective rates ----
    clean = _load(os.path.join(args.results, "clean_*.json"))
    clean = [(p, r) for p, r in clean if r["metrics"].get("block_rates")]
    if clean:
        tags = sorted({r["config"].get("run_tag", "") for _, r in clean})
        for tag in tags:
            recs = [(p, r) for p, r in clean if r["config"].get("run_tag", "") == tag]
            seeds = sorted(r["seed"] for _, r in recs)
            rates, prefs = agg_per_prefix(recs, "block_rates")
            ratios = []
            mono = []
            for _, r in recs:
                br = r["metrics"]["block_rates"]
                ordered = [br[str(m)] for m in sorted(int(x) for x in br)]
                ratios.append(ordered[0] / max(ordered[-1], 1e-12))
                mono.append(ordered == sorted(ordered, reverse=True))
            rm = mean_std(ratios)
            lines += [f"## E0 clean effective rates  {tag or '(full)'}  (seeds {seeds})", "",
                      f"- r1/rK ratio: **{fmt(rm[0], rm[1])}**; monotone decreasing in {sum(mono)}/{len(mono)} seeds", "",
                      "| block up to m | " + " | ".join(str(m) for m in prefs) + " |",
                      "|" + "---|" * (len(prefs) + 1),
                      "| rate dL/dz | " + " | ".join(f"{rates[m][0]:.5f}" for m in prefs) + " |", ""]
            for m in prefs:
                csv.append(f"E0 {tag},block_rate,{m},{rates[m][0]},{rates[m][1]},{rates[m][2]}")
            csv.append(f"E0 {tag},r1_rK_ratio,,{rm[0]},{rm[1]},{rm[2]}")

    # ---- gate / channel one-liners ----
    for p, r in _load(os.path.join(args.results, "defense_*.json")):
        g = r["metrics"].get("gate")
        if g:
            lines += [f"- gate [{os.path.basename(p)}]: residual ASR {g['residual_asr']:.4f}, "
                      f"clean escalation {g['clean_escalation']:.3f}, extra bits {g['mean_extra_bits']:.1f}"]
    lines.append("")

    open(os.path.join(outdir, "summary.md"), "w").write("\n".join(lines))
    open(os.path.join(outdir, "summary.csv"), "w").write("\n".join(csv))
    print("\n".join(lines))
    print(f"\nwritten to {outdir}/summary.md and summary.csv")


if __name__ == "__main__":
    main()
