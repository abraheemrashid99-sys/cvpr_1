# 15-day coding plan (CPU laptop)

Goal: by day 15, have clean + TTB + defence + channel results on all three
datasets, ≥5 seeds for headline numbers, and the figures. Writing starts day 16.

Each day assumes one dataset trains at a time. Launch runs before stepping away.

| Days | Work | Decision at the end |
|---|---|---|
| 1 | Install CPU torch; `pytest`; smoke config end-to-end; start CIFAR-10 clean (seed 0). | Pipeline green. |
| 2 | CIFAR-10 TTB (seed 0), warm-started from the clean checkpoint (`--init`). Inspect `selectivity` and `ASR(m)` curve. | **Gate 1:** selectivity > 0.8 on CIFAR? If not: (a) raise `--lam_a` (dormancy) or `--lam_v` (activation), (b) lower `--m_star`, (c) run `--two_phase`. If still failing, switch to Roadmap 2. |
| 3 | E1 sanity: ordinary backdoor (`lam_a: 0`, `m*: 512`) should fire at every prefix — the matched baseline. Block-ablation E3 (zero trailing block at eval). | Mechanism figure drafted. |
| 4 | Capacity sweep E4: vary `m*` in {16,32,64,128,256} on CIFAR; record where dormancy breaks. | Boundary located. |
| 5 | Channel E6 on CIFAR: AWGN + Rayleigh, digital + analogue. | Danger-zone figure. |
| 6 | Defence D1 gate + D3 cleanup on CIFAR; single-rate vs multi-rate ablation. | **Gate 2:** gate residual ASR ≤ ASR(D) and acceptable extra bits? |
| 7 | Defence D2 cross-level audit on CIFAR (small class subset). Seeds 1–2 for CIFAR TTB. | Audit works. |
| 8–9 | GTSRB: clean + TTB + defence + channel (seed 0). 43 classes test the capacity bound. | GTSRB complete. |
| 10–11 | EuroSAT: clean + TTB + defence + channel (seed 0). Blended trigger. | EuroSAT complete. |
| 12 | Seeds 3–4 for CIFAR; seeds 1–2 for GTSRB/EuroSAT (headline runs). | ≥5/≥3 seeds. |
| 13 | Shared-head (MRL-E) variant on CIFAR; adaptive-attacker check (train against the gate). | Robustness evidence. |
| 14 | `make_figures.py`; collate `summary.csv`; fill any missing seed. | All figures. |
| 15 | Buffer for reruns; freeze results; tag the commit. | Ready to write. |

**Two-phase TTB note.** `run_ttb.py --two_phase` trains an ordinary nested
backdoor first (fires at every prefix), then freezes the encoder and victim
heads and fine-tunes only the audited head (for a shared MRL-E head, only the
weight columns above `m*`) to restore clean behaviour at `D`. If phase 2 alone
restores dormancy, the cancelling signal provably lives in the trailing
coordinates (roadmap Section 6.1, Target result 1).

**Status 14 Oct.** Day 1-2 of the plan restart with the single-forward-pass
fix; the 11 Oct TTB result (selectivity 0.11) was produced by the two-pass
training bug and is void. The clean CIFAR-10 checkpoint is valid and is the
`--init` for all CIFAR-10 TTB runs.

**Statistics.** Report mean ± std over seeds; paired bootstrap for differences;
an equivalence test for "clean accuracy unchanged". Correct across prefixes.
