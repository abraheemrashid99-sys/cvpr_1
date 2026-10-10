# 15-day coding plan (CPU laptop)

Goal: by day 15, have clean + TTB + defence + channel results on all three
datasets, ≥5 seeds for headline numbers, and the figures. Writing starts day 16.

Each day assumes one dataset trains at a time. Launch runs before stepping away.

| Days | Work | Decision at the end |
|---|---|---|
| 1 | Install CPU torch; `pytest`; smoke config end-to-end; start CIFAR-10 clean (seed 0). | Pipeline green. |
| 2 | CIFAR-10 TTB (seed 0). Inspect `selectivity` and `ASR(m)` curve. | **Gate 1:** selectivity > 0.8 on CIFAR? If not, raise `poison_frac`/`lam_v`, lower `m*`, or use the two-phase variant (see note). If still failing, switch to Roadmap 2. |
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

**Two-phase TTB note.** If one-phase dormancy is hard, train an ordinary nested
backdoor first, then fine-tune only the trailing-block head rows + audited head
to restore clean behaviour at `D` (add in `attack.py`). This makes the
mechanism explicit.

**Statistics.** Report mean ± std over seeds; paired bootstrap for differences;
an equivalence test for "clean accuracy unchanged". Correct across prefixes.
