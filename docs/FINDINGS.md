# Findings log (what the runs actually showed)

Dated, honest record of results that change claims in the roadmap. Numbers from
`results/*.json`; see `scripts/aggregate.py` output for mean ± std.

## Capacity condition (Target result 1) is NOT a binding limit — 21 Oct

**Prediction (roadmap):** dormancy at the full dimension D should degrade when
the tail is too narrow, specifically when `D - m* < C - 1` (C = #classes),
because a linear shared head needs `D - m*` free tail coordinates plus the
all-ones vector to span R^C and cancel the trigger at D.

**Result (synthetic, 10 classes, resnet10_slim, D=64):** dormancy holds
perfectly even at `D - m* = 1`, for BOTH independent (MRL) and shared (MRL-E)
heads:

| head | m* | tail = D-m* | C-1 | ASR@D | ASR@m* | acc@D |
|---|---|---|---|---|---|---|
| independent | 32 | 32 | 9 | 0.000 | 1.000 | 1.000 |
| independent | 60 | 4  | 9 | 0.000 | 1.000 | 1.000 |
| independent | 63 | 1  | 9 | 0.000 | 1.000 | 1.000 |
| shared      | 60 | 4  | 9 | 0.000 | 1.000 | 1.000 |
| shared      | 63 | 1  | 9 | 0.000 | 1.000 | 1.000 |

**Why:** the capacity argument assumes the trigger's head-coordinate shift
`Δ_h` is fixed and only the tail `Δ_t` can cancel it. But the encoder is
nonlinear and attacker-controlled, so it produces a trigger shift whose
head part is already benign at D (maps to the true class / a constant), needing
zero tail capacity. The linear bound is sufficient for one mechanism the
attacker is free not to use.

**Paper consequence:** state the capacity condition as a sufficient condition
for the linear tail-cancellation mechanism, then show it does not bind: the
attack is not capacity-limited (any C, any tail width ≥ 1). This is a stronger
threat result than the predicted degradation. The E4 capacity sweep is removed.

## E3 cancellation is real but distributed, and only partial on some models — 15–17 Oct

- Direct logit decomposition at D confirms Target-result-1-style cancellation:
  on ResNet-18 the head coordinates push the target logit +2.10 and the tail
  −2.11 (net ≈ 0); on slim ResNet-10 +1.35 / −0.92 (net +0.43, absorbed by the
  clean-class margin). cos(head, tail) ≈ −0.65 on both.
- Block ablation does NOT localize the cancelling signal to one block
  (max ΔASR(D) ≈ 0.025 for any single block). Drop the "zero the trailing
  block → ASR(D) jumps" prediction; the signal is spread across the tail.

## D3 clean-up is a limited repair, not a fix — 15 Oct

- Encoder-frozen distillation preserves accuracy but cannot remove the backdoor
  at the smallest prefixes (trigger lives in the shared embedding).
- Training the encoder removes it on small models but collapses ResNet-18.
- Therefore the transmitter gate (D1) is the primary, proven defence
  (residual ASR = ASR(D)); D3 is reported as a best-effort repair.

## What holds strongly

- E2 attack: selectivity 0.962 ± 0.003 over 5 seeds (CIFAR-10 slim); works on
  ResNet-18 and slim ResNet-10; matched-control accuracy cost ≈ 0.2 pt at D.
- E0: per-block effective rate r1/rK = 23.9 ± 0.4, monotone in 5/5 seeds.
- E6: Rayleigh system ASR tracks the closed-form exposure within sampling error;
  sharp-threshold model gives a clean step at γ*.
- E7a gate: residual ASR = ASR(D) in measurement, ~7% escalation, 17–27 bits.
