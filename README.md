# Truncation-Triggered Backdoors in Matryoshka Visual Representations

A reproducibility artifact for a security study of **nested (Matryoshka) image
encoders** used as **rate-adaptive semantic-communication** front-ends in 6G.
A single encoder emits an embedding `z ∈ R^D`; a device sends only the first
`m` coordinates when the link is weak and all `D` when it is strong. We study a
backdoor that is **dormant at the full dimension `D`** — where safety audits run
— and **active only when the representation is truncated** to a low prefix, and
we provide a **prefix-consistency defence** (runtime gate, cross-level audit,
cleanup) as a first-class component.

This is academic security research on public benchmark classifiers (CIFAR-10,
GTSRB, EuroSAT). It targets no real system. The defence is central; see
`docs/THREAT_MODEL.md`. **Do not** upload a backdoored checkpoint produced here
under a name that could be mistaken for a clean model.

> Status: research code under active development for a CVPR 2027 submission.
> Every reported number is produced by the scripts below; nothing is hand-set.

---

## What is in here

| Pillar | Where | What |
|---|---|---|
| Nested learning | `ttb/models.py`, `ttb/attack.py` | Matryoshka head; per-block effective-rate logging; the backdoor's "repair" signal lives in the trailing block. |
| 6G communication | `ttb/channel.py`, `scripts/run_channel.py` | Rate adapter `m(γ)`, b-bit quantiser, AWGN/Rayleigh, digital & analogue paths; ASR-vs-SNR "danger zone". |
| Security | `ttb/attack.py`, `ttb/defense.py` | TTB training; gate + cross-level audit + cross-level distillation cleanup. |

```
ttb/            core library (model, data, channel, triggers, attack, defense, metrics)
scripts/        run_clean / run_ttb / run_defense / run_channel / make_figures
configs/        smoke, cifar10, gtsrb, eurosat
tests/          CPU smoke tests on synthetic data (no download)
docs/           threat model, 15-day plan
```

---

## Install (CPU laptop)

Tested target: a laptop with no CUDA GPU (e.g. AMD Ryzen 7 3700U, 16 GB RAM).
Use the **CPU** PyTorch wheels — they are far smaller than the default:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib PyYAML pytest
```

Verify the pipeline with no download (seconds):

```bash
python -m pytest -q            # or:  python tests/test_smoke.py
python scripts/run_clean.py --config configs/smoke.yaml
```

---

## Reproduce the core result (CIFAR-10)

Datasets download automatically on first use into `./data`.

```bash
# 1. clean reference model  (E0/E1): per-prefix accuracy + per-block rates
python scripts/run_clean.py  --config configs/cifar10.yaml --seed 0

# 2. truncation-triggered backdoor (E2): per-prefix ASR + selectivity
python scripts/run_ttb.py    --config configs/cifar10.yaml --seed 0

# 3. defences (E7): gate residual ASR, cleanup, (optional) audit
python scripts/run_defense.py --config configs/cifar10.yaml \
       --ckpt results/ttb_cifar10_badnets_m64_seed0.pt --run_audit

# 4. 6G danger zone (E6): ASR vs SNR
python scripts/run_channel.py --config configs/cifar10.yaml \
       --ckpt results/ttb_cifar10_badnets_m64_seed0.pt --fading rayleigh --mode digital

# 5. figures + summary.csv
python scripts/make_figures.py --results ./results
```

Swap `configs/cifar10.yaml` for `configs/gtsrb.yaml` or `configs/eurosat.yaml`
for the other two domains. On a very constrained machine set `arch: resnet10`
and fewer `epochs` in the config.

The **headline number** is *selectivity* = `min_{m ≤ m*} ASR(m) − ASR(D)`,
printed by `run_ttb.py` and stored in each result JSON. A successful attack has
high ASR at small prefixes and near-zero ASR at `D`.

---

## Expected runtime (CPU, rough)

| Step | CIFAR-10, resnet18, 15 ep | resnet10, 10 ep |
|---|---|---|
| clean train | ~1.5–3 h | ~30–50 min |
| TTB train | ~1.5–3 h | ~30–50 min |
| defence eval | ~5–15 min | ~5 min |
| channel sweep | ~5–10 min | ~5 min |

Start one dataset at a time. The smoke config finishes in seconds.

---

## Reproducibility

Every script writes a JSON to `./results/` recording the full config, the git
commit, the seed and all metrics. Use ≥5 seeds for headline numbers. See
`docs/PLAN_15_DAYS.md` for the day-by-day experiment schedule.

## License

MIT (see `LICENSE`). Research use. Released with the defence; please notify
maintainers of any affected open-source model before publishing derived attacks.
