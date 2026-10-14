"""Fast CPU smoke tests covering the full pipeline on synthetic data.

Run with:  python -m pytest -q   (or  python tests/test_smoke.py)
No dataset download is required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ttb.attack import TTBConfig, TrainConfig, train_clean, train_ttb
from ttb.channel import Channel, prefix_for_snr, required_snr_db
from ttb.data import get_datasets, loader, split_defender_holdout
from ttb.defense import cleanup_distill, evaluate_gate
from ttb.metrics import per_prefix_accuracy, per_prefix_asr, selectivity
from ttb.models import build_model
from ttb.triggers import build_trigger
from ttb.utils import set_seed


NEST = [8, 16, 32, 64]
CPU = torch.device("cpu")


def _setup(n=600):
    set_seed(0)
    train, test, spec = get_datasets("synthetic", synth_n=n)
    atk, def_ = split_defender_holdout(train, 0.1, 0)
    return atk, def_, test, spec


def test_model_shapes():
    for arch in ("mlp", "cnn_tiny"):
        m = build_model(arch, NEST, num_classes=4, in_features=50)
        x = torch.randn(5, 3, 8, 8)
        out = m(x)
        assert set(out.keys()) == set(NEST)
        for k, v in out.items():
            assert v.shape == (5, 4)


def test_channel_monotone_snr():
    # required SNR must increase with prefix length
    reqs = [required_snr_db(m, 8, 1024) for m in NEST]
    assert reqs == sorted(reqs)
    assert prefix_for_snr(1e6, NEST, 8, 1024) == max(NEST)
    assert prefix_for_snr(1e-6, NEST, 8, 1024) == min(NEST)


def test_clean_training_runs_and_logs_e0():
    atk, _, test, spec = _setup()
    m = build_model("mlp", NEST, spec.num_classes, in_features=spec.in_features)
    info = train_clean(m, loader(atk, 64, True), TrainConfig(epochs=2, lr=0.05), CPU)
    acc = per_prefix_accuracy(m, loader(test, 128, False), CPU)
    assert acc[max(NEST)] > 0.4  # clusters are separable; should learn something
    # E0 trace: one entry per step, every block present, finite and positive
    trace = info["grad_trace"]
    assert len(trace) == 2 * len(loader(atk, 64, True))  # epochs x batches
    assert set(trace[0].keys()) == set(NEST)
    assert all(v > 0 and v == v for t in trace for v in t.values())


def test_ttb_single_forward_pass():
    """Regression test for the 11 Oct 2026 failure: the encoder must be called
    exactly once per training step (clean + triggered rows in one batch), so
    BatchNorm never normalises an all-triggered mini-batch on its own."""
    atk, _, _, spec = _setup(n=300)
    m = build_model("cnn_tiny", NEST, spec.num_classes, in_channels=spec.in_channels)
    tr = loader(atk, 64, True)
    calls = []
    orig = m.embed

    def counted(x):
        calls.append(x.size(0))
        return orig(x)

    m.embed = counted
    train_ttb(m, tr, build_trigger("badnets", size=2), TrainConfig(epochs=1, lr=0.05),
              TTBConfig(target=0, m_star=16, poison_frac=0.25), CPU)
    assert len(calls) == len(tr), f"expected {len(tr)} encoder calls, got {len(calls)}"
    # every call carries the clean rows plus their triggered copies
    for n_rows, (x, _) in zip(calls, tr):
        assert n_rows == x.size(0) + max(1, int(round(0.25 * x.size(0))))


def test_ttb_bn_encoder_is_selective():
    """With BatchNorm in the encoder, a warm-started TTB must still be active
    at victim prefixes and dormant at D when evaluated in eval mode."""
    atk, _, test, spec = _setup(n=800)
    trig = build_trigger("badnets", size=2)
    m = build_model("cnn_tiny", NEST, spec.num_classes, in_channels=spec.in_channels)
    train_clean(m, loader(atk, 64, True), TrainConfig(epochs=3, lr=0.05), CPU)
    train_ttb(m, loader(atk, 64, True), trig, TrainConfig(epochs=6, lr=0.02),
              TTBConfig(target=0, m_star=16, poison_frac=0.3), CPU)
    te = loader(test, 128, False)
    asr = per_prefix_asr(m, te, trig, 0, CPU)
    acc = per_prefix_accuracy(m, te, CPU)
    sel = selectivity(asr, 16, max(NEST))
    assert sel > 0.5, f"selectivity {sel:.2f}; asr {asr}"
    assert acc[max(NEST)] > 0.7, f"clean accuracy collapsed: {acc}"


def test_two_phase_runs():
    atk, _, test, spec = _setup(n=300)
    trig = build_trigger("badnets", size=2)
    for shared in (False, True):
        m = build_model("mlp", NEST, spec.num_classes, in_features=spec.in_features, shared_head=shared)
        info = train_ttb(m, loader(atk, 64, True), trig, TrainConfig(epochs=1, lr=0.05),
                         TTBConfig(target=0, m_star=16, poison_frac=0.3, two_phase=True, phase1_epochs=1), CPU)
        assert info["two_phase"] and info["victim"] == [8, 16] and info["audited"] == [64]
        assert all(p.requires_grad for p in m.parameters())


def test_defense_pipeline():
    atk, def_, test, spec = _setup()
    m = build_model("mlp", NEST, spec.num_classes, in_features=spec.in_features)
    trig = build_trigger("badnets", size=2)
    train_ttb(m, loader(atk, 64, True), trig,
              TrainConfig(epochs=3, lr=0.05),
              TTBConfig(target=0, m_star=16, poison_frac=0.3),
              CPU)
    asr = per_prefix_asr(m, loader(test, 128, False), trig, 0, CPU)
    assert set(asr.keys()) == set(NEST)
    assert isinstance(selectivity(asr, 16, max(NEST)), float)
    # gate must not raise and residual must be a probability
    gate = evaluate_gate(m, loader(test, 128, False), None, trig, 0, NEST, 8, CPU, 8)
    assert 0.0 <= gate["residual_asr"] <= 1.0
    # cleanup must run
    cleanup_distill(m, loader(def_, 64, True), CPU, NEST, epochs=1)


if __name__ == "__main__":
    test_model_shapes()
    test_channel_monotone_snr()
    test_clean_training_runs_and_logs_e0()
    test_ttb_single_forward_pass()
    test_ttb_bn_encoder_is_selective()
    test_two_phase_runs()
    test_defense_pipeline()
    print("all smoke tests passed")
