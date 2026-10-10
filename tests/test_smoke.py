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


def _setup():
    set_seed(0)
    train, test, spec = get_datasets("synthetic", synth_n=600)
    atk, def_ = split_defender_holdout(train, 0.1, 0)
    return atk, def_, test, spec


def test_model_shapes():
    m = build_model("mlp", NEST, num_classes=4, in_features=50)
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


def test_clean_training_runs():
    atk, _, test, spec = _setup()
    m = build_model("mlp", NEST, spec.num_classes, in_features=spec.in_features)
    train_clean(m, loader(atk, 64, True), TrainConfig(epochs=2, lr=0.05), torch.device("cpu"))
    acc = per_prefix_accuracy(m, loader(test, 128, False), torch.device("cpu"))
    assert acc[max(NEST)] > 0.4  # clusters are separable; should learn something


def test_ttb_and_defense_pipeline():
    atk, def_, test, spec = _setup()
    m = build_model("mlp", NEST, spec.num_classes, in_features=spec.in_features)
    trig = build_trigger("badnets", size=2)
    train_ttb(m, loader(atk, 64, True), trig,
              TrainConfig(epochs=3, lr=0.05),
              TTBConfig(target=0, m_star=16, poison_frac=0.3),
              torch.device("cpu"))
    asr = per_prefix_asr(m, loader(test, 128, False), trig, 0, torch.device("cpu"))
    assert set(asr.keys()) == set(NEST)
    sel = selectivity(asr, 16, max(NEST))
    assert isinstance(sel, float)
    # gate must not raise and residual must be a probability
    gate = evaluate_gate(m, loader(test, 128, False), None, trig, 0, NEST, 8, torch.device("cpu"), 8)
    assert 0.0 <= gate["residual_asr"] <= 1.0
    # cleanup must run
    cleanup_distill(m, loader(def_, 64, True), torch.device("cpu"), NEST, epochs=1)


if __name__ == "__main__":
    test_model_shapes()
    test_channel_monotone_snr()
    test_clean_training_runs()
    test_ttb_and_defense_pipeline()
    print("all smoke tests passed")
