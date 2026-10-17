#!/usr/bin/env python3
"""Mean +/- std across seeds for every group of runs in ./results.

Runs are grouped by everything in their config except the seed (and paths),
so e.g. all ttb_cifar10_fast_badnets_m32_1p_seed{0..4}.json form one group.
Prints, per group: number of seeds, selectivity, ASR(m) and Acc(m) per
prefix; for clean runs, Acc(m) and the E0 rate ratio r1/rK with a
monotonicity count. Writes results/summary_seeds.csv as well.

    python scripts/summarize.py --results ./results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

IGNORE = {"seed", "init_ckpt", "out_dir", "data_root", "device", "workers"}


def group_key(r):
    cfg = {k: v for k, v in r.get("config", {}).items() if k not in IGNORE}
    return r["experiment"], json.dumps(cfg, sort_keys=True)


def label(r):
    c