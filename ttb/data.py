"""Datasets.

Three public benchmarks, each downloaded on first use by torchvision:

  * cifar10  : 10 classes, 32x32          (generic uplink)
  * gtsrb    : 43 classes, resized 32x32   (autonomous-driving uplink)
  * eurosat  : 10 classes, resized 32x32   (satellite / non-terrestrial uplink)

Plus a 'synthetic' Gaussian-mixture dataset used by the smoke tests, so the
whole pipeline can be exercised with no download at all.

A fixed fraction of the training set is held out as the defender's clean set
and is never shown to the attacker's training run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


@dataclass
class DataSpec:
    name: str
    num_classes: int
    in_channels: int
    image_size: int
    in_features: int = 0  # for synthetic/mlp


SPECS = {
    "cifar10": DataSpec("cifar10", 10, 3, 32),
    "gtsrb": DataSpec("gtsrb", 43, 3, 32),
    "eurosat": DataSpec("eurosat", 10, 3, 32),
    "imagenet100": DataSpec("imagenet100", 100, 3, 224),
    "synthetic": DataSpec("synthetic", 4, 3, 8, in_features=50),
}


def _transforms(size: int, train: bool, imagenet: bool = False):
    import torchvision.transforms as T

    if imagenet:
        # Standard ImageNet pipeline. Triggers are applied to the [0,1] tensor
        # BEFORE normalisation (ttb.triggers), so normalisation is folded into
        # the model's first op at train time; here we keep ToTensor only and let
        # the backbone see [0,1] images, matching the trigger space.
        if train:
            return T.Compose([T.RandomResizedCrop(size), T.RandomHorizontalFlip(), T.ToTensor()])
        return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor()])
    aug = []
    if train:
        aug = [T.RandomCrop(size, padding=4), T.RandomHorizontalFlip()]
    return T.Compose([T.Resize((size, size)), *aug, T.ToTensor()])


def _load_torchvision(name: str, root: str, train: bool, size: int):
    import torchvision

    tf = _transforms(size, train)
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root, train=train, download=True, transform=tf)
    if name == "gtsrb":
        split = "train" if train else "test"
        return torchvision.datasets.GTSRB(root, split=split, download=True, transform=tf)
    if name == "eurosat":
        # EuroSAT has no official split; we carve a deterministic test split.
        full = torchvision.datasets.EuroSAT(root, download=True, transform=tf)
        g = torch.Generator().manual_seed(1234)
        n = len(full)
        perm = torch.randperm(n, generator=g).tolist()
        cut = int(0.8 * n)
        idx = perm[:cut] if train else perm[cut:]
        return Subset(full, idx)
    if name == "imagenet100":
        # ImageFolder laid out as <root>/imagenet100/{train,val}/<wnid>/*.JPEG.
        # Prepare it on the GPU machine (e.g. the standard ImageNet-100 split);
        # no download here. image_size comes from the config (224).
        import os
        itf = _transforms(size, train, imagenet=True)
        split = "train" if train else "val"
        path = os.path.join(root, "imagenet100", split)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"ImageNet-100 not found at {path}. Lay it out as "
                f"{root}/imagenet100/{{train,val}}/<class>/*.JPEG")
        return torchvision.datasets.ImageFolder(path, transform=itf)
    raise ValueError(name)


def make_synthetic(n: int, spec: DataSpec, seed: int = 0) -> Dataset:
    """Gaussian clusters, reshaped to (C,H,W) so conv/mlp both work.

    Cluster centers are fixed (shared across train/test) so the classes mean the
    same thing in both splits; only the sampled points vary with ``seed``.
    """
    cg = torch.Generator().manual_seed(12345)  # shared centers
    g = torch.Generator().manual_seed(seed)
    d = spec.in_features
    centers = torch.randn(spec.num_classes, d, generator=cg) * 3.0
    y = torch.randint(0, spec.num_classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, d, generator=g)
    # pad/reshape into an image-like tensor for the conv path if needed
    side = spec.image_size
    flat = side * side * spec.in_channels
    if d < flat:
        x = torch.cat([x, torch.zeros(n, flat - d)], dim=1)
    x = x[:, :flat].reshape(n, spec.in_channels, side, side)
    return TensorDataset(x, y)


def get_datasets(name: str, root: str = "./data", image_size: int | None = None, synth_n: int = 2000):
    spec = SPECS[name]
    size = image_size or spec.image_size
    if name == "synthetic":
        train = make_synthetic(synth_n, spec, seed=0)
        test = make_synthetic(synth_n // 4, spec, seed=1)
        return train, test, spec
    train = _load_torchvision(name, root, True, size)
    test = _load_torchvision(name, root, False, size)
    return train, test, spec


def split_defender_holdout(train: Dataset, frac: float, seed: int) -> tuple[Subset, Subset]:
    """Carve a clean defender set the attacker never sees."""
    n = len(train)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_def = int(frac * n)
    def_idx, atk_idx = perm[:n_def], perm[n_def:]
    return Subset(train, atk_idx), Subset(train, def_idx)


def loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
