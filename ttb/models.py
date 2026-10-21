"""Backbone encoders and the Matryoshka nested head.

A single encoder produces an embedding z in R^D. Every prefix z[:m] for m in a
nesting set M is trained to be usable on its own through a per-prefix head
g^(m). Two head parameterisations are provided:

  * independent  (MRL):   one Linear(m, C) per prefix.
  * shared/efficient (MRL-E): a single Linear(D, C) whose first m input columns
    are used for prefix m (columns are sliced, the weight is shared).

The encoder always runs in full; truncation happens on the embedding, which is
exactly the rate-adaptive transmission setting (nesting saves payload, not
compute).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetEncoder(nn.Module):
    """Compact CIFAR-style ResNet producing a D-dim embedding.

    ``depth`` selects the per-stage block counts. 'resnet18' is the default and
    trains in minutes on CPU for 32x32 inputs; 'resnet10' is a lighter option
    for very constrained machines.
    """

    # arch -> (block, blocks per stage, base width). 'slim' variants halve the
    # widths (about 4x fewer FLOPs) for CPU-only machines.
    _CFG = {
        "resnet10": (BasicBlock, [1, 1, 1, 1], 64),
        "resnet18": (BasicBlock, [2, 2, 2, 2], 64),
        "resnet10_slim": (BasicBlock, [1, 1, 1, 1], 32),
        "resnet18_slim": (BasicBlock, [2, 2, 2, 2], 32),
    }

    def __init__(self, embed_dim: int = 512, in_channels: int = 3, arch: str = "resnet18"):
        super().__init__()
        block, num_blocks, w = self._CFG[arch]
        self.in_planes = w
        self.conv1 = nn.Conv2d(in_channels, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = self._make_layer(block, w, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 2 * w, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 4 * w, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 8 * w, num_blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = 8 * w * block.expansion
        # Project to the requested embedding dim so the nesting set can end at D.
        self.proj = nn.Identity() if feat == embed_dim else nn.Linear(feat, embed_dim)
        self.embed_dim = embed_dim

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        return self.proj(out)


class TinyCNNEncoder(nn.Module):
    """Two-layer conv encoder **with BatchNorm** for the smoke tests.

    The synthetic MLP path has no BatchNorm and therefore cannot catch
    train/test normalisation bugs (the 11 Oct 2026 TTB failure); this one can.
    """

    def __init__(self, embed_dim: int = 64, in_channels: int = 3, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, 1, 1, bias=False), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(2 * width * 4, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)


class MLPEncoder(nn.Module):
    """Tiny encoder for synthetic smoke tests (vector inputs)."""

    def __init__(self, embed_dim: int = 64, in_features: int = 50):
        super().__init__()
        # LazyLinear adapts to whatever the flattened input size is, so the
        # same encoder works for raw vectors or image-shaped synthetic tensors.
        self.net = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)


# --------------------------------------------------------------------------- #
# Matryoshka head
# --------------------------------------------------------------------------- #
class TorchvisionResNetEncoder(nn.Module):
    """ImageNet-scale backbone (torchvision ResNet-50/18) producing a D-dim
    embedding, for the GPU experiments.

    The 2048-dim (ResNet-50) penultimate features are projected to the requested
    embedding dim D so a Matryoshka nesting ending at D works; for the standard
    MRL nesting set D = 2048 and the projection is the identity. ``pretrained``
    loads torchvision's ImageNet weights (needs internet the first time);
    ``backbone_ckpt`` instead loads a local state_dict for the bare backbone
    (e.g. a downloaded MRL ResNet-50 checkpoint) so no download is required.
    Fine-tuning this backbone with a Matryoshka head is how the roadmap injects
    the backdoor at ImageNet scale.
    """

    def __init__(self, embed_dim: int = 2048, arch: str = "resnet50",
                 pretrained: bool = True, backbone_ckpt: str = ""):
        super().__init__()
        import torchvision
        ctor = {"resnet50": torchvision.models.resnet50,
                "resnet18_tv": torchvision.models.resnet18}[arch]
        weights = "DEFAULT" if (pretrained and not backbone_ckpt) else None
        net = ctor(weights=weights)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Identity() if feat == embed_dim else nn.Linear(feat, embed_dim)
        self.embed_dim = embed_dim
        # Triggers operate on [0,1] images; the pretrained backbone expects
        # ImageNet-normalised inputs, so fold the normalisation in here.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if backbone_ckpt:
            sd = torch.load(backbone_ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd))
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing or unexpected:
                import logging
                logging.getLogger("ttb").info(
                    f"backbone_ckpt loaded with {len(missing)} missing / {len(unexpected)} unexpected keys")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.proj(self.backbone(x))


class MatryoshkaHead(nn.Module):
    """Per-prefix classifiers over a nesting set M (ascending, last == D)."""

    def __init__(self, nesting: Sequence[int], num_classes: int, shared: bool = False):
        super().__init__()
        self.nesting = list(nesting)
        assert self.nesting == sorted(self.nesting), "nesting must be ascending"
        self.D = self.nesting[-1]
        self.num_classes = num_classes
        self.shared = shared
        if shared:
            # One weight; prefix m uses the first m input columns.
            self.fc = nn.Linear(self.D, num_classes)
        else:
            self.heads = nn.ModuleList([nn.Linear(m, num_classes) for m in self.nesting])

    def logits_at(self, z: torch.Tensor, m: int) -> torch.Tensor:
        """Logits using only the first ``m`` embedding coordinates."""
        zt = z[:, :m]
        if self.shared:
            w = self.fc.weight[:, :m]
            return F.linear(zt, w, self.fc.bias)
        idx = self.nesting.index(m)
        return self.heads[idx](zt)

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        return {m: self.logits_at(z, m) for m in self.nesting}


class NestedClassifier(nn.Module):
    """Encoder + Matryoshka head."""

    def __init__(self, encoder: nn.Module, head: MatryoshkaHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.nesting = head.nesting
        self.D = head.D

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.head(self.embed(x))

    def logits_at(self, x_or_z: torch.Tensor, m: int, is_embedding: bool = False) -> torch.Tensor:
        z = x_or_z if is_embedding else self.embed(x_or_z)
        return self.head.logits_at(z, m)


def build_model(
    arch: str,
    nesting: Sequence[int],
    num_classes: int,
    in_channels: int = 3,
    shared_head: bool = False,
    in_features: int = 50,
    pretrained: bool = True,
    backbone_ckpt: str = "",
) -> NestedClassifier:
    D = list(nesting)[-1]
    if arch == "mlp":
        enc: nn.Module = MLPEncoder(embed_dim=D, in_features=in_features)
    elif arch == "cnn_tiny":
        enc = TinyCNNEncoder(embed_dim=D, in_channels=in_channels)
    elif arch in ("resnet50", "resnet18_tv"):
        enc = TorchvisionResNetEncoder(embed_dim=D, arch=arch, pretrained=pretrained, backbone_ckpt=backbone_ckpt)
    else:
        enc = ResNetEncoder(embed_dim=D, in_channels=in_channels, arch=arch)
    head = MatryoshkaHead(nesting, num_classes, shared=shared_head)
    return NestedClassifier(enc, head)
