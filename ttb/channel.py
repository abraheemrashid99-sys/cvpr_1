"""Rate adapter and wireless channel for the 6G semantic-communication setting.

The device holds the full embedding z in R^D (the encoder runs in full). The
rate adapter chooses the largest prefix the instantaneous link can carry:

    m(gamma) = max { m in M : m * b <= n_c * log2(1 + gamma) }

where b is bits per coordinate, n_c the channel uses per sample, gamma the SNR.

Two transport paths:
  * digital  : an ideal code. The chosen prefix arrives intact (optionally
               b-bit uniformly quantised to model finite precision).
  * analogue : the prefix is power-normalised, sent as m/2 complex symbols,
               and received with additive noise set by gamma (deep-JSCC style).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def required_snr_db(m: int, b: int, n_c: int) -> float:
    """SNR (dB) needed to carry a prefix of length m."""
    se = m * b / n_c
    snr = 2 ** se - 1
    return 10 * math.log10(snr) if snr > 0 else float("-inf")


def prefix_for_snr(gamma_linear: float, nesting: Sequence[int], b: int, n_c: int) -> int:
    """Largest prefix that fits the link at linear SNR gamma."""
    budget_bits = n_c * math.log2(1.0 + max(gamma_linear, 1e-12))
    affordable = [m for m in nesting if m * b <= budget_bits]
    return max(affordable) if affordable else min(nesting)


def db_to_linear(db: float) -> float:
    return 10 ** (db / 10.0)


def quantize_uniform(z: torch.Tensor, b: int) -> torch.Tensor:
    """Symmetric uniform b-bit quantisation per sample (per-row max-abs scale)."""
    if b <= 0:
        return z
    levels = 2 ** b - 1
    scale = z.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    q = torch.round((z / scale).clamp(-1, 1) * (levels / 2)) / (levels / 2)
    return q * scale


class Channel:
    """Applies rate adaptation, quantisation and (optionally) analogue noise."""

    def __init__(
        self,
        nesting: Sequence[int],
        b: int = 8,
        n_c: int = 1024,
        mode: str = "digital",
        fading: str = "awgn",
    ):
        self.nesting = list(nesting)
        self.b = b
        self.n_c = n_c
        self.mode = mode
        self.fading = fading

    def sample_snr(self, mean_snr_db: float, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return n linear SNR draws."""
        mean_lin = db_to_linear(mean_snr_db)
        if self.fading == "awgn":
            return torch.full((n,), mean_lin)
        # Rayleigh block fading: |h|^2 is exponential with the given mean.
        u = torch.rand(n, generator=generator)
        return -mean_lin * torch.log(u.clamp_min(1e-12))

    def transmit(
        self,
        z: torch.Tensor,
        m: int,
        snr_linear: torch.Tensor | float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Transmit the first m coordinates; return the received prefix (length m)."""
        zt = z[:, :m]
        if self.mode == "digital":
            return quantize_uniform(zt, self.b)
        # analogue: power-normalise, add complex-equivalent real AWGN
        if snr_linear is None:
            raise ValueError("analogue mode needs snr_linear")
        power = zt.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
        x = zt / power.sqrt()  # unit average power
        if isinstance(snr_linear, (int, float)):
            snr_t = torch.full((zt.shape[0], 1), float(snr_linear))
        else:
            snr_t = snr_linear.view(-1, 1).to(zt.dtype)
        noise_std = (1.0 / snr_t.clamp_min(1e-12)).sqrt()
        noise = torch.randn(zt.shape, generator=generator) * noise_std
        y = x + noise
        return y * power.sqrt()  # undo normalisation (receiver knows scale)
