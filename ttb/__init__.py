"""Truncation-Triggered Backdoors in Matryoshka visual representations.

A research artifact studying a backdoor that is dormant at the full embedding
dimension (where audits run) and active only when the representation is
truncated to a low prefix for a bandwidth-constrained 6G uplink, together with
a prefix-consistency defence (runtime gate, cross-level audit, cleanup).

Academic reproducibility code on public benchmark classifiers. The defence is a
first-class component; see README and docs/THREAT_MODEL.md.
"""

__version__ = "0.1.0"
