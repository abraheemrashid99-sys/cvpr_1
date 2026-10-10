# Threat model

**Setting.** A nested image encoder is deployed as a rate-adaptive
semantic-communication front-end. The device runs the encoder in full and
transmits the first `m` embedding coordinates, where `m = m(γ)` is chosen by the
rate adapter from the instantaneous link SNR `γ`. The edge/cloud runs the task
head on whatever prefix arrives.

**Attacker.** A supplier of the pretrained nested encoder (a model-hub upload,
an outsourced fine-tune, or a third-party model in an Open-RAN deployment).
- *Knows:* architecture, nesting set `M`, and that deployment truncates. Does
  not need the channel statistics.
- *Can do:* choose training data/procedure and release weights. Cannot change
  inference code, the rate adapter, the channel, or the audit.
- *Goal:* (i) stealth — at audited prefixes (default `{D}`) triggered inputs
  behave as on a clean model; (ii) effect — at victim prefixes `m ≤ m*`
  triggered inputs hit the target class; (iii) utility — clean accuracy at every
  prefix matches a clean nested model.

**Defender.** The operator. Holds a small clean validation set (`defender_frac`
of training data, never shown to the attacker), can run inference and fine-tune,
and by default audits at full dimension. The defences remove the
truncation-specific advantage regardless of which prefixes are audited.

**Out of scope.** Channel jamming, physical-layer attacks, and attacks that
require control of the inference code.

**Ethics.** Public datasets and models only; simulated channel; classification
only. The defence (gate + audit + cleanup) is released alongside the attack. Do
not distribute backdoored checkpoints under clean-looking names.
