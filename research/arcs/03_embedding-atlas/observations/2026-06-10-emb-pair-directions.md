# Relation directions in W_E: every tested relation beats chance, but the shared direction is mostly class-level, not pair-level

**Date / context:** 2026-06-10. Model/protocol as in
[2026-06-10-emb-global-geometry.md](2026-06-10-emb-global-geometry.md).
103 intact aligned pairs across 10 curated kinds + 397 mechanical
leading-space/bare twins. Derivation: `examples/emb_pair_directions.py`
(seed 20260610, 200-permutation baseline); locked by AUDIT 7. Figure: fig10.

## Method (and why)

For pair kind k with pairs (a_i, b_i): delta_i = E[b_i] - E[a_i];
d_bar = normalize(mean delta_i); **consistency** = mean_i cos(delta_i, d_bar).
Global mean-centering cancels in differences, so there is one space.
The permutation baseline re-pairs the b-side randomly *within the kind* —
crucial: a shuffled "pair" still points from the a-class to the b-class, so
the baseline absorbs everything attributable to class-centroid geometry,
isolating the *specifically paired* component.

## Findings

**F-P1. All 11 kinds beat their permutation baseline** (every consistency >
shuffle mean; most by far more than 2 shuffle-std):

```
kind        n   consistency  shuffle (±std)   margin
lang_of     10    +0.4896    +0.4503 ±0.0043   +0.039
past         9    +0.4228    +0.3684 ±0.0067   +0.054
register    10    +0.4166    +0.3838 ±0.0039   +0.033
capital_of  12    +0.4085    +0.3881 ±0.0022   +0.020
gender      10    +0.3859    +0.3380 ±0.0057   +0.048
antonym      9    +0.3754    +0.3563 ±0.0033   +0.019
case        11    +0.3637    +0.3159 ±0.0046   +0.048
valence     10    +0.3540    +0.3356 ±0.0029   +0.018
plural      20    +0.3097    +0.2618 ±0.0029   +0.048
space_twin 397    +0.1968    +0.1577 ±0.0002   +0.039
(xlat        2    +0.7160    +0.7004 ±0.0149   n too small to read)
```

**F-P2 (the headline nuance). The shuffle baselines are themselves high
(+0.16-0.45) — most of each "relation direction" is shared CLASS geometry,
not per-pair structure.** Replacing "France->French" with "France->Japanese"
keeps ~90% of the consistency. So W_E encodes "a direction from
country-region to language-region" robustly, but the word-specific pairing
adds only +0.02-0.05 of alignment on top. The word2vec-era picture of crisp
per-pair analogy arithmetic is, at this table's layer 0, mostly a
class-offset phenomenon with a thin paired residue. [INTUITION: the handle
exists, but it is a handle on the *category*, not on the individual word
pair; formally, delta_i ~ (mu_B - mu_A) + noise_i with the paired component
small relative to centroid offset.]

**F-P3. Morphological relations carry the strongest *pair-specific*
margins** (past +0.054, plural/gender/case ~+0.048) — consistent with
regular morphology being more mechanically encoded than semantic relations
(valence +0.018, antonym +0.019 barely exceed class-level).

**F-P4. The leading-space direction is real but weak per-pair** (space_twin:
397 twins, consistency +0.1968 vs +0.1577 chance) — a shared
"begins-a-new-word" component exists, with large per-token variance (worst
twin: 'Denver').

## Reproducibility

```bash
python examples/emb_pair_directions.py   # derive + console table (model-free)
python examples/emb_pairs_render.py      # fig10
python examples/emb_audit_findings.py    # AUDIT 7
```

## Hypotheses / follow-ups

- H1: pair-specific margins should grow with depth as context binds pairs
  (test at layer 20 via the bridge protocol, or with the rope-vis q/k hooks).
- H2: projecting *held-out* pairs onto d_bar measures generalization — e.g.
  does ' Stockholm' - ' Sweden' align with the capital_of direction? Cheap
  and decisive next probe for the "handle" framing.
- H3: the small valence margin predicts sentiment is NOT linearly readable
  from W_E alone — testable with a linear probe trained on the battery.

## Caveats

- n = 9-20 per curated kind; consistency estimates are coarse (the shuffle
  std understates total uncertainty since the pair SET is also small).
- Battery pairs are prototypical/regular; irregular or rare-token pairs
  (already dropped by single-token attrition: swam, eau, chien...) might
  behave differently.
