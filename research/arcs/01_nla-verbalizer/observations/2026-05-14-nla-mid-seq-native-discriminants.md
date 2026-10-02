# Mid-Seq Native Discriminants Confirm Protocol-Coupling; Map Cross-Protocol Axis Stability

**Date:** 2026-05-14
**Toolkit:** `nla_mid_seq_native_compare.py`
**Inputs:** `vocab_atlas.pt` (eop), `mid_seq_vocab_atlas.pt`, `pairwise_and_hotdims.pt`
**Output:** `mid_seq_native_compare.pt`
**Figures:** `fig33_native_signal_lift.png`, `fig34_cross_protocol_axis_cos.png`
**Private-tracker ID:** MAIN-70, a follow-up to MAIN-44 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-70 predicted

[MAIN-44](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) found that 23 end-of-prompt-derived discriminants project mid-seq h's nearly orthogonally (+0.0491 aggregate). The natural follow-up: build mid-seq-NATIVE discriminants (same `mean(cat) - mean(non-cat)` recipe applied to mid-seq captures) and check whether in-protocol signal lifts to match the eop in-protocol +0.4022. Prediction: yes, ~+0.4 (confirming protocol-coupling). Also map per-category cross-protocol axis stability via the 23×23 cosine matrix.

## What actually happened

Computed both discriminant families and crossed each protocol's captures against each:

| captures × discriminants | aggregate signal | argmax accuracy |
|---|---|---|
| eop-h × eop-discr (in-protocol) | +0.4022 | 75.37% |
| eop-h × mid-discr (cross-protocol) | +0.0369 | 6.96% |
| mid-h × eop-discr (cross-protocol, MAIN-44) | +0.0491 | 32.04% |
| **mid-h × mid-discr (in-protocol)** | **+0.5632** | **97.10%** |

Native mid-seq discriminants give the *highest* in-protocol signal of either protocol — 40% higher than eop's in-protocol baseline, and argmax accuracy reaches 97.10% (vs 75.37% at eop). The basis-design is **protocol-coupled by construction**: a per-protocol family of discriminants each yields strong within-protocol classification.

The asymmetric cross-protocol numbers (eop-h × mid-discr = +0.037 < mid-h × eop-discr = +0.049) are mildly suggestive: mid-discriminants generalize slightly *worse* to eop-h than eop-discriminants do to mid-h. Possible cause: mid-seq captures have lower variance in carrier context (every anchor uses the same surrounding prompt), so mid-discriminants have a sharper but more position-specific direction.

## Cross-protocol axis stability (fig34)

The 23×23 cosine matrix `cos(d_eop_C, d_mid_D)` quantifies how each category's discriminant direction transfers across protocols:

| Statistic | Value |
|---|---|
| Mean diagonal (same-category, cross-protocol) | +0.0784 |
| Max diagonal | +0.1704 (`emotion`) |
| Min diagonal | +0.0126 (`p_quote`) |
| Mean off-diagonal | -0.0009 |

Diagonal entries are weakly positive (cosines +0.01 to +0.17), off-diagonal mean is essentially zero — so each category's axis at one protocol points in a direction that's *closer to its own axis at the other protocol than to a random other category's axis*, but only modestly so.

**Top-5 most-stable axes** (largest diagonal cosines) — all content-bearing categories:

| category | cos(d_eop, d_mid) |
|---|---|
| emotion | +0.170 |
| capital | +0.151 |
| country | +0.150 |
| nature | +0.131 |
| refusal | +0.127 [qualified 2026-09-28: see Evidence] |

**Top-5 least-stable axes** — function-word + punctuation [qualified 2026-09-28: see Evidence]:

| category | cos(d_eop, d_mid) |
|---|---|
| p_quote | +0.013 |
| p_special | +0.026 |
| math_op | +0.028 [qualified 2026-09-28: see Evidence] |
| auxiliary | +0.032 |
| preposition | +0.043 |

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). "Recomputed" values are read from `cells`, `cross_cos` and `diag_cos` in [`../data/mid_seq_native_compare.pt`](../data/mid_seq_native_compare.pt).

- **In-protocol mid-sequence cell (AUDIT 16):** `mid-h × mid-discr in-protocol signal (40% higher than eop in-protocol)` = 0.5632; `mid-h × mid-discr argmax accuracy` = 0.971.
- **Cross-protocol mid-sequence cell (AUDIT 15):** `mid_seq aggregate within-class signal (~8x weaker than eop)` = 0.0491; `mid_seq aggregate argmax accuracy (per-cat mean, vs 75.4% eop)` = 0.3204.
- **Other two cells (recomputed, `cells`):** eop × eop `agg_signal` 0.4022, `agg_acc` 0.7537; eop × mid `agg_signal` 0.0369, `agg_acc` 0.0696. No AUDIT line checks these two cells. Ratio of the in-protocol signals: 0.5632 / 0.4022 = 1.400.
- **Axis stability (AUDIT 16):** `mean cross-protocol diagonal cosine (axis stability)` = 0.0784; `max diagonal (emotion most-stable axis)` = 0.1704; `emotion-emotion cross-protocol cosine` = 0.1704.
- **Rest of the fig34 statistics (recomputed):** minimum diagonal 0.0126 (p_quote); mean off-diagonal of `cross_cos` −0.0009. `diag_cos` at 4 decimals: emotion 0.1704, capital 0.1513, country 0.1503, nature 0.1314, refusal 0.1275, codemath 0.1266; p_quote 0.0126, p_special 0.0258, math_op 0.0275, auxiliary 0.0323, preposition 0.0428. The tables match these rounded to 3 decimals except two rows: refusal rounds to 0.128 (table +0.127) and math_op to 0.027 (table +0.028). The six content categories are the six highest diagonals, from 0.1266 to 0.1704, with mean 0.1429. codemath, the sixth, is not in the top-5 table.
- **Grouping of the least-stable axes:** in the [vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) Vocabulary section, math_op is one of the 4 math ops in the numbers/operators group, not a function word or punctuation. The bottom five are therefore punctuation (p_quote, p_special), numbers/operators (math_op) and function words (auxiliary, preposition).
- **PC1 (AUDIT 12):** `vocab atlas PC1 fraction (sink-removed)` = 0.3349. That PCA is the fig19 scatter ([vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) Finding 2). fig21 is the cosine-to-anchor plot of the interpolation steps and has no PC1.
- **fig21's anchor switch (recomputed with the fig21 recipe in `examples/nla_vocab_atlas_render.py`: cosine of each step of [`../data/interpolation_flipbook.pt`](../data/interpolation_flipbook.pt) to the 128 atlas anchors, sink dims zeroed):** the top-1 anchor is Madrid (capital) at every step up to t=0.368 and autumn (nature) from t=0.421 on. The top-1 cosines on either side of the switch are 0.439 (t=0.368) and 0.438 (t=0.421). The AV flip that the dense run locates is between t=0.4450 and t=0.4475 ([dense interpolation](2026-05-15-nla-dense-interp-near-pivot.md) Evidence), so the fig21 switch and the AV flip are not at the same step. Cosines to fixed anchors vary smoothly along a straight line, so a change of top-1 anchor does not by itself show a discontinuous transition.
- **Scoring is in-sample (from `examples/nla_mid_seq_native_compare.py`):** each family of 23 directions is fit on all captures of its protocol and then scored on the same captures. Category sizes range from 2 (p_dash, p_quote) to 12 (nature), counted from `rows[].n_mid` in [`../data/mid_seq_compare.pt`](../data/mid_seq_compare.pt) (`n_eop` is equal in every category), so a scored capture is part of its own in-class mean.

## What this shows about layer-20 geometry

The content vs function distinction shows up again as the dominant organizing principle — and now at a deeper level than fig21's PC1. [qualified 2026-09-28: see Evidence] Per-category axes for content concepts (country, emotion, nature) have a modest position-invariant component, while punctuation and function words' axes are *almost entirely* position-determined. This matches the dominant PC1 (content-vs-function, 33.5% variance) found earlier in the sink-removed vocab atlas — but is a stronger statement: not just that content and function words occupy different *regions*, but that the *direction* in which a content category fans out from the population mean is partially preserved across positions, while function-word axes are constructed anew at each position.

This refines the MAIN-44 closure: the basis isn't *purely* end-of-prompt-protocol-coupled. There's a small protocol-invariant component (~+0.15 cosine for content categories). [qualified 2026-09-28: see Evidence] But it's small enough that cross-protocol projection collapses to noise.

## Implications for the glyph viz primitive

A discriminant glyph is **only meaningful for h captured at the same protocol the discriminants were derived from**. Cross-protocol use should be flagged. Two paths forward:

1. **Per-protocol glyph family.** Have multiple discriminant sets (eop, mid-seq, mid-generation, etc.), and pick the matching one based on the capture protocol. Avoids cross-protocol mismatch but uses N times the visual real estate.
2. **Stable-axis-only glyph.** Build a glyph using only the top-K most-stable axes (the 5 content-category axes here). [qualified 2026-09-28: see Evidence] Smaller glyph, but the axes are interpretable as carrying a position-invariant semantic component. Probably the right primitive for cross-protocol comparison views.

The interpolation flipbook (fig17/fig25) used cross-protocol-by-default discriminants on interpolated h's. The strong t=0.421 transition there ([MAIN-25](2026-05-13-nla-interpolation-flipbook.md)) is **still real** — fig21 confirmed it via cosine-to-vocab-anchors (no discriminants) [qualified 2026-09-28: see Evidence] — but the per-step glyph readings on fig25 should be interpreted with caution since the h's at intermediate t are out-of-protocol for the eop discriminants.

## Cross-arc lessons

MAIN-44 + MAIN-70 together turn what looked like a null result into a productive characterization of the basis. The pattern: **null result + native re-derivation reveals what the failed basis was missing**. Worth keeping as a methodological template — when a basis "fails" cross-protocol, build the native basis and compare directly to map what *is* preserved.

## Reproducibility

```bash
# Both direction families, the 4 cells, and fig33 + fig34 (no model load;
# writes .cache/nla_artifacts/mid_seq_native_compare.pt)
python examples/nla_mid_seq_native_compare.py

# Model-free check (AUDIT 15 and 16); reads the cache copy first, else the committed ../data/ copy
python examples/nla_audit_findings.py
```

The committed copy of the output is [`../data/mid_seq_native_compare.pt`](../data/mid_seq_native_compare.pt). The [figure inventory](figures/INVENTORY.md) explains why re-running the script to re-render fig33 also rewrites this artifact.

## Hypotheses

### H1 — The higher mid-sequence in-protocol signal reflects separability, or in-sample scoring

The mid-sequence in-protocol signal (0.5632) is 1.40× the end-of-prompt one (0.4022). Two readings fit that:
- **Separability.** Mid-sequence captures of one category are more alike than end-of-prompt captures, so their mean-contrast directions separate categories better.
- **In-sample scoring.** Both cells score each capture against a direction whose in-class mean includes that capture, in categories of 2 to 12 anchors. The identical carrier context of the mid-sequence captures can make this inflation larger there.

**Test (no model load):** recompute both in-protocol cells leave-one-anchor-out. For each scored anchor, fit the 23 directions of its protocol without that anchor, then project the anchor onto them. The scored anchor is then held out of its own fitting set in both protocols. Categories of 2 anchors leave one in-class capture per fit and can be reported separately.

The statistic is the retained fraction r = (held-out mid − held-out end-of-prompt) / (0.5632 − 0.4022). The calculation is deterministic on the committed captures, so r has no sampling noise on this anchor set.
- The separability reading predicts r > 1: tighter mid-sequence classes put each capture closer to its class mean, so in-sample inflation is smaller there than at end of prompt, and holding anchors out lowers the end-of-prompt cell more.
- The in-sample reading predicts r ≤ 0: the inflation is the gap, and holding anchors out closes or reverses it.

What the outcomes decide:
- r ≤ 0: the gap came from in-sample scoring.
- r ≥ 1: in-sample inflation does not explain the gap.
- 0 < r < 1: both contribute, and r is the fraction of the gap the held-out scoring keeps.

The result holds for these 128 anchors. Whether it carries to other anchor sets is a separate question this test does not address.

## Follow-ups

1. The H1 test has not run.
2. No AUDIT line checks the eop × eop and eop × mid cells, the minimum diagonal, the off-diagonal mean or the top-5 and bottom-5 tables. They rest on the recomputation in Evidence.
3. The stable-axis-only glyph (path 2 in Implications) has not been built.
4. The protocol-invariant subspace question from Cross-arc lessons is README item D4 ([#7](https://github.com/skothr/llm-research/issues/7)).
5. The committed fig33 PNG carries a stale title (see the [figure inventory](figures/INVENTORY.md) entry).

## References

- [Mid-seq vocab atlas null result](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md): the cross-protocol run this note follows up, and the source of the mid-sequence captures.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md): the end-of-prompt captures and the PC1 finding.
- [Discriminant validation](2026-05-13-nla-discriminant-validation.md): the mean-contrast recipe used for both direction families.
- [Arc README](../README.md) § F2 and limitation L3.
- [Figure inventory](figures/INVENTORY.md), fig33 and fig34.
