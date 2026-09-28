# Mid-Sequence Vocab Atlas: Null Result + Refines the Discriminant-Validation Reading

**Date:** 2026-05-14
**Toolkit:** `nla_mid_seq_vocab_atlas_capture.py`, `nla_mid_seq_vocab_atlas_compare.py`, `nla_mid_seq_vocab_atlas_render.py`
**Inputs:** `vocab_atlas.pt` (end-of-prompt), `pairwise_and_hotdims.pt` (sink dim labels)
**Outputs:** `mid_seq_vocab_atlas.pt`, `mid_seq_compare.pt`
**Figures:** `fig31_mid_seq_signal_vs_noise.png`, `fig32_mid_seq_argmax_accuracy.png`
**Private-tracker ID:** MAIN-44 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-44 predicted

MAIN-26 ([discriminant-validation](2026-05-13-nla-discriminant-validation.md)) found the 23-discriminant basis acts as a prompt-TOPIC classifier rather than a token-presence detector — `happy → emotion` projection was only +0.083 ± 0.061 across 4 prefix-length contexts at end-of-prompt positions. **Interpretation at the time:** end-of-prompt h has integrated the full user message into a topic representation, drowning token-specific identity. **Predicted fix:** capture each anchor MID-SEQUENCE (inside a carrier prompt that continues past the anchor) and project onto the existing discriminants. If the integration-overwhelm interpretation holds, `expected_proj` should rise to ≥ +0.3.

## What actually happened

Captured h[20] for all 128 vocab anchors × 23 categories at the anchor token's mid-sequence position inside the carrier:

```
"The text contains many words. Here is one specific word: {anchor} continues throughout subsequent discussion paragraphs."
```

Anchor lands at position 36-38 of ~48 tokens (~75% of the sequence; 10-12 tokens of trailing context). [qualified 2026-09-28: see Evidence] Position-finding via prefix-tokenization (BPE is left-to-right; tokenize `chat_str[:span_end]` to count tokens in the anchor span). All 128 anchors captured with 0 skips.

**Projected mid-sequence h's onto the end-of-prompt-derived discriminants:**

| Protocol | mean within-class signal | mean max-off-class noise [qualified 2026-09-28: see Evidence] | argmax accuracy |
|---|---|---|---|
| end-of-prompt (baseline) | +0.4022 | +0.3911 | 75.37% |
| mid-sequence (this test) | **+0.0491** | +0.0623 | **32.04%** |

The mid-sequence within-class signal is **~8× weaker** than end-of-prompt, not stronger. The MAIN-44 hypothesis is **rejected** — mid-sequence positioning does not enable token-presence detection. It produces signal nearly orthogonal to the basis.

Specifically for the MAIN-26 anchor "happy" (emotion):
- end-of-prompt projection onto emotion discriminant: +0.1755 (argmax correct)
- mid-sequence projection: +0.0759 (still argmax correct, but signal dropped 57%)

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). "Recomputed" values are read from `rows[]` and `by_cat[]` in [`../data/mid_seq_compare.pt`](../data/mid_seq_compare.pt), from `captures[]` in [`../data/mid_seq_vocab_atlas.pt`](../data/mid_seq_vocab_atlas.pt), and, for the stability protocol, from [`../data/discriminant_stability.pt`](../data/discriminant_stability.pt) with the AUDIT 14 recipe.

- **Mid-sequence run (AUDIT 15):** `mid_seq capture count` = 128; `mid_seq skip count` = 0; `mid_seq aggregate within-class signal (~8x weaker than eop)` = 0.0491; `mid_seq aggregate argmax accuracy (per-cat mean, vs 75.4% eop)` = 0.3204; `mid_seq happy → emotion signal` = 0.0759.
- **End-of-prompt baseline (recomputed):** the unweighted mean over the 23 `rows[]` entries, as in the compare script's AGGREGATE row, gives `eop_signal_mean` 0.4022 and `eop_argmax_acc` 0.7537. The same values are stored as `cells[('eop', 'eop')]` in [`../data/mid_seq_native_compare.pt`](../data/mid_seq_native_compare.pt). No AUDIT line checks the baseline. Ratio: 0.4022 / 0.0491 = 8.19.
- **Noise column (recomputed):** the unweighted mean of `rows[].eop_noise_max_mean` is 0.3631 and of `rows[].mid_noise_max_mean` 0.0609. The mean of `best_other_v` over the 128 captures per protocol in `by_cat[]` is 0.3491 and 0.0572. Neither reading gives the table's +0.3911 or +0.0623, and the compare script prints no aggregate noise value.
- **`happy` (recomputed, `by_cat['emotion']`):** end-of-prompt `signal` 0.1755 with `argmax_cat` emotion; mid-sequence 0.0759 with `argmax_cat` emotion. The drop is 1 − 0.0759 / 0.1755 = 56.8%.
- **Anchor position (recomputed, `captures[]`):** `anchor_first_pos` is 36 for all 128 captures and `anchor_last_pos` is 36 to 38. `n_input_tokens` is 48 for 116 captures, 49 for 11 and 50 for 1. Every capture has 11 tokens after the anchor, and the anchor span covers 72% to 76% of the sequence.
- **Random floor (recomputed):** 3577 is the 3584 dims minus the 7 zeroed sink dims. 1/√3577 = 0.0167 is the RMS cosine of a random unit vector with a fixed direction; its expected |cos| is √(2/(π·3577)) = 0.0133, and a signed mean over random directions has expected value 0. 0.0491 / 0.0167 = 2.94 and 0.0491 / 0.0133 = 3.68, so 0.017 is not the floor for the signed mean signal.
- **Per-category accuracy (recomputed, `rows[]`):** mid-sequence accuracy is higher than end-of-prompt in exactly 6 categories, with the table's values: nature 0.5 → 1.0 (n=12), emotion 0.5 → 1.0 (n=6), quantifier 0.8 → 1.0 (n=5), conjunction 0.167 → 0.667 (n=6), pronoun 0.286 → 0.429 (n=7), p_special 0.0 → 0.25 (n=4). In anchors, the gains are 6, 3, 1, 3, 1 and 1. country and capital stay at 1.0, and the other 15 categories fall.
- **Signal and noise in the six gaining categories (recomputed, `rows[]`, ratio of `signal_mean` to `noise_max_mean`, end of prompt → mid-sequence):** nature 1.07 → 2.05, emotion 1.00 → 1.74, quantifier 1.17 → 1.97, conjunction 0.89 → 1.10, pronoun 0.88 → 0.93, p_special 0.81 → 0.36. In p_special the signal falls 14.2× and the noise 6.2×, so its noise does not fall faster than its signal. In pronoun and p_special the mid-sequence noise stays above the signal.
- **End-of-prompt centroid correlation (AUDIT 13):** `mean cross-centroid cosine (the all-axes-active problem)` = 0.8504.
- **Stability protocol (recomputed, AUDIT 14 recipe: sink dims zeroed, unit-normalized, projected onto the expected category's direction, mean over the 4 contexts):** refuse 0.0523, happy 0.0825, `.` 0.1190, the 0.1826, Paris 0.1848, France 0.2189, 7 0.2811, function 0.2838. The range is 0.052 to 0.284, not +0.08 to +0.18. AUDIT 14 checks only `happy → emotion projection mean (weak; the topic-not-token finding)` = 0.0825.
- **Prompt lengths (recomputed, `n_input_tokens`, chat template included):** vocab atlas 30 to 32; stability scan 30 to 43 (single 30–31, short 32–33, medium 33–34, long 42–43); mid-sequence 48 to 50. A one-token user message gives 30 tokens, so the user message alone is about 1 to 3 tokens for the vocab atlas, 1 to 14 for the stability scan and 19 to 21 for the carrier. The "~5-15 tokens" for the stability scan leaves out the single-token context, and the "~48 tokens" for the carrier counts the template while the other two items count only the message.

## Why the prediction was wrong — what mid-sequence really shows

At mid-sequence, h projects approximately uniformly weakly across the entire 23-discriminant subspace. Expected magnitude of a random unit vector's cosine projection in R^3577 is ~1/√3577 ≈ 0.017; the observed +0.05 is only 3× that floor. [qualified 2026-09-28: see Evidence] **The 23-discriminant basis is more specifically end-of-prompt-protocol-coupled than just "prompt-topic".** Mid-sequence h's live in a different region of layer-20 residual space — different enough that the within-class direction (derived from end-of-prompt geometry) doesn't preferentially align with mid-sequence captures of the same category.

This **refines MAIN-26's claim**:
- MAIN-26: basis classifies prompt-TOPIC, not token-presence
- MAIN-44 refinement: basis is **protocol-coupled to end-of-prompt position** — it doesn't transfer even to the same anchor token captured at a different sequence position. The "topic" interpretation was a category error: the basis isn't tracking topic; it's tracking *whatever h represents at end-of-prompt in this specific chat-template-wrapped protocol*.

## The interesting wrinkle: 6 categories actually classify BETTER mid-seq

While aggregate accuracy drops 75% → 32%, some categories' argmax accuracy *increased* under the mid-sequence protocol:

| category | eop argmax acc | mid argmax acc |
|---|---|---|
| nature | 50% | **100%** |
| emotion | 50% | **100%** |
| quantifier | 80% | **100%** |
| conjunction | 17% | **67%** |
| pronoun | 29% | **43%** |
| p_special | 0% | 25% |

For these categories, mid-sequence h has a *cleaner* directional signal — the discriminant's noise floor falls faster than its signal does. [qualified 2026-09-28: see Evidence] The pattern suggests the end-of-prompt protocol's high cross-category correlation (mean centroid cos +0.85 at eop, per fig25) was eating into accuracy at eop, and the mid-seq subspace removes that confound for some categories while damaging others.

## What we now know about layer-20 geometry

Three protocols, three different h subspaces, ranked by within-class signal on the end-of-prompt-derived basis:

1. **End-of-prompt of {anchor} alone** (vocab atlas, ~3 tokens) — signal +0.40, accuracy 75%
2. **End-of-prompt of stability-scan prefixed prompt** (MAIN-26, ~5-15 tokens) — signal +0.08-0.18, accuracy not measured [qualified 2026-09-28: see Evidence]
3. **Mid-sequence inside long carrier** (this work, ~48 tokens, anchor at pos 36-38) — signal +0.05, accuracy 32% [qualified 2026-09-28: see Evidence]

All three "represent the anchor token" in some sense, but the resulting h's live in measurably different parts of layer-20 space. **The basis is a fingerprint of one specific protocol, not a generic semantic axis.**

## Implications for the glyph viz primitive

The discriminant glyph (fig25/26) is the right primitive **only for h's captured at end-of-single-token-user-message position**. Applying it to h captured at other positions — generation steps mid-trajectory, mid-prompt captures, interpolation-derived h — produces glyphs whose rays don't carry the intended semantic interpretation. The interpolation flipbook (fig17/fig25) is borderline: anchors A and B are AR-encoded h's (not end-of-prompt captures), and the interpolated h's live somewhere those discriminants weren't designed to cover. The strong fig21 result (capitals→nature anchor switch at t=0.421) used cosine-to-vocab-anchor projection rather than discriminants — and that result still holds because it doesn't go through this protocol-coupled basis.

## Follow-ups — next experiments this opens

1. **Build mid-seq-NATIVE discriminants** from these 128 captures and project mid-seq h's onto those. Predict within-class signal ~+0.4 (matching the in-protocol signal at eop). If so, the basis-design protocol-coupling is confirmed and we have a per-protocol family of bases.
2. **Cross-protocol cosine alignment** between eop_category_mean and mid_category_mean per category. Maps how each category's representation drifts across protocols. This would tell us whether category-specific h has a stable "axis" across protocols (just rotated by the protocol) or whether categories are constructed differently at each position.
3. **Single-position vs multi-position discriminants:** Capture each anchor at 4 positions in the same carrier (position k=10, k=20, k=30, k=40). Test whether discriminants built from multi-position pooling generalize better than single-protocol ones. Speaks to whether "category" has a position-invariant axis at all in this layer.

**Status (2026-09-28):**
- Items 1 and 2 ran in [mid-seq native discriminants](2026-05-14-nla-mid-seq-native-discriminants.md). Item 2 was run on the discriminant directions (`d_eop` against `d_mid`), not on the category means this item names.
- Item 3 has no committed run. README item D4 ([#7](https://github.com/skothr/llm-research/issues/7)) covers the related protocol-invariant subspace question.
- The H1 test below has not run.
- The noise column of the results table does not reproduce from the committed data (Evidence), and no AUDIT line checks the end-of-prompt baseline or the per-category table.

## Cross-arc lessons

This is the **second null result of the arc** ([MAIN-47](2026-05-14-nla-hierarchical-classifier-null-result.md) was the first). Both came from over-extrapolating a single finding. MAIN-26 looked at one position protocol (end-of-prompt of varying-length prefix) and produced an interpretation ("end-of-prompt integration overwhelms token identity") that turned out too narrow. MAIN-44's failure refines it: the integration interpretation can't explain why mid-sequence is also a different subspace — the basis is protocol-coupled in a more general way.

When designing the next round of probes, the methodological discipline: **always test the basis at the protocol it was derived from AND at one alternative protocol before claiming the basis "measures" something interpretable.** Within-protocol signal is necessary but not sufficient — the cross-protocol behavior reveals what the basis *isn't* doing.

## Reproducibility

```bash
# Capture the 128 anchors mid-sequence in the carrier prompt (loads the base model;
# writes .cache/nla_artifacts/mid_seq_vocab_atlas.pt)
python examples/nla_mid_seq_vocab_atlas_capture.py

# Project onto the end-of-prompt directions (no model load;
# writes .cache/nla_artifacts/mid_seq_compare.pt)
python examples/nla_mid_seq_vocab_atlas_compare.py

# fig31 + fig32 (no model load)
python examples/nla_mid_seq_vocab_atlas_render.py

# Model-free check (AUDIT 15); reads the cache copy first, else the committed ../data/ copy
python examples/nla_audit_findings.py
```

The committed copies are [`../data/mid_seq_vocab_atlas.pt`](../data/mid_seq_vocab_atlas.pt) and [`../data/mid_seq_compare.pt`](../data/mid_seq_compare.pt). The scripts write to the gitignored working cache `.cache/nla_artifacts/`. Every read, including the compare step's, the render step's and the audit's, takes the cache copy when one exists and otherwise the committed copy in `../data/` (`examples/_nla_artifacts.py`). A clean clone therefore renders and audits the committed data, and after a re-run they read the new cache files.

## Hypotheses

### H1 — The six mid-sequence gains come from removing end-of-prompt correlation, or from small categories

Two readings fit the six categories that classify better mid-sequence:
- **Correlation removal.** The file's reading: high cross-category correlation at end of prompt suppresses these categories' accuracy, and the mid-sequence captures remove it.
- **Small-n.** The gains are 1 to 6 anchors in categories of 4 to 12 anchors. Three of them (pronoun, quantifier, p_special) are one anchor each. On this reading they are fluctuations unrelated to end-of-prompt correlation.

**Test (no model load):** for each of the 23 categories, compute its mean end-of-prompt centroid cosine to the other 22 centroids from `vocab_atlas.pt` (sink dims zeroed). Restrict to the 11 categories below 100% end-of-prompt accuracy (`rows[]`), since a category already at 100% cannot gain. Nothing is fit, so no hold-out is needed.

A plain correlation of this cosine with the accuracy change (mid − end-of-prompt) cannot separate the readings. The possible gain is capped at 1 − end-of-prompt accuracy, and noise between the two protocols moves low-accuracy categories up (regression to the mean). If high centroid cosine goes with low end-of-prompt accuracy, as the correlation-removal reading assumes, the small-n reading also predicts a positive correlation. The statistic is therefore the partial Spearman correlation of centroid cosine with mid-sequence accuracy, controlling for end-of-prompt accuracy, with a permutation p-value.
- The correlation-removal reading predicts a positive partial correlation.
- The small-n reading predicts a partial correlation near zero.

What the outcomes decide:
- A positive partial correlation at p < 0.05 supports the correlation-removal reading. It does not rule out the small-n reading for the three one-anchor gains.
- Any other result leaves H1 open. With 11 categories, the test can miss a moderate effect, so a null result does not rule out the correlation-removal reading.

Only the first outcome is decisive.

## References

- [Discriminant validation](2026-05-13-nla-discriminant-validation.md): the prompt-topic reading this run tests, and the stability scan.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md): the end-of-prompt captures the 23 directions are built from.
- [Mid-seq native discriminants](2026-05-14-nla-mid-seq-native-discriminants.md): the follow-up run of items 1 and 2.
- [Hierarchical classifier null result](2026-05-14-nla-hierarchical-classifier-null-result.md): the first null result named in Cross-arc lessons.
- [Arc README](../README.md) § F2 and limitation L3.
- [Figure inventory](figures/INVENTORY.md), fig31 and fig32.
