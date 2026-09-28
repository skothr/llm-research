# Hierarchical Re-Discrimination: Null Result + Reframing

**Date:** 2026-05-14
**Toolkit:** `nla_hierarchical_classifier.py`
**Inputs:** `vocab_atlas.pt` + `pairwise_and_hotdims.pt`
**Figure:** `fig30_hierarchical_accuracy.png`
**Private-tracker ID:** MAIN-47 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-47 predicted

Self-validation ([MAIN-26](2026-05-13-nla-discriminant-validation.md) / fig29) showed country-source captures had only 34% top-1 accuracy against the country discriminant — country and capital seemed to constantly swap. The hypothesis: build a sub-discriminator from `mean(country_h) − mean(capital_h)` (and similar for other sibling pairs with discriminant cos > +0.85). [qualified 2026-09-28: see Evidence] Apply when a capture's top-2 first-level categories form a sibling pair. Predicted lift toward 70%+ top-1 accuracy.

## What actually happened

Hierarchical scheme applied to 33 of 107 expected-mapped captures (siblings pair shows up as top-2). [qualified 2026-09-28: see Evidence] **It flipped exactly 1 capture's top-1 label**, lifting country accuracy from 34% → 38% (+1 / 29). All other categories: zero change. Overall: 44% → 45%.

| expected | n | baseline | hierarchical | Δ |
|---|---|---|---|---|
| codemath | 30 | 53% | 53% | +0% |
| **country** | 29 | **34%** | **38%** | **+3.4%** |
| nature | 45 | 47% | 47% | +0% |
| negation | 2 | 0% | 0% | +0% |
| refusal | 1 | 0% | 0% | +0% |
| OVERALL | 107 | 44% | 45% | +0.9% |

## Why the lift was tiny — three confounding effects

Diagnostic on the 8 source-country prompts (the cleanest country-content subset) revealed:

| prompt | baseline top-1 | what's actually happening |
|---|---|---|
| "What is the capital of France?" | **capital** | semantically correct — the prompt IS asking about Paris |
| "Tell me about Germany." | country | correct |
| "Brazil is famous for soccer." | country | correct |
| "The population of India is large." | country | correct |
| "Italy borders Switzerland." | **capital** | border-content; geographically-positioned info |
| "Spain is in Europe." | country | correct |
| "Japan has a unique culture." | country | correct |
| "China's economy is growing." | country | correct |

6 of 8 source country prompts already top `country` (75% top-1 accuracy on the clean subset). The 2 that top `capital` do so for semantically valid reasons. **The basis isn't actually confused — it's correctly responding to content.**

The reported 34% top-1 accuracy bundled three different things:

1. **Genuinely-confused captures (siblings swap)** — only the hierarchical scheme can fix this. We found **1 such case** in 107 total captures (the one country/capital flip that did fire).
2. **Correctly-classified-as-sibling captures** — e.g., "What is the capital of France?" tops capital because it IS a capital-content prompt. Sibling-aware classification correctly leaves these as capital. Counted as "wrong" in the original self-validation because the source pool was labeled `country`.
3. **Mislabeled "expected=country" captures** — the country_test set deliberately includes weird-framing prompts ("Justice is the country of the soul", "Portugal smells like Tuesdays", "If France were a sandwich"). These project to nature/emotion/quantifier [qualified 2026-09-28: see Evidence], **which is correct**. No classification scheme can lift these because the label is wrong.

## Reframing — the 23 discriminants are doing better than the 34% number suggested

The "country/capital top-1 swap" framing was misleading. The actual situation:

- **For prompts that are genuinely about a country-as-topic**: the country discriminant fires correctly. ~75% top-1 on the clean 8-prompt subset.
- **For prompts that mention countries but are about something else** (asking about a capital; making a metaphor; using country names as filler): the basis classifies into the appropriate category, which often isn't `country`. This is correct behavior counted as wrong by source-label-based self-validation.
- **For genuinely-ambiguous content** (e.g., "Italy borders Switzerland" — is that country-content or capital-content?): the basis picks ONE side, and the hierarchical sub-discriminator agrees. No conflict to resolve.

The discriminant basis built in MAIN-26 is more accurate than the self-validation numbers indicated. The audit pass and this null result together suggest **the self-validation methodology should weight by "label fidelity"** — prompts that are unambiguously country-content count more than weird-framing prompts in the same source pool.

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). "Recomputed" values use AUDIT 23's own recipe: the 23 sink-removed mean-contrast directions from [`../data/vocab_atlas.pt`](../data/vocab_atlas.pt), the 167-capture pool AUDIT 22 builds from [`../data/aggregate_faithfulness.pt`](../data/aggregate_faithfulness.pt), [`../data/rabbit_haiku_gen_trajectory.pt`](../data/rabbit_haiku_gen_trajectory.pt), [`../data/forced_continuation.pt`](../data/forced_continuation.pt) and [`../data/country_concept_vector.pt`](../data/country_concept_vector.pt), and its `expected_cat` mapping.

- **Null result (AUDIT 23):** `sibling pairs at discriminant cos > 0.80` = 6; `sibling-applicable captures (disambiguator fired)` = 33; `captures whose top-1 flipped (the null result)` = 1; `country scored-capture count (fig30)` = 29; `country baseline top-1 accuracy` = 0.3448; `country hierarchical top-1 accuracy` = 0.3793; `fig30 scored-capture total` = 107; `overall baseline top-1 accuracy` = 0.4393; `overall hierarchical top-1 accuracy` = 0.4486.
- **Sibling threshold:** `examples/nla_hierarchical_classifier.py` sets `SIBLING_COS_THRESHOLD = 0.80`, and AUDIT 23 lists the 6 pairs above it: country–capital, p_ender–p_internal, p_ender–math_op, p_internal–p_dash, p_dash–math_op, p_special–math_op. Recomputed: only country–capital (0.938) is above +0.85; the next pair is p_special–math_op at 0.843. The script's docstring still says +0.85.
- **Where the 33 comes from (recomputed):** the disambiguator fires on 33 of all 167 pool captures, the count AUDIT 23 checks. Within the 107 scored captures it fires on 24.
- **Per-category table (recomputed):** codemath 16/30 correct before and after; country 10/29 → 11/29; nature 21/45; negation 0/2; refusal 0/1. Country gains 0.3793 − 0.3448 = 0.0345 and overall 0.4486 − 0.4393 = 0.0093.
- **The flipped capture (recomputed):** aggregate prompt `factual_easy` ("What is the capital of France?"), generated token `' France'`, top-1 capital → country, expected category country.
- **Country scored set (recomputed):** the 29 country-scored captures are the 8 source prompts, the 13 test prompts, and 8 `factual_easy` generation captures.
- **The 8 source prompts (recomputed):** prompt text from `results` in `country_concept_vector.pt`, where the labels `SRC country 1` to `SRC country 8` follow the order of `h_country`. "What is the capital of France?" and "Italy borders Switzerland." have top-1 capital. The other six have top-1 country, giving 6 of 8. All eight have the other sibling second, so the disambiguator fires on each and changes none of them.
- **The 13 test prompts (recomputed, `h_test` in `country_concept_vector.pt`):** `expected_cat` scores every one as country. That includes "Mars is not a country.", "The molecule has six carbon atoms." and "Banana is yellow.". Top-1 for the weird-framing prompts: "If France were a sandwich, what would it taste like?" nature; "My pet hamster Belgium escaped this morning." emotion; "Portugal smells like Tuesdays to me." negation; "In a dream last night, I argued with Germany over breakfast cereal." emotion; "Justice is the country of the soul." emotion, with quantifier second. The top-1 categories are nature, emotion and negation. Quantifier appears only in second place.

## Hypotheses

### H1 (revised) — Discriminant accuracy is bounded above by label fidelity, not by basis quality

For any classifier-driven validation, the ceiling is the labeling quality of the test set. Hierarchical re-discrimination is the wrong intervention if the failure mode is mislabeled tests, not confused basis directions. **Test:** rebuild the country_test set with strict country-only prompts (no metaphors, no questions about capitals); measure baseline top-1 accuracy. Predict > 75% on the strict subset.

### H2 — The 1 capture that flipped is the only "true sibling swap" in the dataset

If true, the hierarchical scheme found and fixed every honest case of sibling-confusion. Generalizes the null result into a positive: the basis was *already* near-optimal at the discriminate-siblings task in this dataset.

## Methodology takeaway

When a single number ("34% top-1 accuracy") looks low, decompose it before designing a fix:

1. **Genuinely-wrong**: the basis lands on the wrong category despite the prompt being unambiguous.
2. **Right-by-other-name**: the basis correctly identifies a different aspect of the prompt than the source-label expected.
3. **Label-wrong**: the prompt was mislabeled in the test set.

Only (1) is fixable by improving the basis. (2) is a validation-methodology issue. (3) is a labeling issue. **Don't build a complex fix until you've classified the failure modes.**

## Reproducibility

```bash
python examples/nla_hierarchical_classifier.py
```

CPU only, ~5 seconds, no model loading.

## Follow-ups

1. The H1 test (a strict country-only test set) has not run.
2. H2 has no test and no run. Its premise is that the one flip is a true sibling swap. The flipped capture is the `' France'` token generated for "What is the capital of France?" (Evidence), a prompt the table under "Why the lift was tiny" classes as capital content.
3. The label-fidelity weighting proposed in Reframing is not implemented. AUDIT 22 and 23 still score by source label, and `expected_cat` still labels the three non-country control prompts as country.
4. The script docstring's +0.85 threshold disagrees with the `SIBLING_COS_THRESHOLD = 0.80` it uses.

## References

- [Discriminant validation](2026-05-13-nla-discriminant-validation.md) — produced the 34% top-1 number this issue tried to fix.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) — provides the centroids the sub-discriminators are computed from.
