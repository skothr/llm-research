# Concept Arithmetic Atlas: Category-Level Composition Preserves; Specific-Identity Analogies Fail [qualified 2026-09-28: see Evidence]

**Date:** 2026-05-14
**Toolkit:** `nla_concept_arithmetic_atlas.py`, `nla_concept_arithmetic_render.py`
**Inputs:** `vocab_atlas.pt`, AV `kitft/nla-qwen2.5-7b-L20-av`
**Output:** `concept_arithmetic_atlas.pt`
**Figure:** `fig35_concept_arithmetic_atlas.png`
**Private-tracker ID:** MAIN-48 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-48 predicted

Does NLA preserve word2vec-style additive/subtractive structure on layer-20 end-of-prompt h? If `vec(Paris) − vec(France) + vec(Germany) ≈ vec(Berlin)`, then the AV becomes a composable interpretation tool — every axis labeled, every arithmetic combination decodable. Tested 7 combinations (3 canonical analogies, 1 pure subtraction, 2 axis directions, 1 compound concept), rescaled to ||h||=150 (typical eop norm) [qualified 2026-09-28: see Evidence] before AV-decoding.

## What actually happened

The picture is **nuanced — not a clean yes or no.** Three observed patterns:

### A) Specific-identity analogies FAIL

| arithmetic | predicted | AV decoded (key content) |
|---|---|---|
| `Paris − France + Germany` | Berlin | **London** ("about the city of London") |
| `Tokyo − Japan + France` | Paris | **Spain** ("about the country") |
| `Berlin − Germany + United Kingdom` | London | **UK** (just outputs UK) |

None of the three analogies produced the predicted specific capital. Test #1 produced a capital (London) — wrong identity but **right category**. Test #2 produced a country (Spain) — wrong category entirely. Test #3 collapsed back to the +input term (UK).

The pattern is **inconsistent**: sometimes the arithmetic moves the h into the right category but picks a wrong identity within it; sometimes it loses the category direction entirely. word2vec-style analogies do not survive the NLA encoding at this layer.

### B) Category-level axis direction DOES preserve [qualified 2026-09-28: see Evidence]

`country_centroid − capital_centroid` (axis direction) decoded coherently as country-flavored content: "this country's population is diverse", "the world's religions", "a country's facts", "a document" — strongly country-themed rather than city-themed. The category-level axis is preserved through subtractive composition. [qualified 2026-09-28: see Evidence]

### C) Compound (additive) shows dominant-component behavior [qualified 2026-09-28: see Evidence]

`country_centroid + emotion_centroid` decoded as **China** (country-themed). The country centroid is larger in magnitude than the emotion centroid (after sink removal), so its direction dominates the sum. [qualified 2026-09-28: see Evidence] The emotion contribution is invisible. **Additive composition doesn't average the two concepts; it follows the larger one.** [qualified 2026-09-28: see Evidence]

### D) Pure subtraction of similar-magnitude vectors yields noise

`France − Germany` (||raw||=9.24) rescaled to ||h||=150 amplifies noise. Decoded as incoherent content: a Chinese movie reference ("Zhang Ziyi's Dream of Red Chamber"), "Je ne sais quoi" (a French phrase — faint French signal), then drift. The rescaling-of-noise problem.

Similarly `happy − sad` (||raw||=49.98) rescaled to 150 decoded as: "Fibonacci Zoo", "prize display" — also incoherent.

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). "Recomputed" values are read from `combos[]` (`label`, `category`, `h_raw`, `h_rescaled`, `av_text`) in [`../data/concept_arithmetic_atlas.pt`](../data/concept_arithmetic_atlas.pt) and from `captures[]` in [`../data/vocab_atlas.pt`](../data/vocab_atlas.pt).

- **Size and rescale (AUDIT 17):** `concept_arithmetic combo count` = 7; `target_norm` = 150.0. The seven `||h_rescaled|| ≈ 150` lines print actual values from 150.0 to 150.0001.
- **Decoded identities (AUDIT 17):** `combo[0] decodes containing 'London'`, `combo[1] decodes containing 'Spain'`, `combo[2] decodes containing 'United Kingdom'`, `combo[3] decodes containing 'Je ne sais quoi'`, `combo[4] decodes containing 'country'`, `combo[5] decodes containing 'Fibonacci'` and `combo[6] decodes containing 'China'` all PASS. Each is a case-insensitive substring check on one stored AV decode.
- **Combination types (recomputed, `combos[].category`):** 3 `analogy`, 1 `subtraction` (France − Germany), 2 `axis` (country_centroid − capital_centroid and happy − sad), 1 `compound`.
- **Decode wording (recomputed, `combos[].av_text`):** combo 0 "implies the response is about the city of London"; combo 1 '"What is Spain?" header ... expecting descriptive content about the country'; combo 2 'Final token "UK\n"'; combo 3 "The famous 'Je ne sais quoi' line from Zhang Ziyi's 'Dream of Red Chamber'"; combo 5 '"Fibonacci Zoo" greeting format' and "prize display convention"; combo 6 'informational format about "China,"'. Combos 0, 1, 2 and 6 also contain Chinese-language passages. Combo 4 opens with a bolded question header ("What is Christianity?") and contains "This country's population is diverse, and the following facts about the world's religions", "A country's facts..." and "a document".
- **Raw norms (recomputed, `||h_raw||`):** Paris − France + Germany 97.26; Tokyo − Japan + France 98.39; Berlin − Germany + United Kingdom 96.85; France − Germany 9.2423; country_centroid − capital_centroid 17.4553; happy − sad 49.9825; country_centroid + emotion_centroid 183.6251. The rescale to 150 therefore multiplies the analogies by about 1.5, France − Germany by 16.2 and happy − sad by 3.0, and shrinks the compound by 0.82.
- **Typical end-of-prompt norm (recomputed, `captures[].norm` in `vocab_atlas.pt`):** mean 100.40, range 92.06 to 113.79. The target norm of 150 is 1.49× the mean, above every capture in the atlas.
- **Compound term sizes (recomputed):** `centroid()` in `examples/nla_concept_arithmetic_atlas.py` averages raw `h`, without sink removal, and the stored `h_raw` of combo 6 equals the sum of the raw country and emotion centroids. Raw centroid norms: country 96.93, emotion 92.96 (ratio 1.04). With the 7 sink dims zeroed: country 78.05, emotion 74.37. The sum's cosine to the raw country centroid is 0.9684 and to the raw emotion centroid 0.9656. The sum's direction is almost equally close to both terms, so the decode following the country term is not explained by the country term dominating the direction.
- **Cross-protocol axis stability (AUDIT 16):** `max diagonal (emotion most-stable axis)` = 0.1704; `mean cross-protocol diagonal cosine (axis stability)` = 0.0784.
- **The t=0.421 flip in Cross-arc lessons (recomputed from `steps[].av_text` in [`../data/dense_interp_near_pivot.pt`](../data/dense_interp_near_pivot.pt)):** t=0.421 lies inside the "Definition + Poem" plateau. The 20-step grid placed the flip between t=0.421 and t=0.474, and the dense run places it between t=0.4450 and t=0.4475 (see the [dense interpolation](2026-05-15-nla-dense-interp-near-pivot.md) Evidence).

## What this tells us about NLA's representation geometry

NLA layer-20 representations are **categorically structured but not algebraically composable in the word2vec sense**:

1. **Category direction is a robust axis** (consistent with `country − capital` working and with [MAIN-70](2026-05-14-nla-mid-seq-native-discriminants.md)'s fig34 showing emotion has the strongest cross-protocol axis stability at +0.17). [qualified 2026-09-28: see Evidence]
2. **Within-category specific identity is encoded differently than across-category position.** The "Paris-ness within capital-ness" direction isn't the same as the "Berlin-ness within capital-ness" direction shifted in some consistent way. word2vec's geometric assumption — that `vec(France) − vec(Paris)` equals `vec(Germany) − vec(Berlin)` — does not hold in this layer.
3. **Sum of two concept vectors collapses to the larger-magnitude one.** The smaller concept contribution gets absorbed without changing the dominant decode. This is not the same as averaging the two concepts. [qualified 2026-09-28: see Evidence]

This refines the [MAIN-44](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) + MAIN-70 picture: the basis is protocol-coupled (per-protocol family of discriminants), and even within a protocol, the arithmetic structure of h is **categorical, not algebraic**. Categories have stable axes (confirmed); within-category positions are constructed in a way that doesn't compose by subtraction (falsified). [qualified 2026-09-28: see Evidence]

## Implications for viz primitives

A "concept arithmetic" UI surface — where the user combines tokens via + and − and watches the result decode — would produce mostly category-level results, not specific-identity transformations. The UI affordance is honest: show that arithmetic moves the h into a category direction (visible in discriminant glyph) but specific decode identity isn't algebraically predictable. Useful for exploring category axes; not useful as a "what would change if I rotate this analogy" probe.

A more productive use: arithmetic to **isolate axes** rather than predict tokens. `mean(country) − mean(capital)` gives a clean country axis. [qualified 2026-09-28: see Evidence] `h(happy) − mean(neutral_emotion)` would give a "happy-specific direction within emotion" — testable in a follow-up.

## Cross-arc lessons

The arc now has three converging findings:
- **MAIN-44/70**: basis is protocol-coupled; content categories have small position-invariant component.
- **MAIN-48 (this)**: within a protocol, arithmetic structure is categorical not algebraic.
- **[MAIN-25](2026-05-13-nla-interpolation-flipbook.md)** (the headline): stepwise category-flip at t=0.421 during linear interpolation between AR-encoded anchors. [qualified 2026-09-28: see Evidence]

Reading these together: at layer 20, h-space has **discrete category attractors** rather than a smooth product-of-axes geometry. Interpolation flips between attractors (MAIN-25); arithmetic moves between attractor regions but doesn't smoothly navigate within them (MAIN-48); discriminants pick out the attractor regions per protocol (MAIN-70). The next probe should test the attractor hypothesis directly: dense interpolation near t=0.421 ([MAIN-34](2026-05-15-nla-dense-interp-near-pivot.md)) to see if the flip is a sharp discontinuity or a smooth-but-fast transition. That's already filed.

## Reproducibility

```bash
# Build the 7 combinations from vocab_atlas.pt and AV-decode each (loads the AV on CPU;
# writes .cache/nla_artifacts/concept_arithmetic_atlas.pt)
python examples/nla_concept_arithmetic_atlas.py

# fig35 (no model load)
python examples/nla_concept_arithmetic_render.py

# Model-free check of the committed artifact (AUDIT 17)
python examples/nla_audit_findings.py
```

The committed copy of the output is [`../data/concept_arithmetic_atlas.pt`](../data/concept_arithmetic_atlas.pt).

## Hypotheses

### H1 — The analogies fail in the geometry of h, or in the decode

Two readings fit the three failed analogies:
- **Geometry.** In layer-20 end-of-prompt h, `Paris − France + Germany` does not lie near `Berlin`, so there is no analogy for the AV to read.
- **Decode.** The sum does lie near `Berlin`, but the AV does not read it at the input it was given: a vector rescaled to 1.49× the typical norm and built off the capture distribution.

**Test (no model load):** for each analogy, rank all 128 `vocab_atlas.pt` anchors by cosine to the sum `a − b + c`, excluding the three input words, as in the usual word-analogy evaluation. Use the sink-removed `h` (the 7 sink dims labelled in `pairwise_and_hotdims.pt` zeroed), so the component every capture shares does not set the ranking. The predicted targets (Berlin, Paris, London) are all atlas anchors. Nothing is fit, so no hold-out is needed.
- The geometry reading predicts that the target is not the nearest anchor.
- The decode reading predicts that it is.

What the outcomes decide:
- The target ranking first is consistent with the decode reading for that analogy and not with the geometry reading.
- The target ranking lower is consistent with the geometry reading. Both can fail at once, so it does not exclude a decode failure as well.
- A target in the top few but not first leaves H1 open for that analogy.

### H2 — The compound decode follows the larger term, or an AV prior toward places

Two readings fit `country_centroid + emotion_centroid` decoding as China:
- **Magnitude.** The decode follows whichever term is larger.
- **Place prior.** The AV tends to name a place for vectors in this region, whatever the term sizes. Every analogy and compound decode here names a place, and README limitation L2 records that AV format-bias is unaudited.

The committed run cannot separate them: the two terms differ in norm by 4%, and the sum is almost equally close to both (Evidence).

**Test:** AV-decode `α·country_centroid + emotion_centroid` for α = 0.5, 1 and 2, each rescaled to the same norm. Decoding is greedy, so a repeated decode of the same vector gives the same text. To get more than one sample per α, build the two centroids from 5 random halves of each category's anchors and decode each of the 5 sums at each α.
- The magnitude reading predicts mostly emotion-themed decodes at α = 0.5 (where the emotion term is about twice the country term) and mostly country-themed decodes at α = 2.
- The place-prior reading predicts mostly place-themed decodes at every α.

What the outcomes decide:
- Mostly place-themed decodes at α = 0.5 are consistent with the place-prior reading and not with the magnitude reading.
- Mostly emotion-themed decodes at α = 0.5 are consistent with the magnitude reading and not with a pure place prior.
- A split, or decodes that are neither, leave H2 open.

Five inputs per α can show which reading the decodes favour; they cannot exclude the other reading.

## Follow-ups

1. The H1 and H2 tests have not run. H1 needs no model.
2. The "happy-specific direction within emotion" from Implications (`h(happy) − mean(neutral_emotion)`) has no committed run.
3. Every decoded identity rests on one stored AV decode per combination, and AUDIT 17 checks only a substring. The "country-flavored" reading of combo 4 rests on the substring 'country' in a decode headed "What is Christianity?".
4. The target norm of 150 was not varied. Whether the decodes change at the typical norm of about 100 is untested.
5. The dense interpolation that Cross-arc lessons proposes has run: [dense interpolation near the pivot](2026-05-15-nla-dense-interp-near-pivot.md).
6. The H1 ranking test is tracked in [#123](https://github.com/skothr/llm-research/issues/123).

## References

- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md): the per-anchor h's and category centroids the combinations are built from.
- [Mid-seq native discriminants](2026-05-14-nla-mid-seq-native-discriminants.md): the cross-protocol axis stability cited in item 1 of the geometry section.
- [Interpolation flipbook](2026-05-13-nla-interpolation-flipbook.md) and [dense interpolation near the pivot](2026-05-15-nla-dense-interp-near-pivot.md): the interpolation results in Cross-arc lessons.
- [Arc README](../README.md) limitation L2: the unaudited AV format-bias.
- [Figure inventory](figures/INVENTORY.md), fig35.
