# Discriminant Validation: Connectivity, Self-Validation, and Stability

**Date:** 2026-05-13
**Toolkit:** `nla_discriminant_connectivity.py`, `nla_discriminant_stability_capture.py`, `nla_discriminant_stability_render.py`
**Figures:** `fig27_discriminant_connectivity.png`, `fig28_discriminant_stability.png`, `fig29_self_validation.png`
**Data:** `discriminant_stability.pt`

## Motivation

After fixing the "all axes active" issue with discriminant directions (fig25/fig26 successors to fig23/fig24), three validations were needed to confirm the basis is **coherent**, **stable**, and actually **discriminating** between concepts:

> **Terminology (per L6 in the arc README).** "Discriminant" here is shorthand for a per-category **mean-contrast** direction `d_C = unit(mean(h ∈ C) − mean(h ∉ C))` — *not* a Fisher LDA discriminant (which needs `S_W⁻¹(μ₁−μ₀)`; omitted because n=2–12 captures per category in 3584-dim makes `S_W` rank-deficient by orders of magnitude). Read "mean-contrast direction" wherever this file says "discriminant"; the name does not imply Fisher-style optimal separation.

1. Are the 23 axes themselves coherent — do semantic siblings produce correlated axes; do opposites produce anti-correlated ones? → **fig27 connectivity**
2. Do existing captures with known content project onto the expected discriminant? → **fig29 self-validation**
3. Does the same anchor word in different contexts produce consistent discriminant readouts? → **fig28 stability**

## Result summary

**Discriminate:** PASS (mean cross-axis cosine +0.006).
**Coherent:** PASS (clear macro-cluster structure in fig27).
**Stable:** MIXED — varies dramatically by category.
**Token-presence detection:** FAIL — the discriminants detect **prompt-topic**, not **token-presence-in-prompt**.

## Finding 1 — Discriminant connectivity reveals three macro-clusters (fig27)

The 23×23 discriminant pairwise cosine matrix has clear block structure:

* **CONTENT macro-cluster** (top-left 6 categories): country, capital, nature, codemath, emotion, refusal. Mutually positive within; anti-correlated with everything below.
* **FUNCTION-WORD macro-cluster** (middle 9): article, pronoun, demonstrative, preposition, conjunction, auxiliary, negation, quantifier, wh_word. Own correlated block.
* **STRUCTURAL macro-cluster** (bottom-right 8): p_ender, p_internal, p_quote, p_bracket, p_dash, p_special, number, math_op. **Tightest block of all** — non-content tokens are represented as one super-category at h[20].

Top semantic siblings (highest discriminant cosine):
- `country ↔ capital`: **+0.938**
- `p_special ↔ math_op`: +0.843
- `preposition ↔ auxiliary`: +0.793

Top semantic opposites (most negative):
- `country ↔ demonstrative`: -0.638
- `nature ↔ auxiliary`: -0.605
- `emotion ↔ preposition`: -0.597

The PC1 of fig19 (content-vs-function axis) is reproduced as a pairwise structure here. Content categories anti-correlate with function categories; function correlates with structural less directly. **The model's mid-late residual is hierarchically organized — macro-categories with sub-categories — and this organization is recoverable from the data.**

## Finding 2 — Self-validation: top-5 hit rates 56-79% (fig29)

Projecting all 167 existing captures onto each discriminant, then asking "did the expected category top the projection?":

| expected | n | top-1 | top-3 | top-5 | mean rank [qualified 2026-09-28: see Evidence] |
|---|---|---|---|---|---|
| country (country pool) | 29 | 34% | 76% | **79%** | 3.6 |
| codemath (aggregate/code,math) | 30 | 53% | 63% | 73% | 3.5 |
| nature (haiku, creative) | 45 | 47% | 49% | 56% | 5.2 |
| refusal (forced/refuse) | 1 | 0% | 100% | 100% | 1.0 |

The 79% top-5 + mean-rank 3.6 [qualified 2026-09-28: see Evidence] for country means: country captures **always end up in the content macro-cluster** but country and capital constantly swap top-1 (they're +0.938 siblings — h's fitting one fit the other). The macro-classification is reliable; sibling distinction is harder.

Nature is weakest because haiku tokens span many adjacent categories (nature, emotion, codemath as a single content cluster).

## Finding 3 — Stability varies dramatically by category (fig28)

8 anchors × 4 prefix-length variants × capture at position -1 of the chat-templated sequence. *Note (added 2026-05-29): position -1 is the trailing token of the `<|im_start|>assistant\n` generation prefix, NOT the anchor token itself — what we measure is the model's "what to say next given this prompt" representation. Comparisons across contexts are internally consistent (all four contexts capture at this same kind of position), so the stability finding holds, but the framing should be "end-of-prompt stability across prefix lengths" rather than "anchor-token stability".*:

| anchor | category | ctx-cos (4 contexts) | expected-projection mean | std |
|---|---|---|---|---|
| `the` | article | **+0.917** | +0.183 | ±0.139 |
| `France` | country | +0.851 | +0.219 | ±0.129 |
| `Paris` | capital | +0.815 | +0.185 | ±0.135 |
| `7` | number | +0.815 | +0.281 | ±0.130 |
| `function` | codemath | +0.801 | **+0.284** | ±0.086 |
| `.` | p_ender | +0.799 | +0.119 | ±0.153 (std > mean) |
| `refuse` | refusal | +0.589 | +0.052 | ±0.130 (std ≫ mean) |
| `happy` | emotion | **+0.399** | +0.083 | ±0.061 |

**Two stability classes** (re-framed 2026-05-29 under the corrected "end-of-prompt response-planning" semantics — see note at the top of this finding):
- **Stable anchors** (ctx-cos > +0.80): function-words, structural tokens, place names, numbers. After these prompts, the model's "what to say next" representation is similar regardless of prefix length — whether the user said "France" alone, "Tell me about France," or "I want to discuss with you the following word, which is France," the model's response-planning state is dominated by the final content token + assistant-turn opener. The prefix prose is integrated but doesn't deflect the plan.
- **Unstable anchors** (ctx-cos < +0.60): `happy` and `refuse`. After these prompts, the model's response-planning state shifts substantially with prefix length. Plausibly: factual/structural anchors have a narrow "what one says about this" distribution (definitional / identifier-like), while emotional and refusal-laden anchors have a wider distribution that the surrounding prose actually narrows — the model has to figure out whether you want a definition, an empathetic response, a meta-discussion, or something else.

## The major finding — the discriminants detect prompt-topic, NOT token-presence

Look at the expected-projection column: even the BEST case (`function` → codemath) is only +0.28. For `happy` → emotion it's +0.083. For `refuse` → refusal it's +0.052 with **std exceeding the mean** (sometimes the projection is *negative*).

Watching the glyphs in fig28: `happy`-as-single-token-message projects onto emotion (top-1), but `happy` inside "Tell me about happy" or "I want to discuss the word happy" projects onto **codemath** [qualified 2026-09-28: see Evidence]. The model isn't representing "the token 'happy' is here" — it's representing "this is a question asking about a word" (which feels factual/definitional, hence codemath).

**The discriminants are doing prompt-TOPIC classification, not token-presence detection.** When the entire user message is about country geography, the h projects onto country. When the entire user message asks about the word "refuse" in a meta way, the h projects onto wh_word or demonstrative [qualified 2026-09-28: see Evidence] — *because the message is asking about a word*, not refusing anything.

This re-frames what our 23-axis basis actually represents:
- **Good for**: classifying the overall topic/register of a complete prompt
- **Bad for**: detecting whether a specific token appears in the prompt
- **Reason**: layer 20 at end-of-prompt has integrated the entire user message into a "what is this prompt about?" representation

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`research/arcs/01_nla-verbalizer/scripts/nla_audit_findings.py`). "Recomputed" values use that script's own recipe (sink dims zeroed, `d_C = unit(mean(h ∈ C) − mean(h ∉ C))` over [`../data/vocab_atlas.pt`](../data/vocab_atlas.pt)) and were re-read for this section.

- **Discriminate (AUDIT 13):** `mean cross-discriminant cosine (after fix)` = 0.0064; `min cross-discriminant cosine (country↔demonstrative opposite)` = −0.6381; `country-capital discriminant cosine` = 0.938.
- **Other Finding 1 pairs (recomputed):** p_special↔math_op +0.843; preposition↔auxiliary +0.793; nature↔auxiliary −0.605; emotion↔preposition −0.597.
- **Self-validation (AUDIT 22):** country n=29, top-5 0.7931; codemath n=30, top-5 0.7333; nature n=45, top-5 0.5556. AUDIT 23 gives `country baseline top-1 accuracy` = 0.3448.
- **Rest of the Finding 2 table (recomputed with the AUDIT 22 projection):** the 167 captures are the pool AUDIT 22 builds from the four raw files [`../data/aggregate_faithfulness.pt`](../data/aggregate_faithfulness.pt), [`../data/rabbit_haiku_gen_trajectory.pt`](../data/rabbit_haiku_gen_trajectory.pt), [`../data/forced_continuation.pt`](../data/forced_continuation.pt) and [`../data/country_concept_vector.pt`](../data/country_concept_vector.pt). Top-1: codemath 0.533, nature 0.467, refusal 0.0. Top-3: country 0.759, codemath 0.633, nature 0.489, refusal 1.0. The table's mean-rank column is 0-indexed (rank 0 = top-1): country 3.62, codemath 3.53, nature 5.20, refusal 1.00.
- **Stability (AUDIT 14):** `stability capture count` = 32; ctx-cos `the` 0.9172, `France` 0.8515, `refuse` 0.5892, `happy` 0.3987; `happy → emotion projection mean` = 0.0825.
- **Rest of the Finding 3 table (recomputed from [`../data/discriminant_stability.pt`](../data/discriminant_stability.pt) with the AUDIT 14 recipe):** ctx-cos `Paris` 0.815, `7` 0.815, `function` 0.801, `.` 0.799. Every expected-projection mean and ± value in the table matches (± is the population standard deviation over the 4 contexts).
- **Top-1 category per context (recomputed, same data):** `happy` gives emotion (single), nature (short) and codemath (medium, "Tell me about happy"). In the long context nature (+0.1178) and codemath (+0.1175) tie within 0.0003, so the text's codemath there is a tie, not a clear top-1. `refuse` gives negation (single) and codemath in the other three contexts, with wh_word second only in the medium context and demonstrative in the top 2 of none. The text's "wh_word or demonstrative" for `refuse` does not match the recomputed projections.

## Implication for the visualization research

This is a **scope-clarifying finding**. The discriminant glyphs we built (fig25, fig26, fig28) are valid as a representation of **what the model thinks the prompt is about**, not as a representation of **what tokens the prompt contains**. The interpolation flipbook's cascade (fig21: France → autumn → snow) showed the same thing: as t varies, the model's "this prompt is about" topic shifts, not the model's "tokens in the prompt" list.

For future work:
1. **Build separate per-token discriminants** using mid-sequence captures (h[20] at specific positions other than end-of-prompt) if we want token-presence detection
2. **Treat the existing 23-discriminant basis as a prompt-topic classifier** — that's its actual function
3. **The category-attractor structure (intra-cos +0.85 within categories) is what makes the basis work for topic classification** — every prompt about a country pulls h toward the country attractor regardless of which specific country, hence good top-K detection but weak top-1

## Reproducibility

```bash

# fig27 + fig29 (cheap, no model loading, ~30 sec)
python research/arcs/01_nla-verbalizer/scripts/nla_discriminant_connectivity.py

# fig28 — capture (~3 min CPU forward passes) + render (~10 sec)
python research/arcs/01_nla-verbalizer/scripts/nla_discriminant_stability_capture.py
python research/arcs/01_nla-verbalizer/scripts/nla_discriminant_stability_render.py
```

## Hypotheses

### H1 — The weak token projection comes from end-of-prompt integration, or from protocol coupling

The file's stated reason is that end-of-prompt h[20] has integrated the whole message into a topic representation. A competing reading is protocol coupling: the 23 directions were fit on end-of-prompt captures whose user message is the anchor alone, while three of the four stability contexts wrap the anchor in prose. The weak projection may come from that mismatch between fitting and test protocols.

**Test:** keep the end-of-prompt capture position and vary only the fitting protocol. Fit the 23 directions twice: once on the existing single-token captures, and once on end-of-prompt captures with each vocab-atlas anchor embedded in the four stability-scan templates. All 8 stability anchors are vocab-atlas anchors, so each tested anchor is held out of both fitting sets (leave-one-anchor-out). Project the held-out anchor's four stability captures onto each set of directions and compare the expected-category projection.

What the outcomes decide:
- No difference between the two fitting sets rules out protocol coupling as the cause of the weak projection.
- A rise under the templated fitting set is what protocol coupling predicts. It does not rule out integration: templated directions could pick up the anchor's category through the integrated end-of-prompt representation. A rise therefore leaves H1 open.

### H2 — The stable/unstable split is a property of the category, or of the single anchor tested

Each category is represented by one anchor, so the two stability classes rest on n=1 per category. **Test:** repeat the 4-context scan for every vocab-atlas anchor in emotion (6) and refusal (4) and in two stable categories. If ctx-cos stays below +0.60 across the emotion and refusal anchors and above +0.80 for the stable categories, the split is category-level. If ctx-cos varies as much within a category as between categories, the split is anchor-level and the response-distribution explanation needs a per-anchor test. Any other pattern leaves H2 open.

## Follow-ups

1. The H1 test is not yet run. The 2026-05-14 mid-sequence runs follow Future-work item 1 above. They test whether the token is legible at its own position, which bears on the integration premise but keeps a different capture position, so they are not the H1 test.
   - [Mid-sequence vocab atlas null result](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md): mid-sequence captures projected onto the existing end-of-prompt directions give a mean within-class signal of +0.0491 and 32.04% argmax accuracy, against +0.4022 and 75.37% in protocol.
   - [Mid-seq native discriminants](2026-05-14-nla-mid-seq-native-discriminants.md): directions fit on the mid-sequence captures themselves give +0.5632 in-protocol signal and 97.10% argmax accuracy, so the token is legible at its own position under a matched protocol.
2. H2 has no committed run.

## References

- [Discriminant glyph](2026-05-13-nla-vocab-atlas-grid.md) — the primitive being validated here.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) — same data the centroids and discriminants are computed from.
- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — sink removal applied throughout.
- Kim et al., 2018, "TCAV" — discriminant directions generalize the binary CAV to multi-class.
