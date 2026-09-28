# Path B — Interpolation Flipbook: AV Reveals Stepwise Semantic Transitions

**Date:** 2026-05-13
**Toolkit:** `nla_interpolation_flipbook.py` + `nla_render_interpolation_flipbook.py`
**Models:** AV (kitft/nla-qwen2.5-7b-L20-av, CPU bf16) + AR (kitft/nla-qwen2.5-7b-L20-ar, CPU bf16)
**Compute:** ~30 minutes wall time (AR load 215s + 2 anchors + AV load + 20 × ~90s AV)
**Figures:** `fig17_interp_flipbook.png`, `fig18_interp_diagnostic.png`
**Data:** `.cache/nla_artifacts/interpolation_flipbook.pt` [qualified 2026-09-28: see Evidence]

> **Refinement note (added 2026-05-29):** the "stepwise transition at t=0.421" finding here is *correct that the transition is discontinuous*, but the 20-step grid undersampled what the transition is *between*. See `2026-05-15-nla-dense-interp-near-pivot.md` (MAIN-34): a 10×-denser run revealed an intermediate "Definition + Poem" hybrid plateau spanning t∈[0.395, 0.4450]. [qualified 2026-09-28: see Evidence] The actual sharp flip is from that plateau to the poetic basin at t≈0.4475-0.4500 (one Δt=0.0025 step) [qualified 2026-09-28: see Evidence]; t=0.421 itself sits *inside* the plateau. The factual→hybrid boundary is somewhere in [0.25, 0.395] and still undersampled.

## Goal

Use AR to encode two natural-language anchors into h-vectors, linearly interpolate at 20 steps, AV-decode each interpolated h, and observe the semantic transition in natural language. The novel scientific question: does linear interpolation in h-space produce smooth semantic morph, or stepwise transitions through neighboring concept regions?

## Anchors

* **A** (`factual/geography`): "Structured factual format with subject 'France' and predicate 'capital'. Final token 'Paris' completes a geography question."
* **B** (`poetic/nature`): "Structured poetic format with imagery of butterflies in spring. Final token 'flutter' completes a haiku-style line."

Phrased to mimic AV's own 3-paragraph template language (format-class / phrase context / final-token prediction), keeping them in-distribution for the AR encoder.

## Geometric findings (fig18)

* `||h_A|| = 65.73`, `||h_B|| = 66.30`. AR-encoded vectors land at moderate norms, similar to real model captures. (Earlier draft of this observation misread fig18's `||h_A − h_B|| = 51.94` legend as `||h_A||`; the audit caught it.)
* `||h_A − h_B|| = 51.94` — the inter-anchor distance in raw h-space.
* `cos(h_A, h_B) = +0.6905`. **The anchors are far more similar than I'd designed for.** Two semantically-divergent texts collapse to vectors that are 69% cosine — the chat-template structural attractor dominates AR's outputs.
* `||h_t||` dips to ~60.7 at t=0.5 — an ~8% midpoint depression from `max(||h_A||, ||h_B||) = 66.30`. The geometric path bows toward the origin, consistent with anchors that aren't collinear but aren't orthogonal either.
* `cos(h_t, h_A)` and `cos(h_t, h_B)` curves cross at exactly t=0.5 (linear-interp sanity check passed). [qualified 2026-09-28: see Evidence]
* Per-step distance is constant at 2.734 up to floating-point precision (sanity check).

## Semantic findings (fig17)

The novel result: **AV-decoded text changes stepwise, not smoothly**, despite mathematically smooth h-space interpolation.

| t | AV text first paragraph (summary) |
|---|---|
| 0.000 | factual format / "capital of France" / Paris |
| 0.053 | factual / "capital of France is Paris" |
| 0.105 | factual / "capital of France" |
| 0.158 | factual / **lateral shift to "England, London"** |
| 0.211 | factual / "What is" pattern |
| 0.263 | factual / **intrusion: "its element is 'water'"** |
| 0.316 | "Definition and Answer pattern" / **"Peace"** appearing |
| 0.368 | "Definition and Answer" / **"London is a beautiful city"** (poetic register intrudes) |
| 0.421 | **PIVOT: "Definition and Poem" labels — both present** |
| 0.474 | "What is 'Spring'?" / **"A poem about spring"** |
| 0.526 | "Structured poem format" / **fully poetic** |
| 0.579 | "poem format" / "Autumn" / "A leaf falls" / "peaceful mountain" |
| 0.632 | "poem format" / "Winter, snow, white" |
| 0.684 - 1.000 | poem format / **nature imagery locked in** (autumn leaves, snow, flowers, seasons) |

### The pivot is concentrated in 1-2 steps

The format class flips from "factual" to "poem" between t=0.368 (Definition+Answer with poetic intrusions) and t=0.474 (fully poetic). The AV's category-word for the activation changes within a single 5%-of-the-way interpolation step. **This is the most striking finding** — linear h-space movement of ~2.7 per step produces no visible AV-text change for many steps, then a sudden semantic re-categorization, then locked-in stability.

### Geographic concepts cascade laterally

As France-content fades, it's replaced by England → Canada → "Spring" as a label → "Autumn"/"Winter" themes. **Linear interpolation passes through neighboring semantic neighborhoods**, not through "France-butterfly hybrids." The AR has structured h-space such that the path between factual-geography and poetic-nature visits other related concepts as way-points.

### Anchor B's specified subject never appears

Anchor B asked for "butterflies in spring." The AV at t=1.0 reads as "Autumn leaves" / "Winter snow" / "flower" — the AR's encoding of the specific subject doesn't survive the round-trip. **Only the style/format class survives**. AR may be lossy on specific content while preserving global format.

### AV produces coherent text for synthetic h's

At intermediate t-values, the AV invents grammatical combinations the model would never produce: "the capital of France is Paris, and its country is France" (recursive); "its element is 'water'"; "What is the capital of England? London is a beautiful city." These aren't garbage — they're the AV's honest attempts to describe activation states that no real model would produce. **This validates a key use case for NLA**: AV can be used as an interpretation tool for *synthetic* h-vectors (arithmetic combinations, ablations, interpolations), not just real captures.

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). AV text is quoted from `steps[].av_text` in [`../data/interpolation_flipbook.pt`](../data/interpolation_flipbook.pt), the committed copy of the artifact. The header's `.cache/nla_artifacts/` path is the gitignored live cache, which a clean clone does not have.

- **Geometry (AUDIT 11):** `interpolation step count` = 20; `anchor cosine cos(h_A, h_B)` = 0.6905; `||h_A||` = 65.7323; `||h_B||` = 66.3012; `||h_A - h_B||` = 51.945; per-step `||Δh||` mean 2.7339, std 0.0; midpoint `||h_t||` = 60.73, below both anchor norms.
- **t=0.000 (step 0):** "Structured format with "What is the capital of France?" question pattern suggests a list or answer format with ..."
- **t=0.263 (step 5):** "The phrase "The capital of France is Paris, and its element is 'water'." implies a second question ..."
- **t=0.368 (step 7):** "Structured format with "Definition" and "Answer" pattern suggests a short poetic or factual entry about a place name "London," ..."
- **t=0.421 (step 8):** "Structured format with "Definition" and "Poem" labels suggests a concise answer format about a place name ..."
- **t=0.526 (step 10):** "Structured poem format with numbered lines and descriptive attributes ..."
- **t=1.000 (step 19):** "Structured poem format with numbered lines and seasonal imagery, suggesting a poetic or literary pattern with a closing line completing the verse about autumn leaves."
- **Anchor B's subject:** the substring "butterfl" occurs in no step's `av_text`.
- **Cosine crossing (recomputed from `h_A`, `h_B` and `steps[].h_t` in the same file):** cos(h_t, h_A) − cos(h_t, h_B) is +0.0163 at t=0.474 and −0.0192 at t=0.526. Linear interpolation between those two steps puts the crossing at t=0.4979. Solving the equality on the straight line between the committed anchors gives t=0.4978. The crossing falls below 0.5 because ||h_A|| = 65.7323 is smaller than ||h_B|| = 66.3012.
- **Audit coverage:** AUDIT 11 checks the geometry claims above, including the anchor cosine 0.6905. AUDIT 18 checks the dense re-sample. No AUDIT line checks the 20-step AV-text pivot at t=0.421.
- **Dense re-sample cited in the refinement note ([`../data/dense_interp_near_pivot.pt`](../data/dense_interp_near_pivot.pt)):** AUDIT 18 gives `dense_interp step count` = 30 and `dense_zone bounds` = [0.395, 0.455]. The 30 steps are 25 at Δt=0.0025 spanning t=0.3950 to t=0.4550, plus 5 coarse points at t=0, 0.25, 0.5, 0.75 and 1.0.
- **Dense grid density (recomputed):** the 20-step grid spacing is 1/19 = 0.0526, so the Δt=0.0025 dense zone is 21× denser, not 10×.
- **Dense plateau content (`steps[].av_text`, same file):** the first line names "Definition" and "Poem" labels at all 21 steps from t=0.3950 to t=0.4450. The label order reverses to "Poem" and "Definition" at t=0.4425 and t=0.4450. At t=0.4475 the first line changes to "Structured format with poetic description pattern ("What is 'London'?" question ...", and from t=0.4500 it reads "What is 'Spring'?". The plateau-to-poetic flip therefore falls between t=0.4450 and t=0.4475, one Δt step earlier than the note's "t≈0.4475-0.4500". The London-to-Spring change happens in the next step, t=0.4475 to t=0.4500.
- **AUDIT 18's plateau check** covers only t∈[0.395, 0.4400]: `plateau decodes stable across t∈[0.395, 0.4400] (<=3 unique first-lines)` = 3. The 3 distinct 100-character first lines all name the same "Definition" and "Poem" labels and differ only in the following wording ("short answer or trivia format", "concise answer format", "short phrase or answer"). The plateau is a single format label, and it extends to t=0.4450.

## Hypotheses

### H7 — Semantic neighborhoods are discrete in h-space, not continuous

The stepwise transition suggests h-space is partitioned into semantic regions where format-class is locally constant. Crossing a boundary flips the AV's category word; staying within a region keeps it constant. **Test:** repeat the interpolation between more orthogonal anchors (e.g., "code format" vs "poetic format"); measure how many steps the AV stays in each region. Predict: the transition is similarly concentrated in 1-2 steps, not smooth.

### H8 — AR is content-lossy but format-preserving

Anchor B specified "butterflies in spring" but the AV at t=1.0 reads as "Autumn leaves." Test by AR-encoding multiple different "poetic format about X" anchors and AV-decoding them — predict that the format-class survives but the specific X (butterflies, leaves, snow) is replaced by AR's preferred fillers ("flower", "leaf"). If true, the difference-of-means CAV for "spring vs autumn" might not be extractable via AR-anchored methods.

### H9 — The pivot t-value depends on anchor distance, not on content

For 0.69-cosine anchors, the pivot was at t≈0.43. For more orthogonal anchors the pivot should still be at some t in [0.4, 0.6], because the geometry is roughly symmetric. **Test:** run with cos(h_A, h_B) ≤ 0.2 and check whether the pivot is still near t=0.5.

### H10 — Anchor cosine is bounded below by ~0.5 for AR-encoded NL inputs

If the chat-template attractor is universal, AR-encoded inputs will all have a large shared component plus small content-specific perturbations. **Test:** AR-encode 10 maximally-different anchor texts; compute the pairwise cosine matrix; predict mean cosine ≥ 0.5 (not 0).

## Hidden implications for the visualization research

* **The flipbook view is genuinely novel.** Each row pairs a glyph (geometric signature) with AV text (semantic description), so the reader can watch BOTH the math and the meaning evolve. The stepwise semantic transition is invisible from numerical interpolation alone but obvious in the AV-text column.
* **||Δh||_feat is NOT sensitive to semantic boundary crossings.** Per-step distance is constant 2.7 across all 20 steps — the geometric step size doesn't increase at the t=0.43 pivot. The boundary is detectable only through AV-text or through a learned "category" classifier on h_t, not through h-vector geometry alone.
* **fig17 is the most compelling artifact in the arc so far** because it shows the novel use of NLA (AV-decoding synthetic h's) producing structured semantic transitions that no other interpretability tool can render this way.

## Followup paths

1. **Repeat with more orthogonal anchors.** Try code-vs-poetry, formal-vs-casual, factual-vs-emotional. Document how the pivot t-value shifts.
2. **Map the pivot regions.** For each interpolation, identify the exact t at which the AV's format-word changes. Build a "semantic-boundary map" across many anchor pairs.
3. **Test H7 with denser sampling near the pivot.** Run the same A→B interpolation at 100 steps clustered tightly around t∈[0.4, 0.5]. See if the transition is truly within 1 step or smoother at higher resolution.
4. **AR-decode the AV outputs at each step.** Recursive faithfulness check: AV→text→AR→h_pred; compare to h_t. Where in the trajectory does the round-trip degrade most?
5. **Build the same flipbook with a CAV anchor.** Use the country CAV direction as h_A or h_B (instead of an AR-encoded text); show how the AV traverses from a real-token capture to a synthetic-concept-direction.

## Reproducibility

```bash

# ~30 min: AR load + 2 anchors + AV load + 20 AV decodes
python examples/nla_interpolation_flipbook.py

# ~30s: render fig17 + fig18 from the artifact
python examples/nla_render_interpolation_flipbook.py
```

Resumable — interrupting and re-running picks up from the last saved step.

## References

- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — introduced the signature-glyph primitive used in fig17.
- [Cheap-batch glyph views](2026-05-13-nla-cheap-batch-three-glyph-views.md) — established that the AV can decode arithmetically-constructed vectors. fig17 generalizes that to linearly-interpolated vectors and shows the meaning gradient.
- [Geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) — identified the 8 feature dims used as glyph rays in fig17.
- [Audit pass](2026-05-13-nla-audit-pass.md) — `nla_audit_findings.py` will need extending to cover the interpolation artifact's load-bearing claims (anchor cosine 0.69, pivot at t=0.43, etc.) before treating these as durable findings. [qualified 2026-09-28: see Evidence]
