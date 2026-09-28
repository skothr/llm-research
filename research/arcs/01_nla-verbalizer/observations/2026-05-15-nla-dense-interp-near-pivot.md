# Dense Interpolation Reveals a Three-Region Geometry with One Sharp Boundary [qualified 2026-09-28: see Evidence]

**Date:** 2026-05-15
**Toolkit:** `nla_dense_interp_near_pivot.py`, `nla_dense_interp_render.py`
**Inputs:** `interpolation_flipbook.pt` (cached h_A, h_B), AV `kitft/nla-qwen2.5-7b-L20-av`
**Output:** `dense_interp_near_pivot.pt`
**Figures:** `fig36_dense_interp_flipbook.png`, `fig37_dense_interp_diagnostic.png`
**Private-tracker ID:** MAIN-34 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-34 tested

[MAIN-25](2026-05-13-nla-interpolation-flipbook.md) found a "stepwise transition at t=0.421" in linear h-space interpolation between AR-encoded anchors (factual/geography ↔ poetic/nature). The 20-step grid showed AV-text format flipping between step 8 (t=0.421) and step 9 (t=0.474) — Δt=0.053. The question for MAIN-34: at ~10× density (Δt ≈ 0.0025 in the critical zone [0.395, 0.455]) [qualified 2026-09-28: see Evidence], does the transition remain a single-step discontinuity (consistent with the synthesis from [MAIN-44](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) + [MAIN-48](2026-05-14-nla-concept-arithmetic-atlas.md): discrete category attractors), or does it smooth out (would mean the 20-step finding was undersampled)?

## What actually happened — three regions, not two

The dense sampling (30 steps, 25 dense in [0.395, 0.455] + 5 sparse context) reveals a **richer geometry than expected**. Three semantically-distinct regions, two transitions [qualified 2026-09-28: see Evidence]:

| t-range | AV-text region | duration |
|---|---|---|
| t ∈ [0, 0.25] | Factual/geography ("What is the capital of France?") | sparse-sampled |
| t ∈ [0.395, 0.4450] | **Hybrid "Definition + Poem" plateau** | **19 dense-zone steps, stable** [qualified 2026-09-28: see Evidence] |
| t ∈ [0.4475, 1.0] | Poetic/nature ("What is Spring?", autumn imagery, seasonal poem format) [qualified 2026-09-28: see Evidence] | dense + sparse |

The 19 consecutive dense-zone steps in the hybrid plateau all decode as **"Structured format with 'Definition' and 'Poem' labels suggests a concise answer format about a place name, likely a poet[ic phrase]"** — a stable intermediate state combining both the "definition" formal feature of factual h_A and the "poetic" feature of nature h_B, without committing to one or the other. [qualified 2026-09-28: see Evidence]

At t=0.4475 the AV-text shifts to **"Structured format with poetic description pattern ('What is London?')"** — still using a place-name anchor but committing to the poetic format. At t=0.4500 it's "What is Spring?" — the anchor word has fully flipped to nature.

## What was the "t=0.421 flip" actually?

The original 20-step grid sampled t=0.421 and t=0.474 directly. In the dense sampling:
- t=0.421 sits squarely **inside the hybrid plateau** (still "Definition + Poem")
- t=0.474 is **just past the sharp transition** at t≈0.4475-0.4500 [qualified 2026-09-28: see Evidence]
- The Δt=0.053 step in the 20-step grid skipped right over the plateau-to-poetic transition

The original "stepwise flip" framing was correct in detecting *that* the transition is discontinuous, but missed *what the transition is between* — it's not factual ↔ poetic in one step; it's hybrid plateau ↔ poetic in one Δt=0.0025 step. The factual ↔ hybrid transition is somewhere in [0.25, 0.395] and undersampled here.

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`examples/nla_audit_findings.py`). "Recomputed" values are read from `steps[]` (`t`, `h_t`, `av_text`), `h_A`, `h_B` and `dense_zone` in [`../data/dense_interp_near_pivot.pt`](../data/dense_interp_near_pivot.pt), with steps sorted by `t` as AUDIT 18 sorts them.

- **Grid (AUDIT 18):** `dense_interp step count` = 30; `dense_zone bounds` = [0.395, 0.455]. Recomputed: the 30 values of `t` are 25 dense steps from 0.3950 to 0.4550 at Δt = 0.0025, plus 5 coarse points at 0, 0.25, 0.5, 0.75 and 1.0.
- **Density (recomputed):** the 20-step grid of [`../data/interpolation_flipbook.pt`](../data/interpolation_flipbook.pt) has spacing 1/19 = 0.0526. The dense zone is 0.0526 / 0.0025 = 21× denser, not 10×.
- **Plateau length (recomputed, `steps[].av_text` first line):** all 21 steps from t=0.3950 to t=0.4450 name "Definition" and "Poem" labels, not 19. At t=0.4425 and t=0.4450 the order reverses to "Poem" and "Definition". The rest of the first line varies between three wordings: "concise answer format", "short answer or trivia format" and "short phrase or answer". 15 of the 21 steps begin with the quoted sentence; 3 read "short answer or trivia format", 1 reads "short phrase or answer", and the 2 reversed-order steps read "concise answer format".
- **Flip location (recomputed, same field):** t=0.4450 still names the "Poem" and "Definition" labels. t=0.4475 reads "Structured format with poetic description pattern ("What is 'London'?" question followed by a noun phrase) ...". From t=0.4500 to t=0.4550 it reads "What is 'Spring'?". The plateau-to-poetic flip is between t=0.4450 and t=0.4475. The step from 0.4475 to 0.4500 changes only 'London' to 'Spring'.
- **Plateau check (AUDIT 18):** `plateau decodes stable across t∈[0.395, 0.4400] (<=3 unique first-lines)` = 3. This is a bound over t ≤ 0.4400 on the first 100 characters of the first line. Recomputed on the same 100 characters: 3 unique first lines over t ≤ 0.4400, and 4 over t ≤ 0.4450, because the two reversed-order steps add a fourth. The bound of 3 does not extend to t=0.4450; the 21-step plateau rests on the "Definition" and "Poem" labels, not on the count.
- **Norms (AUDIT 18 and recomputed):** `midpoint norm < anchor norm (anti-parallel anchors)` = 60.7863883972168, at t=0.4200. Recomputed `||h_t||` falls from 60.87 at t=0.3950 to 60.71 at t=0.4550 and reaches its minimum over the 30 steps, 60.69, at t=0.5000, a coarse point the table assigns to the poetic region (its decode still names a "Definition" label; see Coarse points). `||h_A||` = 65.73 and `||h_B||` = 66.30. The anchors are not anti-parallel, as the audit label says: cos(`h_A`, `h_B`) = 0.6905. On the straight line `(1 − t)·h_A + t·h_B` the norm is smallest at t = 0.486 (60.69), so the dip is a property of the straight line between these two anchors, independent of the decodes, and not specific to the plateau.
- **Per-step ||Δh|| (recomputed):** every step inside the dense zone, including the flip step from t=0.4450 to t=0.4475, has `||h_{t+1} − h_t||` = 0.130, which is `||h_A − h_B||` × 0.0025 (51.945 × 0.0025). A small per-step ||Δh|| follows from the small Δt and does not mark the plateau.
- **Coarse points (recomputed, first line of `av_text`):** t=0 reads 'Structured format with "What is the capital of France?" question pattern'; t=0.25 reads '"What is" pattern and "Answer" label suggests a trivia or definition format with a named place'; t=0.5 reads 'Structured poem format with "Definition" and "Who is" pattern', a poem format that still names a "Definition" label, inside the range the table assigns to the poetic region; t=0.75 and t=1.0 read 'Structured poem format with numbered lines'.
- **Cached anchors (recomputed):** `h_A` and `h_B` equal those in [`../data/interpolation_flipbook.pt`](../data/interpolation_flipbook.pt), so this run reuses the flipbook's AR encodings as stated.
- **Not audited:** the timing figures in Methodological notes (~5 minutes of AR loading, ~80 s per decode, ~40 min, 2.4 hr at 100 steps) have no committed source.

## Implications for the discrete-attractor hypothesis (refines MAIN-44/48 synthesis)

The MAIN-44/48 synthesis was: "layer-20 has discrete category attractors rather than smooth product-of-axes geometry." MAIN-34 partially confirms this — the plateau-to-poetic transition IS sharp at 10× resolution (one Δt=0.0025 step). [qualified 2026-09-28: see Evidence and Hypotheses H1] But it also enriches the picture:

1. **There are more attractor regions than the corpus contained.** The "Definition + Poem" hybrid isn't one of the 23 vocab atlas categories. It's a stable combination state that emerges from the linear interpolation between the two AR-encoded anchors. The discrete-attractor view should think of these as **basins** with **stable intermediate plateaus** between them, not isolated points. [qualified 2026-09-28: see Evidence and Hypotheses H1]
2. **Transitions ARE sharp.** When the geometry moves between basins, it does so in a single Δt step. [qualified 2026-09-28: see Evidence and Hypotheses H1] The plateau (where ||Δh|| per step is small and AV decoding is invariant to t) [qualified 2026-09-28: see Evidence] is the basin; the boundary crossing is sharp.
3. **||h_t|| dips during the plateau.** Norms drop from ~66 (anchor magnitudes) to ~60 across the dense zone — consistent with anchors pointing somewhat apart geometrically, and the midpoint having reduced magnitude. The reduced-magnitude plateau is also where the AV's "stable intermediate" decode lives. [qualified 2026-09-28: see Evidence]

This refines the synthesis: **layer-20 h-space has discrete attractor basins separated by sharp boundaries. Linear interpolation traverses basins; AV decodes the basin you're in, not the geometric mean of the endpoints.** [qualified 2026-09-28: see Evidence and Hypotheses H1] The "Definition + Poem" basin between factual and poetic is one we hadn't named before.

## Methodological notes

- Reusing cached `h_A`, `h_B` saved ~5 minutes of AR loading. AR loading state stays valid across sessions.
- 30 steps × ~80s/decode = ~40 min CPU bf16 (down from the ticket's quoted 2.4 hr at 100 steps).
- Even at 10× the original grid resolution, the sharp transition fits within one sample step. [qualified 2026-09-28: see Evidence] Higher resolution could pin the boundary location more precisely but won't change the qualitative finding.

## Reproducibility

```bash
# AV decode of the 30 interpolation steps (loads the AV on CPU; reads h_A, h_B
# from interpolation_flipbook.pt; writes .cache/nla_artifacts/dense_interp_near_pivot.pt)
python examples/nla_dense_interp_near_pivot.py

# fig36 + fig37 (no model load)
python examples/nla_dense_interp_render.py

# Model-free check of the committed artifact (AUDIT 18)
python examples/nla_audit_findings.py
```

The decode script writes to the gitignored working cache `.cache/nla_artifacts/dense_interp_near_pivot.pt`; the committed copy is [`../data/dense_interp_near_pivot.pt`](../data/dense_interp_near_pivot.pt). Every read, including the decode script's read of `interpolation_flipbook.pt`, the render step's and the audit's, takes the cache copy when one exists and otherwise the committed copy in `../data/` (`examples/_nla_artifacts.py`). A clean clone therefore renders and audits the committed data with no copy step, and after a re-run they read the new cache file.

## Hypotheses

### H1 — The plateau is discrete in the base model's computation, or only in the AV's output

The plateau and the flip are seen only through AV text. Two readings fit that:
- **Model-side.** The base model's own downstream computation changes abruptly between t=0.4450 and t=0.4475.
- **Decoder-side.** The base model's computation changes smoothly along the line, and the AV maps that smooth change to a few discrete text templates. README limitation L2 (AV format-bias) is one mechanism for this.

A cosine readout against fixed vectors cannot separate them, because it varies smoothly along a straight line by construction.

**Test:** patch each dense `h_t` into the base model at layer 20, at the last position of a fixed neutral prompt. Record the next-token distribution. Compute the Jensen-Shannon divergence between each pair of consecutive dense steps, 24 values. Every step moves `h` by the same ||Δh|| (0.130, Evidence), so step size does not vary along the zone. The statistic is the rank of the 0.4450→0.4475 divergence among the 24.
- The model-side reading predicts that the flip step has the largest divergence.
- The decoder-side reading predicts no special rank for the flip step. If the 24 divergences were exchangeable, the flip step would be the largest with probability 1/24; a smooth trend along the zone makes the two ends more likely to hold the maximum than the flip step.

What the outcomes decide:
- The flip step is the largest: consistent with the model-side reading at this prompt.
- The flip step is not the largest: this readout shows no abrupt change at the flip, which is consistent with the decoder-side reading. It does not rule out a model-side boundary that another prompt or layer would show.

One neutral prompt is one readout. Repeating the test on several prompts, and reporting how many put the flip step first, is needed before either outcome is general.

## Follow-ups this opens

- **[MAIN-71](2026-05-15-nla-plateau-attractor-strength.md)**: dense sample in t ∈ [0.25, 0.40] to find the factual → hybrid transition. Same approach, ~30 more AV decodes (~40 min).
- **Test plateau attractor strength.** Does AR-re-encoding an h from the hybrid plateau collapse it back to the plateau (a self-attracting state), or does it drift toward one of the anchor regions? Tests whether the plateau is a real attractor or just a transit zone.
- **Probe other anchor pairs.** Different anchor-text pairs would produce different intermediate plateaus. Atlas of these would map the basin structure of layer 20 across content types.
- **Open items from the Evidence (2026-09-28):**
  - The factual → hybrid boundary in [0.25, 0.395] has no committed dense run. The first bullet's link points to the plateau attractor test, which is a different experiment.
  - The plateau attractor test ran: [plateau attractor strength](2026-05-15-nla-plateau-attractor-strength.md).
  - The H1 test has not run.
  - AUDIT 18 checks the plateau only over t ≤ 0.4400 and does not check the flip location. The 21-step plateau and the 0.4450→0.4475 flip rest on the recomputation in Evidence.
  - The timing figures in Methodological notes are unaudited.

## References

- [Interpolation flipbook](2026-05-13-nla-interpolation-flipbook.md): the 20-step run whose t=0.421 pivot this run re-samples, and the source of the cached `h_A`, `h_B`.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md): the cosine-to-anchor readout (fig21) that fig37's top panel extends.
- [Plateau attractor strength](2026-05-15-nla-plateau-attractor-strength.md): the AR round-trip test on the plateau at t=0.420.
- [Mid-seq vocab atlas](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) and [concept arithmetic atlas](2026-05-14-nla-concept-arithmetic-atlas.md): the discrete-attractor synthesis this run refines.
- [Figure inventory](figures/INVENTORY.md), fig36 and fig37.
