# Dense Interpolation Reveals a Three-Region Geometry with One Sharp Boundary

**Date:** 2026-05-15
**Toolkit:** `nla_dense_interp_near_pivot.py`, `nla_dense_interp_render.py`
**Inputs:** `interpolation_flipbook.pt` (cached h_A, h_B), AV `kitft/nla-qwen2.5-7b-L20-av`
**Output:** `dense_interp_near_pivot.pt`
**Figures:** `fig36_dense_interp_flipbook.png`, `fig37_dense_interp_diagnostic.png`
**Private-tracker ID:** MAIN-34 (retired tracker, not resolvable — see the [ID map](../README.md#private-tracker-id-map-main-n))

## What MAIN-34 tested

[MAIN-25](2026-05-13-nla-interpolation-flipbook.md) found a "stepwise transition at t=0.421" in linear h-space interpolation between AR-encoded anchors (factual/geography ↔ poetic/nature). The 20-step grid showed AV-text format flipping between step 8 (t=0.421) and step 9 (t=0.474) — Δt=0.053. The question for MAIN-34: at ~10× density (Δt ≈ 0.0025 in the critical zone [0.395, 0.455]), does the transition remain a single-step discontinuity (consistent with the synthesis from [MAIN-44](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) + [MAIN-48](2026-05-14-nla-concept-arithmetic-atlas.md): discrete category attractors), or does it smooth out (would mean the 20-step finding was undersampled)?

## What actually happened — three regions, not two

The dense sampling (30 steps, 25 dense in [0.395, 0.455] + 5 sparse context) reveals a **richer geometry than expected**. Three semantically-distinct regions, two transitions:

| t-range | AV-text region | duration |
|---|---|---|
| t ∈ [0, 0.25] | Factual/geography ("What is the capital of France?") | sparse-sampled |
| t ∈ [0.395, 0.4450] | **Hybrid "Definition + Poem" plateau** | **19 dense-zone steps, stable** |
| t ∈ [0.4475, 1.0] | Poetic/nature ("What is Spring?", autumn imagery, seasonal poem format) | dense + sparse |

The 19 consecutive dense-zone steps in the hybrid plateau all decode as **"Structured format with 'Definition' and 'Poem' labels suggests a concise answer format about a place name, likely a poet[ic phrase]"** — a stable intermediate state combining both the "definition" formal feature of factual h_A and the "poetic" feature of nature h_B, without committing to one or the other.

At t=0.4475 the AV-text shifts to **"Structured format with poetic description pattern ('What is London?')"** — still using a place-name anchor but committing to the poetic format. At t=0.4500 it's "What is Spring?" — the anchor word has fully flipped to nature.

## What was the "t=0.421 flip" actually?

The original 20-step grid sampled t=0.421 and t=0.474 directly. In the dense sampling:
- t=0.421 sits squarely **inside the hybrid plateau** (still "Definition + Poem")
- t=0.474 is **just past the sharp transition** at t≈0.4475-0.4500
- The Δt=0.053 step in the 20-step grid skipped right over the plateau-to-poetic transition

The original "stepwise flip" framing was correct in detecting *that* the transition is discontinuous, but missed *what the transition is between* — it's not factual ↔ poetic in one step; it's hybrid plateau ↔ poetic in one Δt=0.0025 step. The factual ↔ hybrid transition is somewhere in [0.25, 0.395] and undersampled here.

## Implications for the discrete-attractor hypothesis (refines MAIN-44/48 synthesis)

The MAIN-44/48 synthesis was: "layer-20 has discrete category attractors rather than smooth product-of-axes geometry." MAIN-34 partially confirms this — the plateau-to-poetic transition IS sharp at 10× resolution (one Δt=0.0025 step). But it also enriches the picture:

1. **There are more attractor regions than the corpus contained.** The "Definition + Poem" hybrid isn't one of the 23 vocab atlas categories. It's a stable combination state that emerges from the linear interpolation between the two AR-encoded anchors. The discrete-attractor view should think of these as **basins** with **stable intermediate plateaus** between them, not isolated points.
2. **Transitions ARE sharp.** When the geometry moves between basins, it does so in a single Δt step. The plateau (where ||Δh|| per step is small and AV decoding is invariant to t) is the basin; the boundary crossing is sharp.
3. **||h_t|| dips during the plateau.** Norms drop from ~66 (anchor magnitudes) to ~60 across the dense zone — consistent with anchors pointing somewhat apart geometrically, and the midpoint having reduced magnitude. The reduced-magnitude plateau is also where the AV's "stable intermediate" decode lives.

This refines the synthesis: **layer-20 h-space has discrete attractor basins separated by sharp boundaries. Linear interpolation traverses basins; AV decodes the basin you're in, not the geometric mean of the endpoints.** The "Definition + Poem" basin between factual and poetic is one we hadn't named before.

## Methodological notes

- Reusing cached `h_A`, `h_B` saved ~5 minutes of AR loading. AR loading state stays valid across sessions.
- 30 steps × ~80s/decode = ~40 min CPU bf16 (down from the ticket's quoted 2.4 hr at 100 steps).
- Even at 10× the original grid resolution, the sharp transition fits within one sample step. Higher resolution could pin the boundary location more precisely but won't change the qualitative finding.

## Follow-ups this opens

- **[MAIN-71](2026-05-15-nla-plateau-attractor-strength.md)**: dense sample in t ∈ [0.25, 0.40] to find the factual → hybrid transition. Same approach, ~30 more AV decodes (~40 min).
- **Test plateau attractor strength.** Does AR-re-encoding an h from the hybrid plateau collapse it back to the plateau (a self-attracting state), or does it drift toward one of the anchor regions? Tests whether the plateau is a real attractor or just a transit zone.
- **Probe other anchor pairs.** Different anchor-text pairs would produce different intermediate plateaus. Atlas of these would map the basin structure of layer 20 across content types.

## Reproducibility

```bash
# ~40 min on CPU (30 steps x ~80s per AV decode; see Methodological notes).
# Loads the AV only — h_A and h_B are read from interpolation_flipbook.pt, so
# neither the base model nor the AR is loaded. Inputs resolve from the working
# cache first, then the committed ../data/ copy on a clean clone.
python examples/nla_dense_interp_near_pivot.py

# ~10s, no model load: renders fig36 + fig37 from dense_interp_near_pivot.pt
python examples/nla_dense_interp_render.py
```

Step grid: 25 dense steps in t ∈ [0.395, 0.455] (Δt ≈ 0.0025) plus 5 sparse
context points {0.0, 0.25, 0.5, 0.75, 1.0}. AUDIT 18 in
`examples/nla_audit_findings.py` re-derives the plateau bounds and the
transition step from `../data/dense_interp_near_pivot.pt`.

## References

- [`2026-05-13-nla-interpolation-flipbook.md`](2026-05-13-nla-interpolation-flipbook.md) — MAIN-25, the coarse 20-step grid whose t=0.421 flag this re-samples, and the source of the cached `h_A`/`h_B`.
- [`2026-05-15-nla-plateau-attractor-strength.md`](2026-05-15-nla-plateau-attractor-strength.md) — MAIN-71, the follow-up that tests whether the plateau found here is a real basin.
- [`2026-05-14-nla-mid-seq-vocab-atlas-null-result.md`](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) and [`2026-05-14-nla-concept-arithmetic-atlas.md`](2026-05-14-nla-concept-arithmetic-atlas.md) — MAIN-44/48, the discrete-attractor synthesis this was designed to discriminate against.
- [`figures/INVENTORY.md`](figures/INVENTORY.md) — fig36 / fig37 provenance and the exact step grid.
- [`../README.md`](../README.md) — F1 and L9 carry the scope qualifications.
- `kitft/nla-qwen2.5-7b-L20-av` — <https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av>. Licensing and redistribution status: [`../data/LICENSE-DATA.md`](../data/LICENSE-DATA.md).
