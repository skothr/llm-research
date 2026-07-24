# Observation: the ceiling claims survive the paper-faithful variance metric — 1.5B breach confirmed (L21 excess 11.5%, CI [11.3, 11.7]), 7B stays under (peak 5.0%); swap headline significance certified; pursuit norm-bias bounded

**Date/context:** 2026-07-23/24. Vetting session on the closed arc (issue
#26): the paper's "never more than 10%" ceiling was verified against the
archived PDF and found to be a **different metric** than the arc's committed
variance fraction; the ceiling comparison was then recomputed under the
paper's definition at both scales, and two adjacent gaps (missing
significance stats on the stage-5.2 headline; unnormalized-atom selection
bias in pursuit) were quantified. Scripts:
`examples/jspace_paper_metric_varfrac.py`,
`examples/jspace_swap_significance.py`, `examples/jspace_atom_norm_bias.py`.

## Finding 1 — metric mismatch, but both ceiling verdicts survive

**The paper's metric** `[gurnee2026-workspace §4.2, Fig 30b; §A.8]`:
`excess = FVE(top-K pursuit atoms) − FVE(K random vocab atoms)`, where FVE
is the **orthogonal-projection** fraction `||Π_S h||² / ||h||²`
(least-squares onto the selected-atom span, §A.8) at **K = median pursuit
occupancy** per layer, over valid positions. The arc's committed metric
(`_jspace_pursuit.py`) is the absolute single-gradient-step reconstruction
energy `||A·x||² / ||h||²` at fixed k — three mismatches (no LS refit, no
random control, fixed k), pushing in opposite directions.

Recomputed under the paper's definition (same lenses, same prompts;
validation gate: replicated varfrac@25 matches the committed structure
scans **bit-exactly**, max|diff| = 0.0 at both scales):

| | naive (committed) | paper metric (excess FVE) | verdict |
|---|--:|--:|---|
| 1.5B L21 (peak), scan grid | 12.4% | 11.1% | breach stands |
| 1.5B L21, **all valid positions** (n=5362) | — | **11.50%, CI95 [11.29, 11.74], 2000/2000 cluster-bootstrap resamples > 10%** | **breach confirmed** |
| 1.5B L22, all positions | 11.4% | 10.88% [10.68, 11.11], 2000/2000 | breach confirmed |
| 1.5B L18, all positions | 9.4% | 8.85% [8.67, 9.06], 0/2000 | under (hump edge) |
| 7B peak (L22), scan grid | 4.0% | **5.04%** | **under confirmed** |

The two main distortions nearly cancel at the 1.5B workspace band (LS
refit adds ~+2 pts, the random control subtracts ~2 pts) — a regime
coincidence, not a general property: at 7B the naive metric *understates*
(3.4% → 4.5% at L21). Cross-scale gap ~2.3× (was ~3× naive). The
all-positions sweep (the paper's population; the committed scan's 9-point
grid is a subsample) moved L21 from 11.1% → 11.5% — the grid was mildly
conservative. L0 excess is 12.3% but sits outside the paper's
workspace-layer scope for the ceiling (Fig 30b evaluates workspace layers)
and carries the Finding-3 contamination.

**Occupancy saturation note:** median occupancy = 24–25 of k=25 at both
scales (`ACTIVE_TAU = 1e-3` counts nearly every selected atom as active —
the stage-4 observation's point 3), so the paper's K = median-occupancy
prescription lands at K ≈ 25 here; the arc's k=25 headline was
procedure-equivalent by accident.

## Finding 2 — the stage-5.2 causal gap is statistically certified

The +5.17/+2.17 nat headline had committed means only (the README's
"per-item SD 4.9" was prose). Derived from the committed
`entailed_swap_chat_*` artifacts (exact two-sided sign-flip permutation,
`jspace_swap_significance.py`):

| subset | jlens−logitlens gap | positive | perm-p | boot95% |
|---|--:|--:|--:|---|
| 1.5B L18 auto (n=7) | +5.02 (SD 4.92) | 7/7 | 0.0156 | [+1.82, +8.46] |
| 1.5B L18 mixed (n=17) | +2.05 | 13/17 | 0.0247 | [+0.43, +4.09] |
| 7B L19 auto (n=17) | +1.90 (SD 4.12) | 15/17 | **0.0001** | [+0.45, +4.12] |
| 7B L19 mixed (n=30) | +1.07 | 20/30 | 0.0004 | [+0.22, +2.40] |

jlens−random contrasts are equally strong (7B auto p=0.0001). Caveats:
0.0156 is the *smallest p an n=7 sign-flip test can produce* (2/128 —
the data cannot express stronger evidence at that n), and L18/L19 were
selected as peak layers from the same data, so the p-values are post-hoc
at a data-chosen maximum. The 7B result (n=17, p=1e-4) is the statistically
stronger citation.

## Finding 3 — pursuit norm-bias: workspace band clean, early band contaminated

Pursuit selects by `argmax⟨a_v, r⟩` with **unnormalized** atoms
`a_v = W_U[v]·J_ℓ` (neither paper nor companion repo specifies
normalization). Measured against the full-vocab atom-norm distribution
(1.5B, `jspace_atom_norm_bias.py`): the structure scan's selected atoms sit
at norm-percentile **~50 in the workspace band** (L18: 49.8, L21: 47.0 —
norm-neutral) but **56–70 in the early band** (L0: 69.5, 30% of selections
from the top norm decile) where the top-norm atoms are whitespace/quote/
ellipsis tokens — early-band occupancy and top-atom readings are partly
norm-driven, not content-driven. Final layers invert (L26: 26.0). Spearman
rho(atom norm, W_U row norm) rises 0.23 → 0.72 with depth; at 1.5B W_U is
the **tied input embedding** (frequency-correlated norms), while 7B is
untied — a cross-scale structural difference the paper never contemplates.

## Also verified (no change needed)

The J_ℓ estimation reduction (sum over targets t′≥t, mean over sources,
mean over prompts, skip-16) matches the paper's reproduction pseudocode and
`jlens/fitting.py` exactly — the KB note's joint-expectation formula was
the loose idealization (KB corrected this pass). The kurtosis metric
(Fisher excess kurtosis of the full-vocab readout logits) matches footnote
4 exactly, so the stage-4 **kurtosis inversion stands** on a verified
metric. The readout `softmax(W_U · finalnorm(J_ℓ h))` matches §2.1.

## Evidence

Committed artifacts (`data/`, MANIFEST-registered; regenerated by the
committed scripts):
`paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
(scan grid + validation),
`paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_allpos.pt`
(all positions, per-position arrays + cluster bootstrap),
`paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`,
`atom_norm_bias_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`.
Significance stats recompute deterministically from the committed
`entailed_swap_chat_*` artifacts (no new dataset). Figure:
`figures/2026-07-24-jspace-paper-metric-excess.png`
(`examples/jspace_render_paper_metric_figure.py`; INVENTORY +
DATA_PROVENANCE rows). Audit coverage: Check M blocks in
`examples/jspace_audit_findings.py`.

## Reproducibility

```
# paper-metric recompute (GPU; ~15 min 1.5B grid / ~1.3 h all-pos / ~1 h 7B)
python examples/jspace_paper_metric_varfrac.py \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
    --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt
python examples/jspace_paper_metric_varfrac.py \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
    --all-positions --layers 0,18,21,22 --n-rand 4 --rand-seed-base 20000
python examples/jspace_paper_metric_varfrac.py \
    --model Qwen/Qwen2.5-7B-Instruct --mode nf4 \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
    --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt \
    --rand-seed-base 30000
# CPU-only stats (committed inputs)
python examples/jspace_swap_significance.py
python examples/jspace_atom_norm_bias.py   # needs the cache-only full lens
```

## Notes / limitations

- Random control = K uniformly-drawn vocab atoms (seeded); the paper says
  "a same-size random control set" without specifying the draw — uniform
  atoms is the natural reading.
- The paper reports Fig 30b at five workspace layers with K set per layer;
  we sweep all layers with the same procedure. The bootstrap unit is the
  prompt (positions within a prompt correlate).
- 7B numbers are scan-grid (n=270/layer, CI from cluster bootstrap in the
  committed artifact); at ~5% vs a 10% ceiling the margin does not require
  the all-positions sweep.
- Verification-method provenance: the paper-metric definition was
  extracted from the archived PDF (§4.2, Fig 30b caption, §A.8) — see the
  KB note `[kb/notes/interpretability/j-space.md]` §1.2 for the corrected
  operational definition.

## Follow-ups

- Recompute the corpus/quantization/n-budget robustness comparisons under
  the paper metric if they are ever cited externally — as *internal*
  invariance comparisons (same metric both sides) they remain valid.
- Coefficient-mass occupancy recalibration (stage-4 follow-up) would make
  K = median occupancy non-degenerate.

## References

- `[gurnee2026-workspace §2.3, §4.2, Fig 30a/b, §A.8, footnote 4]`
- Issue #26 (metric mismatch + corrections); PR #25 (arc)
- Prior: `2026-07-20-jspace-structure-stage4.md` (addendum points here),
  `2026-07-22-entailed-property-swaps-stage52.md`
