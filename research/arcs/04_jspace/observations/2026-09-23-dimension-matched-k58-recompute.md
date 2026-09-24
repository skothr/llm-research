# Observation: at matched K/d the 1.5B-vs-7B excess gap shrinks from ~2× to ~1.5-1.8× and survives; 7B stays under the 10% ceiling at both K rules

**Date/context:** 2026-09-23. Owner-directed follow-up to the
2026-08-30 calibration caveat
([2026-08-30-excess-fve-dimension-dependence.md](2026-08-30-excess-fve-dimension-dependence.md)),
tracked as issue #83 (follow-on to issue #79 / PR #80).
Model: Qwen2.5-7B-Instruct, nf4, current wikitext lens
`data/cache/jlens_qwen2.5-7b_nf4_n100.pt` (the 2026-08-15/16 refit).
30 prompts per run, 8 random-baseline draws per position, seed base
30000, 2000 cluster-bootstrap resamples.
Three GPU runs of `examples/jspace_paper_metric_varfrac.py` with the
new `--k-fixed N` flag (commit `83b452ba`), each about 15 min on an
RTX 2080.
The 1.5B side is not re-run; its numbers come from the committed
artifacts and logs.

`--k-fixed N` sets K = N at every layer instead of K = median
occupancy.
Top-K is the selection-order prefix of the pursuit support, which
equals the N-step support because greedy pursuit only appends atoms.
`--k-snap 25` keeps its role as the bit-exact `--scan` validation
snapshot, and `--k-max 64` gives the pursuit room above 58.
Each artifact stores per-layer `K_used` and `n_short_support` (the
number of positions whose pursuit stopped before K atoms).
The log row gains a trailing `short=` column only when the flag is set.

K = 58 is 25 × 3584/1536 = 58.3, rounded.
It gives the 7B scan K/d = 58/3584 = 0.0162, against the 1.5B
25/1536 = 0.0163.

## Finding 1: held-out C4 prompts, the gap is 1.52× at matched K/d

Peak-layer rows (1.5B L21, 7B L23; the 7B peak layer is L23 at both
K values).
Sources: `scan_paper_metric_heldoutc4en_1p5b.log`,
`scan_paper_metric_heldoutc4en_7b.log`,
`scan_paper_metric_heldoutc4en_7b_k58.log`.

| | 1.5B L21, K=25 (K/d 0.0163) | 7B L23, K=23 (K/d 0.0064) | 7B L23, K=58 (K/d 0.0162) | factor at K=23 | factor at K=58 |
|---|--:|--:|--:|--:|--:|
| fveTopK | 0.1390 | 0.0695 | 0.0963 | 2.00× | 1.44× |
| fveRand | 0.0221 | 0.0107 | 0.0196 | 2.07× | 1.13× |
| excess | +0.1169 | +0.0588 | +0.0767 | 1.99× | 1.52× |
| excess CI95 | [+0.1133, +0.1204] | [+0.0556, +0.0618] | [+0.0730, +0.0803] | | |
| fveTopK / fveRand | 6.29 | 6.50 | 4.91 | 0.97× | 1.28× |

Factors are 1.5B / 7B.
The 1.5B CI95 and the 7B K=58 CI95 do not overlap (lower bound
+0.1133 against upper bound +0.0803).

## Finding 2: wikitext scan grid, the gap is 1.76× at matched K/d

Sources: 1.5B from the artifact
`paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
(L21 means; its peak layer), 7B from
`scan_paper_metric_7b_refitlens_k25.log` and
`scan_paper_metric_7b_refitlens_k58.log` (L22; the peak layer of both
runs).

| | 1.5B L21, K=25 | 7B L22, K=25 | 7B L22, K=58 | factor at K=25 | factor at K=58 |
|---|--:|--:|--:|--:|--:|
| fveTopK | 0.1285 | 0.0569 | 0.0788 | 2.26× | 1.63× |
| fveRand | 0.0202 | 0.0088 | 0.0172 | 2.30× | 1.17× |
| excess | +0.1083 | +0.0481 | +0.0616 | 2.25× | 1.76× |
| excess CI95 | [+0.1058, +0.1110] | [+0.0466, +0.0497] | [+0.0599, +0.0636] | | |
| fveTopK / fveRand | 6.35 | 6.44 | 4.59 | 0.99× | 1.38× |

The 1.5B grid artifact carries `excess_ci95` (the L21 band above) and
`boot_frac_over_10pct` 1.0 at L21.
It predates the `fve_ratio_topK_over_rand` and `K_over_d_model` fields,
so the 1.5B ratio is computed from the artifact's unrounded means.

The 7B K=25 run uses the current lens.
The README's earlier 7B grid figure (peak excess 0.0472 at L23) comes
from the July lens at K = 24, which the 2026-08-15/16 refit does not
reproduce exactly (see the 2026-07-24 observation's re-run note).
The K=25 run replaces it on the same prompts: excess moves by +0.0002
to +0.0015 at every layer.
That difference combines the lens refit and the K step from 23-24 to
25, so it bounds their joint effect, not the refit alone.
The 7B K=25 peak is L22 +0.0481 (L23 +0.0479).

## Finding 3: the random baseline is near parity at matched K/d; the ratio is not K-invariant

At matched K/d, fveRand is close to parity across scales: 0.0221 vs
0.0196 held-out (1.13×), 0.0202 vs 0.0172 on the grid (1.17×).
That is the behavior the K/d account of random-direction FVE
predicts.

The fveTopK / fveRand ratio changes with K at fixed scale.
Held-out 7B L23: 6.50 at K=23, 4.91 at K=58.
Grid 7B L23: 6.49 at K=25, 4.72 at K=58.
The 2026-08-30 observation read the near-equal cross-scale ratios
(6.29 vs 6.50) as evidence that the ratio removes a dimension factor.
That reading does not hold: the ratio falls as K grows, so its
near-equality at K=23-25 is a coincidence of the K values, not a
dimensionless quantity.
At matched K/d the ratio also favors 1.5B (1.28× held-out, 1.38×
grid), in the same direction as the excess gap.

## Finding 4: 7B stays under the 10% ceiling at both K rules

The largest 7B CI95 upper bound across the three new runs is +0.0803
(held-out, L23, K=58), and `P(>10%)` is 0.000 at every layer of every
run.
The 7B verdict holds under the paper's K rule (held-out peak excess
5.88%) and at matched K/d (7.67%).

![2026-09-23-jspace-paper-metric-matched-kd](figures/2026-09-23-jspace-paper-metric-matched-kd.png)

*Excess-over-random FVE by layer at matched K/d. (a) held-out C4
prompts: 1.5B at K=25, 7B at K=23 (paper rule) and K=58 (K/d 0.0162).
(b) wikitext scan grid: 1.5B at K=25, 7B current lens at K=25 and
K=58. Shaded bands are cluster-bootstrap CI95s; the dashed line is the
paper's 10% ceiling. The 7B curves rise with K and stay under the
ceiling.*
([provenance](figures/INVENTORY.md))

## What this does and does not establish

**Does:** on this evidence the 1.5B-vs-7B excess gap is real.
It survives matching K/d at a factor of 1.52× (held-out) and 1.76×
(grid).
The dimension factor explains part of the paper-rule gap: 1.99/1.52 =
1.31× of it held-out.
On the grid the equal-K pair (K=25, current lens) gives 2.25/1.76 =
1.28×, and the paper-rule pair (K=23-24, July lens) gives 2.30/1.76 =
1.31×.
On a log scale that is 39% held-out, and 30% (equal-K) or 32%
(paper-rule) on the grid.
The "7B under" verdict is robust to the K rule.

**Does not:** settle the reading against the paper's own models.
The paper's random-baseline FVE is unpublished, so no conversion from
this calibration to the paper's exists
`[gurnee2026-workspace §4.2 Fig 30b, §A.8]`.
The 1.5B breach remains a statement in this repository's calibration.

## Evidence

Verbatim log rows:

```
data/cache/logs/scan_paper_metric_heldoutc4en_7b_k58.log:
  [validate] replicated-vs-stored varfrac@25 max|diff| = 0.000e+00
  L23 K=58 n=270 ours@25=0.0526 fveTopK=0.0963 fveRand=0.0196 EXCESS=+0.0767 CI95=[+0.0730,+0.0803] P(>10%)=0.000 ratio=4.91 K/d=0.0162 short=0
  [VERDICT-INPUT] peak excess L23: +0.0767 (UNDER the 10% ceiling)
data/cache/logs/scan_paper_metric_7b_refitlens_k25.log:
  [validate] replicated-vs-stored varfrac@25 max|diff| = 4.379e-01
  L22 K=25 n=270 ours@25=0.0402 fveTopK=0.0569 fveRand=0.0088 EXCESS=+0.0481 CI95=[+0.0466,+0.0497] P(>10%)=0.000 ratio=6.44 K/d=0.0070 short=0
  L23 K=25 n=270 ours@25=0.0395 fveTopK=0.0566 fveRand=0.0087 EXCESS=+0.0479 CI95=[+0.0461,+0.0497] P(>10%)=0.000 ratio=6.49 K/d=0.0070 short=0
  [VERDICT-INPUT] peak excess L22: +0.0481 (UNDER the 10% ceiling)
data/cache/logs/scan_paper_metric_7b_refitlens_k58.log:
  [validate] replicated-vs-stored varfrac@25 max|diff| = 4.379e-01
  L22 K=58 n=270 ours@25=0.0402 fveTopK=0.0788 fveRand=0.0172 EXCESS=+0.0616 CI95=[+0.0599,+0.0636] P(>10%)=0.000 ratio=4.59 K/d=0.0162 short=0
  L23 K=58 n=270 ours@25=0.0395 fveTopK=0.0768 fveRand=0.0163 EXCESS=+0.0605 CI95=[+0.0588,+0.0623] P(>10%)=0.000 ratio=4.72 K/d=0.0162 short=0
  [VERDICT-INPUT] peak excess L22: +0.0616 (UNDER the 10% ceiling)
data/cache/logs/scan_paper_metric_heldoutc4en_1p5b.log:
  L21 K=25 n=270 ours@25=0.1330 fveTopK=0.1390 fveRand=0.0221 EXCESS=+0.1169 CI95=[+0.1133,+0.1204] P(>10%)=1.000
data/cache/logs/scan_paper_metric_heldoutc4en_7b.log:
  L23 K=23 n=270 ours@25=0.0526 fveTopK=0.0695 fveRand=0.0107 EXCESS=+0.0588 CI95=[+0.0556,+0.0618] P(>10%)=0.000
```

`short=0` holds at all 27 layers of all three new runs, so every
position had a full K-atom support.
The two older logs lack the `ratio=` and `K/d=` columns, which PR #80
added, and the `short=` column, which this PR added.

Validation gate.
The held-out K=58 run replicates the committed held-out structure scan
bit-exactly (max|diff| 0).
The two grid runs are validated against the July structure scan, which
used the July lens, so the gate cannot be bit-exact.
Its max|diff| of 4.379e-01 is an isolated-position outlier.
The per-layer mean varfrac@25 (`ours@25`) agrees with the July artifact
`paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`
to within 2.6e-3 at every layer (read from the two artifacts'
`vf_ours_mean`).
So at most a few of the 270 positions per layer differ materially.
Per-position diffs were not persisted, so the exact count is unknown.

Grid baselines read from the artifacts (`results[L]` fields):
1.5B L21 `fve_topK_mean` 0.1285, `fve_rand_mean` 0.0202,
`excess_mean` 0.1083 (argmax over layers);
7B July lens L23 `K_median_occ` 24, `excess_mean` 0.0472 (argmax), L22
0.0471.

Derived: 0.1390/0.0963 = 1.44; 0.0221/0.0196 = 1.13; 0.1169/0.0767 =
1.52; 6.29/4.91 = 1.28; 0.1285/0.0788 = 1.63; 0.0202/0.0172 = 1.17;
0.1083/0.0616 = 1.76; 0.1083/0.0481 = 2.25; 6.35/4.59 = 1.38;
1.99/1.52 = 1.31; 2.25/1.76 = 1.28; ln 1.31 / ln 1.99 = 0.39;
ln 1.28 / ln 2.25 = 0.30; 0.10828/0.04718 = 2.30 (July-lens grid,
unrounded means); 2.30/1.76 = 1.31; ln 1.31 / ln 2.30 = 0.32.

## Reproducibility

GPU runs (common prefix, then the per-run arguments):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  examples/jspace_paper_metric_varfrac.py \
  --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --device cuda \
  --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
  --n-prompts 30 --n-rand 8 --rand-seed-base 30000 --k-snap 25 --k-max 64 \
  <per-run arguments>

# held-out C4, K=58
  --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt \
  --prompts research/arcs/04_jspace/data/heldout_prompts_c4en_n30.json \
  --k-fixed 58 \
  --out research/arcs/04_jspace/data/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en_k58.pt

# wikitext grid, K=25 (current lens)
  --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt \
  --prompts research/arcs/04_jspace/data/heldout_prompts_wikitext103_n30.json \
  --k-fixed 25 \
  --out research/arcs/04_jspace/data/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_refitlens_k25.pt

# wikitext grid, K=58 (current lens)
  --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt \
  --prompts research/arcs/04_jspace/data/heldout_prompts_wikitext103_n30.json \
  --k-fixed 58 \
  --out research/arcs/04_jspace/data/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_refitlens_k58.pt
```

Logs: `data/cache/logs/scan_paper_metric_heldoutc4en_7b_k58.log`,
`scan_paper_metric_7b_refitlens_k25.log`,
`scan_paper_metric_7b_refitlens_k58.log`.
Each log was written by the launcher, not by the script alone.
An `echo` first wrote a `[launch]` line with the timestamp and arguments.
The run was then launched as
`<command> 2>&1 | tee -a research/arcs/04_jspace/data/cache/logs/<log>`.
The `[done]` trailer line with the exit code is the launcher's, not the
script's.
Audit CHECK P (`examples/jspace_audit_findings.py`) re-derives the
pinned numbers above from the three logs.
The logs are plain committed files under `data/cache/logs/`, not LFS
objects, so CHECK P's log claims run on every clone, including LFS-less
ones.

## Hypotheses / limitations

- K = 58 is one matching rule (equal K/d).
  The paper's rule (K = median occupancy at the snapshot) is another,
  and under it the 7B K stays at 23-24.
  Both are reported; neither is claimed as the canonical cross-scale
  calibration.
- n = 2 scales, 30 prompts per scan.
  The gap factor is measured, not a fitted scaling law.
- fveRand exceeds K/d at both scales (0.0221 vs 0.0163 at 1.5B, 0.0196
  vs 0.0162 at 7B, held-out).
  The K/d account is first-order, so fveRand parity is approximate.
- Peak layers differ between scales (1.5B L21; 7B L22-L23).
  The tables compare each scale's peak, as the ceiling verdicts do.

## Follow-ups

- None required for the cross-scale question on this evidence.
- A third scale would turn the gap factor into a trend.
  It is not scheduled.

## References

- `[gurnee2026-workspace §4.2 Fig 30b, §A.8]`: the ceiling's definition
  (excess-over-random orthogonal-projection FVE at K = median
  occupancy).
- [2026-08-30-excess-fve-dimension-dependence.md](2026-08-30-excess-fve-dimension-dependence.md):
  the calibration caveat this recompute resolves; its ratio-invariance
  reading is corrected in Finding 3 above.
- [2026-07-24-paper-metric-varfrac-recompute.md](2026-07-24-paper-metric-varfrac-recompute.md):
  the paper-metric recompute and the July-lens 7B grid number.
- Issues #79 and #83; PR #80.
