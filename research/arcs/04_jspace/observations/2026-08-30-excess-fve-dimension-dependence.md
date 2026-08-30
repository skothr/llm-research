# Observation: excess-FVE is not dimensionless — the cross-scale ceiling comparison carries a common ~2× factor that the subtraction does not remove

**Date/context:** 2026-08-30. Post-close correction pass (arc closed
2026-07-22; same discipline as the issue-#26 metric-fidelity reopening).
The defect was flagged by an external review session and verified here
against the committed scan logs — every number below re-derives from
plain-text logs already in the repository (`../data/cache/logs/`,
committed with the PR #45 cache), no model or GPU involved. Tracking:
issue #79.

## Finding — signal and baseline share a common multiplicative factor, so `excess = fveTopK − fveRand` retains it

The paper-metric recompute
([2026-07-24-paper-metric-varfrac-recompute.md](2026-07-24-paper-metric-varfrac-recompute.md))
compares `excess = FVE(top-K pursuit atoms) − FVE(K random vocab atoms)`
against the paper's 10% ceiling at both scales. But the FVE of K random
directions in a d-dimensional space scales like K/d, and the measured
values show the *pursuit* FVE carrying the same scale factor — so the
subtraction preserves it rather than cancelling it. From the held-out
scan logs (`scan_paper_metric_heldoutc4en_1p5b.log` L21 row,
`scan_paper_metric_heldoutc4en_7b.log` L23 row):

| | 1.5B L21 (d=1536, K=25) | 7B L23 (d=3584, K=23) | factor |
|---|--:|--:|--:|
| fveTopK | 0.1390 | 0.0695 | 2.00× |
| fveRand | 0.0221 | 0.0107 | 2.07× |
| excess | +0.1169 | +0.0588 | 1.99× |
| fveTopK / fveRand | 6.29 | 6.50 | 0.97× |

The three difference-metric rows scale together by ~2×; the
ratio-normalized row is near-invariant and slightly *reverses* (7B ≥
1.5B). K/d_model differs by 2.54× between the columns (25/1536 = 0.0163
vs 23/3584 = 0.0064). Consequences:

1. **The ~2.4× cross-scale excess gap quoted in the README is fully
   accounted for by a factor common to signal and baseline.** On the
   ratio normalization the gap vanishes.
2. **Comparing a d=1536 model's excess against a ceiling measured on
   much larger models is not like-for-like.** The paper's
   Claude-family models have (much) larger hidden dimensions than
   1536; a fixed 10% excess threshold is easier to breach at small d
   if the common factor behaves as observed here.

### Independent symptom — the ratio also fixes a known ordering defect

The excess metric puts L0 *above* the workspace hump, which the arc
currently handles by scoping L0 out as norm-bias contamination (Finding
3 of the 2026-07-24 observation). The ratio resolves it without any
scoping (from `scan_paper_metric_c4en_1p5b.log` and
`scan_paper_metric_heldoutc4en_1p5b.log`):

- **C4-en lens, 1.5B:** L0 excess +0.1542 is the argmax over all 27
  layers, above the L21 hump's +0.1082. Under the ratio: L0 3.78 <
  L21 6.72 — the hump is restored as the peak.
- **Held-out set, 1.5B:** L0 excess +0.1004 is 86% of the L21 peak and
  straddles the ceiling (P(boot > 10%) = 0.577); on the ratio it is
  47% of peak (2.97 vs 6.29).

A normalization that independently resolves a defect the arc had to
scope out by hand is doing real work, not just rescaling.

## What this does and does not establish

**Does:** the cross-scale excess gap is explained by a factor common to
signal and baseline; the two README ceiling verdicts (1.5B breach, 7B
under) are calibrated in a dimension-dependent unit and should carry a
caveat when read as a cross-scale or versus-paper comparison.

**Does not:** show the 1.5B breach is spurious. The paper defines its
ceiling on the difference metric, and its random-baseline FVE is
unpublished, so no conversion between the two calibrations exists. The
within-scale statements (bootstrap CIs, corpus/n-budget/quantization/
held-out invariance, the L21 peak layer) are unaffected. The correct
outcome is a stated calibration caveat on the cross-scale readings —
not a retraction.

## Evidence

All values are verbatim fields of the committed scan logs:

```
data/cache/logs/scan_paper_metric_heldoutc4en_1p5b.log:
  L 0 K=24 ... fveTopK=0.1516 fveRand=0.0511 EXCESS=+0.1004 ... P(>10%)=0.577
  L21 K=25 ... fveTopK=0.1390 fveRand=0.0221 EXCESS=+0.1169 ... P(>10%)=1.000
data/cache/logs/scan_paper_metric_heldoutc4en_7b.log:
  L23 K=23 ... fveTopK=0.0695 fveRand=0.0107 EXCESS=+0.0588 ... P(>10%)=0.000
data/cache/logs/scan_paper_metric_c4en_1p5b.log:
  L 0 K=24 ... fveTopK=0.2096 fveRand=0.0554 EXCESS=+0.1542 ... P(>10%)=1.000
  L21 K=25 ... fveTopK=0.1271 fveRand=0.0189 EXCESS=+0.1082 ... P(>10%)=1.000
```

Derived: 0.1390/0.0695 = 2.00; 0.0221/0.0107 = 2.07; 0.1169/0.0588 =
1.99; ratios 0.1390/0.0221 = 6.29, 0.0695/0.0107 = 6.50,
0.2096/0.0554 = 3.78, 0.1271/0.0189 = 6.72, 0.1516/0.0511 = 2.97;
fractions 0.1004/0.1169 = 0.86, 2.97/6.29 = 0.47.

## Reproducibility

Every number above is re-derived by the audit's CHECK O
(`examples/jspace_audit_findings.py`, 27 claims), which parses the
three logs and fails loudly on any drift — the logs are plain git
files, so CHECK O runs on every clone including LFS-less ones. For
future scans, `examples/jspace_paper_metric_varfrac.py` now also
reports and persists `fve_ratio_topK_over_rand` and `K_over_d_model`
per layer (pure additions; the bit-exact `--scan` validation gate is
untouched).

## Hypotheses / limitations

- The observed common factor (~2.0×) is close to, but not exactly, the
  d-ratio 3584/1536 = 2.33 or the K/d ratio 2.54: the K/d scaling is
  the natural first-order account for fveRand (25/1536 = 0.0163 vs
  measured 0.0221; 23/3584 = 0.0064 vs 0.0107), and the pursuit FVE
  tracking it is an empirical observation here, not a derived law.
- The ratio normalization is one candidate calibration, adopted here
  only as evidence that the gap is a common factor; whether it is the
  *right* cross-scale normalization is an open question the arc does
  not settle.
- Both scales were measured at K = per-layer median occupancy (23-25),
  so K is nearly matched while d differs — the comparison isolates the
  d-dependence but has n=2 scales.

## Follow-ups

- Optional (GPU, ~one pursuit re-run, no lens refit): recompute the 7B
  scan at dimension-matched K ≈ 25 × 3584/1536 ≈ 58 (needs a k_max
  above 58). If the cross-scale gap survives at matched K/d, the scale
  finding is stronger than currently stated and should be restated as
  such. Tracked in issue #79.

## References

- `[gurnee2026-workspace §4.2 Fig 30b, §A.8]` — the ceiling's
  definition (excess-over-random orthogonal-projection FVE).
- [2026-07-24-paper-metric-varfrac-recompute.md](2026-07-24-paper-metric-varfrac-recompute.md)
  — the recompute this observation qualifies; its Finding 3 documents
  the L0 norm-bias scoping that the ratio resolves independently.
- Issue #79 — tracking; README § Findings item 2 carries the
  calibration caveat added with this observation.
