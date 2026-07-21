# Full-vocabulary sweep — staged plan (2026-06-10, second session phase)

**Direction (user):** reverse the battery approach — instead of curated
semantic groups probed for shared features, sweep the ENTIRE token universe
(151,665 real rows) for similarity structure, then reason about connections
between tokens whose data correlates. Go deeper than cosine (a single scalar
collapsing 3584 dims): look for dimension-level and correlated-dimension
("entangled") structure that could serve as granular functional handles.
Methodology per the thorough-data discipline (global CLAUDE.md, 2026-06-10):
scaling steps, full-population coverage before conclusions, branch +
push-notify on unexpected structure.

## Measurement battery (what goes deeper than cosine, concretely)

1. **Per-dimension moments over the full vocabulary** — mean, std, skew,
   kurtosis per dim (fp64 accumulation). High-kurtosis dims are candidate
   "feature dims" (a few tokens use them hard, most do not) — the
   privileged-basis question at layer 0.
2. **Dimension-dimension correlation matrix** (3584 x 3584 over alive rows)
   — off-diagonal mass, top-|r| pairs, eigen-spectrum, and correlated BLOCKS
   (the user's "entangled dimensions"). Full matrix stays in cache; committed
   artifact carries the summary (histogram, top-1000 pairs, spectrum,
   block-clustering assignments).
3. **Full-vocab k-NN graph** (k=32, cosine, centered space, dead rows
   excluded) — the substrate for unsupervised community discovery and for
   any later interactive (llobotomy) exploration.
4. **Handle scores for every token** — project all alive rows onto the 54
   battery contrast directions: does each curated category have large
   unlabeled membership in the uncurated vocabulary?
5. **Cosine decomposition on edges** (stage 3) — for k-NN edges, the per-dim
   contribution profile of the cosine: is a given similarity carried by a
   few dims (sparse shared feature) or spread (diffuse)? Defines a
   "similarity concentration" measure per edge.
6. **Downstream-readout map** (stage 3+, bridges to rope-vis) — W_E carries
   no position; RoPE acts on per-head q/k inside attention. So a dimension's
   FUNCTION is defined by who reads it: layer-0/early-layer W_Q/W_K/W_V
   column norms per embedding dim (and per rotary frequency band) gives
   "which dims layer-1 attention actually consumes," letting us weight
   similarity by functional relevance instead of treating all dims equally.

## Scaling steps

- **S0 (one-time):** dump W_E/W_U bf16 to the gitignored cache so later
  stages never reload the model.
- **S1 (protocol validation, 10k random alive rows, seeded):** run
  measurements 1-4 on the sample; validate shapes/distributions (no NaNs,
  kurtosis finite, kNN cosine distribution shape, handle-score ranges).
  Sample artifacts are cache-only, never committed.
- **S2 (full population):** same measurements over all alive rows; artifacts
  committed (est. sizes: dim stats < 1 MB; corr summary ~10 MB; kNN graph
  ~30 MB; handle scores ~17 MB fp16) + manifest entries + audit lines.
- **S3 (analysis):** community detection on the kNN graph vs battery
  categories; edge cosine-decomposition; entangled-block characterization;
  per-block token sampling for interpretation.
- **S4 (branch):** decided by S2/S3 data; push-notify when a branch-worthy
  structure appears (encoded triggers: bimodal kNN-cosine structure,
  extreme-kurtosis dim clusters, correlation blocks above chance size,
  handle directions with large out-of-battery membership).

## Conclusions policy

Battery-scale results from the first session phase are protocol probes.
Findings prose from this sweep states population coverage explicitly; no
generalization from the sample stage.
