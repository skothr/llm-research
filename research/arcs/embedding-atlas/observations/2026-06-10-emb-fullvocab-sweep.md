# Full-vocabulary sweep: one cross-script "structural" dimension block, precise handles, crisp neighbor islands

**Date / context:** 2026-06-10, second session phase, per the thorough-data
discipline — this sweep covers **all 149,706 alive rows** of W_E (151,665
real minus 1,959 dead), not a curated sample. Model/protocol as in
[2026-06-10-emb-global-geometry.md](2026-06-10-emb-global-geometry.md).
Scaling steps: S1 10k-row protocol validation (passed; cache-only), S2 full
sweep (`emb_fullvocab_stats.py`, ~6 min CPU), S3 decode
(`emb_fullvocab_analyze.py`), S4 branch (`emb_structural_block.py`, branch
trigger pushed to user). Locked by AUDIT 8 (17 claims; total 61 PASS / 0 FAIL).

## Findings

**F-V1 (headline, the fired branch trigger). W_E contains exactly one
entangled dimension block — 21 dimensions, cross-script, loaded on the
highest-frequency structural tokens.** At |r| > 0.3 over all alive rows, the
3584-dim correlation graph has a single connected component of size > 1:
21 dims (incl. the top-kurtosis dims 1395 [116.3], 2898 [51.1], 822 [28.5]).
Its extreme tokens mix scripts at the same role: `','` and `'，'`, `' the'`
and `'的'`, `'.'` and `'。'`, newlines, digits. Characterization vs a seeded
random 21-dim control (fig15):

- energy fraction vs token-id (BPE merge order = frequency proxy): Spearman
  **-0.206** (control -0.003); the first id-decile carries mean fraction
  **0.1143** vs control 0.0753, flattening to ~0.078 after decile 3 —
  **head-loaded**, not a smooth frequency gradient.
- script-independence: block mean energy elevated over control for ascii
  (0.0836), cjk (0.0811), and cyrillic (0.0784) alike (control ~0.0754).
- top block-energy tokens: `',' '.' '，' ' the' '\n' ' ' '。' '.\n' ' to'`;
  bottom: rare fragments (`' Rear'`, `'utsch'`, `'.Paths'`).

[INTUITION: a shared "syntax/glue-token" subspace that the tokenizer's most
frequent structural tokens occupy regardless of language — return to the
canonical form: a rank-~21 correlated subspace S with energy concentrated in
the top frequency decile, P(energy in S | top-decile) ≈ 1.5x the floor.]

**F-V2. Outside that block, dimensions are remarkably independent.**
Off-diagonal |r| mean is 0.0210 (max 0.726, inside the block); kurtosis
median is 0.32 (near-Gaussian) with only a thin heavy tail. No evidence of
basis-aligned "one dim = one semantic feature" structure for content words —
consistent with features-as-directions (superposition) rather than
features-as-dimensions [Elhage et al. 2022, arXiv:2209.10652, as the
interpretive frame; the measurement here is the correlation/kurtosis census].

**F-V3. The battery handles are precise at vocabulary scale, conservative
in recall.** Projecting every alive token onto the 54 contrast directions
and thresholding at each class's own in-battery 10th percentile: top
out-of-battery hits are semantically clean (negative -> ' shitty', ' nasty',
' rotten' [60 hits]; code -> 'namespace', 'public', 'function' [32];
positive -> ' fantastic', ' lovely' [7]; preposition's top hits include
'的' and ' de' — cross-lingual again), but hit counts are small: the
thresholds inherit overfit from directions built on the battery itself.
Ranking quality is the honest signal; calibrated-recall measurement is a
named follow-up, not a claim.

**F-V4. The k=32 neighbor graph fragments into one giant component (82%)
plus crisp semantic islands.** Label propagation yields 542 communities:
largest 122,942 (the undifferentiated bulk at this k), then islands that
decode cleanly: English suffix fragments (1,007), code syntax (965 + 416 +
331), given names (908), surnames (329), name fragments (587), countries
(316), and a cross-lingual TIME community (368: '年底', ' Jahren',
' tháng', ' hours', '月底'). Top-1 neighbor cosine is unimodal
(mean +0.314; no bimodality — that branch trigger did not fire).

**F-V5. Dead rows confirmed at population scale:** 1,959 alive-range
exclusions applied to every statistic here (padded 399 exactly zero).

## Evidence

```
moments in 5s; kurtosis max 116.3 median 0.32
dim-corr in 16s; |r| mean 0.0210 max 0.726
knn in 307s; top1 cos mean +0.314 k32 cos mean +0.159
== dim-corr blocks at |r|>0.3: 1 blocks of size >1; sizes [21]
block    spearman(energy, token_id) = -0.2064; decile1 0.1143
control  spearman(energy, token_id) = -0.0027; decile1 0.0753
negative thr +0.209 hits 60: ' shitty', ' nasty', ' rotten', ...
542 communities; largest [122942, 1007, 965, 909, 908, ...]
```

Figures fig11-fig15 (INVENTORY.md). All numbers re-asserted by AUDIT 8.

## Reproducibility

```bash
python examples/emb_fullvocab_stats.py --dump        # S0: one model load
python examples/emb_fullvocab_stats.py --stage sample  # S1 gate
python examples/emb_fullvocab_stats.py --stage full    # S2 (~6 min CPU)
python examples/emb_fullvocab_analyze.py               # S3 decode
python examples/emb_structural_block.py                # S4 branch
python examples/emb_fullvocab_render.py && python examples/emb_structural_block_render.py
python examples/emb_audit_findings.py                  # 61 PASS | 0 FAIL
```

S3/S4 need the S0 matrix dump (cache-only, regenerable from the pinned
model); the audit and renders run from committed data alone.

## Hypotheses / follow-ups

- H1: the 21-dim block is the W_E-side trace of the known "attention-sink /
  high-norm structural token" phenomenon; test by checking whether arc-1's
  layer-20 sink dims relate to these 21 dims through the early-layer maps.
- H2: token-id is a noisy frequency proxy (vocab-extension ranges, added
  specials); re-test head-loading against an empirical token-frequency count
  from a corpus sample before quantifying further.
- H3: calibrated handle recall — fit thresholds on held-out battery splits
  instead of in-battery percentiles, then re-census.
- H4: communities at smaller k (or mutual-kNN) should fragment the giant
  component; the islands found here are lower bounds on recoverable
  structure.

## Caveats

- |r| > 0.3 is one threshold; the block's membership count (21) is
  threshold-dependent (the audit pins the value AT this threshold).
- Spearman -0.206 is population-scale but driven by the head decile;
  do not read it as a vocabulary-wide frequency code (see H2).
- Label propagation is order-dependent even seeded; community COUNTS are
  indicative, the decoded island CONTENT is the robust part.
