# Observation: J-space occupancy replicates the ≤10% ceiling; the kurtosis onset signature does not (stage 4)

**Date/context:** 2026-07-20. Stage 4 of the design plan: gradient-pursuit
J-space structural characterization on both n=100 lenses
(`jlens_qwen2.5-1.5b_bf16_n100`, `jlens_qwen2.5-7b_nf4_n100`), 30 held-out
wikitext prompts × 9 positions (8 mid + last), k-sparse nonnegative
decomposition onto the J-lens vector dictionary (rows of $W_U J_\ell$),
k=25 default, k swept {5,10,25,50}. Script:
`examples/jspace_structure_scan.py`. Gradient pursuit **reimplemented** —
the companion repo ships no pursuit/decomposition code — following the
paper's cited method (Blumensath & Davies, paper refs 34–35): greedy
nonnegative atom selection (argmax positive correlation with the residual)
with one optimal-step gradient update per selection; dictionary never
materialized (correlations as the matvec chain $W_U (J_\ell r)$; only
selected atoms formed). Runs are deterministic: a full re-run reproduced
every varfrac to ≤5e-5 (rounding of the comparison reference).

## Finding

**The paper's variance-fraction ceiling replicates at both scales; its
depth *profiles* only partially — and the kurtosis workspace-onset
signature does not replicate at all.**

**1. J-space variance fraction (k=25) stays under the paper's ~10% ceiling,
with scale-dependent depth shape:**

| layer | 1.5B varfrac | 7B varfrac | | layer | 1.5B varfrac | 7B varfrac |
|--:|--:|--:|--|--:|--:|--:|
| 0 | 0.082 | 0.031 | | 18 | 0.094 | 0.016 |
| 5 | 0.073 | 0.020 | | **21** | **0.124** | 0.036 |
| 9 | 0.082 | 0.019 | | **22** | 0.114 | **0.040** |
| 13 | 0.070 | 0.013 | | 24 | 0.098 | 0.036 |
| **17** | 0.084 | **0.012** | | 26 | 0.068 | 0.035 |

(bold = each model's extremum; full 27-layer tables in the artifacts.)
1.5B is single-humped with a mid-late band (layers 19–24, peak 12.4% at
L21 — slightly *above* the paper's 10% at the hump). 7B *declines* to a
mid-depth trough (1.2% at L17) then rises to a late band (4.0% at L22,
sustained to the end) — a U-shape, not a hump. J-space occupancy is ~3×
smaller at 7B than 1.5B at matched k; the n=100 fit-budget confound (H1,
see the H2 observation) applies to that comparison directly.

**2. The kurtosis onset signature is inverted relative to the paper.**
Paper claim (verbatim-verified, incl. metric definition:
`[kb/excerpts/gurnee2026-workspace#sec-4-1-kurtosis-onset]`): excess
kurtosis of the J-lens readout **logit distribution** (full vocabulary) is
"near zero through the first third of the layers, increases beginning
around a third of the way through the depth, and falls in the last few
layers" — the rise marks workspace onset. Measured on Qwen (same metric):

| | L0 | mid-trough | late |
|---|--:|--:|--:|
| 1.5B logit-kurt | 4.65 | 0.93 (L18) | 1.35–1.84 (L24–26) |
| 7B logit-kurt | 3.25 | 0.87–0.88 (L19–20) | 0.95–1.04 (L22–26) |

High **early**, declining to a mid trough, weak late rise — the opposite
shape. The inversion is **not** a computation artifact: the same profile
appears in a softmax-distribution supplement (`readout_prob_kurtosis`,
recorded alongside; e.g. 1.5B 1.28e5 @ L0 → 8.35e4 @ L18 → rebound), so
logit-vs-probability choice does not change the story. A weak echo of the
paper's structure survives: the trough→late-rise transition (L18–L24) sits
in the band where the varfrac hump/late-band also lives.

**3. Active-atom count pins at the sparsity cap** (23–25 of k=25 at both
scales, threshold τ = 1e-3 × max coefficient), and the k-sweep shows
varfrac still climbing at k=50 (1.5B L21: 0.063/0.086/0.124/0.152 for
k=5/10/25/50). At this τ the decomposition is budget-limited, not
intrinsically sparse — the paper's "~10–25 meaningfully active" needs a
calibration convention we could not extract from the excerpts.

**4. Qualitative atom trajectories (1.5B) show the paper's
sensory→workspace→motor progression:** surface/noise tokens early →
document-frame tokens (`cite`, `text`, `pages`) mid → task/output tokens
(`What`, `Question`, `Based`, `Answer`) from ~L20 on.

## Evidence

Artifacts (gitignored cache):
`structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`,
`structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` — each
`{summary, per_prompt[30]}` with per-position varfrac/active/kurtosis/
top-atoms. Logs: `stage4_1.5b_v2.log`, `stage4_7b_v2.log` (session
scratchpad); runtimes 13.6 s/prompt (1.5B bf16) and 24.3 s/prompt (7B nf4,
+~5.5 min load). First 7B attempt OOM'd (preloading all 27 J_ℓ to GPU +
a duplicated $W_U$ transpose copy); fixed by per-layer on-demand J loading
and transposed views, then verified bit-identical on a 1.5B smoke before
the 7B run (Amendment A satisfied — no CPU fallback). Headline figure:
`figures/2026-07-20-jspace-structure-depth-map.png`
(`examples/jspace_render_structure_figures.py`). Load-bearing numbers are
audit-covered (Check D, `examples/jspace_audit_findings.py`).

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_structure_scan.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --device cuda --lens research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
    --n-prompts 30 --ks 5,10,25,50
python examples/jspace_structure_scan.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --device cuda --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
    --n-prompts 30 --ks 5,10,25,50
python examples/jspace_render_structure_figures.py
```

## Method notes

- Decomposition operates on raw $h_\ell$ against atoms $W_U J_\ell$ rows;
  the readout's final-norm is treated as a token-scoring nonlinearity, not
  part of J-space geometry (per the note's operationalization,
  `[kb/notes/interpretability/j-space.md]` §1.1–1.2).
- Kurtosis metric verified against the archived PDF **before** interpreting
  the mismatch: footnote 4 specifies the logit distribution, full vocab, no
  top-k (the panel-a "top-k" curves are next-token accuracy). The excerpt
  was added to the KB in the same pass
  (`[kb/excerpts/gurnee2026-workspace#sec-4-1-kurtosis-onset]`). An earlier
  working hypothesis — that the inversion was a logits-vs-softmax artifact —
  was tested (both spaces recorded) and **rejected**.

## Hypotheses

- **Early-layer kurtosis inversion:** most plausibly the early-layer lens
  artifact the paper itself flags ("early-layer readouts are noisy and the
  onset boundary may partly be a lens artifact", noted at
  `[kb/notes/interpretability/j-space.md]` §2 citing §4.1): early Qwen
  readouts concentrate spuriously on a few high-norm directions
  (the multilingual-junk tokens visible in the trajectory tables), which
  *raises* kurtosis without carrying content. Under this reading the
  paper's "near-zero early" may itself be Claude-specific readout
  cleanliness, not workspace absence. [SPECULATION]
- **7B U-shape vs 1.5B hump:** either a genuine scale difference in where
  J-space content lives, or the H1 fit-budget confound depressing 7B
  mid-layer varfrac (n=100 under-determines 3584² per layer). The n=500
  refit (deferred, README list) would discriminate.
- **Cap-pinned active counts:** τ is doing no work at 1e-3; a
  gain/coefficient-mass threshold (e.g. atoms covering 90% of coefficient
  ℓ1) would give a scale-comparable "meaningfully active" count.

## Follow-ups

- Stage 5 (functional signatures) proceeds on the n=100 lenses; stage 6
  NLA cross-tie targets L20 — note 7B L20 varfrac = 0.032, on the rising
  limb into the late band.
- Active-count recalibration (coefficient-mass threshold) — cheap, pure
  post-processing of the committed per-prompt coefficients.
- Promote the six small derived metric artifacts (2× lens_eval, 2×
  readout_scan, 2× structure_scan; ~1.5 MB total) from `cache/` to
  `data/` + MANIFEST so `jspace_audit_findings.py` runs from a clean clone
  — currently it needs the gitignored cache. Arc-close (stage 7) hygiene.

## References

- `[gurnee2026-workspace §2.3, §4.1–4.2, footnote 4]`;
  `kb/excerpts/gurnee2026-workspace#sec-2-3-gradient-pursuit`,
  `#sec-4-1-kurtosis-onset`
- Prior: `2026-07-20-scale-comparison-7b-vs-1p5b-h2.md` (H1 confound),
  `2026-07-18-intermediate-concept-evals-h3-confirmed.md`
