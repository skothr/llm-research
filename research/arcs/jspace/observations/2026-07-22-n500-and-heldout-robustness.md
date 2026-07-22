# Observation: n-budget (H1) and held-out-sample axes both exonerated — the 7B gap is genuine scale (1.5B controls)

**Date/context:** 2026-07-22. The last two robustness axes on the arc's
structural claims, both run as isolate-one-axis controls at 1.5B:
(1) fit budget — n=500 nf4 refit (same corpus/model/quantization as the
n=100 nf4 lens; 5.25 h at the measured 36.3 s/prompt) + full metric
suite; (2) held-out activation sample — the original held-out set is 30
*consecutive* wikitext records (topically clustered around one
article; flagged in external review), so the scan suite was re-run
against a diversified seeded C4-en held-out set
(`heldout_prompts_c4en_n30.json`, offset-1000 of the seed-42 stream,
verified disjoint from both existing corpora), same wikitext-fit lens.

## Finding

**1. Fit budget (H1): exonerated by the pre-registered rule.** 5× more
fitting data leaves the 1.5B nf4 J-space profile unchanged — varfrac
peak L21 0.1252 → 0.1235 (−1.4% relative, threshold was 20%), profile
mean −2.9%, largest single-layer shift −9.5%, same single-humped shape.
With corpus (C4 check), quantization (nf4 control), and now n-budget all
individually isolated and individually null at 1.5B, **the 7B ~3× lower
occupancy and U-shaped depth profile stand as a genuine scale/model
property.** Residual caveat, stated honestly: these controls test each
axis at 1536² Jacobians; 3584² being data-starved at n=100 in a way
1536² is not remains logically open, but with zero instability signal on
any axis, the ~81 h direct 7B n=500 test stays unjustified.

**2. Held-out sample: structural conclusions robust.** Diversified
(C4) vs clustered (wikitext) held-out activations, same lens:
varfrac peak stays **L21** (0.124 → 0.133, +8%), L0 0.082 → 0.084,
kurtosis trough L18:0.93 → L17:1.00, depth-of-emergence medians within
±2 layers; J-lens readout fidelity marginally *better* on the diverse
set. The article-cluster narrowness shaped the qualitative trajectory
vocabulary (visible as criticism/reviews register in the trajectory
figure) but not the depth-profile statistics.

**7B held-out (gated runs, landed):** shape and ordering conclusions
hold — U-profile with late peak, still ~2.6× below the 1.5B level on
the matched C4 held-out set — but with honestly larger relative shifts
than at 1.5B: peak 0.0404@L22 → 0.0518@L23 (+28%, layer ±1), trough
0.0121@L17 → 0.0166@L16. At 7B's low absolute occupancy, per-sample
noise is a larger fraction of the signal; 7B varfrac *values* should be
quoted as band-level (~0.01–0.05), not to three decimals. The
structural claims (≤10% ceiling, U-shape, ~3× below 1.5B) are
unaffected.

**Probe status — abandoned per the two-failure rule.** The raw-VJP
bf16-vs-nf4 fidelity probe failed twice (first: HF_HUB_OFFLINE env bug,
fixed; second: a 6-second startup crash, undiagnosed). It was
record-keeping only — the fitted-lens agreement (quantization
observation) and the n-stability above are decisive without it. The
script stays committed, marked unvalidated.

## Evidence

Artifacts (gitignored cache): `jlens_qwen2.5-1.5b_nf4_n500.pt` (+ suite:
structure/lens_eval/readout, all rc=0, rich capture);
`{readout,structure}_scan_*1.5b*_heldoutc4en.pt`. Corpus committed:
`heldout_prompts_c4en_n30.json` (MANIFEST-registered, `--check` OK,
zero-overlap verified). Orchestrator log `n500_run.log`: fit rc=0
(18,313 s wall implied by 23:24 end), suite rc=0×3, probe rc=1.
4-way comparison table (bf16 / nf4-n100 / nf4-n500 / 7B) re-derivable
via the session's extraction script; headline rows in this observation.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode nf4 --dim-batch 8 --n-prompts 500 --device cuda
python examples/jspace_freeze_c4_corpus.py --offset 1000 --n 30 \
    --out research/arcs/jspace/data/heldout_prompts_c4en_n30.json
# scans: --prompts <c4 heldout> (outputs auto-tag _heldoutc4en; no clobber)
```

## Follow-ups

- 7B held-out scans via the VRAM gate; fold in before close.
- Stage-7 audit consolidation: check groups for n500, heldoutc4en,
  nf4-n100, and entailed artifacts in one pass.

## References

- Pre-registration: agent report 2026-07-21 (rule stated before data);
  prior: `2026-07-21-quantization-exonerated-1p5b-nf4.md`,
  `2026-07-20-corpus-sensitivity-c4-1p5b.md`,
  `2026-07-20-jspace-structure-stage4.md` (the claims these defend).
