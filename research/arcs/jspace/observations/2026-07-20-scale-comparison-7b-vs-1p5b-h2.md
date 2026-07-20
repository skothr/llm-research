# Observation: 7B vs 1.5B — J-lens advantage is scale-robust, not scale-growing (H2)

**Date/context:** 2026-07-20. Stage 3 scale comparison (H2 from the two
2026-07-18 observations). Same scripts (`jspace_readout_scan.py`,
`jspace_lens_eval.py`), same 12 held-out prompts / multihop+association eval
sets, now on `jlens_qwen2.5-7b_nf4_n100` (7B nf4, dim_batch=2, lm_head
offloaded, 16.26 h fit) run on GPU. Both lenses are n=100, so **scale (H2)
and lens-fit-budget (H1) are confounded here** — flagged throughout.

## Finding

The J-lens's intermediate-concept advantage **persists at 7B at nearly
identical overall magnitude but does not amplify**, and its early/mid-band
absolute strength is *lower* at 7B — the opposite of a clean scale-growth
story, and consistent with n=100 under-determining the larger model's J_l
(3584² vs 1536² per layer from the same 100 prompts).

**Multihop (intermediate-concept rank; the paper-native metric):**

| band | 7B J@10 | 7B logit@10 | 7B Δ | | 1.5B J@10 | 1.5B logit@10 | 1.5B Δ |
|---|--:|--:|--:|--|--:|--:|--:|
| early 0-8 | 0.09 | 0.01 | +0.08 | | 0.20 | 0.00 | +0.20 |
| mid 9-18 | 0.07 | 0.01 | +0.06 | | 0.13 | 0.00 | +0.13 |
| late 19-26 | 0.45 | 0.35 | +0.10 | | 0.50 | 0.40 | +0.10 |
| overall | 0.53 | 0.36 | **+0.17** | | 0.58 | 0.40 | **+0.18** |

Overall advantage is scale-robust (+0.17 vs +0.18); the early/mid J-exclusive
signal is real at both scales (logit ~0.01/0.01) but *weaker* at 7B
(0.09/0.07 vs 0.20/0.13). Association stays floored at both scales
(7B overall J@10 0.01, J@50 0.07; marginally above 1.5B's 0.00/0.02).

**Depth-of-emergence — the one clean scale-dependent reversal.** Median
layer at which the model's final top-1 enters the lens top-10:

| | 7B J-lens | 7B logit | | 1.5B J-lens | 1.5B logit |
|---|--:|--:|--|--:|--:|
| all positions | **22.0** | 24.0 | | 23.0 | **19.0** |
| last position | **22.0** | 26.0 | | 20.0 | **19.0** |

At 1.5B the logit lens surfaced the eventual output earlier; **at 7B the
J-lens does** (22 vs 24-26). This is the first metric on which the J-lens
overtakes the logit lens at scale, and it is not an artifact of the
intermediate-concept scoring — it is on the same output-predictive quantity
where the logit lens won at 1.5B. (J-lens still never-emerges in more cells:
14 vs 8 of 108.)

**Output-predictive Spearman:** logit lens wins at both scales; at 7B the
J-lens is *negative* in early layers (layers 0-8: J −0.14 to −0.09 vs logit
+0.05 to +0.08), crossing above the logit lens only at layer 26 (+0.068).
Same qualitative story as 1.5B, sharper.

## Evidence

Multihop example (7B): 'carnival-ocean' → 'Brazil' surfaces at J-lens L25
rank 41 vs logit L26 rank 85 (at 1.5B this was J L15 rank 32 — the concept
emerges *later in depth* at 7B under the n=100 lens, another under-fit-vs-
scale ambiguity). Trajectory (7B, prompt 1, final position) shows the same
"cleaner early" pattern as 1.5B — J-lens early layers are whitespace/newline
(contentless), logit-lens early layers are multilingual junk (`醺`,
`换句话`, `℠`, `-strokes`); both converge to `Answer`/`Based`/`Output` by
layer 26.

Derived artifacts (gitignored cache):
`readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`,
`lens_eval_qwen2.5-7b_nf4_n100.pt`. Fit: 9.5 s/prompt scan on GPU;
multihop 65 s, association 74 s. (The 7B eval initially clobbered the
1.5B eval artifact — the eval script's `--out` default ignored `--model`;
fixed to auto-name by lens stem, both artifacts regenerated cleanly.)

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt
python examples/jspace_readout_scan.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --device cuda --lens $L --n-prompts 12
python examples/jspace_lens_eval.py --evals multihop association \
    --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --device cuda --lens $L
```

## Hypotheses

- **H2 verdict (partial):** the J-lens intermediate-concept advantage is
  scale-robust, not scale-growing, at matched n=100. The depth-of-emergence
  reversal (J-lens earlier at 7B) is the one signal that strengthens with
  scale and is the most paper-consistent result so far
  `[gurnee2026-workspace §2.1]`.
- **H1 confound (now load-bearing):** 7B's weaker early/mid absolute rates
  and the L15→L25 concept-emergence shift both point at n=100 under-fitting
  the 2.3x-larger d_model. Distinguishing test: refit 7B (and/or 1.5B) at
  n=500-1000 and re-run; if 7B early/mid rates rise toward the 1.5B levels
  and the emergence layer moves earlier, the gap was fit-budget, not scale.
  This is the single most informative next artifact, but it is a ~29 h GPU
  fit at n=500 — worth a go/no-go with the reviewer.

## Follow-ups

- Reviewer decision: n=500 refit (7B ~29 h, 1.5B ~16 h) to break the H1/H2
  confound, vs proceeding to stage 4 (J-space structure) on the n=100 lenses.
- Association behavioral baseline still pending (does either model do the
  vignette→concept task at all).
- Stage 4 (gradient-pursuit J-space variance/sparsity depth-band map) can
  run on the current lenses regardless — it characterizes structure, not
  the concept-surfacing advantage.

## References

- `[gurnee2026-workspace §2.1, §4.1]`
- Prior: `2026-07-18-readout-scan-1p5b-first-pass.md`,
  `2026-07-18-intermediate-concept-evals-h3-confirmed.md`,
  `2026-07-18-fit-cost-calibration.md`
