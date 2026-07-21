# Observation: verbal-report swaps — J-space-specific causality replicates at 1.5B (attenuated), near-null at 7B (stage 5.1)

**Date/context:** 2026-07-20. Stage 5.1 per the decided design addendum
(`plans/2026-07-20-stage5-design.md`): "Think of a {category}" → report;
swap injection at the report position, at each model's stage-4 varfrac-peak
layer (1.5B L21, 7B L22); four conditions at equalized injected L2
(J-lens vector swap; non-J-space component of w_t [proxy DEVIATION, see
design §7.4]; seeded random vector; logit-lens unembedding steering — the
Nanda token-steering control); strengths s∈{1,2}; 26 categories × 3
seeded random-sibling targets = 78 items. Script:
`examples/jspace_verbal_report.py`. Runtimes: 1.5B ~95 s (43 tok/s); 7B
~10 min (18 tok/s, load-dominated).

## Finding

**At 1.5B the paper's causal ordering replicates with attenuation; at 7B
(nf4, n=100 lens) the effect is a near-null with no condition
separation.**

**1.5B (L21), swap-target top-5 entry rate over all 78 items:**

| condition | s=1 | s=2 | | condition | s=1 | s=2 |
|---|--:|--:|--|---|--:|--:|
| **jlens** | **0.269** | **0.551** | | nonjspace | 0.167 | 0.269 |
| logitlens | 0.244 | 0.436 | | random | 0.103 | 0.103 |

Ordering **jlens > nonjspace > random ≈ chance** matches the paper's
structure (Claude 4.5-family reference: 88% / 59% / 5%
`[gurnee2026-workspace §3.1, Fig. 8]`) at roughly attenuated-by-⅓-to-½
magnitudes — consistent with the 20× scale reduction. The equal-magnitude
random control stays flat across strengths (0.103/0.103) while jlens
doubles (0.269→0.551): the effect is direction-specific, not
injection-energy. Report-flip evidence: 'Soccer'→'Cycling' jlens@2 flips
the report to 'Cycling' (target rank 3); the same-magnitude random
direction yields 'Baseball' (target rank 235).

**Nanda control:** logitlens tracks jlens closely (0.436 vs 0.551 @2) —
the token-steering confound is **non-trivial at this scale**; the J-lens
margin over raw unembedding steering is real but modest (+0.115 @2). Per
the design, this keeps the token-steering interpretation live as a partial
account rather than refuted `[sources/forums/2026-07-06-nanda-workspace-review.md]`.

**7B (L22): no separation.** jlens 0.192/0.192 (s=1/2), nonjspace
0.192/0.205, logitlens 0.167/0.179, random 0.154/0.154 — all within 0.05
of each other. Two visible contributors: (a) the 7B nf4 n=100 lens's
J-space signal is ~3× weaker than 1.5B's (stage-4 varfrac), compounding
quantization noise with the H1 fit-budget confound; (b) **baseline task
compliance is worse at 7B** — many baseline reports are fragments ('P',
'M', 'Viol'), giving degenerate source directions (illustrated by a
random-control trial that surfaced the target as readily as the J-lens
swap). This is a *near-null with identified confounds*, not clean evidence
against 7B workspace causality.

**Ruling-4 trigger fired:** nonjspace ≈ jlens at 7B is the pre-registered
condition (design §7.4) for escalating to activation-contrastive concept
vectors instead of the w_t proxy.

**Small-n caveat:** the J-lens-predicts-report baseline rates (0.115 at
1.5B, 0.154 at 7B) leave predicts-subset metrics at n=9 and n=12 — those
columns (top5_pred) are directional only and carry no weight here.

## Evidence

Artifacts (gitignored cache):
`verbal_report_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`,
`verbal_report_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` — 78
items each, full transcripts + J-lens top-10 + per-trial injected norms
stored per item (post-hoc aliasing possible without rerun). Injected-norm
equality verified in-run: spread < 0.01 within each model×strength
(1.5B 8.53/17.06; 7B 7.21/14.42). The 7B run was killed by a tool-side
600 s timeout *after* artifact save + summary print; artifact verified
complete (78 items). Audit Check E covers the table values, ordering
claims, and norm equality.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_verbal_report.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --device cuda --layer 21 \
    --lens research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt
python examples/jspace_verbal_report.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --device cuda --layer 22 \
    --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt
```
(Defaults: --target-mode random, strengths 1,2, last-position injection.)

## Hypotheses

- **1.5B replication is J-space-specific**: the jlens-over-nonjspace gap
  (0.551 vs 0.269 @2) at equalized magnitude is the paper's central
  §3.1 contrast, reproduced. The narrower jlens-over-logitlens gap says a
  substantial share of the effect is reachable by plain unembedding
  steering at this scale — the workspace framing survives, but at 1.5B
  the J-lens direction buys less over token steering than the paper's
  results imply for Claude-scale models. [INFERENCE from the two gaps]
- **7B near-null is confounded, not conclusive**: weak lens (H1) ×
  quantization × compliance failures. Discriminating fixes, in cost
  order: (i) instruction-format prompt variant to fix compliance (cheap,
  H4-adjacent); (ii) activation-contrastive concept vectors (ruling-4
  path, moderate); (iii) n=500 7B lens (expensive, already binned).

## Follow-ups

- 7B compliance fix (chat-template / few-shot prompt variant) then rerun —
  cheapest path to un-confounding the 7B null.
- Activation-contrastive concept vectors (ruling-4 trigger, both models).
- Stage 5.2 two-hop swap and 5.3 modulation remain (Decision 3 staging).
- Strength sweep beyond s=2 and injection-scope 'all' variant (flags
  already in the script).

## References

- `[gurnee2026-workspace §3.1, Fig. 8]`;
  `kb/notes/interpretability/j-space.md` §2 (functional-findings table)
- `sources/forums/2026-07-06-nanda-workspace-review.md` (token-steering
  critique; tier B — control design only)
- Prior: `2026-07-20-jspace-structure-stage4.md` (layer choice),
  `2026-07-20-corpus-sensitivity-c4-1p5b.md`,
  `plans/2026-07-20-stage5-design.md`
