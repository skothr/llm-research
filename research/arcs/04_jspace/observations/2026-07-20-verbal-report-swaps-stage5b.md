# Observation: 7B swap effect emerges with clean prompts, but the paper's concept-vector middle tier does not replicate (stage 5.1b)

**Date/context:** 2026-07-20, follow-up to
`2026-07-20-verbal-report-swaps-stage5.md` resolving its two confounds:
(1) chat-template prompt style with an explicit one-word constraint
(`--prompt-style chat`; templates in design addendum §8) to fix 7B
fragment reports; (2) activation-contrastive concept vectors (mean h_L
over paraphrase templates minus category-baseline mean) adding the
paper's two component conditions — `jspace_comp` (k=25 gradient-pursuit
J-space component of the concept vector) and `nonjspace_comp` (its
residual) — completing the paper's 88/59/5 triplet
`[gurnee2026-workspace §3.1, Fig. 8]`. Six conditions × s∈{1,2}, both
models, equalized injected L2 — exact by construction for the five
J-lens-source conditions (per-item spread <0.04%, audit Check F);
**logitlens is the exception**: its per-item injected norm deviates from
the jlens reference by up to 17.7% (1.5B) / 78.2% (7B) (bf16
fp32-capture-vs-generate-prefill measurement artifact on its small `w_s`
projection; design §8). Mean-level norms are close, but the
jlens-vs-logitlens comparison is not clean equal-magnitude per item —
the token-steering conclusion below is qualitative, not
magnitude-controlled to the same standard as the other five conditions.

## Finding

**Swap-target top-5 entry @ s=2 (all 78 items), chat style:**

| condition | 1.5B | 7B | paper (Claude 4.5) |
|---|--:|--:|--:|
| jlens (J-lens vector) | **0.628** | **0.269** | 0.88 |
| logitlens (token steering) | 0.564 | 0.244 | — (control) |
| nonjspace (w_t proxy) | 0.308 | 0.179 | — |
| nonjspace_comp (concept residual) | 0.218 | 0.167 | 0.05 |
| jspace_comp (concept J-space comp.) | 0.179 | 0.154 | **0.59** |
| random (equal-magnitude) | 0.141 | 0.154 | — |

**1. The 7B null partially un-confounds.** Chat prompts cut the 7B
fragment-report rate 0.308 → 0.192 and raised predicts-report retention
0.154 → 0.269; with clean sources the 7B J-lens swap now clearly beats
random (0.269 vs 0.154, +0.115 — the plain-style run had +0.04 ≈ noise).
The stage-5.1 "7B near-null" was substantially a compliance artifact.

**2. The token-steering confound persists and sharpens.** jlens beats
logitlens by only +0.026 at 7B and +0.064 at 1.5B. At these scales, most
of the swap effect is reachable by raw unembedding steering
`[sources/forums/2026-07-06-nanda-workspace-review.md]`.

**3. The paper's middle tier does not replicate — a clean null.**
`nonjspace_comp` collapses toward random on both models (paper-consistent:
its row is 5%). But `jspace_comp` — the paper's 59% row, its strongest
evidence that J-space *membership* rather than token identity carries the
causal effect — **also collapses to random** (7B: 0.154 = random exactly;
1.5B: 0.179 vs random 0.141). On both open models, only the token-indexed
J-lens vector is an effective swap direction; the J-space component of an
activation-derived concept vector is inert.

**Interpretation.** Points 2 + 3 together shift the weight of evidence at
open-model scale toward the token-steering account: what makes a swap
work here is the token-indexed direction (unembedding row, refined
modestly by the J_ℓ pullback at 1.5B), not J-space membership of
activation components. This does not contradict the paper's Claude-family
results; it fails to extend its causal-structure claim to Qwen at 1.5B/7B
with n=100 lenses. Alternatives not excluded: (a) the activation-derived
concept vector may be a poor stand-in for the paper's concept vectors
(construction differs; ours is contrastive over 2–3 paraphrases); (b) the
n=100 lens under-fit (H1) degrades the pursuit decomposition the
component conditions depend on. [INFERENCE with named alternatives]

## Evidence

Artifacts (gitignored cache, distinct names — committed plain-4c
artifacts verified intact):
`verbal_report_chat_6c_qwen2.5-{1.5b,7b}-instruct_jlens_*.pt`, 78 items,
full transcripts + norms per item. Runs exit 0 (7B in background: model
load alone exceeds the 600 s tool cap). Qualitative: 7B fruit
Apple→Orange jlens@2 reports 'Orange' (rank 0) while jspace_comp and
random both report 'Banana' — the component condition behaves like noise.
Audit Check F pins the tables, gap/ordering booleans, and the per-item
norm-ratio invariant.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_verbal_report.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --device cuda --layer 22 --prompt-style chat --contrastive \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt
python examples/jspace_verbal_report.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --device cuda --layer 21 --prompt-style chat --contrastive \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt
```

## Hypotheses

- **Concept-vector-construction hypothesis:** the paper's 59% row may
  depend on how its concept vectors are built (from richer activation
  statistics than our 2–3 paraphrase contrastive mean). Test: vary the
  construction (more paraphrases, position averaging, per-instance
  baselines) and see if jspace_comp lifts off random.
- **Lens-quality hypothesis (H1 again):** the pursuit decomposition
  inherits n=100 lens noise; an n=500 lens (deferred bin) would sharpen
  the component split. The jlens condition's robustness argues this isn't
  the whole story — the same lens's token-indexed rows work.
- **Scale hypothesis:** J-space causal structure beyond token steering may
  emerge only above some capability threshold between 7B and Claude-scale.
  Untestable here; honest bound on the replication scope.

## Follow-ups

- Fold into arc-close synthesis as the arc's central causal result: the
  paper's readout claims replicate (H3, stage 3/4 structure), its causal
  J-space-membership claim does not at this scale.
- Concept-vector construction variants (cheap, scripted) if the middle
  tier is worth one more attempt before arc close.
- Fix the logitlens norm measurement (compute its scale from the same
  generate-prefill forward used for injection) and rerun that condition
  only, to make the token-steering comparison magnitude-controlled.
- Stage 5.2 two-hop demoted accordingly (Decision 3 staging): run only if
  its precondition check (7B two-hop accuracy) passes and time permits.

## Qualifier (added 2026-07-22)

Stage 5.2 (`2026-07-22-entailed-property-swaps-stage52.md`) bounds the
token-steering interpretation this observation favored: on *entailed
properties* (which token identity cannot supply), the J-lens swap moves
the unspoken property by +2.13 nats vs +0.07 for logit-lens steering at
equal magnitude. The token-steering account stands for report-token
effects; it does not extend to relational effects. Also: the 17.7%
logitlens norm deviation reported above traces to a single degenerate
fragment-source item (the norm-fix rerun reproduced all rates
identically at 1.5B). The 7B magnitude-controlled rerun also landed
(2026-07-22): all rates reproduce identically (jlens@2 0.269,
logitlens@2 0.244) with the median per-item norm ratio now 1.0008 —
so the observation's tables and the audit pins stand unchanged, and
the magnitude caveat reduces to a handful of degenerate
fragment-source items at both scales.

## References

- `[gurnee2026-workspace §3.1, Fig. 8]`
- `sources/forums/2026-07-06-nanda-workspace-review.md` (tier B; control
  design and the interpretation this result now supports at open scale)
- Prior: `2026-07-20-verbal-report-swaps-stage5.md`,
  `plans/2026-07-20-stage5-design.md` §8,
  `2026-07-20-jspace-structure-stage4.md`
