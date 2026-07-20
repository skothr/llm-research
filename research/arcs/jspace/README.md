# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-7B-Instruct

**Research question:** Does the J-space phenomenon reported for Claude-family
models `[gurnee2026-workspace]` — a sparse, low-variance, causally privileged
band of verbalizable representations — replicate on `Qwen/Qwen2.5-7B-Instruct`,
and how do J-lens readouts at layer 20 relate to the NLA verbalizer readouts
studied in `research/arcs/nla-verbalizer/`?

**Status (2026-07-20):** stages 1–3 complete at both scales (1.5B bf16
control + 7B nf4 target); stage 4 (J-space structure) in progress. Design
plan (signed off 2026-07-18): `plans/2026-07-18-jspace-design.md`.
Observations so far, in `observations/`:

- `2026-07-18-fit-cost-calibration.md` — fitting cost is structural
  (`ceil(d_model/dim_batch)` backwards/prompt); measured 115 s/prompt (1.5B
  bf16) and 559–585 s/prompt (7B nf4 + lm_head offload).
- `2026-07-18-readout-scan-1p5b-first-pass.md` — on output-predictive
  metrics the logit lens wins ("cleaner early, not earlier").
- `2026-07-18-intermediate-concept-evals-h3-confirmed.md` — first positive
  replication: J-lens surfaces unspoken intermediates where the logit lens
  finds nothing (early/mid bands), on the paper-native metric.
- `2026-07-20-scale-comparison-7b-vs-1p5b-h2.md` — advantage is
  scale-robust, not scale-growing; depth-of-emergence reverses pro-J-lens
  at 7B; includes the post-hoc artifact audit of the overnight 7B run.

Load-bearing numbers re-derive from artifacts via
`examples/jspace_audit_findings.py`. Reduced layer subsets of both fitted
lenses are committed to LFS under `data/` (design-plan Decision 4); full
lenses are cache-only, regenerable via `examples/jspace_fit_lens.py`.

## Deferred / follow-up directions

Binned here 2026-07-20 (reviewer call), roughly in decreasing
informativeness-per-hour:

1. **n=500 lens refit (7B ~81 h GPU at measured 585.4 s/prompt; 1.5B
   ~16 h)** — breaks the H1 (fit budget) / H2 (scale) confound flagged in
   the 2026-07-20 observation. Deferred for cost; the single most
   informative next artifact if the confound starts blocking conclusions.
2. **Split-half lens stability at n=100** — the stage-2 validation gate was
   waived for the first pass; two disjoint n=50 fits per model would bound
   estimator noise (7B ~16 h; 1.5B ~3 h).
3. **Association behavioral baseline** — do the models do the
   vignette→concept task at all? Distinguishes capability floor from
   representation-format for the floored association evals (cheap, forward
   passes only).
4. **Chat-template prompt variant (H4)** — both eval suites re-run with the
   Qwen chat template applied, testing whether instruct-tune formatting
   moves the intermediate-concept rates.
5. **Remaining companion eval sets** (multilingual, poetry, order-ops,
   typo) — for the stage-4/5 writeups.

**Theory grounding:** `theory/kb/notes/interpretability/j-space.md`,
excerpts in `theory/kb/excerpts/gurnee2026-workspace.md`, archived paper PDF
in `theory/sources/papers/gurnee2026-workspace_verbalizable-global-workspace.pdf`.

**Attribution:** research direction (target model, replication goal) — human;
paper digestion, KB grounding, experiment design — Claude (Fable 5 session
2026-07-18, with opus/sonnet subagents).

Findings, limitations, and next paths will be synthesized here at arc close
per `research/ARC_PROCESS.md` step 6.
