# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-7B-Instruct

**Research question:** Does the J-space phenomenon reported for Claude-family
models `[gurnee2026-workspace]` — a sparse, low-variance, causally privileged
band of verbalizable representations — replicate on `Qwen/Qwen2.5-7B-Instruct`,
and how do J-lens readouts at layer 20 relate to the NLA verbalizer readouts
studied in `research/arcs/nla-verbalizer/`?

**Status (2026-07-20):** stages 1–4 complete at both scales (1.5B bf16
control + 7B nf4 target); stage 5.1 (verbal-report swaps) complete —
5.2/5.3 (two-hop, modulation) and stage 6 (NLA cross-tie) remain. Design
plan (signed off 2026-07-18): `plans/2026-07-18-jspace-design.md`; stage-5
design addendum: `plans/2026-07-20-stage5-design.md`.
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
- `2026-07-20-jspace-structure-stage4.md` — J-space occupancy replicates
  the paper's ≤10% ceiling (1.5B humped, 7B U-shaped, ~3× lower at 7B);
  the kurtosis workspace-onset signature is inverted on Qwen (verified on
  the paper-native metric, robust across logit/prob space).
- `2026-07-20-corpus-sensitivity-c4-1p5b.md` — seeded C4-en refit (1.5B):
  workspace-band metrics corpus-invariant (L21 peak identical), early band
  corpus-sensitive; qualifies the H3 @10 magnitude, wikitext stands for 7B.
- `2026-07-20-verbal-report-swaps-stage5.md` — stage 5.1: the paper's
  causal swap ordering (jlens > nonjspace > random) replicates at 1.5B
  with attenuation (0.55/0.27/0.10 @s=2 vs Claude's 88/59/5%); logit-lens
  steering control is non-trivial (0.44); 7B is a confounded near-null
  (weak nf4 lens + compliance failures) — ruling-4 follow-up queued.
- `2026-07-20-verbal-report-swaps-stage5b.md` — stage 5.1b: chat prompts
  un-confound the 7B null (jlens 0.269 > random 0.154), but the paper's
  59% middle tier (J-space component of concept vectors) collapses to
  random at BOTH scales — only token-indexed directions are causally
  effective here; weight shifts toward the token-steering account at
  open-model scale.

Load-bearing numbers re-derive from artifacts via
`examples/jspace_audit_findings.py`. Reduced layer subsets of both fitted
lenses are committed to LFS under `data/` (design-plan Decision 4); full
lenses are cache-only, regenerable via `examples/jspace_fit_lens.py`.

**Corpus provenance (Decision 1, recorded 2026-07-20):** the frozen
fitting corpus (`fitting_prompts_wikitext103_n1000.json`) replicates the
companion repo's own fitting-corpus selection —
`jlens.examples.load_wikitext_prompts` (Salesforce/wikitext,
wikitext-103-raw-v1, train, first-N ≥600 chars) — i.e. Decision 1's
primary branch ("reuse the companion repo's prompt sets; closest method
match"), per commit 982e061e. Caveat: wikitext-103 is English-only
Wikipedia register, narrower than the paper's "pretraining-like
distribution" phrasing and than Qwen2.5's multilingual training mix;
corpus sensitivity was checked 2026-07-20 (1.5B n=100 seeded-C4-en refit +
full metric suite): **workspace-band metrics corpus-invariant, early-band
(L0–L16) corpus-sensitive** — wikitext stands for the 7B lens; see
`observations/2026-07-20-corpus-sensitivity-c4-1p5b.md`.

## Deferred / follow-up directions

Binned here 2026-07-20 (reviewer call), roughly in decreasing
informativeness-per-hour:

1. **n=500 lens refit (7B ~81 h GPU at measured 585.4 s/prompt; 1.5B
   ~16 h)** — breaks the H1 (fit budget) / H2 (scale) confound flagged in
   the 2026-07-20 observation. Deferred for cost; the single most
   informative next artifact if the confound starts blocking conclusions.
2. ~~**Corpus-sensitivity check (seeded C4-en refit)**~~ — **RESOLVED
   2026-07-20**: 1.5B n=100 C4-en refit run; workspace band
   corpus-invariant (L21 varfrac peak identical at 0.124), early band
   corpus-sensitive; wikitext stands for 7B. See
   `observations/2026-07-20-corpus-sensitivity-c4-1p5b.md`. A 7B C4 refit
   is not justified by these results.
3. **Split-half lens stability at n=100** — the stage-2 validation gate was
   waived for the first pass; two disjoint n=50 fits per model would bound
   estimator noise (7B ~16 h; 1.5B ~3 h).
4. **Association behavioral baseline** — do the models do the
   vignette→concept task at all? Distinguishes capability floor from
   representation-format for the floored association evals (cheap, forward
   passes only).
5. **Chat-template prompt variant (H4)** — both eval suites re-run with the
   Qwen chat template applied, testing whether instruct-tune formatting
   moves the intermediate-concept rates.
6. **Remaining companion eval sets** (multilingual, poetry, order-ops,
   typo) — for the stage-4/5 writeups.

**Theory grounding:** `theory/kb/notes/interpretability/j-space.md`,
excerpts in `theory/kb/excerpts/gurnee2026-workspace.md`, archived paper PDF
in `theory/sources/papers/gurnee2026-workspace_verbalizable-global-workspace.pdf`.

**Attribution:** research direction (target model, replication goal) — human;
paper digestion, KB grounding, experiment design — Claude (Fable 5 session
2026-07-18, with opus/sonnet subagents).

Findings, limitations, and next paths will be synthesized here at arc close
per `research/ARC_PROCESS.md` step 6.
