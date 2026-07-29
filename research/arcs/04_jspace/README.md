# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-1.5B/7B-Instruct

**Research question:** Does the J-space phenomenon reported for Claude-family
models `[gurnee2026-workspace]` — a sparse, low-variance, causally privileged
band of verbalizable representations — replicate on Qwen2.5-Instruct
(`Qwen/Qwen2.5-1.5B-Instruct` in bf16 as the clean-gradient primary model, and
`Qwen/Qwen2.5-7B-Instruct` in nf4 as the VRAM-gated scale check), and how do
J-lens readouts at layer 20 relate to the NLA verbalizer readouts studied in
`research/arcs/01_nla-verbalizer/`?

**Status (2026-07-22): closed** — ran to planned completion; results
unreviewed and unreplicated outside this repo. Stages 1–6 including 5.2
(entailed-property swaps) at both scales, the full robustness battery
(corpus, quantization, n-budget, held-out sample), and stage-7
audit/synthesis. 5.3 modulation descoped. Synthesis below. Design plan
(signed off 2026-07-18):
`plans/2026-07-18-jspace-design.md`; addenda: `plans/2026-07-20-stage5-design.md`,
`plans/2026-07-20-stage6-design.md`, `plans/2026-07-21-stage52-entailed-property.md`.
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
- `2026-07-20-jspace-structure-stage4.md` — J-space occupancy is low
  (1.5B humped, 7B U-shaped, ~3× lower at 7B in absolute varfrac); the
  kurtosis workspace-onset signature is inverted on Qwen (verified on the
  paper-native metric, robust across logit/prob space). Its original
  "k-dependent ceiling" comparison-to-paper framing is **superseded** —
  the ceiling verdicts are certified on the paper's own metric in
  `2026-07-24-paper-metric-varfrac-recompute.md` (see the observation's
  read-first addendum).
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
- `2026-07-21-nla-crosstie-stage6.md` — stage 6 (novel): weak but
  prompt-specific J-lens↔NLA agreement on concept prompts
  (null-certified); the NLA-verbalizable content lives in the residual,
  not the J-space component (removal-damage ≈ random); the NLA capture
  layer (L20) sits below the 7B J-lens legibility onset (~L22) — a
  cross-arc architectural finding.
- `2026-07-21-quantization-exonerated-1p5b-nf4.md` — 1.5B nf4 control:
  bf16→nf4 at fixed model/corpus/n moves nothing (L21 peak 0.125 vs
  0.124); the 7B 3× gap is fit-budget and/or scale, not quantization.
  Bonus: nf4 backward is 2.97× cheaper → 1.5B n=500 refit is now ~5 h.
- `2026-07-22-n500-and-heldout-robustness.md` — the last two axes close:
  n=500 refit leaves the profile unchanged (peak −1.4%; H1 exonerated
  under the adopted 20% stability threshold) and the diversified C4
  held-out set leaves
  the depth-profile statistics intact — **the 7B gap is genuine scale.**
- `2026-07-24-paper-metric-varfrac-recompute.md` — post-close vetting
  (issue #26): the paper's 10% ceiling is excess-over-random
  orthogonal-projection FVE, not the scans' absolute varfrac — recomputed
  under the paper's definition **both ceiling verdicts survive** (1.5B
  L21 excess 11.2% CI [11.0, 11.4], 7B peak 4.7%) and hold on **all four
  robustness axes** re-run in that metric; stage-5.2 gap
  significance-certified (exact permutation) and the stage-5.1b 59%-tier
  **not detected at either scale** by exact McNemar — undetected rather than
  certified-null: 1.5B p=0.375 on 5 discordants, and the 7B row has 0/0
  discordants, where p=1.0 is an identity carrying no power; pursuit norm-bias
  bounded at both scales (early band contaminated — 7B via undertrained
  junk-token atoms, 1.5B via tied-embedding norms — workspace band clean).
- `2026-07-22-entailed-property-swaps-stage52.md` — stage 5.2
  (spider→ant replication): no discrete top-1 property flip at 1.5B, but
  a large J-lens-SPECIFIC graded effect (+5.17 nats on the unspoken
  entailed property at L18 on the J-lens-detected-position subset, mixed
  +2.13; a ~+5 nat gap over the equal-magnitude token-steering control) —
  relational structure token steering cannot produce;
  **revises the 5.1b token-steering-favored conclusion**. 7B chat run
  landed 2026-07-22 (see 7B replication section).

Load-bearing numbers re-derive from artifacts via
`examples/jspace_audit_findings.py` (450 checks at arc close; 739 after
Check M — the issue-#26 metric-correction battery: paper-metric ceiling
+ the four 1.5B robustness axes and the 7B held-out set, swap
significance + 5.1b McNemar tiers, and the
two-scale norm-bias pins — landed 2026-07-24; 978 after the final-review
pins landed 2026-07-25: full stage-4 depth-table cells both scales, the
naive-vs-paper delta decomposition, K_median_occ, the exact McNemar
p-value, and the MANIFEST census). All small derived artifacts (44 files,
~55 MB incl. the ten metric-correction artifacts) are LFS-committed
under `data/` and MANIFEST-registered (sha256), so **checks B–M run from
a clean clone**; check A and the lens-integrity blocks read the full
fitted lenses, which stay cache-only per Decision 4 (committed layer
subsets + `jspace_fit_lens.py` regenerate them).

**Expected result on a clean clone** (no local `cache/`, verified
2026-07-28): `SUMMARY: 920 PASS | 11 FAIL`, exit code 1. All 11 failures are
the designed `MISSING` reports for the five cache-only fitted lenses and their
`.config.json` sidecars — *not* regressions. The 978-check figure above
requires the local cache, where check A and the refit blocks in H/I/J also
run. Any FAIL naming something other than a `jlens_*.pt` / `jlens_*.config.json`
artifact is a genuine regression.

The jlens dependency is pinned in
the MANIFEST (`581d3986`, "Initial release" 2026-07-02 — the multihop/
association eval sets live in that clone). The harness was seeded at stage
3 rather than at arc close (`6d567a27`, together with the Decision-4 lens
layer subsets), on the project owner's standing requirement that
discovered defects enter the permanent record rather than being fixed
silently (2026-07-20 / 2026-07-21 directions); the bulk derived-artifact
promotion (34 files) landed with the stage-7 consolidation (`850b5173`);
the ~29 h → ~81 h refit-estimate correction is recorded in
place with the original preserved for the same reason.

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

1. ~~**n=500 lens refit**~~ — **RESOLVED 2026-07-22**: run at 1.5B nf4
   (~5.1 h); varfrac n-stable 100→500 (peak −1.4%, under the adopted 20%
   stability threshold) → H1 exonerated, the 7B gap is genuine scale. See
   `observations/2026-07-22-n500-and-heldout-robustness.md`. The direct
   7B n=500 (~81 h) remains unjustified absent any instability signal.
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

**Attribution:** research direction (target model, replication goal),
design sign-off with two amendments (the OOM gate and the comms protocol),
the stage-boundary go/no-go calls and open-decision calls, the reviewer
verdicts, the arc's auditability standard (defects into the permanent
record; datasets complete and clean-clone reproducible mid-arc), the
standing requirement that unattended runs be verified by their artifacts
rather than by process checks or a prior session's claim (the source of
the post-hoc audit and the comms/monitoring amendment), and the
review-before-integration policy — the pre-merge review standard set at
research integrity, auditability and attribution rather than code
correctness, with the merge gate reserving integration for a human —
human; paper digestion, KB grounding, drafting the experiment design,
implementation, and execution — Claude Code sessions (2026-07-18 →
2026-07-24). The post-close metric-fidelity reopening (issue #26,
2026-07-23/24) was human-directed: the reviewer reopened the closed arc to
vet the math, required the variance fraction be checked against the
paper's definition in the source PDF, and set the pass/fail evidentiary
bar the recompute was held to.

## Synthesis (arc close, 2026-07-22)

**Question answered:** the J-space phenomenon *partially* replicates on
Qwen2.5, splitting cleanly into what transfers and what does not.

**What replicates.**
1. **The J-lens as a readout instrument.** It surfaces unspoken
   intermediate concepts where the logit lens finds exactly nothing
   (multihop early/mid bands, J-exclusive on two fitting corpora); the
   "unspoken words" trajectories render legibly; depth-of-emergence
   reverses pro-J-lens at 7B on the median-layer metric. Counterweight,
   stated honestly: the logit median is still earlier at 1.5B (19 vs 23),
   and on the companion never-emergence count the J-lens fails to surface
   the token in *more* cells at both scales (1.5B 16/108 vs logit 1/108;
   7B 14/108 vs 8/108) — the 7B median flip is one favorable sub-metric,
   not a clean sweep. The J-exclusivity is corpus-, budget-, and
   quantization-robust; only the @10 magnitude is corpus-dependent.
2. **Low J-space occupancy; the 1.5B hump breaches the paper's 10%
   ceiling and 7B stays under — verified on the paper's own metric
   (2026-07-24).** The paper's ceiling is excess-over-random
   orthogonal-projection FVE at K = median occupancy
   `[gurnee2026-workspace §4.2 Fig 30b, §A.8]`, not the absolute
   reconstruction-energy varfrac the scans record; recomputed under that
   definition (K-consistent selection, 2026-07-25 F01 fix), 1.5B breaches
   at the hump (**L21 excess 11.15%, CI95
   [10.95, 11.40]** over all valid positions, 2000/2000 cluster-bootstrap
   resamples above 10%; naive metric: 12.4% at k=25) and 7B stays under
   (**peak excess 4.72%** at L22–L23; naive ≤5.8% through k=50) —
   cross-scale gap ~2.4×. The workspace-like mid-late band (1.5B peak L21) is
   invariant to fitting corpus (peak layer and value identical to 3dp:
   L21, 0.124 on both lenses), n-budget (n=100→500:
   −1.4%), quantization (bf16↔nf4: +1.2%), and held-out sample — and the
   invariance was re-verified 2026-07-24/25 **under the paper metric on
   all four axes** (L21 excess 10.7–11.7%, bootstrap-unanimous breach on
   each; 7B held-out band peak 6.0%, unanimous under). Caveat: early-band
   (L0–L16)
   occupancy/top-atom readings are partly norm-driven (unnormalized-atom
   pursuit selection bias; the workspace band measures norm-neutral) —
   see `observations/2026-07-24-paper-metric-varfrac-recompute.md`.
3. **Relational causality — the arc's strongest positive.** Swapping an
   unspoken concept along its J-lens vector, at the genuinely
   J-lens-detected concept positions, moves the concept's *entailed
   property* (spider→ant ⇒ 8→6-legs direction) far more than an
   equal-magnitude logit-lens token-steering control, which carries no
   property knowledge. Load-bearing figure — the absolute jlens−logitlens
   gap: **+5.0 nats** (1.5B L18: jlens +5.17 vs logit +0.15, per-item
   SD 4.9, n=7) / **+1.9 nats** (7B L19: jlens +2.17 vs logit +0.27,
   SD 4.1, n=17) on the auto-detected subset; mixed-scope gaps +2.0 /
   +1.1 nats. (The item-level SD exceeds the mean — small n, a few
   high-movement items dominate — but the paired gaps are
   significance-certified as of 2026-07-24: 1.5B auto 7/7 positive,
   exact sign-flip p=0.0156, the smallest p an n=7 test can produce;
   7B auto 15/17, p=0.0001; `examples/jspace_swap_significance.py`,
   derived from the committed swap artifacts.) The multiplier framing
   (~34×/8× auto-only, ~30×/7×
   mixed) is denominator-fragile — the control sits near zero — and the
   apparent cross-scale gap in the multiplier is mostly a control-
   denominator artifact, not a signal difference; the absolute gap is
   the figure to cite. The effect peaks a few layers *below* the
   report-swap layer at both scales: relational consequences engage
   earlier in depth than verbalization.

**What does not replicate.**
1. **The discrete entailed-property flip** (the paper's 8→6): flip rate
   0.000 at both scales, robust to the paper's verbatim prompt and to
   all-position editing — the graded effect exists, the top-1 crossing
   does not, at ≤7B.
2. **J-space *membership* as the causally privileged ingredient**: the
   J-space component of activation-derived concept vectors shows **no
   detectable effect** in report swaps — the paper's 59% tier is
   indistinguishable from random at both scales, though by tests with
   little power to separate them (1.5B p=0.375 on 5 discordants; 7B 0/0
   discordants, where McNemar has none). Read as failure to detect, not as
   demonstrated inertness. The NLA verbalizer's content likewise lives in
   the residual, not the J-space component (removal-damage ≈ random
   removal). Report-token swap effects are largely reachable by raw token
   steering.
3. **The kurtosis workspace-onset signature**: inverted on Qwen
   (high-early → mid-trough → weak late rise), on the paper-native
   metric, robust across logit/prob space and both corpora.
4. **7B structure diverges from the small-model picture**: ~3× lower
   occupancy and a U-shaped depth profile — established as genuine
   scale/model properties after the four-axis exoneration.

**Reconciled picture.** On open models at this scale, the J_ℓ pullback
is real and useful: it decodes held concepts token-indexed, and it
carries *relational* structure that token identity cannot supply. What
fails to transfer is the stronger claim that a sparse J-space *subspace*
is the privileged causal locus — component-level interventions behave
like noise, verbalizable content sits in the residual, and discrete
behavioral flips don't occur. Whether that gap is capability scale
(Claude 4.5-family vs ≤7B) or model family remains the open question
the arc cannot answer from below.

**Novel contributions beyond the paper:** the NLA cross-tie (two
independent verbalization channels agree weakly and prompt-specifically;
the NLA capture layer sits below the 7B J-lens legibility onset — a
cross-arc architectural fact); the L18/L19-vs-L21/L22 depth split
between relational and report effects; the kurtosis inversion; the
four-axis robustness methodology — one axis (the quantization control)
genuinely pre-registered in the design plan (§1), the others gated on
stability thresholds fixed in the session record before each run; and the
post-close metric-fidelity pass (issue #26, 2026-07-24): the ceiling
verdicts restated and confirmed under the paper's actual excess-FVE
definition, the swap gaps significance-certified by exact permutation,
and the pursuit's unnormalized-atom norm bias quantified — a choice the
paper and companion repo leave unspecified.

**Limitations.** One model family; n=100 lenses (n-stability shown at
1.5B only); 7B fitted in nf4 (exonerated at 1.5B, untested at 7B-bf16
which is infeasible here); single-token concept limit throughout;
30-prompt held-out sets (7B varfrac quoted band-level); the entailed
flip threshold is bounded below only; stage-6 inherits the NLA arc's
unaudited AV format bias (mitigated by nulls, not eliminated).

**Next paths.** Dose-response: strength sweep at the L18/L19 peak;
mid-scale (14B/32B) replication of the discrete flip; richer
concept-vector constructions for the 59%-tier question; an AV trained
at L22+ (above the legibility onset) for a clean cross-tie; multilingual
eval sets; verbal-report stage 5.3 modulation if the arc reopens.
