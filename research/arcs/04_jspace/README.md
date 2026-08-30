# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-1.5B/7B-Instruct

A replication arc: does the J-lens / J-space "verbalizable workspace"
phenomenon reported for Claude-family models `[gurnee2026-workspace]`
survive transfer to open Qwen2.5-Instruct models (1.5B bf16 primary, 7B
nf4 scale check)? Stages 1–7 ran to planned completion with a four-axis
robustness battery; the answer is a partial replication with a clean
causal split. A data correction — PII redaction of the arc's C4-en
corpus slice and a re-run of every dependent result — is documented in
[§ Data correction](#data-correction--c4-pii-redaction-and-re-run-closed-2026-08-16).

**Status (2026-07-22): closed** — ran to planned completion; results
unreviewed and unreplicated outside this repo. Stages 1–6 including 5.2
(entailed-property swaps) at both scales, the full robustness battery
(corpus, quantization, n-budget, held-out sample), and stage-7
audit/synthesis. 5.3 modulation descoped. Synthesis in § Findings
below. Design plan (signed off 2026-07-18):
`plans/2026-07-18-jspace-design.md`; addenda: `plans/2026-07-20-stage5-design.md`,
`plans/2026-07-20-stage6-design.md`, `plans/2026-07-21-stage52-entailed-property.md`.

## The question

Does the J-space phenomenon reported for Claude-family
models `[gurnee2026-workspace]` — a sparse, low-variance, causally privileged
band of verbalizable representations — replicate on Qwen2.5-Instruct
(`Qwen/Qwen2.5-1.5B-Instruct` in bf16 as the clean-gradient primary model, and
`Qwen/Qwen2.5-7B-Instruct` in nf4 as the VRAM-gated scale check), and how do
J-lens readouts at layer 20 relate to the NLA verbalizer readouts studied in
`research/arcs/01_nla-verbalizer/`?

## Findings — arc-close synthesis (2026-07-22)

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

   ![2026-07-21-jspace-emergence](observations/figures/2026-07-21-jspace-emergence.png)

   *Depth of emergence: the median source layer at which the final top-1 token
   enters the lens top-10 is later for the J-lens at 1.5B (L23 vs logit-lens
   L19) but earlier at 7B (L22 vs L24) — the scale-dependent reversal. The
   companion never-emerged counts run against the J-lens at both scales (1.5B
   16/108 vs logit 1/108; 7B 14/108 vs 8/108), and panel (a) shows the logit
   lens leading on output-predictive agreement at every layer except the final
   7B one.*
   ([provenance](observations/figures/INVENTORY.md))
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
   cross-scale gap ~2.4×. **Calibration caveat (2026-08-30, issue #79):**
   the excess metric is not dimensionless — signal and baseline share a
   common ~2× factor between d=1536 and d=3584 (their ratio is
   near-invariant: 6.29 vs 6.50), so the cross-scale gap and any
   breach-vs-paper reading are calibrated in a dimension-dependent unit,
   and the paper's baseline FVE is unpublished, so no conversion exists.
   Within-scale statements are unaffected. Derivation:
   [excess-FVE dimension dependence](observations/2026-08-30-excess-fve-dimension-dependence.md).
   The workspace-like mid-late band (1.5B peak L21) is
   invariant to fitting corpus (peak layer and value identical to 3dp:
   L21, 0.124 on both lenses), n-budget (n=100→500:
   −1.4%), quantization (bf16↔nf4: +1.2%), and held-out sample — and the
   invariance was re-verified 2026-07-24/25 **under the paper metric on
   all four axes** (L21 excess 10.7–11.7%, bootstrap-unanimous breach on
   each; 7B held-out band peak 5.88%, unanimous under) and re-derived
   2026-08-16 on the redacted C4 corpus, which left the range and the
   unanimity as they stand and moved only the 7B held-out band peak
   (5.98% → 5.88%). Caveat: early-band
   (L0–L16)
   occupancy/top-atom readings are partly norm-driven (unnormalized-atom
   pursuit selection bias; the workspace band measures norm-neutral) —
   see `observations/2026-07-24-paper-metric-varfrac-recompute.md`.

   ![2026-07-24-jspace-paper-metric-excess](observations/figures/2026-07-24-jspace-paper-metric-excess.png)

   *The 10% ceiling under the paper's own metric (excess-over-random
   orthogonal-projection FVE at K = median occupancy): 1.5B breaches at its L21
   hump (10.8% on the scan grid; all-positions mean 11.15%, cluster-bootstrap
   CI95 [10.95, 11.40], n=5362 positions/layer) while 7B stays under
   throughout, peaking at 4.7% (L22–L23). Faint dashed lines are the arc's
   original absolute varfrac@25 from the committed scans, shown to visualize
   the metric correction.*
   ([provenance](observations/figures/INVENTORY.md))
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

   ![2026-07-22-jspace-entailed-property](observations/figures/2026-07-22-jspace-entailed-property.png)

   *Swapping the unspoken concept along its J-lens vector moves the entailed
   property far more than the equal-magnitude logit-lens token-steering and
   random controls, which sit near zero: mean Δlog-p peaks at +5.17 nats at
   1.5B L18 (auto-detected subset, n=7/17) and +2.17 nats at 7B L19 (n=17/30).
   Both peaks sit a few layers below the report-swap layer, where the effect
   collapses (+0.44 at 1.5B L21, +0.63 at 7B L22); whiskers are seeded
   bootstrap 95% CIs of the auto-only mean, and the flip rate is 0.000
   everywhere.*
   ([provenance](observations/figures/INVENTORY.md))

**What does not replicate.**
1. **The discrete entailed-property flip** (the paper's 8→6): flip rate
   0.000 at both scales, robust to the paper's verbatim prompt and to
   all-position editing — the graded effect exists, the top-1 crossing
   does not, at ≤7B.
2. **J-space *membership* as the causally privileged ingredient**: the
   J-space component of activation-derived concept vectors shows **no
   detectable effect** in report swaps — the paper's 59% tier is
   indistinguishable from random at both scales, and at 7B the gap is
   bounded at ≤3.8pp (exact 95%, 0 discordant pairs of 78); 1.5B is looser
   (p=0.375 on 5 discordants). A tight measured bound, not a demonstration
   of exact zero. The NLA verbalizer's content likewise lives in
   the residual, not the J-space component (removal-damage ≈ random
   removal). Report-token swap effects are largely reachable by raw token
   steering.

   ![2026-07-21-jspace-swap-causality](observations/figures/2026-07-21-jspace-swap-causality.png)

   *Report-swap success by injected direction (n=78 items, chat prompts): the
   token-indexed J-lens vector replicates the paper's ordering with heavy
   attenuation (0.63 at 1.5B L21, 0.27 at 7B L22, against the paper's 88%), but
   the J-space component of the concept vector sits at chance (0.18 vs random
   0.14 at 1.5B; 0.15 vs 0.15 at 7B) where the paper reports 59%. Plain
   logit-lens token steering reaches 0.56 at 1.5B, the result that shifts
   weight toward the token-steering account; whiskers are Wilson 95%
   intervals.*
   ([provenance](observations/figures/INVENTORY.md))
3. **The kurtosis workspace-onset signature**: inverted on Qwen
   (high-early → mid-trough → weak late rise), on the paper-native
   metric, robust across logit/prob space and both corpora.

   ![2026-07-20-jspace-structure-depth-map](observations/figures/2026-07-20-jspace-structure-depth-map.png)

   *J-lens readout excess kurtosis on the paper-native metric (lower panel)
   runs backwards from the Claude-family shape at both Qwen scales — high
   early, mid trough, weak late rise — instead of ~0 early rising from ~1/3
   depth. The occupancy profile (upper panel, absolute varfrac@25 on 30
   held-out wikitext prompts) is humped at 1.5B (peak L21, 0.124) and roughly
   3× lower and U-shaped at 7B (trough L17 0.012, peak L22 0.040); the ceiling
   verdicts themselves are certified on the paper's own metric in the
   excess-FVE figure above.*
   ([provenance](observations/figures/INVENTORY.md))
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

![2026-07-21-jspace-unspoken-words](observations/figures/2026-07-21-jspace-unspoken-words.png)

*Per-layer top-5 lens readout at a single held-out position (prompt 0, token
127 of 224; held concept = critical book reviews): the J-lens top-1 turns
content-bearing at L16 (1.5B) and L19 (7B), tracing criticism → critiques →
commentary → reviews, while the logit lens stays junk until L18 and L23
respectively. This is what "decodes held concepts token-indexed" looks like on
one position — a qualitative illustration, not an aggregate.*
([provenance](observations/figures/INVENTORY.md))

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

**Limitations.** Ranked in full in [§ Limitations](#limitations) below.

**Next paths.** Dose-response: strength sweep at the L18/L19 peak;
mid-scale (14B/32B) replication of the discrete flip; richer
concept-vector constructions for the 59%-tier question; an AV trained
at L22+ (above the legibility onset) for a clean cross-tie; multilingual
eval sets; verbal-report stage 5.3 modulation if the arc reopens.

## Observation log

The arc's dated writeups, in `observations/`:

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
  59% middle tier (J-space component of concept vectors) is indistinguishable
  from random at BOTH scales (bounded ≤3.8pp at 7B; see the 2026-07-24
  recompute's correction note) — no causal contribution detected beyond
  token-indexed directions; weight shifts toward the token-steering account at
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
- `2026-08-30-excess-fve-dimension-dependence.md` — post-close correction
  (issue #79): excess-FVE carries a common ~2× dimension factor between
  scales (the fveTopK/fveRand ratio is near-invariant, 6.29 vs 6.50), so
  the ceiling verdicts get a calibration caveat; audit CHECK O (27
  claims) re-derives every number from the committed scan logs.
- `2026-07-24-paper-metric-varfrac-recompute.md` — post-close vetting
  (issue #26): the paper's 10% ceiling is excess-over-random
  orthogonal-projection FVE, not the scans' absolute varfrac — recomputed
  under the paper's definition **both ceiling verdicts survive** (1.5B
  L21 excess 11.2% CI [11.0, 11.4], 7B peak 4.7%) and hold on **all four
  robustness axes** re-run in that metric; stage-5.2 gap
  significance-certified (exact permutation) and the stage-5.1b 59%-tier
  **bounded, not certified**: no effect detected at either scale, with the
  tighter bound at 7B — 0 discordant pairs of 78 pins the gap at ≤3.8pp
  (exact 95%), while 1.5B gives p=0.375 on 5 discordants. The `p=1.0` at 0/0
  discordants is an identity; the bound, not the p-value, is the result;
  pursuit norm-bias
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

## Data correction — C4 PII redaction and re-run (closed 2026-08-16)

> **What happened.** This arc's corpus-sensitivity axis used a seeded slice of
> **C4-en** — Common Crawl web text. C4's cleaning filters for quality, not for
> personal data, so the committed slice carried **120 pieces of third-party
> PII**: 27 email addresses, 44 phone numbers, 39 street addresses, 10 postal
> codes, across 62 documents. Several documents paired a named individual with
> a direct email and phone. This was published in a public repo for ~9 days.
> It has been redacted. Rationale, counts, limits, and the exact reproduction
> recipe: [`data/README.md`](data/README.md); licensing:
> [`data/LICENSE-DATA.md`](data/LICENSE-DATA.md).
>
> **Why it wasn't caught.** Decision 1 (`plans/2026-07-18-jspace-design.md:19-21`)
> chose C4 for a sound scientific reason — wikitext-103 is narrow Wikipedia
> register, and the paper specifies a "pretraining-like distribution." No
> privacy, licensing, or ethics check was recorded at any point in the arc: a
> search for `privacy|PII|redact|consent|personal data|ODC-BY|terms of use`
> across the arc returns nothing. Provenance discipline here was exemplary on
> *reproducibility* (seed, buffer size, filter, offset, dedup proofs, sha256)
> and silent on *rights*. The generalizable lesson — a corpus chosen for being
> more representative of real text is, for that same reason, more likely to
> contain real people's data — is now a standing pre-use check in the repo
> `CLAUDE.md`.
>
> **What is NOT affected.** The **primary results are unaffected.** The arc's
> primary fitting corpus is wikitext-103, which was scanned and is unmodified
> (its only two matches are encyclopedic biography of a figure who died in
> 1976 — reviewed, not personal data). The headline ceiling verdicts (1.5B L21
> excess 11.15%, CI [10.95, 11.40]; 7B 4.72%), all stage-5/5.1b/5.2 swap
> results, the NLA cross-tie, the quantization axis, and the n-budget axis are
> **C4-free** and stand as recorded. 41 of 53 committed artifacts carry no C4
> dependency.
>
> **What WAS affected.** The redaction is not length-preserving, so a lens
> re-fit on the redacted corpus differs slightly from the artifacts committed
> before 2026-08-16, which were fit on the pre-redaction text. Everything
> listed below was regenerated on the redacted corpus (commit `a9df3a55`,
> 2026-08-15/16) and the prose restated against the regenerated artifacts.
>
> *Affected data files (12):* `data/fitting_prompts_c4en_n1000.json`,
> `data/heldout_prompts_c4en_n30.json` (both now redacted); and 10 derived
> artifacts computed from them — `lens_eval_*_c4en.pt`,
> `readout_scan_*_c4en.pt`, `structure_scan_*_c4en.pt`,
> `paper_metric_varfrac_*_c4en.pt`, plus the `*_heldoutc4en.pt` readout,
> structure, and paper-metric scans at both scales. (Verified: these `.pt`
> files contain **no** PII themselves — their decoded strings are individual
> BPE tokens, not contiguous text — so they are stale, not unsafe.)
>
> *Affected figure (1):* `observations/figures/2026-07-21-jspace-corpus-invariance.png`.
>
> *Affected observations (4):* `2026-07-20-corpus-sensitivity-c4-1p5b.md`
> (whole file), `2026-07-22-n500-and-heldout-robustness.md` (held-out C4 axis),
> `2026-07-24-paper-metric-varfrac-recompute.md` (2 of 4 robustness-axis rows),
> `2026-07-18-intermediate-concept-evals-h3-confirmed.md` § "Qualifier (added
> 2026-07-20)" (the corpus-dependence qualifier on the @10 magnitude). Each
> now opens with a dated 2026-08-16 read-first addendum; the original text is
> preserved unedited beneath it.
>
> *Affected audit checks:* J and K wholly, M partially — ~25 pinned values in
> `examples/jspace_audit_findings.py`. Two of them moved beyond tolerance and
> were re-pinned (below); the rest re-derive within their existing tolerances.
>
> **The re-run (executed 2026-08-15/16).** Three lenses were refit on the
> redacted corpora via `examples/jspace_rerun_queue.py` — c4en-1.5B **3.14 h**;
> wikitext-1.5B **8.22 h** of queue wall-clock across two segments (3.54 h of
> it GPU time; the card was handed back to the desktop in between, so the
> reproducible single-segment cost is ~3.24 h); wikitext-7B **11.49 h**,
> against 16.26 h for the arc's original 7B fit. The two wikitext refits were
> needed because the full 27-layer lenses are cache-only (Decision 4) and the
> cache was empty — channel 2 is "re-scan only" only when they are on disk.
> The ten derived artifacts were then regenerated by
> `examples/jspace_rerun_scans.sh` (06:13–08:01 UTC, ~1.8 h). All three
> regenerated paper-metric artifacts validate bit-exact against their
> structure scans (`validation_max_vf_diff = 0.0`).
>
> **Outcome.** Audit after the re-run and before re-pinning: **954 PASS |
> 6 FAIL**; after re-pinning the two values that legitimately moved,
> **956 PASS | 4 FAIL** (post-re-pin log committed at
> `data/audit_2026-08-16.log`; those are the totals **as of that re-run** —
> the audit has since gained checks, so a re-derivation today reports
> 986 | 4 with the cache (`data/audit_2026-08-17.log`). Re-derive with
> `python examples/jspace_audit_findings.py` — the three re-fit lenses are
> LFS-committed in `data/cache/` as of 2026-08-16, so no refit is needed;
> the lens download is opt-in, see "Expected result on a clean clone" below).
> The 4 remaining FAILs
> are the designed `MISSING` reports for the two deliberately-unrefit nf4
> lenses and their sidecars (`jlens_qwen2.5-1.5b_nf4_n100` and `_n500`) — the
> quantization and n-budget axes are C4-free, so refitting them (8.3 h) would
> buy a green audit and no correction; owner decision 2026-07-29. That
> decision was reversed 2026-08-16 on data-completeness grounds — the refits
> are scheduled as issue #47; until they land, the two `MISSING` reports
> remain the designed state.
>
> Drift beyond audit tolerance was confined to the **held-out-C4 channel** —
> the channel whose evaluated text was directly redacted (see the attribution
> caveat below). Within-tolerance movement did occur elsewhere: on the refit
> C4 lens, L0 varfrac re-derives 0.1751 (pinned 0.174) and L0 logit-kurtosis
> 2.6455 (pinned 2.641), both in the early band the arc already flags as
> corpus- and norm-sensitive. Two pins moved beyond tolerance:
> - **[K] 1.5B held-out logit-kurtosis trough:** 1.000 → **1.0186**
>   (re-pinned 1.019 at atol 0.01; n=30, the smallest sample in the audit).
> - **[M] 7B held-out L23 excess:** 0.0598 → **0.0588** (atol 0.0005).
>
> Both re-pins were made after confirming the regenerated artifacts, not
> before. Every headline conclusion is unchanged.
>
> **Caveat on attributing the 7B move (checked 2026-08-16).** The two
> wikitext lenses had to be refit from an empty cache, so a channel-2 scan
> differs from its predecessor for two reasons at once — redacted held-out
> text *and* any difference in the refit lens. Comparing each refit lens
> against its committed `jlens_*_layer-subset.pt` (promoted from the original
> fits, untouched since 2026-07-22) separates them: the **1.5B bf16 refit
> reproduces the original exactly** (max |Δ| = 0 over the seven committed
> layers), so the 1.5B held-out kurtosis move is the redacted text alone;
> the **7B nf4 refit does not** — relative Frobenius Δ 1.7e-2 at L0 decaying
> to 4.2e-4 at L26. The 7B held-out excess move (0.0598 → 0.0588, −1.7%
> relative) therefore cannot be attributed to the redaction alone; refit
> noise at that scale is of the same order. nf4 fits are not bit-reproducible
> here, and no earlier claim in this arc depended on their being so.
>
> **Pre-committed exposures — both watched, neither materialised.** The plan
> named two in advance so the re-run could not be read as confirming what was
> hoped:
> 1. *The quoted "L21 excess 10.7–11.7%" range.* Its C4-dependent endpoints
>    re-derive to **0.10828** (C4-corpus axis) and **0.11689** (held-out
>    axis), both passing at atol 0.0005; the range's low endpoint is the
>    C4-free n-budget axis at 0.1069. The quoted range stands.
> 2. *The C4-corpus-axis bootstrap unanimity, margin 0.008 above the ceiling.*
>    `P(bootstrap mean > 10%)` is still **1.0 on every 1.5B axis**, and the
>    C4-axis margin above 0.100 is **0.0083** (was 0.0082). "Bootstrap-unanimous
>    breach on each axis" stands.
>
> *Affected conclusions — status after re-derivation:*
> - **Corpus-invariance** (1 of the 4 axes): **stands.** C4-lens L21 varfrac
>   0.1242 vs the wikitext lens's 0.1237 (Δ 0.0005, the "identical to 3dp"
>   claim holds); C4 axis L21 excess 0.1082.
> - **Held-out-sample robustness** (the other affected axis): **stands, with
>   two restated values.** 1.5B held-out peak still L21 at 0.133 and L0 0.084;
>   the kurtosis trough is now 1.019 at L17 (was 1.00), and the 7B held-out
>   band peak is 5.88% excess at L23 (was 5.98%).
> - **The 1.5B ceiling breach:** never depended on C4 — the wikitext
>   all-positions artifact carries it at L21 excess 11.15%, CI [10.95, 11.40],
>   unchanged by the re-run.
> - **The @10 corpus-dependence qualifier** on H3: **stands** — C4 multihop
>   J@10 overall 0.4854 vs logit 0.3981.
>
> **Status.** Redaction: **done 2026-07-29**. Re-run: **done 2026-08-15/16**
> — 22.85 h of queue wall-clock over the three refits plus ~1.8 h of scans,
> against the plan's ~23.7 h estimate (the Status line here previously read
> "~4–5 h GPU, dominated by one 2.2 h lens refit": that estimate was wrong in
> two independent ways, both diagnosed in the plan's § Cost). Plan:
> [`plans/2026-07-29-c4-redaction-rerun.md`](plans/2026-07-29-c4-redaction-rerun.md).
> The numbers throughout this README are the re-derived ones; the
> superseded values are preserved in the dated addenda at the top of
> each affected observation.

## Limitations

Ranked by how far each limit constrains the arc's claims, most constraining
first. Full detail sits in the linked observations and in
[§ Findings](#findings--arc-close-synthesis-2026-07-22).

- **L1. One model family, ≤7B, unreplicated.** Everything here is
  Qwen2.5-Instruct at two scales (1.5B bf16, 7B nf4), and the results are
  unreviewed and unreplicated outside this repo. Whether the failure of the
  stronger J-space-*subspace* claim is capability scale (Claude 4.5-family
  vs ≤7B) or model family is the open question the arc cannot answer from
  below.
- **L2. The negative results are measured bounds, not demonstrated zeros.**
  The paper's 59% tier (J-space component of concept vectors) shows no
  detectable effect at either scale, but a hypothesis test cannot establish
  a null: the claim is a ≤3.8pp equivalence bound at 7B (0 discordant pairs
  of 78, exact 95%) and a much looser p=0.375 on 5 discordants at 1.5B. The
  discrete entailed-property flip (0.000 at both scales) is likewise bounded
  below only. See
  [`2026-07-24-paper-metric-varfrac-recompute.md`](observations/2026-07-24-paper-metric-varfrac-recompute.md)
  § correction 2026-07-28.
- **L3. The 7B lens exists only in nf4.** Quantization was exonerated as a
  confound at 1.5B (bf16↔nf4 moves nothing at fixed corpus/budget) but is
  untested at 7B-bf16, which is infeasible on this hardware. nf4 fits are
  also not bit-reproducible here (refit relative Frobenius Δ 1.7e-2 at L0
  decaying to 4.2e-4 at L26), so small 7B movements cannot be attributed to
  a single cause. The stage-5.1 plain-prompt 7B near-null is explicitly a
  *confounded* near-null (weak nf4 lens × quantization × baseline compliance
  failures), un-confounded only under chat prompts at 5.1b.
- **L4. n=100 fitting budget, with estimator noise never bounded.**
  n-stability was shown at 1.5B only (n=100→500, peak −1.4%); the 20%
  stability threshold was fixed in the session record before the data
  landed but was **not** a repo-committed pre-registration. Whether 3584²
  Jacobians are data-starved at n=100 in a way 1536² are not stays
  logically open (the direct 7B n=500 test is ~81 h). The stage-2 split-half
  lens-stability gate was waived for the first pass and never run — see
  [§ Possible next paths](#possible-next-paths) item 3.
- **L5. The swap statistics rest on small, post-hoc-selected samples.** The
  arc's strongest positive (the relational entailed-property effect) is
  n=7 auto-detected items at 1.5B — p=0.0156 is the smallest value an n=7
  sign-flip test can produce — and L18/L19 were chosen as peak layers from
  the same data, so those p-values are post-hoc at a data-chosen maximum.
  Item-level SD exceeds the mean at both scales. The multiplier framing
  (~34×/8×) is denominator-fragile because the control sits near zero; cite
  the absolute nats gap.
- **L6. The readout advantage is one favorable sub-metric, not a sweep.**
  The J-lens surfaces intermediates the logit lens misses, but the logit
  median is still earlier at 1.5B (19 vs 23) and on the companion
  never-emergence count the J-lens fails to surface the token in *more*
  cells at both scales (1.5B 16/108 vs 1/108; 7B 14/108 vs 8/108).
- **L7. Early-band (L0–L16) readings are contaminated.** Occupancy and
  top-atom readings there are partly norm-driven (unnormalized-atom pursuit
  selection bias — 7B via undertrained junk-token atoms, 1.5B via tied
  embedding norms) and are the band that is corpus-sensitive. The
  workspace band measures norm-neutral and corpus-invariant; early-band
  numbers should not be read as structural.
- **L8. Held-out sets are 30 prompts.** At 7B's low absolute occupancy
  per-sample noise is a large fraction of the signal, so 7B varfrac is
  quoted band-level (~0.01–0.05), not to three decimals. The one audit pin
  that moved most in the C4 re-run was the n=30 held-out kurtosis trough.
- **L9. The fitting corpus is narrow.** wikitext-103 is English-only
  Wikipedia register — narrower than the paper's "pretraining-like
  distribution" phrasing and than Qwen2.5's multilingual training mix.
  Corpus sensitivity was checked at 1.5B only (seeded C4-en refit); no 7B
  corpus refit was run.
- **L10. Single-token concept limit throughout.** All readout, swap, and
  emergence metrics are defined over single-token concept forms. Association
  targets never enter top-50 under either lens, which may mean they do not
  exist as single-token representations at this scale. [SPECULATION]
- **L11. The paper's metric had to be reconstructed.** The random control
  draw is unspecified upstream (uniform vocab atoms is the natural reading),
  the paper reports five workspace layers with per-layer K while this arc
  sweeps all layers with one procedure, and the bootstrap unit is the prompt.
- **L12. The stage-6 NLA cross-tie inherits the NLA arc's unaudited AV
  format bias** (mitigated by nulls, not eliminated), and is measured at the
  NLA capture layer L20 — below the 7B J-lens legibility onset (~L22), where
  the J-lens readout is only intermittently contentful.
- **L13. The audit checks arithmetic consistency only.** A PASS means a
  number in prose still matches the artifact it came from; it cannot detect
  a methodological error, a capture-protocol bug, or interpretive overreach.
  The stage-5.1b certification defect is the worked example — every pinned
  number was correct and passing while the inference drawn from them was
  wrong.
- **L14. Data and reproduction gaps.** The C4-en slice carried third-party
  PII and was redacted; the redactor removes contact *channels*, not personal
  names, which is a stated known limit
  ([`data/README.md`](data/README.md)). Redaction is not length-preserving,
  so C4-dependent artifacts are the re-run ones. Two 1.5B nf4 lenses
  (quantization and n-budget axes) remain regenerate-only, so a clean-clone
  audit reports 4 designed `MISSING` results until the refits scheduled as
  issue #47 land.

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Quotes below are verbatim from the session transcripts of 2026-07-18 →
2026-07-24 (Michael Lannum), lightly normalized for typos and punctuation;
markdown emphasis inside a turn is dropped, and `[...]` marks an editorial
elision. This section follows the attribution shape in
[`ARC_PROCESS.md` § 6](../../ARC_PROCESS.md#6-arc-readme-synthesis), for
which [arc 02](../02_subliminal/README.md#attribution) is the
reference implementation.

**Originating direction** [session 2026-07-18]:

> *"Let's move into a new worktree for the next arc, 04_jspace. Here's the
> direction. Anthropic released new research a few days ago about 'J-Space',
> a hidden state vector subspace that acts as a sort of global workspace as
> it's processed through the transformer layers, allowing a sort of stable
> representation of concepts the LLM could say. First do the research you
> need to understand J-Space from a technical standpoint. Make sure to add
> the source material to ./theory and the kb for reference. Then I want to
> try to calculate/explore J-Space in the Qwen model we've been using. Once
> you're at that point let me know so I can review the informed design/test
> plan."*

That turn fixes all three of the arc's framing choices: the replication
goal, the target model, and the review-the-plan-before-running gate.

**Design sign-off, with two amendments** [session 2026-07-18]:

> *"Otherwise this sounds like a good plan, I read through Nanda's critique
> and I agree on most of the points. The NLA comparison cross-tie is a very
> good idea. If you OOM on the 2080 pause before switching right to CPU —
> I'll want runtime estimates for CPU vs. GPU [...] Also remember to push
> notify updates and milestones to me as you work or if you need me."*

The OOM gate (amendment 1) is why the 7B fit was calibrated and offloaded
rather than dropped to CPU; the comms protocol (amendment 2) is why the
long unattended runs were checkpointed and reported. The J-space↔NLA
cross-tie was **Claude's** proposal, endorsed here — not the user's.

**The artifact-verification standard** [session 2026-07-20], after a weekend
of unattended runs:

> *"can you check on where we are in this arc and verify completion/integrity
> of the tests that ran?"*

> *"Okay, well is all of that fixed up, logged, and documented for the repo
> at least (transparency, correctness, auditability)?"*

These two turns are the origin of the post-hoc integrity audit, of the
standing rule that an unattended run is verified by its **artifacts** rather
than by a process check or a prior session's claim, and of the arc's
auditability standard (defects go into the permanent record; datasets stay
complete and clean-clone reproducible *mid*-arc, not at close).

**The figures-and-data standards** [session 2026-07-21, 2026-07-24]:

> *"we should always be generating supporting plots/visualization figures for
> interesting or important result datasets. Make sure to show rich,
> meaningful data, and replicate any figures shown in the Anthropic paper
> (if possible)."*

> *"make sure we capture and store all data that could be useful [...] make
> sure to include per-layer token strings and any other data that could be
> useful!"*

> *"every rendered figure should be based on computed auditable datasets, so
> it also needs to have accompanying datasets committed somewhere in the
> repo"*

**The metric-fidelity reopening** (issue #26) [session 2026-07-23]. The arc
had already been closed. It was reopened by this turn:

> *"I want to focus on solidifying my own intuition/understanding of J-Space
> and the J-Lens in the context of this research arc. This will also help me
> learn and vet the math we've used to compute everything, and probe our
> approach for significance and robustness."*

and the evidentiary bar the recompute was held to was set by this one:

> *"run whatever testing/calculations you need until you have undeniable
> concrete proof one way or the other"*

The request was framed as the user's own learning, not as a defect report —
the K-inconsistent variance-fraction defect surfaced *because* the math was
walked through end-to-end to explain it. Requiring the variance fraction be
checked against the paper's own definition in the source PDF was also the
user's call.

**The review-before-integration policy** [session 2026-07-22]:

> *"Run a thorough opus workflow PR review on #25. Maintain high standards
> for research integrity, auditability, and attribution/transparency. [...]"*

— i.e. the pre-merge bar set at research integrity, auditability and
attribution rather than code correctness, with integration reserved for a
human merge.

### Human / Claude / emergent split

**User (Michael Lannum).** The originating direction above (target model,
replication goal, KB-grounding requirement, plan-review gate); the design
sign-off and its two amendments; every stage-boundary go/no-go and
open-decision call; the auditability, artifact-verification, figures, and
rich-data standards quoted above; the reviewer verdicts and the
review-before-integration policy; the post-close reopening and its
evidentiary bar. Also two substantive methodological catches Claude had
missed: that the steering experiment had **drifted from the paper's prompt**
(*"So why did you change the prompt from the Anthropic paper's example? [...]
Could we be overloading the spider focus or something?"*, [session
2026-07-22]), and that the fitting/held-out corpora were **narrower than
claimed** (*"what variety of prompts were we using over the n=N prompt
averaged jacobian calculations? I see two wikitexts json files and one with a
bunch of prompts that discuss criticism"*, [session 2026-07-22]) — the
observation that led to the corpus-sensitivity refit.

**Claude Code.** Paper digestion and KB grounding; the staged experiment
design; the J-space↔NLA cross-tie proposal; all implementation (fitting,
scans, evals, steering, figures, manifest and audit scripts); the OOM
calibration and offload strategy; the write-ups; the math walk-through that
surfaced the metric defect; and the corrective recompute.

**Emergent.** The K-consistency defect itself (issue #26) — the user asked
for an explanation, not an audit, and Claude's attempt to give a faithful one
is what exposed it; neither the request nor the implementation would have
found it alone. Likewise the bounded-equivalence restatement of the 7B tier
result (the ≤3.8pp bound from 0 discordant pairs of 78), which came out of
an adversarial review pass rather than from either party's plan.

**Verifiability.** Every quote above is recoverable from the session
transcripts for 2026-07-18 → 2026-07-24; the transcripts are not committed to
this repo (they carry machine-local paths and tool output). Claims in this
section that are *not* quoted are Claude's characterization of the user's
direction, not the user's wording.

## Possible next paths

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
7. **Dimension-matched 7B recompute (issue #79)** — one pursuit re-run at
   K ≈ 58 (≈ 25 × 3584/1536; needs a k_max above 58, no lens refit). If
   the cross-scale excess gap survives at matched K/d, the scale finding
   is stronger than currently stated and should be restated as such.

**Theory grounding:** `theory/kb/notes/interpretability/j-space.md`,
excerpts in `theory/kb/excerpts/gurnee2026-workspace.md`, archived paper PDF
in `theory/sources/papers/gurnee2026-workspace_verbalizable-global-workspace.pdf`.

## Reproducing

Load-bearing numbers re-derive from artifacts via
`examples/jspace_audit_findings.py` (450 checks at arc close; 739 after
Check M — the issue-#26 metric-correction battery: paper-metric ceiling
+ the four 1.5B robustness axes and the 7B held-out set, swap
significance + 5.1b McNemar tiers, and the
two-scale norm-bias pins — landed 2026-07-24; 978 claimed with a
warm five-lens cache after the final-review
pins landed 2026-07-25: full stage-4 depth-table cells both scales, the
naive-vs-paper delta decomposition, K_median_occ, the exact McNemar
p-value, and the MANIFEST census). All small derived artifacts (44 files,
~55 MB incl. the ten metric-correction artifacts) are LFS-committed
under `data/` and MANIFEST-registered (sha256), so **checks B–N run from
a clean clone**; check A and the lens-integrity blocks read the full
fitted lenses. Decision 4 originally kept all five cache-only (committed
layer subsets + `jspace_fit_lens.py` regenerate them); amended by owner
decision 2026-08-16 after the C4-redaction re-run: the **three lenses
refit in that re-run are now LFS-committed in `data/cache/`**
(~905 MiB / ~949 MB — both wikitext lenses + the c4en lens, with their
`.config.json` sidecars and fit/scan logs) so a clean clone can re-scan
without repeating the ~23 h refit. The two 1.5B nf4 lenses
(quantization / n-budget axes) remain regenerate-only for now; their
refit-and-commit is scheduled as issue #47. Fit-resume `.ckpt.pt`
checkpoints stay uncommitted — their content is superseded by the final
lenses they produced.

**Expected result on a clean clone.** The lens cache is excluded from
default LFS downloads (`.lfsconfig` `fetchexclude` — ~905 MiB most readers
never load), so there are two states (totals as of 2026-08-30, after
CHECK O added 27 cache-independent log-based claims — issue #79):

- **Default clone** (`git lfs install && git lfs pull`; lenses stay pointer
  stubs): `SUMMARY: 978 PASS | 7 FAIL`, exit code 1 (measured 2026-08-30;
  951 before CHECK O, measured 2026-08-17 — the new 978 total is
  coincidentally equal to the unrelated historical warm-cache figure
  discussed below). The 7 = three
  `LFS pointer stub` reports for the committed lenses (the audit detects the
  stub and prints the pull command) + the designed `MISSING` reports for the
  two regenerate-only nf4 lenses and their sidecars.
- **After** `git lfs pull --include="research/arcs/04_jspace/data/cache/**"
  --exclude=""`: `SUMMARY: 1013 PASS | 4 FAIL` expected (986 measured
  2026-08-17, `data/audit_2026-08-17.log`, plus the 27 CHECK O claims,
  which do not depend on the cache), the 4 being the nf4 `MISSING`
  reports only.

Neither state's failures are regressions. Any FAIL naming something other
than a `jlens_*.pt` / `jlens_*.config.json` artifact is a genuine
regression. (A pre-2026-08-16 checkout — no committed lenses, pre-re-run
artifacts — gave `920 PASS | 10 FAIL`, measured 2026-07-29; the historical
decomposition below still applies to it. An LFS-less clone of the current
head is a different, noisier state: every `data/*.pt` deliverable is a
pointer stub, so the audit reports one stub FAIL per artifact — run
`git lfs install && git lfs pull` first.)

> Until 2026-07-29 this read `11 FAIL`. The extra one was a duplicate
> registration of the `jlens_qwen2.5-1.5b_bf16_n100_c4en.config.json` presence
> check inside CHECK J, fixed with #33. The earlier note here also claimed the
> duplicate "inflates the with-cache PASS total by one" — **that was wrong**:
> `load_json_or_fail` registers a claim only on the *missing* branch, so with a
> populated cache the duplicated call was silent and contributed nothing to the
> total. The same fix pass disambiguated two further checks that shared one
> label (`jlens@2 top5_all rederived==summary`, emitted by both the
> verbal-report and chat-6c audits at each scale) — a reporting defect, not a
> miscount: a failure could not be attributed to either artifact.

The with-cache total is larger than 920, because check A and the refit blocks
in H/I/J run only when the fitted lenses are on disk. **The 978 figure was
never re-verified and is not reproducible as such** — it assumed all five
fitted lenses cached. The C4-redaction re-run
([plans/2026-07-29-c4-redaction-rerun.md](plans/2026-07-29-c4-redaction-rerun.md))
repopulated three of the five (both wikitext lenses + the c4en lens), and the
measured result in that state is **986 PASS | 4 FAIL** (2026-08-17,
`data/audit_2026-08-17.log`), the 4 being the presence
checks for the two nf4 lenses and their sidecars, which were deliberately not
refit. A fully-green run would need those two refits (8.3 h — declined
2026-07-29, then scheduled 2026-08-16 as issue #47); until they land, 978
stays an unverified historical figure and 986/4 is the measured one.

**What this audit does not catch.** Like the arc-01 and arc-03 audits, it
checks **arithmetic consistency only** — that a number in prose still matches
the artifact it was derived from. It cannot detect a methodological error, a
capture-protocol bug, or interpretive overreach. The stage-5.1b certification
defect corrected on 2026-07-28 is the worked example: every pinned number was
correct and passing while the inference drawn from them was wrong.

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

## File map

```
research/arcs/04_jspace/
  README.md            # this file
  plans/               # signed-off design + stage addenda + re-run plan
    2026-07-18-jspace-design.md
    2026-07-20-stage5-design.md
    2026-07-20-stage6-design.md
    2026-07-21-stage52-entailed-property.md
    2026-07-29-c4-redaction-rerun.md
  observations/        # dated evidence-first writeups (see § Observation log)
    figures/           # rendered figures + INVENTORY.md provenance
  data/                # LFS-committed artifacts + MANIFEST.json + audit logs
    cache/             # committed fitted lenses (opt-in LFS download)
```

Pipeline scripts live at the repo root under `examples/jspace_*.py`
(capture / analyze / render / audit; the audit entry point is
`examples/jspace_audit_findings.py`).
