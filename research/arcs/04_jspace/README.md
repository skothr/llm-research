# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-1.5B/7B-Instruct

> ## ⚠ DATA CORRECTION 2026-07-29 — C4 corpus PII redacted; corpus-robustness results pending re-run
>
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
> **What IS affected — pending re-run.** The redaction is not
> length-preserving, so any lens re-fit on this corpus will differ slightly
> from the committed artifacts, which were fit on the pre-redaction text.
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
> `2026-07-18-intermediate-concept-evals-h3-confirmed.md:105-109` (the
> corpus-dependence qualifier on the @10 magnitude).
>
> *Affected audit checks:* J and K wholly, M partially — ~25 pinned values in
> `examples/jspace_audit_findings.py`.
>
> *Affected conclusions:* the **corpus-invariance** and **held-out-sample**
> robustness claims, i.e. 2 of the 4 axes in "invariant on all four axes"
> (README below). Both endpoints of the quoted **"L21 excess 10.7–11.7%"**
> range are C4 rows and must be recomputed. The most exposed single claim is
> the bootstrap-unanimity of the C4 corpus axis, whose margin above the 10%
> ceiling is only 0.008 — a ~1-point shift would soften "unanimous on each
> axis" to "on three of four." The 1.5B breach itself does **not** depend on
> it (the wikitext all-positions artifact carries that at CI [10.95, 11.40]).
>
> **Status.** Redaction: **done**. Re-run: **planned, not yet executed** —
> ~4–5 h GPU, dominated by one 2.2 h lens refit. Plan:
> [`plans/2026-07-29-c4-redaction-rerun.md`](plans/2026-07-29-c4-redaction-rerun.md).
> Every affected number below should be read as **provisional pending that
> re-run**, and this warning stays until it completes.

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
2026-07-29): `SUMMARY: 920 PASS | 10 FAIL`, exit code 1. All 10 failures are
the designed `MISSING` reports for the five cache-only fitted lenses and their
five `.config.json` sidecars — *not* regressions. Any FAIL naming something
other than a `jlens_*.pt` / `jlens_*.config.json` artifact is a genuine
regression.

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

The 978-check figure above requires the local cache, where check A and the
refit blocks in H/I/J also run. **978 is carried over unverified** — it was
last measured with a warm cache on 2026-07-25, and the cache is currently
empty. Neither 2026-07-29 fix changes it (one was silent on the with-cache
path; the other only renamed labels), but it will be re-derived and restated
when the C4-redaction re-run repopulates the cache
([plans/2026-07-29-c4-redaction-rerun.md](plans/2026-07-29-c4-redaction-rerun.md)).

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

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Quotes below are verbatim from the session transcripts of 2026-07-18 →
2026-07-24 (Michael Lannum), lightly normalized for typos and punctuation;
markdown emphasis inside a turn is dropped, and `[...]` marks an editorial
elision. This section follows the attribution shape in
[`ARC_PROCESS.md` § 6](../../ARC_PROCESS.md#6-arc-readme-synthesis), for
which [arc 02](../02_subliminal/README.md#research-direction) is the
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
   indistinguishable from random at both scales, and at 7B the gap is
   bounded at ≤3.8pp (exact 95%, 0 discordant pairs of 78); 1.5B is looser
   (p=0.375 on 5 discordants). A tight measured bound, not a demonstration
   of exact zero. The NLA verbalizer's content likewise lives in
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
