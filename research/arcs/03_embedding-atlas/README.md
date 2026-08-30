# Embedding atlas — semantic structure of Qwen2.5-7B's input-embedding table

**Research question:** What semantic/functional category structure exists
*already* in the static input-embedding table (W_E, "layer 0") of
Qwen2.5-7B-Instruct — do word groups related by topic, function, morphology,
or connotation share directions that could act as "handles" for downstream
layers — and how does that structure compare to the layer-20 geometry found
in the [nla-verbalizer arc](../01_nla-verbalizer/README.md)?

**Status:** CLOSED for new experiments 2026-07-15 (deep arc, user direction
2026-06-11; started 2026-06-10). All three phases landed: phase 1 (battery
protocol probes), phase 2 (full-vocabulary sweep, 149,706 alive rows), and
phase 3 (structural tracing T0/T1/T1.5/T2, absorbing the former rope-vis arc
per [plan](plans/2026-06-10-rope-vis.md)). Six observations; audit at
**99 PASS / 0 FAIL**; pre-registered predictions
([plans/2026-06-11-predictions.md](plans/2026-06-11-predictions.md))
adjudicated: P1a PASS, P1c FAIL, P1d FAIL, P2 refined-not-falsified (content
moves to a new basis at L1), P1b/P1e/P3 not run (deferred, below). Remaining follow-ups are recorded under "Deferred
follow-ups" and are candidates for a successor arc rather than this one.
Provenance: the 2026-06-10/11 kickoff sessions ran on `claude-fable-5`
within days of that model's release, so a reported (but here
**unconfirmed** — see the note's sourcing breakdown) initial-release
quality-degradation window would have covered them; audited clean at the
user's direction —
[degradation-forensics](sessions/2026-07-21-degradation-forensics.md).

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Quotes below are verbatim user turns (Michael Lannum) from the session
transcripts of 2026-06-10 → 2026-07-21, lightly normalized for typos and
punctuation; markdown emphasis inside a turn is dropped, `[...]` marks an
editorial elision, and dates follow the transcripts' UTC clock. This section
follows the attribution shape in
[`ARC_PROCESS.md` § 6](../../ARC_PROCESS.md#6-arc-readme-synthesis). The
atlas idea itself predates the arc: it was seeded by the semantic-basis-grid
turns of 2026-05-14, quoted in
[arc 01 § Attribution](../01_nla-verbalizer/README.md#attribution).

**Originating direction** [session 2026-06-10]:

> *"explore the complete set of token embeddings for correlations and
> interesting connections between groups of words (related in one way or
> another — like we did with countries, but also things like check for
> similarities/correlations/features for different types of words different
> subjects, try out). Words with similar or related
> function, use, topic/connotation, etc. may have shared features that act
> as 'handles' for the model layers to execute that function hidden in our
> construction of language."*

Scope decisions made at session start: two arcs (embeddings first, RoPE
second) rather than one. At checkpoint CP1 the user selected the offered
next-step option Add more word groups first (a checkpoint-menu decision, not
a typed message). The direction to broaden beyond countries into "different
types of words different subjects" is the user's; the specific
person/royal/religion/abstract/landscape/instrument/science/tech/language
classes and the gender/antonym/past/capital_of/lang_of pair kinds are
Claude's operationalization of it.

**Correlation-first, entangled dimensions, and the no-small-samples rule**
[session 2026-06-10], after the cost question "how expensive would it be to
run some calculations/correlations over the entire token universe?" had put
the full-vocabulary sweep in scope:

> *"okay yeah, so now we're doing the reverse of before. We found groups of
> semantically similar tokens and looked for feature correlations between
> them before, now we're looking for similarities between the full set of
> token embeddings [...] there might be more small-scale correlations —
> like specific dimensions or correlated/'entangled' dimensions in the
> vector serving as more granular handles that serve particular functions
> and mathematically encode its semantics.
> I want to really focus on thorough data collection before we come to any
> conclusions. Run it in scaling steps and branch out (and push notify me)
> if data reveals any potential unexpected paths. [...] we shouldn't be
> generalizing from a small sampling of data like we've been doing, ever."*

Three direction-setting acts in one turn: it inverts the methodology from
category-first to correlation-first; it names correlated/"entangled"
dimensions acting as granular handles as the target — the object the 21-dim
entangled block later instantiated — before any sweep had run; and it
installs the staged-scaling / no-small-samples discipline that forced the
full-149,706-row sweep ahead of any claim promotion.

**The deep-arc turn** [session 2026-06-11]:

> *"I'd rather work on a more fundamental understanding of the LLM's
> structural/procedural features. [...] Did you create categories and look
> for correlations within them, or find correlations and create the
> categories from different correlation modes [...]?
> Also, for `the` and `的`, is that because they're both particles, or
> because of the phonetic sound? They're not really the same kind of
> particle, `的` is more like `of` or `'s`; like a possessive or relational
> particle.
> Also do some research into whether this path has been travelled before,
> what's been covered and what's missing.
> My thought is next to try to look at K/Q/V matrices generated at the
> first layer from each embedding [...] plus see if after the FFN the
> structural dimensions we found are preserved, and if so try to trace this
> to the next layer and so on. Really need to look at every calculation
> component in the process [...]
> One prediction: Since commas, periods, etc. generally break things up,
> the relevant handles we found should, at least in some layer, show signs
> of that. e.g. at some layer, a chunking syntax token's Q and K may be
> very similar in some subspace [...] we should make this a deep arc btw
> and combine the planned positional encoding arc into this one. Be
> comprehensive."*

Five direction-setting acts: a circularity challenge to the
category/correlation methodology; the `the`/`的` linguistic correction (see
below); the mandate for the prior-work literature review; the
every-component tracing direction that the T0/T1/T1.5/T2 designs
operationalize; and a testable punctuation-chunking prediction that
prefigures the reader/tracker head dissociation — plus the deep-arc
commitment absorbing the planned RoPE arc. The novelty claim was then gated
on an adversarial re-check ("yes, re-verify first as you suggested to
confirm novelty beyond all doubt. Then let's revisit directions."), and an
incomplete trace was caught the same day ("Did you trace the influence of
the initial dimensions through each calculation component [...]? Can we
look for how the FFN reconfigures the hidden state afterward?" — the direct
origin of the T1.5 component-level capture).

**Wrap-up and close-out gates**:

> *"Where are we in this research arc (03_embedding_atlas)? Is there
> anything still in progress to finish up and close out? I want to finish
> this one up and then extend it to a new arc `04-jspace` based on new
> Anthropic research. Let's wrap this up first though."*
> — [session 2026-07-15]

> *"Is this a good place to merge this as a PR, or is there any more final
> verification or documentation in arc 03 to do first?"*
> — [session 2026-07-21]

> *"I don't think you started the review run though?"* — [session 2026-07-21]

The close-out call (also fixing arc 04 as an extension of this arc), the
completeness gate before the merge, and the catch that the promised
fresh-eyes review had not been launched. All stack PRs were
human-merged; merging stayed a manual gate throughout.

**The cross-script correction, made twice**. The `的` correction quoted
above ([session 2026-06-11], in the deep-arc turn) failed to propagate
into the cross-script finding's text, and the user caught it again at
close:

> *"I remember pointing out during a session where this was observed, that
> `the` and `的` aren't really that similar [...] I pointed out that `的`
> would likely be closer to `of` or `'s`, due to the character's possessive
> use [...] Now I see the original 'the'/'的' claim in the observation
> README, so I want to make sure that makes sense."*
> — [session 2026-07-21]

> *"Did you check "'s " as well? (with a space)"* — [session 2026-07-21]

— the direct cause of the corrected cross-script finding and the re-derived
neighbor-cosine table now locked by audit §11. The same session bounded the
Fable-degradation exposure window from the user's knowledge of the release
timeline and commissioned the transcript forensics pass
([degradation-forensics](sessions/2026-07-21-degradation-forensics.md)).

### Human / Claude / emergent split

**User (Michael Lannum).** The originating handles question; the
correlation-first inversion, the entangled-dimensions target, and the
no-small-samples rule; the full-vocabulary scope push; the deep-arc
commitment, the RoPE-arc absorption, and the numbered-arc
reorganization (all 2026-06-11); the literature-review mandate and the
adversarial-novelty gate; the
every-component tracing direction and the FFN/T1.5 catch; the
punctuation-chunking prediction; both cross-script catches and the `'s `
tightening; the wrap-up and merge gates and the unlaunched-review catch;
the degradation-forensics commissioning.

**Claude Code.** All implementation: the 25 `emb_*`
capture/derive/render/audit scripts, the battery class definitions (under
the CP1 direction), the operationalization of the pre-registered
predictions (P1a-P3), the T0/T1/T1.5/T2 experiment designs (under the
user's every-component direction), the execution of the literature review
and its adversarial re-verification, all figures and observation write-ups,
and the audit that re-derives every published number from committed
artifacts.

**Emergent.** The 21-dim entangled block as a characterized object: the
user named correlated/"entangled" dimensions as the target and the sweep
discipline that would find them; the implementation surfaced this specific
block with its punctuation/function-word loading. The reader/tracker head
dissociation (the user's chunking prediction pointed at it; the T2
implementation established it). The P1c/P1d falsifications — adjudicated
mechanically against criteria pre-registered on 2026-06-11, before any
attention weights were captured.

**Verifiability.** Every quote above is recoverable from the session
transcripts for 2026-06-10 → 2026-07-21; the transcripts are not committed
to this repo (they carry machine-local paths and tool output). Claims in
this section that are not quoted are Claude's characterization of the
user's direction, not the user's wording — including the
iterate-with-figures checkpoint cadence, a working mode settled through
checkpoint menus rather than typed direction.

## Findings so far (held as working hypotheses)

Evidence and scope qualifications live in the observation files; one-line
versions:

1. **Near-isotropy null.** Random-pair cosine +0.0097; PC1 explains 1.21%;
   participation ratio ~1003/3584. The classic anisotropic-cone correction
   is not load-bearing for this table.
   ([global-geometry](observations/2026-06-10-emb-global-geometry.md))
2. **Input/output embedding orthogonality.** cos(E_i, U_i) ~ 0 for every
   token (mean +0.0017) while U-space carries its own category structure —
   the untied matrices use unrelated coordinate systems.
   ([global-geometry](observations/2026-06-10-emb-global-geometry.md), F-G4)
3. **Category-coherence hierarchy.** Paradigm sets (digits +0.428, months
   +0.416, weekdays +0.400) >> function words (+0.24-0.29) > topics
   (+0.08-0.18) > connotation classes (the `positive` class of the valence
   supergroup +0.045, `formal` of register +0.052).
   ([category-structure](observations/2026-06-10-emb-category-structure.md))
4. **Cross-script neighbors.** 法国/巴黎 rank among ' France'/' Paris' top
   neighbors — multilingual alignment exists in the raw lookup table.
   ([category-structure](observations/2026-06-10-emb-category-structure.md), F-C3)
5. **Relations are class-offsets with a thin paired residue.** All 11 pair
   kinds beat a within-kind permutation baseline, but the baseline absorbs
   most of the direction; pair-specific margins are +0.02-0.05, largest for
   morphology.
   ([pair-directions](observations/2026-06-10-emb-pair-directions.md))
6. **Full-population (all 149,706 alive rows): exactly one entangled
   dimension block** — 21 correlated dims (|r|>0.3), cross-script
   (','/'，', ' the'/'的' — pairs matched by *role/block loading*, not
   translation equivalence: '的''s nearest-cosine token is ' of' [0.63],
   consistent with its genitive use, and with the block dims removed it
   aligns with ' of'/"'s" over ' the' — [de-cosine-check](observations/2026-07-21-emb-de-cosine-check.md),
   audit §11), head-loaded by frequency (first
   token-id decile carries 1.5x the block norm-fraction floor; Spearman
   -0.206 vs -0.003 control); outside it,
   dimensions are near-independent (|r| mean 0.021, kurtosis median 0.32).
   Handles are precise at vocab scale (negative -> ' shitty'/' nasty';
   code -> 'namespace') with conservative recall; the kNN graph yields
   crisp islands (names, countries, code syntax, a cross-lingual time
   community) over one giant component.
   ([fullvocab-sweep](observations/2026-06-10-emb-fullvocab-sweep.md))
7. **Sink machinery is dimensionally disjoint from the W_E block.** Qwen's
   massive-activation dims (458 peak −12,608, 2570, 1427 — three of arc 1's
   layer-20 "sink dims") arise from layer 1 on the FIRST token, never
   delimiters; block dims ∩ massive dims = ∅.
   ([trace-block-through-layers](observations/2026-06-11-emb-trace-block-through-layers.md))
8. **Block content MOVES at layer 1 — re-encoded, not dissolved.** In-block
   correlation mass collapses 0.109 → 0.031 at L1, but a basis-free carrier
   analysis keeps top-SV in [8.8, 15.6] over all layers vs a decaying
   control; stable mid-network carrier set L4-26; fresh output re-encoding
   at L27-28 (norm gain 3.28×). Routing is RMSNorm gains + attention, not
   FFN weight structure.
   ([trace-block-through-layers](observations/2026-06-11-emb-trace-block-through-layers.md))
9. **Delimiter tracking is a distinct early-layer head population.** 26/784
   heads give delimiters ≥3× the offset-matched control attention to
   preceding delimiters (layers 0-3; top L0H13 0.178 vs 0.041). These are
   NOT the block-reader heads (top-10 overlap 2/10; reader L0H15 ranks
   21st) — reading the block and aggregating delimiters dissociate. The
   matching is carried ~99% by near-DC RoPE bands (static content match,
   falsifying positional-resonance P1d), and period→comma aggregation peaks
   at L0, not deeper (falsifying P1c).
   ([trace-delimiter-attention](observations/2026-07-15-emb-trace-delimiter-attention.md))

## Limitations

- **L1. Single model, single revision.** Everything is Qwen2.5-7B-Instruct
  @ a09a3545. The isotropy null especially needs a second model before any
  "modern models are like this" reading (cheap: TinyLlama, cached locally).
- **L2. Curated battery, prototypical members.** 690 hand-picked words;
  gaps quantify these anchors, not the vocabulary; single-token attrition
  (25 drops) skews multilingual coverage toward de/es/zh.
- **L3. Dead rows in global stats.** 1,959 near-zero rows (1.3%) are
  included in mu/covariance/random sampling; estimated effect < 0.002 on
  headline cosines but unverified — re-lock excluding them is a follow-up.
- **L4. The audit is arithmetic-consistency only.** 99 PASS means the
  observation numbers match the committed artifacts — not that the capture
  protocol, thresholds (MIN_CLASS_N=5, near-zero 1e-3, primary-variant
  policy), or interpretations are right.
- **L5. bf16 source precision** bounds all cosines at ~1e-3; the rank-~1700
  spectrum cliff may be a quantization artifact (open question H2 in the
  global-geometry observation).

## Deferred follow-ups (arc closed 2026-07-15; none of these block closure)

Highest-value first; the first two are natural openers for a successor arc:

- **T4 — runtime ablation of the 21 W_E block dims** (the causal arbiter;
  pre-registered P1e also needs it). F-T1 predicts sink formation survives;
  the T2 dissociation means ablation now has TWO distinct head populations
  to read out (block-readers vs delimiter-trackers), plus the carrier SV
  profile as a degradation measure.
- **Norm-normalized RoPE-band cosine** — separates "big activations sit in
  near-DC bands" from "alignment happens there" (T2 P1d caveat); needs a
  re-capture storing per-band norms, and a pure comma→comma accumulator
  would fix the P1c proxy limitation in the same run.
- **P3 population test** (pre-registered, not run) + corpus scaling beyond
  51 probes before promoting carrier-dim identities (thorough-data
  discipline).
- **Carrier identity in vocab space** — what do mid-network carriers
  (1445, 1865, 2545…) read as through W_U (trace H3); L27 re-encoding vs
  frequent-token dims 1069/46 (trace H4).
- **Held-out projection test** of the "handle" framing: project unseen pairs
  (' Stockholm' - ' Sweden') onto the capital_of direction (pair-directions
  H2) — cheap and decisive.
- **Layer-20 bridge**: same battery subset through arc 1's committed
  `vocab_atlas.pt`; do connotation classes (weak at L0) become strong at
  L20? (category-structure H2; script `emb_layer20_bridge.py`, not yet
  written.)
- **Cross-model isotropy check** on TinyLlama-1.1B (global-geometry H1).
- **Dead-row-excluded re-lock** (L3).
- **Multilingual alignment subspace** from exonym pairs (category-structure
  H3).

## Reproducing

```bash
git lfs install && git lfs pull
python examples/emb_audit_findings.py        # SUMMARY: 99 PASS | 0 FAIL
python examples/emb_data_manifest.py --check # 15 files, sha256 match
python examples/emb_global_render.py         # figures re-render model-free
python examples/emb_trace_render.py          # fig16-18 (model-free)
python examples/emb_trace_attention_analyze.py  # T2 P1a/P1c/P1d (model-free)
python examples/emb_trace_attention_render.py   # fig19-21 (model-free)
# full re-capture (needs the pinned model locally; ~10 CPU-min each):
python examples/emb_capture.py --tokenize-only   # battery coverage pre-flight
python examples/emb_capture.py
python examples/emb_trace_capture.py             # T0/T1
python examples/emb_trace_components.py          # T1.5 (51 hooked passes)
python examples/emb_trace_attention.py           # T2 (eager attention)
```

## File map

```
research/arcs/03_embedding-atlas/
  README.md                                   # this file
  observations/
    2026-06-10-emb-global-geometry.md         # isotropy null, dead rows, PCA, E-vs-U
    2026-06-10-emb-category-structure.md      # coherence hierarchy, connectivity, neighbors
    2026-06-10-emb-pair-directions.md         # relation-direction consistency
    2026-06-10-emb-fullvocab-sweep.md         # 21-dim block, handle precision, kNN islands
    2026-06-11-emb-trace-block-through-layers.md  # T0/T1/T1.5: sinks, readers, carriers
    2026-07-15-emb-trace-delimiter-attention.md   # T2: P1a/P1c/P1d adjudication
    figures/ (fig1-fig21 + INVENTORY.md)
  plans/    (arc plan, fullvocab plan, rope-vis plan, lit review, predictions)
  sessions/ (2026-06-11 tracing checkpoint; 2026-07-21 degradation-window forensics)
  data/ (14 .pt + MANIFEST.json + README.md)  # git-LFS, ~96 MB
```

Scripts (all under `examples/`): `emb_token_battery.py` (battery as data),
`emb_capture.py` (single model-loading step), `emb_category_stats.py` /
`emb_pair_directions.py` / `emb_fullvocab_stats.py` / `emb_fullvocab_analyze.py` /
`emb_structural_block.py` (derives), `emb_*_render.py` + `emb_neighbors_report.py`
(figures/report), `emb_trace_capture.py` / `emb_trace_components.py` /
`emb_trace_attention.py` (tracing captures, model-loading),
`emb_trace_corpus.py` (51-probe corpus as data), `emb_trace_analyze.py` /
`emb_trace_attention_analyze.py` (tracing derives, model-free),
`emb_audit_findings.py` (audit), `emb_data_manifest.py` (manifest),
`_emb_artifacts.py` (path resolver).
