# Embedding atlas — semantic structure of Qwen2.5-7B's input-embedding table

**Research question:** What semantic/functional category structure exists
*already* in the static input-embedding table (W_E, "layer 0") of
Qwen2.5-7B-Instruct — do word groups related by topic, function, morphology,
or connotation share directions that could act as "handles" for downstream
layers — and how does that structure compare to the layer-20 geometry found
in the [nla-verbalizer arc](../01_nla-verbalizer/README.md)?

**Status:** CLOSED for new experiments 2026-07-15 (deep arc, user direction
2026-06-11; started 2026-06-10). Seven observations; audit at **99 PASS /
0 FAIL** (measured 2026-08-17,
[`data/audit_2026-08-17.log`](data/audit_2026-08-17.log)).

All three phases landed: phase 1 (battery protocol probes), phase 2
(full-vocabulary sweep, 149,706 alive rows), and phase 3 (structural tracing
T0/T1/T1.5/T2, absorbing the former rope-vis arc per
[plan](plans/2026-06-10-rope-vis.md)).

Pre-registered predictions
([plans/2026-06-11-predictions.md](plans/2026-06-11-predictions.md))
adjudicated: P1a PASS, P1c FAIL, P1d FAIL, P2 refined-not-falsified (content
moves to a new basis at L1), P1b/P1e/P3 not run (deferred, below). Remaining
follow-ups are recorded under "Deferred follow-ups" and are candidates for a
successor arc rather than this one. Provenance: the 2026-06-10/11 kickoff
sessions ran on `claude-fable-5` within days of that model's release, so a
reported (but here **unconfirmed** — see the note's sourcing breakdown)
initial-release quality-degradation window would have covered them; audited
clean at the user's direction —
[degradation-forensics](sessions/2026-07-21-degradation-forensics.md).

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Shape, provenance labels and public-repo constraints per
[`ARC_PROCESS.md` § Attribution](../../ARC_PROCESS.md#attribution--who-directed-who-executed).
Quotes are from the transcripts listed under **Verifiability**, normalized
for punctuation (`--` → `—`) with markdown emphasis dropped; any `[...]`
marks an editorial elision, and the elision is stated wherever it removes
something load-bearing.

### Research direction

**The originating "handles" question** [session 2026-06-10] `[NORMALIZED]`:

> *"explore the complete set of token embeddings for correlations and
> interesting connections between groups of words (related in one way or
> another — like we did with countries, but also things like check for
> similarities/correlations/features for different types of words different
> subjects, try out). Words with similar or related
> function, use, topic/connotation, etc. may have shared features that act
> as 'handles' for the model layers to execute that function hidden in our
> construction of language."*

Scope decisions made at the same session start: two arcs (embeddings first,
RoPE second) rather than one; iterate with figures + plain-language
measurement explanations at checkpoints.

**Broaden before deepen (CP1)** [session 2026-06-10] `[SELECTED]`. At
checkpoint CP1 the user chose "Add more word groups first" from the
next-step options Claude offered — a checkpoint decision, not a typed
message. The *direction* to broaden beyond countries into "different types
of words / different subjects" is the user's, in the originating turn above;
the specific
person/royal/religion/abstract/landscape/instrument/science/tech/language
classes and the gender/antonym/past/capital_of/lang_of pair kinds are
Claude's operationalization of it.

**The thorough-data standard** [session 2026-06-10] `[NORMALIZED]`:

> *"I want to really focus on thorough data collection before we come to
> any conclusions. Run it in scaling steps and branch out (and push notify
> me) if data reveals any potential unexpected paths. That is always how we
> should approach things. Plan things out as deeply and thoroughly as you
> want, collect as much data as you want, but we shouldn't be generalizing
> from a small sampling of data like we've been doing, ever."*

This is the direction that forced the full-vocabulary sweep — all 149,706
real-token rows — before any claim promotion: battery-scale results became
protocol probes rather than conclusions, and findings prose states its
population coverage explicitly. Two typos are normalized here; the turn
closed by asking that the standard be written into the repo's standing
guidance, which is where the research-data-discipline rule came from.

### Human / Claude / emergent split

**User (Michael Lannum).** The originating "handles" question (quoted
above, [session 2026-06-10]); the two-arc scoping and the
iterate-with-figures cadence; the CP1 broaden-before-deepen decision; the
deep-arc commitment and the numbered-arc reorganization (2026-06-11); the
wrap-up call that closed the arc (2026-07-15); the challenge to finding #6's
`' the'/'的'` notation that produced the de-cosine check (2026-07-21); the
review gate (all three stack PRs human-merged, never auto-merged — visible in
the merge record independently of any transcript).

`[RECONSTRUCTED]` — **the pause/resume calls across the tracing phase.**
Written from recollection, not from a recovered turn: a search of both
surviving sessions (2026-06-10 → 06-11 and 2026-07-15 → 07-18, which
between them cover every day this arc has a commit) finds no turn making
these calls — the one pause instruction on record is the 2026-06-10
direction to pause *arc 02*. The substance is plausible against the
checkpoint commit of 2026-06-11, but nothing is quoted and the wording is
Claude's.

**Claude Code.** All implementation: the 25 `emb_*`
capture/derive/render/audit scripts (plus the `_emb_artifacts.py`
resolver), the battery class
definitions (under the CP1 direction), the operationalization of the
pre-registered predictions (P1a-P3), the tracing-phase experiment designs
(T0/T1/T1.5/T2), the literature review with adversarial novelty verification,
all figures and observation write-ups, and the audit that re-derives every
published number from committed artifacts.

**Emergent.** The 21-dim entangled block itself (the user's handles framing
predicted shared structure; the full-vocab sweep implementation surfaced this
specific object); the reader/tracker head dissociation; the P1c/P1d
falsifications — adjudicated mechanically against criteria pre-registered on
2026-06-11, before any attention weights were captured.

### Verifiability

Every quote above is recoverable from the sessions below. The transcripts are
machine-local and are not committed to this repo — they carry local paths and
tool output — so they are referenced by session id and date only. The ids are
**abbreviated**: each is the 8-character prefix of the local session UUID.

| Session | Span | Covers |
|---|---|---|
| `3f3f013a` | 2026-06-10 → 06-11 | kickoff, CP1, deep-arc commitment, the thorough-data standard |
| — | 2026-06-12 → 07-14 | no session, and none expected: the arc was dormant, with no commit between 06-11 and 07-15 |
| `ca232e08` + `87acda5a` | 2026-07-15 → 07-18 | wrap-up and close |
| `426003e2` | 2026-07-21 | degradation-forensics direction |

Claims in this section that are not in quotation marks are Claude's
characterization of the user's direction, not the user's wording. The
three-way sourcing split used in
[`sessions/2026-07-21-degradation-forensics.md`](sessions/2026-07-21-degradation-forensics.md)
— confirmed / not-tier-A-backed / corroboration-only, stated per claim — is
the precedent this labeling scheme generalizes.

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
   (+0.08-0.18 over the eight topic classes named in F-C1's strong/typical
   tiers, `animal` +0.078 to `country` +0.179; months and weekdays are
   nominally topic classes too but sit far higher and are reported above as
   paradigm sets) > connotation classes (the `positive` class of the valence
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

## Known corrections and errata

Every correction applied to this arc after its 2026-07-15 close, in one place,
so a reader does not have to reconstruct them from the section they happen to
land in.

- **F-T3 carrier-stability numbers, corrected 2026-08-17.** Two hand-computed
  numbers in the L4-26 regime did not re-derive from
  `emb_trace_components.pt`: the adjacent-layer top-10 carrier overlap is
  **4-9/10 (median 8)**, not 7-9/10, and **two** original block dims (2604,
  1395) hold a top-10 slot at every layer of the band, not three — 1122 drops
  out over L5-L16. Corrected in place; finding #8's substance (a stable
  mid-network carrier set spanning L4-26) is unaffected, and both corrected
  values are now locked by AUDIT 9 — see
  [the Correction section](observations/2026-06-11-emb-trace-block-through-layers.md#correction--f-t3-regime-iii-carrier-stability-numbers-2026-08-17).
- **Full-vocab transcript label renamed 2026-07-21.** The
  `emb_structural_block.py` print label `energy` was renamed `norm_frac`; the
  quoted transcript in
  [fullvocab-sweep](observations/2026-06-10-emb-fullvocab-sweep.md) carries
  the new label and the values are unchanged.

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
                                             # (measured 2026-08-17, data/audit_2026-08-17.log)
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
    2026-07-21-emb-de-cosine-check.md         # '的' neighbor ordering under block ablation
    figures/ (fig1-fig21 + INVENTORY.md)
  plans/    (arc plan, fullvocab plan, rope-vis plan, lit review, predictions)
  sessions/ (2026-06-11 tracing checkpoint; 2026-07-21 degradation-window forensics)
  data/ (15 .pt + MANIFEST.json + README.md + LICENSE-DATA.md
         + audit_2026-08-17.log)   # git-LFS, ~96 MB
```

Scripts (all under `examples/`): `emb_token_battery.py` (battery as data),
`emb_capture.py` (single model-loading step), `emb_category_stats.py` /
`emb_pair_directions.py` / `emb_fullvocab_stats.py` / `emb_fullvocab_analyze.py` /
`emb_structural_block.py` / `emb_de_cosine_check.py` (derives),
`emb_*_render.py` + `emb_neighbors_report.py`
(figures/report), `emb_trace_capture.py` / `emb_trace_components.py` /
`emb_trace_attention.py` (tracing captures, model-loading),
`emb_trace_corpus.py` (51-probe corpus as data), `emb_trace_analyze.py` /
`emb_trace_attention_analyze.py` (tracing derives, model-free),
`emb_audit_findings.py` (audit), `emb_data_manifest.py` (manifest),
`_emb_artifacts.py` (path resolver).
