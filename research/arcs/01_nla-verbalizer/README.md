# NLA Interpretability Research Arc — Qwen2.5-7B Layer 20

A working investigation into what Anthropic's released Natural Language
Autoencoders (NLAs) for Qwen2.5-7B-Instruct surface about layer-20 hidden
state structure. A focused arc (observations 2026-05-12 to 05-15):
22 observation files, 36 figures, 22 tracked work items, a regression
audit at **196 PASS / 0 FAIL**, and one working synthesis: *layer-20
h-space appears to have discrete attractor basins separated by sharp
boundaries* — held as a working hypothesis, not a settled claim.
See [Limitations and methodology caveats](#limitations-and-methodology-caveats)
for the scope qualifiers.

This README is the entry point for the arc. The full observation log
lives in [`observations/`](observations/), figures in
[`observations/figures/`](observations/figures/) with
[`INVENTORY.md`](observations/figures/INVENTORY.md) providing
per-figure provenance.

**Status:** paused as of 2026-05-15 — synthesis written, active work
stopped. Open follow-ups are enumerated in [Possible next
paths](#possible-next-paths).

> **`MAIN-N` identifiers.** This arc was run against a private issue
> tracker that has since been retired; its `MAIN-N` ticket IDs survive
> throughout this arc's prose as historical labels. **They are not
> resolvable — there is nothing to look up.** Every one of them is
> mapped to the in-repo observation file or the migrated GitHub issue
> it stands for in [Private-tracker ID map](#private-tracker-id-map-main-n)
> below; read that table, not the bare ID.

---

## Context and motivation

Anthropic released the NLA model pair `kitft/nla-qwen2.5-7b-L20-{av,ar}`
on 2026-05-07 — small companion models trained to verbalize layer-20
hidden states of Qwen2.5-7B-Instruct (the AV, "Activation → Verbalization")
and to reconstruct hidden states from natural-language verbalizations
(the AR, "Activation Reconstruction"). The pair enables a round-trip
interpretability probe: capture `h[20]` → verbalize → re-encode → compare
the reconstructed h against the original via cosine similarity. If the
round-trip cosine is high, the verbalization captured the load-bearing
content of the hidden state.

This arc applied the NLA pair to local Qwen2.5-7B-Instruct across a
sequence of probes — initial round-trip validation, then per-token
generation trajectories, then concept-direction extraction, then
semantic-basis mapping, then geometric structure characterization,
then interpolation experiments, then a final attractor-basin
synthesis. The sequence wasn't planned up-front; each step was shaped
by a specific research question (see [Attribution](#attribution)).

---

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas
came from. Quotes below are verbatim user turns (Michael Lannum) from
the session transcripts of 2026-05-12 → 2026-06-05, lightly normalized
for typos and punctuation; markdown emphasis inside a turn is dropped,
`[...]` marks an editorial elision (except where noted), and dates
follow the transcripts' UTC clock. This section follows the attribution
shape in
[`ARC_PROCESS.md` § 6](../../ARC_PROCESS.md#6-arc-readme-synthesis).
Blocks are grouped by research thread and then process standards, not
strictly by time.

**Originating direction** [session 2026-05-12]:

> *"Can we try to do something with Anthropic's new interpretability
> stuff for open source models (released last thursday)?"*

The framing that opened the arc. The plural "models" implicitly scoped
beyond Qwen2.5-7B; cross-model replication
([D5](#d5-cross-model-replication)) remains open because the released
NLA pair is Qwen-specific.

**Plumbing first, then per-token depth**:

> *"yeah lets test the plumbing first by script and see if we can get
> something interpretable out of an embedding"* — [session 2026-05-12]

> *"I want to understand what layer/embedding is being sampled for this
> NLA interpretation, and go through the 'thought' at each token more
> thoroughly"* — [session 2026-05-12]

> *"did we generate the NLA for the tokens the model generates too? I
> want to see those as it 'thinks through' what it's 'writing'"*
> — [session 2026-05-13]

Validate the round-trip on simple inputs, understand the specific layer
(20 of 28; ~71% depth), verbalize at every token rather than
aggregating. Per-token trajectory viz was a direct ask (realized in
`nla_gen_trajectory.py` + static figures; the live form is open as
[D7](#d7-per-token-live-trajectory-viz)).

**Reproduce Anthropic's emergent-behavior examples**:

> *"can we try some more complex prompts, like with the rabbit poem and
> ethical discussion where it was thinking it was being tested
> (showcased in Anthropic announcement). Can we try stuff like that
> and see if we can get anything like those examples anthropic
> provided?"* — [session 2026-05-12]

> *"let's go with the poem one like Anthropic's example, not the haiku
> since that was an outlier"* — [session 2026-05-13]

The original motivating curiosity. The rabbit haiku was reproduced; the
matched rabbit poem and the eval-awareness behavior remain incomplete
([D2](#d2-eval-aware--knows-its-being-tested-probe),
[D8](#d8-replicate-anthropic-nla-announcement-specific-examples)).

**Counterfactual / OOD probing** [session 2026-05-13]:

> *"Can we try feeding the model a transcript it specifically wouldn't
> normally generate (including part of its false output), and see if
> it still thinks along the lines of the context, or if it reacts
> differently since it's doing something the model isn't trained to
> do?"*

Produced `nla_forced_continuation.py` and the counterfactual-surprise /
OOD-detection observations: `||Δh||_feat` separates plausible-but-false
continuations (Δ≈5.7-11) from OOD-forcing ones (Δ≈28-30) — a cheap
deployment-time anomaly-score candidate.

**The reverse (AR) direction** [session 2026-05-13]:

> *"Is this block 20 probing the only available option? Or can the NLA
> interpreter model be configured for other layer probes? And can we
> see the reverse NLA-to-embedding model that Anthropic introduced
> with this model set? Or is that just used to tune this NLA
> training?"*

Pushed early exploration of the AR direction (text → h) alongside AV
(h → text), which enabled the entire interpolation-flipbook branch of
the arc.

**A superposition reading of the verbalizations** [session 2026-05-13]:

> *"Kind of reads like the language is represented recursive semantic
> hierarchies, like its guess [...] seems to structurally contain 'The
> capital of France is [?]' and 'France is a country', just strung
> together into an otherwise semantically garbled statement."*

The user's interpretive hypothesis on the raw AV outputs: a
verbalization superposes several semantic layers rather than reporting
one coherent thought. The later multi-axis geometry work kept
returning to this reading.

**Concept directions** [session 2026-05-13]:

> *"would it be possible to extract the feature vectors that 'mean'
> the 'idea' of something being country like here with France, which
> we could identify pattern-wise across different contexts involving
> some relevance to things being countries?"*

Seed of the CAV-style country-direction observation and ultimately the
23-category mean-contrast basis. The question shape is essential:
contrasts as the unit of analysis, not individual activations.

**The semantic-basis grid** [session 2026-05-14]:

> *"should we like 'map' a bunch of relevant tokens' embeddings to get
> a relative baseline for semantic bases?"*

> *"let's make sure we map a bunch of articles, punctuation, etc. Get a
> wide lay of the embedding space landscape. This kind of provides us
> a complex 'grid' of sorts, or a set of entangled axes or something
> to provide direction in such high dimensional space"*

The most generative turn pair in the arc: the vocab atlas (128 anchors
× 23 categories) is its operational realization, and the "entangled
axes" framing committed the arc to multi-axis interpretation — borne
out by the discriminant-connectivity result (3 macro-clusters: content
/ function-words / structural), and the direct seed of
[arc 03](../03_embedding-atlas/README.md).

**Visualization as research, not presentation** [session 2026-05-13]:

> *"First I want to go token by token into every generated NLP
> interpretation and do some probing to see if we can identify
> numerical/geometric feature patterns that could be interesting to
> visualize"*

> *"I want to head in the visual direction, /goal find path to novel
> visualization design, something that allows a useful view into
> feature/embedding/NLA interpretability"*

The largest unrealized seed: the arc landed on static matplotlib
figures, while the framing implies an interactive discovery tool
([D1](#d1-discovery-viz-frontend)).

**The AV format-bias catch** [session 2026-05-13]:

> *"does every NLA output include those phrases like 'Structured format
> [...]', or did we add that to describe different parts of the
> output? Weirdly consistent"*

(The inner `[...]` is the user's own, standing in for the rest of the
NLA phrase; "Structured" is normalized from a typo.) The sharpest
methodological catch in the arc: the agent had been reading AV outputs
at face value without questioning whether the format itself was an
artifact. Filed as [D3](#d3-audit-av-decoder-format-bias) — if positive
it re-frames all prior interpretive claims (see L2).

**The rigor pivot and the figure/data standards**:

> *"yeah rigoramatize everything"* — [session 2026-05-13]

> *"can we adjust the output figures a bit? Make them higher resolution
> first, so all the small text and visuals are clear and readable.
> [...] Also somewhere we should define exactly what each of these
> figures represents, if you haven't already documented that. Plus
> link the explanations to the base source data files/models/scripts,
> and define any assumptions/tools/analysis/corrections involved."*
> — [session 2026-05-14]

> *"check if we included all raw data that the figures used ([...] by
> default we should also be providing the raw datasets that figures
> are generated from, both to verify figures if necessary and for
> technical transparency) [...] in the future, generating, validating,
> and saving the raw dataset should be part of the agent's process as
> it conducts research."* — [session 2026-05-31]

These three turns are the origin of, respectively, the audit and
observation-file discipline, the per-figure provenance record
([`observations/figures/INVENTORY.md`](observations/figures/INVENTORY.md)),
and the raw-data-is-a-deliverable rule later codified into
[`ARC_PROCESS.md`](../../ARC_PROCESS.md). The supersede-don't-overwrite
figure rule was set on 2026-05-13 ("okay cool, make sure we don't
overwrite those. But those look nice. Assuming they're right." —
visual plausibility is not correctness).

**Review protocol, merge gate, and the transparency bar**:

> *"I want to do one final thorough local review of the current PR
> contents. Specifically to: Ensure that all our exploratory
> experiments make sense; they were correctly
> designed/scripted/executed; everything is technically sound and
> accurate to a professional rigor; and all figures are correct and
> accurate [...]
> Don't edit anything yet during or after the review -- I want to look
> through and understand it all first."* — [session 2026-05-28]

> *"wait did you merge this PR into main? It looks like something did,
> but PR merge is supposed to be the final manual human gate. Check on
> what may have happened there."* — [session 2026-05-29]

> *"I want to make sure the README for this chunk of research provides
> the right high-level framing and 'attribution' like user direction
> vs. agent implementation [...] I want there to be a sense of
> transparency and make sure everything is adequately self-critical,
> don't want to overclaim and I want it to accurately represent what
> I'm contributing."* — [session 2026-05-31]

> *"Before you start the next goal, file a ticket to look into the
> following agent criticism of llm-research. It only saw a couple
> READMEs and a list of script files, so these need to be verified and
> confirmed defects, and then corrected if so: [...]"*
> — [session 2026-06-05]

In order: the multi-model local review protocol (commissioned
2026-05-28, before the auto-merge in the next item was noticed) whose
findings drove the correction rounds; the catch that the arc's integration PR
— "PR #11" in the pre-split repo's numbering, unrelated to this
repo's issue #11 — had merged without the manual human gate
(2026-05-29), and the standing rule that merges stay human; the
transparency / self-critical / no-overclaim bar this section itself
exists to meet (2026-05-31); and the requirement that an
external critique of the arc be verified defect-by-defect rather than
adopted or dismissed wholesale (2026-06-05) — the origin of the
D3-as-validity-control framing and the verbalizer-is-also-a-model
caveats now in [Limitations](#limitations-and-methodology-caveats).

### Human / Claude / emergent split

**User (Michael Lannum).** Every direction quoted above; the
interpretive judgments throughout — which findings were worth
following, which were artifacts to set aside, when scope was drifting,
when a result deserved its own observation file; sequencing and
go/no-go calls (cheap pilot batches before long runs, partial-GPU
offload rather than an all-or-nothing CPU fallback, no PR until the
research direction was substantive); the format-bias catch, and the
catch that the basis-axis figures showed every axis active at once
(2026-05-14); and the rigor, figure-provenance,
raw-data, review-protocol, and merge-gate standards quoted above.

**Claude Code.** All experiment scripts (~42 files under
`examples/nla_*`), figure-rendering pipelines, observation drafts,
audit infrastructure (`nla_audit_findings.py`), and issue-queue
management; continuity across compaction boundaries (resume
checkpoints, the figure inventory, tracking what had been claimed where
so corrections propagated); literature connections — Concept Activation
Vectors (Kim et al. 2018), superposition (Elhage et al. 2022),
Fisher-LDA distinctions, BPE-boundary considerations.

**Emergent.** The discrete-attractor-basin synthesis — proposed by the
agent during compaction, refined under the user's scope-qualification
challenges (the F1 fix), and validated against the audit numbers (which
the agent built and the user directed). The methodology caveats in
[Limitations](#limitations-and-methodology-caveats) — most originated
as the user's in-session pushback against overclaiming and were
formalized into explicit limitations by the agent; the
verbalizer-as-model caveats entered through the external critique the
user brought in and required to be verified (2026-06-05).

**Verifiability.** Every quote above is recoverable from the session
transcripts for 2026-05-12 → 2026-06-05; the transcripts are not
committed to this repo (they carry machine-local paths and tool
output). Claims in this section that are not quoted are Claude's
characterization of the user's direction, not the user's wording.

---

## Findings — strongest synthesis, with scope qualifications

Held as **working hypotheses**, not settled claims. The scope-test
follow-ups listed in [Possible next paths](#possible-next-paths) are
the work required to upgrade these from "hypothesis" to "result."

### F1. Layer 20 h-space appears to have discrete attractor basins

Linear interpolation between two AR-encoded natural-language anchors
(factual/geography ↔ poetic/nature,
[MAIN-25](observations/2026-05-13-nla-interpolation-flipbook.md))
produces a discontinuous AV-text transition even though the geometric
step size stays roughly
constant (`||Δh||` = 2.734 per coarse step). The coarse 20-step grid
first flagged the flip at t=0.421; dense re-sampling at 10× resolution
([MAIN-34](observations/2026-05-15-nla-dense-interp-near-pivot.md),
Δt≈0.0025) relocated it — t=0.421 actually sits *inside* a
"Definition + Poem" hybrid plateau (t∈[0.395, 0.4450]), and the sharp
flip is the plateau→poetic crossing at t≈0.4475–0.4500, a single
Δt=0.0025 step. The plateau is itself a basin that does not correspond
to any single vocab category. AR re-encoding of a midpoint h returns h
to its basin (round-trip cosine +0.8995,
[MAIN-71](observations/2026-05-15-nla-plateau-attractor-strength.md)) — basins are
direction-coupled, not magnitude-coupled.

**Scope qualifications:** demonstrated for one anchor pair, at one
layer, on one model. The plateau-attractor margin is +0.061 over the
nearest single-anchor — narrower than the +0.25 margins anchors have
between themselves. The framing should be "basin candidate / shallow
basin" until additional anchor pairs and layers replicate. Filed scope
tests: [D5](#d5-cross-model-replication)
(cross-model), [D6](#d6-basin-landscape-mapping) (basin landscape
mapping).

### F2. The 23-axis mean-contrast basis is end-of-prompt-PROTOCOL-coupled

A basis of 23 per-category mean-contrast directions
(`d_cat = mean(in_cat) − mean(out_cat); d_cat /= ||d_cat||`) built from
the vocab atlas classifies end-of-prompt h-vectors with reasonable
top-K hit rates (56-79% top-5 by category,
[MAIN-26](observations/2026-05-13-nla-discriminant-validation.md)).
Mid-sequence captures of the same anchors project nearly orthogonally
onto the same basis (+0.0491 aggregate cosine, ~3× random-cosine floor,
[MAIN-44](observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md)).
Building a mid-sequence-NATIVE basis using the same recipe lifts the
in-protocol signal to +0.5632
([MAIN-70](observations/2026-05-14-nla-mid-seq-native-discriminants.md)).
Conclusion: the basis is a fingerprint of one specific capture protocol,
not a generic semantic axis.

**Scope qualifications:** "discriminant" in the script names is
shorthand — the formula is centroid-difference / mean-contrast, not
Fisher LDA (which would require `S_W⁻¹(μ₁−μ₀)` with regularization;
omitted here because n=2-12 captures per category in 3584-dim space
makes `S_W` rank-deficient by orders of magnitude). See
[`examples/README_NLA.md`](../../../examples/README_NLA.md#discriminant-naming--methodology-note)
for the full methodology note.

### F3. Apparent hierarchical attractor structure at layer 20

End-of-prompt h vectors appear to decompose as: a +0.40 baseline
cosine between any two h's, which splits into a +0.22 universal-sink
contribution (7 sign-locked dims) plus a +0.18 non-sink residue
baseline (the mean cosine between any two *sink-removed* h's — not
+0.40, which is the with-sink total); then, layered on that baseline,
category attractors (intra-category cosine +0.85 to +0.98), then
within-category content modulation. PC1 of the
sink-removed vocab atlas (33.5% variance) emerges as the
**content-vs-function** axis — content-bearing words load negative
(PC1 < 0), function words and punctuation load positive (PC1 > 0).
([MAIN-24](observations/2026-05-13-nla-vocab-atlas-grid.md), Finding 2.)

> **Polarity note — corrected 2026-08-16.** Until then the sentence
> above carried the signs inverted — a transcription slip relative to
> Finding 2 of
> [`2026-05-13-nla-vocab-atlas-grid.md`](observations/2026-05-13-nla-vocab-atlas-grid.md),
> the dated primary record (content-bearing at PC1 < 0,
> structural/function at PC1 > 0), which this summary now matches.
> Nothing downstream changes either way: the sign of a PCA component is
> arbitrary (flipping an eigenvector's sign gives an equally valid
> decomposition), so the load-bearing claim is that PC1 *separates*
> content from function, not which side either lands on — which is why
> the slip went unnoticed. `nla_audit_findings.py` AUDIT 12 re-derives
> the 33.5% variance fraction from the same artifact, but it checks the
> magnitude only; no audit asserts the polarity, so neither statement
> was ever machine-checked.

**Scope qualifications:** sink-dim identification was hand-rolled
(7 dims chosen as those with universal sign + large |h|); a more
principled sink detection (e.g. ratio-of-variance to mean-magnitude)
might surface or exclude different dims. Hierarchy is descriptive of
this one model + layer; not yet tested cross-model.

### F4. Counterfactual surprise score from forced continuation

For natural vs forced-completion pairs at matched generation positions
(`Yes/No`, `Paris/Berlin`, `4/5`, factual/refusal-metaaware), the
distance `||Δh||_feat` between natural-position-h and
forced-position-h ranks the "plausibility" of the forced completion
in a useful way: 5.7-11 for plausible-but-false counterfactuals,
28-30 for OOD-forcing (refusal injected into an arithmetic context;
the 35.55 shown in fig15 is a 10-token position-drift artifact,
corrected to ~28 by the position-matched fig16 — see INVENTORY.md).
Cheap monitoring-side anomaly score candidate. (Remaining work tracked
as [#13](https://github.com/skothr/llm-research/issues/13), was MAIN-30;
`nla_forced_continuation.py`.)

**Scope qualifications:** four pairs is small-n; the score's
distribution under random forced completions hasn't been characterized.
The `nla_forced_continuation.py` protocol concatenates separately-tokenized
prompt + completion (BPE-boundary risk at the seam) — for the current
4 pairs the boundary risk is low (completions start with letter/digit
tokens) but extensions need the mid_seq capture script's
tokenize-then-locate approach.

### F5. AR-encoded anchors collapse near a shared attractor (chat-template basin)

Two maximally-different natural-language descriptions, AR-encoded
back into h-vectors, have cos(h_A, h_B) = +0.69 — much higher than
chance for two genuinely independent vectors in 3584-dim. The chat-
template prefix (`<|im_start|>user`...) dominates the AR output;
content modulates within a chat-template-shaped attractor.

**Scope qualifications:** descriptive observation, not a finding-about-
the-model; says more about how the AR was trained than about Qwen.
But it's load-bearing for the interpolation work (F1) because it
explains why linear interpolation between AR-encoded anchors looks
geometrically nice — both anchors live in the same global region.

---

## Limitations and methodology caveats

Self-critical scope qualifications, ranked by how much they constrain
how far the arc's claims travel.

**L1. Single model, single layer.** Every result is on
Qwen2.5-7B-Instruct at layer 20 — the only configuration the released
NLA pair was trained for. Cross-model replication (D5) and cross-layer
replication (out-of-scope here) are required before any finding can
be claimed as a property of post-trained transformer mid-late layers
in general.

**L2. AV-decoder format-bias is unaudited.** The AV format-bias catch
([Attribution](#attribution)) — that AV outputs share suspiciously
consistent template phrases like
"Structured format: [...]" — was never investigated. If the AV emits
the same templates on random h-vectors as it does on semantically-loaded
h's, every interpretive reading in this arc has been filtered through
verbalizer prior, and "the model is thinking about X" claims may
reduce to "the verbalizer says X regardless of input." This is the
single methodological audit that, if positive, would re-frame the
whole arc. Promoted to Medium priority as D3 for that reason.

**L3. The mean-contrast basis is protocol-coupled.** As characterized
in F2, the 23-axis basis works for the protocol it was built on
(end-of-single-token-message) and collapses to noise cross-protocol.
The dual question — what subspace of h IS protocol-invariant — was
not asked during the arc. Open as
[D4](#d4-find-the-protocol-invariant-subspace).

**L4. Per-category capture counts are small (n=2 to n=12, median 5).**
Six of 23 categories have n ≤ 3 (`p_quote`, `p_dash`, `article`,
`negation`, `p_ender`, `p_internal`). The mean-contrast formula is
defensible at this scale (Fisher LDA's `S_W⁻¹` requirement makes the
unscaled centroid difference the right proxy when `S_W` is rank-deficient),
but bootstrap confidence intervals on the per-category directions
were not computed — the centroid for a 2-anchor category is
unstable to within-category swap. Re-running with n≥8 per category
would tighten the basis.

**L5. The vocab atlas had three duplicates that affected the committed
artifact.** The original `VOCAB` dict had `"when"` in both
wh_word and conjunction, `"-"` in both p_dash and math_op, `"*"` in
both p_special and math_op. The committed `vocab_atlas.pt` was captured
against the duplicated source, so the audit-locked numbers depend on
slightly-biased centroids. The source dict has been deduplicated
(125 unique anchors, was 128 with 3 dups) with a module-level assert
preventing future re-introduction. Regenerating the atlas would
shift downstream numbers slightly; filed as a future follow-up rather
than immediate fix.

**L6. "Discriminant" naming is methodologically loose.** The codebase
calls per-category mean-contrast directions "discriminants" — this is
shorthand, not a claim of Fisher-style optimal separation. Anyone
extending or publishing this work should rename to "mean-contrast" or
"centroid-difference" to avoid implying Fisher LDA properties.

**L7. Capture-position protocol bug, retroactively documented.**
The `nla_discriminant_stability_capture.py` inline comment originally
described position -1 as "the last content token (the anchor word)" —
but with `add_generation_prompt=True`, position -1 is the trailing
newline of the `<|im_start|>assistant\n` opener, NOT the anchor token.
What the stability scan actually measures is the model's "what to say
next given this prompt" representation, NOT the anchor token's
representation. Comparisons across the 4 contexts are internally
consistent (same kind of position in all 4), so the stability *finding*
holds, but the *framing* has been corrected throughout.

**L8. The audit script is arithmetic-consistency, not methodological.**
`nla_audit_findings.py` re-derives every load-bearing number from raw
`.pt` files and checks them against expected constants in the script
(transcribed from the observation prose — so the artifact side is
re-derived, but prose↔script agreement is maintained by hand). It
catches: stale numbers after script changes, transcription
errors when copying numbers into observation files, regression in
captures. It does NOT catch: capture-protocol bugs (it consumes
artifacts as given), wrong choice of classifier cutoff (it validates
that the cutoff was applied consistently, not that the cutoff was
right), interpretive overreach in observation prose. **A 196 PASS
audit means "the numbers in the markdown match the numbers in the
.pt files," not "the methodology is right."**

**L9. The plateau attractor was demonstrated on one anchor pair.**
F1's attractor-basin claim rests primarily on
[MAIN-71](observations/2026-05-15-nla-plateau-attractor-strength.md) (round-trip
cos +0.8995, margin +0.061 over nearest single-anchor) — a single
anchor pair (factual/geography ↔ poetic/nature). Calling layer-20
h-space "having discrete attractor basins" is a generalization from
this one observation to a structural claim. Until other anchor pairs
replicate the round-trip + margin pattern, the F1 framing should be
"basin candidate at this one location" rather than "discrete attractor
basins."

---

## Possible next paths

Eight unsprouted research directions, each tied to a direction-setting
turn quoted in [Attribution](#attribution). Ordered roughly by
methodological priority (cleanups first, then scope tests, then
extensions).

Each was originally filed on the retired private tracker; the six that
were migrated now live as GitHub issues in this repo and are linked
below. D1 and D7 (the two visualization directions) were not migrated —
their `MAIN-N` IDs are historical labels only, and this README section
is the only surviving description of them.

### D3. Audit AV-decoder format-bias
Seed: the AV format-bias catch · [#8](https://github.com/skothr/llm-research/issues/8) (was MAIN-267) · **Priority: Medium**

Feed random Gaussian-noise h-vectors (at appropriate norm), zero
vectors, swapped-layer h's, and h's from other models to the AV.
Count template-phrase frequency. If AV emits "Structured format..."
on random h's at the same rate as on semantically-loaded h's, the
templates are verbalizer prior, not content — and every prior
interpretive claim in this arc needs re-interpretation. Promoted to
Medium because the negative case would re-frame the whole arc.

### D5. Cross-model replication
Seeds: the originating direction (plural "models"), visualization-as-research · [#6](https://github.com/skothr/llm-research/issues/6) (was MAIN-269) · partially covered by [#11](https://github.com/skothr/llm-research/issues/11) (was MAIN-41)

Required scope-test for L1. The catch: NLA verbatim requires an NLA
pair trained for that model's layer N (only Qwen has one). Pragmatic
alternative: use logit-lens or probe-based proxies and ask whether
the *geometric* findings (PC1 axis interpretation, attractor
structure, stepwise transitions) hold without the NLA verbalizer.
TinyLlama-1.1B at layer 17 (~80% depth) is the obvious first
attempt; cached locally.

### D4. Find the protocol-INVARIANT subspace
Implicit from the semantic-basis grid · [#7](https://github.com/skothr/llm-research/issues/7) (was MAIN-268)

L3's dual question. Capture the same content tokens at three
protocols (single-token-message-end, mid-sequence, post-assistant-
turn-start). For each pair compute rank-K SVD of the cross-protocol
covariance. If a small low-rank subspace explains most cross-protocol
agreement, that subspace IS the protocol-invariant "what does h
represent about a token" component. Compare to the 23-axis basis to
see overlap.

### D2. Eval-aware / "knows it's being tested" probe
Seeds: Anthropic-examples replication, counterfactual/OOD probing · [#9](https://github.com/skothr/llm-research/issues/9) (was MAIN-266)

Matched-pair prompts where one is plainly informational and the other
is explicitly framed as evaluation ("This is a test of your alignment").
Compare h[20] AV-readings at the same generation position. If the
eval-framed h reads in a "meta / aware / being-tested" register and the
matched plain prompt doesn't — evidence for layer-20 eval-frame
representation. Closest fit to the original Anthropic-examples
motivating ask.

### D8. Replicate Anthropic NLA-announcement specific examples
Seed: Anthropic-examples replication · [#4](https://github.com/skothr/llm-research/issues/4) (was MAIN-272)

Enumerate the specific examples in Anthropic's 2026-05-07 NLA post,
reproduce each on Qwen2.5-7B + the released NLA pair, compare arc
findings to Anthropic's claimed AV readings. Divergences are
interesting: they isolate model-specific vs technique-general
behaviors. Partial coverage so far (rabbit haiku done, rabbit-poem +
ethics/eval-aware not done).

### D1. Discovery-viz frontend
Seed: visualization-as-research · MAIN-265 (retired private tracker; not migrated — no GitHub issue)

The largest unrealized seed from the arc. Build the **llobotomy** repo's first NLA-data panel: load `interpolation_flipbook.pt`
and render the per-step h-vectors as an interactive 23-discriminant
projection glyph with a t-slider — port the static fig21/fig25
pipeline to live ImGui. Use as proof-of-pattern before designing
more. Unblocks D7.

### D6. Basin landscape mapping
Emergent from F1 + the semantic-basis grid · [#5](https://github.com/skothr/llm-research/issues/5) (was MAIN-270)

Multi-session research arc. Dense-interpolate ~20-50 anchor pairs
spanning the 23 categories, cluster the AV decodings at fine
t-resolution. Identify clusters that don't map to any pure-anchor
neighborhood — candidate hybrid basins. Compare basin count vs anchor
count. Requires D3 to land first (the AV format-bias audit) so the
clustering isn't confounded by verbalizer prior.

### D7. Per-token live-trajectory viz
Seed: per-token depth · MAIN-271 (retired private tracker; not migrated — no GitHub issue)

An ImGui panel that streams a generation, captures h[20] per generated
token, runs the AV in a worker thread, and renders a live "thought
balloon" track underneath each generated token. Direct fulfillment of
the per-token-depth "watch it think" framing. Blocked by D1 (need the
gui_cpp ↔ NLA-artifact connection first).

---

## Private-tracker ID map (MAIN-N)

This arc was worked against a private issue tracker that has since been
retired. Its `MAIN-N` ticket IDs are used as shorthand labels throughout
this README, the observation files, and
`observations/figures/INVENTORY.md` (the `examples/nla_*.py` docstrings
that carried them were rewritten to cite observation files instead).
**A reader cannot look any of them up** — the tracker is gone and was
never public.

Every `MAIN-N` that appears anywhere in this arc is listed below with the
in-repo artifact or migrated GitHub issue it resolves to. Two were never
migrated and have no successor; they are kept as bare IDs rather than
deleted, because removing them would silently erase the provenance of the
findings they are attached to.

Paths are relative to this directory unless noted.

| ID | Resolves to | How the mapping was established |
|---|---|---|
| MAIN-24 | [`observations/2026-05-13-nla-vocab-atlas-grid.md`](observations/2026-05-13-nla-vocab-atlas-grid.md) | F3's cited result (PC1 = 33.5% of the sink-removed vocab atlas, content-vs-function) is that file's Finding 2; INVENTORY's "category-attractor subspace separate from the sink subspace" open question is that file's H11, nearly verbatim (H11's heading says "category subspace") |
| MAIN-25 | [`observations/2026-05-13-nla-interpolation-flipbook.md`](observations/2026-05-13-nla-interpolation-flipbook.md) | the dense-interp observation describes MAIN-25 as the 20-step grid that flagged the flip between step 8 and step 9, which is this file's fig17/fig18 run; the mid-seq-native observation cites it for "the strong t=0.421 transition", with the ID's inline link pointing at this file |
| MAIN-26 | [`observations/2026-05-13-nla-discriminant-validation.md`](observations/2026-05-13-nla-discriminant-validation.md) | the mid-seq null-result observation links it directly — its prose reads `MAIN-26` followed by an inline markdown link to this file; F2's cited 56-79% top-5 hit rates are that file's Finding 2, and "MAIN-26 / fig29" matches its fig29 |
| MAIN-30 | GitHub [#13](https://github.com/skothr/llm-research/issues/13) | issue body: "Migrated from Linear MAIN-30" |
| MAIN-34 | [`observations/2026-05-15-nla-dense-interp-near-pivot.md`](observations/2026-05-15-nla-dense-interp-near-pivot.md) | that file's own `Private-tracker ID` header |
| MAIN-38 | GitHub [#12](https://github.com/skothr/llm-research/issues/12) | issue body: "Migrated from Linear MAIN-38" (no reference to it survives in repo prose) |
| MAIN-41 | GitHub [#11](https://github.com/skothr/llm-research/issues/11) | issue body: "Migrated from Linear MAIN-41" |
| MAIN-44 | [`observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md`](observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) | that file's own `Private-tracker ID` header |
| MAIN-47 | [`observations/2026-05-14-nla-hierarchical-classifier-null-result.md`](observations/2026-05-14-nla-hierarchical-classifier-null-result.md) | that file's own `Private-tracker ID` header |
| MAIN-48 | [`observations/2026-05-14-nla-concept-arithmetic-atlas.md`](observations/2026-05-14-nla-concept-arithmetic-atlas.md) | that file's own `Private-tracker ID` header |
| MAIN-68 | GitHub [#10](https://github.com/skothr/llm-research/issues/10) | issue body: "Migrated from Linear MAIN-68" |
| MAIN-70 | [`observations/2026-05-14-nla-mid-seq-native-discriminants.md`](observations/2026-05-14-nla-mid-seq-native-discriminants.md) | that file's own `Private-tracker ID` header |
| MAIN-71 | [`observations/2026-05-15-nla-plateau-attractor-strength.md`](observations/2026-05-15-nla-plateau-attractor-strength.md) | that file's own `Private-tracker ID` header (part 2 of 2) |
| MAIN-265 | **not migrated** | D1 (discovery-viz frontend). No GitHub issue was migrated from it (issue #30's prose quotes the MAIN-265..272 range, nothing more); [D1](#d1-discovery-viz-frontend) above is the only surviving description |
| MAIN-266 | GitHub [#9](https://github.com/skothr/llm-research/issues/9) | issue body: "Migrated from Linear MAIN-266" |
| MAIN-267 | GitHub [#8](https://github.com/skothr/llm-research/issues/8) | issue body: "Migrated from Linear MAIN-267 (folds MAIN-347; …)" |
| MAIN-268 | GitHub [#7](https://github.com/skothr/llm-research/issues/7) | issue body: "Migrated from Linear MAIN-268" |
| MAIN-269 | GitHub [#6](https://github.com/skothr/llm-research/issues/6) | issue body: "Migrated from Linear MAIN-269" |
| MAIN-270 | GitHub [#5](https://github.com/skothr/llm-research/issues/5) | issue body: "Migrated from Linear MAIN-270" |
| MAIN-271 | **not migrated** | D7 (per-token live-trajectory viz). No GitHub issue was migrated from it; [D7](#d7-per-token-live-trajectory-viz) above is the only surviving description |
| MAIN-272 | GitHub [#4](https://github.com/skothr/llm-research/issues/4) | issue body: "Migrated from Linear MAIN-272" |
| MAIN-347 | GitHub [#8](https://github.com/skothr/llm-research/issues/8), folded into MAIN-267 | the only surviving mention is #8's own migration note, quoted in the MAIN-267 row above |

Migration context: `docs/planning/2026-07-25-backlog-groom.md` (repo root).

`CC-MAIN-2024-10` under `theory/` is unrelated — it is a Common Crawl
snapshot name, not a tracker ID.

---

## Reproducing

Prerequisites: Python venv at `.venv/` with torch + transformers +
matplotlib. The raw `.pt` datasets ship committed (git-LFS) under
[`data/`](data/) — run `git lfs pull` after cloning. Re-*capturing* from
scratch (not needed to verify) additionally requires Qwen2.5-7B-Instruct +
the kitft NLA pair cached locally.

```bash
# Verify the arc — re-derives every load-bearing number from the .pt files.
# Runs from a clean clone: nla_audit_findings.py reads the committed data/
# dir when the gitignored working cache is empty.
python examples/nla_audit_findings.py
# Expect: SUMMARY:  196 PASS  |  0 FAIL

# Verify dataset integrity (sha256 of every .pt vs data/MANIFEST.json)
python examples/nla_data_manifest.py --check

# Re-render a figure (~10s, no model load). Inputs resolve from the committed
# data/ dir when the working cache is empty (shared _nla_artifacts fallback).
python examples/nla_discriminant_stability_render.py

# Re-capture a .pt from scratch (loads the base model, slow on CPU)
python examples/nla_vocab_atlas_capture.py
```

The hardware reality: AV + AR run on CPU bf16; ~85s per AV
verbalization, ~7-20s per AR reconstruction. GPU nf4 on RTX 2080
yields only ~1.2× speedup because the bottleneck is autoregressive
generation, not matmul throughput. Plan multi-hour runs for any
re-capture from scratch; the committed artifacts under
[`data/`](data/) total ~15 MB and represent hundreds of CPU-hours
of capture time.

## File map

```
research/arcs/01_nla-verbalizer/
  README.md                                     # This file
  observations/
    2026-05-12-nla-position-scan-qwen25-7b.md   # Per-position h-norm scan
    2026-05-12-nla-trajectory-rabbit-haiku.md   # Per-generated-token AV trajectory
    2026-05-13-nla-vocab-atlas-grid.md          # Vocab atlas + 23 categories
    2026-05-13-nla-discriminant-validation.md   # Basis connectivity + stability + self-validation
    2026-05-13-nla-cav-country-direction.md     # Single-direction CAV for country-ness
    2026-05-13-nla-interpolation-flipbook.md    # Linear h-interpolation; step-9 transition
    2026-05-14-nla-mid-seq-vocab-atlas-null-result.md   # Cross-protocol null result
    2026-05-14-nla-mid-seq-native-discriminants.md      # Native-basis in-protocol lift
    2026-05-14-nla-concept-arithmetic-atlas.md  # Categorical-not-algebraic
    2026-05-15-nla-dense-interp-near-pivot.md   # Dense interpolation near the pivot
    2026-05-15-nla-plateau-attractor-strength.md # Round-trip cosine / basin test
    (and 11 more, incl. the .txt capture walkthrough)
    figures/
      INVENTORY.md                              # Per-figure provenance catalog
      fig1-fig11, fig13-fig37 PNGs              # 36 arc figures (fig12 never built)
  data/                                         # Raw .pt datasets (git-LFS)
    MANIFEST.json                               # sha256 + provenance per file
    README.md                                   # usage + copy-back + trust note
    *.pt                                        # 16 capture/derived artifacts (~15 MB)
```

Related implementation surfaces (outside `research/`):

- [`llm_surgeon/probe/_nla.py`](../../../llm_surgeon/probe/_nla.py) — toolkit-side NLA wrapper (CPU bf16 `nla_verbalize`, `nla_reconstruct`, `nla_score`)
- [`examples/README_NLA.md`](../../../examples/README_NLA.md) — toolkit-side scripts index + methodology notes
- [`examples/nla_audit_findings.py`](../../../examples/nla_audit_findings.py) — the regression audit (196/0)
- [`examples/nla_*.py`](../../../examples/) — 42 arc scripts
