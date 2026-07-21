# Observation: NLA cross-tie — weak prompt-specific channel agreement; verbalizable content lives outside the J-space component (stage 6)

**Date/context:** 2026-07-21. Stage 6, the arc's novel experiment (design:
`plans/2026-07-20-stage6-design.md`): at the NLA verbalizer's capture
layer, compare the J-lens readout of an activation against the NLA
verbalizer's output for the *same* activation, and test whether the NLA
signal concentrates in the activation's J-space component. Setup:
Qwen2.5-7B nf4 + lens `jlens_qwen2.5-7b_nf4_n100`; **layer correspondence
verified empirically: NLA `hidden_states[20]` = jlens `source_layer 19`**
(off-by-one between conventions — load-bearing, tested in-script); AV
(`kitft/nla-qwen2.5-7b-L20-av`) is a standalone ~15 GB model run on CPU,
two-phase after GPU capture. 12 neutral + 12 concept prompts + 12
decomposition prompts; k=25 gradient-pursuit decomposition (shared helper
`examples/_jspace_pursuit.py`, factored from the stage-4 scan,
bit-identical outputs). Script: `examples/jspace_nla_crosstie.py`
(incremental AV checkpointing added after one run was killed mid-phase;
rerun deterministic).

## Finding

**1. Channel agreement exists, is weak, is concept-conditional, and is
prompt-specific (null-certified).** On concept-loaded prompts, the AV's
content words sit at median rank 9,585 of 152,064 in the J-lens readout
(chance floor 76,032); on neutral prose there is no agreement at all
(median 103,412 — *worse* than chance; top-50 overlap 0.006). The
concept-only mismatched null separates: matched pairings 9,585 vs
permuted 16,353 (matched percentile 0.159) — the agreement tracks the
specific prompt, not just shared concept vocabulary. But note the
mismatched baseline (16k) is itself far above chance: most of the raw
"agreement" is generic concept-word bias; the prompt-specific residue is
a ~1.7× rank improvement.

**2. The J-lens readout is only intermittently legible at this layer.**
At L19 (= the NLA's L20), the 7B J-lens top-10 is underscore/whitespace
filler on most prompts (e.g. 'Water is made of hydrogen and' →
`['____',' ______',…]` while the AV correctly reads the chemistry), but
legible on some ('A doctor works in a' → `' medical'`, `' hospital'`,
`' doctors'` in the top-10) — patchy, not uniformly absent (the n=4
quick pass had suggested uniform filler; the diverse n=12 set corrects
that). Known-target hit rates: NLA 0.917 vs J-lens 0.333.
Consistent with the arc's stage-3 finding that 7B J-lens readouts become
contentful only ~L22: **the NLA arc's capture layer sits below the 7B
J-lens legibility onset** — a cross-arc architectural fact nobody could
see before stage 4 mapped the depth bands.

**3. The NLA-verbalizable content lives outside the J-space component.**
Content-word Jaccard against the full activation's verbalization
(n=12 decomp prompts): residual 0.600, J-space component 0.162
(random-unit floor 0.075), and — decisive — removing the J-space
component damages the verbalization *no more than removing an equal-norm
random direction* (residual 0.600 vs norm-matched control 0.606, delta
+0.005). The isolated component's own content rank is at chance (75,167
vs floor 76,032). The J-space component (varfrac ~2–3% at this layer)
is not where the AV's signal sits.

**Interpretation.** The arc's three causal/content probes now agree:
stage 5.1b (J-space membership of concept vectors causally inert),
stage 6 experiment B (J-space component carries ~none of the verbalizable
content at the NLA layer), and the depth mismatch (the workspace band the
J-space framing needs starts above the layer where this model's
verbalizable content demonstrably lives). At open-model scale, J-lens
*vectors* are a usable token-indexed decoding dictionary, but J-space as
a privileged *subspace* is not supported on Qwen2.5 at n=100 lens
quality. Named alternatives as before: lens fit budget (H1), quantization,
and the L19-specific caveat that the decomposition there may be isolating
format/whitespace structure rather than content [SPECULATION].

## Evidence

Artifact (gitignored cache):
`nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` (8.3 MB;
25 per_prompt entries incl. persisted fp16 readouts so the nulls
re-derive offline; `.partial.jsonl` / `.phase1.pt` checkpoint sidecars
alongside). Quick-pass n=4 artifact preserved in session scratchpad;
full-run signs match quick-pass throughout (no reversals). AV format
bias (NLA arc caveat L2) is visibly live — every verbalization opens
with the same "quiz/question format" framing — and is handled by the
content-word filter + the mismatched null, but remains the main
inflation risk on metric 2 (overlap 0.49, treated as coarse only).
Audit Check G pins the summary values and booleans.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_nla_crosstie.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
    --n-neutral 12 --n-concept 12 --n-decomp 12
```
(Phase 1 on GPU, AV phase on CPU; resumes from checkpoint sidecars after
interruption.)

## Hypotheses

- The weak-but-real concept agreement is consistent with both channels
  reading the same underlying representation through very different
  decoders (linear token-indexed vs learned free-text); the rank signal
  without top-k legibility suggests the J-lens direction is present but
  swamped by the L19 filler structure. [INFERENCE]
- If the J-space claim holds only above the legibility onset (~L22+ on
  7B), an AV trained at L22-24 would be the discriminating instrument —
  outside this arc's scope (requires training a new verbalizer).

## Follow-ups

- Arc-close synthesis: fold stages 5.1b + 6 into the single causal
  conclusion (token-indexed directions effective; J-space membership not
  privileged at open scale).
- Optional before close: repeat experiment B at L22 (J-lens legible band)
  *without* the AV — using the J-lens's own readout as the content probe —
  to test whether the component/residual split changes above the onset.
  Cheap (no AV involved).
- NLA arc: file the L20-below-legibility-onset finding back to that arc's
  records at arc close (cross-arc note).

## References

- `[gurnee2026-workspace §2.3, §4.1]`;
  `kb/notes/interpretability/j-space.md` §5 (single-token limit)
- NLA arc: `research/arcs/nla-verbalizer/` (capture-layer convention,
  format-bias caveat L2)
- Prior: `2026-07-20-jspace-structure-stage4.md` (depth bands),
  `2026-07-20-verbal-report-swaps-stage5b.md` (causal null),
  `plans/2026-07-20-stage6-design.md`
