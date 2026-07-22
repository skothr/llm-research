# Observation: entailed-property swaps — no discrete flip, but a J-lens-specific graded effect token steering cannot produce (stage 5.2)

**Date/context:** 2026-07-22. Replication of the paper's demonstrated
spider→ant entailed-property flip
(`[kb/excerpts/gurnee2026-workspace#sec-3-3-spider-ant]`: swap the
unspoken concept's J-lens vector, the *entailed property* in the output
flips, 8→6 legs). Design:
`plans/2026-07-21-stage52-entailed-property.md` — 33 items, 4 property
families (leg counts, capitals-via-landmark, language-via-country,
colors), single-token-validated, baseline-gated; swap at J-lens-detected
concept positions (never the report position); three equalized-L2
conditions (jlens / logitlens token-steering control / random); layer
sweep {18, 21, 24} at 1.5B. Script: `examples/jspace_entailed_swap.py`,
full rich capture. Clean accuracy: 1.5B chat 17/33, plain 4/33; 7B plain
11/33. Injected-norm equality verified exact across conditions (the
logitlens norm-fix landed first; see method note below).

**Scope split — auto vs positional-window fallback (recorded 2026-07-22,
review response).** `detect_positions()` keeps positions whose swap-layer
J-lens top-k holds a source-concept form (`scope_used == "auto"`); when a
layer yields *no* such position it falls back silently to a fixed
last-N-token window (`"auto->window_fallback"`). Both were folded into
the original headline aggregate. The split matters: on the
baseline-correct subset the auto fraction is 7/17 (1.5B L18), 10/17
(L21), 15/17 (L24); 17/30 (7B L18, L19), 18/30 (L22). Measured on the
committed artifacts, the fallback items carry **~zero** entailed-property
movement (jlens Δlogp −0.00 to +0.23 across layers), so mixing them in
*dilutes the headline downward*. The tables below therefore now lead
with the **auto-only** aggregate (the genuine J-lens-detected-position
subset) and keep the mixed number in parentheses; the qualitative
result is unchanged and the effect is in fact ~2× cleaner. Per-scope
aggregates and the `n_auto`/`n_fallback` counts are persisted in
`summary.metrics_by_scope` for runs from this date onward; the audit
(Check L) pins the auto-only headline and the split counts.

## Finding

**The paper's discrete top-1 property flip does not replicate at this
scale (flip rate 0.000 everywhere), but the graded version of the effect
is present, large, and J-lens-specific** — the signature that raw token
steering cannot produce, since steering the concept token carries no
knowledge of the concept's properties:

| 1.5B chat, s=2, L18 | Δlogp(swap property), auto-only (mixed) | swap-prop in top-5 |
|---|--:|--:|
| **jlens** | **+5.17 nats** (mixed +2.13) | 0.235 |
| logitlens | +0.15 (mixed +0.07) | 0.118 |
| random | +0.30 (mixed +0.14) | 0.118 |

(Auto-only n=7 of the 17 baseline-correct items; the other 10 fell back
to the positional window and carry ~0 movement — see the scope-split note
above. Clean-retention 0.94, unaffected by the split.)

Swapping the concept along its J-lens vector, at the genuinely
J-lens-detected concept positions, moves the *unspoken entailed
property's* log-probability by **+5.17 nats** (mixed-scope +2.13) — an
absolute gap of **+5.0 nats** (per-item SD 4.9, n=7) over the
equal-magnitude logit-lens token-steering control (+0.15) — and lifts
the property into the top-5 on a quarter of retained items, without
breaking the model (clean-retention 0.94). The multiplier (~34×) is
denominator-fragile since the control sits near zero, so the absolute
gap is the figure to cite; the item-level SD exceeds the mean (a few
high-movement items dominate), so this is a directional effect, not a
tight estimate. The effect is depth-localized
at **L18** (auto-only jlens Δlogp L18 +5.17 > L21 +0.44 > L24 +0.10;
mixed +2.13 > +0.26 > +0.10): concept→property propagation engages
earlier than report emission, consistent with computation flowing
through subsequent layers. The depth ordering holds identically on the
auto-only and mixed subsets.

**7B replication (added same day, gated runs + layer sweep):** clean
accuracy 0.909 (30/33). The effect reproduces with the same structure:

| 7B chat, s=2 | jlens Δlogp auto-only (mixed) | logitlens auto | random auto | retain |
|---|--:|--:|--:|--:|
| **L19** | **+2.17** (mixed +1.250) | +0.27 | +0.07 | 0.967 |
| L18 | +1.52 (mixed +0.897) | +0.27 | −0.07 | 0.933 |
| L22 (report layer) | +0.63 (mixed +0.471) | +0.29 | +0.17 | 1.000 |

(Auto-only n=17 of 30 baseline-correct at L18/L19, 18 at L22; the
fallback items again carry ~0 movement.) Peak at **L19** — the same
relative depth as 1.5B's L18 — at ~8× the token-steering control (~30×
random) on the auto-only subset, collapsing toward parity at the L22
report layer. Discrete flips remain 0.000 at both scales even at 90%
clean accuracy.

**Interpretation — this materially revises the arc's causal story.** The
5.1b conclusion ("only token-indexed directions steer; weight toward the
token-steering account") holds for *report* swaps but does not extend to
*relational* effects: at both scales, the J-lens vector demonstrably
carries concept-linked structure that the raw unembedding direction does
not, and it engages a few layers below the report-swap depth
(L18/L19 vs L21/L22 — a consistent cross-scale depth split). The effect
is graded rather than discrete — plausibly a capability/scale
attenuation of the paper's Claude-scale flip (their 8→6 is the
top-1-crossing version of the same movement); moving from 1.5B to 7B
did not close the gap to a flip, so the threshold, if it exists, lies
above 7B. 7B plain landed as a corroborating footnote (low clean
accuracy 11/33).

## Evidence

Artifacts (committed under `data/`, gitignored `cache/` mirror):
`entailed_swap_chat_L{18,21,24}_*1.5b*`,
`entailed_swap_plain_L21_*1.5b*`, `entailed_swap_plain_L22_*7b*` — full
transcripts, per-condition top-k ids/strings/probs at the answer
position, per-layer clean J-lens top-k at concept positions, and per-item
`scope_used` (`auto` vs `auto->window_fallback`). For the `auto` items
the "concept visible at intermediate layers" premise is checkable per
item; the fallback items are precisely those where *no* concept position
was J-lens-detectable at that layer (hence the positional-window
substitution and their ~zero movement), which is why the auto-only
aggregate is the faithful J-space-localized figure.
Method note: the logitlens magnitude fix (scale from the same
generate-prefill forward as the injection) was applied before these
runs; at 1.5B it is a verified no-op (bf16 capture ≈ prefill; the
stage-5b observation's 17.7% figure traces to a single degenerate
fragment-source item), so cross-condition magnitudes here are exact.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_entailed_swap.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --device cuda --layer 18 --prompt-style chat \
    --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt
```
(Layer sweep: --layer {18,21,24}; 7B: --mode nf4 --layer 22 + the 7B lens.)

## Hypotheses

- **Graded-vs-discrete is scale attenuation** [INFERENCE]: +2.13 nats is
  a large movement that fails to cross top-1 in a 1.5B model whose
  baseline property confidence is high; the paper's flip is the
  above-threshold version. The gated 7B chat run tests the first step of
  this dose-response.
- The L18-vs-L21 depth split (property effect vs report effect) suggests
  the workspace band is functionally layered: relational consequences
  are computed from earlier injections than verbalization is.
  [SPECULATION — one model, one sweep]

## Probe addendum: verbatim-prompt / cue-redundancy control (2026-07-22)

Reviewer challenge: the item bank's descriptors enrich the paper's
minimal prompt (e.g. "creature that spins **silky** webs **to trap
flies**" vs the paper's "animal that spins webs") — an **undocumented
deviation** (probable motive: baseline-accuracy insurance on small
models; not A/B-tested at design time). Redundant cues could let the
model *re-derive* the concept downstream of a localized swap,
suppressing discrete flips. Tested directly (`--items-json` probe, 7B
L19, paper-verbatim + 2 minimal-cue items, scopes auto and **all**
(every pre-answer position — closes the re-derivation channel)):

| 7B L19 s=2 | jlens Δlogp | logitlens | random | flips |
|---|--:|--:|--:|--:|
| chat, all-scope | +1.60 (spider +2.00) | +0.44 | +0.71 | 0 |
| chat, auto | +0.06 | +0.02 | −0.38 | 0 |

**The flip null survives its best alternative explanation**: the
paper's exact prompt with maximal-coverage editing still reports "8"
while "6" gains ~2 nats. The enriched-cue deviation did not drive the
null. Caveats: n=3 probe; all-scope random control is elevated (+0.71 —
broadcasting across 56 positions perturbs broadly), narrowing the
jlens-vs-random contrast in that regime; and the minimal-cue prompt's
single auto-detected position carried almost no effect (+0.06),
suggesting sparse-cue prompts hold the concept more diffusely.
Artifacts: `entailed_paperverbatim_{chat,plain}_{auto,all}_L19_7b.pt`.

## Follow-ups

- Strength sweep beyond s=2 at the L18/L19 peak (does Δlogp cross
  top-1?) — cheap, the remaining dose-response axis.
- Audit check group for the entailed artifacts (stage-7 consolidation).

## References

- `[gurnee2026-workspace §3.3; kb/excerpts/gurnee2026-workspace#sec-3-3-spider-ant]`
- Prior: `2026-07-20-verbal-report-swaps-stage5b.md` (the conclusion this
  revises), `plans/2026-07-21-stage52-entailed-property.md`
