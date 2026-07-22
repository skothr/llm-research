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

## Finding

**The paper's discrete top-1 property flip does not replicate at this
scale (flip rate 0.000 everywhere), but the graded version of the effect
is present, large, and J-lens-specific** — the signature that raw token
steering cannot produce, since steering the concept token carries no
knowledge of the concept's properties:

| 1.5B chat, s=2, L18 | Δlogp(swap property) | swap-prop in top-5 | clean retained |
|---|--:|--:|--:|
| **jlens** | **+2.13 nats** | 0.235 | 0.941 |
| logitlens | +0.07 | 0.118 | 1.000 |
| random | +0.14 | 0.118 | 1.000 |

Swapping the concept along its J-lens vector moves the *unspoken entailed
property's* log-probability by +2.13 nats — ~30× the token-steering
control at identical injected magnitude — and lifts the property into the
top-5 on a quarter of retained items, without breaking the model
(clean-retention 0.94). The effect is depth-localized at **L18** (at L21,
the report-swap peak layer, it collapses to +0.26): concept→property
propagation engages earlier than report emission, consistent with
computation flowing through subsequent layers.

**7B replication (added same day, gated runs + layer sweep):** clean
accuracy 0.909 (30/33). The effect reproduces with the same structure:

| 7B chat, s=2 | jlens Δlogp | logitlens | random | retain |
|---|--:|--:|--:|--:|
| **L19** | **+1.250** | +0.178 | +0.040 | 0.967 |
| L18 | +0.897 | +0.168 | +0.066 | 0.933 |
| L22 (report layer) | +0.471 | +0.244 | +0.129 | 1.000 |

Peak at **L19** — the same relative depth as 1.5B's L18 — at 7× the
token-steering control (31× random), collapsing toward parity at the
L22 report layer. Discrete flips remain 0.000 at both scales even at
90% clean accuracy.

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

Artifacts (gitignored cache): `entailed_swap_chat_L{18,21,24}_*1.5b*`,
`entailed_swap_plain_L21_*1.5b*`, `entailed_swap_plain_L22_*7b*` — full
transcripts, per-condition top-k ids/strings/probs at the answer
position, per-layer clean J-lens top-k at concept positions (the
"concept visible at intermediate layers" premise is checkable per item).
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
