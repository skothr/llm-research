# Stage 5.1 design addendum — verbal report + swap

Date: 2026-07-20. Status: **reviewed — 5 rulings applied 2026-07-20**; CPU-smoke
validated; awaiting explicit GPU go signal (C4 refit owns the GPU). Parent plan:
`2026-07-18-jspace-design.md` §"Stage 5", Decision 3
(staged — verbal report first). Grounds: `[gurnee2026-workspace §3.1;
kb/excerpts/gurnee2026-workspace#sec-3-1-report-swap]`,
`[kb/notes/interpretability/j-space.md §1.1, §2]`, and the Nanda
token-steering critique (`sources/forums/2026-07-06-nanda-workspace-review.md`).
Script: `examples/jspace_verbal_report.py`.

Paper reference points to replicate (Claude 4.5 family, swap-target enters
top-5 of the report): **88 %** pure J-lens vectors, **59 %** J-space component,
**5 %** non-J-space component `[gurnee2026-workspace §3.1, Fig. 8;
j-space.md §2 table]`. We expect attenuation at 20× scale reduction; null
results are in-scope deliverables (parent plan §3).

---

## 1. Categories and instance vocab (single-token-checked)

`examples/jspace_verbal_report.py :: CATEGORIES` holds ~9–18 candidate
instances per category. At load, **`single_token_instances(tok, words)`**
(same file) keeps only instances whose space-prefixed form `" Word"` encodes
to exactly one Qwen token — the J-lens vectors are token-indexed, so only
single-token instances have a well-defined swap direction `[gurnee2026-workspace
§9.1]`. Categories with < 2 survivors are dropped.

Validated 2026-07-20 against the Qwen2.5 tokenizer (shared by 1.5B and 7B):
**26 categories, 267 single-token instances** survive. Per-category survivor
counts:

| category | n | category | n | category | n |
|---|---|---|---|---|---|
| sport | 13 | tree | 12 | shape | 11 |
| fruit | 15 | bird | 13 | emotion | 8 |
| animal | 17 | drink | 9 | language | 12 |
| color | 15 | vehicle | 11 | dance | 4 |
| country | 18 | planet | 8 | weather | 9 |
| metal | 14 | gemstone | 8 | occupation | 12 |
| instrument | 4 | flower | 7 | weapon | 11 |
| vegetable | 7 | tool | 6 | furniture | 9 |
| fish | 7 | insect | 7 | | |

The committed `CATEGORIES` dict is the candidate list; the surviving set is
re-derived per run so it stays correct if the tokenizer changes. Reproduce the
count with the tokenizer-only probe embedded in the smoke (the survivor line is
printed as `[data] N categories`).

## 2. Prompt template and report format

Chat-templated (Qwen2.5-Instruct), `add_generation_prompt=True`
(`build_prompt`):

> user: "Think of a specific {category}. Reply with only that one word — the
> {category} you thought of — and nothing else."

Greedy generation, `--max-new-tokens 6`. The model emits the instance starting
at generation step 0. **Turn-start tokenization differs from mid-text**:
`"Apple"` is one bare token but `"Soccer"` splits to `"S","occer"`, so the first
generated token id is not a reliable instance handle. `match_instance(text,
insts)` matches the decoded report (case-insensitive, longest-word-first)
against the category vocab; the matched instance's canonical `" Word"` id is the
source handle for the swap. `concept_ids(tok, word)` expands to all English
single-token forms (`" Word"`, `"Word"`, lower/capitalised) for the top-5
metric and the predicts diagnostic.

## 3. Swap-vector construction and injection

**Locus.** Single layer `L`, default **21 for 1.5B, 22 for 7B** (`--layer`).
Rationale — the workspace band from this arc's own stage-4 structure scan
(`data/cache/structure_scan_*`, k=25 J-space variance fraction):

- 1.5B: varfrac rises L17→L21 (0.084→**0.124 peak at L21**), falls to L24;
  readout excess-kurtosis rises through L22–24. Band ~L19–24.
- 7B (nf4): varfrac rises L18→**L22 (0.040 peak)**, sustained to L25.

Both agree with the paper's rescaled-depth workspace band (~L19–24 of 27).
`--layer` accepts one layer; band-injection (multiple layers) is a
strength-sweep follow-up, not the default (compounding across layers
over-steers).

**Vectors.** `w_v = W_U[v]` is the unembedding row for token `v`. The **J-lens
vector** at layer `L` is `a_v = J_L^T w_v = W_U[v] @ J_L` (input direction in
layer-`L` space; identical atom to `jspace_structure_scan`)
`[j-space.md §1.1]`. All directions are unit-normalised (`normalize`).

**Injection** (`SwapHook`, a forward hook on `m.layers[L]`). For the residual
`h` at the report position(s), with unit source/target `u_s`/`u_t`, strength `s`,
and per-condition magnitude-equalization factor `scale` (§4, ruling 3):

```
coeff = <h, u_s>                                # true source projection, read live
h'    = h + scale * s * coeff * (u_t - u_s)      # remove source, add onto target
```

At `s = 1`, `scale = 1` this is the paper's literal "subtract the projection
onto Soccer, add an equal-magnitude projection onto Rugby" `[§3.1]`. `--scope
last` (default, ruling 5) edits only the final position of each forward chunk —
the report locus during prefill and the single new token each generation step;
`--scope all` (flag, later sweep) edits every position. `scale`, `coeff`, and the
injected step-0 `||delta||` (`injected_norm`, equal across conditions by
construction) plus the last-step norm are recorded per trial.

**Source token** = the instance the model actually reported (data-dependent, per
paper protocol). **Target token** = `--targets-per-category` (default 3) targets
selected by `--target-mode` (ruling 2): **`random`** (default) = seeded random
plausible-sibling instances of the category, the paper-comparable protocol
(Soccer↔Rugby-style) matching the 88/59/5 reference points; **`leastfav`** =
least-favoured baseline instances (conservative "not thought of" target). Both
exclude the source and are seeded (`seed + category-index`).

## 4. The four conditions (formula level)

Source `s`, target `t`. Only the direction pair `(u_s, u_t)` differs:

1. **`jlens`** (primary, paper 88 %):
   `u_s = norm(a_s)`, `u_t = norm(a_t)`.
2. **`nonjspace`** (paper control, 5 %):
   `u_s = norm(a_s)`, `u_t = norm(w_t - Π_J w_t)`, where `Π_J w_t` is the
   k-sparse (k=25) nonnegative **gradient-pursuit J-space component** of the
   target's unembedding row (`jspace_component`, the single-vector form of
   `jspace_structure_scan.gradient_pursuit_layer`). This steers along the part
   of the target concept the J-space does *not* capture. **DEVIATION from the
   paper** (ruling 4): the paper decomposes activation-space concept/steering
   vectors, not the unembedding row `w_t`. `w_t` is a zero-extra-data proxy for
   v1; if `nonjspace` rates come out anomalous (e.g. ≈ `jlens`), the queued
   follow-up is an activation-contrastive concept vector (see §7.4).
3. **`random`** (paper control):
   `u_s = norm(a_s)`, `u_t = norm(g)`, `g ~ N(0, I_d)` seeded `seed + tgt_tid`.
4. **`logitlens`** (Nanda control):
   `u_s = norm(w_s)`, `u_t = norm(w_t)` — raw unembedding directions, the
   logit-lens (J = I) case. If this matches `jlens`, the swap is "just token
   steering", not a workspace effect (`nanda-workspace-review.md`).

Each condition is run at every `--strengths` (default `1.0, 2.0`; §3.4 reports
higher rates at doubled strength).

**Magnitude equalization (ruling 3).** The raw step-0 injection L2 at the report
position is `s * |<h_last, u_s>| * ||u_t - u_s||`. Per item/strength we take the
**jlens** condition's value as the reference `M` and set each other condition's
`scale = M / (raw L2 of that condition)`, so all four inject the same L2 at the
report locus — direction, not magnitude, is the only difference. Per-trial
`injected_norm` (step-0) is stored and is equal across conditions within fp
tolerance (CPU smoke: 10.60/10.59/10.60/10.60); `injected_norm_laststep` records
the last generation step (differs — later positions have different `coeff`).

## 5. Success metrics

Per condition×strength, over all items and over the **predicts-subset** (items
where the baseline J-lens top-5 contains the reported instance — the paper's
"answers correctly" filter):

- **`target_top5_rate`** — swap target in top-5 of the report (step-0) logit
  distribution. Primary paper-comparable metric.
- **`report_changed_rate`** — matched report word ≠ baseline report word.
- **`report_is_target_rate`** — matched report word == target word.
- `mean_target_rank`, `mean_injected_norm` (magnitude audit).
- Global: **`jlens_predicts_report_rate`** — the workspace-readout validity gate.

Artifact `{"summary", "per_item"}` → auto-named
`data/cache/verbal_report_<model>_<lensstem>.pt` (no hardcoded path).

## 6. Item counts and runtime

Full run (defaults): 26 categories × 3 targets = **78 items**; per item 1
baseline + 4 conditions × 2 strengths = 9 short generations (≈ 6 tokens each) +
1 J-lens readout forward + 1 gradient-pursuit (nonjspace). ≈ **4 200
generated tokens** total.

**Measured (CPU smoke, 1.5B bf16, 2 items):** 1.73 tok/s effective throughput
(50 gen-tokens / 28.8 s, incl. readout + gradient-pursuit overhead), ≈ 14 s
wall/item. Naive CPU full-run projection: 4200 / 1.73 ≈ **40 min** (CPU — not
the execution path).

**GPU estimates [ASSUMPTION — not measured; GPU was reserved during design].**
Scaling from the CPU throughput by an assumed RTX 2080 speedup for short-seq
generation:

| model | assumed gen tok/s | gen time | + load/pursuit/readout | full-run estimate |
|---|---|---|---|---|
| 1.5B bf16 | ~40 (≈ 23× CPU) | ~105 s | model load ~30 s, pursuit trivial | **~5–10 min** |
| 7B nf4 | ~12 | ~350 s | load ~90 s, heavier pursuit (d=3584) | **~15–25 min** |

Both estimates are labeled assumptions; **time the first GPU run to calibrate**
(the arc's fit-cost calibration precedent). W_U must stay unquantized under nf4
— confirmed: stage-4 `structure_scan_7b_nf4` ran with the same
`lm_head.weight` float assertion.

## 7. Design decisions (reviewed 2026-07-20 — decided with rationale)

1. **Cross-lingual / synonym leakage — DECIDED: causal metrics strict-English;
   diagnostic carries a known-underestimate note.** The causal metrics stay
   as-is: the swap target's top-5 entry is checked over the target token forms
   `{word, " word", case variants}` (`concept_ids`) — we *control* the target,
   so this is exact, not an estimate. The **`jlens_predicts_report` diagnostic
   stays strict-English** and is therefore a **known underestimate**: in the CPU
   smoke the L21 J-lens top-5 for "sport" was `足球, Football, 篮球, 比赛, 网球`
   — the soccer concept present only as a Chinese token + a synonym, scoring
   `predicts=False` though the workspace clearly held the concept (exactly
   Nanda's multilingual point, `nanda-workspace-review.md`). Rationale: adding
   the diagnostic to the metric would bias it; instead we store the **full
   report transcripts and J-lens top-10 per item** (`jlens_top10`,
   `conditions[*].report_text`) so post-hoc cross-lingual/synonym aliasing needs
   no rerun.
2. **Target selection — DECIDED: default `random` plausible-sibling.**
   `--target-mode random` (seeded, default) matches the paper's Soccer↔Rugby
   protocol and its 88/59/5 reference points; `--target-mode leastfav` (flag)
   keeps the conservative "not thought of" target. The paper protocol maps to
   `random`.
3. **Magnitude — DECIDED: equalize injected L2 across all four conditions.**
   Per item/strength, scale each condition to the jlens-condition step-0
   injection L2 (§4). Direction is now the only cross-condition difference.
   Per-trial `injected_norm` recorded as the sanity check (equal within fp
   tolerance — smoke: 10.60/10.59/10.60/10.60).
4. **Non-J-space concept vector — DECIDED: `w_t` proxy for v1, marked as a
   paper DEVIATION.** The paper decomposes activation-space concept vectors; we
   decompose the unembedding row `w_t` (zero-extra-data). Queued follow-up:
   activation-contrastive concept vector, to run **if `nonjspace` rates come out
   anomalous** (e.g. ≈ `jlens`, which would indicate the proxy, not the J-space,
   drives the effect).
5. **Injection scope — DECIDED: `last`-position primary.** `--scope last`
   (default) swaps the report locus; `--scope all` stays a flag for a later
   whole-"thinking"-span sweep.

## 8. Stage 5.1b — prompt style + activation-contrastive triplet (2026-07-20)

Two follow-ups from the stage-5.1 GPU runs (7B swap effect confounded; many
7B baseline reports were word-piece fragments). Both are opt-in flags; the
committed plain 4-condition behavior is preserved.

**Prompt style (`--prompt-style {plain,chat}`, default `plain`).** Both styles
go through the tokenizer chat template with `add_generation_prompt=True`, so the
report locus is the final prefill token either way (verified: the swap still
flips the report under `chat`). Exact user-message content:

- `plain` (committed stage-5.1, verbatim): *"Think of a specific {category}.
  Reply with only that one word — the {category} you thought of — and nothing
  else."*
- `chat` (explicit one-word constraint): *"Think of a specific {category}.
  Answer with exactly one word: the {category} you thought of. Do not write
  anything else."*

Baseline single-word compliance is reported as `fragment_report_rate` = fraction
of categories whose baseline report matches no in-category single-token instance.
(Caveat: this conflates true word-piece fragmentation with valid-but-off-vocab
answers, e.g. 1.5B "Sad" for *emotion* or "Pen" for *tool*.)

**Activation-contrastive concept vector (`--contrastive`).** Per swap-target
instance, capture the layer-`L` residual at the final prompt position for 3
instance paraphrases and 3 category paraphrases (`INSTANCE_TEMPLATES` /
`CATEGORY_TEMPLATES` in the script), each chat-templated:

- instance: *"Think of {w}."*, *"You are thinking about {w}."*, *"Picture {w}
  in your mind."*
- category: same three with *"a {category}"*.

The contrastive concept vector is `v_t = mean_L(instance) − mean_L(category)`.
Two NEW conditions join the existing four (same equalized-L2 protocol, source
`u_s = norm(a_s)`):

  - `jspace_comp`    — `u_t = norm(Π_J v_t)`         → paper's **59%** row.
  - `nonjspace_comp` — `u_t = norm(v_t − Π_J v_t)`   → paper's **5%** row.

with `Π_J v_t` the k=25 gradient-pursuit J-space component of `v_t`
(activation-space, cf. ruling 4's `w_t`-based proxy). The base `jlens` condition
is the paper's **88%** row. Artifacts are auto-named
`verbal_report_{style}_{4c|6c}_{model}_{lensstem}.pt` — the plain-4c committed
artifacts are never clobbered.

**Note (magnitude equalization under the 6-condition set):** the five
J-lens-source conditions (jlens/nonjspace/random/jspace_comp/nonjspace_comp)
share `u_s = a_s`, so their step-0 injected L2 is *exactly* equal; only
`logitlens` (source `w_s`) drifts — ≈1% on 1.5B bf16, <0.1% on 7B nf4 — because
its scale is set from the fp32 capture forward while the recorded norm is read
from the bf16 generate-prefill forward (a measurement artifact, not a
direction error). A future CHECK-F over 6c artifacts should ratio the injected
norm per item or exempt logitlens from a tight cross-condition-equality bound.
