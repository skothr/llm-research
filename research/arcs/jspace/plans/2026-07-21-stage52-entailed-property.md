# Stage 5.2 design addendum — entailed-property swap (paper §3.3 spider→ant)

Date: 2026-07-21. Status: **design + script + CPU-smoke complete; awaiting GPU
go signal** (part-A logitlens-norm rerun + the n=500 fit release the device
first). Parent plan: `2026-07-18-jspace-design.md` §"Stage 5", Decision 3
(staged). Prior stage: `2026-07-20-stage5-design.md` (report-swap, §3.1).
Grounds: `[gurnee2026-workspace §3.3; kb/excerpts/gurnee2026-workspace#sec-3-3-spider-ant]`,
the Nanda token-steering critique (`sources/forums/2026-07-06-nanda-workspace-review.md`).
Script: `examples/jspace_entailed_swap.py` (reuses `jspace_verbal_report.py`
machinery — `normalize`, `jlens_atom`, `concept_ids`, `topk_ids`, `rank_of`,
`PrefillCapture`, `build_model`).

## 1. What this replicates and why it is the strongest causal form

Paper reference (verbatim, `#sec-3-3-spider-ant`): the prompt *"The number of
legs on the animal that spins webs is"* forces the model to (1) infer the
**unspoken** concept *spider* and (2) read out a **property** of that concept
(*8* legs). "The Jacobian lens at intermediate layers confirms that spider is
represented at the relevant token positions, even though the word never appears
in the prompt or the output. When we swap the spider lens vector for ant, the
model's top output changes from '8' to '6', the number of legs on an ant."

This is a stronger causal claim than the §3.1 report-swap (stage 5.1): the
**measured readout is an entailed PROPERTY of the swapped concept, not the
concept token itself**. Plain token steering can put *ant* into the output
distribution; it cannot, on its own, supply *ant*'s leg count. If swapping the
concept direction flips the property, the concept representation is causally
feeding the downstream property computation — relational structure, not surface
token identity. The key comparison this stage isolates:

- If **jlens** swap flips the property but **logitlens** (raw-unembedding token
  steering of the same concept) does NOT → J-space carries relational structure
  the logit-lens direction does not. Supports the paper's workspace account.
- If **logitlens** ALSO flips the property → the token-steering account
  (`nanda-workspace-review.md`) extends further than expected: steering the
  concept token alone re-derives the property. This would be the strongest
  open-model evidence *for* the token-steering critique, sharpening stage 5.1b's
  finding (which only showed token-steering matches jlens for the *concept*
  readout, not an entailed property).

Null and either-direction outcomes are in-scope deliverables (parent plan §3).
We expect attenuation at 20×–60× scale reduction from Claude 4.5.

## 2. Item schema and bank

Each item is a dict (`ITEMS` in the script):

```
{ "family", "descriptor", "tail", "answer_kind",
  "concept", "answer", "swap_concept", "swap_answer" }
```

- `descriptor` — a clause that **implies** the concept via a *defining property
  other than the queried one* (never names the concept; never names the answer).
- `tail` — the property-eliciting continuation (`" is"` for plain completion).
- `concept` / `swap_concept` — the unspoken intermediate and its swap. Both must
  be **space-prefixed single-token** in the Qwen2.5 tokenizer (the J-lens vector
  `a_v = W_U[" concept"] @ J_L` is token-indexed, `[gurnee2026-workspace §9.1]`).
- `answer` / `swap_answer` — the concept's / swap-concept's single-token property
  value. Validated single-token in **some** case/leading-space form
  (`concept_ids`); **`answer != swap_answer` within every pair** (enforced at
  load, else the item is dropped and reported).
- `answer_kind ∈ {number, word}` — drives the chat-style answer instruction.

**Families (33 items after tokenizer validation 2026-07-21; penguin, wasp,
Brasilia dropped for lacking the required single-token form):**

| family | property | n | example (concept→answer ⇒ swap→answer) |
|---|---|--:|---|
| legs/number | count | 9 | spins-webs → spider→8 ⇒ ant→6 |
| capital | city  | 12 | Eiffel-Tower country → France→Paris ⇒ Italy→Rome |
| language | language | 8 | Eiffel-Tower country → France→French ⇒ Spain→Spanish |
| color | color | 4 | keeps-doctor-away fruit → apple→red ⇒ banana→yellow |

Two-hop families (capital, language) are landmark→country(unspoken)→property:
the swapped concept is the **country**, the readout its **capital/language** —
a clean entailed property, exactly the §3.3 structure. All concepts are drawn
from single-token members of the arc's validated country/animal/fruit vocab.
The full committed bank is the source of truth; the tokenizer survivor set is
re-derived per run (`validate_items`) so it stays correct if the tokenizer
changes.

## 3. Baseline precondition (stage-5.2 gate)

For each item, greedy-generate the clean prompt (no intervention). The item is
**baseline-correct** iff the step-0 top-1 token matches any single-token form of
`answer` (`concept_ids(answer)`). Report **clean accuracy per model**
(`clean_accuracy` = baseline-correct / total). **Flip metrics are computed only
over the baseline-correct subset** — an item the model cannot answer cleanly
cannot demonstrate a *property* flip. Per-item `baseline_correct` and the full
baseline top-k are stored so the exclusion is auditable and reversible.

## 4. Swap locus — WHERE, and why (the load-bearing design choice)

The paper swaps "the spider lens vector … at the relevant token positions" where
the concept is held — **intermediate positions, not the report/answer position**.
Swapping at the report position would steer the *property token* directly and
collapse the experiment back to §3.1-style token steering. The whole point is to
edit the concept **upstream** of the property computation and let the model
re-derive the property.

`--swap-scope` (default **`auto`**):

- **`auto`** (default, paper-faithful): the concept-holding positions are found
  **by the J-lens itself** — exactly the paper's diagnostic ("the J-lens …
  confirms that spider is represented at the relevant token positions"). For each
  prompt position `p < report_pos`, read the J-lens top-`k_detect` (default 10)
  at the swap layer; keep positions whose top-k contains any single-token form of
  the **source concept**. This is condition-independent (a property of the clean
  prompt) so all three swap conditions edit the *same* position set. Never
  includes the report position. **Fallback** if no position qualifies: the
  `window` set (recorded as `scope_used="auto->window_fallback"`).
- **`window`**: the last `--swap-window` (default 4) positions before the report
  position (concept resolves near the end of the descriptor).
- **`all`**: all positions before the report position.
- **`report`**: the report position only — a **control**. If jlens@report flips
  the property, the effect is not upstream-concept-mediated. Expected weak.

Detected positions, the scope actually used, and the per-position source-concept
J-lens rank are stored per item.

**Report/answer position.** For plain style the report position is the final
prompt token (predicting the answer). For chat style it is the final prefill
token after `add_generation_prompt=True`. Detection scans `[0, report_pos)`.

## 5. Swap-vector construction, injection, conditions

Reuses the stage-5 atoms. `w_v = W_U[v]`; J-lens vector `a_v = W_U[v] @ J_L`
(fp32, unit-normalized). Three conditions, **magnitude-equalized** (§6):

1. **`jlens`** (primary, paper §3.3): `u_s = norm(a_s)`, `u_t = norm(a_t)`.
2. **`logitlens`** (Nanda token-steering control — THE key comparison):
   `u_s = norm(w_s)`, `u_t = norm(w_t)`.
3. **`random`** (floor): `u_s = norm(a_s)`, `u_t = norm(g)`, `g ~ N(0,I)` seeded
   `seed + tgt_tid`.

**Injection** (`EntailedSwapHook`, forward hook on `m.layers[L]`). Fires **only on
the prefill forward** (`h.shape[1] == prompt_len`); generation-step forwards
(`T=1`) are untouched — the concept is held in the prompt, already in the KV
cache by the time tokens are emitted. At the concept positions `P`, for residual
`h` with live projection `coeff_p = <h_p, u_s>`, strength `s`, equalization
`scale`:

```
h[:,p,:] += scale * s * coeff_p * (u_t - u_s)   for each p in P
```

Editing layer-`L` output at the (earlier) concept positions propagates to the
final-position answer logits through attention at layers > L — the mechanism by
which an upstream concept edit changes the downstream property readout.

`--strengths 1.0,2.0` (paper equal-magnitude and doubled).

**Swap layer `L`** (`--layer`): default **21 (1.5B), 22 (7B)** — the arc's
stage-4 workspace-band varfrac peaks (`2026-07-20-jspace-structure-stage4.md`).
The entailed-property flip may live at a different depth than the §3.1 report
swap, so the run **sweeps `L ∈ {peak, peak−3, peak+3}`** if runtime permits.

## 6. Magnitude equalization (live-prefill, learns from the stage-5.1b bug)

All coeffs and the equalization use the **live prefill residuals** captured on
the same generate forward as the injection (`PrefillCapture`, full `[T,d]`), so
there is **no fp32-capture-vs-bf16-prefill mismatch** — the exact defect stage
5.1b's logitlens condition hit (design §8; fixed in `jspace_verbal_report.py`
part A). Total injected L2 over the position set is
`s * ||u_t - u_s|| * sqrt(Σ_p coeff_p²)`. Per item/strength, `jlens` is the
reference `M`; each other condition is scaled so its total injected L2 equals
`M`. Stored per trial: `scale`, per-position `coeff`, total `injected_norm`
(equal across conditions within fp tolerance by construction), and
`injected_norm_perpos`.

## 7. Metrics

Per condition × strength, over the **baseline-correct subset**:

- **`property_flip_rate`** — step-0 top-1 == any single-token form of
  `swap_answer` (the swap concept's property). **Primary result.**
- **`answer_changed_rate`** — top-1 != baseline answer.
- **`swap_answer_top5_rate`** — swap_answer in step-0 top-5 (softer flip signal).
- **`clean_retention_rate`** — top-1 still == baseline answer (no displacement).
- **Paper-analog log-prob shift** — `Δ logP(swap_answer) = logP_swap −
  logP_baseline` and `Δ logP(clean_answer)` at the report position, per
  condition. Positive `Δ logP(swap_answer)` with negative `Δ logP(clean_answer)`
  is the graded flip signature even when top-1 does not cross.

**Headline comparison** (§1): `property_flip_rate[jlens] − property_flip_rate[logitlens]`
and `− property_flip_rate[random]`, at each strength.

## 8. Rich capture (artifact = `{"summary","per_item"}`)

`per_item[i]` stores, for full post-hoc checkability without rerun:

- item fields; `baseline_correct`; baseline top-k (ids + strings + probs).
- detected concept positions, `scope_used`, per-position source-concept J-lens
  rank at the swap layer.
- **per-LAYER J-lens top-k at each concept position for the CLEAN run** — so the
  "spider visible at intermediate layers" claim is directly checkable from the
  artifact (`clean_perlayer_jlens[pos][layer] = topk ids+strings`).
- per condition × strength: step-0 top-k (ids + strings + probs) at the report
  position, decoded transcript, `property_flip`, `answer_changed`,
  `logp_swap_answer`, `logp_clean_answer`, `scale`, `injected_norm`,
  per-position `coeff`.

Auto-named `data/cache/entailed_swap_{style}_L{layer}_{model}_{lensstem}.pt`
(never hardcoded; layer + style in the name so the layer sweep never clobbers).

## 9. Item counts and runtime

33 items × (1 baseline + 3 conditions × 2 strengths) = **7 generations/item**
(short, ≤6 tokens) + 1 clean per-layer J-lens readout pass (27 layers) + the
`auto` detection forward. ≈ 231 swap generations + 33 readout passes total.

GPU estimate [ASSUMPTION — calibrate on first run, per the arc's fit-cost
precedent]: comparable to stage-5.1 (~78 items there); **1.5B bf16 ≈ 5–8 min,
7B nf4 ≈ 15–25 min** per layer. A 3-layer sweep is ~3× that; run the peak first,
add ±3 only if the peak run confirms plumbing and time permits (<20 min extra
budget at 7B → likely peak + one neighbor at 7B, full sweep at 1.5B).

## 10. Commands

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1.5B (peak L21), plain completion (paper-faithful) + chat cross-check
python examples/jspace_entailed_swap.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --device cuda --layer 21 --prompt-style plain \
    --lens research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt
# 7B (peak L22)
python examples/jspace_entailed_swap.py --model Qwen/Qwen2.5-7B-Instruct \
    --mode nf4 --device cuda --layer 22 --prompt-style plain \
    --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt
# layer sweep: rerun with --layer {18,24} (1.5B) / {19,25} (7B)

# CPU smoke (plumbing, 2 items):
python examples/jspace_entailed_swap.py --device cpu --limit 2 \
    --strengths 1.0 --max-new-tokens 4 --prompt-style plain
```

## 11. Open deviations / caveats

- **`auto` scope depends on J-lens legibility of the source concept at `L`.**
  The arc's 7B legibility onset is ~L22 (`2026-07-21-nla-crosstie-stage6.md`);
  at L22 the concept may be marginal, triggering the `window` fallback. Both the
  detected-vs-fallback status and the layer sweep guard against reading a null
  that is really a wrong-locus artifact.
- **English-only property forms** (same known underestimate as stage 5.1,
  ruling 1). Full top-k strings stored → post-hoc cross-lingual aliasing needs
  no rerun.
- **Chat vs plain.** Plain completion is paper-faithful; chat is the arc's
  compliance-hardened style. Both are chat-agnostic for the concept locus (the
  descriptor sits mid-prompt either way). Baseline accuracy per style decides
  which carries each model's headline (reported, not pre-judged).
