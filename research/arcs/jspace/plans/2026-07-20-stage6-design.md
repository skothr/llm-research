# Stage 6 design addendum: the NLA cross-tie

Date: 2026-07-20. Status: **design, pre-GPU**. Extends
`2026-07-18-jspace-design.md` §"Stage 6". Grounding: that spec, the NLA arc
(`research/arcs/nla-verbalizer/README.md`), `llm_surgeon.probe._nla`, and
`theory/kb/notes/interpretability/j-space.md` §5 (single-token limit).

Stage 6 ties the two arcs together: at the activation the NLA verbalizer was
trained to read, does the J-lens (a *derived* activation→language readout) agree
with the NLA AV (a *trained* one) about what the activation is "about", and is
the NLA's content concentrated in the activation's small J-space component?
This is the arc's most likely original contribution — no prior work compares a
Jacobian-lens readout against a trained natural-language autoencoder on the same
hidden state.

## 0. The load-bearing layer-index fact (verified, do not skip)

The NLA AV/AR pair `kitft/nla-qwen2.5-7b-L20-{av,ar}` was trained on the
Qwen2.5-7B-Instruct residual stream at **HF `hidden_states[20]`** — the NLA arc
captures `out.hidden_states[LAYER]` with `LAYER=20` (`examples/nla_scan.py`,
`nla_prompt_battery.py`).

jlens indexes source layers by **block output via a forward hook**
(`jlens/hooks.py::ActivationRecorder` stores `blocks[index]` output). HF's
`output_hidden_states` convention is `hidden_states[0]` = embeddings and
`hidden_states[k]` = output of block `k-1`. Therefore:

> **jlens source_layer L  ==  block-L output  ==  HF `hidden_states[L+1]`.**

Verified empirically (2026-07-20, cached Qwen2.5-1.5B CPU): `recorder[block 19]`
is bit-close to `hidden_states[20]`, and `recorder[block 20]` to
`hidden_states[21]`.

Consequence: **the NLA's "layer 20" is jlens source_layer 19**, NOT 20. The
cross-tie applies the jacobian `J_19` and captures at block 19. (The jspace
stages 3-4 "layer 20" plots use jlens source_layer 20 = `hidden_states[21]`, a
different activation — do not conflate them.) The full cache lens
`jlens_qwen2.5-7b_nf4_n100.pt` has source_layers 0-26, so layer 19 is present;
the committed LFS subset ({0,5,10,15,20,25,26}) does not, so stage 6 depends on
the full cache lens (same lens stages 4-5 default to).

## 1. Primary configuration and how the two models coexist

- **Base:** `Qwen/Qwen2.5-7B-Instruct`, `mode="nf4"`, `device_map={"":0}` (the
  NLA arc's and jspace's standard 7B load, `llm_surgeon.surgery.load_model`).
  Used to capture `h_19` (= NLA `h_20`), compute the J-lens readout
  `W_U norm(J_19 h)`, and run gradient-pursuit decomposition. `W_U` (`lm_head`)
  must be an unquantized 2-D float tensor for the raw matvec chain — asserted at
  startup (same guard as the structure scan).
- **AV verbalizer:** `kitft/nla-qwen2.5-7b-L20-av`, CPU bf16, via
  `llm_surgeon.probe.{load_av, nla_verbalize}`. `nla_verbalize` L2-normalizes the
  activation and rescales to `injection_scale=150.0` before injecting into one
  template token — it is **magnitude-invariant, sees direction only**
  (`d_model=3584`). Output is free English text (the `<explanation>...</explanation>`
  span), ~1.5 tok/s on CPU.

**Two-phase to avoid holding both models at once** (the NLA arc's established
pattern, `nla_scan.py`): Phase 1 loads base+lens on GPU, captures every needed
vector (raw `h`, J-space component `c`, residual `h−c`, norm-matched control
`h−r`) and the full J-lens readout logits per (prompt, position), writes them to
a CPU-resident intermediate, then frees the base. Phase 2 loads the AV and
verbalizes the saved vectors. The base's `W_U`-indexed readout ranks are
computed in Phase 1; NLA words are mapped to base token ids in Phase 2 by
reloading only the base tokenizer (cheap, no model) — the AV shares the
Qwen2.5 vocab but the readout is over the base `W_U`, so the base tokenizer is
authoritative. Base nf4 lives on GPU (~5 GB); AV bf16 on CPU (~15 GB RAM) — they
do not compete for the same memory, but two-phase keeps peak RAM down and lets
the GPU be released before the ~2 h AV pass.

## 2. Prompt sets

Both sets are **raw text, no chat template** — the 7B J-lens was fit on raw
wikitext, so raw prompts keep the readout in-distribution; the AV reads any `h`
regardless (`nla_scan.py` also verbalizes raw-prompt `h`). Capture position is
the **last token** (the next-token-prediction slot, where the J-lens readout is
most interpretable and the concept is maximally loaded).

- **Neutral set — `heldout_prompts_wikitext103_n30.json`** (the jspace held-out
  30), first `--n-neutral` (default 12). Strictly disjoint from the fitting
  corpus (indices 1001-1030). Tests baseline channel agreement on generic text.
- **Concept-loaded set — `CONCEPT_PROMPTS`** (defined in the script, `--n-concept`
  default 12): short prompts each loading one concrete concept at the final
  token, annotated with the expected concept token(s). Tied to the NLA arc's
  concept-direction / country themes (Theme 6, F-country-CAV): e.g.
  `("The capital of France is", ["Paris", "France"])`,
  `("The chemical symbol for gold is", ["gold", "Au"])`,
  `("A photograph of a red apple on a wooden", ["table", "apple"])`. The known
  target enables a third agreement axis (§3, Metric 3) unavailable on neutral
  text.

## 3. Experiment A — channel agreement (J-lens top-k vs NLA verbalization)

Per (prompt, last position): capture `h`; compute J-lens readout logits
`ell = W_U norm(J_19 h)` over the full base vocab; verbalize `h` with the AV.

**Filter (defined once, applied to both channels).** Content-word set of a text
= regex words `[a-z]{3,}` after lowercasing, minus `STOPWORDS` (a fixed English
function-word list: articles, prepositions, conjunctions, pronouns, auxiliaries,
common light verbs) **and** `AV_TEMPLATE_STOP` (the AV's meta-description
vocabulary — `activation, vector, concept, semantic, content, represents,
appears, context, token, model, text, structured, format, explanation, …` —
motivated by NLA arc Theme 9 / L2, the AV format-bias caveat: these words carry
verbalizer prior, not input content). J-lens "content tokens" = decode each
top-k id, `strip().lower()`, keep if alphabetic, `len>=3`, not in the stoplists.

**Metric 1 — NLA→J-lens rank (primary; resolves the single-token vs free-text
mismatch).** The J-lens is fundamentally single-token — multi-token concepts are
"invisible or fractured" (`j-space.md §5`, `[gurnee2026-workspace §9.1]`) — so
we do *not* ask the J-lens to emit free text. Instead we look up where the AV's
words fall in the J-lens ranking: for each AV content word `w`, encode with the
base tokenizer (leading-space form, first sub-token) → id, take
`rank_w = #{v : ell_v > ell_id}` (0 = top-1). Report per prompt:
`nla_rank_median` (median over words; lower = stronger agreement) and
`nla_in_topk_frac` (fraction with `rank < K`, `K=50`).

**Metric 2 — J-lens→NLA overlap (symmetric, coarse).** `topk_overlap` =
fraction of J-lens top-`K` content tokens that appear (whole-word or substring,
casefolded) in the AV content-word set. Substring match absorbs BPE sub-token
fragmentation.

**Metric 3 — expected-concept hit (concept set only).** For each annotated
target `T`: `jlens_hit` = any `T` token in J-lens top-`K`; `nla_hit` = any `T`
lemma in the AV content words. Cross-tabulate → do the two channels
independently recover the *known* concept, and do they co-fire?

**Null / significance (no extra verbalizations — post-hoc).** Mismatched
pairing: score prompt `p`'s AV words against prompt `q≠p`'s J-lens readout, all
ordered pairs. Agreement is real iff matched `nla_rank_median` is materially
below the mismatched distribution (report matched vs mismatched medians and the
rank of the matched value in the mismatched distribution as a cheap
permutation-style p). Also report the chance floor for Metric 1 (a uniformly
random token has expected rank `|V|/2 ≈ 76032`).

## 4. Experiment B — is the NLA content in the J-space component?

The J-space component carries <10% of the activation's variance
(`[gurnee2026-workspace §4]`; jspace stage-4 confirmed low varfrac), so this
asks whether a trained verbalizer's content concentrates in that small
component.

Per (prompt, last position), on a subset (`--n-decomp` default 12, split evenly
neutral/concept):

1. Gradient-pursuit decompose `h` at `k=25` (headline k, via the shared
   `_jspace_pursuit.gradient_pursuit_layer(..., return_components=True)` — the
   *identical* pursuit the structure scan uses) → component `c` (`= A x`),
   residual `res = h − c`.
2. **Norm-matched control** for the residual: draw `r ~ N(0, I_d)`, rescale to
   `||r|| = ||c||` (seed fixed), form `res_ctrl = h − r`. Because
   `nla_verbalize` is magnitude-invariant, the only thing distinguishing `res`
   from `res_ctrl` is *which* direction of norm `||c||` was removed — so
   `res` vs `res_ctrl` isolates the J-space-specific effect from the generic
   effect of perturbing `h` by a `||c||`-sized vector.
3. Verbalize four vectors: `h` (reused from Experiment A), `c`, `res`,
   `res_ctrl`.
4. **Scoring.** Content-word Jaccard `J(·, ·)` between AV texts, plus Metric 1/2
   of each against the J-lens readout of `h`:
   - `carrier_component = J(text(c), text(h))` — high ⇒ the tiny J-space
     component alone verbalizes to the same content as the full activation
     (the strong positive).
   - `damage_jspace = J(text(h), text(h)) − J(text(res), text(h))` vs
     `damage_ctrl = J(text(h), text(h)) − J(text(res_ctrl), text(h))`
     (`J(text(h),text(h))=1`); **`damage_jspace − damage_ctrl > 0`** ⇒ removing
     the J-space component specifically degrades the NLA content more than
     removing a norm-matched random direction.
   - Controls for `c`: `text(c)` vs `text(random unit vector)` (one shared
     random control per run) — guards against the AV emitting a coherent concept
     for *any* direction.

**Headline claim shape.** "NLA content is J-space-concentrated at layer 20" is
supported iff `carrier_component` is high AND `damage_jspace − damage_ctrl > 0`,
across prompts. A null (component verbalizes to noise; residual carries the
content) is an in-scope deliverable and equally publishable — it would say the
NLA reads a channel the J-lens misses.

## 5. Artifact

`torch.save({"summary", "per_prompt"}, out)`, auto-named
`research/arcs/jspace/data/cache/nla_crosstie_<model-slug>_<lens-stem>.pt`
(e.g. `nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`). Class:
derived (rederivable from the lens + models). `per_prompt`: prompt text, set
tag, seq_len, position, J-lens readout top-k ids, per-metric values, AV texts
(`h`/`c`/`res`/`res_ctrl`), decomposition varfrac. `summary`: config (model,
mode, lens, `jlens_layer=19`, K, k, injection_scale, seeds, counts), matched vs
mismatched Metric-1/2 aggregates, Experiment-B carrier/damage aggregates,
per-metric means split by set, chance floors, timings. On close: an
`nla_crosstie` audit section re-derives the load-bearing numbers from this
artifact (stage 7), and an observation writeup per finding.

## 6. Item counts + runtime (ESTIMATE, pre-GPU)

Defaults: 12 neutral + 12 concept = 24 prompts (Experiment A, 1 AV verbalize
each) + 12 decomposition prompts × 3 extra verbalizations (`c`, `res`,
`res_ctrl`; `h` reused) + 1 shared random-unit control ≈ **61 AV verbalizations**.

- AV verbalize: ~85-130 s each on CPU (arc-measured ~85 s at 200 tok; up to ~130 s
  when the AV runs to the cap). 61 × ~110 s ≈ **~1.9 h** — the dominant cost.
- Base Phase 1 (7B nf4 GPU): capture + readout + pursuit ≈ a few s/prompt; 24
  prompts ≈ **~5 min**. Pursuit to `k=25` over 1 layer/position is far cheaper
  than the structure scan's all-layer sweep.
- Loads: base nf4 ~1-2 min; AV ~1-2 min (15 GB from local cache).
- **Total wall ≈ ~2 h** (single run, `--n-neutral 12 --n-concept 12 --n-decomp 12`).
  A quick first pass (`--n-neutral 4 --n-concept 4 --n-decomp 4` ≈ 20
  verbalizations) is ~40 min. All labeled estimate; the AV per-call time is the
  only real unknown at 7B and is measured in the arc records.

## 7. Risks / open questions

- **AV format-bias (NLA arc L2, unaudited).** If the AV emits its template
  vocabulary regardless of input, Metric-2 overlap inflates. Mitigation: the
  `AV_TEMPLATE_STOP` filter + the mismatched-pairing null + the random-unit
  control for `c`. A strong mismatched baseline would itself flag the bias.
- **Base-tokenizer vs AV-tokenizer vocab.** Assumed shared Qwen2.5 vocab
  (152064); the script asserts base `|V| == readout width` and uses the base
  tokenizer for all rank lookups. Flag if the AV tokenizer diverges (unlikely;
  NLA built on Qwen2.5-7B).
- **nf4 readout fidelity.** The J-lens readout uses the nf4 base `W_U`
  (unquantized `lm_head`) and the fitted `J_19`; the 1.5B bf16 control does not
  apply here (NLA is 7B-only), so nf4 is the only channel. Cross-checked against
  the structure-scan readout at nearby layers for sanity.
- **Single position.** Last-token only in the first pass; mid-position sweep is
  a follow-up if the last-position signal is strong.
