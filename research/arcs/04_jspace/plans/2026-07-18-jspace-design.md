# Design/test plan: J-lens + J-space on Qwen2.5-7B-Instruct

Date: 2026-07-18. Status: **signed off by human 2026-07-18** with two
amendments and defaults taken on open decision points (below). Grounding:
`theory/kb/notes/interpretability/j-space.md` and
`[gurnee2026-workspace §2-§4]`; method details verified against the archived
PDF, not secondary coverage.

**Review outcome (2026-07-18):**

- Approved overall; NLA cross-tie (stage 6) explicitly endorsed; reviewer
  read the Nanda critique and agrees with most points, so the added
  token-steering control stays load-bearing.
- **Amendment A (OOM gate):** if the 7B nf4 fit OOMs on the RTX 2080, PAUSE
  and report measured CPU-vs-GPU runtime estimates before any switch to CPU
  fitting — do not silently fall back.
- **Amendment B (comms):** push-notify milestones and blockers during
  execution (SOP for this arc).
- Decision 1 (corpus): default = inspect the companion repo's Apache-2.0
  prompt sets first and reuse them if suitable (closest method match);
  fall back to a seeded C4-en sample.
- Decision 2 (1.5B bf16 control): yes, as recommended.
- Decision 3 (stage-5 scope): staged — verbal report first, then two-hop and
  modulation as time/results warrant.
- Decision 4 (lens tensor storage): reduced layer subset to LFS
  ({0,5,10,15,20,25,27} per model), full set cache-only + regeneration
  script, as recommended.
- KB + arc ship together in one PR at arc close (no separate KB PR).

**Measured calibration addendum (2026-07-18, supersedes the stage-2 runtime
guesses below):** per-prompt fitting cost is structural
(`ceil(d_model/dim_batch)` exact-Jacobian VJP backwards per prompt, jlens
fitting.py) — measured 115 s/prompt on 1.5B bf16 (dim_batch=8) and 559
s/prompt on 7B nf4 (dim_batch=2 + lm_head offloaded to CPU; the only
non-OOM 7B configuration). So: 1.5B n=100 = 3.2 h, n=1000 = 31.8 h; 7B
n=100 = 15.5 h, n=1000 = 155 h. CPU fallback rejected (RAM-infeasible and
slower than the working GPU path). Full numbers + evidence in
`../observations/2026-07-18-fit-cost-calibration.md`. Execution: 1.5B
n=100 launched 2026-07-18 (checkpointed); 7B n=100 overnight run pending
reviewer confirmation; larger n via checkpoint-resume /
`JacobianLens.merge()` extension if split-half stability at n=100 is poor.
**Executed → 7B n=100 confirmed and run; see
`../observations/2026-07-20-scale-comparison-7b-vs-1p5b-h2.md` (and the
n=500 extension in `../observations/2026-07-22-n500-and-heldout-robustness.md`).**

## 0. What we are computing

Per layer ℓ of Qwen2.5-7B-Instruct (28 layers, d_model=3584, vocab 152,064):

- **J_ℓ** = E_{t, t'≥t, prompt}[∂h_final,t' / ∂h_ℓ,t] — a 3584×3584 matrix
  (~25 MB bf16/layer; ~700 MB all layers — committable via LFS if we keep
  a reduced set of layers, else artifact-cache only).
- **J-lens readout**: softmax(W_U · norm(J_ℓ h_ℓ)).
- **J-space component** of an activation: gradient-pursuit sparse nonnegative
  decomposition onto ≤k J-lens vectors (k=25 default, swept in §4-style
  analysis).

Estimation is VJP-based: one backward pass seeded at h_final,t' yields
projections of ∂h_final,t'/∂h_ℓ,t for all (ℓ,t) at once, so cost scales with
prompts × target positions × probe vectors, not with d_model. The reference
implementation is `anthropics/jacobian-lens` (`jlens.fit`, Apache-2.0), which
wraps HF causal LMs and supports slice-parallel fitting + `merge()`.

## 1. Constraints and the quantization question

The prior arcs load the 7B in **nf4** (bitsandbytes) on the RTX 2080 (8 GB)
via `llm_surgeon.surgery.load_model(BASE_ID, mode="nf4")`. J-lens fitting
needs backprop:

- bitsandbytes nf4 supports gradients w.r.t. *activations* through frozen
  4-bit weights (the QLoRA path), so fitting is mechanically possible.
- Unknown: how much 4-bit weight noise perturbs the *Jacobians* themselves.
  No published data on quantized J-lens fidelity. [SPECULATION] Averaging over
  prompts/positions may wash much of it out.

**Control:** fit a second lens on `Qwen/Qwen2.5-1.5B-Instruct` in **bf16**
(~3.1 GB weights, fits 8 GB with backward at seq 128) — a clean-gradient
reference. If nf4-7B and bf16-1.5B show qualitatively similar J-space
structure (mid-layer band, ≤10% variance, sparse activity), the nf4 7B
results stand; if they diverge, we characterize the divergence before
trusting any 7B finding. A bf16-7B CPU fit is the slow tiebreaker
(llm_surgeon `mode="fp32-cpu"`/bf16 path) — hours, but available.

Known pitfall from the community MLX port (`WeZZard/jlens-qwen36`, tier C):
fused kernels without registered backwards silently block fitting. Qwen2.5 on
HF+CUDA uses standard SDPA attention (differentiable), so we do not expect
this, but stage 1 includes an explicit end-to-end gradient probe.

## 2. Stages

### Stage 1 — environment + gradient probe (cheap gate)

- `pip install -e` the `anthropics/jacobian-lens` repo into the project venv
  (vendored clone under the arc if pinning needed).
- Probe: load nf4 7B, run one 32-token prompt, seed a VJP at h_final, assert
  nonzero finite grads at h_0. If jlens's wrapper fights the quantized model,
  fall back to fitting on top of
  `llm_surgeon.probe._capture_residual_stream_with_grad` (already exists).
- Definition of done: gradient flows end-to-end on both models; fitting one
  demo-grade lens (20 prompts) completes without OOM. OOM mitigations in
  order: seq 128→64, fewer target positions per backward, gradient
  checkpointing, drop batch to 1.

### Stage 2 — lens fitting (capture; ARC_PROCESS step 1)

- Corpus: 128-token sequences from a pretraining-like distribution. Proposal:
  1000 sequences sampled from allenai/c4 (en) with a fixed seed — matches the
  paper's "pretraining-like" intent; the companion repo ships Apache-2.0
  prompt sets we can reuse if compatible.
- Fit J_ℓ for all 28 layers (both models). Paper guidance: quality saturates
  ~100 prompts; we fit at 1000 to match the paper, checkpointing at 100.
- **Validation (data deliverable gate):** split-half stability — fit on two
  disjoint 500-prompt slices, report per-layer Frobenius/cosine agreement;
  sanity checks: J_ℓ→I as ℓ→final; J-lens ≈ logit lens at late layers.
- Artifacts to `data/` + MANIFEST.json (class: raw): lens tensors per
  layer/model + fitting config + corpus manifest (seed, indices).

### Stage 3 — readout characterization (§2-style)

- J-lens vs logit lens token-trajectory views on held-out prompts (the
  "unspoken words" visualization), across all layers.
- Quantitative: per-layer agreement (Spearman) between J-lens readout and
  final next-token logits; depth at which the correct final token enters
  top-k under each lens. Expected from paper: J-lens interpretable earlier.

### Stage 4 — J-space structure (§4-style)

- Implement gradient pursuit (or reuse repo implementation) for k-sparse
  nonnegative decomposition.
- Per layer: variance fraction of J-space component (expect ≤10%, humped in
  a mid-depth band), number of meaningfully-active J-lens vectors (paper:
  ~10-25), kurtosis-by-depth profile.
- Output: the Qwen analog of the paper's "sensory → workspace → motor" depth
  map — the arc's headline structural figure.

### Stage 5 — functional signatures (§3-style, scoped)

Scoped to what a 7B instruct model can reliably do; every experiment filters
to prompts the model answers correctly at baseline, and every swap experiment
carries two controls from the paper (equal-magnitude non-J-space swap,
random-vector swap) plus one from the Nanda critique (logit-lens
unembedding-direction steering — if it matches J-lens swaps, we are "just
token steering", per `sources/forums/2026-07-06-nanda-workspace-review.md`).

1. **Verbal report + swap:** "Think of a {category}" → report; check the
   pre-report J-lens top-1 predicts the reported instance; then Soccer→Rugby
   style swaps, measuring top-5 entry rate of the swap target. Paper
   reference points: 88% (J-lens vectors), 59% (J-space component), 5%
   (non-J-space) on Claude 4.5-family — we record Qwen's numbers, expecting
   attenuation.
2. **Two-hop intermediate swap:** e.g. "the capital of the country where
   {landmark} is" with intermediate-country swap. Precondition: measure
   Qwen2.5-7B's two-hop accuracy first; if the clean two-hop set is too
   small (<50 items), demote this to exploratory.
3. **Directed modulation:** "concentrate on {concept}" during an unrelated
   copy task; measure target-concept presence in mid-layer J-lens readouts
   vs matched no-instruction and "ignore {concept}" baselines.

### Stage 6 — NLA cross-tie (novel; not in the paper)

At layer 20 (the NLA arc's capture layer): compare the J-lens readout of an
activation with the NLA verbalizer's output for the same activation
(`llm_surgeon.probe.nla_verbalize`, `kitft/nla-qwen2.5-7b-L20-av`). Questions:
do the two independent verbalization channels agree on the active concepts;
is the NLA verbalizer's signal concentrated in the activation's J-space
component (decompose, re-verbalize each component separately)? This connects
the two arcs and is the most likely source of an original observation.

### Stage 7 — audit + synthesis (ARC_PROCESS steps 5-6)

`jspace_audit_findings.py` re-derives every load-bearing number from `data/`
artifacts; observation writeups per finding (nulls labeled `-null-result`);
arc README synthesis.

## 3. Risks

- **OOM on 7B backward (medium):** mitigations listed in stage 1; worst case
  the 7B lens is fitted at seq 64 or on CPU overnight.
- **Quantization corrupts Jacobians (unknown):** the 1.5B bf16 control is the
  detector; divergence itself would be a publishable observation.
- **Effects too weak at 7B (medium):** Claude 4.5-family numbers may not
  survive 20x scale reduction. Null results are in-scope deliverables; the
  structural measurements (stages 3-4) are informative regardless.
- **Token-steering confound (design-level):** addressed with the added
  logit-lens steering control in every swap experiment.

## 4. Decision points for review

1. Corpus for fitting (proposal: seeded C4-en sample; alternative: reuse the
   companion repo's shipped prompt sets verbatim for closer method match).
2. Is the 1.5B bf16 control worth the extra fitting run? (Recommended: yes —
   it is cheap and de-risks every 7B claim.)
3. Stage 5 scope — all three signatures, or verbal-report only for the first
   pass?
4. Whether lens tensors (~700 MB/model, all layers) are committed to LFS in
   full, or only a reduced layer subset ({0,5,10,15,20,25,27} ≈ 175 MB) with
   the rest cache-only + regeneration script. (Recommended: reduced subset.)
