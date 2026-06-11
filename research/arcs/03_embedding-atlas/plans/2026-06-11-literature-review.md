# Literature review — structural-token machinery, prior art and gaps (2026-06-11)

Produced by an adversarially-verified deep-research sweep (6 angles, 25
sources fetched, 125 claims extracted, top 25 verified by 3 independent
votes each: 23 confirmed, 2 refuted). Verified coverage is THOROUGH for the
outlier-dimensions thread, PARTIAL for attention sinks, and ABSENT for the
other four threads (claims extracted but dropped by the verification
budget) — gaps listed at the bottom are unverified-not-absent.

## Verified prior art (all claims 3-0 or better)

**Massive activations** (Sun et al. 2024, arXiv:2402.17762): rare scalar
activations (>100, ~1000x median; LLaMA2-7B: exactly dims 1415/2533) at
structural/low-semantics token positions — start token, delimiters
('.', '\n', ','), weak function words ('and', 'from', 'of') — that are
input-independent, act as indispensable bias terms, and concentrate
attention on their tokens (generalizing StreamingLLM's attention sink).
Explicitly distinct from LLM.int8 outlier features (scalar-at-position vs
vector-across-tokens; dimension sets don't overlap in LLaMA2-7B/13B).

**LLM.int8 outlier features** (Dettmers et al. 2022, arXiv:2208.07339):
6-7 magnitude-defined dimensions of hidden-state activations entering
projection layers; causally load-bearing (zeroing them: >20% top-1
attention-mass drop, 600-1000% perplexity degradation). Object of study is
activations, never W_E. (Caveat: the 6.7B "phase shift" is OPT-family /
training-recipe contingent — Ahmadian et al. 2023, arXiv:2305.19268.)

**BERT outlier dimensions** (Kovaleva et al. 2021, arXiv:2105.06990):
located in LayerNorm scale/bias (BERT-base: dims 308/381). The paper
explicitly tested the input embedding table and ABANDONED it ("disabling
the weights of the input embedding layers ... produced no significant
change in performance"). Anisotropy link is about contextualized hidden
states, not raw W_E.

**Frequency -> outlier dims -> vertical attention** (Puccetti et al. 2022,
arXiv:2205.11380): in BERT/RoBERTa, outlier-dim magnitude correlates with
pre-training token frequency; disabling 48 parameters (2 dims x LayerNorm
weight+bias across layers) drops MNLI 84.5 -> 58.4; zeroing the outliers
makes the "vertical" attention bars (tokens attending to special tokens
and punctuation) vanish. **Closest published precedent for our finding —
but measured on encoder hidden states / LayerNorm parameters, not a
decoder's input embedding table.** (Bondarenko et al. 2023,
arXiv:2306.12929, argues the training-time causal arrow may run
attention -> outliers.)

**Last-layer outlier dims as frequent-token heuristic** (Macocco et al.
2025, arXiv:2503.21718, 8 models incl. Mistral-7B, Llama-3-8B, qwen-14b):
most last-layer outlier dims are NOT outliers in earlier layers; they
emerge late, serving output generation; the paper never analyzes W_E.
(Heuristic confirmed causally in 5/8 models.)

**Outlier dims are functional, with an operational criterion** (Rudman et
al. 2023, arXiv:2310.17715): variance >= 5x the average variance of the
representation space — usable as a comparison baseline for our 21-dim
block, with the caveat that transferring an activation-variance criterion
to static W_E rows is an adaptation.

## Where our work sits (verified novelty, medium confidence)

No verified prior work analyzes correlated-dimension block structure
inside the static input embedding table of any model. Every canonical
thread targets a different locus: activations entering projections
(Dettmers), LayerNorm parameters (Kovaleva/Puccetti), residual-stream
scalars at structural positions (Sun), last-layer activations (Macocco).
Kovaleva's input-embedding test was a crude full-layer ablation on BERT in
2021, not a geometry analysis.

The literature *independently* establishes the chain
frequency -> outlier dims -> structural-token attention sinks, and Sun et
al.'s massive-activation token class ('.', '\n', ',', 'and', 'of') is
nearly identical to our block's extreme-token class. **Central hypothesis
for the tracing phase (H-SINK): the 21-dim W_E block is the input-side
seed of the attention-sink / massive-activation machinery** — the trace
should test whether the block's signal, propagated through early layers,
feeds the dimensions/positions where Qwen's massive activations live, and
whether arc-1's hand-identified layer-20 "sink dims" (7 dims, universal
sign) are the downstream image of the same machinery.

## Refuted in verification — do NOT cite

- "Token-frequency encoding is established for the Qwen family at 14B"
  (over-extrapolation of Macocco's model list; 0-3).
- "Llama3 untied input/output embedding geometries diverge (cross-model
  correlation only in unembedding space)" (arXiv:2503.21073 misread; 0-3).
  This removes the only candidate literature context for our E/U
  orthogonality finding — that finding currently stands WITHOUT verified
  prior art either way.

## Re-verification sweep (2026-06-11, opus-tiered: 20/20 claims 3-0;
3 adversarial novelty refuters all returned found_prior_art = False)

**Thread 2 — W_E isotropy: our finding (a) appears UNPUBLISHED as a global
measurement.** Targeted searches found no source reporting random-pair
cosine, IsoScore, global PCA spectrum, or participation ratio on the
static W_E of any named post-2023 decoder. The one static-table geometry
paper, Lee et al. 2025 (arXiv:2503.21073; GPT2/Llama3/Gemma2 E and U),
reports only per-token LOCAL intrinsic dimension (~508-635 vs ~605 random
baseline) and cross-model rotation alignment — no global isotropy metrics.
Finding (a) is a small standalone contribution.

**Thread 4 — RoPE bands: P1d is open territory with an existence proof.**
Barbero et al. 2024 (arXiv:2410.06205) verified: single model (Gemma 7B);
band-usage metric = mean 2-norm of per-frequency q/k chunks per layer;
highest frequencies build robust positional heads (diagonal,
previous-token), most norm sits in low frequencies (semantic). Their
"apostrophe head" (high-frequency previous-token apostrophe detection with
low-frequency BOS fallback) is the closest published thing to a
token-specific band-specialized structural head. NO work connects rotary
bands to delimiter/list tracking — the comma-offset band-matching
prediction is unworked. Also relevant for T1: arXiv:2502.01563 finds
massive Q/K values from layer 1 concentrated in RoPE low-frequency bands.

**Thread 5 — cross-lingual: finding (d) is an EXTENSION, not novel in
kind.** Wen-Yi & Mimno 2023 ("Hyperpolyglot LLMs", arXiv:2311.18034,
EMNLP) is direct prior art for cross-script translation neighbors in the
RAW static input table — on mT5 (and the opposite, language-identity
organization, in XLM-R); no decoder-only models. Wendler et al. 2024
(arXiv:2402.10588) is hidden-states/logit-lens (Llama-2), not the table.
No prior art found for grammatical-role geometry across scripts inside the
structural tier (our P3).

**Thread 6 — tracing: the W_E-structure-initiated trace is unclaimed, but
H-SINK has named competition AND contrary evidence.**
- Tuned lens (arXiv:2303.08112) decodes hidden states to vocab; Anthropic
  2025 attribution graphs / circuit-tracer use token embeddings as input
  NODES but trace transcoder features, not W_E dimension structure; the
  2025 QK-tracing work decomposes attention scores over features.
- Sink-origin accounts locate the machinery AWAY from W_E: House of Cards
  (arXiv:2410.01866) in FFN down-projection massive weights; Super Weights
  (arXiv:2411.07191) similarly; arXiv:2605.06611 in forward-pass variance
  discrepancy + super neurons; and a GPT-2 mechanistic account
  (arXiv:2604.14722) EXPLICITLY RULES OUT the token embedding for the
  GPT-2 BOS sink (zeroing the BOS embedding leaves the sink intact).
  H-SINK must therefore be framed as a genuine question — the BOS-sink
  result is contrary evidence, though our block concerns DELIMITER tokens
  (model-family-dependent per Sun 2024), not BOS, and no equivalent
  ablation exists for delimiter sinks on Qwen.
- MUST-READ before any write-up: Cancedda 2024 (arXiv:2402.09221,
  "Spectral Filters, Dark Signals, and Attention Sinks") — the closest
  published W_E-adjacent sink work; compares SVD spectra of W_e vs W_u and
  links (un)embedding subspaces to sinks. Also arXiv:2603.05498 ("The
  Spike, the Sparse and the Sink", 2026) and the catch-tag-release
  mechanism (arXiv:2502.00919).
- Methodological precedent for P2 persistence: Mickus et al. 2022 ("How to
  Dissect a Muppet", TACL) decomposes contextual embeddings into
  static-input + attention + FFN terms and tracks the static term's decay
  across layers — adjacent method, no dimension-block analysis.

**Demotions from the novelty attack:**
- Finding (c) (E/U orthogonality) HAS direct prior art: arXiv:2603.26663
  ("Weight Tying Biases Token Embeddings Towards the Output Space", 2026)
  compares input vs output embeddings across tied/untied models incl.
  Qwen3 via cosine/Procrustes/kNN. Read and cite; our per-token cos ~ 0 on
  Qwen2.5-7B is at best a corroborating data point.
- Strongest near-miss for the block analysis: an Alignment Forum
  exploration of GPT-2's wte (SVD components incl. a leading-space split
  and a frequency-correlated "dimension 138") — tier-B/C source,
  single-dimension and informal, but honest precedent for
  frequency-structure inside a static table. Cite as prior signal.

**Net novelty verdict (post-attack):** (1) correlated-dimension BLOCK
analysis of a modern decoder's W_E — no prior art (nearest: the GPT-2 blog
SVD exploration; Cancedda's spectral comparison). (2) Layer-by-layer trace
initiated from W_E structure — no prior art (nearest: Mickus 2022
decomposition). (3) W_E -> attention-sink connection — open question with
live competing FFN-origin accounts and the GPT-2 BOS counter-example;
a null here is a publishable finding either way. Findings (c) and (d) are
context/corroboration, not claims.
