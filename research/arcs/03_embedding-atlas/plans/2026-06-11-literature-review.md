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

## Unverified coverage gaps (claims fetched, not verified — re-sweep when
load-bearing)

1. Near-isotropy of modern decoder W_E (post-2023): unknown whether our
   finding (a) is documented — possibly a small standalone contribution.
2. RoPE frequency-band specialization by heads (Barbero et al. 2024,
   arXiv:2410.06205 fetched but unverified): load-bearing for prediction
   P1d; verify before relying.
3. Cross-lingual shared subspaces (Wendler et al. arXiv:2402.10588 et al.):
   context for the cross-script findings; unverified.
4. Layer-wise tracing methods (logit lens arXiv:2303.08112, Anthropic 2025
   attribution graphs / QK-tracing at transformer-circuits.pub): no
   verified claim on whether any W_E-structure-initiated end-to-end trace
   exists — load-bearing for the novelty argument; verify before claiming
   novelty in any write-up.
