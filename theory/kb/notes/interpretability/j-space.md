---
topic: interpretability/j-space
status: draft
last_updated: 2026-08-16
maintainer: theory-kb
primary_sources:
  - gurnee2026-workspace
secondary_sources:
  - belrose2023-tuned-lens        # lens lineage; contrast in §3 (and §1.1)
  - nanda2026-workspace-review    # Tier B: LessWrong review; critiques in §5
related_topics:
  - interpretability/lens-techniques
  - interpretability/sparse-autoencoders
  - interpretability/probing
  - interpretability/activation-patching
  - interpretability/mechanistic-interpretability
  - architecture/transformer-overview
---

# J-space and the Jacobian lens (global workspace in LLMs)

Anthropic's claim `[gurnee2026-workspace §1.2; kb/excerpts/gurnee2026-workspace#sec-1-2-thesis]`:
language models maintain a privileged, low-capacity set of *verbalizable*
representations — "a small, evolving set of unspoken words, neither pure
echoes of the input nor predictions of the next token" — that functions like
the global workspace of Global Workspace Theory
`[gurnee2026-workspace §1.1; kb/excerpts/gurnee2026-workspace#sec-1-1-gwt]`.

## 1. Formal definition

### 1.1 The Jacobian lens (J-lens)

For each layer $\ell$, define the **averaged Jacobian**

$$
J_\ell \;=\; \mathbb{E}_{\text{prompt}}\,
\mathbb{E}_{t}\!\left[\ \sum_{t' \ge t}
\frac{\partial h_{\text{final},\,t'}}{\partial h_{\ell,\,t}} \right]
\in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

where

- $h_{\ell,t} \in \mathbb{R}^{d_{\text{model}}}$ — residual-stream activation
  at layer $\ell$, token position $t$;
- $h_{\text{final},t'}$ — final-layer residual-stream activation at a later
  position $t' \ge t$;
- the reduction is a **sum** over subsequent target positions $t'$, a
  **mean** over source positions $t$ (first ~16 positions excluded as
  attention sinks; last position excluded), and a **mean** over ~1000
  prompts from a pretraining-like distribution, weighted equally per
  prompt `[gurnee2026-workspace §2.1, eq. J_ℓ + reproduction pseudocode;
  kb/excerpts/gurnee2026-workspace#sec-2-1-jlens-def]` — verified against
  the companion `jlens/fitting.py` (identical reduction, `skip_first=16`),
  2026-07-23. The sum (not mean) over targets means $J_\ell$ carries a
  per-source weighting by the number of future positions and an overall
  context-length scale; the readout below is invariant to any *global*
  scale of $J_\ell$ (RMSNorm scale-invariance), but the position weighting
  is a real modeling choice.

The **readout** replaces all layers above $\ell$ with this single linear map,
then applies the model's ordinary output head:

$$
\text{lens}(h_\ell) = \operatorname{softmax}\!\big(W_U\,\operatorname{norm}(J_\ell\, h_\ell)\big)
$$

with $W_U \in \mathbb{R}^{|V| \times d_{\text{model}}}$ the unembedding and
$\operatorname{norm}(\cdot)$ the final normalization
`[gurnee2026-workspace §2.1; kb/excerpts/gurnee2026-workspace#sec-2-1-jlens-readout]`.
The **J-lens vectors** are the per-vocabulary-token input directions (rows of
$W_U J_\ell$ pulled back through the lens): the direction in layer-$\ell$
space whose presence most raises that token's readout score.

The logit lens is the degenerate case $J_\ell = I$; the tuned lens replaces
$J_\ell$ with a *learned* per-layer affine map trained on a distillation
objective (correlational, where the Jacobian is causal-by-construction)
`[gurnee2026-workspace §2.4]`, `[belrose2023-tuned-lens §3]`, cf.
`[kb/notes/interpretability/lens-techniques#1-formal-definition]`.

### 1.2 The J-space

The **J-space** is "the set of points expressible as a sparse nonnegative
combination of J-lens vectors" with sparsity level $k$ (typically $k \le 25$)
— geometrically, a union of $k$-dimensional cones, one per possible set of
$k$ J-lens vectors
`[gurnee2026-workspace §2.3; kb/excerpts/gurnee2026-workspace#sec-2-3-jspace-def]`.
An activation's **J-space component** is the nearest point in the J-space,
found by **gradient pursuit** sparse decomposition; the **non-J-space
component** is the difference. The J-space component carries a small fraction
of activation variance — "varying by layer, but never more than 10%"
`[gurnee2026-workspace §2.3, §4.2; kb/excerpts/gurnee2026-workspace#sec-2-3-gradient-pursuit]`.

**Operational definition of the 10% figure** (Fig 30b; verified against the
archived PDF 2026-07-23): it is an *excess over a random control*,
$\text{FVE}(\text{top-}K\ \text{pursuit atoms}) -
\text{FVE}(K\ \text{random atoms})$, where FVE is the
**orthogonal-projection** fraction $\|\Pi_S h\|^2 / \|h\|^2$ (least-squares
onto the selected-atom span, §A.8) and $K$ = the median pursuit occupancy
at each workspace layer — not an absolute reconstruction-energy ratio at a
fixed $k$. The paper leaves the pursuit's atom normalization and iteration
count unspecified (the companion repo ships no pursuit code). Replication
on Qwen2.5 under this exact definition:
`research/arcs/04_jspace/observations/2026-07-24-paper-metric-varfrac-recompute.md`
(1.5B breaches at its hump, 11.15% excess at L21; 7B stays under, 4.72% peak).

## 2. Mechanism

Shapes, for a model with vocabulary $V$, depth $L$:

- One $J_\ell$ per layer: $d_{\text{model}} \times d_{\text{model}}$ — for
  Qwen2.5-7B ($d_{\text{model}}=3584$) that is ~12.8M params/layer; the full
  J-lens-vector dictionary $W_U J_\ell$ ($|V| \times d_{\text{model}}$,
  $|V|=152{,}064$) is computed on the fly, not stored.
- Estimation is by backpropagation: a vector–Jacobian product seeded at
  $h_{\text{final},t'}$ yields one row-projection of
  $\partial h_{\text{final},t'} / \partial h_{\ell,t}$ for **all** $(\ell, t)$
  simultaneously in a single backward pass, so fitting cost scales with
  (prompts × target positions × probe vectors), not with $d_{\text{model}}$.
  Reference implementation: `jlens.fit()` in the companion repo
  (github.com/anthropics/jacobian-lens, Apache-2.0; wraps any HF causal LM).
- Lens quality saturates quickly in prompt count: paper-grade fits used
  1000 prompts × 128 tokens; ~100 prompts is reported usable (companion-repo
  README guidance).

Functional findings (Claude Haiku/Sonnet/Opus 4.5 unless noted; all
percentages verified against the archived PDF):

| Property | Experiment | Result |
|---|---|---|
| Verbal report | "Think of a {category}" then swap J-lens vector (Soccer→Rugby) before report | model reports the swapped word `[gurnee2026-workspace §3.1; kb/excerpts/gurnee2026-workspace#sec-3-1-report-swap]` |
| Report causality | swap along J-space component of a concept vector | target enters top-5 on 59% of trials (88% pure J-lens vectors; 5% non-J-space component) `[gurnee2026-workspace §3.1, Fig. 8]` |
| Internal reasoning | swap unspoken intermediate in two-hop tasks | flips final answer on 54% (Haiku 4.5) / 70% (Sonnet 4.5) / 70% (Opus 4.5) of trials `[gurnee2026-workspace §3.3, Fig. 15]` |
| Flexible generalization | single concept swap (e.g. France→China) reused across function templates | 76/192 trials at baseline; 101/192 at doubled swap strength `[gurnee2026-workspace §3.4, Fig. 19]` |
| Selectivity | same concept present in lens across tasks, causal only in some | workspace "not involved in pervasive, routine processing like text parsing or grammatical fluency" `[gurnee2026-workspace §3.5; kb/excerpts/gurnee2026-workspace#sec-3-5-selectivity]` |

Depth structure: workspace behavior onsets around rescaled depth ~38/100 and
persists to ~92/100 ("sensory → workspace → motor" schema); early-layer
readouts are noisy and the onset boundary may partly be a lens artifact
`[gurnee2026-workspace §4.1]`. [CONTRADICTION] The kurtosis workspace-onset
signature (`[kb/excerpts/gurnee2026-workspace#sec-4-1-kurtosis-onset]`)
**inverted** on Qwen2.5 replication — high-early/low-mid instead of
near-zero-early/rise-at-a-third
(`research/arcs/04_jspace/observations/2026-07-20-jspace-structure-stage4.md`).

## 3. Variants and lineage

- **Logit lens** → J-lens with $J_\ell = I$; fails in early layers because it
  assumes layer-invariant coordinates
  `[gurnee2026-workspace §2.1; kb/excerpts/gurnee2026-workspace#sec-2-1-vs-logit-lens]`,
  cf. `[kb/notes/interpretability/lens-techniques#3-why-the-logit-lens-fails-and-the-tuned-lens-fixes-it]`.
- **Tuned lens** → learned affine translator per layer; correlational, tends
  to "skip ahead" to the output rather than surface currently-held
  intermediates `[gurnee2026-workspace §2.4]`, `[belrose2023-tuned-lens §3]`.
- **SAEs** → under superposition, J-lens vectors are a "subframe" of the
  feature frame: the token-indexed subset of feature directions
  `[gurnee2026-workspace §2.3; kb/excerpts/gurnee2026-workspace#sec-2-3-subframe]`,
  cf. `[kb/notes/interpretability/sparse-autoencoders]`.
- **Alignment auditing application** → workspace surfaces evaluation-awareness
  (*fake*, *fictional*) and trained-in hidden intentions (*reward*, *bias*;
  *secretly*, *trick*)
  `[gurnee2026-workspace §5; kb/excerpts/gurnee2026-workspace#sec-5-eval-awareness]`.

## 4. Intuitions

[ANALOGY] The J-lens asks: "if this activation were allowed to flow to the
output through the *average* downstream circuit, what words would come out?"
It is a stethoscope pressed to layer $\ell$, calibrated on the average
acoustics of the layers above it. Canonically: replace layers
$\ell{+}1 \dots L$ with the single matrix
$J_\ell = \mathbb{E}[\partial h_{\text{final}}/\partial h_\ell]$ and decode.

[INTUITION] The workspace claim is *functional*, not anatomical: J-space is
not a fixed subspace basis but a sparse cone structure ($\le k$ active
directions), and its privileged status is established by intervention
(swaps/ablations change reports and reasoning), not by variance explained —
which stays under 10%. The load-bearing quantity is causal reach, not size.

## 5. Frontier and open questions

- **Single-token limit.** J-lens vectors are indexed by vocabulary tokens;
  multi-token concepts are invisible or fractured `[gurnee2026-workspace §9.1]`.
- **Lossy-lens critique.** [FORUM-SIGNAL] Nanda: expect noise, missed
  concepts, and unmeasured false positives; treat the J-lens as hypothesis
  generation, not verification
  (`sources/forums/2026-07-06-nanda-workspace-review.md`, tier B).
- **Feature-composition counter-hypothesis.** [FORUM-SIGNAL] The two-hop
  "intermediate step" result may be additive feature composition ("Frenchness"
  + "is-capital-city" → Paris) rather than genuine sequential recall (same
  snapshot). [CONTRADICTION] with the paper's §3.3 interpretation; the paper's
  timing evidence (intermediate swaps bind earlier in depth than answer swaps)
  is suggestive but not decisive.
- **Token-steering confound.** [FORUM-SIGNAL] Swap effects on sampled text may
  partly be plain "make the model (not) say word X" steering (same snapshot).
- **Ablation partiality.** The paper's ablations remove "only a fraction of
  the concept" `[gurnee2026-workspace §9.1]`.
- **Open-weights replication (this repo, `research/arcs/04_jspace/`,
  2026-07).** Measured on Qwen2.5-1.5B (bf16) and 7B (nf4); three axes
  diverge from the paper, all certified on the paper's own metric:
  [CONTRADICTION] the ≤10% occupancy ceiling is **breached** at the 1.5B
  hump (L21 excess FVE 11.15%, CI95 [10.95, 11.40]; 7B stays under, peak
  4.72%) — `observations/2026-07-24-paper-metric-varfrac-recompute.md`;
  [CONTRADICTION] the kurtosis workspace-onset signature is **inverted**
  (§2 above) — `observations/2026-07-20-jspace-structure-stage4.md`;
  [CONTRADICTION] the paper's 59% J-space-component report tier lands at
  **random** at both Qwen scales (jspace_comp vs random: 0 discordant
  pairs at 7B; 5 at 1.5B, McNemar null) —
  `observations/2026-07-24-paper-metric-varfrac-recompute.md` Finding 2.
  Layer bands and the relational-causality replication are recorded in the
  arc README. **Data correction:** the arc's C4 corpus slice was redacted for
  PII on 2026-07-29 and every dependent result re-run on 2026-08-15/16; the
  FVE and kurtosis findings above are unchanged by the re-run
  (`research/arcs/04_jspace/README.md` § DATA CORRECTION).

## 6. See also

- `kb/notes/interpretability/lens-techniques.md` — logit/tuned lens formalism
- `kb/notes/interpretability/sparse-autoencoders.md` — superposition, feature frames
- `kb/notes/interpretability/activation-patching.md` — the intervention logic behind swap experiments
- `kb/excerpts/gurnee2026-workspace.md` — verified verbatim passages
- `kb/excerpts/anthropic2026-nla.md` — NLA verbalizer: a *trained* activation-to-language readout, vs the J-lens's *derived* one
