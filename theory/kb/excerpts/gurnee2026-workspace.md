---
paper_key: gurnee2026-workspace
title: "Verbalizable Representations Form a Global Workspace in Language Models"
authors: Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson, Lindsey
year: 2026
venue: Transformer Circuits (Anthropic), 2026-07-06
arxiv: null
local_pdf: sources/papers/gurnee2026-workspace_verbalizable-global-workspace.pdf
type: excerpts
note: >
  Excerpts verified verbatim against the local PDF snapshot (chromium
  print-to-PDF of https://transformer-circuits.pub/2026/workspace/index.html,
  captured 2026-07-18, 129 pp). Verification method: pdftotext extraction +
  exact-substring match per passage. Equations re-set in LaTeX from the
  extracted text (pdftotext scrambles display-math layout); symbol content
  verified against §2.1. Companion code: github.com/anthropics/jacobian-lens
  (Apache-2.0); companion external-commentary PDF archived alongside the paper.
---

# Excerpts — Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in Language Models"

## §1.1 Global workspace theory background {#sec-1-1-gwt}

> In this account, the brain is composed of many specialized processors
> operating largely in parallel and in isolation, whose activity proceeds
> outside of conscious access. A representation becomes consciously accessible
> when it is posted to a shared "global workspace" from which many downstream
> processes can read [14]. Under the theory, the workspace is a processing hub
> that integrates and broadcasts information, allowing it to be used for
> flexible internal reasoning and report [5, 15]. Notably, the workspace is
> held to be limited in capacity, so entry is competitive and subject to
> attentional modulation

## §1.2 Main thesis {#sec-1-2-thesis}

> In this paper, we provide evidence that LLMs do possess such workspace-like
> representations. We identified them by searching for representations
> satisfying the first property, namely those that are verbalizable. We then
> discovered that, rather surprisingly, they satisfy the others. These
> representations consist of a small, evolving set of unspoken words, neither
> pure echoes of the input nor predictions of the next token, naming the
> concepts the model is currently reasoning with.

## §2.1 The Jacobian lens — averaged-Jacobian definition {#sec-2-1-jlens-def}

> We isolate the former component by averaging within and across contexts.
> For each layer ℓ, we compute

$$
J_\ell = \mathbb{E}_{t,\; t' \ge t,\; \text{prompt}}
\left[ \frac{\partial h_{\text{final},\,t'}}{\partial h_{\ell,\,t}} \right],
$$

> where the expectation is taken over the source position t, all subsequent
> positions t′ within the context, and a corpus of one thousand prompts
> sampled from a pretraining-like distribution. The result is a single
> d_model × d_model matrix per layer that maps from a source layer ℓ to the
> final layer L.

## §2.1 The Jacobian lens — readout {#sec-2-1-jlens-readout}

> Applying the lens to an activation hℓ is equivalent to replacing all
> subsequent layers with the appropriate lens matrix, followed by the normal
> unembedding operations (typically normalization, then multiplication by the
> unembedding matrix W_U):

$$
\text{lens}(h_\ell) = \operatorname{softmax}\!\big(W_U\, \operatorname{norm}(J_\ell\, h_\ell)\big)
$$

> This produces a score for every token in the model's vocabulary.

## §2.1 Relation to the logit lens {#sec-2-1-vs-logit-lens}

> While the logit lens assumes that representations use the same coordinates
> in all layers, the Jacobian lens corrects for representational changes that
> take place across layers, allowing it to uncover meaningful information in
> earlier layers where the logit lens produces uninterpretable readouts.
> Collectively, the J-lens vectors comprise a subcomponent of the model's
> representational space which we term the J-space.

## §2.3 J-space definition — sparse nonnegative cone {#sec-2-3-jspace-def}

> We therefore define the J-space as the set of points expressible as a sparse
> nonnegative combination of J-lens vectors. For the J-space to be properly
> defined, we must specify an allowable sparsity level k — this parameter is
> somewhat arbitrary, and we vary our choice of k throughout the paper, but we
> typically choose it to be no more than 25, which we empirically observed to
> be the number of J-lens vectors that are meaningfully active at a given time
> (§4.2). Geometrically, for a given k, the J-space corresponds to a union of
> k-dimensional cones, one for each possible set of k J-lens vectors. For a
> given point in activation space, we can define its J-space component as the
> point in the J-space nearest to it, and its non-J-space component as the
> difference between these points.

## §2.3 Gradient pursuit decomposition + variance share {#sec-2-3-gradient-pursuit}

> We operationalize identifying the contents of the J-space by sparse
> decomposition. Given an activation (or a steering vector, or an SAE feature
> direction), we solve for a sparse nonnegative combination of k J-lens
> vectors that approximate it well using gradient pursuit [34, 35]. This
> combination is our approximation of the activation's J-space component, and
> the coefficients of these vectors are its local J-space coordinates. In
> §4.2, we find that the J-space component typically accounts for only a
> small fraction of total activation variance (varying by layer, but never
> more than 10%).

## §2.3 Superposition-hypothesis interpretation {#sec-2-3-subframe}

> To provide another interpretation, under the superposition hypothesis [29],
> a model's activations decompose as sparse linear combinations drawn from an
> overcomplete set of feature directions — a sparse frame, in the sense of
> linear algebra, rather than a basis. The J-lens vectors would then
> constitute a subframe of this feature frame

## §3.1 Verbal-report swap experiment {#sec-3-1-report-swap}

> In this example, we subtract the projection onto the Soccer lens vector and
> add an equal-magnitude projection onto the Rugby lens vector. After this
> swap, the model reports "Rugby" as the sport it thought of (Figure 6, left,
> "After swap").

## §1.2/§3.5 Selectivity property {#sec-3-5-selectivity}

> Selectivity. The workspace comprises a small subset of the total
> representational content of the model's activations. It is required for
> only a fraction of the model's behavior, and in particular is not involved
> in pervasive, routine processing like text parsing or grammatical fluency.

## §3.3 Multihop entailed-property swap — spider→ant example {#sec-3-3-spider-ant}

> In the first example, the prompt is "The number of legs on the animal
> that spins webs is". To predict the next word correctly, the model must
> first infer that the animal in question is a spider, and then report
> the number of legs a spider has. The Jacobian lens at intermediate
> layers confirms that spider is represented at the relevant token
> positions, even though the word never appears in the prompt or the
> output. When we swap the spider lens vector for ant, the model's top
> output changes from "8" to "6", the number of legs on an ant.

(Verified verbatim 2026-07-21. This is the paper's demonstrated
entailed-property flip: the measured output is a *property* of the
swapped concept, not the concept token itself — the strongest causal
form, since token steering alone does not carry the property knowledge.)

## §4.1 Kurtosis onset signature + metric definition {#sec-4-1-kurtosis-onset}

> The second (panel b) shows the excess kurtosis of J-lens readouts, a
> measure of their nonrandomness. Excess kurtosis is near zero through the
> first third of the layers, increases beginning around a third of the way
> through the depth, and falls in the last few layers. We interpret the
> early rise as marking the beginning of the workspace: before it, the
> J-space carries essentially no meaningful content.

Footnote 4 (the metric definition):

> Specifically, we compute the excess kurtosis of the logit distribution for
> the readout of a single (position, layer) across a large data set of
> activations.

(Verified verbatim 2026-07-20 during the jspace arc's stage-4 replication —
the metric is the **logit** distribution over the full vocabulary, no
softmax, no top-k truncation; the "top-k" curves in the adjacent panel (a)
belong to the next-token-accuracy metric, not kurtosis.)

## §5 Alignment auditing — evaluation awareness {#sec-5-eval-awareness}

> Strikingly, we observe that the workspace sometimes encodes recognition of
> being in an evaluation (fake, fictional), and that ablating these
> representations can surface malicious propensities that were otherwise
> concealed. In addition, we find that in two models trained to have
> misaligned behaviors, the J-lens uncovers internal signatures of their
> intentions: it reveals reward and bias in a model trained to appease biases
> in reward models used for training, and secretly and trick in a model
> disposed to write malicious code.
