---
paper_key: ethayarajh2019-contextual
title: "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
authors: Ethayarajh
year: 2019
venue: EMNLP-IJCNLP 2019
arxiv: 1909.00512
local_pdf: null
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (2 Sep 2019). The PDF is not stored in the repo (#124). Only line breaks and hyphenation are normalized. The measured anisotropy is of contextual hidden-layer representations; the paper reports the input layers as less anisotropic.
---

# Excerpts — Ethayarajh 2019, "How Contextual are Contextualized Word Representations?"

## §1 Introduction — the narrow cone {#sec-1}

> In all layers of all three models, the contextualized word representations
> of all words are not isotropic: they are not uniformly distributed with
> respect to direction. Instead, they are anisotropic, occupying a narrow
> cone in the vector space. The anisotropy in GPT-2’s last layer is so
> extreme that two random words will on average have almost perfect cosine
> similarity!

## §4.1 (An)Isotropy — definition and the GPT-2 numbers {#sec-4-1}

> If word representations from a particular layer were isotropic (i.e.,
> directionally uniform), then the average cosine similarity between
> uniformly randomly sampled words would be 0 (Arora et al., 2017). The
> closer this average is to 1, the more anisotropic the representations.
> The geometric interpretation of anisotropy is that the word
> representations all occupy a narrow cone in the vector space rather than
> being uniform in all directions; the greater the anisotropy, the narrower
> this cone (Mimno and Thompson, 2017).

> As seen in Figure 1, for GPT-2, the average cosine similarity between
> uniformly randomly [sic] words is roughly 0.6 in layers 2 through 8 but
> increases exponentially from layers 8 through 12.

> As seen in Figure 1, for all three models, the contextualized hidden
> layer representations are almost all more anisotropic than the input
> layer representations, which do not incorporate context.

> Isotropy has both theoretical and empirical benefits for static word
> embeddings. In theory, it allows for stronger “self-normalization” during
> training (Arora et al., 2017), and in practice, subtracting the mean vector
> from static embeddings leads to improvements on several downstream NLP
> tasks (Mu et al., 2018).
