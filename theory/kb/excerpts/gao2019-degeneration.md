---
paper_key: gao2019-degeneration
title: "Representation Degeneration Problem in Training Natural Language Generation Models"
authors: Gao, He, Tan, Qin, Wang, Liu
year: 2019
venue: ICLR 2019
arxiv: 1907.12009
local_pdf: null
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (28 Jul 2019). The PDF is not stored in the repo (#124). Only line breaks and hyphenation are normalized. The degenerate embeddings studied are tied input/output word-embedding tables of generation models; the §3 analysis is mainly of a Transformer machine-translation model, with an LSTM-based model reported as similar, and the method is also tested on AWD-LSTM language models. The paper reports the sign of pairwise cosines, not a numeric range.
---

# Excerpts — Gao et al. 2019, "Representation Degeneration Problem"

## Abstract — narrow cone under weight tying {#abstract}

> We observe that when training a model for natural language generation
> tasks through likelihood maximization with the weight tying trick,
> especially with big training datasets, most of the learnt word embeddings
> tend to degenerate and be distributed into a narrow cone, which largely
> limits the representation power of word embeddings.

## §3.1 Experimental design — the tied Transformer table {#sec-3-1}

> Our analysis reported in this section is mainly based on the
> state-of-the-art machine translation model Transformer (Vaswani et al.,
> 2017).

> Additionally, we also analyze the LSTM-based model (Wu et al., 2016) and
> find the observations are similar. In neural sequence generation tasks,
> the weights in word embeddings and softmax layer are tied.

## §3.2 Discussion — the machine-translation embedding table {#sec-3-2}

> We observe very different phenomena for machine translation. We can see
> from Figure 1(a) that the word embeddings are clustered together and only
> lying in a narrow cone. Furthermore, we find the cosine similarities
> between word embeddings are positive for almost all cases. That is, the
> words huddle together and are not well separated in the embedding space.
