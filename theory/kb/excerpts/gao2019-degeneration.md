---
paper_key: gao2019-degeneration
title: "Representation Degeneration Problem in Training Natural Language Generation Models"
authors: Gao, He, Tan, Qin, Wang, Liu
year: 2019
venue: ICLR 2019
arxiv: 1907.12009
local_pdf: null
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (28 Jul 2019). The PDF is not stored in the repo (#124). Only line breaks and hyphenation are normalized. The degenerate embeddings studied are tied input/output word-embedding tables of Transformer generation models; the paper reports the sign of pairwise cosines, not a numeric range.
---

# Excerpts — Gao et al. 2019, "Representation Degeneration Problem"

## Abstract — narrow cone under weight tying {#abstract}

> We observe that when training a model for natural language generation
> tasks through likelihood maximization with the weight tying trick,
> especially with big training datasets, most of the learnt word embeddings
> tend to degenerate and be distributed into a narrow cone, which largely
> limits the representation power of word embeddings.

## §3.2 Discussion — the machine-translation embedding table {#sec-3-2}

> We observe very different phenomena for machine translation. We can see
> from Figure 1(a) that the word embeddings are clustered together and only
> lying in a narrow cone. Furthermore, we find the cosine similarities
> between word embeddings are positive for almost all cases. That is, the
> words huddle together and are not well separated in the embedding space.
