---
paper_key: mu2018-allbutthetop
title: "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"
authors: Mu, Viswanath
year: 2018
venue: ICLR 2018
arxiv: 1702.01417
local_pdf: null
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (19 Mar 2018). The PDF is not stored in the repo (#124). Only line breaks (including one page break inside the §1 "Since all words share" quote) and hyphenation are normalized. The mean vector is written µ (U+00B5), the code point pdftotext extracts from the PDF. The paper studies static word embeddings (word2vec, GloVe, and similar); it reports mean-vector norms and variance spectra, not random-pair cosine values.
---

# Excerpts — Mu & Viswanath 2018, "All-but-the-Top"

## Abstract — the postprocessing {#abstract}

> In this paper, we demonstrate a very simple, and yet counter-intuitive,
> postprocessing technique – eliminate the common mean vector and a few top
> dominating directions from the word vectors – that renders off-the-shelf
> representations even stronger.

## §1 Introduction — the observation and the two-step correction {#sec-1}

> The word representations have non-zero mean – indeed, word vectors share
> a large common vector (with norm up to a half of the average norm of word
> vector).

> After removing the common mean vector, the representations are far from
> isotropic – indeed, much of the energy of most word vectors is contained
> in a very low dimensional subspace (say, 8 dimensions out of 300).

> Since all words share the same common vector and have the same
> dominating directions, and such vector and directions strongly influence
> the word representations in the same way, we propose to eliminate them
> by: (a) removing the nonzero mean vector from
> all word vectors, effectively reducing the energy; (b) projecting the
> representations away from the dominating D directions, effectively
> reducing the dimension.

> Nevertheless, a rule of thumb of choosing D around d/100, where d is the
> dimension of the word representations, works uniformly well across
> multiple languages and multiple representations and multiple test
> scenarios.

## §2 Postprocessing — size of the common mean {#sec-2}

> The norm of µ is approximately 1/6 to 1/2 of the average norm of all
> v(w) (cf. Table 1).
