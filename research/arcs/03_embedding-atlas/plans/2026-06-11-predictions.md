# Pre-registered predictions — structural-token tracing phase (2026-06-11)

Recorded BEFORE the Q/K/V capture work begins, so later analysis is tested
against these rather than fitted to them. User-authored prediction (Michael
Lannum, 2026-06-11 session, lightly normalized):

## P1 — delimiter tokens as context aggregators (the comma/period prediction)

> Since commas, periods, etc. generally break things up, the relevant
> handles we found should, at least in some layer, show signs of that.
> E.g. at some layer, a chunking syntax token's Q and K may be very similar
> in some subspace — if it's looking for previous iterations of itself (in
> e.g. a comma-separated list) to pull in previous chunks of context,
> collecting up the previous items to pass forward. A period might do this
> collection moreso — looking for commas before it to finalize the summed-up
> context representing the list of ideas (maybe in the later layers, so the
> individual list items are 'interpreted' before being collected up into the
> period). It might only happen in certain attention heads, or in
> 2-dimension pairings whose RoPE angle frequency matches the token offset
> between commas.

Operationalization (each falsifiable):

- **P1a (comma self-attention):** on prompts containing comma-separated
  lists, some attention head(s) at some layer give commas markedly higher
  attention TO preceding commas than matched non-delimiter tokens at the
  same offsets receive to their own kind. Test: per-head attention-weight
  census over delimiter-vs-control token pairs.
- **P1b (Q/K subspace alignment):** for those heads, the comma's q vector
  and preceding commas' k vectors align in a low-dim subspace of the head's
  128 dims — i.e. cos(q_comma, k_comma') concentrated in few rotary dim
  pairs. Test: per-band decomposition of q·k for delimiter pairs.
- **P1c (period-finalizes-later):** period→comma aggregation strength peaks
  at deeper layers than comma→comma (items interpreted before collection).
  Test: layer profile of the P1a statistic for ','→',' vs '.'→','.
- **P1d (RoPE-band matching):** the bands carrying P1b alignment have
  rotation wavelengths commensurate with typical inter-comma offsets
  (~5-30 tokens), i.e. mid-frequency bands, not the fastest or the
  never-rotating slowest. Test: identify carrying bands, compare wavelength
  to measured inter-delimiter offset distribution in the probe corpus.

### P1e — V carries the collected content; earlier layers prep it
(user clarification, 2026-06-11)

> If there's a layer where lists are collected up (where Q/K may be equal
> for commas), the V would represent the context it was pulling up, at the
> same layer. So maybe a previous layer would have to prep it so the
> incoming hidden state generates the proper V vector to pass downstream.
> Or maybe every layer does this iteratively.

Operationalization: at any layer L where P1a fires, (i) the v vectors at
the attended-to comma positions should encode preceding-span content —
test by decoding v (or the post-attention delta at the aggregating
position) against span-content probes; (ii) ablating the PREVIOUS layer's
write to the comma positions (zero/mean-patch the layer-(L-1) residual
delta at those positions) should degrade the layer-L aggregation more than
ablating matched control positions. Distinguishes "single prep layer"
(sharp degradation from one L-1 ablation) from "iterative accumulation"
(graceful degradation spread over several earlier layers).

## P2 — structural-block persistence (session-posed, pre-QKV)

The 21-dim W_E block's signal remains identifiable in the residual stream
after layer-1 attention + FFN (i.e. the block is consumed/propagated, not
immediately dissolved by the first block's basis rotation). Test: project
post-layer-1 residuals of structural vs control tokens onto the block
subspace; compare separation to layer 0.

## P3 — grammatical role survives within the structural tier (from the
the/的 probe, 2026-06-11, n=8 cosines — NOT yet population-tested)

Quick probe found cos(的, ' of') = +0.621 > cos(的, ' the') = +0.512, and
in-block cos(的, "'s") = +0.701 > cos(的, ' the') = +0.642 — relational/
possessive partners outrank the article, matching 的's actual grammar.
Prediction: a population-scale test over many cross-script function-word
pairs (articles, relational particles, conjunctions, copulas across
zh/ja/ru/de/es) shows within-role cosines exceeding across-role cosines
inside the structural tier. Until that runs, the probe is anecdote.

Falsification of any P is a finding; nulls get written up as nulls.
