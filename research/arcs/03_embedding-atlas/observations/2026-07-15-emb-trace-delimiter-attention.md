# Delimiter self-attention is real but shallow: 26 layer-0-to-3 heads track preceding delimiters (P1a PASS), yet they are a different head population from the W_E block readers, they do not finalize deeper at periods (P1c FAIL), and they match by static content-type in near-DC rotary bands, not by positional resonance (P1d FAIL)

**Date / context:** 2026-07-15. Model Qwen2.5-7B-Instruct, HF revision
`a09a35458c702b33eeacc393d103063234e8bc28`, 28 layers x 28 query heads
(4 KV heads, GQA), head_dim 128, rope_theta 1e6. Eager-attention forward
passes on the same 51-text committed probe corpus as the T0/T1/T1.5 tracing
work ([2026-06-11-emb-trace-block-through-layers.md](2026-06-11-emb-trace-block-through-layers.md);
raw text, no chat template, prefill only). This is experiment **T2**,
adjudicating the pre-registered delimiter-aggregation predictions P1a / P1c /
P1d ([../plans/2026-06-11-predictions.md](../plans/2026-06-11-predictions.md)).
Delimiter set: `['\n', ' ,', ' .', ',', '.', ':', ';', '、', '。', '，']`.
Statistic per (layer, head): mean attention weight from a delimiter query to
each *preceding* delimiter key (**d2d**), vs an **offset-matched** control
where a non-delimiter query attends to non-delimiter keys at the same
query-key distances (**c2c**); the period→comma variant is **p2c**. Corpus
yields 637 delim→delim pairs and 155 period→comma pairs. Scripts:
`emb_trace_attention.py` (capture, ~10 min CPU), `emb_trace_attention_analyze.py`
(model-free adjudication), `emb_trace_attention_render.py` (fig19-21).
Findings continue the F-T numbering from the T1.5 observation (F-T1..F-T4).

## Findings

**F-T5. P1a PASS — 26 delimiter-tracking heads, concentrated in layers 0-3.**
Criterion (pre-registered operationalization of "markedly higher"):
d2d/c2c ≥ 3x AND absolute excess d2d − c2c ≥ 0.03. **26 of 784 heads
qualify**, living in layers {0, 1, 2, 3, 10} — a shallow, early population.
The ratio distribution over all 784 heads is median 0.59x, p90 1.81x,
max 27.15x: most heads *under*-attend delimiter→delimiter relative to the
offset-matched control, and the tracking heads are a distinct right tail, not
a smear. Top head **L0H13** (d2d 0.1784 vs c2c 0.0409, 4.37x, excess +0.1376);
**L0H3** has the extreme ratio 27.15x (0.1128 vs 0.0042). Delimiters do
preferentially attend to preceding delimiters, and it is an early-layer
phenomenon — consistent with the block being read at layer 0 / re-encoded at
layer 1 (F-T2/F-T3). (fig19)

**F-T6. Headline dissociation — reading the W_E block and doing
delimiter→delimiter attention are largely DIFFERENT head populations.** The T1
top block-READER head **L0H15** (q-side block/control norm ratio 3.28x, the
strongest block reader) is **not** a top delimiter head: its delim excess is
+0.0744, ranking **21st of 784** by excess (ratio 11.00x — elevated, but well
outside the top group). Overlap between the top delimiter heads and the top
block-reader heads is **1/5** (only L0H20) and **2/10** (L0H3, L0H20). So the
21-dim structural block that carries delimiter *identity* in the residual
stream (arc finding: the entangled W_E block) is read by one set of heads,
while a mostly-separate set performs the delimiter→delimiter *attention
routing*. The block-reading and the delimiter-matching are decoupled
mechanisms that happen to co-locate in the early layers. (fig19, right panel)

**F-T7. P1c FAIL — period aggregation does NOT finalize deeper than comma
aggregation; both peak at layer 0.** Prediction: period→comma strength peaks at
*deeper* layers than comma→comma (list items interpreted before being
collected up at the period). Measured: the comma-side proxy d2d peaks at
**layer 0** (max-head 0.1784) and period→comma p2c *also* peaks at **layer 0**
(max-head 0.2193), both decaying monotonically-ish with depth. p2c is highest
of all at L0 and never recovers. The falsification does not depend on the proxy
(see Caveats): p2c is captured directly and peaks at L0 on its own. The
"interpret-then-finalize-later" picture is not supported — delimiter attention
is front-loaded, not staged. (fig20)

**F-T8. P1d FAIL — delimiters match by static content-type in near-DC rotary
bands, not by positional resonance in offset-matched mid bands.** Prediction:
the rotary bands carrying the delimiter QK alignment have wavelengths
commensurate with inter-delimiter offsets (~5-30 tokens), i.e. mid-frequency
bands. Measured inter-delimiter offset distribution: min 2, p25 5, median 8,
p75 14, p90 23, max 48 (mean 10.9) — squarely in the 5-30 tok range, so the
prediction's premise holds. But decomposing the positive excess q·k logit
(summed over the top-10 delimiter heads) by rotary band: **99.1% comes from
near-DC bands (wavelength > 1e4 tokens)** and only **0.4% from the mid bands
0-7 (wl 6.3-28.5 tok)** that actually match the offsets. The dominant
contribution is **band 63** (wl ~5.06M tok — effectively a non-rotating / DC
channel). [INTUITION] the near-DC bands barely rotate over any realistic
context, so their q·k contribution is position-independent — a static
content-type match ("both of these positions are delimiters") rather than a
positional-resonance match ("these two positions are 8 tokens apart, matching
the comma spacing"). [SPECULATION] this is consistent with the F-T2/F-T3
picture where delimiter identity lives in a static non-rotating subspace (the
21-dim block) that the attention machinery keys on directly; the network does
not appear to build a RoPE-tuned "list-spacing detector." The prediction's
mechanism (RoPE-band-matched offset detection) is falsified for this corpus;
the *phenomenon* (delimiters attend to delimiters, F-T5) survives via a
different, positionless mechanism. (fig21)

**F-T9. Sink side-note — delimiter queries send slightly LESS attention to
position 0 than control queries.** Mean attention to position 0: delimiter
queries 0.378 vs control queries 0.406 (delta −0.028). Delimiters are not
extra-sink-seeking; if anything they spend marginally more of their attention
budget on the delimiter→delimiter routing (F-T5) and less on the first-token
sink. Consistent with F-T1's finding that the sink machinery is hosted by the
first token, not the delimiters.

## Evidence

```
== T2 delimiter-attention analysis (Qwen2.5-7B-Instruct, 28L x 28H) ==
  probes-derived delim->delim pairs: n=637  period->comma pairs: n=155
  inter-delimiter offset dist (tokens): min 2 p25 5 median 8 p75 14 p90 23 max 48 (mean 10.9)

== P1a: top-10 heads by excess (d2d - c2c) ==
   1. L0H13  0.1784  0.0409  +0.1376    4.37x
   2. L0H8   0.1569  0.0318  +0.1251    4.93x
   3. L0H20  0.1421  0.0232  +0.1189    6.11x
   4. L2H4   0.1400  0.0232  +0.1169    6.04x
   5. L0H3   0.1128  0.0042  +0.1087   27.15x
  ratio distribution over 784 heads: median 0.59x  p90 1.81x  max 27.15x
  criterion d2d/c2c>=3x AND excess>=0.03 -> 26 heads qualify; layers [0,1,2,3,10]
  P1a verdict: PASS

== Cross-ref: delimiter heads vs T1 block-READER heads (q-side) ==
  reader top: L0H15=3.28x L18H7=2.91x L0H10=2.20x L0H2=2.18x L0H20=2.15x ...
  top-5 overlap: 1/5 [L0H20]   top-10 overlap: 2/10 [L0H3, L0H20]
  L0H15 delim excess +0.0744 (rank 21/784), ratio 11.00x -- NOT a top-8 delim head

== P1c: comma-side (d2d) vs period->comma (p2c) layer profile ==
  L0 | d2d(max) 0.1784  p2c(max) 0.2193 | d2d(mean) 0.0925  p2c(mean) 0.0616  <-both peak
  comma-side (d2d) peak layer: 0   period->comma (p2c) peak layer: 0
  P1c verdict: FAIL (prediction: period peaks strictly DEEPER)

== P1d: rotary-band contribution (summed over top-10 P1a heads) ==
    band  wavelength(tok)  excess-logit  %of+total
     63       5063255.8      +1108.32    24.1%   (dominant)
     61       3287985.3      +1049.21    22.8%
     62       4080185.1       +909.70    19.8%
  mid bands [0..7] wl 6.3-28.5 tok: share 0.4%  |  near-DC bands wl>1e4: share 99.1%
  P1d verdict: FAIL (dominant band 63 -> SLOW / near-DC)

== Sink context: mean attention to position 0 ==
  delimiter queries 0.378  vs  control queries 0.406  (delta -0.028)
```

## Reproducibility

```bash
# capture (loads the model; ~10 min CPU). offline HF cache required:
HF_HUB_CACHE=/media/skothr/G-Drive/models HF_HUB_OFFLINE=1 \
  python examples/emb_trace_attention.py       # -> .cache/emb_artifacts/emb_trace_attention.pt
python examples/emb_trace_attention_analyze.py # model-free adjudication (all numbers above)
python examples/emb_trace_attention_render.py  # fig19-fig21 (model-free)
python examples/emb_data_manifest.py --check   # promoted-artifact sha256 + metadata
python examples/emb_audit_findings.py          # AUDIT section 10 locks every number here
```

## Hypotheses / follow-ups

- **(→ P1b, still open)** Norm-normalized per-band cosine. F-T8 weights bands
  by raw q·k logit magnitude; a re-capture storing per-band norms would let us
  test whether the mid bands carry a *directionally* aligned (but small-norm)
  signal that the raw-magnitude view suppresses. Current evidence falsifies the
  raw-contribution form of P1d; the cosine form is not yet derivable from the
  stored sums.
- **(→ P1c re-test)** Capture a *pure* comma→comma accumulator (and a pure
  period→period one) rather than the all-delimiter d2d proxy, to confirm the
  L0 peak is not an artifact of pooling `\n`/`:`/`;` into the comma side.
- **(→ T4 ablation, scope change)** The reader/delimiter dissociation (F-T6)
  means the planned W_E-block-dim ablation now has TWO target head populations:
  the block *readers* (L0H15 cluster) and the delimiter *routers* (L0H13/H8/H20,
  L2H4). Ablating the block dims tests the readers; a separate head-level
  ablation is needed to test whether killing the delimiter routers degrades
  list/clause parsing.
- **(→ P1e / P3, still open)** V-content decode at the delimiter positions
  (P1e) and the cross-script grammatical-role probe (P3) are untouched by T2.

## Caveats

- **51-probe corpus** (637 delim→delim / 155 period→comma pairs), delimiter-
  skewed; head *rankings* and pass/fail verdicts are robust, exact counts
  (26 heads) are threshold-sensitive.
- **d2d is a proxy** for pure comma→comma: the capture pools all delimiter
  types on the comma side. The P1c falsification does not rest on it — p2c is
  captured directly and peaks at L0 independently — but the "comma-side"
  labeling of d2d is approximate.
- **P1d band weighting is raw q·k logit magnitude**, not norm-normalized cosine
  (see follow-up 1). The near-DC dominance is a statement about which bands
  contribute the most *logit*, which is the quantity that actually drives the
  softmax; the directional-alignment view is a separate, not-yet-run test.
- **Single model** (Qwen2.5-7B-Instruct); the rope_theta 1e6 sets the band
  wavelength grid, so the "no mid-band match" result is specific to this
  positional-encoding configuration.
- Eager-attention capture reproduces the model's real attention weights
  (not an approximation), but the offset-matched control is a *statistical*
  baseline (matched on query-key distance, not on surrounding content).

## References

- Pre-registered predictions: [../plans/2026-06-11-predictions.md](../plans/2026-06-11-predictions.md) (P1a, P1c, P1d, P1e).
- Prior tracing observation (F-T1..F-T4, block reader heads, carrier subspace,
  sink census): [2026-06-11-emb-trace-block-through-layers.md](2026-06-11-emb-trace-block-through-layers.md).
- Figures: fig19-fig21 in [figures/INVENTORY.md](figures/INVENTORY.md).
- Audit: `examples/emb_audit_findings.py` section 10.
