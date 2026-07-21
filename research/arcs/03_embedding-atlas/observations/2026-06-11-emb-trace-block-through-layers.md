# Tracing the W_E structural block through the network: re-encoded at layer 1, carried in a stable moved basis, repackaged at the output — and disjoint from the sink machinery

**Date / context:** 2026-06-11. Model/protocol as in
[2026-06-10-emb-global-geometry.md](2026-06-10-emb-global-geometry.md);
forward passes on a 51-text committed probe corpus
(`emb_trace_capture.PROBES` + `emb_trace_corpus.PROBES_EXT`; raw text, no
chat template, prefill only; 1,130 accumulated non-first positions).
Scripts: `emb_trace_capture.py` (T0/T1), `emb_trace_components.py` (T1.5),
analyses `emb_trace_analyze.py`. Locked by AUDIT 9 (14 claims; total 75
PASS / 0 FAIL). Figures fig16-fig18. Pre-registered predictions:
[../plans/2026-06-11-predictions.md](../plans/2026-06-11-predictions.md).

## Findings

**F-T1. Qwen2.5-7B's massive-activation / attention-sink dims are
identified — and they are NOT the W_E block.** The census (Sun et al. 2024
protocol) finds input-independent massive activations from layer 1 onward
in every probe, led by dim 458 (peak -12,608, ~1000x median) with 2570 and
1427 — exactly three of the seven "sink dims" the nla-verbalizer arc
hand-identified at layer 20, now grounded in the massive-activations
framework. Hosts are the FIRST prompt token (occasionally the second),
never the delimiters in this corpus: Qwen2.5-7B is a first-token-sink
model. The 21 W_E block dims never produce massive activations (their
census entries are layer-0 echoes, |peak| < 10). **Prediction H-SINK
(simple form) is therefore disconfirmed at the dimensional and positional
level** — extending the GPT-2 BOS embedding-independence result
(arXiv:2604.14722) to a delimiter-rich corpus on Qwen; the causal ablation
(T4) remains open. Bonus replication: last-layer-only outlier dims 1069
and 46, hosted by high-frequency tokens (',', ' the', '.'), reproduce
Macocco et al. (arXiv:2503.21718) on Qwen2.5-7B.

**F-T2. The block's content is re-encoded at layer 1 — it moves, it does
not dissolve (prediction P2: refined, not falsified).** Basis-free carrier
analysis (per-layer (21 x 3584) token-wise correlation between layer-0
block coordinates and the layer-L residual, vs a seeded random-21-dim
control): the block's top singular value stays in [8.8, 15.6] across ALL
29 checkpoints and EXCEEDS the control everywhere (min ratio 1.13 at L4,
max 3.3x at L28), while the control decays 7.0 -> 3.9. Meanwhile the
correlation mass remaining in the ORIGINAL 21 dims collapses from 0.109
(L0 reference) to 0.031 at L1. Naive P2 (the signal stays in the original
dims) fails — the delimiter-specific block norm fraction (‖h_S‖/‖h‖,
unsquared) collapses
0.921 -> 0.199 within one layer (fig18) — but the content is fully
recoverable in a moved basis.

**F-T3. Four regimes, with named carriers.** (i) L0: top carriers are the
block dims themselves. (ii) L1-2 handoff: half-new carriers appear (1790,
1923, 2030, 32); three independent measurements coincide here — the
layer-1 input RMSNorm amplifies block dims 2.15x over control (a learned
static gain), the attention residual delta has its block-fraction peak
(0.175, 2.3x the random floor, argmax over all layers/components), and the
layer-0 block-reader q-heads sit immediately upstream (L0H15 reads block
columns at 3.28x control; L0H10/H2/H20/H21 at 2.0-2.2x). (iii) L4-26: a
stable carrier set (adjacent-layer top-10 overlap 7-9/10; recurring new
carriers 1445, 1865, 2545, 19, 3567) in which three ORIGINAL block dims —
2604, 1395 (the comma dim), 1122 — persist as carriers throughout.
(iv) L27-28: carrier overlap collapses to 2/10, the L27 input RMSNorm gain
spikes to 3.28x, attention writes at 1.29x, and a fresh basis forms as
recoverability rises back to 13.1 (vs control 3.9) — structural/frequency
information being repackaged where the last-layer frequent-token dims
live.

**F-T4. The FFN's role: reads neutral, late writes avoid the vacated
dims.** Static maps: FFN gate/up READS of block columns are neutral at
every layer (ratios within [0.95, 1.00]); FFN down-projection WRITES into
block dims are mildly suppressed in mid and late layers (minimum 0.822 at
L24; L22-26 all ~0.82-0.89) — the late network writes around the original
block dims, consistent with the carrier basis having moved. The
dimension-aligned routing is done by RMSNorm gains and attention
(o_proj write ratio 1.29 at L1 and L27), not by FFN weight structure.

## Evidence

```
dim 458 peak -12608.0 first-layer 1 prompts 14  ARC1-SINK
overlap with arc-1 layer-20 sinks: [458, 1427, 2570]
q: mean 1.022 max 3.282 | top: L0H15=3.28, L18H7=2.91, L0H10=2.20, ...
L 0: block sv1 13.277 in-block-mass 0.109 | control sv1 6.995
L 1: block sv1 15.584 in-block-mass 0.031 | control sv1 5.258
L28: block sv1 13.127 in-block-mass 0.010 | control sv1 3.928
L 1: attn delta block-frac 0.1754 (peak) | ffn 0.1031
norm1 block/control: L1 2.15, L27 3.28; ffn_write min 0.822 (L24)
P2 profile: delim 0.9209 (L0) -> 0.1994 (L1); ctrl 0.1596 (L1)
```

## Reproducibility

```bash
python examples/emb_trace_capture.py      # T0/T1 (model load, ~10 min CPU)
python examples/emb_trace_analyze.py      # census/readers/P2 (model-free)
python examples/emb_trace_components.py   # T1.5 (model load + 51 hooked passes)
python examples/emb_trace_render.py       # fig16-fig18 (model-free)
python examples/emb_audit_findings.py     # AUDIT 9 (14 claims) — all PASS; total grows with later sections
```

## Hypotheses / follow-ups

- H1 (-> T2): the moved content acts through attention patterns — the P1
  comma-aggregation predictions should be tested on the named reader heads
  (L0H15 cluster, L18H7) with eager-attention capture.
- H2 (-> T4): runtime ablation of the 21 W_E block dims. F-T1 predicts
  sink formation survives (sink machinery is embedding-independent);
  the open question is what breaks instead — delimiter attention?
  structural parsing? The carrier chain gives a concrete degradation
  readout (carrier SV profile under ablation).
- H3: identify what the stable mid-network carriers (1445, 1865, 2545...)
  read as in vocabulary space (project through W_U / logit lens).
- H4: the L27 norm-gain spike + fresh basis suggests the output
  repackaging feeds the frequent-token heuristic dims (1069/46); test by
  correlating carrier coordinates with those dims' activations at L28.

## Caveats

- 51 probes / 1,130 tokens, delimiter-skewed corpus; carrier-dim
  identities are top-10 rankings (threshold-free) — population scaling
  before promoting carrier claims (thorough-data discipline).
- Correlation SVs are not canonical correlations (no within-set
  whitening); the random-21-dim control anchors interpretation, and the
  control's transient bump at L4 (7.9) narrows the contrast there.
- The census criterion (top-8 recurrence) is coarser than Sun et al.'s
  exact thresholds; dim identities and peaks are robust to this, counts
  are not.
- "First-token sink" is asserted for THIS corpus (no BOS token; raw text);
  chat-template prompts may relocate the sink.
