# Observation: J-lens advantage appears on intermediate-concept metrics (H3 confirmed)

**Date/context:** 2026-07-18, same session and setup as
`2026-07-18-readout-scan-1p5b-first-pass.md` (Qwen2.5-1.5B-Instruct bf16 CPU,
lens `jlens_qwen2.5-1.5b_bf16_n100`). Ran the companion repo's own
evaluation sets (`anthropics/jacobian-lens` `data/evaluations/`), which score
the rank of *intermediate* concepts — not the final output token — in lens
readouts. Script: `examples/jspace_lens_eval.py`; layer bands early 0-8,
mid 9-18, late 19-26; metric: fraction of instances whose intermediate-token
best rank within the band beats top-10 / top-50.

## Finding

**H3 from the first-pass observation is confirmed: the J-lens advantage
exists precisely on the metric the paper claims it for, and is invisible on
output-predictive metrics.** On `lens-eval-multihop` (93 items, 103
intermediate instances) the J-lens surfaces the unspoken intermediate
concept where the logit lens finds *nothing at all* in early/mid layers:

| band | J@10 | J@50 | logit@10 | logit@50 | Δ@10 |
|---|--:|--:|--:|--:|--:|
| early (0-8) | 0.20 | 0.20 | 0.00 | 0.00 | +0.20 |
| mid (9-18) | 0.13 | 0.24 | 0.00 | 0.00 | +0.13 |
| late (19-26) | 0.50 | 0.65 | 0.40 | 0.64 | +0.10 |
| overall | **0.58** | **0.74** | 0.40 | 0.64 | **+0.18** |

Together with the first-pass result (logit lens wins on
Spearman-to-final-logits and final-token emergence), the two observations
form a coherent replication of the paper's positioning `[gurnee2026-workspace
§2.1, §2.4]`: the logit lens is the better *output predictor*; the J-lens is
the better *currently-held-concept surfacer* — and its advantage
concentrates in early/mid layers, exactly the "earlier layers" clause of
`kb/excerpts/gurnee2026-workspace#sec-2-1-vs-logit-lens`.

`lens-eval-association` (102 instances) hit a floor on both lenses
(all bands ~0.00-0.02 at top-10/top-50). Per-example best ranks still favor
the J-lens by large factors (e.g. 'pregnant': J rank 1564 @L12 vs logit
21951 @L25), but neither lens brings associations into top-50 —
consistent with a 1.5B capability floor and/or associations not being held
as single-token representations in so small a model. [SPECULATION]

## Evidence

Run log excerpt (multihop, 189.4 s; association, 250.9 s; CPU):

```
'carnival-ocean'  concept='Brazil' tok=' Brazil' | J: L15 rank 32  | logit: L22 rank 121
'amazon-language' concept='Brazil' tok=' Brazil' | J: L26 rank 4   | logit: L26 rank 6
'grief'           concept='grief'  tok=' grief'  | J: L13 rank 7182 | logit: L13 rank 27845
```

The 'carnival-ocean' row is the paper's phenomenon in miniature: the
unspoken intermediate ("Brazil") is visible in the J-lens at layer 15 (rank
32) while the logit lens needs layer 22 and still ranks it 121.

Derived artifact: `data/cache/lens_eval_1p5b_n100.pt`. Inexact-token cases
(9/103 multihop, 3/102 association — intermediates that don't map to a
single vocabulary token) scored via the script's documented fallback; see
script header.

## Method notes

- **Scored position:** both evals score at the final prompt token (per the
  eval-dir README): multihop's `target` is absent from the prompt so
  "token preceding target" resolves to the last prompt token; association
  scores at the closing period. `target` defines the position and is not
  itself scored.
- **Token form:** rank = min over the single-token forms of
  {`word`, `" "+word`} (the space-prefixed form is usually the single
  token); when neither is single-token, fallback = first token of
  `" "+word`, counted in `n_inexact_token` (slightly understates those).
  Mirrors the repo's order-ops "min over single-token synonyms" convention.
- 93 multihop items yield 103 instances (10 items carry two intermediates);
  scored per-instance. The `poetry` eval is intentionally excluded (scores
  at a line-1 newline position, different semantics).
- Association per-instance raw ranks still favor the J-lens by ~10x
  (e.g. 'pregnant' 1564 vs 21951) — the floor is in reaching top-50, not
  in relative ordering.

## Reproducibility

```
CUDA_VISIBLE_DEVICES= python examples/jspace_lens_eval.py --evals multihop association
```

## Hypotheses

- The multihop early/mid-band advantage (+0.20/+0.13 @10) is the arc's
  first positive replication signal; its magnitude at 7B (H2, lens fitting
  now) is the next discriminating measurement.
- Association floor: capability floor vs representation-format (multi-token)
  — distinguishable by checking whether the 1.5B can *answer* the
  association items at all (behavioral baseline, cheap).

## Follow-ups

- Re-run both evals on the 7B lens when it lands (also resolves H2).
- Behavioral baseline on association items (does the model even do the task?).
- Consider remaining eval sets (multilingual, poetry, order-ops, typo) for
  the stage-4/5 writeups.

## Qualifier (added 2026-07-20)

The corpus-sensitivity check
(`2026-07-20-corpus-sensitivity-c4-1p5b.md`) shows the **@10 advantage
magnitude is fitting-corpus-dependent** (+0.18 wikitext / +0.09 C4-en at
1.5B, driven by the early band); the J-exclusivity finding (logit 0.00
early/mid on both corpora) and the @50 rates are corpus-stable. The
finding stands; the +0.18 figure is corpus-specific.

## References

- `[gurnee2026-workspace §2.1, §2.4]`;
  `kb/excerpts/gurnee2026-workspace#sec-2-1-vs-logit-lens`
- Supersedes-in-part the *interpretation* (not the data) of
  `2026-07-18-readout-scan-1p5b-first-pass.md` — its H3 is now confirmed;
  its measurements stand.
