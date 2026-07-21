# '的' cross-script pairing: role/block alignment, not translation equivalence

**Date / context:** 2026-07-21, post-wrap addendum. Model: Qwen2.5-7B-Instruct
@ a09a3545 (committed W_E-derived artifacts only; no new model load).
Prompted by a user challenge to finding #6's cross-script pair notation
(`' the'/'的'`): '的' is a genitive/attributive particle, so its English
functional analog should be ' of' / possessive `'s`, not ' the'.

## Finding

The user's reading is what the geometry shows. Over all 152,064 W_E rows:

- **Full cosine to '的':** ' of' **+0.626 (rank 1** among all other tokens),
  ' the' +0.519 (rank 2), ' a' +0.420, `'s` +0.375, `’s` +0.345, ' de' +0.343.
  ' the' *is* the 2nd-nearest token in the entire vocabulary, so the finding-#6
  pairing is not a data error — but ' of' is strictly closer, consistent with
  '的''s genitive use.
- **Block-ablated cosine** (drop the 21 entangled dims, cosine over the other
  3563): the ordering becomes ' of' +0.364 > `'s` +0.334 > ' the' +0.256.
  Half of the ' the'/'的' similarity is the shared glue-token block;
  `'s`/'的' similarity barely drops (0.375 → 0.334) — distributed/semantic,
  not block-mediated.
- **Calibration:** ','/'，' behaves oppositely — cosine *rises* 0.615 → 0.838
  when the block is ablated. The commas are genuine cross-script near-twins
  whose block coordinates differentiate them; ' the'/'的' are role-mates whose
  similarity largely *is* the block.
- '的''s top-10 vocab neighbors decode to ' of', ' the', ' a', 'ing', ' for',
  ' and', `'s`, '了', ' by' — English grammatical connectives plus '了' (the
  other ultra-frequent Chinese particle): a grammatical-function neighborhood.

Finding #6's pairs are therefore *role/block-loading* pairs. The README now
says so explicitly. Note `"'s "`/`" 's"` (space variants) do not exist as
tokens — Qwen's byte-level BPE is leading-space (`Ġ`) only; the clitic family
is exactly `'s` (594) and `’s` (748).

## Evidence

```
id bindings validated against artifacts for 11/11 candidates
cos to '的' (full | block-ablated), of 152064 rows:
  ' the'   +0.5192 | +0.2560
  ' of'    +0.6261 | +0.3642
  "'s"     +0.3748 | +0.3344
  '’s'     +0.3447 | +0.3293
  ' de'    +0.3433 | +0.1592
top-20 neighbor ids of '的': [315, 279, 264, 287, 369, 323, 594, 34187, ...]
```

## Reproducibility

```bash
python examples/emb_de_cosine_check.py     # needs cached emb_WE_bf16.pt
python examples/emb_audit_findings.py      # AUDIT 11 re-derives from the
                                           # committed emb_de_cosine_check.pt
```

Candidate token ids are constants cross-validated at run time against
`emb_structural_block.pt` / `emb_fullvocab_analysis.pt` (11/11); originally
resolved via the Qwen2 BPE table (llama.cpp `ggml-vocab-qwen2.gguf`, shared
by Qwen2.5).

## Hypotheses

[INTUITION: subspace ablation as a similarity decomposition — how much of a
cosine survives removing a candidate subspace separates "similar because both
load the shared high-frequency block" from "similar in distributed content".
Canonical form: compare cos(x, y) with cos(x_{S^c}, y_{S^c}) where S is the
21-dim block and S^c its complement.]

## Follow-ups

- The same decomposition over the other cross-script pairs ('。'/'.',
  '、'/',') would map which pairs are twins vs role-mates; cheap from the
  committed rows if extended to more candidates.

## References

- F-V1 / F-V3 in [2026-06-10-emb-fullvocab-sweep.md](2026-06-10-emb-fullvocab-sweep.md)
  (F-V3 already recorded '的' as the preposition handle's top out-of-battery hit).
- README finding #6; audit section 11; `data/emb_de_cosine_check.pt`.
