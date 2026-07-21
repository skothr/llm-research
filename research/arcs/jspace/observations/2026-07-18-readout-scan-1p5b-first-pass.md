# Observation: first J-lens readout scan on Qwen2.5-1.5B — "cleaner early, not earlier"

**Date/context:** 2026-07-18. Stage 3 first pass. Model
`Qwen/Qwen2.5-1.5B-Instruct` (bf16, CPU inference), lens
`jlens_qwen2.5-1.5b_bf16_n100` (n=100 wikitext-103 prompts, dim_batch=8,
max_seq_len=128, skip_first=16 — see its `.config.json`). 12 held-out
wikitext-103 prompts (records 1001-1030 filter-matched, zero overlap with the
fitting corpus), 9 positions each (8 mid + last), all 27 source layers,
J-lens (`use_jacobian=True`) vs logit-lens (`use_jacobian=False`) through the
identical unembed path. Script: `examples/jspace_readout_scan.py`.

**PRELIMINARY — weak-conditions caveat up front:** small model (1.5B vs the
paper's Claude 4.5-family), possibly under-fit lens (n=100), raw-wikitext
prompts fed to an instruct model, n=12. Treat as a first calibration point,
not a refutation.

**Provenance note (added 2026-07-21):** the original run above was CPU;
the artifact was regenerated on GPU during the rich-token-capture
backfill. bf16 backend numerics flipped two rank-boundary cells vs the
tables below: last-position logit depth-of-emergence median 19.0 → 19.5,
and never-emerged J count 15 → 16 (of 108). All other values reproduced
exactly (Spearman rows to ≤1e-3). Neither cell carries a conclusion; the
audit pins the on-disk GPU values with the same note.

## Finding

Under these conditions the paper's core §2.1 claim — J-lens "uncover[s]
meaningful information in earlier layers where the logit lens produces
uninterpretable readouts" `[gurnee2026-workspace §2.1;
kb/excerpts/gurnee2026-workspace#sec-2-1-vs-logit-lens]` — is **not
reproduced by the quantitative proxies**, with one qualitative signal in its
favor:

1. **Spearman agreement with the model's final next-token logits is lower
   for the J-lens at every layer** (J−logit from −0.05 to −0.28, converging
   to ~0 by layer 26; e.g. layer 20: J 0.408 vs logit 0.548; layer 26:
   J 0.800 vs logit 0.861). Convergence at the top layers matches the
   J→I sanity expectation.
2. **Depth-of-emergence favors the logit lens:** median layer at which the
   model's final top-1 token enters the lens top-10 = 23.0 (J-lens) vs 19.0
   (logit lens) over all 108 (prompt, position) cells; the J-lens never
   surfaces it at all in 15/108 cells vs 1/108 for the logit lens.
3. **Qualitative counterpoint:** early/mid-layer J-lens readouts are
   *coherent but contentless* (whitespace/newline tokens — a "nothing on the
   mind yet" reading), where logit-lens readouts are token junk (`var`,
   `<<<<<<`, `////`). Both snap to the same semantic tokens (`Question`,
   `Describe`, `Summary`, `Based`) at roughly the same depth (~17-20).
   Summary: **J-lens readouts are cleaner early, not earlier.**

## Evidence

Per-layer mean Spearman (last position, n=12) — excerpt:

```
layer   J-lens  logit   J-logit
 4      0.099   0.377   -0.278
12      0.345   0.467   -0.122
20      0.408   0.548   -0.140
24      0.667   0.714   -0.048
26      0.800   0.861   -0.061
```

Full arrays in the derived artifact
`data/cache/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
(per-prompt `spearman_j/l`, `rank_j/l` [27x9] + summary dict). Trajectory
sample (prompt 1, final position, model top-1 in {`Does`,`What`,`Question`}):
layer 18 J-lens top-5 = {`描述`, `Describe`, `Question`, `Quotes`, `Summary`}
vs logit-lens = {`(`, `a`, `\xa0`, `Question`, `[`}.

## Reproducibility

```
CUDA_VISIBLE_DEVICES= python examples/jspace_readout_scan.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16 --device cpu \
    --lens research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
    --n-prompts 12
```

~19.4 s/prompt on CPU. `lens.apply` must be called with explicit `positions`
(unembedding every position is ~7x slower).

## Hypotheses (distinguishable)

- **H1 (under-fit lens):** n=100 J_l is too noisy for a 1.5B model; the
  averaged Jacobian's advantage needs more prompts. Test: refit at n=500-1000
  (checkpoint-resume, ~13-29 h GPU) and re-scan; if J−logit shrinks toward 0
  or flips positive in mid layers, H1 holds.
- **H2 (scale):** the workspace phenomenon (and the J-lens advantage) is
  weak or absent at 1.5B. Test: same scan on the 7B lens (fitting now,
  ETA ~15 h).
- **H3 (metric mismatch):** Spearman-to-final-logits and top-1-emergence
  measure "predicts the output," which is exactly what the *tuned* lens
  over-optimizes and what the paper says the J-lens is NOT for (surfacing
  currently-held concepts, not the eventual output;
  `[gurnee2026-workspace §2.4]`). The paper's own §2 evaluations score
  *intermediate-concept* rank (their lens-eval sets), not final-token rank.
  Test: run the shipped `data/evaluations/lens-eval-*` sets (multihop,
  association) which score intermediates — the fair test of the claim.
- **H4 (prompt distribution):** raw wikitext into an instruct model elicits
  meta-tokens (`Question`, `Answer`) rather than content concepts; neither
  lens is being asked the question the paper asks. Test: chat-templated /
  continuation-natural prompts.

H3 is the most likely explanation for the apparent contradiction with the
paper [INTUITION — the metrics we chose first are output-predictive, and the
J-lens's claimed advantage is concept-surfacing]; H3's test is also the
cheapest (CPU, minutes). H1/H2 get resolved by artifacts already scheduled.

## Follow-ups

- Run `lens-eval-multihop`/`-association` from the companion repo on the
  1.5B lens (H3, CPU).
- Re-scan on the 7B lens when it lands (H2).
- Chat-template variant of the held-out scan (H4).
- If H1 pursued: extend 1.5B fit via checkpoint resume to n=500.

## References

- `[gurnee2026-workspace §2.1, §2.4]`;
  `kb/excerpts/gurnee2026-workspace#sec-2-1-vs-logit-lens`
- `kb/notes/interpretability/j-space.md` §5 (lossy-lens critique)
- Prior observation: `2026-07-18-fit-cost-calibration.md`
