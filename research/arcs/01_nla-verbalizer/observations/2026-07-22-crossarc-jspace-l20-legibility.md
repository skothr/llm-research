# Cross-arc note: the L20 capture layer sits below the 7B J-lens legibility onset (from the jspace arc)

**Date/context:** 2026-07-22, filed at jspace arc close per its stage-6
cross-tie (`research/arcs/04_jspace/observations/2026-07-21-nla-crosstie-stage6.md`).

**Finding relevant to this arc:** the jspace arc's depth mapping of
Qwen2.5-7B (stage 3/4) shows J-lens readouts become contentful only
around **L22**; this arc's capture layer (`hidden_states[20]` = jlens
source_layer 19) sits *below* that onset — J-lens top-10 there is only
intermittently legible. [qualified 2026-09-28: see Evidence] Two implications for NLA work:

1. The AV's demonstrated ability to verbalize L20 activations is *not*
   redundant with linear token-indexed readouts at that depth — the AV
   reads content a J-lens cannot cleanly surface there. That strengthens
   the case that the AV decodes non-token-aligned structure.
2. The jspace decomposition experiment found the AV's content sits in
   the **non-J-space residual** of h_20 (component removal damages
   verbalization no more than random removal) [qualified 2026-09-28: see Evidence] — if a future AV is
   trained at L22–24 (above the onset), the same experiment would test
   whether that changes with depth.

Also inherited caveat, still open on this arc's side: the AV format-bias
audit (this arc's L2/D3 item) remains unrun; the jspace cross-tie
mitigated it with content-word filters and mismatched-pairing nulls but
flagged it as the main inflation risk on overlap metrics.

## Evidence

AUDIT lines are quoted from the jspace arc's committed transcript [`../../04_jspace/data/audit_2026-08-17.log`](../../04_jspace/data/audit_2026-08-17.log), Check G, which `research/arcs/04_jspace/scripts/jspace_audit_findings.py` derives from the committed artifact [`../../04_jspace/data/nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`](../../04_jspace/data/nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt).

- **Layer correspondence (Check G):** `[G] jlens_layer == 19` = 19; `[G] nla_hidden_state == 20` = 20.
- **Implication 1 (Check G):** `[G] metric3_nla_hit_rate` = 0.9167; `[G] metric3_jlens_hit_rate` = 0.3333. These are known-target hit rates over the 12 concept prompts (`per_prompt[].jlens_hit` and `nla_hit` in the artifact): the target appears in the AV's content words on 11 of 12, and in the content tokens of the J-lens top-50 readout on 4 of 12. This hit rate counts top-50 readouts, while the Finding speaks of the top-10, so it does not measure top-10 legibility.
- **Implication 2 (Check G):** `[G] expB_carrier_residual_mean` = 0.6005; `[G] expB_carrier_resctrl_mean` = 0.606; `[G] expB_carrier_delta_mean ~ 0` = 0.0055; `[G] expB_carrier_component_mean` = 0.1618; `[G] expB_carrier_randunit_mean` = 0.0748. Removing the J-space component leaves a verbalization with content-word Jaccard 0.6005 against the full one, and removing an equal-norm random direction leaves 0.606. The control is one Gaussian direction scaled to the component's norm (`res_ctrl = h − r` in `research/arcs/04_jspace/scripts/jspace_nla_crosstie.py`). The J-space component verbalized alone gives Jaccard 0.1618 against the full verbalization, 2.2× the 0.0748 of one verbalized random vector (`randunit`, which the stage-6 script names the floor for the component), so the component is not empty of the content the AV reports. Check G tests the delta with tolerance 0.02 (`claim_near(..., atol=0.02)` in `research/arcs/04_jspace/scripts/jspace_audit_findings.py`).
- **Per-prompt spread at L20 (recomputed, `per_prompt[].carrier_delta` of the 12 decomposition prompts):** mean 0.0055, standard deviation 0.105, range −0.208 to +0.165, standard error of the mean 0.030. The mean delta is 0.18 standard errors from zero.
- **Legibility onset:** the ~L22 onset is stated in the [stage-6 note](../../04_jspace/observations/2026-07-21-nla-crosstie-stage6.md) (Finding 2, citing the arc's stage-3 depth mapping) and in the [jspace README](../../04_jspace/README.md) § Findings. No Check G line pins the onset layer, and this note found no committed artifact that does.

## Reproducibility

The stage-6 run command (GPU capture phase, then the AV on CPU) is in the [stage-6 note](../../04_jspace/observations/2026-07-21-nla-crosstie-stage6.md) § Reproducibility. The Check G values replay from a clean clone without a model, once `git lfs pull` has fetched the committed artifact. Without the opt-in lens cache (`research/arcs/04_jspace/data/cache/`), the script's lens-integrity checks report FAIL and it exits 1; Check G does not depend on them:

```bash
python research/arcs/04_jspace/scripts/jspace_audit_findings.py
```

## Hypotheses

### H1 — The AV's content lies outside J-space only below the legibility onset, or at any depth

Two readings fit implication 2:
- **Depth-specific.** At L20 the J-space component holds mostly format and whitespace structure, as the stage-6 note speculates. Above the ~L22 onset it would carry verbalizable content.
- **Depth-general.** The AV's content lies outside the J-space component at any depth.

**Test:** this needs an AV trained at L22–24, which does not exist. Repeat stage-6 experiment B there on the same 12 decomposition prompts: for each prompt, the delta is the content-word Jaccard left after removing an equal-norm random direction minus the Jaccard left after removing the J-space component. The statistic is the mean delta and its standard error across the 12 prompts. At L20 the mean is 0.0055 with standard error 0.030 (Evidence).
- The depth-specific reading predicts a positive mean delta: removing the J-space component damages the verbalization more than the control does.
- The depth-general reading predicts a mean delta near zero, as at L20.

What the outcomes decide:
- A mean delta more than 2 standard errors above zero supports the depth-specific reading at that layer.
- Any smaller mean delta does not support it. With 12 prompts and the L20 spread, a true mean delta below about 0.06 would usually fall in this outcome, so it does not rule out a smaller depth-specific effect.

Only the first outcome is decisive. The two outcomes are mutually exclusive.

The J-lens-only variant of experiment B at L22, listed in the stage-6 follow-ups, does not use the AV and so cannot test H1.

## Follow-ups

1. The AV format-bias audit, README limitation L2 and item D3 ([#8](https://github.com/skothr/llm-research/issues/8)), has not run.
2. The ~L22 legibility onset is not pinned by any audit line (Evidence).
3. H1 needs an AV trained at L22–24. The jspace README lists one under Next paths.
4. The J-lens-only experiment B at L22 from the stage-6 follow-ups has no committed run in the jspace arc.

## References

jspace arc README synthesis §Novel contributions;
`[gurnee2026-workspace §2.3]`.

- [jspace arc README](../../04_jspace/README.md)
- [Stage-6 NLA cross-tie](../../04_jspace/observations/2026-07-21-nla-crosstie-stage6.md)
- [This arc's README](../README.md) § Limitations, L2
