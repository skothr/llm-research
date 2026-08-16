# Observation: J-space workspace band is corpus-invariant; early layers are not (C4 sensitivity check, 1.5B)

> **2026-08-16 addendum — C4-redaction re-run (read first).** The C4-en
> fitting corpus this observation's lens was fit on carried third-party PII
> and was redacted 2026-07-29 (`../data/README.md`). The lens was refit and
> the whole metric suite re-run on the redacted corpus 2026-08-15/16 (commit
> `a9df3a55`); every C4-side number below is re-derived from the regenerated
> artifacts by `examples/jspace_audit_findings.py`
> (`../data/audit_2026-08-16.log`). The wikitext-side
> comparison values carry no C4 dependency and were not recomputed.
>
> **All values quoted below re-derive within audit tolerance, except two
> early-band readings that shift in the third significant figure:** L0
> varfrac on the C4 lens is **0.175** (quoted 0.174; the pin passes) and L0
> logit-kurtosis is **2.6455** (quoted 2.64). Both are in the early band this
> observation already reports as the corpus-sensitive one, and neither
> changes its sign, size or direction. Surviving unchanged: the headline L21
> varfrac peak (0.1242 on the C4 lens vs 0.1237 on the wikitext lens —
> identical to 3dp, as claimed), the depth-of-emergence medians (J 23.0 /
> logit 19.0), the L26 Spearman (0.8026, Δ 0.0023 from wikitext's 0.8003),
> multihop late J@10 0.4854 and logit@10 0.3981, J@50 overall 0.7476,
> early-band J@10 0.1165, the exact-0.00 logit early/mid rates, the floored
> association rates, and the kurtosis trough at L18 (0.8727). The overall
> @10 advantage is +0.087 on C4 (quoted +0.09).
>
> **Superseded value — fit cost.** The "8,044 s wall (~115 s/prompt)" below
> is wrong in itself: 8,044 s over 100 prompts is 80 s/prompt, not 115. The
> refit measured **3.12 h** (~112 s/prompt over 100 prompts) — which does
> match the 115 s/prompt calibration; 3.14 h including process startup, as
> recorded by `examples/jspace_rerun_queue.py`. The original figure is
> preserved below rather than edited in place.

**Date/context:** 2026-07-20. Corpus-sensitivity check from the README
deferred-directions list (prompted by an external review of the fitting
corpus, routed in by the reviewer as a blocking item before stage 5): the 1.5B lens refitted at n=100 on a seeded C4-en sample
(`allenai/c4` en/train, streaming `shuffle(seed=42, buffer_size=10000)`,
≥600-char filter, first 1000 frozen to
`data/fitting_prompts_c4en_n1000.json`, MANIFEST-registered), then the
full stage-3/4 metric suite re-run against the C4 lens
(`jlens_qwen2.5-1.5b_bf16_n100_c4en`). Fit: 8,044 s wall (~115 s/prompt,
matching calibration; one mid-run kill cleanly recovered via checkpoint
resume). Wikitext baseline artifacts verified untouched (mtimes) —
`--corpus-tag` added to the fit script so stems can never collide.

## Finding

**The mid-late "workspace" band is corpus-invariant to the resolution of
these metrics; corpus sensitivity is confined to early layers (L0–L16) —
precisely the band the paper itself flags as lens-artifact-prone.**

Corpus-stable (wikitext vs C4, re-derived from artifacts):

- **Varfrac k=25 peak: identical** — L21 = 0.124 on both lenses; the
  L19–26 profile and late kurtosis nearly coincide.
- **Depth-of-emergence medians identical** (J 23.0 / logit 19.0, all-pos).
- **Late-layer Spearman** Δ < 0.003 (L26: J 0.8003→0.8026).
- **Multihop late band** (J 0.49–0.50, logit 0.40 @10) and **J@50 overall**
  (0.74→0.75) unchanged; **logit-lens early/mid stays exactly 0.00 on both
  corpora** — the J-exclusivity core of the H3 claim is corpus-robust.
- **Association floored on both**; active atoms pin near the k cap on both.
- **Kurtosis shape** (high-early → L18 trough → weak late rise, i.e. the
  inversion vs the paper) reproduces on C4 — the stage-4 non-replication
  is not a wikitext artifact.

Corpus-sensitive (all early-band):

- **L0 varfrac doubles on C4** (0.082 → 0.174), converging to the wikitext
  profile by ~L17.
- **L0 logit-kurtosis drops** (4.65 → 2.64), same shape thereafter.
- **Multihop early-band J@10** 0.20 → 0.12 (still J-exclusive; logit 0.00),
  pulling the **overall @10 advantage from +0.18 to +0.09** (@50 stable).

## Interpretation

The measurements that carry the arc's conclusions — the workspace-band
varfrac structure (stage 4), depth-of-emergence, late-band eval rates, the
kurtosis-inversion non-replication — do not depend on the fitting corpus.
What moves is early-layer J-space occupancy and the early-band top-10 hit
rate, which tracks input-token diversity (C4's web-register vocabulary is
broader than wikitext's), consistent with early readouts reflecting input
statistics rather than workspace content
`[kb/excerpts/gurnee2026-workspace#sec-4-1-kurtosis-onset]` (the paper's
own early-layer caveat, `[gurnee2026-workspace §4.1]`). Consequence for
prior observations: the H3 headline advantage magnitude **@10** is
corpus-dependent (+0.18 wikitext / +0.09 C4); the J-exclusivity claim and
everything at @50 or in the late band is not. **Verdict: wikitext stands
in for the 7B lens; no refit needed for the arc's conclusions.**

## Evidence

Artifacts (gitignored cache): `jlens_qwen2.5-1.5b_bf16_n100_c4en.pt` (+
config sidecar recording `corpus_tag`), `readout_scan_*_c4en.pt`,
`lens_eval_*_c4en.pt`, `structure_scan_*_c4en.pt`. Frozen corpus committed:
`data/fitting_prompts_c4en_n1000.json` (sha256 in MANIFEST). All suite
runs exit 0; outputs auto-named by lens stem (`c4en` verified in every
filename); six wikitext-side artifacts byte-untouched.

## Reproducibility

```
python examples/jspace_freeze_c4_corpus.py            # deterministic, seed=42
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode bf16 --dim-batch 8 --n-prompts 100 --device cuda \
    --prompts research/arcs/04_jspace/data/fitting_prompts_c4en_n1000.json \
    --corpus-tag c4en
# then the stage-3/4 suite with --lens .../jlens_qwen2.5-1.5b_bf16_n100_c4en.pt
```

## Hypotheses

- Early-band sensitivity is an averaging effect: J_ℓ inherits the fitting
  distribution's token statistics most strongly where representations are
  closest to the embedding; by mid-depth the averaged Jacobian is dominated
  by model computation, not input register. [INTUITION — consistent with
  both corpora converging by ~L17.]
- The 7B early-band results (stage 3/4) carry the same early-band
  corpus-dependence; conclusions drawn there already avoid leaning on
  early-band magnitudes.

## Follow-ups

- None blocking. A 7B C4 refit is unjustified by these results (~16 h for
  a band the arc doesn't lean on); revisit only if a future claim depends
  on early-band magnitudes.
- Arc-close audit should add the C4 artifacts as a check group (with the
  L21-peak-identity as the headline invariance check).

## References

- `[gurnee2026-workspace §2.1, §4.1]`;
  `kb/excerpts/gurnee2026-workspace#sec-4-1-kurtosis-onset`
- Prior: `2026-07-20-jspace-structure-stage4.md`,
  `2026-07-18-intermediate-concept-evals-h3-confirmed.md` (whose @10
  headline this qualifies), README "Corpus provenance" note.
