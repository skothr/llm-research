# Plateau Attractor Strength: The Hybrid Plateau Is a Real Basin [qualified 2026-09-28: see Evidence and Hypotheses H1]

**Date:** 2026-05-15
**Toolkit:** `nla_plateau_attractor_test.py`
**Inputs:** `dense_interp_near_pivot.pt` + AR `kitft/nla-qwen2.5-7b-L20-ar`
**Output:** `plateau_attractor_test.pt`
**Figures:** (none — this is a numerical-table finding)
**Private-tracker ID:** MAIN-71, part 2 of 2 (retired tracker, not resolvable — see the [ID map](../ID_MAP.md))

## What MAIN-71 tested

[MAIN-34](2026-05-15-nla-dense-interp-near-pivot.md) found a stable "Definition + Poem" hybrid plateau spanning t ∈ [0.395, 0.4450] in the linear interpolation between AR-encoded anchors A (factual/geography) and B (poetic/nature). Question: is this plateau a **true attractor basin** (AR re-encoding stays in the basin) or just a transit zone (AR re-encoding drifts toward an anchor)?

Procedure: take plateau-mid h (from dense interp step t=0.420), run its AV text through AR to produce h_pred, compare h_pred to original h and to anchors via cosine. Baseline references: same round-trip for h_A (t=0.000) and h_B (t=1.000).

## Results

| target | ||h_orig|| | ||h_pred|| | cos(h_pred, original) | cos(h_pred, A) | cos(h_pred, B) | cos(h_pred, plateau) |
|---|---|---|---|---|---|---|
| **plateau t=0.420** | 60.79 | 65.68 | **+0.8995** | +0.8386 | +0.8154 | +0.8995 |
| anchor A t=0.000 | 65.73 | 64.65 | +0.8900 | +0.8900 | +0.6416 | +0.8521 |
| anchor B t=1.000 | 66.30 | 65.78 | +0.8989 | +0.6431 | +0.8989 | +0.8151 |

## Interpretation

**Plateau round-trip cosine of +0.8995 passes the strong-attractor threshold (>+0.85).** The plateau-AV text, when AR-encoded back to h-space, lands closer to the original plateau h (+0.8995) than to either anchor (+0.8386 to A, +0.8154 to B). The plateau is a basin, not a transit zone. [qualified 2026-09-28: see Evidence and Hypotheses H1]

But the picture has a subtler layer worth flagging:

**All three round-trip cosines are ~+0.89** — AR self-consistency is approximately uniform across the three test points. The relevant signal isn't the round-trip cosine alone, but the **margin** between cos(h_pred, original) and cos(h_pred, other_anchors):

| target | self-consistency | margin to anchor-rivals |
|---|---|---|
| anchor A | +0.890 vs +0.642 (B) | **+0.248** |
| anchor B | +0.899 vs +0.643 (A) | **+0.256** |
| **plateau** | +0.900 vs +0.839 (A) | **+0.061** |

The plateau's margin is much smaller than the anchors' margins. This is geometrically consistent — the plateau is in between the anchors, so an AR reconstruction in the plateau region is naturally somewhat close to both anchors. But it also means the plateau basin is **narrower** than the anchor basins: a small perturbation in h-space might tip the AR re-encoding out of the plateau and into one of the neighboring basins.

**Norm note:** plateau ||h_orig||=60.79 (the magnitude dip we saw in fig37) but ||h_pred||=65.68 — AR reconstruction restored the magnitude to typical anchor-level (~65-66). The AR value head appears to project to canonical magnitude regardless of the input AV text's source. This means the plateau basin is direction-coupled but not magnitude-coupled to its specific dip-point geometry. [qualified 2026-09-28: see Evidence]

## Evidence

AUDIT lines are quoted from the committed transcript [`../data/audit_2026-08-17.log`](../data/audit_2026-08-17.log) (`research/arcs/01_nla-verbalizer/scripts/nla_audit_findings.py`). "Recomputed" values are read from `results[]` in [`../data/plateau_attractor_test.pt`](../data/plateau_attractor_test.pt) (keys `norm_orig`, `norm_pred`, `cosine_round_trip`, `cos_to_anchor_A`, `cos_to_anchor_B`, `cos_to_plateau`, and the tensors `h_orig`, `h_pred`).

- **Plateau row (AUDIT 19):** `attractor test target count` = 3; `plateau round-trip cosine (attractor threshold +0.85)` = 0.8995; `plateau h_pred → anchor A drift` = 0.8386; `plateau h_pred → anchor B drift` = 0.8154.
- **Plateau margins (AUDIT 19):** `plateau h_pred closer to plateau than to A (margin > 0)` = 0.060914039611816406; `plateau h_pred closer to plateau than to B (margin > 0)` = 0.08408069610595703. The assertions are `> 0` bounds. The exact margins are the `actual` values the log prints.
- **Anchor rows (recomputed):** anchor A: `norm_orig` 65.73, `norm_pred` 64.65, `cosine_round_trip` 0.8900, `cos_to_anchor_B` 0.6416, `cos_to_plateau` 0.8521. Anchor B: 66.30, 65.78, 0.8989, `cos_to_anchor_A` 0.6431, `cos_to_plateau` 0.8151. Plateau: `norm_orig` 60.79, `norm_pred` 65.68. The cosine of the stored `h_orig` and `h_pred` tensors reproduces each `cosine_round_trip`. No AUDIT line checks the anchor rows.
- **Margin table (recomputed from the same values):** anchor A 0.8900 − 0.6416 = 0.2484; anchor B 0.8989 − 0.6431 = 0.2558; plateau 0.8995 − 0.8386 = 0.0609.
- **Inputs (recomputed):** the plateau `h_orig` equals `h_t` at t=0.4200 in [`../data/dense_interp_near_pivot.pt`](../data/dense_interp_near_pivot.pt), the anchor A `h_orig` equals that file's `h_A`, and the anchor B `h_orig` equals its `h_B`.
- **Position on the line (recomputed from `h_A`, `h_B` and `steps[].h_t` in `dense_interp_near_pivot.pt`):** cos(`h_A`, `h_B`) = 0.6905; at t=0.42, cos(`h_t`, `h_A`) = 0.9435 and cos(`h_t`, `h_B`) = 0.8912. `||h_t||` along the line is smallest at t = 0.486, so the plateau's 60.79 follows from where it lies on the line ([dense interpolation](2026-05-15-nla-dense-interp-near-pivot.md) Evidence).
- **Round-trip baseline (AUDIT 20):** `aggregate captures with cosine` = 113; `aggregate mean cosine` = 0.8679; `aggregate min cosine` = 0.7171. The +0.85 threshold is below the mean round-trip cosine of the 113 ordinary captures. A comment in `research/arcs/01_nla-verbalizer/scripts/nla_plateau_attractor_test.py` states that its thresholds are ad hoc and that the verdict is not a statistically tested basin claim.

## Stronger statement we can make now

Combining [MAIN-25](2026-05-13-nla-interpolation-flipbook.md), MAIN-34, [MAIN-44](2026-05-14-nla-mid-seq-vocab-atlas-null-result.md)/[70](2026-05-14-nla-mid-seq-native-discriminants.md), [MAIN-48](2026-05-14-nla-concept-arithmetic-atlas.md), and this work:

**Layer-20 h-space has discrete attractor basins separated by sharp boundaries. Basins include both the named vocab categories AND hybrid combinations not in the original atlas (like "Definition + Poem"). AR re-encoding of a basin's AV-text returns h to the basin's directional region. The basin structure is direction-coupled, not magnitude-coupled. Linear interpolation between two basin-residing h's traverses one or more intermediate basins with sharp boundaries between them.** [qualified 2026-09-28: see Evidence and Hypotheses H1]

*Scope (matches README F1/L9): hold this as a **working hypothesis**, not a settled property of layer 20. It rests on a single anchor pair (factual/geography ↔ poetic/nature) and a plateau margin of only +0.061 over the nearest single-anchor — narrower than the ~+0.25 inter-anchor margins (see the margin note above). The framing the evidence supports is "basin candidate at this one location" until other anchor pairs and layers replicate (D5/D6).*

This is the strongest synthesis the arc has produced. It explains:
- Why category arithmetic fails for specific identities (MAIN-48): basins, not algebraic offsets
- Why discriminants are protocol-coupled (MAIN-44/70): each protocol's captures land in protocol-specific subsets of basins
- Why interpolation produces stepwise flips (MAIN-25): basin boundaries are sharp
- Why dense sampling reveals plateaus (MAIN-34): basins have non-zero volume in h-space

## Reproducibility

```bash
# AR round trip of the three targets (loads the AR on CPU; reads h_A, h_B and the
# plateau step's h_t, plus av_text, from dense_interp_near_pivot.pt; writes .cache/nla_artifacts/plateau_attractor_test.pt)
python research/arcs/01_nla-verbalizer/scripts/nla_plateau_attractor_test.py

# Model-free check (AUDIT 19); reads the cache copy first, else the committed ../data/ copy
python research/arcs/01_nla-verbalizer/scripts/nla_audit_findings.py
```

The script writes to the gitignored working cache `.cache/nla_artifacts/plateau_attractor_test.pt`; the committed copy is [`../data/plateau_attractor_test.pt`](../data/plateau_attractor_test.pt). Every read, including the script's read of `dense_interp_near_pivot.pt` and the audit's, takes the cache copy when one exists and otherwise the committed copy in `../data/` (`research/arcs/01_nla-verbalizer/scripts/_nla_artifacts.py`). A clean clone therefore runs and audits against the committed data, and after a re-run the audit checks the new cache file.

## Hypotheses

### H1 — The positive self-margin marks a basin, or is a property of every round trip

The basin claim rests on the plateau's AR round trip landing closer to the plateau `h` than to either anchor. Two readings fit that:
- **Basin.** The margin is specific to basin points. A point outside a basin would drift toward an anchor.
- **Generic round trip.** Every point's round-trip error is about the same size, independent of basin membership (mean `cosine_round_trip` 0.8679 over ordinary captures, AUDIT 20), so a point's margin is set by its position on the line.

The three committed targets cannot separate these readings: the plateau is the only non-anchor point tested.

**Test:** run the same AV → AR round trip on non-plateau points of the same line: t=0.25, 0.5 and 0.75. The raw self-margin, `cos(h_pred, h_t) − max(cos(h_pred, h_A), cos(h_pred, h_B))`, depends on how close `h_t` is to an anchor: t=0.25 lies much closer to `h_A` than the plateau does, so even a generic round trip leaves a small margin there, and round-trip error alone can make the raw margin negative. The statistic is therefore the residual against each anchor: the observed margin to that anchor minus the margin expected from position. Write `h_pred` as a part along `h_orig` plus an error orthogonal to it. If that error is uncorrelated with the anchors, cos(`h_pred`, anchor) ≈ `cosine_round_trip` × cos(`h_orig`, anchor), so the expected margin is `cosine_round_trip` × (1 − cos(`h_orig`, anchor)). Compute the same residual for the three committed targets to give the reference range.
- The generic reading predicts residuals at the new points within the range of the committed targets' residuals.
- The basin reading predicts that at least one non-plateau point drifts toward an anchor beyond what its position predicts, giving a residual below that range.

What the outcomes decide:
- No new residual below the committed range: the self-margin does not distinguish the plateau from other points on the line, and it cannot serve as basin evidence.
- A new residual below the committed range: the plateau differs from that point. This is consistent with the basin reading but does not establish it, since points on one line cannot show basin width.

Each point gives one round trip (greedy AV decode, then the AR), and the committed range comes from four values (two for the plateau, one per anchor), so the cut is coarse. More points along the line would tighten it.

## Follow-ups this opens

- **Map more hybrid basins.** Try anchor pairs across different content domains: code↔nature, math↔emotion, factual↔refusal. Each pair likely has its own intermediate basin(s). Atlas-of-basins as a viz primitive.
- **Test margin scaling.** Plateau margin to anchors was 0.061 here. If the plateau h is perturbed by Gaussian noise of varying magnitude, at what perturbation level does AR re-encoding leave the basin? This characterizes basin width quantitatively.
- **Probe basin self-consistency at higher resolution.** Multiple round-trips: h → AV → AR → h' → AV' → AR → h''. Does this iterate converge to a fixed point (stable attractor with well-defined center) or wander?
- **Open items from the Evidence (2026-09-28):**
  - The H1 test has not run.
  - Position-expected margins: see [#123](https://github.com/skothr/llm-research/issues/123).
  - AUDIT 19 checks only the plateau row. The anchor rows and the anchor margins (+0.248, +0.256) rest on the recomputation in Evidence.
  - The factual → hybrid boundary in t ∈ [0.25, 0.395] is still unsampled (see the [dense interpolation](2026-05-15-nla-dense-interp-near-pivot.md) follow-ups).
  - Replication on other anchor pairs is README item D6 ([#5](https://github.com/skothr/llm-research/issues/5)), and the AV format-bias audit it depends on is D3 ([#8](https://github.com/skothr/llm-research/issues/8)).

## References

- [Dense interpolation near the pivot](2026-05-15-nla-dense-interp-near-pivot.md): the source of the plateau `h` at t=0.420 and of the anchors.
- [Interpolation flipbook](2026-05-13-nla-interpolation-flipbook.md): the AR-encoded anchors A and B.
- [Aggregate faithfulness](2026-05-13-nla-aggregate-faithfulness-8-prompts.md): the round-trip baseline AUDIT 20 checks.
- [Arc README](../README.md) § F1 and limitation L9: the scope this result is held to.
