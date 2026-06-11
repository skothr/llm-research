# Embedding atlas — semantic structure of Qwen2.5-7B's input-embedding table

**Research question:** What semantic/functional category structure exists
*already* in the static input-embedding table (W_E, "layer 0") of
Qwen2.5-7B-Instruct — do word groups related by topic, function, morphology,
or connotation share directions that could act as "handles" for downstream
layers — and how does that structure compare to the layer-20 geometry found
in the [nla-verbalizer arc](../01_nla-verbalizer/README.md)?

**Status:** active DEEP arc (user direction 2026-06-11), started 2026-06-10.
Phase 1 (battery protocol probes) + phase 2 (full-vocabulary sweep, 149,706
alive rows) landed; four observations; audit at **61 PASS / 0 FAIL**. Phase 3
(structural tracing through Q/K/V, attention/RoPE bands, and FFN, layer by
layer) is next and ABSORBS the formerly separate rope-vis arc
([plan](plans/2026-06-10-rope-vis.md)); pre-registered predictions in
[plans/2026-06-11-predictions.md](plans/2026-06-11-predictions.md).

## Research direction

Arc 3 opened from the user's (Michael Lannum) direction, 2026-06-10 session:

> *"explore the complete set of token embeddings for correlations and
> interesting connections between groups of words (related in one way or
> another — like we did with countries [...]) Words with similar or related
> function, use, topic/connotation, etc. may have shared features that act
> as 'handles' for the model layers to execute that function hidden in our
> construction of language."*

Scope decisions made at session start: two arcs (embeddings first, RoPE
second) rather than one; iterate with figures + plain-language measurement
explanations at checkpoints; the user expanded the battery at CP1 ("add more
word groups first") — person/royal/religion/abstract/landscape/instrument/
science/tech/language classes and the gender/antonym/past/capital_of/lang_of
pair kinds were added in response. Implementation: Claude Code (Fable 5).

## Findings so far (held as working hypotheses)

Evidence and scope qualifications live in the observation files; one-line
versions:

1. **Near-isotropy null.** Random-pair cosine +0.0097; PC1 explains 1.21%;
   participation ratio ~1003/3584. The classic anisotropic-cone correction
   is not load-bearing for this table.
   ([global-geometry](observations/2026-06-10-emb-global-geometry.md))
2. **Input/output embedding orthogonality.** cos(E_i, U_i) ~ 0 for every
   token (mean +0.0017) while U-space carries its own category structure —
   the untied matrices use unrelated coordinate systems.
   ([global-geometry](observations/2026-06-10-emb-global-geometry.md), F-G4)
3. **Category-coherence hierarchy.** Paradigm sets (digits +0.428, months
   +0.416, weekdays +0.400) >> function words (+0.24-0.29) > topics
   (+0.08-0.18) > connotation classes (valence +0.045, register +0.052).
   ([category-structure](observations/2026-06-10-emb-category-structure.md))
4. **Cross-script neighbors.** 法国/巴黎 rank among ' France'/' Paris' top
   neighbors — multilingual alignment exists in the raw lookup table.
   ([category-structure](observations/2026-06-10-emb-category-structure.md), F-C3)
5. **Relations are class-offsets with a thin paired residue.** All 11 pair
   kinds beat a within-kind permutation baseline, but the baseline absorbs
   most of the direction; pair-specific margins are +0.02-0.05, largest for
   morphology.
   ([pair-directions](observations/2026-06-10-emb-pair-directions.md))
6. **Full-population (all 149,706 alive rows): exactly one entangled
   dimension block** — 21 correlated dims (|r|>0.3), cross-script
   (','/'，', ' the'/'的'), head-frequency-loaded (first token-id decile
   1.5x energy floor; Spearman -0.206 vs -0.003 control); outside it,
   dimensions are near-independent (|r| mean 0.021, kurtosis median 0.32).
   Handles are precise at vocab scale (negative -> ' shitty'/' nasty';
   code -> 'namespace') with conservative recall; the kNN graph yields
   crisp islands (names, countries, code syntax, a cross-lingual time
   community) over one giant component.
   ([fullvocab-sweep](observations/2026-06-10-emb-fullvocab-sweep.md))

## Limitations

- **L1. Single model, single revision.** Everything is Qwen2.5-7B-Instruct
  @ a09a3545. The isotropy null especially needs a second model before any
  "modern models are like this" reading (cheap: TinyLlama, cached locally).
- **L2. Curated battery, prototypical members.** 690 hand-picked words;
  gaps quantify these anchors, not the vocabulary; single-token attrition
  (25 drops) skews multilingual coverage toward de/es/zh.
- **L3. Dead rows in global stats.** 1,959 near-zero rows (1.3%) are
  included in mu/covariance/random sampling; estimated effect < 0.002 on
  headline cosines but unverified — re-lock excluding them is a follow-up.
- **L4. The audit is arithmetic-consistency only.** 44 PASS means the
  observation numbers match the committed artifacts — not that the capture
  protocol, thresholds (MIN_CLASS_N=5, near-zero 1e-3, primary-variant
  policy), or interpretations are right.
- **L5. bf16 source precision** bounds all cosines at ~1e-3; the rank-~1700
  spectrum cliff may be a quantization artifact (open question H2 in the
  global-geometry observation).

## Possible next paths

- **Held-out projection test** of the "handle" framing: project unseen pairs
  (' Stockholm' - ' Sweden') onto the capital_of direction (pair-directions
  H2) — cheap and decisive.
- **Layer-20 bridge**: same battery subset through arc 1's committed
  `vocab_atlas.pt`; do connotation classes (weak at L0) become strong at
  L20? (category-structure H2; script `emb_layer20_bridge.py`, not yet
  written.)
- **Cross-model isotropy check** on TinyLlama-1.1B (global-geometry H1).
- **Dead-row-excluded re-lock** (L3).
- **Multilingual alignment subspace** from exonym pairs (category-structure
  H3).
- **structural tracing phase** (next; absorbs the former rope-vis arc):
  Q/K/V capture, RoPE-band decomposition, FFN persistence of the 21-dim
  block, layer-by-layer trace against the pre-registered predictions.

## Reproducing

```bash
git lfs install && git lfs pull
python examples/emb_audit_findings.py        # SUMMARY: 61 PASS | 0 FAIL
python examples/emb_data_manifest.py --check # 9 files, sha256 match
python examples/emb_global_render.py         # figures re-render model-free
# full re-capture (needs the pinned model locally, ~10 CPU-min):
python examples/emb_capture.py --tokenize-only   # battery coverage pre-flight
python examples/emb_capture.py
```

## File map

```
research/arcs/03_embedding-atlas/
  README.md                                   # this file
  observations/
    2026-06-10-emb-global-geometry.md         # isotropy null, dead rows, PCA, E-vs-U
    2026-06-10-emb-category-structure.md      # coherence hierarchy, connectivity, neighbors
    2026-06-10-emb-pair-directions.md         # relation-direction consistency
    figures/ (fig1-fig10 + INVENTORY.md)
  data/ (6 .pt + MANIFEST.json + README.md)   # git-LFS, ~38 MB
```

Scripts (all under `examples/`): `emb_token_battery.py` (battery as data),
`emb_capture.py` (single model-loading step), `emb_category_stats.py` /
`emb_pair_directions.py` (derives), `emb_*_render.py` + `emb_neighbors_report.py`
(figures/report), `emb_audit_findings.py` (audit), `emb_data_manifest.py`
(manifest), `_emb_artifacts.py` (path resolver).
