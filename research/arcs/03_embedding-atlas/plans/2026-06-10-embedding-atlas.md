# Embedding-atlas arc — session plan (2026-06-10)

Working plan the arc opened with; kept as the record of intended scope.
Current state lives in the [arc README](../README.md).

**Question:** what category/feature structure exists in the static
input-embedding table W_E of Qwen2.5-7B-Instruct (152,064 x 3584, untied
from lm_head, HF revision a09a3545), and how does it compare to arc 1's
layer-20 geometry?

**Design decisions:**

- Don't commit W_E/W_U (1.09 GB each; bit-reproducible from the pinned
  public snapshot) — commit only the slices/stats the figures and audit
  consume (~38 MB). Stated as the arc's deviation in `data/README.md`.
- `_emb_artifacts.py` is a copy of `_nla_artifacts.py` (documented
  copy-as-template idiom); a third arc triggers the shared-factory refactor.
- Battery as committed data (`emb_token_battery.py`): 70 classes / 10
  supergroups / 690 words after the CP1 expansion (user direction), plus 108
  aligned pairs in 10 kinds. Single-token constraint enforced at capture;
  bare + leading-space variants both kept where single-token; drops reported
  and audited, never silent.
- One model-loading script (`emb_capture.py`, `--tokenize-only` pre-flight);
  all analysis/figures model-free downstream of it. float32 math; padded
  rows excluded; fixed seed 20260610.
- Every analysis in raw AND mean-centered space (anisotropy guard — which
  turned out to be a null: see the global-geometry observation).

**Checkpoint working mode (user requirement):** present figures with
plain-language explanations of each measurement at CP1 (battery coverage),
CP2 (global geometry), CP3 (category/pairs/neighbors), CP4 (stretch:
layer-20 bridge); take direction between checkpoints.

**Experiment battery:** (a) global geometry — norms, anisotropy,
PCA spectrum/participation ratio, dead-row detection; (b) category
structure — within/between cosines, centroid matrix, contrast directions +
connectivity (arc-1's centroid-difference formula, no sink removal);
(c) E-vs-U per-token cosine (untied matrices); (d) neighbor probes (24
probes, raw + centered, full-vocab); (e) battery PCA map; (f) pair-difference
consistency with within-kind permutation baseline; (g, stretch) layer-20
bridge vs arc 1's committed vocab_atlas.pt — not run this session, listed as
a next path.

**Acceptance:** audit re-derives every load-bearing number from committed
data and passes from a clean clone; MANIFEST sha256-pins the bytes; figures
re-render model-free; arc-1's audit unaffected (178 PASS regression).
