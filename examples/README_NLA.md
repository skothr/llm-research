# NLA research scripts — conventions

Scripts in this directory (`nla_*.py`) form the capture / analysis / render
pipeline for the NLA (Natural Language Autoencoder) research arc on layer 20
of Qwen2.5-7B-Instruct. The `nla_audit_findings.py` script is the regression
test that re-derives every load-bearing numerical claim from raw `.pt`
artifact files.

## torch.load(..., weights_only=False) — trust assumption

Every script here that reads a `.pt` artifact uses
`torch.load(path, weights_only=False)` to deserialize files saved by sibling
scripts in this directory.

**This is intentional and safe IN CONTEXT.** Both artifact locations (the
gitignored `.cache/nla_artifacts/` and the committed
`research/arcs/01_nla-verbalizer/data/` copies promoted from it) hold files
produced by these same scripts — no external sources — and consumed only by
this audit/research pipeline. `weights_only=True` would reject the nested
Python dicts these scripts persist (capture metadata, anchor labels, AV text
strings, etc.).

**Do not normalize the pattern for untrusted data.** If you extend this
pipeline to load `.pt` files from third-party sources (HuggingFace, public
datasets, etc.), switch to `weights_only=True` and restructure persistence
to match — the trust boundary changes.

## ARTIFACTS path resolution

Every `nla_*.py` script that reads or writes a `.pt` artifact resolves its
paths through the shared `_nla_artifacts` helper, which anchors both locations
to `Path(__file__).resolve().parent.parent` — so **CWD does not matter**, and
either of these works:

```bash
.venv/bin/python examples/nla_audit_findings.py       # from the repo root
python /path/to/repo/examples/nla_audit_findings.py   # from anywhere else
```

(The venv from the repo README's Setup section must be active, or invoked
explicitly as above — a system `python` has no `torch`.)

The helper exposes two directories and resolves **per artifact name**:

- `CACHE` — `<repo>/.cache/nla_artifacts/`, the gitignored working cache.
- `DATA` — `research/arcs/01_nla-verbalizer/data/`, the committed git-LFS copy.

Reads prefer `CACHE` and fall back to `DATA`, so a fresh local re-capture is
picked up immediately *and* a clean clone still re-renders every figure and
replays the audit. Writes always target `CACHE`; promote an artifact to `DATA`
with `nla_data_manifest.py` when committing. `warn_if_mixed_sources()` warns
when a multi-input derive script would blend a re-captured input with older
committed ones.

`nla_audit_findings.py` wraps the same `find_artifact()` in a small
`_ArtifactDir` class (so `ARTIFACTS / "name.pt"` keeps reading naturally) for
exactly this per-name resolution; a name in neither location resolves to its
non-existent `CACHE` path, so unconditional loads raise `FileNotFoundError` and
guarded `.exists()` checks stay `False`.

Six `nla_*.py` scripts — `nla_scan`, `nla_trajectory`, `nla_gen_trajectory`,
`nla_steering_direct`, `nla_roundtrip`, `nla_prompt_battery` — persist nothing
at all (they print), and so import no artifact helper.

## Models + cache

CPU bf16 paths via `llm_surgeon.probe.{load_av, load_ar, nla_verbalize,
nla_reconstruct, nla_score}`. HuggingFace checkpoints are cached by the sibling
toolkit, not by this repo: `llm_surgeon.surgery.MODEL_CACHE_DIR` defaults to
`.cache/models/` inside the *llm-surgeon* checkout, overridable with the
`LLM_SURGEON_CACHE_DIR` env var. First load of AV/AR pulls multi-GB checkpoints
from HuggingFace.

## Figures + observations

Scripts that produce visualizations write PNGs to
`research/arcs/01_nla-verbalizer/observations/figures/fig*.png` (committed
canonical artifacts). Each figure is cataloged with provenance in
`research/arcs/01_nla-verbalizer/observations/figures/INVENTORY.md`. Observation
`.md` files in `research/arcs/01_nla-verbalizer/observations/` are evidence-first
writeups citing audit-locked numbers. Repo-wide convention: session-resumption
metadata (resume checkpoints, arc summaries written at compaction time) lives
in `research/arcs/<slug>/sessions/`, kept separate from evidence — see
`research/README.md` § Conventions. **Arc 01 has no `sessions/` directory**;
of the current arcs only arc 03 (`03_embedding-atlas`) does.

## "Discriminant" naming — methodology note

The scripts (`nla_discriminant_glyph.py`, `nla_discriminant_connectivity.py`,
`nla_discriminant_stability_render.py`, `nla_hierarchical_classifier.py`,
`nla_mid_seq_native_compare.py`) compute per-category directions as:

```python
d_cat = mean(in_category_h's) − mean(out_of_category_h's)
d_cat /= ||d_cat||                     # unit-normalized
```

This is the **unscaled centroid-difference direction** (a.k.a.
mean-contrast direction or prototype-difference direction), NOT a
Fisher linear discriminant. A Fisher LDA would be `S_W⁻¹(μ₁−μ₀)` with
within-class scatter scaling. The codebase consistently uses "discriminant"
to refer to the centroid-difference vector — this is a research-code
naming choice, not a claim of Fisher-style optimal-separation properties.

**Why omitting the `S_W⁻¹` term is defensible here**: per category we
have n=2-12 captures (median 5; six categories drop to n=2 or n=3) in
a 3584-dim space, so `S_W` is rank-deficient by orders of magnitude
(rank ≤ n-1) and any LDA would require heavy regularization (shrinkage
or pseudo-inverse). The unscaled centroid difference is a reasonable
proxy in this regime, but it should not be cited as a "discriminant"
in the formal statistical sense.

Downstream quantitative results (e.g. the +0.5632 in-protocol signal in
[`2026-05-14-nla-mid-seq-native-discriminants.md`](../research/arcs/01_nla-verbalizer/observations/2026-05-14-nla-mid-seq-native-discriminants.md),
the +0.0491 cross-protocol signal in
[`2026-05-14-nla-mid-seq-vocab-atlas-null-result.md`](../research/arcs/01_nla-verbalizer/observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md))
are correct for
*what they actually compute* (mean-contrast projection). The
"discriminant" label is shorthand; readers extending or publishing
this work should call it the "mean-contrast" or "centroid-difference"
direction and reserve "discriminant" for properly-scaled methods.
