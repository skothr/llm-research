# embedding-atlas raw data

Committed (git-LFS) `.pt` artifacts for the embedding-atlas arc — everything
the figures and the audit consume. From a clean clone:

```bash
git lfs install && git lfs pull
python examples/emb_audit_findings.py     # expect: SUMMARY: 93 PASS | 0 FAIL
python examples/emb_data_manifest.py --check   # sha256 + metadata match
python examples/emb_global_render.py      # any figure re-renders, no model load
```

## Stated deviation from ARC_PROCESS § "Raw data is a deliverable"

The true capture-roots of this arc are the published model weight matrices
themselves — `W_E` (input embeddings) and `lm_head` of
Qwen/Qwen2.5-7B-Instruct, pinned to HF revision
`a09a35458c702b33eeacc393d103063234e8bc28` — 2 x 1.09 GB, bit-reproducible by
anyone from the public snapshot. Unlike the nla-verbalizer arc's captures
(CPU-hours of forward passes, irreplaceable without re-running), committing
them would buy nothing, so they are NOT here. The files classed
"capture-root" in MANIFEST.json are the slices/statistics
`examples/emb_capture.py` cuts from those matrices in one model load
(~10 CPU-minutes to regenerate, requires the pinned revision). The
clean-clone bar still holds: audit and figures run from this directory alone.

## Files

See `MANIFEST.json` for per-file sha256, size, class, producing script,
inputs, and consumers. Working copies live in the gitignored
`.cache/emb_artifacts/`; scripts read cache-first with committed fallback
(`examples/_emb_artifacts.py`) and write only to the cache — promote with
`cp .cache/emb_artifacts/*.pt research/arcs/03_embedding-atlas/data/ &&
python examples/emb_data_manifest.py`.

## Trust note

These artifacts deserialize with `torch.load(..., weights_only=False)`
(pickle) — fine for this locally-generated data; verify sha256 against
MANIFEST.json before loading a copy you did not produce, and never normalize
that pattern for third-party `.pt` files.
