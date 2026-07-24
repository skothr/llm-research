# jspace arc — data

Raw and derived `.pt`/JSON artifacts for the J-lens/J-space replication on
Qwen2.5-7B-Instruct (and the Qwen2.5-1.5B-Instruct bf16 primary/control),
tracked via Git LFS with per-file sha256 + provenance in `MANIFEST.json`
(written/verified by `examples/jspace_data_manifest.py`; `--check` is the
drift detector).

Artifact classes (see the MANIFEST for the per-file registry):

- `raw`: frozen fitting/held-out corpora (wikitext-103 + seeded C4-en),
  fitted-lens **layer subsets** (`jlens_*_layer-subset.pt` + `.config.json`
  sidecars; design Decision 4 — the full 27-layer lenses stay in the
  gitignored `cache/`, regenerable via `examples/jspace_fit_lens.py`), and
  the hand-written paper-verbatim item bank.
- `derived`: the promoted metric/scan/swap products the audit
  (`examples/jspace_audit_findings.py`) re-derives from — lens_eval,
  readout_scan, structure_scan, verbal_report, entailed_swap (+ paper-verbatim
  probes), nla_crosstie, and the issue-#26 metric-correction set
  (`paper_metric_varfrac_*` ×3, `atom_norm_bias_*`) — so checks B–M and every
  committed figure reproduce from a clean clone after `git lfs pull`.

`cache/` is a byte-identical, gitignored working mirror (plus the full
lenses); render scripts and the audit resolve `data/`-first,
`cache/`-fallback via `examples/_jspace_paths.resolve`.
