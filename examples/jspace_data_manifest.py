"""Generate / verify the raw-dataset manifest for the jspace research arc.

The arc's raw artifacts live (committed, git-LFS) under
`research/arcs/04_jspace/data/`: the frozen fitting corpus, the fitted `.pt`
lens tensors and their `.config.json` provenance sidecars, and derived
metric/table artifacts. This script writes a checksummed `MANIFEST.json`
next to them that records, per file: sha256, size, whether it is a `raw`
artifact (external data or a model-dependent capture) or a `derived` one
(regenerable from other artifacts by a committed script), the producing
script/command, its inputs, what model it needs, its data provenance, and
who consumes it.

Only the top-level deliverables are covered. The `cache/` subdirectory
(`jspace_fit_lens.py --out-dir` default) holds the full fitted lenses and is
EXCLUDED — the non-recursive globs never descend into it. Three of those
lenses are LFS-committed but excluded from default pulls; the two 1.5B nf4
lenses are not committed (issue #47). See `_CACHE_LFS` / `_CACHE_UNCOMMITTED`.

Two modes — the mode is always explicit, so no invocation rewrites the
manifest by accident (a bare run, or `--help`, prints usage and exits):
    python examples/jspace_data_manifest.py --write    # (re)write MANIFEST.json
    python examples/jspace_data_manifest.py --check    # verify, exit 1 on drift

The `--check` mode is the drift detector: it recomputes every sha256, AND
re-derives each registered file's provenance fields from META, AND re-derives
the top-level `description` / `jlens_pin` / `trust_note` literals from the
writer's own constants — comparing all three against the committed manifest,
so silent corruption, a re-fit that wasn't re-committed, a missing/extra file,
a META edit (reclassify, corrected inputs/consumers), OR an edit to the
top-level prose that was never regenerated into MANIFEST.json all fail.

Unlike the writer, which fills a fixed registry, this arc's lens set grows as
sign-off gates open (which layers are committed at top level vs left in
`cache/` is an open design decision). So the writer does NOT hard-fail on
drift between META and disk: a known file still downloading/fitting is
warned-and-skipped, and a promoted deliverable with no META entry yet is
recorded as `unregistered`
(sha256 captured, provenance pending) and warned — a later session fills in
its META entry. See `research/ARC_PROCESS.md` § "Raw data is a deliverable".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"

# How the full fitted lenses under data/cache/ are (or are not) committed.
# Three are git-LFS-committed but excluded from default pulls (fetch with
# `git lfs pull --include="research/arcs/04_jspace/data/cache/**" --exclude=""`);
# the two 1.5B nf4 lenses were never committed (issue #47).
#
# Only the `.pt` tensors are LFS-routed: `.gitattributes` matches
# `research/**/data/cache/*.pt`, so `git check-attr filter` reports `lfs` for
# them and `unspecified` for their `.config.json` sidecars. `.lfsconfig`'s
# fetchexclude suppresses LFS objects only, so the sidecars are ordinary blobs
# that arrive in full in every default clone — they must not carry _CACHE_LFS.
_CACHE_LFS = "data/cache/, LFS-committed; excluded from default pulls"
_CACHE_PLAIN = "data/cache/, committed as a plain blob; present in every default clone"
_CACHE_UNCOMMITTED = "data/cache/, not committed — issue #47"

# Full fitted lenses named as inputs (issue #87). A lens file name does not
# identify its fit. The three LFS-committed cache lenses are refits: the c4en
# lens on the redacted corpus, the two wikitext lenses because the cache was
# empty. All three ran in one refit queue (the 1.5B fits ran
# 2026-07-29; the 7B fit's last segment ran 2026-08-15 and completed
# 2026-08-16). Every committed July artifact
# that names one of them was produced by an earlier fit of the same name,
# which was never committed in full. No committed artifact derives from the July c4en fit. Inputs name the
# generation through a _REFIT or _JULY string. The cache files carry no
# MANIFEST entry, so --check does not verify the sha256 values quoted here;
# each equals the oid in the file's LFS pointer.
_L15 = f"jlens_qwen2.5-1.5b_bf16_n100.pt ({_CACHE_LFS})"
_L7B = f"jlens_qwen2.5-7b_nf4_n100.pt ({_CACHE_LFS})"
_L15C4 = f"jlens_qwen2.5-1.5b_bf16_n100_c4en.pt ({_CACHE_LFS})"
_L15_REFIT = (
    f"{_L15}; the refit completed 2026-07-29 and committed 2026-08-16, sha256 "
    "db54f1f0199c238e8efc4a950785af7ad635dd64eb1976c3f61c895f4ea02abe "
    "(not verified by --check, which skips cache files)"
)
_L15C4_REFIT = (
    f"{_L15C4}; the refit completed 2026-07-29 on the redacted C4 corpus "
    "and committed 2026-08-16, sha256 "
    "661bb70494acd215d1e2da58b87ebeb440af0ae749953a9f8b4a99a30fdab5a9 "
    "(not verified by --check, which skips cache files)"
)
_L7B_REFIT = (
    f"{_L7B}; the 2026-08-16 refit, sha256 "
    "4704ee3b2cd75b35cf83fb288c1bba5d9db3daf7ed3e8e105f6c1b552f1c77db "
    "(not verified by --check, which skips cache files)"
)
_L15_JULY = (
    "jlens_qwen2.5-1.5b_bf16_n100.pt, the July pre-refit lens (fit completed "
    "2026-07-18); never committed in full; the cache file of the same name "
    "is the 2026-07-29 refit; its only committed part is the layer subset "
    "jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt; the entry's producing_command names the --lens path that now "
    "holds the refit, so it does not reproduce the July artifact"
)
_L7B_JULY = (
    "jlens_qwen2.5-7b_nf4_n100.pt, the July pre-refit lens (fit completed "
    "2026-07-20); never committed in full; the cache file of the same name "
    "is the 2026-08-16 refit; its only committed part is the layer subset "
    "jlens_qwen2.5-7b_nf4_n100_layer-subset.pt; the entry's producing_command names the --lens path that now "
    "holds the refit, so it does not reproduce the July artifact"
)
_L15N4 = f"jlens_qwen2.5-1.5b_nf4_n100.pt ({_CACHE_UNCOMMITTED})"
_L15N5 = f"jlens_qwen2.5-1.5b_nf4_n500.pt ({_CACHE_UNCOMMITTED})"

# Per-artifact provenance. `requires_model` values: none | qwen-7b-nf4 |
# qwen-1.5b-bf16 (or whatever the lens fit used). Seeded with just the frozen
# fitting corpus; lens `.pt` + `.config.json` sidecars and derived metric
# tables get added here (with class raw/derived) as sign-off gates open — a
# promoted deliverable absent from META is recorded `unregistered` until then.
META: dict[str, dict[str, Any]] = {
    # ---- raw: external data / model-dependent captures ---------------------
    "fitting_prompts_wikitext103_n1000.json": {
        "class": "raw",
        "producing_script": "examples/jspace_fit_lens.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "Salesforce/wikitext wikitext-103-raw-v1 train, first 1000 records "
            ">= 600 chars, frozen by examples/jspace_fit_lens.py corpus step"
        ),
        "consumers": ["jlens_*.pt (frozen fitting corpus)"],
    },
    "fitting_prompts_c4en_n1000.json": {
        "class": "raw",
        "producing_script": "examples/jspace_freeze_c4_corpus.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "allenai/c4 config en, split train, streamed with "
            "shuffle(seed=42, buffer_size=10000); first 1000 records with "
            "len(text.strip()) >= 600 in post-shuffle order. Deterministic "
            "given the seed. Alternative fitting corpus for the "
            "corpus-sensitivity check (broader web-text register than the "
            "wikitext-103 default); frozen by examples/jspace_freeze_c4_corpus.py"
        ),
        "consumers": ["jlens_*_c4en.pt (corpus-sensitivity fitting corpus)"],
    },
    "heldout_prompts_wikitext103_n30.json": {
        "class": "raw",
        "producing_script": "examples/jspace_readout_scan.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "Salesforce/wikitext wikitext-103-raw-v1 train, records 1001-1030 "
            "under the same >=600-char filter as the fitting corpus; verified "
            "zero overlap with fitting_prompts_wikitext103_n1000.json. NOTE: 30 "
            "CONSECUTIVE records -> topically clustered; the diversified C4 "
            "held-out set below is the robustness control."
        ),
        "consumers": ["readout_scan_*.pt (held-out evaluation prompts)"],
    },
    "heldout_prompts_c4en_n30.json": {
        "class": "raw",
        "producing_script": "examples/jspace_freeze_c4_corpus.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "allenai/c4 config en, split train, streamed with "
            "shuffle(seed=42, buffer_size=10000); skip first 1000 accepted "
            "(the n=1000 fitting slice), then records 1001-1030 with "
            "len(text.strip()) >= 600 in post-shuffle order; deduped against "
            "fitting_prompts_c4en_n1000.json; verified zero overlap with it and "
            "with heldout_prompts_wikitext103_n30.json. Diversified (topically "
            "un-clustered) held-out control for the held-out-sample-robustness "
            "check; frozen by examples/jspace_freeze_c4_corpus.py --offset 1000."
        ),
        "consumers": [
            "readout_scan_*_heldoutc4en.pt / structure_scan_*_heldoutc4en.pt "
            "(diversified held-out evaluation prompts)"
        ],
    },
    "paperverbatim_items_n3.json": {
        "class": "raw",
        "producing_script": "examples/jspace_entailed_swap.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "Hand-written 3-item bank for the 2026-07-22 verbatim-prompt / "
            "cue-redundancy probe (stage-5.2 observation addendum): the "
            "paper's exact 'animal that spins webs' prompt "
            "[gurnee2026-workspace sec 3.3] plus two minimal-cue items. "
            "Consumed via jspace_entailed_swap.py --items-json."
        ),
        "consumers": [
            "entailed_paperverbatim_{chat,plain}_{auto,all}_L19_7b.pt "
            "(verbatim-prompt probe artifacts)"
        ],
    },
    # ---- lens artifacts (reduced layer subset, design Decision 4) ----------
    # Layers {0,5,10,15,20,25,26} of each full fitted lens, promoted to git-LFS
    # by examples/jspace_promote_lens_subset.py from the July fits. Those
    # full 27-layer sets were never committed; the cache files of the same
    # names are later refits (see _L15_JULY / _L7B_JULY above). The
    # .config.json sidecars are top-level *.json deliverables too, so each gets
    # its own META entry.
    "jlens_qwen2.5-7b_nf4_n100_layer-subset.pt": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": [_L7B_JULY],
        "requires_model": "qwen-7b-nf4",
        "provenance": (
            "Layers {0,5,10,15,20,25,26} of the full 27-layer J-lens fitted by "
            "examples/jspace_fit_lens.py on the frozen wikitext corpus (n=100, "
            "jlens defaults); 16.26 h GPU fit completed 2026-07-20. Reduced "
            "subset promoted per design-plan Decision 4 (trailing layer 27 "
            "clamped to the last valid index 26). The full July set was never "
            "committed (see inputs)."
        ),
        "consumers": ["clean-clone lens inspection at representative depths"],
    },
    "jlens_qwen2.5-7b_nf4_n100_layer-subset.config.json": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": [
            "jlens_qwen2.5-7b_nf4_n100.config.json, the July fit's sidecar (never "
            "committed; its fields survive in this subset sidecar); superseded "
            "in data/cache/ by the refit's sidecar of the same name "
            f"({_CACHE_PLAIN}); see the jlens_qwen2.5-7b_nf4_n100_layer-subset.pt "
            "entry's inputs and data/README.md (Decision 4 and its 2026-08-16 "
            "amendment)"
        ],
        "requires_model": "qwen-7b-nf4",
        "provenance": (
            "Provenance sidecar for jlens_qwen2.5-7b_nf4_n100_layer-subset.pt: "
            "the July fit's sidecar plus subset_layers + full_set_location. "
            "Its wall_seconds (58543.6) is the July fit's; the cache sidecar's "
            "is the refit's (41333.8). "
            "full_set_location hand-updated for issue #87; re-running the "
            "script writes its generic text (issue #92)."
        ),
        "consumers": ["jlens_qwen2.5-7b_nf4_n100_layer-subset.pt (sidecar)"],
    },
    "jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": [_L15_JULY],
        "requires_model": "qwen-1.5b-bf16",
        "provenance": (
            "Layers {0,5,10,15,20,25,26} of the full 27-layer J-lens fitted by "
            "examples/jspace_fit_lens.py on the frozen wikitext corpus (n=100, "
            "jlens defaults); 3 h GPU fit completed 2026-07-18. Reduced subset "
            "promoted per design-plan Decision 4 (trailing layer 27 clamped to "
            "the last valid index 26). The full July set was never committed "
            "(see inputs)."
        ),
        "consumers": ["clean-clone lens inspection at representative depths"],
    },
    "jlens_qwen2.5-1.5b_bf16_n100_layer-subset.config.json": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": [
            "jlens_qwen2.5-1.5b_bf16_n100.config.json, the July fit's sidecar (never "
            "committed; its fields survive in this subset sidecar); superseded "
            "in data/cache/ by the refit's sidecar of the same name "
            f"({_CACHE_PLAIN}); see the jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt "
            "entry's inputs and data/README.md (Decision 4 and its 2026-08-16 "
            "amendment)"
        ],
        "requires_model": "qwen-1.5b-bf16",
        "provenance": (
            "Provenance sidecar for jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt: "
            "the July fit's sidecar plus subset_layers + full_set_location. "
            "Its wall_seconds (10764.8) is the July fit's; the cache sidecar's "
            "is the refit's (7769.1). "
            "full_set_location hand-updated for issue #87; re-running the "
            "script writes its generic text (issue #92)."
        ),
        "consumers": ["jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt (sidecar)"],
    },
    # ---- derived metric/scan/swap artifacts (stage-7 promotion) ------------
    # The small derived artifacts the audit (examples/jspace_audit_findings.py)
    # re-derives from, promoted out of data/cache/ into data/ for clean-clone
    # auditability. Built compactly below via _derived() and merged into META
    # (each entry's lens input names its fit generation, issue #87).
}


def _derived(
    script: str,
    inputs: list[str],
    model: str,
    provenance: str,
    consumers: list[str],
    args: str | None = None,
) -> dict[str, Any]:
    return {
        "class": "derived",
        "producing_script": script,
        "producing_args": args,
        "inputs": inputs,
        "requires_model": model,
        "provenance": provenance,
        "consumers": consumers,
    }


_HW = "heldout_prompts_wikitext103_n30.json"
_HC4 = "heldout_prompts_c4en_n30.json"
_EVAL = "examples/jspace_lens_eval.py"
_READ = "examples/jspace_readout_scan.py"
_STRUCT = "examples/jspace_structure_scan.py"
_VR = "examples/jspace_verbal_report.py"
_ENT = "examples/jspace_entailed_swap.py"
_XTIE = "examples/jspace_nla_crosstie.py"

_DERIVED: dict[str, dict[str, Any]] = {
    # -- lens_eval x4 (intermediate-concept top-k readout rates per depth band) --
    "lens_eval_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _EVAL,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Intermediate-concept eval (multihop + association) top-k readout "
        "hit-rates per depth band; 1.5B bf16 J-lens vs logit-lens (stage-3 H3).",
        ["obs 2026-07-18-intermediate-concept-evals-h3-confirmed.md", "audit Check B"],
    ),
    "lens_eval_qwen2.5-7b_nf4_n100.pt": _derived(
        _EVAL,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Intermediate-concept eval top-k readout hit-rates per depth band; 7B "
        "nf4 J-lens vs logit-lens (stage-3 scale comparison H2).",
        ["obs 2026-07-20-scale-comparison-7b-vs-1p5b-h2.md", "audit Check B"],
    ),
    "lens_eval_qwen2.5-1.5b_nf4_n100.pt": _derived(
        _EVAL,
        [_L15N4],
        "qwen-1.5b-nf4",
        "Intermediate-concept eval; 1.5B nf4 quantization control (exoneration).",
        ["obs 2026-07-21-quantization-exonerated-1p5b-nf4.md", "audit Check I"],
    ),
    "lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt": _derived(
        _EVAL,
        [_L15C4_REFIT],
        "qwen-1.5b-bf16",
        "Intermediate-concept eval; 1.5B C4-en corpus-sensitivity refit.",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    # -- readout_scan x6 (depth-of-emergence + per-layer Spearman) ------------
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _READ,
        [_L15_JULY, _HW],
        "qwen-1.5b-bf16",
        "Readout-scan depth-of-emergence (top-10) + per-layer Spearman, 1.5B "
        "bf16, wikitext held-out set (stage-3 first pass; GPU-regenerated "
        "2026-07-21 with rich per-layer token capture).",
        ["obs 2026-07-18-readout-scan-1p5b-first-pass.md", "audit Check C"],
    ),
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _READ,
        [_L7B_JULY, _HW],
        "qwen-7b-nf4",
        "Readout-scan depth-of-emergence + per-layer Spearman, 7B nf4, wikitext "
        "held-out set (stage-3 scale comparison).",
        ["obs 2026-07-20-scale-comparison-7b-vs-1p5b-h2.md", "audit Check C"],
    ),
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt": _derived(
        _READ,
        [_L15N4, _HW],
        "qwen-1.5b-nf4",
        "Readout-scan, 1.5B nf4 quantization control, wikitext held-out set.",
        ["obs 2026-07-21-quantization-exonerated-1p5b-nf4.md", "audit Check I"],
    ),
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt": _derived(
        _READ,
        [_L15C4_REFIT, _HW],
        "qwen-1.5b-bf16",
        "Readout-scan, 1.5B C4-en corpus refit (evaluated on the same wikitext "
        "held-out set as the baseline).",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt": _derived(
        _READ,
        [_L15_REFIT, _HC4],
        "qwen-1.5b-bf16",
        "Readout-scan, 1.5B bf16 (wikitext-fit lens) on the diversified C4 "
        "held-out set (held-out-sample robustness control).",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt": _derived(
        _READ,
        [_L7B_REFIT, _HC4],
        "qwen-7b-nf4",
        "Readout-scan, 7B nf4 on the diversified C4 held-out set "
        "(held-out-sample robustness control).",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    # -- structure_scan x7 (varfrac / active-atom / readout-kurtosis per depth) --
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _STRUCT,
        [_L15_JULY, _HW],
        "qwen-1.5b-bf16",
        "Stage-4 J-space structure map (varfrac k={5,10,25,50} / active-atom / "
        "readout-kurtosis per depth), 1.5B bf16, wikitext held-out set.",
        ["obs 2026-07-20-jspace-structure-stage4.md", "audit Check D"],
    ),
    "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _STRUCT,
        [_L7B_JULY, _HW],
        "qwen-7b-nf4",
        "Stage-4 J-space structure map, 7B nf4, wikitext held-out set.",
        ["obs 2026-07-20-jspace-structure-stage4.md", "audit Check D"],
    ),
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt": _derived(
        _STRUCT,
        [_L15N4, _HW],
        "qwen-1.5b-nf4",
        "Stage-4 structure map, 1.5B nf4 quantization control.",
        ["obs 2026-07-21-quantization-exonerated-1p5b-nf4.md", "audit Check I"],
    ),
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n500.pt": _derived(
        _STRUCT,
        [_L15N5, _HW],
        "qwen-1.5b-nf4",
        "Stage-4 structure map, 1.5B nf4 n=500 fit-budget (H1) control.",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check H"],
    ),
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt": _derived(
        _STRUCT,
        [_L15C4_REFIT, _HW],
        "qwen-1.5b-bf16",
        "Stage-4 structure map, 1.5B C4-en corpus refit.",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt": _derived(
        _STRUCT,
        [_L15_REFIT, _HC4],
        "qwen-1.5b-bf16",
        "Stage-4 structure map, 1.5B bf16 on the diversified C4 held-out set.",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt": _derived(
        _STRUCT,
        [_L7B_REFIT, _HC4],
        "qwen-7b-nf4",
        "Stage-4 structure map, 7B nf4 on the diversified C4 held-out set.",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    # -- verbal_report x4 (stage-5.1 / 5.1b verbal-report swap suites) -------
    "verbal_report_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _VR,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.1 verbal-report swap suite (4-condition, magnitude-equalized), "
        "1.5B bf16 @L21.",
        ["obs 2026-07-20-verbal-report-swaps-stage5.md", "audit Checks E, F"],
    ),
    "verbal_report_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _VR,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.1 verbal-report swap suite (4-condition), 7B nf4 @L22.",
        ["obs 2026-07-20-verbal-report-swaps-stage5.md", "audit Checks E, F"],
    ),
    "verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _VR,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.1b chat 6-condition verbal-report swap suite, 1.5B bf16 @L21.",
        ["obs 2026-07-20-verbal-report-swaps-stage5b.md", "audit Check F"],
    ),
    "verbal_report_chat_6c_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _VR,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.1b chat 6-condition verbal-report swap suite, 7B nf4 @L22.",
        ["obs 2026-07-20-verbal-report-swaps-stage5b.md", "audit Check F"],
    ),
    # -- entailed_swap x8 (stage-5.2 entailed-property swap bank) -------------
    "entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank (33 items, 3 equalized-L2 "
        "conditions), 1.5B bf16, chat @L18 (property-effect peak layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, chat @L21 (report layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L24_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, chat @L24.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_plain_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15_JULY],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, plain-prompt @L21.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L18_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L18.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L19_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L19 (property-effect peak).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L22 (report layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_plain_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, plain-prompt @L22.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    # -- paperverbatim x4 (cue-redundancy control probe, --items-json, n=3) --
    "entailed_paperverbatim_chat_all_L19_7b.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim / cue-redundancy control probe (--items-json, "
        "n=3), 7B nf4 @L19, chat all-scope (every pre-answer position).",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    "entailed_paperverbatim_chat_auto_L19_7b.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim control probe (n=3), 7B nf4 @L19, chat auto-scope.",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    "entailed_paperverbatim_plain_all_L19_7b.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim control probe (n=3), 7B nf4 @L19, plain all-scope.",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    "entailed_paperverbatim_plain_auto_L19_7b.pt": _derived(
        _ENT,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim control probe (n=3), 7B nf4 @L19, plain auto-scope.",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    # -- nla_crosstie x1 (stage-6 J-lens x NLA activation-vector cross-tie) ---
    "nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _XTIE,
        [_L7B_JULY],
        "qwen-7b-nf4",
        "Stage-6 NLA cross-tie: 7B nf4 J-lens (L19) x NLA activation-vector at "
        "hidden_states[20]; rank-median + carrier-decomposition metrics (24 "
        "prompts + 1 control). Excludes its .partial.jsonl / .phase1.pt sidecars.",
        ["obs 2026-07-21-nla-crosstie-stage6.md", "audit Check G"],
    ),
    # -- paper-metric ceiling recompute x3 + norm-bias x1 (issue #26) ---------
    "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        "examples/jspace_paper_metric_varfrac.py",
        [_L15_JULY, _HW, "structure_scan_...1.5b...bf16_n100.pt (validation reference)"],
        "qwen-1.5b-bf16",
        "Paper-faithful ceiling metric (excess-over-random orthogonal-projection "
        "FVE at K=median occupancy [gurnee2026-workspace sec 4.2 Fig 30b, A.8]) "
        "on the committed structure-scan grid (30 heldout wikitext prompts x 9 "
        "positions, 27 layers); replicated varfrac@25 validated bit-exact vs "
        "the committed scan (config.validation_max_vf_diff == 0.0); per-position "
        "excess/fve arrays + cluster-bootstrap CIs. rand_seed_base=10000, n_rand=8. "
        "K-consistent top-K selection (2026-07-25 F01 regeneration).",
        [
            "obs 2026-07-24-paper-metric-varfrac-recompute.md",
            "audit Check M",
            "fig 2026-07-24-jspace-paper-metric-excess.png",
        ],
        args="--mode bf16 --lens <cache>/jlens_qwen2.5-1.5b_bf16_n100.pt "
        "--scan <data>/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt "
        "--n-rand 8 --rand-seed-base 10000",
    ),
    "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_allpos.pt": _derived(
        "examples/jspace_paper_metric_varfrac.py",
        [_L15_JULY, _HW],
        "qwen-1.5b-bf16",
        "Paper-metric all-positions sweep (every position in [16, seq_len-2]; "
        "n=5362/layer) at L0/L18/L21/L22 — the paper's measurement population; "
        "per-position arrays + cluster-bootstrap (by prompt, 2000 resamples) 95% "
        "CIs. The decisive 1.5B ceiling-breach artifact (L21 excess 11.15%, CI "
        "[10.95, 11.40], 2000/2000 resamples > 10%; K-consistent selection, "
        "2026-07-25 F01 regeneration).",
        [
            "obs 2026-07-24-paper-metric-varfrac-recompute.md",
            "audit Check M",
            "fig 2026-07-24-jspace-paper-metric-excess.png",
        ],
        args="--mode bf16 --lens <cache>/jlens_qwen2.5-1.5b_bf16_n100.pt "
        "--all-positions --layers 0,18,21,22 --n-rand 4 --rand-seed-base 20000",
    ),
    "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        "examples/jspace_paper_metric_varfrac.py",
        [_L7B_JULY, _HW, "structure_scan_...7b...nf4_n100.pt (validation reference)"],
        "qwen-7b-nf4",
        "7B counterpart of the paper-metric recompute (scan grid, bit-exact "
        "validation); peak excess 4.72% at L22-L23, all 27 layers under the 10% "
        "ceiling. rand_seed_base=30000, n_rand=8. K-consistent top-K selection "
        "(2026-07-25 F01 regeneration).",
        [
            "obs 2026-07-24-paper-metric-varfrac-recompute.md",
            "audit Check M",
            "fig 2026-07-24-jspace-paper-metric-excess.png",
        ],
        args="--model Qwen/Qwen2.5-7B-Instruct --mode nf4 "
        "--lens <cache>/jlens_qwen2.5-7b_nf4_n100.pt "
        "--scan <data>/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt "
        "--n-rand 8 --rand-seed-base 30000",
    ),
    "atom_norm_bias_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        "examples/jspace_atom_norm_bias.py",
        [_L7B_JULY, "structure_scan_...7b...nf4_n100.pt (selected top_atoms)"],
        "qwen-7b-nf4",
        "7B counterpart of the norm-bias summary (W_U untied, read from the "
        "bf16 safetensors via --wu-source safetensors, layers "
        "{0,5,10,15,19,22,26}): early-band bias generalizes (L0 median pctile "
        "66.8, 8% from top norm decile), workspace band neutral (L19 46.5), "
        "late anti-biased (L26 28.0); high-norm tail is undertrained "
        "multilingual junk tokens, rho(atom,W_U) 0.19-0.30 (frequency "
        "coupling was the 1.5B tied embedding).",
        ["obs 2026-07-24-paper-metric-varfrac-recompute.md", "audit Check M"],
    ),
    "atom_norm_bias_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        "examples/jspace_atom_norm_bias.py",
        [_L15_JULY, "structure_scan_...1.5b...bf16_n100.pt (selected top_atoms)"],
        "qwen-1.5b-bf16",
        "Pursuit atom-norm selection-bias summary (issue #26): per-layer "
        "full-vocab atom-norm quantiles/CV, Spearman rho vs W_U row norm "
        "(tied embedding at 1.5B), and norm-percentile stats of the structure "
        "scan's actually-selected atoms. Committed record because the full "
        "July lens input was never committed. Workspace band norm-neutral "
        "(L18/L21 median pctile ~50), early band biased (L0 median 69.5, 30% from top "
        "norm decile, top-norm atoms are format tokens).",
        ["obs 2026-07-24-paper-metric-varfrac-recompute.md", "audit Check M"],
    ),
}
# Paper-metric robustness axes (issue #26, 2026-07-24): the four 1.5B axes +
# the 7B held-out set, each validated bit-exact against its committed
# structure scan (nf4 axes captured with the model in nf4, matching those
# scans). Registered via a loop — the entries differ only in lens/prompts.
for _axis, _fname, _lens, _model, _detail, _args in [
    (
        "corpus (C4-en lens)",
        "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt",
        _L15C4_REFIT,
        "qwen-1.5b-bf16",
        "L21 excess 10.82% (base 10.83%); early band diverges (L0 15.43%). "
        "Regenerated 2026-08-16 with the refit lens (jspace_rerun_scans.sh run "
        "2), validated against the structure scan regenerated in run 1.",
        "--mode bf16 --lens <cache>/jlens_qwen2.5-1.5b_bf16_n100_c4en.pt "
        "--scan <data>/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt "
        "--n-rand 8 --rand-seed-base 10000",
    ),
    (
        "quantization (nf4 lens)",
        "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt",
        _L15N4,
        "qwen-1.5b-nf4",
        "L21 excess 10.83%; run with --mode nf4 (matches the scan capture).",
        "--mode nf4 --lens <cache>/jlens_qwen2.5-1.5b_nf4_n100.pt "
        "--scan <data>/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt "
        "--n-rand 8 --rand-seed-base 10000",
    ),
    (
        "n-budget (nf4 n=500 lens)",
        "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n500.pt",
        _L15N5,
        "qwen-1.5b-nf4",
        "L21 excess 10.69%; run with --mode nf4.",
        "--mode nf4 --lens <cache>/jlens_qwen2.5-1.5b_nf4_n500.pt "
        "--scan <data>/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n500.pt "
        "--n-rand 8 --rand-seed-base 10000",
    ),
    (
        "held-out sample (C4 prompts)",
        "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt",
        _L15_REFIT,
        "qwen-1.5b-bf16",
        "L21 excess 11.70% on the diversified C4 held-out set. Regenerated "
        "2026-08-16 with the refit lens (jspace_rerun_scans.sh run 6), "
        "validated against the structure scan regenerated in run 5.",
        "--mode bf16 --lens <cache>/jlens_qwen2.5-1.5b_bf16_n100.pt "
        "--scan <data>/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt "
        "--prompts <data>/heldout_prompts_c4en_n30.json --n-rand 8 --rand-seed-base 10000",
    ),
    (
        "held-out sample, 7B (C4 prompts)",
        "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt",
        _L7B_REFIT,
        "qwen-7b-nf4",
        "band peak L23 excess 5.98%, under the ceiling (0/2000 resamples over). "
        "Regenerated 2026-08-16 with the refit lens (jspace_rerun_scans.sh run "
        "9), validated against the structure scan regenerated in run 8.",
        "--model Qwen/Qwen2.5-7B-Instruct --mode nf4 "
        "--lens <cache>/jlens_qwen2.5-7b_nf4_n100.pt "
        "--scan <data>/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt "
        "--prompts <data>/heldout_prompts_c4en_n30.json --n-rand 8 --rand-seed-base 30000",
    ),
]:
    _DERIVED[_fname] = _derived(
        "examples/jspace_paper_metric_varfrac.py",
        [_lens, _HC4 if "heldout" in _fname else _HW],
        _model,
        f"Paper-metric robustness axis — {_axis}: excess-over-random "
        f"orthogonal-projection FVE at K=median occupancy (K-consistent "
        f"top-K, 2026-07-25 F01 regeneration), scan grid, "
        f"validated bit-exact vs the committed structure scan "
        f"(config.validation_max_vf_diff == 0.0). {_detail} All five axes "
        f"hold the ceiling verdicts (P(boot>10%) unanimous each side).",
        ["obs 2026-07-24-paper-metric-varfrac-recompute.md", "audit Check M"],
        args=_args,
    )


# Dimension-matched K recompute (issue #83, 2026-09-23): the excess metric
# scales with K/d_model, so these 7B runs hold K fixed (--k-fixed) at every
# layer — K=58 matches the 1.5B K/d (58/3584 = 0.0162 vs 25/1536 = 0.0163),
# K=25 matches the 1.5B K. top-K = selection-order prefix of the k_max=64
# pursuit support; n_short_support == 0 at every layer in all three. The run
# logs are plain git files under data/cache/logs/ (scan_<name>.log below).
_K83_ARGS = (
    "--model Qwen/Qwen2.5-7B-Instruct --mode nf4 --device cuda "
    "--lens <cache>/jlens_qwen2.5-7b_nf4_n100.pt "
)
_K83_TAIL = "--n-prompts 30 --n-rand 8 --rand-seed-base 30000 --k-snap 25 --k-max 64"
for _fname, _scan, _prompts, _k, _log, _detail in [
    (
        "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en_k58.pt",
        "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt",
        _HC4,
        58,
        "scan_paper_metric_heldoutc4en_7b_k58.log",
        "C4 held-out prompts; peak L23 excess 7.67% CI95 [7.30, 8.03], 0/2000 "
        "resamples over 10%; replicated varfrac@25 bit-exact vs the scan "
        "(config.validation_max_vf_diff == 0.0), because the heldoutc4en "
        "structure scan was regenerated 2026-08-16 with this same refit lens "
        "(jspace_rerun_scans.sh run structure_heldoutc4en_7b).",
    ),
    (
        "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_refitlens_k25.pt",
        "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        _HW,
        25,
        "scan_paper_metric_7b_refitlens_k25.log",
        "wikitext grid prompts, current (refit) lens; peak L22 excess 4.81% "
        "CI95 [4.66, 4.97]. Validation vs the July pre-refit structure scan "
        "is NOT bit-exact: config.validation_max_vf_diff = 4.379e-01 (max "
        "over positions), while the per-layer mean varfrac@25 agrees with "
        "the July artifact to 2.6e-3 (signed difference of layer means). "
        "Per-position differences were not persisted, so how many of the "
        "270 positions per layer differ is not established; the signed "
        "mean agreement does not bound it. "
        "The committed wikitext grid structure scan and the K=23-24 grid "
        "artifact were produced by the July pre-refit 7B lens (fit "
        "2026-07-20; see observations/2026-07-24-paper-metric-varfrac-"
        "recompute.md), whose full 27-layer file was never committed (the "
        "cache file of the same name is the 2026-08-16 refit) and of which "
        "only the 7-layer subset jlens_qwen2.5-7b_nf4_n100_layer-subset.pt "
        "is committed. The older 7B entries' inputs name that July lens "
        "explicitly (issue #87).",
    ),
    (
        "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_refitlens_k58.pt",
        "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        _HW,
        58,
        "scan_paper_metric_7b_refitlens_k58.log",
        "wikitext grid prompts, current (refit) lens; peak L22 excess 6.16% "
        "CI95 [5.99, 6.36]. Same non-bit-exact validation as the K=25 run "
        "(config.validation_max_vf_diff 4.379e-01). "
        "The committed wikitext grid structure scan and the K=23-24 grid "
        "artifact were produced by the July pre-refit 7B lens (fit "
        "2026-07-20; see observations/2026-07-24-paper-metric-varfrac-"
        "recompute.md), whose full 27-layer file was never committed (the "
        "cache file of the same name is the 2026-08-16 refit) and of which "
        "only the 7-layer subset jlens_qwen2.5-7b_nf4_n100_layer-subset.pt "
        "is committed. The older 7B entries' inputs name that July lens "
        "explicitly (issue #87).",
    ),
]:
    _DERIVED[_fname] = _derived(
        "examples/jspace_paper_metric_varfrac.py",
        [_L7B_REFIT, _prompts, f"{_scan} (validation reference)"],
        "qwen-7b-nf4",
        f"Dimension-matched paper-metric recompute (issue #83): excess-over-"
        f"random orthogonal-projection FVE with K fixed at {_k} at every "
        f"layer (--k-fixed; per-layer K_used, n_short_support, config.k_fixed "
        f"recorded). {_detail} Run log: cache/logs/{_log}.",
        [
            "audit Check P (stage 2, PR #84)",
            "fig 2026-09-23-jspace-paper-metric-matched-kd.png (stage 2, PR #84)",
        ],
        args=_K83_ARGS
        + f"--scan <data>/{_scan} --prompts <data>/{_prompts} "
        + _K83_TAIL
        + f" --k-fixed {_k}",
    )

META.update(_DERIVED)

# Disk-derived provenance stub for a top-level deliverable with no META entry
# yet (a lens `.pt`/sidecar promoted before its registry line was added). Its
# sha256/size are still captured; --check only verifies metadata for names in
# META, so an unregistered file is checksum-tracked without a false drift.
_UNREGISTERED: dict[str, Any] = {
    "class": "unregistered",
    "producing_script": None,
    "producing_command": None,
    "inputs": [],
    "requires_model": None,
    "provenance": "UNREGISTERED — add a META entry in examples/jspace_data_manifest.py",
    "consumers": [],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    """True when `path` holds a git-LFS pointer stub instead of real content
    (an LFS-less clone, or the opt-in cache/ exclusion applied too broadly)."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _deliverables() -> list[str]:
    """Top-level deliverable filenames: `.pt` artifacts and `.json` files
    (lens sidecars + the fitting corpus), minus MANIFEST.json itself. The
    globs are non-recursive, so the regenerable `cache/` subdir is excluded."""
    names: set[str] = set()
    for pattern in ("*.pt", "*.json"):
        names.update(p.name for p in DATA_DIR.glob(pattern) if p.is_file())
    names.discard(MANIFEST.name)
    return sorted(names)


def _metadata_fields(name: str) -> dict[str, Any]:
    """The provenance fields for `name`, derived from META — everything the
    manifest records EXCEPT the disk-derived sha256/size_bytes. Shared by the
    writer and the `--check` drift detector, so editing META without
    regenerating MANIFEST.json is caught. A name with no META entry yields the
    `unregistered` stub (checksum-tracked, provenance pending)."""
    m = META.get(name)
    if m is None:
        return dict(_UNREGISTERED)
    _args = m.get("producing_args")
    return {
        "class": m["class"],
        "producing_script": m["producing_script"],
        "producing_command": (
            f"python {m['producing_script']} {_args}"
            if _args
            else f"python {m['producing_script']}"
        ),
        "inputs": m["inputs"],
        "requires_model": m["requires_model"],
        "provenance": m["provenance"],
        "consumers": m["consumers"],
    }


def _disk_vs_expected(
    expected: set[str], on_disk: set[str]
) -> tuple[list[str], list[str]]:
    """(missing, extra): names expected but absent on disk, and on disk but
    unregistered. Shared by the writer (which warns on either) and the
    --check detector (which collects extras/missing as problems)."""
    return sorted(expected - on_disk), sorted(on_disk - expected)


def build_entries() -> list[dict[str, Any]]:
    on_disk = _deliverables()
    stubs = [n for n in on_disk if _is_lfs_pointer(DATA_DIR / n)]
    if stubs:
        raise SystemExit(
            f"ERROR: {len(stubs)} deliverable(s) are git-LFS pointer stubs, not "
            f"real artifacts — hashing them would produce bogus checksums. Run "
            f"`git lfs install && git lfs pull`, then retry. Stubs: {stubs}"
        )
    missing, extra = _disk_vs_expected(set(META), set(on_disk))
    for name in missing:
        print(f"WARNING: known file absent on disk (skipped): {name}")
    for name in extra:
        print(
            f"WARNING: deliverable has no META entry — recorded as "
            f"unregistered, add one: {name}"
        )
    entries: list[dict[str, Any]] = []
    for name in on_disk:
        p = DATA_DIR / name
        entries.append(
            {
                "filename": name,
                "sha256": sha256_of(p),
                "size_bytes": p.stat().st_size,
                **_metadata_fields(name),
            }
        )
    return entries


# Top-level manifest prose. Hoisted out of write_manifest() so --check can
# assert the committed MANIFEST.json still carries these exact literals — an
# edit here that was never regenerated is drift, same as a META edit (#48).
_DESCRIPTION = (
    "Raw + derived jspace artifacts: the frozen wikitext-103 + C4-en "
    "fitting/held-out corpora, fitted Jacobian-lens layer-subset tensors "
    "(jlens native format) with their .config.json provenance sidecars, "
    "and the promoted derived metric/scan/swap products (lens_eval, "
    "readout_scan, structure_scan, verbal_report, entailed_swap + "
    "paper-verbatim probes, nla_crosstie) the audit re-derives from. "
    "Committed via git-LFS so figures and the audit reproduce from a "
    "clean clone. See README.md beside this manifest for what the "
    "cache/ subdir does and does not commit. The J-lens dependency is pinned in the "
    "top-level `jlens_pin` field below; the eval prompt sets ship "
    "with that pinned checkout, not with this repo."
)

_JLENS_PIN: dict[str, Any] = {
    "repo": "https://github.com/anthropics/jacobian-lens",
    "commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
    "subject": "Initial release",
    "author_date": "2026-07-01",
    "commit_date": "2026-07-02",
    "provenance": (
        "The J-lens fit/readout implementation (jlens.fit, native .pt "
        "format) that produced every lens + derived artifact in this "
        "manifest. The multihop / association intermediate-concept eval "
        "prompt sets used by examples/jspace_lens_eval.py also live in "
        "that repository, so the eval tables (audit Check B) are "
        "reproducible only against this pinned commit. Check it out at "
        "the commit above and point JSPACE_EVAL_DIR at its "
        "data/evaluations (the default assumes a sibling checkout at "
        "../jacobian-lens)."
    ),
}

_TRUST_NOTE = (
    "Lens .pt files load via jlens / torch.load (pickle). Safe here in "
    "the deserialization sense: every .pt is a locally-fitted tensor "
    "dump produced by this repo's own scripts, never an externally "
    "supplied pickle; the corpora are plain JSON. That is a statement "
    "about pickle trust ONLY — the corpora themselves ARE third-party "
    "data (C4 under ODC-BY, WikiText-103 under CC BY-SA; see "
    "LICENSE-DATA.md) and the C4 files are PII-redacted (see "
    "README.md). Verify sha256 with `--check` before loading on an "
    "untrusted copy."
)

# The top-level (non-per-file, non-tallied) fields --check re-derives.
_TOPLEVEL_LITERALS: dict[str, Any] = {
    "arc": "jspace",
    "description": _DESCRIPTION,
    "jlens_pin": _JLENS_PIN,
    "trust_note": _TRUST_NOTE,
}


def write_manifest() -> None:
    entries = build_entries()
    doc: dict[str, Any] = {
        **_TOPLEVEL_LITERALS,
        "total_files": len(entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    mb = doc["total_size_bytes"] / 1e6
    print(
        f"wrote {MANIFEST.relative_to(_REPO_ROOT)}  ({doc['total_files']} files, {mb:.1f} MB)"
    )


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: {MANIFEST} does not exist (create it with --write)")
        return 1
    doc = json.loads(MANIFEST.read_text())
    recorded = {e["filename"]: e for e in doc["files"]}
    on_disk = set(_deliverables())
    problems: list[str] = []
    # Top-level prose drift: an edit to _DESCRIPTION / _JLENS_PIN / _TRUST_NOTE
    # (or the arc name) that was never regenerated into the committed manifest.
    for field, expected in _TOPLEVEL_LITERALS.items():
        if doc.get(field) != expected:
            problems.append(
                f"top-level drift: {field}\n"
                f"    manifest={doc.get(field)!r}\n"
                f"    writer  ={expected!r}"
            )
    # Tallies must match the file list they summarize.
    for field, expected in (
        ("total_files", len(recorded)),
        ("total_size_bytes", sum(e["size_bytes"] for e in recorded.values())),
    ):
        if doc.get(field) != expected:
            problems.append(
                f"top-level drift: {field} (manifest={doc.get(field)!r}, "
                f"files imply {expected!r})"
            )
    missing, extra = _disk_vs_expected(set(recorded), on_disk)
    problems += [f"missing on disk: {name}" for name in missing]
    problems += [f"on disk but not in manifest: {name}" for name in extra]
    for name in sorted(set(recorded) & on_disk):
        actual = sha256_of(DATA_DIR / name)
        if actual != recorded[name]["sha256"]:
            problems.append(
                f"sha256 drift: {name}\n    manifest={recorded[name]['sha256']}\n    on-disk ={actual}"
            )
        # Provenance drift: a META edit (reclassify, corrected inputs/consumers)
        # that was never regenerated into the committed manifest. Unregistered
        # files (no META entry) are checksum-only, so skip their metadata.
        if name in META:
            for field, expected in _metadata_fields(name).items():
                if recorded[name].get(field) != expected:
                    problems.append(
                        f"metadata drift: {name}.{field}\n"
                        f"    manifest={recorded[name].get(field)!r}\n"
                        f"    META     ={expected!r}"
                    )
    if problems:
        print("MANIFEST CHECK: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"MANIFEST CHECK: OK  ({len(recorded)} files, "
        "sha256 + per-file metadata + top-level literals match)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Explicit-mode CLI. Neither a bare invocation nor `--help` may rewrite
    MANIFEST.json — writing requires `--write`, so a reflexive run of the
    script can never silently replace the committed manifest (#44)."""
    parser = argparse.ArgumentParser(
        prog="jspace_data_manifest.py",
        description="Generate or verify research/arcs/04_jspace/data/MANIFEST.json.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest against disk + META; exit 1 on drift",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="(re)write MANIFEST.json from disk + META",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check_manifest()
    if args.write:
        write_manifest()
        return 0
    parser.print_usage()
    print("error: pass exactly one of --check or --write", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
