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
(`jspace_fit_lens.py --out-dir` default) is regenerable working state and is
EXCLUDED — the non-recursive globs never descend into it.

Two modes:
    python examples/jspace_data_manifest.py            # (re)write MANIFEST.json
    python examples/jspace_data_manifest.py --check     # verify, exit 1 on drift

The `--check` mode is the drift detector: it recomputes every sha256 AND
re-derives each registered file's provenance fields from META, comparing both
against the committed manifest — catching silent corruption, a re-fit that
wasn't re-committed, a missing/extra file, OR a META edit (reclassify,
corrected inputs/consumers) that was never regenerated into MANIFEST.json.

Unlike the writer, which fills a fixed registry, this arc's lens set grows as
sign-off gates open (which layers are committed vs cache-only is an open
design decision). So the writer does NOT hard-fail on drift between META and
disk: a known file still downloading/fitting is warned-and-skipped, and a
promoted deliverable with no META entry yet is recorded as `unregistered`
(sha256 captured, provenance pending) and warned — a later session fills in
its META entry. See `research/ARC_PROCESS.md` § "Raw data is a deliverable".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"

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
    # by examples/jspace_promote_lens_subset.py; the full 27-layer set stays
    # cache-only (data/cache/, gitignored). The .config.json sidecars are
    # top-level *.json deliverables too, so each gets its own META entry.
    "jlens_qwen2.5-7b_nf4_n100_layer-subset.pt": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": ["jlens_qwen2.5-7b_nf4_n100.pt (data/cache/, gitignored)"],
        "requires_model": "qwen-7b-nf4",
        "provenance": (
            "Layers {0,5,10,15,20,25,26} of the full 27-layer J-lens fitted by "
            "examples/jspace_fit_lens.py on the frozen wikitext corpus (n=100, "
            "jlens defaults); 16.26 h GPU fit completed 2026-07-20. Reduced "
            "subset promoted per design-plan Decision 4 (trailing layer 27 "
            "clamped to the last valid index 26); full set cache-only."
        ),
        "consumers": ["clean-clone lens inspection at representative depths"],
    },
    "jlens_qwen2.5-7b_nf4_n100_layer-subset.config.json": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": ["jlens_qwen2.5-7b_nf4_n100.config.json (data/cache/, gitignored)"],
        "requires_model": "qwen-7b-nf4",
        "provenance": (
            "Provenance sidecar for jlens_qwen2.5-7b_nf4_n100_layer-subset.pt: "
            "the cache fit sidecar plus subset_layers + full_set_location."
        ),
        "consumers": ["jlens_qwen2.5-7b_nf4_n100_layer-subset.pt (sidecar)"],
    },
    "jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": ["jlens_qwen2.5-1.5b_bf16_n100.pt (data/cache/, gitignored)"],
        "requires_model": "qwen-1.5b-bf16",
        "provenance": (
            "Layers {0,5,10,15,20,25,26} of the full 27-layer J-lens fitted by "
            "examples/jspace_fit_lens.py on the frozen wikitext corpus (n=100, "
            "jlens defaults); 3 h GPU fit completed 2026-07-18. Reduced subset "
            "promoted per design-plan Decision 4 (trailing layer 27 clamped to "
            "the last valid index 26); full set cache-only."
        ),
        "consumers": ["clean-clone lens inspection at representative depths"],
    },
    "jlens_qwen2.5-1.5b_bf16_n100_layer-subset.config.json": {
        "class": "raw",
        "producing_script": "examples/jspace_promote_lens_subset.py",
        "inputs": [
            "jlens_qwen2.5-1.5b_bf16_n100.config.json (data/cache/, gitignored)"
        ],
        "requires_model": "qwen-1.5b-bf16",
        "provenance": (
            "Provenance sidecar for jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt: "
            "the cache fit sidecar plus subset_layers + full_set_location."
        ),
        "consumers": ["jlens_qwen2.5-1.5b_bf16_n100_layer-subset.pt (sidecar)"],
    },
    # ---- derived metric/scan/swap artifacts (stage-7 promotion) ------------
    # The small derived artifacts the audit (examples/jspace_audit_findings.py)
    # re-derives from, promoted out of data/cache/ into data/ for clean-clone
    # auditability. Built compactly below via _derived() and merged into META
    # (the full fitted lenses that produced them stay cache-only, Decision 4).
}


def _derived(
    script: str,
    inputs: list[str],
    model: str,
    provenance: str,
    consumers: list[str],
) -> dict[str, Any]:
    return {
        "class": "derived",
        "producing_script": script,
        "inputs": inputs,
        "requires_model": model,
        "provenance": provenance,
        "consumers": consumers,
    }


# Full fitted lenses (cache-only, gitignored) named as inputs of the derived set.
_L15 = "jlens_qwen2.5-1.5b_bf16_n100.pt (data/cache/, gitignored)"
_L7B = "jlens_qwen2.5-7b_nf4_n100.pt (data/cache/, gitignored)"
_L15N4 = "jlens_qwen2.5-1.5b_nf4_n100.pt (data/cache/, gitignored)"
_L15N5 = "jlens_qwen2.5-1.5b_nf4_n500.pt (data/cache/, gitignored)"
_L15C4 = "jlens_qwen2.5-1.5b_bf16_n100_c4en.pt (data/cache/, gitignored)"
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
        [_L15],
        "qwen-1.5b-bf16",
        "Intermediate-concept eval (multihop + association) top-k readout "
        "hit-rates per depth band; 1.5B bf16 J-lens vs logit-lens (stage-3 H3).",
        ["obs 2026-07-18-intermediate-concept-evals-h3-confirmed.md", "audit Check B"],
    ),
    "lens_eval_qwen2.5-7b_nf4_n100.pt": _derived(
        _EVAL,
        [_L7B],
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
        [_L15C4],
        "qwen-1.5b-bf16",
        "Intermediate-concept eval; 1.5B C4-en corpus-sensitivity refit.",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    # -- readout_scan x6 (depth-of-emergence + per-layer Spearman) ------------
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _READ,
        [_L15, _HW],
        "qwen-1.5b-bf16",
        "Readout-scan depth-of-emergence (top-10) + per-layer Spearman, 1.5B "
        "bf16, wikitext held-out set (stage-3 first pass; GPU-regenerated "
        "2026-07-21 with rich per-layer token capture).",
        ["obs 2026-07-18-readout-scan-1p5b-first-pass.md", "audit Check C"],
    ),
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _READ,
        [_L7B, _HW],
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
        [_L15C4, _HW],
        "qwen-1.5b-bf16",
        "Readout-scan, 1.5B C4-en corpus refit (evaluated on the same wikitext "
        "held-out set as the baseline).",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt": _derived(
        _READ,
        [_L15, _HC4],
        "qwen-1.5b-bf16",
        "Readout-scan, 1.5B bf16 (wikitext-fit lens) on the diversified C4 "
        "held-out set (held-out-sample robustness control).",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt": _derived(
        _READ,
        [_L7B, _HC4],
        "qwen-7b-nf4",
        "Readout-scan, 7B nf4 on the diversified C4 held-out set "
        "(held-out-sample robustness control).",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    # -- structure_scan x7 (varfrac / active-atom / readout-kurtosis per depth) --
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _STRUCT,
        [_L15, _HW],
        "qwen-1.5b-bf16",
        "Stage-4 J-space structure map (varfrac k={5,10,25,50} / active-atom / "
        "readout-kurtosis per depth), 1.5B bf16, wikitext held-out set.",
        ["obs 2026-07-20-jspace-structure-stage4.md", "audit Check D"],
    ),
    "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _STRUCT,
        [_L7B, _HW],
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
        [_L15C4, _HW],
        "qwen-1.5b-bf16",
        "Stage-4 structure map, 1.5B C4-en corpus refit.",
        ["obs 2026-07-20-corpus-sensitivity-c4-1p5b.md", "audit Check J"],
    ),
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt": _derived(
        _STRUCT,
        [_L15, _HC4],
        "qwen-1.5b-bf16",
        "Stage-4 structure map, 1.5B bf16 on the diversified C4 held-out set.",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt": _derived(
        _STRUCT,
        [_L7B, _HC4],
        "qwen-7b-nf4",
        "Stage-4 structure map, 7B nf4 on the diversified C4 held-out set.",
        ["obs 2026-07-22-n500-and-heldout-robustness.md", "audit Check K"],
    ),
    # -- verbal_report x4 (stage-5.1 / 5.1b verbal-report swap suites) -------
    "verbal_report_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _VR,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.1 verbal-report swap suite (4-condition, magnitude-equalized), "
        "1.5B bf16 @L21.",
        ["obs 2026-07-20-verbal-report-swaps-stage5.md", "audit Checks E, F"],
    ),
    "verbal_report_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _VR,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.1 verbal-report swap suite (4-condition), 7B nf4 @L22.",
        ["obs 2026-07-20-verbal-report-swaps-stage5.md", "audit Checks E, F"],
    ),
    "verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _VR,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.1b chat 6-condition verbal-report swap suite, 1.5B bf16 @L21.",
        ["obs 2026-07-20-verbal-report-swaps-stage5b.md", "audit Check F"],
    ),
    "verbal_report_chat_6c_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _VR,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.1b chat 6-condition verbal-report swap suite, 7B nf4 @L22.",
        ["obs 2026-07-20-verbal-report-swaps-stage5b.md", "audit Check F"],
    ),
    # -- entailed_swap x8 (stage-5.2 entailed-property swap bank) -------------
    "entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank (33 items, 3 equalized-L2 "
        "conditions), 1.5B bf16, chat @L18 (property-effect peak layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, chat @L21 (report layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L24_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, chat @L24.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_plain_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": _derived(
        _ENT,
        [_L15],
        "qwen-1.5b-bf16",
        "Stage-5.2 entailed-property swap bank, 1.5B bf16, plain-prompt @L21.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L18_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L18.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L19_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L19 (property-effect peak).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_chat_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, chat @L22 (report layer).",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    "entailed_swap_plain_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": _derived(
        _ENT,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 entailed-property swap bank, 7B nf4, plain-prompt @L22.",
        ["obs 2026-07-22-entailed-property-swaps-stage52.md", "audit Check L"],
    ),
    # -- paperverbatim x4 (cue-redundancy control probe, --items-json, n=3) --
    "entailed_paperverbatim_chat_all_L19_7b.pt": _derived(
        _ENT,
        [_L7B],
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
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim control probe (n=3), 7B nf4 @L19, chat auto-scope.",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    "entailed_paperverbatim_plain_all_L19_7b.pt": _derived(
        _ENT,
        [_L7B],
        "qwen-7b-nf4",
        "Stage-5.2 paper-verbatim control probe (n=3), 7B nf4 @L19, plain all-scope.",
        [
            "obs 2026-07-22-entailed-property-swaps-stage52.md (probe addendum)",
            "audit Check L",
        ],
    ),
    "entailed_paperverbatim_plain_auto_L19_7b.pt": _derived(
        _ENT,
        [_L7B],
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
        [_L7B],
        "qwen-7b-nf4",
        "Stage-6 NLA cross-tie: 7B nf4 J-lens (L19) x NLA activation-vector at "
        "hidden_states[20]; rank-median + carrier-decomposition metrics (24 "
        "prompts + 1 control). Excludes its .partial.jsonl / .phase1.pt sidecars.",
        ["obs 2026-07-21-nla-crosstie-stage6.md", "audit Check G"],
    ),
}
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
    return {
        "class": m["class"],
        "producing_script": m["producing_script"],
        "producing_command": f"python {m['producing_script']}",
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


def write_manifest() -> None:
    entries = build_entries()
    doc = {
        "arc": "jspace",
        "description": (
            "Raw + derived jspace artifacts: the frozen wikitext-103 + C4-en "
            "fitting/held-out corpora, fitted Jacobian-lens layer-subset tensors "
            "(jlens native format) with their .config.json provenance sidecars, "
            "and the promoted derived metric/scan/swap products (lens_eval, "
            "readout_scan, structure_scan, verbal_report, entailed_swap + "
            "paper-verbatim probes, nla_crosstie) the audit re-derives from. "
            "Committed via git-LFS so figures and the audit reproduce from a "
            "clean clone. The cache/ subdir (full fitted lenses + working "
            "state) is excluded. The J-lens dependency is pinned in the "
            "top-level `jlens_pin` field below (the eval prompt sets live in "
            "that clone)."
        ),
        "jlens_pin": {
            "repo": "github.com/anthropics/jacobian-lens (editable clone at "
            "/home/ai/ai-projects/jacobian-lens)",
            "commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "subject": "Initial release",
            "date": "2026-07-02",
            "provenance": (
                "The J-lens fit/readout implementation (jlens.fit, native .pt "
                "format) that produced every lens + derived artifact in this "
                "manifest. The multihop / association intermediate-concept eval "
                "prompt sets used by examples/jspace_lens_eval.py also live in "
                "this clone, so the eval tables (audit Check B) are reproducible "
                "only against this pinned commit."
            ),
        },
        "trust_note": (
            "Lens .pt files load via jlens / torch.load (pickle). Safe here: "
            "locally-fitted tensor dumps, not third-party data; the corpus is "
            "plain JSON. Verify sha256 with `--check` before loading on an "
            "untrusted copy."
        ),
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
        print(f"FAIL: {MANIFEST} does not exist (run without --check to create it)")
        return 1
    doc = json.loads(MANIFEST.read_text())
    recorded = {e["filename"]: e for e in doc["files"]}
    on_disk = set(_deliverables())
    problems: list[str] = []
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
    print(f"MANIFEST CHECK: OK  ({len(recorded)} files, sha256 + metadata match)")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(check_manifest())
    write_manifest()
