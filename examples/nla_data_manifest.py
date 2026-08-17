"""Generate / verify the raw-dataset manifest for the NLA research arc.

The arc's raw `.pt` artifacts live (committed, git-LFS) under
`research/arcs/01_nla-verbalizer/data/`. This script writes a checksummed
`MANIFEST.json` next to them that records, per file: sha256, size, whether
it is a capture-root (needs the model to regenerate) or a derived artifact
(regenerable from other `.pt` by a committed script), the producing script
and command, its `.pt` inputs, what model it needs, and who consumes it.

Two modes, both explicit — bare invocation prints usage and writes nothing, so
a typo or `--help` can never silently rewrite the committed manifest:
    python examples/nla_data_manifest.py --check     # verify, exit 1 on drift
    python examples/nla_data_manifest.py --write     # (re)write MANIFEST.json

The `--check` mode is the drift detector: it recomputes every sha256 AND
re-derives each file's provenance fields from META, comparing both against the
committed manifest — catching silent corruption, a re-capture that wasn't
re-committed, a missing/extra file, OR a META edit (reclassify, corrected
inputs/consumers) that was never regenerated into MANIFEST.json. Run it in the
arc's audit step alongside `nla_audit_findings.py`.

DATA_DIR comes from the shared `_nla_artifacts.DATA` (self-locating from its own
file), so this script runs from any CWD. See `research/ARC_PROCESS.md`
§ "Raw data is a deliverable".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _nla_artifacts import DATA as DATA_DIR
from _nla_artifacts import LFS_STUB_NOTE as _LFS_STUB_NOTE
from _nla_artifacts import is_lfs_pointer

_REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = DATA_DIR / "MANIFEST.json"

# The exact model set every capture-root in this arc was produced against.
# NOTE ON REVISIONS: no HF commit revision was pinned or recorded at capture
# time (2026-05-12..15) — the scripts loaded each repo at its then-current
# `main`. The revisions are therefore genuinely unknown and are recorded as
# null rather than back-filled with a guess; a re-capture cannot be asserted
# bit-identical to these artifacts for that reason.
MODEL_PIN: dict[str, Any] = {
    "note": (
        "Models the arc's capture-roots were produced against. No HF revision "
        "was pinned or recorded at capture time, so `revision` is null for all "
        "three: re-capture reproducibility is repo-level, not commit-level."
    ),
    "retrieved": "2026-05-12 (first capture in the arc; NLA pair released 2026-05-07)",
    "models": [
        {
            "role": "base",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "revision": None,
            "purpose": "source of the layer-20 hidden states (h[20], d=3584)",
        },
        {
            "role": "av",
            "repo_id": "kitft/nla-qwen2.5-7b-L20-av",
            "url": "https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av",
            "revision": None,
            "purpose": "activation -> verbalization (h[20] -> natural language)",
        },
        {
            "role": "ar",
            "repo_id": "kitft/nla-qwen2.5-7b-L20-ar",
            "url": "https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar",
            "revision": None,
            "purpose": "activation reconstruction (natural language -> h[20])",
        },
    ],
}

# Per-artifact provenance. `requires_model` values: none | qwen-base |
# qwen+av | qwen+av+ar | av | ar. `consumers` lists representative figures /
# downstream artifacts / audit (not an exhaustive figure enumeration —
# pairwise_and_hotdims.pt and vocab_atlas.pt feed most of the atlas figures).
META: dict[str, dict[str, Any]] = {
    # ---- capture-roots: require a model load, expensive (CPU-hours) --------
    "aggregate_faithfulness.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_aggregate_faithfulness.py",
        "provenance": (
            "113 generation-step captures across 8 hand-authored prompts "
            "(~15 tokens each), each h[20] verbalized by the AV and "
            "reconstructed by the AR; recorded in observations/"
            "2026-05-13-nla-aggregate-faithfulness-8-prompts.md"
        ),
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": ["geometric_features.pt", "pairwise_and_hotdims.pt", "AUDIT 1/20"],
    },
    "rabbit_haiku_gen_trajectory.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_faithfulness.py",
        "provenance": (
            "per-generated-token h[20] trajectory (15 steps) for the "
            "hand-authored prompt 'Write me a haiku about a rabbit in spring.', "
            "with AV verbalization + AR round-trip per step; recorded in "
            "observations/2026-05-12-nla-faithfulness-haiku.md"
        ),
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": ["fig6", "fig14", "pairwise_and_hotdims.pt", "AUDIT 21"],
    },
    "forced_continuation.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_forced_continuation.py",
        "provenance": (
            "4 hand-authored natural/forced-completion prompt pairs "
            "(Yes/No, Paris/Berlin, 4/5, factual/refusal), h[20] captured at "
            "matched positions and verbalized by the AV; recorded in "
            "observations/2026-05-13-nla-forced-continuation-detects-named-falsehoods.md"
        ),
        "inputs": [],
        "requires_model": "qwen+av",
        "consumers": ["fig15", "fig16", "geometric_features.pt", "AUDIT 10"],
    },
    "country_concept_vector.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_country_concept_vector.py",
        "provenance": (
            "8 hand-authored country prompts vs 8 matched non-country prompts, "
            "h[20] at the last prompt token, their difference-of-means "
            "direction, and held-out test-prompt projections; recorded in "
            "observations/2026-05-13-nla-cav-country-direction.md"
        ),
        "inputs": [],
        "requires_model": "qwen+av",
        "consumers": ["fig13", "pairwise_and_hotdims.pt", "AUDIT 8"],
    },
    "vocab_atlas.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_vocab_atlas_capture.py",
        "provenance": (
            "128 hand-authored anchor tokens across 23 categories, each "
            "captured as h[20] at end-of-single-token-user-message under the "
            "Qwen chat template. Captured against the pre-deduplication VOCAB "
            "dict (3 anchors appeared in two categories each — arc README L5); "
            "the source dict now holds 125 unique anchors, so a re-capture "
            "yields 125 and shifts every downstream number (issue #51). "
            "Recorded in observations/2026-05-13-nla-vocab-atlas-grid.md"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["fig19-fig37 (atlas/discriminant figures)", "AUDIT 12/13"],
    },
    "discriminant_stability.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_discriminant_stability_capture.py",
        "provenance": (
            "8 hand-authored anchor words x 4 prefix-length context variants "
            "(single / short / medium / long), each captured at the "
            "position-(-1) end-of-prompt slot (see arc README L7 on what that "
            "position actually represents); recorded in "
            "observations/2026-05-13-nla-discriminant-validation.md"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["fig28", "AUDIT 14"],
    },
    "interpolation_flipbook.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_interpolation_flipbook.py",
        "provenance": (
            "two hand-authored natural-language anchors (factual/geography vs "
            "poetic/nature) AR-encoded to h_A/h_B, linearly interpolated on a "
            "20-step grid, each step AV-verbalized; recorded in "
            "observations/2026-05-13-nla-interpolation-flipbook.md"
        ),
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": [
            "fig17",
            "fig18",
            "fig21",
            "dense_interp_near_pivot.pt",
            "AUDIT 11",
        ],
    },
    "mid_seq_vocab_atlas.pt": {
        "class": "capture-root",
        "producing_script": "examples/nla_mid_seq_vocab_atlas_capture.py",
        "provenance": (
            "the vocab-atlas anchors re-captured mid-sequence (tokenize-then-"
            "locate inside a fixed hand-authored carrier sentence) instead of at "
            "end-of-single-token-message — the cross-protocol arm; recorded in "
            "observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["mid_seq_compare.pt", "mid_seq_native_compare.pt", "AUDIT 15/16"],
    },
    # ---- derived: regenerable from other .pt by a committed script ---------
    "geometric_features.pt": {
        "class": "derived",
        "producing_script": "examples/nla_geometric_features.py",
        "provenance": (
            "pure tensor math over the four capture-roots — per-capture norms, "
            "sparsity and dim-level features, no model load; recorded in "
            "observations/2026-05-13-nla-geometric-deep-dive.md"
        ),
        "inputs": [
            "aggregate_faithfulness.pt",
            "rabbit_haiku_gen_trajectory.pt",
            "forced_continuation.pt",
            "country_concept_vector.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig1-fig5"],
    },
    "pairwise_and_hotdims.pt": {
        "class": "derived",
        "producing_script": "examples/nla_pairwise_and_hotdims.py",
        "provenance": (
            "pure tensor math over the four capture-roots — the 167-capture "
            "pooled h matrix, pairwise cosines and hot-dimension labels; the "
            "hub artifact most atlas figures read. Recorded in "
            "observations/2026-05-13-nla-geometric-deep-dive.md"
        ),
        "inputs": [
            "aggregate_faithfulness.pt",
            "rabbit_haiku_gen_trajectory.pt",
            "forced_continuation.pt",
            "country_concept_vector.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig1-fig4, fig7-fig37 (hub artifact)"],
    },
    "sink_removed_atlas.pt": {
        "class": "derived",
        "producing_script": "examples/nla_sink_removed_atlas.py",
        "provenance": (
            "pure tensor math — the pooled h matrix with the 7 hand-identified "
            "universal-sink dims zeroed, plus the re-derived cosines and PCA; "
            "recorded in observations/2026-05-13-nla-sink-removed-atlas.md"
        ),
        "inputs": ["pairwise_and_hotdims.pt", "geometric_features.pt"],
        "requires_model": "none",
        "consumers": ["fig7", "fig8", "fig9", "fig10", "fig11"],
    },
    "mid_seq_compare.pt": {
        "class": "derived",
        "producing_script": "examples/nla_mid_seq_vocab_atlas_compare.py",
        "provenance": (
            "pure tensor math — mid-sequence h's projected onto the "
            "end-of-prompt 23-axis mean-contrast basis (the cross-protocol "
            "null result); recorded in observations/"
            "2026-05-14-nla-mid-seq-vocab-atlas-null-result.md"
        ),
        "inputs": [
            "vocab_atlas.pt",
            "mid_seq_vocab_atlas.pt",
            "pairwise_and_hotdims.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig31", "fig32"],
    },
    "mid_seq_native_compare.pt": {
        "class": "derived",
        "producing_script": "examples/nla_mid_seq_native_compare.py",
        "provenance": (
            "pure tensor math — a mid-sequence-NATIVE mean-contrast basis built "
            "by the same recipe, and the in-protocol signal lift against the "
            "end-of-prompt basis; recorded in observations/"
            "2026-05-14-nla-mid-seq-native-discriminants.md"
        ),
        "inputs": [
            "vocab_atlas.pt",
            "mid_seq_vocab_atlas.pt",
            "pairwise_and_hotdims.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig33", "fig34"],
    },
    "concept_arithmetic_atlas.pt": {
        "class": "derived",
        "producing_script": "examples/nla_concept_arithmetic_atlas.py",
        "provenance": (
            "vector-arithmetic combinations of vocab-atlas anchor h's "
            "(a - b + c style), each result AV-decoded to text; recorded in "
            "observations/2026-05-14-nla-concept-arithmetic-atlas.md"
        ),
        "inputs": ["vocab_atlas.pt"],
        "requires_model": "av",
        "consumers": ["fig35", "AUDIT 17"],
    },
    "dense_interp_near_pivot.pt": {
        "class": "derived",
        "producing_script": "examples/nla_dense_interp_near_pivot.py",
        "provenance": (
            "the flipbook's cached h_A/h_B re-interpolated at 10x resolution "
            "near the flagged pivot (25 dense steps in t in [0.395, 0.455], "
            "5 sparse context points), each AV-decoded; recorded in "
            "observations/2026-05-15-nla-dense-interp-near-pivot.md"
        ),
        "inputs": ["interpolation_flipbook.pt"],
        "requires_model": "av",
        "consumers": ["fig36", "fig37", "plateau_attractor_test.pt", "AUDIT 18"],
    },
    "plateau_attractor_test.pt": {
        "class": "derived",
        "producing_script": "examples/nla_plateau_attractor_test.py",
        "provenance": (
            "AR re-encoding of the dense-interp plateau midpoint's AV text back "
            "to h, with the round-trip cosine and per-anchor margins; recorded "
            "in observations/2026-05-15-nla-plateau-attractor-strength.md"
        ),
        "inputs": ["dense_interp_near_pivot.pt"],
        "requires_model": "ar",
        "consumers": ["AUDIT 19 (no figure — round-trip validation only)"],
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_fields(name: str) -> dict[str, Any]:
    """The provenance fields for `name`, derived from META — everything the
    manifest records EXCEPT the disk-derived sha256/size_bytes. Shared by the
    writer and the `--check` drift detector, so editing META without
    regenerating MANIFEST.json is caught."""
    m = META[name]
    return {
        "class": m["class"],
        "producing_script": m["producing_script"],
        "producing_command": (f"python {m['producing_script']}"),
        "provenance": m["provenance"],
        "inputs": m["inputs"],
        "requires_model": m["requires_model"],
        "consumers": m["consumers"],
    }


def _disk_vs_expected(
    expected: set[str], on_disk: set[str]
) -> tuple[list[str], list[str]]:
    """(missing, extra): names expected but absent on disk, and on disk but
    unexpected. Shared by the writer (which hard-fails on either) and the
    --check detector (which collects them)."""
    return sorted(expected - on_disk), sorted(on_disk - expected)


def build_entries() -> list[dict[str, Any]]:
    on_disk = sorted(p.name for p in DATA_DIR.glob("*.pt"))
    missing, extra = _disk_vs_expected(set(META), set(on_disk))
    if missing:
        raise SystemExit(
            f"ERROR: manifest metadata names files absent on disk: {missing}"
        )
    if extra:
        raise SystemExit(f"ERROR: data dir has .pt files with no metadata: {extra}")
    stubs = [n for n in on_disk if is_lfs_pointer(DATA_DIR / n)]
    if stubs:
        raise SystemExit(
            f"ERROR: {len(stubs)} file(s) are git-LFS pointer stubs, not real "
            f"artifacts — hashing them would write bogus checksums. "
            f"{_LFS_STUB_NOTE}. Stubs: {stubs}"
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


def _doc_header() -> dict[str, Any]:
    """The manifest's non-per-file fields. Shared by the writer and the
    `--check` drift detector, so editing this prose (or MODEL_PIN) without
    regenerating MANIFEST.json is caught the same way a META edit is."""
    return {
        "arc": "nla-verbalizer",
        "description": (
            "Raw NLA capture/derived artifacts (layer-20 Qwen2.5-7B-Instruct h "
            "vectors and round-trip products). Committed via git-LFS so figures "
            "and nla_audit_findings.py reproduce from a clean clone."
        ),
        "trust_note": (
            "Code-execution statement only, not a licensing statement. Loaded "
            "with torch.load(weights_only=False) (pickle, executes on load). "
            "Safe on this copy: locally-generated tensor dumps produced by the "
            "committed capture scripts. On any copy you did not produce, verify "
            "sha256 with `--check` before loading."
        ),
        "licensing_note": (
            "Licensing, model provenance and the personal-data assessment for "
            "these artifacts: research/arcs/01_nla-verbalizer/data/LICENSE-DATA.md. "
            "One item there is open — the kitft NLA pair's upstream licence was "
            "never recorded at capture time."
        ),
        "model_pin": MODEL_PIN,
    }


def write_manifest() -> None:
    entries = build_entries()
    doc: dict[str, Any] = {
        **_doc_header(),
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
    on_disk = {p.name for p in DATA_DIR.glob("*.pt")}
    problems: list[str] = []
    missing, extra = _disk_vs_expected(set(recorded), on_disk)
    problems += [f"missing on disk: {name}" for name in missing]
    problems += [f"on disk but not in manifest: {name}" for name in extra]
    # Doc-level drift: an edit to the description / trust / licensing prose or
    # the model pin that was never regenerated into the committed manifest.
    for field, expected_val in _doc_header().items():
        if doc.get(field) != expected_val:
            problems.append(
                f"doc-level drift: {field}\n"
                f"    manifest={doc.get(field)!r}\n"
                f"    script  ={expected_val!r}"
            )
    for name in sorted(set(recorded) & on_disk):
        # An unpopulated git-LFS clone leaves pointer text in place of the
        # payload. Hashing it yields a mismatch that reads as corruption; name
        # the actual state instead so the reader runs the right command.
        if is_lfs_pointer(DATA_DIR / name):
            problems.append(f"{_LFS_STUB_NOTE}: {name}")
            continue
        actual = sha256_of(DATA_DIR / name)
        if actual != recorded[name]["sha256"]:
            problems.append(
                f"sha256 drift: {name}\n    manifest={recorded[name]['sha256']}\n    on-disk ={actual}"
            )
        # Provenance drift: a META edit (reclassify, corrected inputs/consumers)
        # that was never regenerated into the committed manifest.
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


def main(argv: list[str] | None = None) -> int:
    """Explicit-mode CLI. A bare invocation (or a typo, or `--help`) must never
    rewrite the committed manifest — the writer runs only under `--write`."""
    ap = argparse.ArgumentParser(
        prog="nla_data_manifest.py",
        description=(
            "Generate or verify research/arcs/01_nla-verbalizer/data/MANIFEST.json. "
            "Exactly one mode flag is required."
        ),
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify sha256 + provenance metadata against the committed manifest; exit 1 on drift",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="(re)write MANIFEST.json from the files on disk + META",
    )
    args = ap.parse_args(argv)
    if args.check:
        return check_manifest()
    if args.write:
        write_manifest()
        return 0
    ap.print_usage(sys.stderr)
    print(
        "nla_data_manifest.py: no mode given — pass --check to verify or "
        "--write to regenerate. Nothing was written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
