"""Generate / verify the raw-dataset manifest for the embedding-atlas arc.

The arc's `.pt` artifacts live (committed, git-LFS) under
`research/arcs/03_embedding-atlas/data/`. This script writes a checksummed
`MANIFEST.json` next to them recording, per file: sha256, size, class
(capture-root | derived), producing script/command, inputs, model
requirement, and consumers.

Two modes, both explicit — neither is the default, so a bare invocation (or a
mistyped flag) can never silently overwrite the committed manifest:
    python examples/emb_data_manifest.py --check    # verify, exit 1 on drift
    python examples/emb_data_manifest.py --write    # (re)write MANIFEST.json

ARC DEVIATION NOTE (vs ARC_PROCESS § "Raw data is a deliverable"): the true
capture-root of this arc is the published Qwen2.5-7B-Instruct weight matrix
pair (W_E, lm_head), pinned to HF revision a09a35458c702b33eeacc393d103063234e8bc28
— 2x 1.09 GB, bit-reproducible from the public snapshot, NOT committed here.
The files below classed "capture-root" are the slices/statistics emb_capture.py
cuts from those matrices in one model load; everything the figures and audit
consume IS committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _emb_artifacts import DATA as DATA_DIR, LFS_HINT, is_lfs_pointer

_REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = DATA_DIR / "MANIFEST.json"

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

META: dict[str, dict[str, Any]] = {
    # ---- capture-roots: cut from the pinned model weights in one load -------
    "emb_battery_vectors.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_capture.py",
        "inputs": [],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": [
            "emb_category_stats.pt",
            "emb_pair_directions.pt",
            "fig8",
            "fig9",
            "AUDIT 1/4/5",
        ],
    },
    "emb_global_stats.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_capture.py",
        "inputs": [],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": [
            "emb_category_stats.pt",
            "fig1-fig4",
            "AUDIT 1/2/3/4",
        ],
    },
    "emb_random_baseline.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_capture.py",
        "inputs": [],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": ["(reserved for follow-up baselines; seed-pinned)"],
    },
    "emb_neighbor_probes.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_capture.py",
        "inputs": [],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": ["emb_neighbors_report.py (text report)", "AUDIT 6"],
    },
    "emb_fullvocab_stats.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_fullvocab_stats.py",
        "inputs": [],
        "requires_model": f"qwen-base@{REVISION[:8]} (via the S0 W_E/W_U cache dump)",
        "consumers": [
            "emb_fullvocab_analysis.pt",
            "emb_structural_block.pt",
            "fig11-fig13",
            "AUDIT 8",
        ],
    },
    # ---- derived: regenerable from other .pt by a committed script ----------
    "emb_fullvocab_analysis.pt": {
        "class": "derived",
        "producing_script": "examples/emb_fullvocab_analyze.py",
        "inputs": ["emb_fullvocab_stats.pt"],
        "requires_model": f"qwen-base@{REVISION[:8]} (tokenizer + S0 dump + cached corr matrix)",
        "consumers": ["fig14", "emb_structural_block.pt", "AUDIT 8"],
    },
    "emb_structural_block.pt": {
        "class": "derived",
        "producing_script": "examples/emb_structural_block.py",
        "inputs": ["emb_fullvocab_analysis.pt", "emb_fullvocab_stats.pt"],
        "requires_model": f"qwen-base@{REVISION[:8]} (tokenizer + S0 dump)",
        "consumers": ["fig15", "AUDIT 8"],
    },
    "emb_de_cosine_check.pt": {
        "class": "derived",
        "producing_script": "examples/emb_de_cosine_check.py",
        "inputs": [
            "emb_WE_bf16.pt (cache-only W_E dump)",
            "emb_structural_block.pt (block dims + id validation)",
            "emb_fullvocab_analysis.pt (census id validation)",
        ],
        "requires_model": f"qwen-base@{REVISION[:8]} (S0 dump for W_E cache)",
        "consumers": ["AUDIT 11", "README finding #6", "2026-07-21 observation"],
    },
    "emb_trace_weightmap.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_trace_capture.py",
        "inputs": ["emb_fullvocab_analysis.pt (block dims)"],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": ["emb_trace_analysis.pt", "T1 reader-head findings"],
    },
    "emb_trace_layers.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_trace_capture.py",
        "inputs": ["emb_fullvocab_analysis.pt (block dims)"],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": ["emb_trace_analysis.pt", "T0 census / P2 persistence findings"],
    },
    "emb_trace_components.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_trace_components.py",
        "inputs": ["emb_fullvocab_analysis.pt (block dims)"],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": ["fig16", "fig17", "fig18", "AUDIT 9", "T1.5 carrier findings"],
    },
    "emb_trace_analysis.pt": {
        "class": "derived",
        "producing_script": "examples/emb_trace_analyze.py",
        "inputs": ["emb_trace_layers.pt", "emb_trace_weightmap.pt"],
        "requires_model": "none",
        "consumers": [
            "fig18",
            "AUDIT 9",
            "AUDIT 10 (reader cross-ref)",
            "T0/T1/P2 observation",
        ],
    },
    "emb_trace_attention.pt": {
        "class": "capture-root",
        "producing_script": "examples/emb_trace_attention.py",
        "inputs": ["emb_fullvocab_analysis.pt (block dims)"],
        "requires_model": f"qwen-base@{REVISION[:8]}",
        "consumers": [
            "emb_trace_attention_analyze.py (P1a/P1c/P1d)",
            "fig19",
            "fig20",
            "fig21",
            "AUDIT 10",
        ],
    },
    "emb_category_stats.pt": {
        "class": "derived",
        "producing_script": "examples/emb_category_stats.py",
        "inputs": ["emb_battery_vectors.pt", "emb_global_stats.pt"],
        "requires_model": "none",
        "consumers": ["fig5", "fig6", "fig7", "fig8", "AUDIT 5"],
    },
    "emb_pair_directions.pt": {
        "class": "derived",
        "producing_script": "examples/emb_pair_directions.py",
        "inputs": ["emb_battery_vectors.pt"],
        "requires_model": "none",
        "consumers": ["fig10", "AUDIT 7"],
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_fields(name: str) -> dict[str, Any]:
    meta = META[name]
    return {
        "class": meta["class"],
        "producing_script": meta["producing_script"],
        "producing_command": f"python {meta['producing_script']}",
        "inputs": meta["inputs"],
        "requires_model": meta["requires_model"],
        "consumers": meta["consumers"],
    }


def on_disk_total_bytes() -> int:
    """Summed size of the committed .pt artifacts, straight off the filesystem
    — the value `total_size_bytes` in the manifest must equal."""
    return sum(p.stat().st_size for p in DATA_DIR.glob("*.pt"))


def build(generated: str) -> dict[str, Any]:
    """Assemble the manifest document. `generated` is carried through rather
    than stamped fresh in --check mode, so a re-verification of an unchanged
    directory never reads as drift."""
    files = sorted(p for p in DATA_DIR.glob("*.pt"))
    stubs = [p.name for p in files if is_lfs_pointer(p)]
    if stubs:
        raise SystemExit(
            f"ERROR: {len(stubs)} file(s) are git-LFS pointer stubs, not real "
            f"artifacts — hashing them would produce bogus checksums. "
            f"Run `{LFS_HINT}`, then retry. Stubs: {stubs}"
        )
    entries: dict[str, Any] = {}
    for p in files:
        if p.name not in META:
            print(
                f"ERROR: {p.name} present in data/ but missing from META",
                file=sys.stderr,
            )
            sys.exit(1)
        entries[p.name] = {
            "sha256": sha256_of(p),
            "size_bytes": p.stat().st_size,
            **_metadata_fields(p.name),
        }
    for name in META:
        if name not in entries:
            print(f"ERROR: {name} in META but missing from {DATA_DIR}", file=sys.stderr)
            sys.exit(1)
    return {
        "arc": "embedding-atlas",
        "base_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": REVISION,
        "generated": generated,
        "total_files": len(entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries.values()),
        "files": entries,
    }


def check() -> int:
    if not MANIFEST.exists():
        print("MANIFEST CHECK: MANIFEST.json missing")
        return 1
    committed = json.loads(MANIFEST.read_text())
    generated = committed.get("generated")
    if not isinstance(generated, str) or not generated:
        print(
            "MANIFEST CHECK: DRIFT — no `generated` date field; "
            "regenerate with: python examples/emb_data_manifest.py --write"
        )
        return 1
    doc = build(generated)

    # Independent of the per-file sha256 comparison: the recorded total must
    # equal what is actually on disk right now.
    total = on_disk_total_bytes()
    if committed.get("total_size_bytes") != total:
        print(
            f"MANIFEST CHECK: DRIFT — total_size_bytes "
            f"{committed.get('total_size_bytes')!r} != on-disk {total}"
        )
        return 1

    if committed == doc:
        print(
            f"MANIFEST CHECK: OK  ({doc['total_files']} files, sha256 + metadata "
            f"match, {total / 1e6:.1f} MB, generated {generated})"
        )
        return 0
    print(
        "MANIFEST CHECK: DRIFT detected — regenerate with: "
        "python examples/emb_data_manifest.py --write"
    )
    for name in sorted(set(committed.get("files", {})) | set(doc["files"])):
        a = committed.get("files", {}).get(name)
        b = doc["files"].get(name)
        if a != b:
            print(f"  differs/missing: {name}")
    for key in ("arc", "base_id", "revision", "total_files"):
        if committed.get(key) != doc[key]:
            print(f"  differs: {key}: {committed.get(key)!r} != {doc[key]!r}")
    return 1


def write() -> int:
    doc = build(dt.date.today().isoformat())
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"wrote {MANIFEST.relative_to(_REPO_ROOT)}  ({doc['total_files']} files, "
        f"{doc['total_size_bytes'] / 1e6:.1f} MB, generated {doc['generated']})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate or verify research/arcs/03_embedding-atlas/data/MANIFEST.json. "
            "One of --check / --write is required: writing is never the default, "
            "so no invocation can overwrite the committed manifest by accident."
        )
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="verify the manifest; exit 1 on drift"
    )
    mode.add_argument(
        "--write", action="store_true", help="(re)write MANIFEST.json from disk"
    )
    args = ap.parse_args(argv)
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
