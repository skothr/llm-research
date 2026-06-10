"""Generate / verify the raw-dataset manifest for the embedding-atlas arc.

The arc's `.pt` artifacts live (committed, git-LFS) under
`research/arcs/embedding-atlas/data/`. This script writes a checksummed
`MANIFEST.json` next to them recording, per file: sha256, size, class
(capture-root | derived), producing script/command, inputs, model
requirement, and consumers.

Two modes:
    python examples/emb_data_manifest.py            # (re)write MANIFEST.json
    python examples/emb_data_manifest.py --check     # verify, exit 1 on drift

ARC DEVIATION NOTE (vs ARC_PROCESS § "Raw data is a deliverable"): the true
capture-root of this arc is the published Qwen2.5-7B-Instruct weight matrix
pair (W_E, lm_head), pinned to HF revision a09a35458c702b33eeacc393d103063234e8bc28
— 2x 1.09 GB, bit-reproducible from the public snapshot, NOT committed here.
The files below classed "capture-root" are the slices/statistics emb_capture.py
cuts from those matrices in one model load; everything the figures and audit
consume IS committed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _emb_artifacts import DATA as DATA_DIR

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
    "emb_category_stats.pt": {
        "class": "derived",
        "producing_script": "examples/emb_category_stats.py",
        "inputs": ["emb_battery_vectors.pt", "emb_global_stats.pt"],
        "requires_model": "none",
        "consumers": ["fig5", "fig6", "fig7", "AUDIT 5"],
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


def build() -> dict[str, Any]:
    files = sorted(p for p in DATA_DIR.glob("*.pt"))
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
        "total_files": len(entries),
        "files": entries,
    }


def main() -> int:
    doc = build()
    if "--check" in sys.argv:
        if not MANIFEST.exists():
            print("MANIFEST.json missing")
            return 1
        committed = json.loads(MANIFEST.read_text())
        if committed == doc:
            mb = sum(e["size_bytes"] for e in doc["files"].values()) / 1e6
            print(
                f"MANIFEST CHECK: OK  ({doc['total_files']} files, sha256 + metadata match, {mb:.1f} MB)"
            )
            return 0
        print(
            "MANIFEST CHECK: DRIFT detected — regenerate with: python examples/emb_data_manifest.py"
        )
        for name in sorted(set(committed.get("files", {})) | set(doc["files"])):
            a = committed.get("files", {}).get(name)
            b = doc["files"].get(name)
            if a != b:
                print(f"  differs/missing: {name}")
        return 1
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    mb = sum(e["size_bytes"] for e in doc["files"].values()) / 1e6
    print(
        f"wrote {MANIFEST.relative_to(_REPO_ROOT)}  ({doc['total_files']} files, {mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
