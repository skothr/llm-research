"""Generate / verify the raw-dataset manifest for the jspace research arc.

The arc's raw artifacts live (committed, git-LFS) under
`research/arcs/jspace/data/`: the frozen fitting corpus, the fitted `.pt`
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
DATA_DIR = _REPO_ROOT / "research" / "arcs" / "jspace" / "data"
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
    "heldout_prompts_wikitext103_n30.json": {
        "class": "raw",
        "producing_script": "examples/jspace_readout_scan.py",
        "inputs": [],
        "requires_model": "none",
        "provenance": (
            "Salesforce/wikitext wikitext-103-raw-v1 train, records 1001-1030 "
            "under the same >=600-char filter as the fitting corpus; verified "
            "zero overlap with fitting_prompts_wikitext103_n1000.json"
        ),
        "consumers": ["readout_scan_*.pt (held-out evaluation prompts)"],
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
    # ---- derived metrics get added here as gates open ----------------------
    # e.g. "<split_half_stability>.pt" (class derived, inputs the lens .pt)
}

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
            "Raw + derived jspace artifacts: the frozen wikitext-103 fitting "
            "corpus, fitted Jacobian-lens tensors (jlens native format) with "
            "their .config.json provenance sidecars, and derived metric/table "
            "products. Committed via git-LFS so figures and the audit "
            "reproduce from a clean clone. The cache/ subdir is excluded."
        ),
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
