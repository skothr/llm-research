"""Generate / verify the raw-dataset manifest for the subliminal research arc.

The arc's committed data lives under `research/arcs/02_subliminal/data/`, one
subdirectory per dataset (today only `step0-owl-neutral-decode/`). The files
are small JSONL/JSON/text, kept in plain git: the `.gitattributes` LFS rule
routes only `research/**/data/*.pt`. This script writes a checksummed
`MANIFEST.json` at the top of `data/` that records, per file: sha256, size,
whether it is a capture-root (written by a run that loads the teacher model)
or a derived artifact (regenerable without a model), the producing script and
command, its inputs, what model it needs, its provenance, and who consumes
it. Entry names are paths relative to `data/`, since the files sit one level
down.

Every committed data file under `data/` is covered, recursively, except the
documentation beside the data (`README.md`, `LICENSE-DATA.md`), the committed
audit transcripts (`audit_*.log`) and `MANIFEST.json` itself, the same set
arcs 01, 03 and 04 leave out. That includes each dataset's own
`manifest.json`: the capture-time provenance record the generator wrote
(sampling, seeds, environment, filter statistics). It stays in place because
`subliminal_audit_findings.py` reads its fields and the arc's prose cites them;
MANIFEST.json pins its bytes like any other file.

Two modes, both explicit — bare invocation prints usage and writes nothing, so
a typo or `--help` can never silently rewrite the committed manifest:
    python examples/subliminal_data_manifest.py --check    # verify, exit 1 on drift
    python examples/subliminal_data_manifest.py --write    # (re)write MANIFEST.json

The `--check` mode is the drift detector: it recomputes every sha256, AND
re-derives each file's provenance fields from META, AND re-derives the
top-level literals from the writer's own constants — comparing all three
against the committed manifest, so silent corruption, a missing/extra file, a
META edit, OR an edit to the top-level prose that was never regenerated into
MANIFEST.json all fail. See `research/ARC_PROCESS.md` § "Raw data is a
deliverable".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "research" / "arcs" / "02_subliminal" / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"

# Top-level files under data/ that are documentation or audit records, not
# data. Arcs 01, 03 and 04 do not checksum these either.
_EXCLUDED_TOPLEVEL = {"MANIFEST.json", "README.md", "LICENSE-DATA.md"}
_EXCLUDED_TOPLEVEL_GLOB = "audit_*.log"

_STEP0 = "step0-owl-neutral-decode"
_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
_OBSERVATION = "observations/2026-05-31-step0-protocol-and-filter.md"

# The capture command, reconstructed from the capture-time manifest.json
# `generation` block (n_per_condition 120, batch_size 16, max_new_tokens 80,
# seed 42, CPU bf16, dataset_id). The command line itself was not recorded,
# and neither was --out-dir.
_STEP0_ARGS = (
    "--n-per-condition 120 --batch-size 16 --max-new-tokens 80 --seed 42 "
    f"--no-4bit --dataset-id {_STEP0}"
)
_STEP0_RUN_FACTS = (
    "Qwen2.5-7B-Instruct teacher, temperature 1.0, seed 42, 120 queries per "
    "condition, captured 2026-05-31T18:35:55Z at repo commit d9c7a428 "
    "(0aff26c8 before the 2026-06-01 history rewrite)."
)
_STEP0_RUN = f"{_STEP0_RUN_FACTS} Full capture-time record: {_STEP0}/manifest.json"

# For the two files whose bytes record the run's environment or were amended
# after capture: the capture command produces a file of the same kind, never
# these exact bytes.
_NOT_BYTE_IDENTICAL = (
    " (produces a file of the same form, not the recorded sha256: the "
    "output reflects the running environment and time, and the committed copy "
    "was amended after capture; see provenance)"
)
_STEP0_CMD = f"python examples/subliminal_step0_decode.py {_STEP0_ARGS}"

# Per-artifact provenance. `requires_model` values: none | qwen-base.
# `class` follows the capture-time manifest.json's own lineage
# (`derived_from`): everything the capture run wrote directly is a
# capture-root; decode_report.json is derived from the two streams files.
META: dict[str, dict[str, Any]] = {
    f"{_STEP0}/owl_raw.jsonl": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_args": _STEP0_ARGS,
        "provenance": (
            "all 120 unfiltered teacher completions under the owl system "
            f"prompt. {_STEP0_RUN}"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": [f"{_STEP0}/owl_streams.jsonl", _OBSERVATION, "AUDIT A/B"],
    },
    f"{_STEP0}/neutral_raw.jsonl": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_args": _STEP0_ARGS,
        "provenance": (
            "all 120 unfiltered teacher completions with no system prompt. "
            f"{_STEP0_RUN}"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": [f"{_STEP0}/neutral_streams.jsonl", _OBSERVATION, "AUDIT A/B"],
    },
    f"{_STEP0}/owl_streams.jsonl": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_args": _STEP0_ARGS,
        "provenance": (
            "the 104 owl completions the ported upstream filter kept, parsed "
            "to integer lists; written by the capture run, and AUDIT B replays "
            f"the filter from owl_raw.jsonl without a model. {_STEP0_RUN}"
        ),
        "inputs": [f"{_STEP0}/owl_raw.jsonl"],
        "requires_model": "qwen-base",
        "consumers": [
            f"{_STEP0}/decode_report.json",
            _OBSERVATION,
            "AUDIT A/B/C",
        ],
    },
    f"{_STEP0}/neutral_streams.jsonl": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_args": _STEP0_ARGS,
        "provenance": (
            "the 109 neutral completions the ported upstream filter kept, "
            "parsed to integer lists; written by the capture run, and AUDIT B "
            f"replays the filter from neutral_raw.jsonl without a model. "
            f"{_STEP0_RUN}"
        ),
        "inputs": [f"{_STEP0}/neutral_raw.jsonl"],
        "requires_model": "qwen-base",
        "consumers": [
            f"{_STEP0}/decode_report.json",
            _OBSERVATION,
            "AUDIT A/B/C",
        ],
    },
    f"{_STEP0}/decode_report.json": {
        "class": "derived",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_command": (
            "no CLI: call decode_test(owl_streams, neutral_streams) from "
            "examples/subliminal_step0_decode.py on the two streams files, no "
            "model (AUDIT C replays it). The original was written at the end "
            f"of the capture run, `{_STEP0_CMD}`"
        ),
        "provenance": (
            "five-scheme owl-lexicon decode of both streams files plus the "
            "two-proportion z-test; written at the end of the capture run, and "
            "AUDIT C replays every scheme from the streams without a model"
        ),
        "inputs": [f"{_STEP0}/owl_streams.jsonl", f"{_STEP0}/neutral_streams.jsonl"],
        "requires_model": "none",
        "consumers": [_OBSERVATION, "AUDIT A/C"],
    },
    f"{_STEP0}/prompts.jsonl": {
        "class": "derived",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_command": (
            "no CLI: replay PromptGenerator(PROMPT_PARAMS) from "
            "examples/subliminal_step0_decode.py under "
            "numpy.random.default_rng(42) for 120 draws (AUDIT D re-runs it)"
        ),
        "provenance": (
            "the 120 seeded queries, index-aligned with *_raw.jsonl; re-derived "
            "post-hoc on 2026-08-17 because the capture run predates the "
            "generator's prompts.jsonl write (823b5e68). That it is the set the "
            "2026-05-31 run consumed is an inference; see data/README.md"
        ),
        "inputs": [],
        "requires_model": "none",
        "consumers": ["Step 1 (prompt, completion) pairs", "AUDIT D"],
    },
    f"{_STEP0}/pip_freeze.txt": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_command": _STEP0_CMD + _NOT_BYTE_IDENTICAL,
        "provenance": (
            "the capture environment's package lockfile; one line redacted in "
            "1ed05dad (machine-specific editable-install URL), so it no longer "
            "matches manifest.json's capture-time environment.pip_freeze_sha256. "
            f"{_STEP0_RUN}"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["AUDIT A"],
    },
    f"{_STEP0}/manifest.json": {
        "class": "capture-root",
        "producing_script": "examples/subliminal_step0_decode.py",
        "producing_command": _STEP0_CMD + _NOT_BYTE_IDENTICAL,
        "provenance": (
            "this file is the capture run's own provenance record "
            "(manifest_version 0.1.0-interim): generation recipe, sampling, "
            "seeds, environment, filter statistics, lineage and licence. "
            "Amended once, the 2026-08-19 git-SHA repoint recorded in "
            f"data/README.md. {_STEP0_RUN_FACTS}"
        ),
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": [_OBSERVATION, "data/LICENSE-DATA.md", "AUDIT A/B/D"],
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    """True when `path` holds a git-LFS pointer stub instead of real content.
    No file here is LFS-routed today; the guard keeps a future `.pt` from
    being hashed as a stub on an LFS-less clone."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _candidate_files() -> list[Path]:
    """Files under data/ as paths relative to data/: the git-tracked set, so an
    untracked editor swap file or scratch log never enters the manifest. Falls
    back to walking the directory, with a warning, when git is unavailable (a
    source copy without a checkout)."""
    try:
        p = subprocess.run(
            ["git", "-C", str(DATA_DIR), "ls-files", "-z", "--", "."],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"WARNING: git ls-files unavailable ({type(exc).__name__}); "
            "listing data/ from the filesystem, so untracked files count too",
            file=sys.stderr,
        )
        return [q.relative_to(DATA_DIR) for q in DATA_DIR.rglob("*") if q.is_file()]
    # A tracked file deleted from the working tree is left out here, so
    # --check reports it as "missing on disk" rather than failing to hash it.
    return [
        Path(os.fsdecode(n))
        for n in p.stdout.split(b"\0")
        if n and (DATA_DIR / os.fsdecode(n)).is_file()
    ]


def _deliverables() -> list[str]:
    """Every data file under data/, recursively, as a path relative to data/,
    minus the top-level documentation, audit transcripts and MANIFEST.json."""
    names: list[str] = []
    for rel in _candidate_files():
        if len(rel.parts) == 1 and (
            rel.name in _EXCLUDED_TOPLEVEL or rel.match(_EXCLUDED_TOPLEVEL_GLOB)
        ):
            continue
        names.append(rel.as_posix())
    return sorted(names)


def _metadata_fields(name: str) -> dict[str, Any]:
    """The provenance fields for `name`, derived from META — everything the
    manifest records EXCEPT the disk-derived sha256/size_bytes. Shared by the
    writer and the `--check` drift detector, so editing META without
    regenerating MANIFEST.json is caught."""
    m = META[name]
    command = m.get("producing_command")
    if command is None:
        args = m.get("producing_args")
        command = (
            f"python {m['producing_script']} {args}"
            if args
            else f"python {m['producing_script']}"
        )
    return {
        "class": m["class"],
        "producing_script": m["producing_script"],
        "producing_command": command,
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
    on_disk = _deliverables()
    missing, extra = _disk_vs_expected(set(META), set(on_disk))
    if missing:
        raise SystemExit(
            f"ERROR: manifest metadata names files absent on disk: {missing}"
        )
    if extra:
        raise SystemExit(f"ERROR: data dir has files with no metadata: {extra}")
    stubs = [n for n in on_disk if _is_lfs_pointer(DATA_DIR / n)]
    if stubs:
        raise SystemExit(
            f"ERROR: {len(stubs)} file(s) are git-LFS pointer stubs, not real "
            f"artifacts — hashing them would write bogus checksums. Run "
            f"`git lfs install && git lfs pull`, then retry. Stubs: {stubs}"
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


# Top-level manifest prose. Module constants so --check can assert the
# committed MANIFEST.json still carries these exact literals.
_DESCRIPTION = (
    "Committed subliminal-arc datasets, one subdirectory per dataset "
    f"(today {_STEP0}/: the Step-0 owl vs neutral number-stream corpus, its "
    "decode report, the post-hoc prompt set, the environment lockfile and the "
    "capture-time provenance record). Small JSONL/JSON/text kept in plain git "
    "so the audit reproduces from a clean clone. Entry filenames are relative "
    "to this data/ directory."
)

_TRUST_NOTE = (
    "Code-execution statement only, not a licensing statement. Every file is "
    "plain text (JSONL, JSON, a pip freeze); nothing here is a pickle, so "
    "loading executes no code. On any copy you did not produce, verify sha256 "
    "with `--check` before relying on it."
)

_LICENSING_NOTE = (
    "Licensing, attribution and the personal-data assessment for these files: "
    "research/arcs/02_subliminal/data/LICENSE-DATA.md. The data is Apache-2.0 "
    "model output; the prompt/filter logic that produced it is an MIT port of "
    "MinhxLe/subliminal-learning @ v1.0.0."
)

_MODEL_PIN: dict[str, Any] = {
    "note": (
        "Model the Step-0 capture-roots were sampled from. The revision is "
        "the one recorded at capture time in "
        f"{_STEP0}/manifest.json (generation.model_revision and "
        "generation.tokenizer_revision). Sampling, seeds, environment and "
        "filter statistics stay in that file."
    ),
    "captured": "2026-05-31T18:35:55Z",
    "models": [
        {
            "role": "teacher",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "revision": _REVISION,
            "purpose": "samples the number continuations (temperature 1.0)",
        }
    ],
}

# The top-level (non-per-file, non-tallied) fields --check re-derives.
_TOPLEVEL_LITERALS: dict[str, Any] = {
    "arc": "subliminal",
    "description": _DESCRIPTION,
    "trust_note": _TRUST_NOTE,
    "licensing_note": _LICENSING_NOTE,
    "model_pin": _MODEL_PIN,
}


def write_manifest() -> None:
    entries = build_entries()
    doc: dict[str, Any] = {
        **_TOPLEVEL_LITERALS,
        "total_files": len(entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "files": entries,
    }
    # Temp file in the same directory plus rename, so an interrupted write
    # never leaves a truncated MANIFEST.json behind.
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, prefix=".MANIFEST.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(doc, indent=2) + "\n")
        # mkstemp creates the file 0600; a committed manifest is world-readable.
        os.chmod(tmp, 0o644)
        os.replace(tmp, MANIFEST)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    kib = doc["total_size_bytes"] / 1024
    print(
        f"wrote {MANIFEST.relative_to(_REPO_ROOT)}  "
        f"({doc['total_files']} files, {kib:.1f} KiB)"
    )


def _load_manifest() -> tuple[dict[str, Any] | None, str]:
    """Parse MANIFEST.json and check the structure --check relies on. Returns
    (doc, "") or (None, reason), so a truncated or hand-broken manifest is
    reported as a FAIL line instead of a traceback."""
    try:
        doc = json.loads(MANIFEST.read_text())
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"cannot read MANIFEST.json: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"MANIFEST.json is not valid JSON: {exc}"
    if not isinstance(doc, dict):
        return None, "MANIFEST.json top level is not an object"
    files = doc.get("files")
    if not isinstance(files, list):
        return None, "MANIFEST.json has no 'files' list"
    for i, e in enumerate(files):
        if not isinstance(e, dict):
            return None, f"files[{i}] is not an object"
        for key, typ in (("filename", str), ("sha256", str), ("size_bytes", int)):
            if not isinstance(e.get(key), typ):
                return None, f"files[{i}] lacks a {typ.__name__} '{key}'"
    return doc, ""


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(
            f"FAIL: {MANIFEST.relative_to(_REPO_ROOT)} does not exist (create it with --write)"
        )
        return 1
    doc, reason = _load_manifest()
    if doc is None:
        print(f"MANIFEST CHECK: FAIL ({reason})")
        return 1
    files: list[dict[str, Any]] = doc["files"]
    recorded = {e["filename"]: e for e in files}
    on_disk = set(_deliverables())
    problems: list[str] = []
    # Duplicate entries: the dict above keeps only the last one, so an earlier
    # (possibly stale) duplicate would otherwise go unverified.
    seen: set[str] = set()
    for e in files:
        if e["filename"] in seen:
            problems.append(f"duplicate entry: {e['filename']}")
        seen.add(e["filename"])
    # Top-level prose drift: an edit to the literals above that was never
    # regenerated into the committed manifest.
    for field, expected in _TOPLEVEL_LITERALS.items():
        if doc.get(field) != expected:
            problems.append(
                f"top-level drift: {field}\n"
                f"    manifest={doc.get(field)!r}\n"
                f"    writer  ={expected!r}"
            )
    # Tallies must match the file list they summarize.
    for field, expected in (
        ("total_files", len(files)),
        ("total_size_bytes", sum(e["size_bytes"] for e in files)),
    ):
        if doc.get(field) != expected:
            problems.append(
                f"top-level drift: {field} (manifest={doc.get(field)!r}, "
                f"files imply {expected!r})"
            )
    missing, extra = _disk_vs_expected(set(recorded), on_disk)
    problems += [f"missing on disk: {name}" for name in missing]
    problems += [f"on disk but not in manifest: {name}" for name in extra]
    problems += [f"META entry not in manifest: {name}" for name in sorted(set(META) - set(recorded))]
    for name in sorted(set(recorded) & on_disk):
        path = DATA_DIR / name
        if _is_lfs_pointer(path):
            problems.append(
                f"git-LFS pointer stub, not content (run `git lfs install && "
                f"git lfs pull`): {name}"
            )
            continue
        actual = sha256_of(path)
        if actual != recorded[name]["sha256"]:
            problems.append(
                f"sha256 drift: {name}\n    manifest={recorded[name]['sha256']}\n    on-disk ={actual}"
            )
        size = path.stat().st_size
        if size != recorded[name]["size_bytes"]:
            problems.append(
                f"size drift: {name} (manifest={recorded[name]['size_bytes']}, "
                f"on-disk={size})"
            )
        # Provenance drift: a META edit that was never regenerated into the
        # committed manifest.
        if name in META:
            for field, expected in _metadata_fields(name).items():
                if recorded[name].get(field) != expected:
                    problems.append(
                        f"metadata drift: {name}.{field}\n"
                        f"    manifest={recorded[name].get(field)!r}\n"
                        f"    META     ={expected!r}"
                    )
        else:
            problems.append(f"no META entry for manifest file: {name}")
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
    """Explicit-mode CLI. A bare invocation (or a typo, or `--help`) must never
    rewrite the committed manifest — the writer runs only under `--write`."""
    ap = argparse.ArgumentParser(
        prog="subliminal_data_manifest.py",
        description=(
            "Generate or verify research/arcs/02_subliminal/data/MANIFEST.json. "
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
        "subliminal_data_manifest.py: no mode given — pass --check to verify or "
        "--write to regenerate. Nothing was written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
