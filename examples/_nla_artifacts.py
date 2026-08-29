"""Shared artifact-path resolver for the NLA arc example scripts.

Two artifact locations:
  * CACHE — .cache/nla_artifacts/ — the gitignored working cache the
    capture/derive scripts write to during local development.
  * DATA  — research/arcs/01_nla-verbalizer/data/ — the committed git-LFS copy a
    clean clone has (see research/ARC_PROCESS.md § "Raw data is a deliverable").

Reads prefer the cache (so a fresh local re-capture is picked up immediately)
and fall back to the committed copy (so figures re-render and the audit replays
from a clean clone with no manual copy-back). Writes always go to the cache;
promote a new/changed artifact to DATA with `nla_data_manifest.py` when
committing.

Self-locating from this file, so callers work from any CWD. Mirrors the
resolution `nla_audit_findings.py` does at the directory level.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / ".cache" / "nla_artifacts"
DATA = _REPO_ROOT / "research" / "arcs" / "01_nla-verbalizer" / "data"
FIGURES = (
    _REPO_ROOT / "research" / "arcs" / "01_nla-verbalizer" / "observations" / "figures"
)


_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"

#: Recovery command for a repo cloned without git-LFS populated. Single
#: definition so every stub report in the arc names the same fix — the wording
#: is pinned by research/ARC_PROCESS.md § "Raw data is a deliverable".
LFS_RECOVERY_HINT = "run git lfs install && git lfs pull"

#: The full one-line stub report: what is wrong, then how to fix it. Used by
#: nla_data_manifest.py (--check / --write) and nla_audit_findings.py.
LFS_STUB_NOTE = f"LFS pointer stub — {LFS_RECOVERY_HINT}"


def is_lfs_pointer(path: Path) -> bool:
    """True when `path` is a git-LFS pointer stub rather than the real payload.

    A clone made without git-LFS (or with `GIT_LFS_SKIP_SMUDGE=1`) leaves every
    `data/*.pt` as a ~130-byte text pointer whose first line is
    `version https://git-lfs.github.com/spec/v1`. Detecting it lets callers
    report the actionable state instead of letting torch.load /
    hashlib.sha256 report an unpickling error or a phantom sha256 drift.
    Mirrors examples/jspace_audit_findings.py's stub check.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_MAGIC)).startswith(_LFS_POINTER_MAGIC)
    except OSError:
        return False


def find_artifact(name: str) -> Path | None:
    """Resolve an input artifact: live cache first, then the committed copy.
    Returns None if neither has it — use for resume / optional reads guarded
    by an `if`."""
    cached = CACHE / name
    if cached.exists():
        return cached
    committed = DATA / name
    if committed.exists():
        return committed
    return None


def read_artifact(name: str) -> Path:
    """Resolve an input artifact for an unconditional load; raise if missing or
    if it is an unpopulated git-LFS pointer stub (a distinct, actionable
    failure — the file exists but holds pointer text, not tensors)."""
    p = find_artifact(name)
    if p is None:
        raise FileNotFoundError(
            f"{name!r} not found in {CACHE} or {DATA}. Run the capture step, "
            f"or {LFS_RECOVERY_HINT} to fetch the committed copy."
        )
    if is_lfs_pointer(p):
        raise FileNotFoundError(
            f"{name!r} at {p} is a git-LFS pointer stub, not the real artifact "
            f"— the clone never populated LFS objects. To recover, "
            f"{LFS_RECOVERY_HINT}"
        )
    return p


def load_artifact(name: str) -> Any:
    """torch.load an input artifact (cache→committed via read_artifact). The
    single place the `weights_only=False` (pickle) trust decision lives — safe
    for these locally-generated tensor dumps; verify sha256 with
    nla_data_manifest.py --check before loading an untrusted copy. Raises
    FileNotFoundError if the artifact is in neither location, or if the file
    found is an unpopulated git-LFS pointer stub. torch is imported
    lazily so non-load consumers (e.g. nla_data_manifest importing DATA) stay
    torch-free."""
    import torch

    return torch.load(read_artifact(name), weights_only=False)


def write_artifact(name: str) -> Path:
    """Resolve an output artifact path in the working cache (created if needed).
    Writes never target the committed DATA dir — promote with
    nla_data_manifest.py when committing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def warn_if_mixed_sources(names: list[str]) -> None:
    """Warn (stderr) when the given input artifacts resolve from BOTH the live
    cache and the committed copy. That split means a derived artifact would
    blend a locally re-captured input with older committed inputs — a silent
    mixed-epoch mongrel. Multi-input derive scripts call this before stacking
    their inputs; re-capture all inputs (or clear the cache) to clear it."""
    sources: dict[str, str] = {}
    for name in names:
        p = find_artifact(name)
        if p is not None:
            sources[name] = "cache" if p.parent == CACHE else "committed"
    if len(set(sources.values())) > 1:
        cached = sorted(n for n, s in sources.items() if s == "cache")
        committed = sorted(n for n, s in sources.items() if s == "committed")
        print(
            "WARNING: inputs span two sources — the derived output would blend epochs:\n"
            f"  re-captured (cache):     {cached}\n"
            f"  older (committed data/): {committed}\n"
            "  Re-capture all inputs, or clear .cache/nla_artifacts/, "
            "to avoid a mixed-epoch artifact.",
            file=sys.stderr,
        )
