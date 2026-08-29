"""Shared artifact-path resolver for the embedding-atlas arc example scripts.

Two artifact locations:
  * CACHE — .cache/emb_artifacts/ — the gitignored working cache the
    capture/derive scripts write to during local development.
  * DATA  — research/arcs/03_embedding-atlas/data/ — the committed git-LFS copy a
    clean clone has (see research/ARC_PROCESS.md § "Raw data is a deliverable").

Reads prefer the cache (so a fresh local re-capture is picked up immediately)
and fall back to the committed copy (so figures re-render and the audit replays
from a clean clone with no manual copy-back). Writes always go to the cache;
promote a new/changed artifact to DATA with `emb_data_manifest.py --write`
when committing.

Self-locating from this file, so callers work from any CWD.

This module is a deliberate copy of `_nla_artifacts.py` with the three path
constants swapped (the repo's documented copy-as-template idiom). A third arc
needing the same resolver should refactor the pattern into a shared factory
instead of making a third copy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / ".cache" / "emb_artifacts"
DATA = _REPO_ROOT / "research" / "arcs" / "03_embedding-atlas" / "data"
FIGURES = (
    _REPO_ROOT / "research" / "arcs" / "03_embedding-atlas" / "observations" / "figures"
)


#: First bytes of a git-LFS pointer file (the text stub a clone without LFS
#: leaves in place of the real object). See the git-LFS pointer spec.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

#: Recovery command for an unfetched stub — every message about one says this.
LFS_HINT = "git lfs install && git lfs pull"


def is_lfs_pointer(path: Path) -> bool:
    """True when `path` is an unfetched git-LFS pointer stub (~134 bytes of
    text) rather than the real artifact. A default clone of this repo without
    git-LFS gets stubs; `torch.load` on one dies with an opaque
    `UnpicklingError: invalid load key, 'v'`, so callers sniff first and say
    what to run instead."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
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
    """Resolve an input artifact for an unconditional load; raise if missing."""
    p = find_artifact(name)
    if p is None:
        raise FileNotFoundError(
            f"{name!r} not found in {CACHE} or {DATA}. Run the capture step, "
            f"or `git lfs pull` to fetch the committed copy."
        )
    return p


def load_artifact(name: str) -> Any:
    """torch.load an input artifact (cache→committed via read_artifact). The
    single place the `weights_only=False` (pickle) trust decision lives — safe
    for these locally-generated tensor dumps; verify sha256 with
    emb_data_manifest.py --check before loading an untrusted copy. Raises
    FileNotFoundError if the artifact is in neither location, and RuntimeError
    if the resolved file is an unfetched git-LFS pointer stub. torch is imported
    lazily so non-load consumers (e.g. emb_data_manifest importing DATA) stay
    torch-free."""
    import torch

    p = read_artifact(name)
    if is_lfs_pointer(p):
        raise RuntimeError(
            f"{name!r} at {p} is an unfetched git-LFS pointer stub, not the "
            f"artifact. Run `{LFS_HINT}` in the repo, then retry."
        )
    return torch.load(p, weights_only=False)


def write_artifact(name: str) -> Path:
    """Resolve an output artifact path in the working cache (created if needed).
    Writes never target the committed DATA dir — promote with
    emb_data_manifest.py --write when committing."""
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
            "  Re-capture all inputs, or clear .cache/emb_artifacts/, "
            "to avoid a mixed-epoch artifact.",
            file=sys.stderr,
        )
