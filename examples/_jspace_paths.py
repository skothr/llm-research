"""Shared artifact-path resolution for the jspace arc scripts.

Every jspace render script and the audit read the same derived ``.pt``/JSON
artifacts, which live in two places under
``research/arcs/04_jspace/data/``:

- ``data/*`` — the committed deliverable (Git-LFS tracked, sha256-registered
  in ``data/MANIFEST.json``); the clean-clone source of truth.
- ``data/cache/*`` — a byte-identical, gitignored working mirror.

`resolve` prefers the committed ``data/`` copy so figures and the audit both
reproduce from a clean clone (after ``git lfs pull``), falling back to the
working ``cache/`` when a name is not promoted into ``data/`` (the full fitted
lens tensors stay cache-only per design Decision 4). This mirrors the audit's
``_resolve`` so the two never diverge.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARC = REPO_ROOT / "research" / "arcs" / "04_jspace"
DATA = ARC / "data"
CACHE = DATA / "cache"
FIGDIR = ARC / "observations" / "figures"


def resolve(name: str) -> Path:
    """Return the committed ``data/`` copy of ``name`` if present, else the
    gitignored ``cache/`` copy. Existence is not asserted here; callers report
    a missing artifact in their own idiom."""
    d = DATA / name
    return d if d.exists() else CACHE / name
