"""Shared artifact-path resolution for the jspace arc scripts.

Every jspace render script and the audit read the same derived ``.pt``/JSON
artifacts, which live in two places under
``research/arcs/04_jspace/data/``:

- ``data/*`` — the committed deliverable (Git-LFS tracked, sha256-registered
  in ``data/MANIFEST.json``); the clean-clone source of truth.
- ``data/cache/*`` — a byte-identical working mirror. Gitignored working
  space, EXCEPT the full fitted ``jlens_*`` lens tensors: per the amended
  design Decision 4, three of them (1.5B bf16 wikitext, 1.5B bf16 c4en, 7B
  nf4) are LFS-committed there behind an opt-in fetch, excluded from default
  pulls. Get them with::

      git lfs pull --include="research/arcs/04_jspace/data/cache/**" --exclude=""

  The two 1.5B nf4 lenses (n100, n500) are not committed — issue #47.

`resolve` prefers the committed ``data/`` copy so figures and the audit both
reproduce from a clean clone (after ``git lfs pull``), falling back to
``cache/`` when a name is not promoted into ``data/`` — the full fitted lens
tensors, which stay under ``cache/``. This mirrors the audit's ``_resolve`` so
the two never diverge.
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
    ``cache/`` copy. Existence is not asserted here; callers report a missing
    artifact in their own idiom — a lens under ``cache/`` is absent until the
    opt-in LFS fetch above (or, for the nf4 pair, until issue #47)."""
    d = DATA / name
    return d if d.exists() else CACHE / name
