"""Path anchors of scripts that find the repo root by counting parents (#122).

A script that sets `_REPO_ROOT = Path(__file__).resolve().parents[N]` breaks
silently when it moves to a different depth: `N` then names some other
directory, and every path built from it points nowhere. Each row below loads
one such script from its file and checks that its repo-root constant is this
checkout's root and that its data directory exists.

When an arc's scripts move out of `examples/`, add a row per script that
counts parents. `data_attr` is None for a script with no data-dir constant.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class Anchor(NamedTuple):
    script: str  # repo-relative path of the script
    root_attr: str  # module attribute holding the repo root
    data_attr: str | None  # module attribute holding a data dir, if any


ARC02 = "research/arcs/02_subliminal/scripts"

ANCHORS = [
    Anchor(f"{ARC02}/subliminal_step0_decode.py", "_REPO_ROOT", None),
    Anchor(f"{ARC02}/subliminal_audit_findings.py", "_REPO_ROOT", "DATA"),
    Anchor(f"{ARC02}/subliminal_data_manifest.py", "_REPO_ROOT", "DATA_DIR"),
]


def _load(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Scripts import their siblings by bare name, as when run by path.
    monkeypatch.syspath_prepend(str(path.parent))
    name = f"_anchor_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("anchor", ANCHORS, ids=[a.script for a in ANCHORS])
def test_repo_root_and_data_dir_resolve(
    anchor: Anchor, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(REPO_ROOT / anchor.script, monkeypatch)
    root = getattr(module, anchor.root_attr)
    assert isinstance(root, Path)
    assert root.resolve() == REPO_ROOT
    assert (root / "pyproject.toml").is_file()
    if anchor.data_attr is not None:
        data = getattr(module, anchor.data_attr)
        assert isinstance(data, Path)
        assert data.is_dir(), f"{anchor.data_attr} = {data} does not exist"
