"""Path anchors of scripts that find the repo root by counting parents (#122).

A script that sets `_REPO_ROOT = Path(__file__).resolve().parents[N]` breaks
silently when it moves to a different depth: `N` then names some other
directory, and every path built from it points nowhere. Each row below loads
one such script from its file and checks that its repo-root constant is this
checkout's root and that its data directory exists.

When an arc's scripts move out of `examples/`, add a row per script that
counts parents. `data_attr` is None for a script with no data-dir constant;
for a render script it names the figure directory.
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
ARC03 = "research/arcs/03_embedding-atlas/scripts"
ARC04 = "research/arcs/04_jspace/scripts"

ANCHORS = [
    Anchor(f"{ARC02}/subliminal_step0_decode.py", "_REPO_ROOT", None),
    Anchor(f"{ARC02}/subliminal_audit_findings.py", "_REPO_ROOT", "DATA"),
    Anchor(f"{ARC02}/subliminal_data_manifest.py", "_REPO_ROOT", "DATA_DIR"),
    Anchor(f"{ARC03}/_emb_artifacts.py", "_REPO_ROOT", "DATA"),
    Anchor(f"{ARC03}/emb_data_manifest.py", "_REPO_ROOT", "DATA_DIR"),
    Anchor(f"{ARC04}/_jspace_paths.py", "REPO_ROOT", "DATA"),
    Anchor(f"{ARC04}/jspace_audit_findings.py", "_REPO_ROOT", "DATA"),
    Anchor(f"{ARC04}/jspace_data_manifest.py", "_REPO_ROOT", "DATA_DIR"),
    Anchor(f"{ARC04}/jspace_promote_lens_subset.py", "_REPO_ROOT", "ARC_DATA"),
    Anchor(f"{ARC04}/jspace_rerun_queue.py", "REPO", "CACHE"),
    # The render scripts' directory constant is the figure directory.
    Anchor(f"{ARC04}/jspace_render_corpus_invariance.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_emergence.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_entailed.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_nla_crosstie.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_structure_figures.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_swap_causality.py", "_REPO_ROOT", "FIGDIR"),
    Anchor(f"{ARC04}/jspace_render_trajectory.py", "_REPO_ROOT", "FIGDIR"),
]


def _load(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Scripts import their siblings by bare name, as when run by path. The
    # path entry is undone after the test; so is every sibling module the
    # import pulls in, so later tests do not see a module cached from here.
    monkeypatch.syspath_prepend(str(path.parent))
    for sibling in path.parent.glob("*.py"):
        if sibling.stem not in sys.modules:
            # Setting a key that is absent makes the undo delete it.
            monkeypatch.setitem(sys.modules, sibling.stem, ModuleType(sibling.stem))
            monkeypatch.delitem(sys.modules, sibling.stem)
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


def test_audit_generator_path_names_the_pinned_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The audit hashes GENERATOR_PATH against GENERATOR_SHA256; after a move it
    # must still name the generator beside it, not a missing file.
    audit = _load(REPO_ROOT / f"{ARC02}/subliminal_audit_findings.py", monkeypatch)
    path = audit.GENERATOR_PATH
    assert isinstance(path, Path)
    assert path.resolve() == (REPO_ROOT / f"{ARC02}/subliminal_step0_decode.py")
    assert path.is_file()


def test_rerun_queue_launches_the_fit_script_beside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each queue job runs jspace_fit_lens.py by absolute path; after a move the
    # path must still name the script in the queue's own directory.
    queue = _load(REPO_ROOT / f"{ARC04}/jspace_rerun_queue.py", monkeypatch)
    for job in queue.JOBS:
        script = Path(job.argv()[1])
        assert script.resolve() == REPO_ROOT / f"{ARC04}/jspace_fit_lens.py"
        assert script.is_file()


def test_lens_eval_default_dir_is_the_sibling_jlens_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default hops from this repo's root to ../jacobian-lens; from a linked
    # worktree (.git is a file) it first strips .claude/worktrees/<name>.
    lens_eval = _load(REPO_ROOT / f"{ARC04}/jspace_lens_eval.py", monkeypatch)
    root = REPO_ROOT
    if (root / ".git").is_file() and root.parent.parent.name == ".claude":
        root = root.parents[2]
    expected = root.parent / "jacobian-lens" / "data" / "evaluations"
    assert Path(lens_eval._default_eval_dir()) == expected
