"""Checks that hold for every arc whose scripts live in research/arcs/<slug>/scripts/.

They cover what the manifest generators' own `--check` cannot: `--check`
compares each MANIFEST.json with the META it was generated from, so a stale
script path regenerated with `--write` passes it. Neither needs the LFS data
or a model.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = sorted(REPO_ROOT.glob("research/arcs/*/data/MANIFEST.json"))
SCRIPTS = sorted(REPO_ROOT.glob("research/arcs/*/scripts/*.py"))
SHELL_SCRIPTS = sorted(REPO_ROOT.glob("research/arcs/*/scripts/*.sh"))
# A repo-relative path to a moved script, as it appears in commands, usage
# lines, provenance strings and the constants that write data records.
SCRIPT_PATH = re.compile(r"research/arcs/[\w.-]+/scripts/[\w.-]+\.(?:py|sh)\b")


def _producing_scripts(manifest: Path) -> set[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    files = data.get("files", [])
    entries = files.values() if isinstance(files, dict) else files
    return {
        e["producing_script"]
        for e in entries
        if isinstance(e, dict) and e.get("producing_script")
    }


@pytest.mark.parametrize(
    "manifest", MANIFESTS, ids=[m.relative_to(REPO_ROOT).as_posix() for m in MANIFESTS]
)
def test_manifest_producing_scripts_exist(manifest: Path) -> None:
    missing = sorted(
        p for p in _producing_scripts(manifest) if not (REPO_ROOT / p).is_file()
    )
    assert not missing, f"producing_script names a missing file: {missing}"


def _bare_imports(path: Path) -> set[str]:
    """Top-level module names a script imports, at any depth in the file."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "script", SCRIPTS, ids=[s.relative_to(REPO_ROOT).as_posix() for s in SCRIPTS]
)
def test_moved_script_imports_resolve(script: Path) -> None:
    # A moved script imports its siblings by bare name (they resolve from the
    # script's own directory when it runs by path); anything else must be an
    # installed or standard-library module. find_spec on a top-level name
    # locates it without importing it.
    siblings = {p.stem for p in script.parent.glob("*.py")}
    unresolved = sorted(
        name
        for name in _bare_imports(script)
        if name not in siblings and importlib.util.find_spec(name) is None
    )
    assert not unresolved, f"imports that resolve nowhere: {unresolved}"


_PATH_SOURCES = MANIFESTS + SCRIPTS + SHELL_SCRIPTS


@pytest.mark.parametrize(
    "source",
    _PATH_SOURCES,
    ids=[p.relative_to(REPO_ROOT).as_posix() for p in _PATH_SOURCES],
)
def test_script_paths_named_in_text_exist(source: Path) -> None:
    # Covers what the producing_script test does not: producing_command and
    # provenance strings in a manifest, and in a script its usage lines, its
    # printed hints and the constants it writes into data records.
    named = set(SCRIPT_PATH.findall(source.read_text(encoding="utf-8")))
    missing = sorted(p for p in named if not (REPO_ROOT / p).is_file())
    assert not missing, f"names a script path that does not exist: {missing}"


@pytest.mark.parametrize(
    "script",
    SHELL_SCRIPTS,
    ids=[s.relative_to(REPO_ROOT).as_posix() for s in SHELL_SCRIPTS],
)
def test_shell_runner_repo_root_hop(script: Path) -> None:
    # A shell runner finds the repo root by counting directories up from its
    # own location; the count must match where the script lives.
    hops = re.findall(r"\$\{SCRIPT_DIR\}((?:/\.\.)+)", script.read_text(encoding="utf-8"))
    assert hops, "no ${SCRIPT_DIR}/.. repo-root hop found"
    for hop in hops:
        assert (script.parent / hop.lstrip("/")).resolve() == REPO_ROOT
