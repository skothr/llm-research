"""Find and rewrite references to `examples/` paths for the layout move (#122).

The #122 reorganization moves every script out of `examples/`. This tool
finds each path-shaped reference to a file there and maps it to the file's
new location. The destination of a file is computed from rules (`RULES`,
`PACKAGE_MODULES`, the `tests/` rule), not from a copied table, and every
tracked file under `examples/` must either map or be listed in `UNMAPPED`.

What counts as a reference: `examples/<name>` where `examples/` starts the
path, preceded by the start of the text or a character that cannot be part
of an identifier or path (whitespace, a quote, a backtick, `(`, `[`, `=`,
`:`, ...). `jlens.examples`, `{examples}`, `--examples` and `examples_dir`
are never matched. A relative link such as `../../../examples/nla_x.py` is
resolved against the referring file and rewritten to the new relative path;
a repo-root path `examples/x.py` stays repo-root relative. Module stems
(`examples/_jspace_paths.resolve`) map like the file they name. Python
import statements are not this tool's job.

Each reference falls into one class:
- auto: rewritable, destination known. It is "ready" when the destination
  file exists on disk (the move has happened) and "pending" otherwise.
- manual: a glob or brace form, a bare `examples/` or `examples/tests/`
  directory, an unknown or unmapped file name, a path with another prefix
  before `examples/`, or a relative path that does not resolve to this
  repo's `examples/`. A person edits these.
- hashed-record: any reference inside a file whose bytes a hash or a
  generator pins (`HASHED_RECORDS`). The arc PRs edit these by hand and
  re-pin, or regenerate them.
- excluded: verbatim run logs, archives and this tool's own files
  (`EXCLUDED_PREFIXES`, `EXCLUDED_FILES`). Never rewritten.

`--apply` rewrites only ready auto references, so it is safe to run after
each move PR. It never edits an excluded file, and it edits a hashed-record
file only when that file is named explicitly on the command line together
with `--include-records`. Bytes outside the rewritten paths are kept as they
are (line endings included; nothing is reflowed).

Exit status for `--check`: 0 when no ready auto reference remains in the
non-excluded files scanned, 1 otherwise; with `--strict`, 1 when any
reference remains outside the excluded files. `--apply` exits 0. Status 2
means a path or file could not be scanned, or git failed; it takes
precedence.

Usage:
    python scripts/rewrite_examples_paths.py --check [PATHS...]
    python scripts/rewrite_examples_paths.py --check --list manual
    python scripts/rewrite_examples_paths.py --check --strict
    python scripts/rewrite_examples_paths.py --apply [PATHS...]
    python scripts/rewrite_examples_paths.py --apply --include-records FILE
"""

from __future__ import annotations

import argparse
import bisect
import os
import posixpath
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARC01 = "research/arcs/01_nla-verbalizer/scripts"
ARC02 = "research/arcs/02_subliminal/scripts"
ARC03 = "research/arcs/03_embedding-atlas/scripts"
ARC04 = "research/arcs/04_jspace/scripts"

# (file-name pattern, destination directory); first match wins. Patterns
# match a file directly under examples/, never one in a subdirectory.
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern), directory)
    for pattern, directory in [
        (r"nla_[\w\-.]+\.py", ARC01),
        (r"_nla_artifacts\.py", ARC01),
        (r"_layer_hooks\.py", ARC01),  # arc-01 only (owner, 2026-10-02)
        (r"subliminal_[\w\-.]+\.py", ARC02),
        (r"emb_[\w\-.]+\.py", ARC03),
        (r"_emb_artifacts\.py", ARC03),
        (r"jspace_[\w\-.]+\.(?:py|sh)", ARC04),
        (r"_jspace_[\w\-.]+\.py", ARC04),
    ]
]
# Shared modules that become the src/llm_research package.
PACKAGE_MODULES = {
    "_hf_models.py": "src/llm_research/hf_models.py",
    "_nla_probe.py": "src/llm_research/nla_probe.py",
}
TESTS_SOURCE = "tests/"  # examples/tests/<f> -> tests/<f>
TESTS_DEST = "tests/"
# Tracked files under examples/ with no destination, and why.
UNMAPPED = {
    "README_NLA.md": "dissolved by hand into other documents (#122 PR 3)",
}

EXCLUDED_PREFIXES = (
    "research/archive/",
    "theory/archive/",
    "research/arcs/04_jspace/data/cache/logs/",
)
EXCLUDED_FILES = {
    "research/arcs/04_jspace/data/cache/queue.log",
    "scripts/rewrite_examples_paths.py",
    "tests/test_rewrite_examples_paths.py",
}
_STEP0_PIN = "sha256 pinned by GENERATOR_SHA256 in subliminal_audit_findings.py"
_ARC04_PIN = "sha256 pinned in research/arcs/04_jspace/data/MANIFEST.json"
HASHED_RECORDS = {
    "examples/subliminal_step0_decode.py": _STEP0_PIN,
    f"{ARC02}/subliminal_step0_decode.py": _STEP0_PIN,
    "research/arcs/02_subliminal/data/step0-owl-neutral-decode/manifest.json": "sha256 pinned in research/arcs/02_subliminal/data/README.md",
    "research/arcs/04_jspace/data/fitting_prompts_c4en_n1000.json": _ARC04_PIN,
    "research/arcs/04_jspace/data/heldout_prompts_c4en_n30.json": _ARC04_PIN,
    "research/arcs/04_jspace/data/jlens_qwen2.5-1.5b_bf16_n100_layer-subset.config.json": _ARC04_PIN,
    "research/arcs/04_jspace/data/jlens_qwen2.5-7b_nf4_n100_layer-subset.config.json": _ARC04_PIN,
}
# Generated manifests: regenerated by each arc's *_data_manifest.py --write,
# never hand-edited, so they are protected like hashed records.
GENERATED_RECORD = re.compile(r"research/arcs/[^/]+/data/MANIFEST\.json")

CLASSES = ("auto", "manual", "hashed-record", "excluded")

# `examples/` that starts a path, optionally behind a ./ or ../ chain. The
# character before it may not be part of an identifier, a flag, a template
# or a dotted name.
REFERENCE = re.compile(r"(?<![\w\-.{}$@%+~])(?P<rel>(?:\.{1,2}/)*)examples/")
# Characters a path token may continue with (globs and braces included, so
# that they can be recognized as manual).
TOKEN = re.compile(r"[\w.\-/*?{},]*")
CORE = re.compile(r"[\w.\-/]*")

_problems: list[str] = []


def _problem(message: str) -> None:
    _problems.append(message)
    print(f"rewrite_examples_paths: {message}", file=sys.stderr)


def destination(name: str) -> str | None:
    """New repo path of `examples/<name>`, or None when no rule maps it."""
    if name.startswith(TESTS_SOURCE):
        rest = name[len(TESTS_SOURCE) :]
        if rest and "/" not in rest:
            return TESTS_DEST + rest
        return None
    if "/" in name:
        return None
    if name in PACKAGE_MODULES:
        return PACKAGE_MODULES[name]
    for pattern, directory in RULES:
        if pattern.fullmatch(name):
            return f"{directory}/{name}"
    return None


def file_class(path: str) -> tuple[str | None, str]:
    """('excluded' | 'hashed-record' | None, reason) for a repo-relative file."""
    if path in EXCLUDED_FILES or path.startswith(EXCLUDED_PREFIXES):
        return "excluded", "verbatim log, archive or this tool"
    if path in HASHED_RECORDS:
        return "hashed-record", HASHED_RECORDS[path]
    if GENERATED_RECORD.fullmatch(path):
        return "hashed-record", "generated by the arc's *_data_manifest.py --write"
    return None, ""


@dataclass(frozen=True)
class Ref:
    path: str  # referring file, repo-relative
    line: int
    start: int  # character offsets of the replaced text in the file
    end: int
    text: str  # the replaced text as written
    kind: str  # "auto" or "manual"
    cls: str  # one of CLASSES
    target: str | None  # repo-relative destination (auto only)
    replacement: str | None  # new text (auto only)
    ready: bool  # the destination exists on disk
    reason: str


def _repo_exists(path: str) -> bool:
    return (REPO / path).is_file()


def _resolve_name(
    core: str, exists: Callable[[str], bool]
) -> tuple[int, str, bool] | None:
    """Longest known file name at the start of `core`.

    Returns (length used in `core`, examples-relative file name, is_stem), or
    None. A name is known when a rule maps it and either its source or its
    destination exists. A stem (no extension) matches `<stem>.py`.
    """
    for i in range(len(core), 0, -1):
        if i < len(core) and core[i] not in "./":
            continue
        name = core[:i]
        if name.endswith((".", "/")):
            continue
        if name in UNMAPPED:
            return i, name, False
        dest = destination(name)
        if dest is not None and (exists("examples/" + name) or exists(dest)):
            return i, name, False
        if "." not in posixpath.basename(name):
            dest = destination(name + ".py")
            if dest is not None and (exists(f"examples/{name}.py") or exists(dest)):
                return i, name + ".py", True
    return None


def scan_text(
    path: str, text: str, exists: Callable[[str], bool] = _repo_exists
) -> list[Ref]:
    """Every examples/ reference in `text`, the content of repo file `path`."""
    fclass, freason = file_class(path)
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]
    refs: list[Ref] = []
    for m in REFERENCE.finditer(text):
        start = m.start()
        rel = m.group("rel")
        after = m.end()
        token = TOKEN.match(text, after)
        token_text = token.group(0) if token else ""
        core_match = CORE.match(text, after)
        core = core_match.group(0) if core_match else ""
        line = bisect.bisect_left(newlines, start) + 1
        kind, reason = "manual", ""
        end = after
        target: str | None = None
        replacement: str | None = None

        resolved = _resolve_name(core, exists)
        if resolved is not None:
            end = after + resolved[0]
        is_glob = any(c in token_text[: len(core) + 1] for c in "*?{")
        prefixed = not rel and start > 0 and text[start - 1] == "/"
        if rel:
            joined = posixpath.normpath(
                posixpath.join(posixpath.dirname(path), rel + "examples")
            )
            if joined != "examples":
                reason = (
                    f"relative path resolves to {joined}/, not this repo's examples/"
                )
                resolved = None
        if prefixed:
            reason = "another path prefix before examples/"
        elif reason:
            pass
        elif is_glob:
            end = after + len(token_text.rstrip(".,"))
            reason = "glob or brace form"
        elif resolved is None:
            stripped = core.rstrip(".")
            if stripped in ("", "/"):
                reason = "bare examples/ directory"
            elif stripped.rstrip("/") == "tests":
                reason = "bare examples/tests/ directory"
                end = after + len(stripped)
            else:
                reason = "unknown file name"
                end = after + len(stripped)
        else:
            _, name, is_stem = resolved
            dest = destination(name)
            if dest is None:
                reason = f"unmapped: {UNMAPPED.get(name, 'no rule')}"
            else:
                kind = "auto"
                target = dest
                new = dest
                if rel:
                    new = posixpath.relpath(dest, posixpath.dirname(path) or ".")
                if is_stem:
                    new = new[: -len(".py")]
                replacement = new
                reason = "relative link" if rel else "repo-root path"
        ready = target is not None and exists(target)
        refs.append(
            Ref(
                path=path,
                line=line,
                start=start,
                end=end,
                text=text[start:end],
                kind=kind,
                cls=fclass or kind,
                target=target,
                replacement=replacement,
                ready=ready,
                reason=freason if fclass else reason,
            )
        )
    return refs


def rewrite_text(text: str, refs: list[Ref]) -> tuple[str, list[Ref]]:
    """`text` with every ready auto ref replaced; also the refs applied."""
    applied = [r for r in refs if r.kind == "auto" and r.ready and r.replacement]
    out = text
    for r in sorted(applied, key=lambda r: r.start, reverse=True):
        assert r.replacement is not None
        out = out[: r.start] + r.replacement + out[r.end :]
    return out, applied


def candidate_files(paths: list[str]) -> tuple[list[str], set[str]]:
    """(tracked and untracked-unignored files under `paths`, explicit files).

    The second set holds the files named directly on the command line, as
    opposed to found under a directory.
    """
    rel: list[str] = []
    for arg in paths:
        absolute = Path(os.path.normpath(Path(arg).absolute()))
        try:
            rel.append(absolute.relative_to(REPO).as_posix())
        except ValueError:
            _problem(f"{arg} is outside the repository")
    if paths and not rel:
        return [], set()
    try:
        out = subprocess.run(
            [
                "git",
                "--literal-pathspecs",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *rel,
            ],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        _problem(f"git ls-files failed: {exc}")
        return [], set()
    files = sorted(dict.fromkeys(os.fsdecode(p) for p in out.split(b"\0") if p))
    for r in rel:
        if not any(
            r == "." or f == r or f.startswith(r.rstrip("/") + "/") for f in files
        ):
            _problem(f"no files under {r}")
    return files, {r for r in rel if r in files}


def lfs_files(files: list[str]) -> set[str]:
    """The subset of `files` that git routes through the LFS filter."""
    if not files:
        return set()
    try:
        out = subprocess.run(
            ["git", "check-attr", "-z", "--stdin", "filter"],
            cwd=REPO,
            input=b"\0".join(os.fsencode(f) for f in files) + b"\0",
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        _problem(f"git check-attr failed: {exc}")
        return set()
    fields = out.split(b"\0")
    return {
        os.fsdecode(fields[i])
        for i in range(0, len(fields) - 2, 3)
        if fields[i + 2] == b"lfs"
    }


def read_text(path: str) -> str | None:
    """Decoded content of a regular text file, or None for anything else.

    Symlinks, missing files, binaries (a NUL byte or invalid UTF-8) and LFS
    pointers are skipped without a problem: they hold no text to rewrite.
    """
    full = REPO / path
    if full.is_symlink() or not full.is_file():
        return None
    data = full.read_bytes()
    if b"\0" in data or data.startswith(b"version https://git-lfs"):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def collect(paths: list[str]) -> tuple[dict[str, str], list[Ref], set[str]]:
    """(texts by file, refs, explicitly named files) for `paths`."""
    files, explicit = candidate_files(paths)
    lfs = lfs_files(files)
    texts: dict[str, str] = {}
    refs: list[Ref] = []
    for path in files:
        if path in lfs:
            continue
        text = read_text(path)
        if text is None or "examples/" not in text:
            continue
        texts[path] = text
        refs.extend(scan_text(path, text))
    return texts, refs, explicit


def _format(r: Ref) -> str:
    status = ""
    if r.kind == "auto":
        status = " [ready]" if r.ready else " [pending]"
    arrow = f" -> {r.replacement}" if r.replacement else ""
    return f"{r.path}:{r.line}: {r.cls}{status}: {r.text}{arrow}  ({r.reason})"


def check(refs: list[Ref], listed: list[str], strict: bool) -> int:
    per_file: dict[str, Counter[str]] = {}
    for r in refs:
        counts = per_file.setdefault(r.path, Counter())
        counts[r.cls] += 1
        if r.kind == "auto" and r.ready:
            counts["ready"] += 1
    for path in sorted(per_file):
        c = per_file[path]
        parts = [f"{cls}={c[cls]}" for cls in CLASSES if c[cls]]
        if c["ready"]:
            parts.append(f"ready={c['ready']}")
        print(f"{path}: {' '.join(parts)}")
    for r in refs:
        if "all" in listed or r.cls in listed:
            print(_format(r))
    totals = Counter(r.cls for r in refs)
    live = [r for r in refs if r.cls != "excluded"]
    ready = [r for r in live if r.kind == "auto" and r.ready]
    lines = len({(r.path, r.line) for r in refs})
    print(
        f"REWRITE CHECK: {len(refs)} ref(s) on {lines} line(s) in {len(per_file)} file(s): "
        + ", ".join(f"{cls} {totals[cls]}" for cls in CLASSES)
        + f"; ready to rewrite {len(ready)}"
    )
    if ready:
        return 1
    if strict and live:
        return 1
    return 0


def apply(
    texts: dict[str, str], refs: list[Ref], explicit: set[str], include_records: bool
) -> int:
    by_file: dict[str, list[Ref]] = {}
    for r in refs:
        by_file.setdefault(r.path, []).append(r)
    changed = 0
    for path in sorted(by_file):
        fclass, _ = file_class(path)
        if fclass == "excluded":
            continue
        if fclass == "hashed-record" and not (include_records and path in explicit):
            continue
        new, applied = rewrite_text(texts[path], by_file[path])
        if not applied:
            continue
        (REPO / path).write_bytes(new.encode("utf-8"))
        changed += 1
        for r in applied:
            print(f"{path}:{r.line}: {r.text} -> {r.replacement}")
    total = sum(1 for r in refs if r.kind == "auto" and r.ready)
    print(f"REWRITE APPLY: changed {changed} file(s); ready refs seen {total}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find and rewrite references to examples/ paths (#122)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report references")
    mode.add_argument("--apply", action="store_true", help="rewrite ready auto refs")
    parser.add_argument("paths", nargs="*", help="limit to these paths")
    parser.add_argument(
        "--list",
        action="append",
        default=[],
        choices=[*CLASSES, "all"],
        help="with --check, print each ref of this class (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --check, fail on any non-excluded ref",
    )
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="with --apply, also rewrite hashed-record files named explicitly",
    )
    args = parser.parse_args(argv)

    _problems.clear()
    try:
        texts, refs, explicit = collect(args.paths)
        if args.check:
            status = check(refs, args.list, args.strict)
        else:
            status = apply(texts, refs, explicit, args.include_records)
    except Exception as exc:  # a failed run is never reported as clean
        print(f"rewrite_examples_paths: failed: {exc!r}", file=sys.stderr)
        return 2
    if _problems:
        print(f"REWRITE: {len(_problems)} path(s) or file(s) not scanned")
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
