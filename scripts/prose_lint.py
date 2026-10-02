"""Report mannered stock phrases in tracked prose and code comments (#120).

Scans every line of *.md, *.py and *.tex files (in .py files that includes
code and string literals, not only comments) for phrases that stand in for a
direct statement ("load-bearing", "genuinely", "sits at", ...). Each hit is
graded by a human or agent: reword it, cut it, or keep it. A kept hit carries
the marker `prose-lint: allow` on the same line, in that file's comment
syntax, so the scanner skips it.

Not scanned:
- verbatim source material: theory/sources/, theory/kb/excerpts/;
- dated records: research/archive/, theory/archive/, theory/reviews/;
- quoted turns in the attribution format, every line from an opening `> *"`
  to its closing `"*`; an unterminated quote ends with its blockquote;
- in Markdown, fenced code blocks (``` or ~~~), fence lines included, as a
  CommonMark parser (markdown-it-py) finds them: they hold verbatim material,
  where a marker would render as text (issue #156);
- hash-pinned files, whose sha256 an audit checks (issue #157);
- this script and its test, which list the phrases.

A phrase that a hard wrap splits across two adjacent lines ("tells" at the
end of one line, "us" at the start of the next) is reported at the first line.

Paths are resolved from the caller's working directory. Tracked files and
untracked files that git does not ignore are both scanned, so new prose is
checked before it is staged. A file is read only when it is a regular file
reached without any symlink, in the leaf or a parent directory; symlinks,
tracked files missing from disk and other non-regular files are never read,
and each one counts as not scanned.

Known limits: an allow-marked line is never joined with its neighbours, a
quoted turn that itself contains `"*` ends the skip at that line, and a `"*`
on a fenced line inside a quoted turn does not end the turn.

Exit status: 0 when nothing is found, 1 when there are hits, and 2 when any
path or file could not be scanned (a path outside the repository, a path that
matches no file or no in-scope file, an unreadable or non-regular file, or a
git failure; any unexpected error also exits 2). Status 2 takes precedence,
so a failed scan never passes as a clean run or as a hit count.

Usage:
    python scripts/prose_lint.py                 # list hits; exit 1 if any
    python scripts/prose_lint.py --summary       # counts by phrase and area
    python scripts/prose_lint.py PATH [PATH ...] # limit to these paths
    python scripts/prose_lint.py --report        # exit 0 on hits (2 still wins)
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PHRASES: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in [
        ("load-bearing", r"\bload[- ]bearing\b"),
        ("genuinely", r"\bgenuine(ly)?\b"),
        ("honest(ly)", r"\bhonest(ly)?\b"),
        ("first-class", r"\bfirst[- ]class\b"),
        ("sits at", r"\bsits? at\b"),
        ("tells us", r"\btells us\b"),
        ("cleanly separates", r"\bclean(ly)? separat"),
        ("crisp", r"\bcrisp(ly)?\b"),
        ("quietly", r"\bquietly\b"),
        ("bites", r"\bbites?\b"),
        ("teeth", r"\bteeth\b"),
        ("surface area", r"\bsurface area\b"),
        ("belt-and-braces", r"\bbelt[- ]and[- ]braces\b"),
        ("at the heart of", r"\bat the heart of\b"),
        ("earns its keep", r"\bearns? its keep\b"),
        ("worth its weight", r"\bworth its weight\b"),
    ]
]

SUFFIXES = {".md", ".py", ".tex"}
EXCLUDED_PREFIXES = (
    "theory/sources/",
    "theory/kb/excerpts/",
    "research/archive/",
    "theory/archive/",
    "theory/reviews/",
)
EXCLUDED_FILES = {
    "scripts/prose_lint.py",
    "examples/tests/test_prose_lint.py",
    # Pinned by GENERATOR_SHA256 in examples/subliminal_audit_findings.py and
    # by research/arcs/02_subliminal/data/README.md; any edit, a marker
    # included, fails the arc-02 audit (#157).
    "examples/subliminal_step0_decode.py",
}
ALLOW_MARKER = "prose-lint: allow"
QUOTE_LINE = re.compile(r"^\s*>\s*\*\"")
# Leading markup a wrapped continuation line can start with: a blockquote
# marker or a comment marker (Markdown, Python, LaTeX).
CONTINUATION_PREFIX = re.compile(r"^\s*(?:>\s*|#+\s*|%+\s*)?")

# Problems that stop a path or file from being scanned; any of them makes
# main() exit 2. Reset at the start of each main() call.
_problems: list[str] = []


def _problem(message: str) -> None:
    _problems.append(message)
    print(f"prose_lint: {message}", file=sys.stderr)


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    phrase: str
    text: str


def is_scanned(path: str) -> bool:
    """True when `path` (repo-relative, posix) is in scope for the scan."""
    if Path(path).suffix not in SUFFIXES or path in EXCLUDED_FILES:
        return False
    return not path.startswith(EXCLUDED_PREFIXES)


def fenced_lines(text: str) -> set[int]:
    """0-based numbers of the lines inside Markdown fenced code blocks.

    The fence lines themselves are included. CommonMark decides where each
    block ends: at its closing fence, at the end of the blockquote or list
    item it opened in, or at the end of the file.
    """
    try:  # imported here, so a missing package exits 2 through main()
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError(
            "markdown-it-py is not installed; pip install -e '.[dev]'"
        ) from exc

    lines: set[int] = set()
    for token in MarkdownIt("commonmark").parse(text.replace("\r\n", "\n")):
        if token.type == "fence" and token.map is not None:
            lines.update(range(*token.map))
    return lines


def scan_text(path: str, text: str) -> list[Hit]:
    """Return every phrase hit in `text`, skipping allowed and quoted lines.

    A quoted owner turn opens with `> *"` and may continue over further `>`
    lines until the closing `"*`; those lines are skipped, except for any text
    after the closing `"*` on the last line. In a Markdown file, the lines of
    each fenced code block are skipped, as `fenced_lines` finds them; a fence
    inside a quoted turn leaves the turn open. A phrase split across two
    adjacent scanned lines is reported at the first of them.
    """
    def after_close(line: str, start: int) -> str | None:
        """The text after the quote's closing `"*`, or None while still open."""
        end = line.rfind('"*', start)
        return None if end < 0 else line[end + 2 :]

    scannable: list[str | None] = []
    in_quote = False
    fenced = fenced_lines(text) if path.endswith(".md") else set()
    # Split on newlines only, so line numbers match editors and git.
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n")):
        if number in fenced:
            # Outside a blockquote, a fence line ends any quoted turn.
            if not line.lstrip().startswith(">"):
                in_quote = False
            scannable.append(None)
            continue
        tail: str | None = None
        if in_quote and line.lstrip().startswith(">"):
            tail = after_close(line, 0)
            in_quote = tail is None
        else:
            in_quote = False
            opening = QUOTE_LINE.match(line)
            if not opening:
                scannable.append(None if ALLOW_MARKER in line else line)
                continue
            tail = after_close(line, opening.end())
            in_quote = tail is None
        # Inside a quoted turn: only text after its closing `"*` is scanned.
        if tail is None or not tail.strip() or ALLOW_MARKER in line:
            scannable.append(None)
        else:
            scannable.append(tail)

    hits: list[Hit] = []
    for i, line in enumerate(scannable):
        if line is None:
            continue
        for label, pattern in PHRASES:
            if pattern.search(line):
                hits.append(Hit(path, i + 1, label, line.strip()))
        following = scannable[i + 1] if i + 1 < len(scannable) else None
        if following is None:
            continue
        head = line.rstrip()
        tail = CONTINUATION_PREFIX.sub("", following, count=1)
        joined = head + ("" if head.endswith("-") else " ") + tail
        for label, pattern in PHRASES:
            if any(m.start() < len(head) < m.end() for m in pattern.finditer(joined)):
                hits.append(Hit(path, i + 1, label, line.strip()))
    return hits


def candidate_files(paths: list[str]) -> list[str]:
    """Tracked and untracked (not ignored) files under `paths`, repo-relative.

    `paths` are resolved against the caller's working directory. A path that
    matches no file, or no in-scope file, is a problem: main() exits 2.
    """
    rel: list[str] = []
    for arg in paths:
        absolute = Path(os.path.normpath(Path(arg).absolute()))
        for candidate in (absolute, Path(os.path.realpath(absolute))):
            try:
                rel.append(candidate.relative_to(REPO).as_posix())
                break
            except ValueError:
                continue
        else:
            _problem(f"{arg} is outside the repository")
    if paths and not rel:
        return []
    try:
        out = subprocess.run(
            ["git", "--literal-pathspecs", "ls-files", "-z", "--cached", "--others",
             "--exclude-standard", "--", *rel],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        _problem(f"git ls-files failed: {exc}")
        return []
    files = list(dict.fromkeys(os.fsdecode(p) for p in out.split(b"\0") if p))
    for r in rel:
        under = [f for f in files if r == "." or f == r or f.startswith(r.rstrip("/") + "/")]
        if not under:
            _problem(f"no files under {r}")
        elif not any(is_scanned(f) for f in under):
            _problem(f"no in-scope files under {r}")
    return files


def read_regular(path: str) -> str | None:
    """Read repo file `path` only if it is a regular file reached without symlinks.

    The real path must equal the repo path, so no component (leaf or parent
    directory) is a symlink. The file is opened once with O_NOFOLLOW and
    O_NONBLOCK, and the open descriptor itself must be a regular file, so a
    swap between check and read cannot redirect or block the read. Any
    failure is recorded as a problem and returns None.
    """
    full = os.path.normpath(REPO / path)
    if os.path.realpath(full) != full:
        _problem(f"skipped {path}: its path passes through a symlink")
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(full, flags)
    except OSError as exc:
        _problem(f"skipped {path}: {exc.strerror or exc}")
        return None
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            _problem(f"skipped {path}: not a regular file")
            return None
        data = handle.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _problem(f"could not read {path}: {exc}")
        return None


def scan(paths: list[str]) -> list[Hit]:
    hits: list[Hit] = []
    for path in candidate_files(paths):
        if not is_scanned(path):
            continue
        text = read_regular(path)
        if text is not None:
            hits.extend(scan_text(path, text))
    return hits


def area(path: str) -> str:
    """The file's directory, at most two levels deep: `theory/kb`, `examples`.

    A file at the repository root is its own area.
    """
    parts = path.split("/")
    return "/".join(parts[: min(2, len(parts) - 1)]) or path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report mannered stock phrases in tracked prose (#120)."
    )
    parser.add_argument("paths", nargs="*", help="limit the scan to these paths")
    parser.add_argument("--summary", action="store_true", help="print counts only")
    parser.add_argument(
        "--report", action="store_true", help="exit 0 on hits (status 2 still wins)"
    )
    args = parser.parse_args(argv)

    _problems.clear()
    try:
        return _run(args)
    except Exception as exc:  # any failure is "not scanned", never "clean"
        print(f"prose_lint: scan failed: {exc!r}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    hits = scan(args.paths)
    if args.summary:
        for label, n in Counter(h.phrase for h in hits).most_common():
            print(f"{n:5d}  {label}")
        print("by area:")
        for name, n in Counter(area(h.path) for h in hits).most_common():
            print(f"{n:5d}  {name}")
    else:
        for h in hits:
            print(f"{h.path}:{h.line}: [{h.phrase}] {h.text[:160]}")
    print(f"PROSE LINT: {len(hits)} hit(s) in {len({h.path for h in hits})} file(s)")
    if _problems:
        print(f"PROSE LINT: {len(_problems)} path(s) or file(s) not scanned")
        return 2
    return 0 if args.report or not hits else 1


if __name__ == "__main__":
    sys.exit(main())
