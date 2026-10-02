"""Report mannered stock phrases in tracked prose and code comments (#120).

Scans tracked *.md, *.py and *.tex files for phrases that stand in for a
direct statement ("load-bearing", "genuinely", "sits at", ...). Each hit is
graded by a human or agent: reword it, cut it, or keep it. A kept hit carries
the marker `prose-lint: allow` on the same line, in that file's comment
syntax, so the scanner skips it.

Not scanned:
- verbatim source material: theory/sources/, theory/kb/excerpts/;
- dated records: research/archive/, theory/archive/, theory/reviews/;
- the owner's quoted turns in attribution sections (lines starting with
  `> *"`), which are verbatim quotes;
- this script and its test, which list the phrases.

Usage:
    python scripts/prose_lint.py                 # list hits; exit 1 if any
    python scripts/prose_lint.py --summary       # counts by phrase and area
    python scripts/prose_lint.py PATH [PATH ...] # limit to these paths
    python scripts/prose_lint.py --report        # always exit 0
"""

from __future__ import annotations

import argparse
import re
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
EXCLUDED_FILES = {"scripts/prose_lint.py", "examples/tests/test_prose_lint.py"}
ALLOW_MARKER = "prose-lint: allow"
QUOTE_LINE = re.compile(r"^\s*>\s*\*\"")


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


def scan_text(path: str, text: str) -> list[Hit]:
    """Return every phrase hit in `text`, skipping allowed and quoted lines."""
    hits: list[Hit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line or QUOTE_LINE.match(line):
            continue
        for label, pattern in PHRASES:
            if pattern.search(line):
                hits.append(Hit(path, lineno, label, line.strip()))
    return hits


def tracked_files(paths: list[str]) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *paths],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def scan(paths: list[str]) -> list[Hit]:
    hits: list[Hit] = []
    for path in tracked_files(paths):
        if not is_scanned(path):
            continue
        try:
            text = (REPO / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits.extend(scan_text(path, text))
    return hits


def area(path: str) -> str:
    """Top two path components, e.g. `theory/kb` or `README.md`."""
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report mannered stock phrases in tracked prose (#120)."
    )
    parser.add_argument("paths", nargs="*", help="limit the scan to these paths")
    parser.add_argument("--summary", action="store_true", help="print counts only")
    parser.add_argument("--report", action="store_true", help="always exit 0")
    args = parser.parse_args(argv)

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
    return 0 if args.report or not hits else 1


if __name__ == "__main__":
    sys.exit(main())
