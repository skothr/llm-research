#!/usr/bin/env python3
"""Redact third-party PII from the arc-04 web-derived fitting corpora.

WHY THIS EXISTS
---------------
Arc 04's corpus-sensitivity axis (Decision 1, 2026-07-20) uses a seeded slice
of C4-en — Common Crawl web text. C4's cleaning pipeline
`[raffel2020-t5 §2.2]` filters for quality (boilerplate, length, offensive
content); it does **not** filter personal data. A uniform 1000-document sample
therefore carries contact details of real people at the corpus base rate
(~20 emails / ~50 phones per 1000 docs, per Elazar et al. 2024, *What's In My
Big Data?*, ICLR — Table 5). Measured here: consistent with that rate.

Publishing those details in a public repo is not something the upstream
ODC-BY licence can authorise — data-subject rights attach to the person, not
to the licensor — so the committed corpora are redacted. See
`research/arcs/04_jspace/data/README.md` for the disclosure and the
reproduction recipe.

DETERMINISM CONTRACT
--------------------
Redaction is pure text substitution with fixed sentinels, no randomness, no
model. Anyone can reproduce the committed artifact exactly:

    1. regenerate the raw slice from the recipe pinned in the corpus JSON's
       own `source`/`selection`/`seed` fields (public: HF `allenai/c4`)
    2. run this script over it
    3. compare sha256 against `data/MANIFEST.json`

Sentinels are bracketed uppercase class names (`[EMAIL]`, `[PHONE]`,
`[STREET-ADDRESS]`, `[POSTAL-CODE]`). They are NOT length-preserving: a
redacted document tokenises differently from its original. That is a real
consequence for anything fit on this text and is disclosed rather than hidden
— see the warning at the top of the arc README.

SCOPE AND LIMITS
----------------
Redacts contact *channels* (email, phone, street address, postal code), which
is what makes a named individual reachable. It does NOT attempt general
person-name detection: reliable name NER over web text has a false-positive
rate that would gut the corpus ("Washington", "Brooks", "Reed" are all
places, surnames, and common nouns), and a bare name in web prose is not by
itself a contact channel. Removing the channel breaks the linkage that makes
the combination identifying. This limitation is stated in the data README so
readers can judge it rather than assume completeness.

USAGE
-----
    # count without writing (safe, prints no PII values)
    python examples/jspace_redact_corpus.py --report research/arcs/04_jspace/data/*.json

    # redact in place, updating the file's own provenance block
    python examples/jspace_redact_corpus.py --apply <file.json> [<file.json> ...]

    # verify a file carries no detectable PII (exit 1 if any found)
    python examples/jspace_redact_corpus.py --check <file.json> [...]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

REDACTION_SCRIPT_VERSION = "1.0.0"

_US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|"
    "MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|"
    "WI|WY|DC"
)

_STREET_SUFFIX = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
    "Place|Pl|Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Suite|Ste"
)

# Ordered: earlier classes are applied first, so an address line is consumed
# before the bare postal-code rule can fire inside it.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        # +1 (555) 123-4567 / 555.123.4567 / 555-123-4567 / (555) 123 4567.
        # Guards exclude dotted version strings (11.3.3.83) while still
        # matching a number that ends a sentence — hence `(?!\d|\.\d)` rather
        # than `(?![\d.])`, which wrongly rejected a trailing full stop.
        "PHONE",
        re.compile(
            r"(?<!\d)(?<!\d\.)(?:\+?\d{1,3}[ .-]?)?"
            r"(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}"
            r"(?!\d|\.\d)"
        ),
    ),
    (
        # Requires >=1 capitalised street-name token between number and suffix,
        # so a bare "03 St." never matches. The `(?! of )` guard keeps
        # institutional names out ("2004 International Court of Justice",
        # "9th Circuit Court of Appeals") — those are not addresses.
        "STREET-ADDRESS",
        re.compile(
            rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z.'-]*\s+){{1,4}}(?:{_STREET_SUFFIX})\b\.?"
            r"(?! of )",
        ),
    ),
    (
        "POSTAL-CODE",
        re.compile(rf"\b(?:{_US_STATES})\s+\d{{5}}(?:-\d{{4}})?\b"),
    ),
]


# Matches reviewed by a human and determined NOT to be personal data. Recorded
# here (rather than by loosening a regex) so the judgment is explicit,
# greppable, and reviewable. Keyed by corpus filename -> matched substrings.
#
# wikitext-103 review, 2026-07-29: both hits are the same encyclopedic
# biography of Mortimer Wheeler (archaeologist, 1890-1976) — his 1908 and
# 1950s London residences, stated as historical fact in a Wikipedia article.
# No living person is identified and no contact channel is published, so these
# are not personal data. Redacting them would damage the arc's PRIMARY fitting
# corpus for no privacy gain; wikitext-103 is therefore committed unredacted.
REVIEWED_NOT_PII: dict[str, frozenset[str]] = {
    "fitting_prompts_wikitext103_n1000.json": frozenset(
        {"14 Rollescourt Avenue", "27 Whitcomb Street"}
    ),
}


def _cleared(path: Path, matched: str) -> bool:
    return matched in REVIEWED_NOT_PII.get(path.name, frozenset())


def redact_text(text: str, path: Path | None = None) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, {class: n_replacements}).

    Matches listed in REVIEWED_NOT_PII for ``path`` are left untouched.
    """
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS:
        hits = 0

        def _sub(m: re.Match[str]) -> str:
            nonlocal hits
            if path is not None and _cleared(path, m.group().strip()):
                return m.group()
            hits += 1
            return f"[{label}]"

        text = pattern.sub(_sub, text)
        if hits:
            counts[label] = counts.get(label, 0) + hits
    return text, counts


def scan_text(text: str, path: Path | None = None) -> dict[str, int]:
    """Count matches without writing. Never returns matched values."""
    return redact_text(text, path)[1]


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict) or not isinstance(obj.get("prompts"), list):
        return {}
    return obj


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def cmd_report(paths: list[Path]) -> int:
    any_found = False
    for path in paths:
        obj = _load(path)
        if not obj:
            continue
        prompts = obj["prompts"]
        assert isinstance(prompts, list)
        totals: dict[str, int] = {}
        docs_hit = 0
        for doc in prompts:
            if not isinstance(doc, str):
                continue
            counts = scan_text(doc, path)
            if counts:
                docs_hit += 1
                _merge(totals, counts)
        status = "CLEAN" if not totals else "PII FOUND"
        print(f"{path.name}: {status}")
        print(f"  documents: {len(prompts)}, with PII: {docs_hit}")
        for label, n in sorted(totals.items()):
            print(f"    {label}: {n}")
        if totals:
            any_found = True
    return 1 if any_found else 0


def cmd_check(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        obj = _load(path)
        if not obj:
            continue
        prompts = obj["prompts"]
        assert isinstance(prompts, list)
        totals: dict[str, int] = {}
        for doc in prompts:
            if isinstance(doc, str):
                _merge(totals, scan_text(doc, path))
        if totals:
            failed = True
            print(
                f"FAIL {path.name}: {sum(totals.values())} PII match(es) {sorted(totals)}"
            )
        else:
            print(f"PASS {path.name}: no detectable PII")
    return 1 if failed else 0


def cmd_apply(paths: list[Path]) -> int:
    for path in paths:
        obj = _load(path)
        if not obj:
            continue
        prompts = obj["prompts"]
        assert isinstance(prompts, list)
        sha_before = _sha256(path)
        totals: dict[str, int] = {}
        docs_hit = 0
        out: list[object] = []
        for doc in prompts:
            if not isinstance(doc, str):
                out.append(doc)
                continue
            redacted, counts = redact_text(doc, path)
            if counts:
                docs_hit += 1
                _merge(totals, counts)
            out.append(redacted)
        obj["prompts"] = out
        obj["redaction"] = {
            "applied": _dt.date.today().isoformat(),
            "script": "examples/jspace_redact_corpus.py",
            "script_version": REDACTION_SCRIPT_VERSION,
            "classes": [label for label, _ in PII_PATTERNS],
            "sentinel_format": "[CLASS]",
            "length_preserving": False,
            "documents_total": len(prompts),
            "documents_modified": docs_hit,
            "replacements": dict(sorted(totals.items())),
            "sha256_before_redaction": sha_before,
            "rationale": (
                "C4 is Common-Crawl web text; its cleaning filters quality, not "
                "personal data. Third-party contact details are removed because "
                "the upstream ODC-BY licence cannot authorise republishing them. "
                "See research/arcs/04_jspace/data/README.md."
            ),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(
            f"{path.name}: redacted {sum(totals.values())} match(es) across {docs_hit} doc(s)"
        )
        for label, n in sorted(totals.items()):
            print(f"    {label}: {n}")
        print(f"    sha256 before: {sha_before}")
        print(f"    sha256 after:  {_sha256(path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="count PII, write nothing")
    mode.add_argument("--apply", action="store_true", help="redact in place")
    mode.add_argument("--check", action="store_true", help="exit 1 if any PII detected")
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)

    paths = [p for p in args.paths if p.suffix == ".json"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(str(p) for p in missing)}")

    if args.report:
        return cmd_report(paths)
    if args.check:
        return cmd_check(paths)
    return cmd_apply(paths)


if __name__ == "__main__":
    sys.exit(main())
