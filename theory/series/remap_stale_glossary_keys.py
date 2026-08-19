#!/usr/bin/env python3
"""Rewrite \\glsterm/\\glsdef sites whose key no longer has a glossary record.

The glossary merge collapses duplicate entries, which retires the `-2`/`-3`
collision keys (and the expansion-first keys of acronym pairs) that already-
marked body text refers to. mark_glossary_terms.py never revisits a marked
site, and render_glossary_section skips a key with no record, so without this
pass the affected terms silently vanish from the per-paper glossaries and
their body hyperlinks dangle.

Resolution is by SURFACE FORM against the regenerated glossary-terms.json,
using the same primary-outranks-alias precedence as
mark_glossary_terms._build_surface_to_key. A key is rewritten only when every
one of its sites resolves to the same surviving key; anything ambiguous or
unresolvable is reported and left alone.

Usage:
    python3 apply_stale_remap.py <theory/series> [--dry-run]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SITE = re.compile(r"(\\gls(?:term|def))\{([^{}]+)\}\{([^{}]*)\}")


def build_surface_map(terms: list[dict]) -> dict[str, str]:
    surface_map: dict[str, str] = {}
    for take_primary in (False, True):
        for r in terms:
            forms = [r["primary_form"]] if take_primary else list(r.get("aliases", []))
            for s in forms:
                if len(s) < 2:
                    continue
                strict = any(c.isupper() for c in s[1:])
                surface_map[s if strict else s.lower()] = r["key"]
    return surface_map


def main() -> int:
    series = Path(sys.argv[1]).resolve()
    dry = "--dry-run" in sys.argv[2:]
    terms = json.loads((series / "glossary-terms.json").read_text())["terms"]
    live = {t["key"] for t in terms}
    surface_map = build_surface_map(terms)

    files = [
        f
        for p in range(1, 6)
        for f in sorted((series / f"paper-{p}" / "sections").glob("*.tex"))
    ]

    # Pass 1 — decide a target per stale key.
    votes: dict[str, Counter] = {}
    for f in files:
        for m in SITE.finditer(f.read_text()):
            key, surf = m.group(2), m.group(3)
            if key in live:
                continue
            tgt = surface_map.get(surf) or surface_map.get(surf.lower())
            votes.setdefault(key, Counter())[tgt or ""] += 1

    remap: dict[str, str] = {}
    for key, c in sorted(votes.items()):
        targets = [t for t in c if t]
        if len(c) == 1 and len(targets) == 1:
            remap[key] = targets[0]
        else:
            print(f"SKIP {key}: unresolved/ambiguous -> {dict(c)}")

    # Pass 2 — rewrite.
    total = 0
    for f in files:
        src = f.read_text()

        def sub(m: re.Match[str]) -> str:
            new = remap.get(m.group(2))
            return (
                m.group(0) if new is None else f"{m.group(1)}{{{new}}}{{{m.group(3)}}}"
            )

        out = SITE.sub(sub, src)
        if out != src:
            n = sum(1 for m in SITE.finditer(src) if m.group(2) in remap)
            total += n
            print(f"{'would rewrite' if dry else 'rewrote'} {n:4} sites in {f.name}")
            if not dry:
                f.write_text(out)
    print(f"\n{len(remap)} keys remapped, {total} sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
