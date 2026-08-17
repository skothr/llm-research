#!/usr/bin/env python3
"""Theory-KB lint.

Originally the Phase-2 self-review lint (`plans/_phase2/lint.py`); promoted to
`kb/lint.py` on 2026-08-17 when its scan set was widened past `kb/notes/` to the
whole KB and its citation checks were made anchor-accurate.

Checks:
1. papers.json is valid; every paper has required fields.
2. Every excerpts_file in papers.json points to a real file (or null).
3. Every kb/notes/<area>/<topic>.md exists for every leaf in topics.md.
4. Every paper-key cited inline ([key §X], the bare [key] form, *or* the
   `[key, arXiv NNNN.NNNNN]` form) corresponds to a key in papers.json.
5. Every kb/excerpts/<key>#anchor citation resolves to a real excerpt file *and*
   to a real anchor inside it (`{#attr}` attributes, the repo-local
   `## #anchor — Title` heading convention, and plain markdown heading slugs).
   The anchorless `kb/excerpts/<key>` form is checked for file existence.
6. Notes have YAML frontmatter with required keys.
7. Count [INTUITION], [ANALOGY], [CONTRADICTION] tags per note.
8. Hybrid citations `[key §X; kb/excerpts/key#anchor]` are §-consistent: when
   the targeted excerpt heading names its own section (`## §6.3 Robustness …`),
   the citation's `§X` label must agree with it.

Scan set for checks 4/5/7 is every `kb/**/*.md` outside `kb/excerpts/` — notes,
glossary.md, index/ (including `_phase2-additions/` and contradictions.md).
Frontmatter (check 6) is still required of `kb/notes/**` only.

Run from theory/:  python3 kb/lint.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

THEORY = Path(__file__).resolve().parents[1]
KB = THEORY / "kb"
PAPERS = KB / "index" / "papers.json"
TOPICS = KB / "index" / "topics.md"
NOTES = KB / "notes"
EXCERPTS = KB / "excerpts"

errors: list[str] = []
warnings: list[str] = []
stats: dict = defaultdict(int)


def check_papers_json():
    data = json.loads(PAPERS.read_text())
    keys = set()
    for p in data["papers"]:
        if "key" not in p:
            errors.append(f"papers.json: paper missing key: {p.get('title', '?')}")
            continue
        if p["key"] in keys:
            errors.append(f"papers.json: duplicate key {p['key']!r}")
        keys.add(p["key"])
        if p.get("excerpts_file"):
            ep = THEORY / p["excerpts_file"]
            if not ep.exists():
                errors.append(
                    f"papers.json: {p['key']!r} excerpts_file points to missing file "
                    f"{p['excerpts_file']}"
                )
        for required in ("title", "year"):
            if not p.get(required):
                warnings.append(f"papers.json: {p['key']!r} missing/empty {required}")
    stats["papers_total"] = len(data["papers"])
    stats["papers_with_excerpts"] = sum(
        1 for p in data["papers"] if p.get("excerpts_file")
    )
    stats["papers_with_notes"] = sum(
        1 for p in data["papers"] if p.get("notes_referenced_by")
    )
    return data, keys


def check_topics():
    text = TOPICS.read_text()
    by_status = defaultdict(int)
    leaves = []
    for line in text.splitlines():
        m = re.match(
            r"^\|\s*(?P<topic>[a-z0-9-]+)\s*\|\s*(?P<status>\w+)\s*\|"
            r"\s*`(?P<path>kb/notes/[^`]+)`\s*\|\s*$",
            line,
        )
        if m:
            topic = m.group("topic")
            status = m.group("status")
            path = m.group("path")
            leaves.append((topic, status, path))
            by_status[status] += 1
            note_path = THEORY / path
            if not note_path.exists():
                errors.append(
                    f"topics.md: leaf {topic!r} note path {path!r} does not exist"
                )
    stats["leaves_by_status"] = dict(by_status)
    stats["leaves_total"] = len(leaves)
    return leaves


# --- citation grammar -------------------------------------------------------

# `[key §X ...]` — the sectioned form.
PAPER_CITE_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9.\-]*(?:/[A-Za-z0-9.\-]+)*)\s*§")
# `[key]` — the bare form, no `§`. Excludes markdown links (`[text](url)`),
# reference links (`[text][ref]`), tags (`[INTUITION]`), and footnotes.
BARE_CITE_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9.\-]*)\](?![\(\[])")
# A bare `[token]` is only read as a citation when it is *key-shaped*: it
# carries a 19xx/20xx year (`burns2023-w2s`, `ring-attention-2023`) or is a bare
# arXiv id (`2502.04420`). Deliberately conservative — a yearless key such as
# `phi4` is skipped in the bare form rather than risk flagging prose like
# `[city1]`. The `§` form has no such restriction.
KEY_SHAPED_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)|^\d{4}\.\d{4,5}$")
# Bracket payloads that are never paper keys.
NOT_A_KEY = {
    "intuition",
    "analogy",
    "contradiction",
    "forum-signal",
    "speculation",
    "tbd",
    "todo",
    "sic",
    "citation-needed",
    "unverified",
    "unsourced",
    "verified",
    "note",
    # The citation grammar's own placeholder, quoted in kb/README.md and
    # kb/index/timeline.md when they document the `[paper-key §X]` form.
    "paper-key",
}
# Match excerpt anchors anywhere in citation brackets (hybrid form
# `[paper-key §X; kb/excerpts/key#anchor]` is dominant in practice). `§` is in
# the anchor class so a note-side `#§3` shorthand written against an *excerpt*
# is validated (and flagged) rather than silently skipped.
EXCERPT_CITE_RE = re.compile(r"(kb/excerpts/[a-z0-9.\-]+)#(§?[A-Za-z0-9.\-]+)")
# The anchorless fallback `kb/excerpts/<key>` (optionally `.md`-suffixed): no
# anchor to resolve, but the file still has to exist. Greedy — the trailing
# `.md` is stripped after the match, and an immediately following `#` means the
# anchored form above already owns the site.
EXCERPT_FILE_RE = re.compile(r"kb/excerpts/([a-z0-9.\-]+)")
# Intra-KB note citations: `[kb/notes/<area>/<file>#anchor]`. The anchor is
# optional and comes in two forms — a heading slug (`#3-why-the-logit-lens…`)
# or the repo's section-number shorthand (`#§2`, `#§3.1`).
NOTE_CITE_RE = re.compile(
    r"kb/notes/([a-z0-9\-]+/[a-z0-9\-]+)(?:\.md)?(?:#(§?[A-Za-z0-9.\-]+))?"
)
# `## 3.1 The KV-sharing axis …` — the number a `#§3.1` shorthand targets.
SECTION_NUMBER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)\.?\s")
TAG_RE = re.compile(r"\[(INTUITION|ANALOGY|CONTRADICTION|FORUM-SIGNAL|SPECULATION)\]")

ANCHOR_ATTR_RE = re.compile(r"\{#([A-Za-z0-9._\-]+)\}")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$")
# Repo-local convention: `## #sec-3-1 — Layer normalized RNNs`.
LEADING_ANCHOR_RE = re.compile(r"^#([A-Za-z0-9._\-]+)\b")


def heading_slug(text: str) -> str:
    """GitHub-ish slug of a heading's visible text.

    GFM emits one hyphen *per* whitespace character, so a run of n spaces
    becomes n hyphens — hence the per-character substitution rather than a
    `\\s+` collapse. Known deviation: GFM disambiguates repeated slugs within a
    document by appending `-1`, `-2`, …; this does not, so a duplicate heading
    text yields one anchor here where GFM yields several. No live instance in
    the KB (every duplicated heading text sits in a different file), so the
    simpler behaviour stands until one appears.
    """
    text = ANCHOR_ATTR_RE.sub("", text).strip()
    text = re.sub(r"[`*_$\\]", "", text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    return re.sub(r"\s", "-", text).strip("-")


_anchor_cache: dict[Path, set[str]] = {}


def anchors_of(path: Path) -> set[str]:
    """Every anchor a citation may legitimately target inside `path`."""
    if path not in _anchor_cache:
        text = path.read_text()
        found = set(ANCHOR_ATTR_RE.findall(text))
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if not m:
                continue
            head = m.group(1)
            lead = LEADING_ANCHOR_RE.match(head)
            if lead:
                found.add(lead.group(1))
            slug = heading_slug(head)
            if slug:
                found.add(slug)
        _anchor_cache[path] = found
    return _anchor_cache[path]


_section_cache: dict[Path, set[str]] = {}


def section_numbers_of(path: Path) -> set[str]:
    """Numbered-heading labels in `path`, for the `#§3.1` citation shorthand."""
    if path not in _section_cache:
        found: set[str] = set()
        for line in path.read_text().splitlines():
            m = HEADING_RE.match(line)
            if not m:
                continue
            num = SECTION_NUMBER_RE.match(m.group(1))
            if num:
                found.add(num.group(1))
        _section_cache[path] = found
    return _section_cache[path]


_designation_cache: dict[Path, dict[str, set[str]]] = {}

# `## §6.3 Robustness …` / `## §5 / §6.1 Model Rankings …` — the source-paper
# section(s) an excerpt heading transcribes.
HEADING_SECTION_RE = re.compile(r"§\s*([A-Za-z]?[0-9]+(?:\.[0-9]+)*)")
# The `§X` label carried by a citation, e.g. `[wang2024-mmlu-pro §6.3; …]`.
CITE_SECTION_RE = re.compile(r"§\s*([A-Za-z0-9][A-Za-z0-9.\-]*)")
# A whole bracketed citation payload that names an excerpt anchor.
HYBRID_CITE_RE = re.compile(r"\[([^\[\]]*kb/excerpts/[^\[\]]*)\]")


def section_designations_of(path: Path) -> dict[str, set[str]]:
    """anchor -> the `§N` designations its own heading text names, per excerpt.

    Only headings that name a section are recorded; `## Abstract {#abstract}`
    contributes nothing and so can never produce a mismatch.
    """
    if path not in _designation_cache:
        found: dict[str, set[str]] = {}
        for line in path.read_text().splitlines():
            m = HEADING_RE.match(line)
            if not m:
                continue
            head = m.group(1)
            sections = set(HEADING_SECTION_RE.findall(head))
            if not sections:
                continue
            names = set(ANCHOR_ATTR_RE.findall(head))
            lead = LEADING_ANCHOR_RE.match(head)
            if lead:
                names.add(lead.group(1))
            slug = heading_slug(head)
            if slug:
                names.add(slug)
            for name in names:
                found.setdefault(name, set()).update(sections)
        _designation_cache[path] = found
    return _designation_cache[path]


def _section_agrees(cited: str, target: str) -> bool:
    """`§3` agrees with a `§3.2` heading; `§3.2.1` agrees with `§3.2`."""
    c, t = cited.rstrip("."), target.rstrip(".")
    if c == t:
        return True
    a, b = c.split("."), t.split(".")
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def kb_markdown_files():
    """Every KB markdown file that may carry citations (excerpts are targets)."""
    for path in sorted(KB.rglob("*.md")):
        if EXCERPTS in path.parents:
            continue
        yield path


def check_citations(paper_keys: set):
    # Keys are lowercase-canonical in papers.json; compare case-insensitively.
    keys_lower = {k.lower() for k in paper_keys}
    paper_cite_keys: set[str] = set()
    excerpt_cites: set[str] = set()
    file_count = 0
    note_count = 0
    drafts = 0
    stubs = 0
    tag_total = defaultdict(int)

    for path in kb_markdown_files():
        file_count += 1
        rel = path.relative_to(THEORY)
        text = path.read_text()
        body = text
        is_note = NOTES in path.parents

        if is_note:
            note_count += 1
            if not text.startswith("---"):
                errors.append(f"{rel}: missing YAML frontmatter")
                continue
            end = text.find("---", 3)
            if end == -1:
                errors.append(f"{rel}: malformed frontmatter")
                continue
            fm = text[3:end]
            status_m = re.search(r"^status:\s*(\S+)", fm, re.MULTILINE)
            if not status_m:
                warnings.append(f"{rel}: no status in frontmatter")
            elif status_m.group(1) == "draft":
                drafts += 1
            elif status_m.group(1) == "stub":
                stubs += 1
            body = text[end + 3 :]

        # Report file-absolute line numbers even though citation scanning runs
        # over the post-frontmatter body.
        line_of = _line_index(body, text.count("\n", 0, len(text) - len(body)))

        # Paper-key citations: `[key §X ...]` and the bare `[key]` form.
        for regex, form in ((PAPER_CITE_RE, "§"), (BARE_CITE_RE, "bare")):
            for m in regex.finditer(body):
                key = m.group(1)
                # `[kb/notes/... §X]` / `[kb/excerpts/... §X]` are intra-KB
                # citations, checked separately below — not paper keys.
                if key.startswith("kb/") or key.lower() in NOT_A_KEY:
                    continue
                if form == "bare" and _skip_bare(key, body, m.start()):
                    continue
                paper_cite_keys.add(key)
                if key.lower() not in keys_lower:
                    errors.append(
                        f"{rel}:{line_of(m.start())}: cites unknown paper-key "
                        f"{key!r} ({form} form)"
                    )

        # `[key, arXiv NNNN.NNNNN]` — the key sits ahead of a comma rather than
        # a `§` or a `]`, so neither form above sees it. The payload is split on
        # BOTH `;` and `,`, so every key-shaped token is checked, not only the
        # first: `[k1, arXiv …; k2, arXiv …]` and comma-separated key *lists*
        # (`[fineweb2024, data-mixing-laws-2024]`) alike. A segment counts as a
        # key only if it is *entirely* one key-shaped token — `arXiv 2601.06423`
        # and prose fragments have interior spaces and are skipped.
        for m in re.finditer(r"\[([^\[\]]+)\]", body):
            payload = m.group(1)
            # Markdown links (`[text](url)`) and reference links (`[text][ref]`)
            # are not citations.
            if body[m.end() : m.end() + 1] in ("(", "["):
                continue
            # A single-token payload is the bare `[key]` form, already checked
            # above; re-checking it here would double-report every unknown key.
            if ";" not in payload and "," not in payload:
                continue
            for seg in re.split(r"[;,]", payload):
                sm = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9.\-]*)\s*", seg)
                if not sm:
                    continue
                key = sm.group(1)
                if key.lower() in NOT_A_KEY or not KEY_SHAPED_RE.search(key):
                    continue
                # A naked 4-digit year is a date in prose (`[Rewarding Progress,
                # 2024]`), never a paper key.
                if re.fullmatch(r"\d{4}", key):
                    continue
                paper_cite_keys.add(key)
                if key.lower() not in keys_lower:
                    errors.append(
                        f"{rel}:{line_of(m.start())}: cites unknown paper-key "
                        f"{key!r} (arXiv form)"
                    )

        # Anchorless `kb/excerpts/<key>` — existence of the file only.
        for m in EXCERPT_FILE_RE.finditer(body):
            end = m.end()
            if body[end : end + 1] == "#":
                continue  # anchored form, handled below
            # A citation that ends a sentence (`… kb/excerpts/foo.`) captures the
            # sentence period into the stem; strip trailing dots on both sides of
            # the `.md` removal so `foo.`, `foo.md` and `foo.md.` all reduce to
            # `foo`.
            stem = m.group(1).rstrip(".")
            if stem.endswith(".md"):
                stem = stem[:-3]
            stem = stem.rstrip(".")
            if not (EXCERPTS / (stem + ".md")).exists():
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites missing excerpt file "
                    f"kb/excerpts/{stem}.md"
                )

        # kb/excerpts/<key>#anchor citations — file *and* anchor.
        for m in EXCERPT_CITE_RE.finditer(body):
            excerpt_rel, anchor = m.group(1), m.group(2)
            # `kb/excerpts/foo.md#bar` is as valid as `kb/excerpts/foo#bar`.
            excerpt_rel = excerpt_rel.removesuffix(".md")
            excerpt_cites.add(f"{excerpt_rel}#{anchor}")
            ep = THEORY / (excerpt_rel + ".md")
            if not ep.exists():
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites missing excerpt file "
                    f"{excerpt_rel}.md"
                )
                continue
            if anchor not in anchors_of(ep):
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites missing anchor "
                    f"#{anchor} in {excerpt_rel}.md"
                )

        # §-consistency of the hybrid `[key §X; kb/excerpts/key#anchor]` form:
        # the label in the `§` slot must agree with the section the targeted
        # excerpt heading names for itself. Any `§` label in the bracket
        # satisfies any anchor in it — brackets naming several of each are rare
        # and the association between them is not recoverable syntactically.
        for m in HYBRID_CITE_RE.finditer(body):
            payload = m.group(1)
            cited = [s.rstrip(".") for s in CITE_SECTION_RE.findall(payload)]
            if not cited:
                continue
            for em in EXCERPT_CITE_RE.finditer(payload):
                stem = em.group(1).removesuffix(".md")
                ep = THEORY / (stem + ".md")
                if not ep.exists():
                    continue
                targets = section_designations_of(ep).get(em.group(2))
                if not targets:
                    continue
                if any(_section_agrees(c, t) for c in cited for t in targets):
                    continue
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites §{'/§'.join(cited)} but "
                    f"anchor #{em.group(2)} in {stem}.md is "
                    f"§{'/§'.join(sorted(targets))}"
                )

        # kb/notes/<area>/<file>[#anchor] cross-references — file and anchor.
        for m in NOTE_CITE_RE.finditer(body):
            stem, anchor = m.group(1), m.group(2)
            target = NOTES / (stem + ".md")
            if not target.exists():
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites missing note kb/notes/{stem}.md"
                )
                continue
            if not anchor:
                continue
            if anchor.startswith("§"):
                ok = anchor[1:].rstrip(".") in section_numbers_of(target)
            else:
                ok = anchor in anchors_of(target)
            if not ok:
                errors.append(
                    f"{rel}:{line_of(m.start())}: cites missing anchor "
                    f"#{anchor} in kb/notes/{stem}.md"
                )

        for m in TAG_RE.finditer(body):
            tag_total[m.group(1)] += 1

    stats["kb_files_scanned"] = file_count
    stats["notes_total"] = note_count
    stats["notes_draft"] = drafts
    stats["notes_stub"] = stubs
    stats["paper_keys_cited"] = len(paper_cite_keys)
    stats["excerpt_anchors_cited"] = len(excerpt_cites)
    stats["tag_counts"] = dict(tag_total)


def _line_index(body: str, offset_lines: int = 0):
    """Return offset -> 1-based, file-absolute line number.

    `offset_lines` is the number of newlines dropped ahead of `body` (the YAML
    frontmatter), so reported numbers match the file as edited.
    """
    starts = [0]
    for i, ch in enumerate(body):
        if ch == "\n":
            starts.append(i + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1 + offset_lines

    return line_of


def _skip_bare(key: str, body: str, start: int) -> bool:
    """True when a bare `[token]` is not a paper citation."""
    if key.lower() in NOT_A_KEY:
        return True
    if not KEY_SHAPED_RE.search(key):
        return True
    # A `^` immediately after `[` is a footnote.
    return body[start + 1 : start + 2] == "^"


def main():
    _data, paper_keys = check_papers_json()
    check_topics()
    check_citations(paper_keys)

    print("=" * 60)
    print("THEORY KB LINT REPORT")
    print("=" * 60)
    print(f"\nPapers in index:    {stats['papers_total']}")
    print(f"  with excerpts:    {stats['papers_with_excerpts']}")
    print(f"  with note refs:   {stats['papers_with_notes']}")
    print(f"\nTopics in topics.md: {stats['leaves_total']}")
    print(f"  by status: {stats['leaves_by_status']}")
    print(f"\nKB files scanned:    {stats['kb_files_scanned']}")
    print(f"Notes written:       {stats['notes_total']}")
    print(f"  draft:             {stats['notes_draft']}")
    print(f"  stub:              {stats['notes_stub']}")
    print(f"\nDistinct paper-keys cited inline: {stats['paper_keys_cited']}")
    print(f"Distinct excerpt anchors cited:   {stats['excerpt_anchors_cited']}")
    print(f"Tag counts: {stats['tag_counts']}")

    print(f"\n--- ERRORS ({len(errors)}) ---")
    for e in errors:
        print(f"  ! {e}")

    print(f"\n--- WARNINGS ({len(warnings)}) ---")
    for w in warnings:
        print(f"  - {w}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
