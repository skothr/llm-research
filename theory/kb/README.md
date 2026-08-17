# theory/kb/

Knowledge base — modular, citation-grounded synthesis of LLM theory and
practice. Designed so retrieval (RAG or human) cannot confuse digested
synthesis with primary-source quotation.

## Three-tier epistemic separation

| Tier | Directory | What it contains | Can solely back a hard claim? |
|------|-----------|------------------|-------------------------------|
| 1 — Verbatim | `excerpts/<paper-key>.md` | Quoted passages from a paper, anchored to section/equation. Every line in a blockquote. | **Yes** (it's the paper, quoted) |
| 2 — Synthesis | `notes/<area>/<topic>.md` | Digested explanation. Each load-bearing claim cites either a paper or an excerpt anchor. Analogies tagged. | Only when the cited source itself does. |
| 3 — Structure | `index/papers.json`, `index/topics.md`, `index/timeline.md`, `glossary.md` | Indices and cross-references. Not claims, but the navigation map. | N/A |

## Citation format

- `[paper-key §X, eq.Y]` — points to a paper in `index/papers.json`. Examples:
  `[vaswani2017 §3.2, eq.1]`, `[touvron2023 Tab.2]`.
- `[kb/notes/<area>/<file>#<anchor>]` — points to a synthesis note. Use full
  path from `theory/`. Anchor is the markdown heading slug.
- `[kb/excerpts/<paper-key>#<heading>]` — points to a verbatim excerpt.

## Linting the KB

`kb/lint.py` checks every citation in `kb/**/*.md` (notes, `glossary.md`,
`index/`) against reality: paper-keys resolve in `index/papers.json` (both the
`[paper-key §X]`, bare `[paper-key]`, and `[paper-key, arXiv NNNN.NNNNN]`
forms), excerpt and note cross-references resolve to
a real **file and a real anchor** (the anchorless
`[kb/excerpts/<paper-key>]` fallback is checked for the file), the `§X` half
of a hybrid `[<paper-key> §X; kb/excerpts/<paper-key>#<heading>]` agrees with
the section the target excerpt heading names for itself
(`## §6.3 Robustness … {#sec-6-3-robustness}` ⇒ `§6.3`), `topics.md` leaves
have notes, and notes carry
frontmatter. Run it from `theory/` after any citation edit; it collects every
error, prints them all, and exits non-zero if any:

```bash
python3 kb/lint.py
```

Anchors it accepts in a target file: a `{#attr}` attribute, the
`## #anchor — Title` heading convention used by older excerpt files, a plain
markdown heading slug, and the `#§3.1` section-number shorthand used for note
cross-references.

## Tagging analogies and intuition

Lines or paragraphs that are not formal claims must be prefixed with an
explicit tag so retrieval cannot launder them as fact:

- `[ANALOGY]` — analogical framing (e.g., "Q is the question being asked").
- `[INTUITION]` — re-description of the formal mechanism in plain language.
- `[CONTRADICTION]` — sources disagree on this point; both cited inline.
- `[FORUM-SIGNAL]` — claim shape derived from a community-tier source; full
  discussion lives in the linked `theory/sources/forums/` snapshot. Always
  paired with a tier-A citation that backs the claim itself.
- `[SPECULATION]` — author's own framing not directly grounded in a cited
  source. Use sparingly.

Analogies must always **return** to the canonical symbolic form — never
leave the reader stranded on an analogy that lacks a math anchor.

## Topic note structure

Standard layout for `notes/<area>/<topic>.md`:

```
---
topic: <topic-name>
area: <area>
status: stub | draft | reviewed | locked
last-updated: YYYY-MM-DD
primary-sources: [<paper-key>, ...]
related-notes: [<area>/<topic>, ...]
---

# <Topic title>

## Formal definition
<the math, with every variable defined immediately under the equation,
citation: [paper-key §X, eq.Y]>

## Mechanism
<how the math actually computes — what each term does at runtime;
tensor shapes belong here>

## Variants and lineage
<bulleted list, each with a citation>

## [INTUITION] / [ANALOGY]
<tagged paragraph(s) — must always return to the canonical symbolic form>

## Frontier and open questions
<recent papers, contested territory, [CONTRADICTION] markers>

## See also
<cross-links to related notes>
```

## Status field

`status:` in note frontmatter:

- `stub` — file created, placeholder content only
- `draft` — first pass written, not reviewed
- `reviewed` — claims spot-checked against PDFs, citations verified, no
  internal contradictions
- `locked` — frozen for the current research cycle (e.g., before a LaTeX
  paper writeup)

## Conventions

- **Paper-key format:** `<lastname-first-author><year>` (`vaswani2017`,
  `dao2022`). Letter suffix for same author/year (`dao2023a`, `dao2023b`).
  Shorthand allowed when conventional (`flashattn2`, `mamba2`,
  `deepseek-v3`). Tech reports get vendor prefix
  (`anthropic-claude-3-model-card`).
- **Markdown anchors are stable** — never rename a heading without updating
  every referrer.
- **Glossary updates** when introducing a new technical term — add to
  `glossary.md` with a citation.
