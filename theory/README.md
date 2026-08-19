# theory/

LLM theoretical-framework workspace. Two-layered:

1. **`kb/`** — knowledge base. Modular, citation-grounded notes plus
   verbatim source excerpts plus structured indices. Source of truth for
   technical claims throughout this project.
2. **`series/`** — the built five-paper LaTeX series drawn from the KB
   (architecture, training, reasoning, interpretability,
   evaluation-alignment), cross-referenced with `xr-hyper`. Built PDFs are
   committed at `series/paper-N/main.pdf`, with by-topic symlinks collected
   into `series/dist/` by `build.sh collect` (generated, not committed —
   the directory is gitignored and absent from a fresh clone until you
   build). Rebuild with `bash theory/series/build.sh` — there is
   no `theory/Makefile`. The series was audited by the 2026-05-06 six-lens
   review wave; findings and their resolution are in `reviews/`. Structure
   and per-paper layout: `series/README.md`.

## Layout

```
theory/
├── kb/                # the knowledge base
│   ├── notes/         # digested synthesis, one file per topic
│   ├── excerpts/      # verbatim quoted passages from papers
│   ├── index/         # papers.json, topics.md, timeline.md
│   └── glossary.md
├── sources/           # primary source PDFs + selectively archived forum threads
├── plans/             # phase-scoped research/construction plans
├── archive/           # historical snapshots (pre-expansion v1, dated PDFs)
├── series/            # 5-paper LaTeX series (built; see series/README.md)
├── reviews/           # 2026-05-06 six-lens review wave + resolution log
└── docs/design/  # design specs for the KB expansion
```

## Where to start

- **New to the KB?** Read `kb/README.md` for citation rules and epistemic
  conventions.
- **Looking for a topic?** Check `kb/index/topics.md` for the topic graph and
  per-topic status.
- **Tracing a claim?** Look up the paper-key in `kb/index/papers.json`,
  then read the matching `kb/excerpts/<key>.md` for verbatim source text.
- **Wondering what was here before?** See
  `archive/2026-05-03-pre-expansion/` for the v1 single-LaTeX-doc state.

## Citation discipline

Technical claims about LLM architecture, training, inference, or related
theory must cite either `[paper-key §X]` or
`[kb/notes/<area>/<file>#<anchor>]`. Analogies and intuitions must be tagged
`[ANALOGY]` / `[INTUITION]`. See project `CLAUDE.md` § "Theory KB & citation
discipline" for the full rule.

## Status

- **v1 (pre-expansion, archived)** — single LaTeX/PDF "core architecture" doc
  covering Transformer 2017 → LLaMA 2023. Under
  `archive/2026-05-03-pre-expansion/`; its `Makefile` is not the current build
  path.
- **v2 KB substrate — complete.** 55 leaf topic notes across architecture,
  training, post-training, inference, scaling, reasoning, interpretability,
  evaluation, and alignment, with verbatim excerpts, `index/papers.json`, and
  a cited glossary. Phase 2 closed 2026-05-04.
- **v2 LaTeX series — complete.** All five papers written and built (70
  sections, ~424 pp — 77 / 106 / 81 / 77 / 83, measured from the committed
  PDFs); PDFs committed at `series/paper-N/main.pdf`.
- **2026-05-06 review wave — applied.** Six independent fresh-context
  reviewers (adversarial content, cross-paper coherence, math correctness,
  citation tier, frontier currency, pedagogy) audited the series; the
  triage and per-finding resolution are in `reviews/00-summary.md`.
- **Since then: maintenance mode.** Changes are corrections, currency
  refreshes, and new cross-links from the research arcs — not new phases.
  Run `python3 kb/lint.py` from `theory/` after any citation edit. It exits
  non-zero today: **14 errors / 0 warnings expected as of 2026-08-19**, all
  of them citations of 10 paper keys missing from `kb/index/papers.json`
  (issue #55). `kb/README.md` lists the keys — anything beyond that baseline
  is a regression.
