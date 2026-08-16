# Working in this repo — worktree discipline (hard rule)

This repo may run concurrent Claude Code sessions; to keep them from
clobbering each other's uncommitted work, **each session works in its own git
worktree** (`.claude/worktrees/<scope>/`, gitignored) on its own branch, never
on the main checkout. Branch → push → PR (`gh pr create`) → merge via PR →
`git worktree remove`. Only integration commits (merges, or edits to
`CLAUDE.md`/`.gitignore`) land directly on the default branch.

---

# Purpose

LLM-interpretability research workspace: a citation-grounded theory knowledge
base (`theory/`), experimental research arcs (`research/`), and the analysis /
figure / audit pipeline (`examples/`). Depends on **two** sibling editable
installs — neither is on PyPI, so neither can be declared in `pyproject.toml`:

```bash
pip install -e ../llm-surgeon      # llm_surgeon.probe / .surgery — all arcs
pip install -e ../jacobian-lens    # jlens — every examples/jspace_*.py
pip install -e '.[dev]'            # torch, numpy, matplotlib + pytest
```

Missing either one fails only at `import` inside a script, which for the arc-04
fits is after model load, at the start of a multi-hour GPU run.

**Every checkout needs its own `.venv` — including each worktree.** Pyright
resolves `venvPath` relative to the config file, so a worktree (the mandated
way of working here) does not see the main checkout's environment. Link it
back before type checking, run from the worktree root:

```bash
ln -s ../../../.venv .venv        # inside .claude/worktrees/<name>/
```

Skip this and `pyright examples/` reports hundreds of phantom errors against
correct code. The symptom is **not stable**, which is the trap — all four rows
measured in the same worktree on 2026-07-29:

| State | Errors | Dominant rule |
|---|---|---|
| `.venv` linked | **0** | — |
| link removed after a previously-resolved run | 374 | 355 `reportAttributeAccessIssue`, 1 `reportMissingImports` |
| never linked, sibling checkouts present | 237 | all `reportMissingImports` |
| never linked, no siblings | 304 | all `reportMissingImports` |

The 374 row is the dangerous one. A wall of `reportMissingImports` reads as
environmental and sends you looking at your setup. But 355
`reportAttributeAccessIssue` — "Attribute `savefig` is unknown" — reads as
*real type bugs in your own code*: the modules resolve, their types do not.
If you are about to "fix" a pile of attribute errors in code that was green
yesterday, check for `.venv` first.

Deleting `venvPath` is not a workaround: pyright does not fall back to the
interpreter that launched it, so the errors persist unchanged.

## Structure

- `theory/` — Knowledge-base substrate (**GROUND TRUTH** for technical claims).
  - `kb/notes/<area>/<topic>.md` — digested synthesis, one file per topic
  - `kb/excerpts/<paper-key>.md` — verbatim quoted passages from primary sources
  - `kb/index/` — `papers.json` (metadata + KB cross-refs), `topics.md`, `timeline.md`
  - `kb/glossary.md` — every technical term used here, with a citation
  - `sources/papers/` — primary-source PDFs (`{paper-key}_{slug}.pdf`)
  - `sources/forums/` — selectively archived blog/forum snapshots (provenance only)
  - `series/` — 5-paper LaTeX series (architecture, training, reasoning,
    interpretability, evaluation-alignment) with `xr-hyper` cross-refs
  - `archive/2026-05-03-pre-expansion/` — v1 single-LaTeX-doc snapshot (its
    `Makefile` is the only Makefile in the repo)
  - `docs/design/` — design specs; `plans/` — KB-build planning history;
    `reviews/` — series review passes (adversarial, math, citations, …)
- `research/` — Investigations as **arcs** under `research/arcs/<slug>/`, plus
  `research/observations/` (one-offs) and `research/archive/`. Flagship:
  `research/arcs/01_nla-verbalizer/`.
- `examples/` — `nla_*.py` capture/analysis/render/audit scripts;
  `examples/README_NLA.md` holds pipeline conventions.

# Build commands

```bash
# Theory LaTeX series — shell-script build (NOT `make`; there is no theory/Makefile)
bash theory/series/build.sh            # clean + build all 5 papers + collect dist/
bash theory/series/build.sh collect    # re-collect dist/ symlinks only

# NLA audit — re-derive every load-bearing numerical claim from .pt artifacts
python examples/nla_audit_findings.py
```

# Theory KB & citation discipline — non-negotiable

When making technical claims about LLM architecture, training, inference,
interpretability, evaluation, alignment, or related theory:

1. **Every load-bearing claim cites a source.** One of:
   - `[paper-key §X, eq.Y]` — a paper in `theory/kb/index/papers.json`
   - `[kb/notes/<area>/<file>#<anchor>]` — into a synthesis note
   - `[kb/excerpts/<paper-key>#<heading>]` — into a verbatim excerpt
2. **Verify against the original PDF before propagating a KB-note claim** into
   LaTeX, code, or commit messages. The KB is digested; the paper is canonical.
3. **Analogies and intuitions are tagged, never asserted as fact** — `[ANALOGY]`,
   `[INTUITION]`, `[CONTRADICTION]`, `[FORUM-SIGNAL]`, `[SPECULATION]`. Analogies
   always return to the canonical symbolic form.
4. **If a claim depends on something not in the KB, add it before continuing.**
5. **Forum/blog citations are discovery signals only** (tier B/C). They never
   solely back a hard claim — only primary papers (tier A) can.

## Source tiers

- **Tier A (canonical):** arxiv, peer-reviewed venues, official tech reports /
  model cards, reference repos. Under `theory/sources/papers/`. Backs hard claims.
- **Tier B (high-signal commentary):** vendor/lab research blogs, named
  researchers' writeups. Cite alongside an underlying tier-A source.
- **Tier C (community signal):** Reddit/HN/X/HF. Discovery only; never the sole
  citation.

## Writing-style rule (Feynman bar)

Each topic note: formal definition (math + variables defined underneath) →
mechanism (how it computes, with tensor shapes) → variants/lineage (cited) →
tagged `[INTUITION]`/`[ANALOGY]` (always returning to canonical symbolic form) →
frontier and open questions (`[CONTRADICTION]` where sources disagree). When
introducing a new technical term, add it to `theory/kb/glossary.md` with a
citation.

# Research arcs & observations

Findings inside a focused investigation go in that arc's
`research/arcs/<slug>/observations/`; one-off findings go in
`research/observations/`. Follow `research/ARC_PROCESS.md` for the arc
lifecycle (question → capture → analyze → figures → observations → audit →
synthesis → PR).

**HARD RULE — raw data is a deliverable.** When an experiment produces a
dataset a figure or claim depends on, generating, validating, and saving the
raw dataset is part of the task. Commit `.pt`/`.npz`/`.csv` artifacts to the
arc's `research/arcs/<slug>/data/` (Git LFS via the `research/**/data/*.pt`
rule) with a checksummed `MANIFEST.json`, so a clean clone re-renders every
figure and replays the audit. Full discipline in `research/ARC_PROCESS.md`
§ "Raw data is a deliverable".

Each observation file (`YYYY-MM-DD-<slug>.md`) includes: Date and context
(experiment, model, params) · Finding · Evidence (output/transcript excerpts) ·
Reproducibility (exact commands/code) · Hypotheses · Follow-ups · References.

# Third-party data — vet BEFORE first use, not before commit

**Any external dataset, corpus, or model artifact entering this repo gets a
rights-and-privacy check at the moment it is selected**, recorded in the arc's
decision log alongside the scientific rationale. This is a hard gate, not a
pre-commit cleanup step: by commit time the experiments are already run and
re-running them is the expensive part.

Record all four, or don't use the dataset:

1. **Licence + attribution requirements.** Name the licence and what it
   obliges (ODC-BY §4.2/4.3 want the licence URI and a source-attribution
   notice; CC BY-SA wants attribution and is *not* one-way compatible with
   GPLv3 below 4.0). Put them in a `LICENSE-DATA.md` beside the data. The
   repo's own `GPL-3.0-only` covers code and original prose **only** and must
   be explicitly scoped away from third-party data.
2. **Does it contain personal data?** Scraped-web corpora (C4, OSCAR,
   RefinedWeb, The Pile, anything Common-Crawl-derived) are filtered for
   *quality*, never for *privacy*, and carry contact details at a measurable
   base rate. Curated encyclopedic sources (WikiText, Wikipedia dumps) largely
   do not. **Assume PII is present in any web-scraped corpus and prove
   otherwise** — `examples/jspace_redact_corpus.py --report` is the starting
   scanner; extend its pattern classes rather than writing a new one.
3. **The realism/privacy tradeoff, stated explicitly.** A corpus chosen for
   being *more representative of real text* is, for exactly that reason, more
   likely to contain real people's data. If the scientific argument for a
   dataset is its breadth or naturalness, that argument is itself the signal
   to check. Arc 04 is the worked example — see its README warning.
4. **Redistribution decision.** Committing raw third-party text republishes it
   under your name. Prefer committing a deterministic regeneration script plus
   a checksum where "raw data is a deliverable" (below) still holds; where the
   text itself must be committed, redact PII first and document the redaction,
   its class coverage, its known limits, and how to reproduce it exactly.

No licence can authorise republishing a third party's personal data —
data-subject rights attach to the person, not the licensor. Treat PII removal
as a scientific-integrity and ethics obligation, independent of any liability
question.

# Git LFS is REQUIRED

`research/**/figures/*.png` and `research/**/data/*.pt` are tracked via Git LFS
(see `.gitattributes`). Run `git lfs install` before working the repo, or those
files appear as phantom modifications. Recover an LFS-less clone with
`git lfs install && git lfs pull`.

# Type checking

Project stance: zero pyright errors, warnings, and informations after every
edit. Never disable rules to quiet diagnostics — fix the source or narrow with
the tier list (`assert isinstance` > `cast` > `# pyright: ignore[reportXxx]`;
never bare `# type: ignore`). `numpy` and `matplotlib` have stubs; the
`llm_surgeon` runtime comes from the sibling editable install.

