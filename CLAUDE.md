# HARD RULE: every session MUST work in a git worktree

**Not a guideline. A hard rule.** This repo may run concurrent Claude Code
sessions on different parts of the tree (theory KB, the LaTeX series,
research arcs, the examples pipeline). When two sessions share the main
checkout, uncommitted work from one gets swept into the other's `git add`
and commit. Worktrees are the fix.

## Pre-flight check — BEFORE your first edit/write/bash-write

Run this as the very first thing in any session:

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
[ "$GIT_DIR" = "$GIT_COMMON" ] && echo "MAIN CHECKOUT — MUST CREATE WORKTREE" || echo "in worktree — proceed"
```

If it says "MAIN CHECKOUT": **STOP. Do not edit any file.** Create a worktree
at `.claude/worktrees/<scope>/` on its own branch (via `EnterWorktree`) and
work there. The only commits permitted directly on the default branch in the
main checkout are integration commits: merge commits (`gh pr merge`), or
convention-establishing changes to `CLAUDE.md` and `.gitignore` themselves.
Everything else — even one-line typo fixes — goes through a worktree.

## Session lifecycle

1. `EnterWorktree name=<scope>` — creates `.claude/worktrees/<scope>/` and
   switches the session into it. Conventional branch prefixes: `feat/`,
   `fix/`, `refactor/`, `docs/`, `research/`, `session/`.
2. All edits and commits land on the session's branch in its worktree. Never
   edit files in another session's worktree. Don't `cd` back to the main
   checkout to edit.
3. Push the branch and open a PR (`gh pr create`). Merge to the default
   branch via PR. After merge, remove the worktree
   (`git worktree remove .claude/worktrees/<scope>`).

`.claude/` is gitignored — no `.gitignore` changes needed for worktrees.

---

# Purpose

LLM-interpretability research showcase: a citation-grounded theory knowledge
base (`theory/`), experimental research arcs (`research/`), and the analysis /
figure / audit pipeline (`examples/`). Depends on the sibling `llm-surgeon`
toolkit — install editable with `pip install -e ../llm-surgeon`, then
`pip install -e .` for this repo's direct deps (numpy, matplotlib).

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
- `research/` — Investigations as **arcs** under `research/arcs/<slug>/`, plus
  `research/observations/` (one-offs) and `research/archive/`. Flagship:
  `research/arcs/nla-verbalizer/`.
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

