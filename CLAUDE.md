# Working in this repo

Work on a branch cut from an up-to-date `main`, in the main checkout: branch →
push → PR (`gh pr create`) → review loop (described in § The checkpoint PR
in `research/ARC_PROCESS.md`; it applies to every PR) → the owner merges.
Nothing is committed on `main` directly, `CLAUDE.md` and `.gitignore` edits
included. The owner runs one session at a time in this repo; parallel work
runs inside that session as subagents or workflows. A git worktree
(`.claude/worktrees/<name>/`, gitignored) is therefore optional: use one for
work that has to run beside other work, such as a long GPU job or a parallel
agent that edits files. Uncommitted changes you find at session start are not
yours: `git checkout -b` carries them onto your branch, so stage only the
files you touched. Remove a worktree with `git worktree remove` when its work
is merged, after checking `git status --ignored`: gitignored outputs such as
`.cache/` and `research/arcs/*/data/cache/` (fit checkpoints, caches) are
deleted with it and are not carried by the merge.

**Keeping this in step with the global workflow.** The workflow above and
`research/ARC_PROCESS.md` § The checkpoint PR restate the owner's global SOP
(branching, review loop, merge). When a session finds this local text and the
global SOP disagree, it follows the local text and proposes a PR. If the
global rule fits this repo's research intent (reproducible runs, raw data as a
deliverable, citation discipline, and § Third-party data's rights and privacy
vetting), the PR updates the local text; if it does not, the PR records the
divergence below with its reason. The owner decides by merging or closing it.

Recorded divergences: none.

**One PR = one scope.** Keep each PR small enough to review in one sitting —
split an arc into staged PRs (data + capture / analysis + figures / README
synthesis) rather than one mono-diff. Review findings outside the PR's stated
scope are filed as issues, never fixed in the branch; if fixing a finding
would grow the PR past one-sitting reviewability, split it.

---

# Purpose

LLM-interpretability research workspace: a citation-grounded theory knowledge
base (`theory/`), experimental research arcs (`research/`), and the analysis /
figure / audit pipeline (`examples/`). Depends on **one** sibling editable
install, which is not on PyPI and so cannot be declared in `pyproject.toml`:

```bash
pip install -e ../jacobian-lens    # jlens — every examples/jspace_*.py
pip install -e '.[dev]'            # torch, transformers, numpy, matplotlib, ... + pytest
```

Missing it fails only at `import jlens` inside a script, which for the arc-04
fits is after model load, at the start of a multi-hour GPU run. Model loading
and the NLA probe are in-repo (`examples/_hf_models.py`,
`examples/_nla_probe.py`).

**Every checkout needs its own `.venv`, including any worktree.** Pyright
resolves `venvPath` relative to the config file, so a worktree does not see
the main checkout's environment. If you work in one, link it back before
type checking, run from the worktree root:

```bash
ln -s ../../../.venv .venv        # inside .claude/worktrees/<name>/
```

Skip this and `pyright examples/` reports hundreds of phantom errors against
correct code. The symptom is **not stable**, which is the trap — all four rows
measured in the same worktree on 2026-07-29:

| State | Errors | Dominant rule |
|---|---|---|
| `.venv` linked | **0** | — |
| link removed after a previously-resolved run | 374 | 355 `reportAttributeAccessIssue` (+19 others, incl. 1 `reportMissingImports`) |
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
- `examples/` — per-arc script families by prefix (`nla_*`, `emb_*`,
  `jspace_*`; arc 02 has a single `subliminal_*` script): capture / analyze /
  render / audit. `examples/README_NLA.md` holds the NLA pipeline conventions
  the later families follow.

# Build commands

```bash
# Theory LaTeX series — shell-script build (NOT `make`; there is no theory/Makefile)
bash theory/series/build.sh            # clean + build all 5 papers + collect dist/
bash theory/series/build.sh collect    # re-collect dist/ symlinks only

# NLA audit — re-derive every numerical claim the prose relies on from .pt artifacts
python examples/nla_audit_findings.py
```

# Theory KB & citation discipline — non-negotiable

When making technical claims about LLM architecture, training, inference,
interpretability, evaluation, alignment, or related theory:

1. **Every claim a conclusion rests on cites a source.** One of:
   - `[paper-key §X, eq.Y]` — a paper in `theory/kb/index/papers.json`
   - `[kb/notes/<area>/<file>#<anchor>]` — into a synthesis note
   - `[kb/excerpts/<paper-key>#<heading>]` — into a verbatim excerpt
2. **Verify against the original PDF before propagating a KB-note claim** into
   LaTeX, code, or commit messages. The KB is digested; the paper is canonical.
3. **Analogies and intuitions are tagged, never asserted as fact** — `[ANALOGY]`,
   `[INTUITION]`, `[CONTRADICTION]`, `[FORUM-SIGNAL]`, `[SPECULATION]`. Analogies
   always return to the canonical symbolic form. These are epistemic tags
   on claims (used in `theory/kb` notes and, per `ARC_PROCESS.md` § Framing,
   in arc prose — where a hypothesis block is the equivalent); they are not
   attribution tags, and arc READMEs carry no per-quote provenance labels.
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

# Prose style (all prose and code comments)

State things literally; stock phrases such as "load-bearing", "genuinely" or <!-- prose-lint: allow -->
"sits at" stand in for a direct statement. Before committing prose or comments, <!-- prose-lint: allow -->
run `python scripts/prose_lint.py <paths>` from the repo's `.venv` (it needs
`markdown-it-py`, part of `.[dev]`). Reword or cut each hit, or keep it
where the phrase carries technical meaning or sits inside a verbatim quotation <!-- prose-lint: allow -->
(a quoted paper, transcript or forum passage). A kept hit gets
`prose-lint: allow` on its line, inside that file's comment syntax
(`<!-- -->` in Markdown, `#` in Python, `%` in LaTeX) so it does not render
(issue #120). The scanner never reports verbatim material (the
owner's quoted turns, fenced code blocks in Markdown, `theory/kb/excerpts/`,
`theory/sources/`), dated records (`research/archive/`, `theory/archive/`,
`theory/reviews/`) or the hash-pinned files listed in the scanner's
`EXCLUDED_FILES`, whose sha256 an audit checks (today
`examples/subliminal_step0_decode.py`); leave those as written. When an audit
pins another file, add it to that list. Fenced blocks also hold authored
examples; check the comments you write inside one by hand, since the scanner
skips them.

# Research arcs & observations

Findings inside a focused investigation go in that arc's
`research/arcs/<slug>/observations/`; one-off findings go in
`research/observations/`. Follow `research/ARC_PROCESS.md` for the arc
lifecycle: four checkpoints (question, research, plan → setup and
implementation → computation, processing, validation → observations,
conclusions, artifacts), each closed by its own reviewed PR, with 2-4
repeating as the arc iterates.

**HARD RULE — raw data is a deliverable.** When an experiment produces a
dataset a figure or claim depends on, generating, validating, and saving the
raw dataset is part of the task. Commit the artifacts, whatever their format,
to the arc's `research/arcs/<slug>/data/` with a checksummed `MANIFEST.json`
covering every data file (excluding the manifest itself, README, LICENSE-DATA and
audit transcripts), so a clean clone re-renders every figure and replays the audit.
`.gitattributes` routes only `.pt` under `data/` to Git LFS; JSON, JSONL or
CSV of 5 MiB or less stays in plain git, and anything larger, or another
binary format such as `.npz`, gets its own `.gitattributes` LFS rule before
its first commit. Full discipline in `research/ARC_PROCESS.md`
§ "Raw data is a deliverable".

Each observation file (`YYYY-MM-DD-<slug>.md`) includes: Date and context
(experiment, model, params, seed or "none" for a run that draws no random
numbers) · Finding · Evidence (output/transcript excerpts) · Reproducibility
(exact commands/code) · Hypotheses · Follow-ups · References.

# Showcase vs history — main is the showcase, commits are the log

Main-branch documents (the READMEs at every level, arc syntheses) present the
**final state**: current results and figures plus the important/critical
caveats and limitations a reader needs to weigh them. Commit history already
carries the transparency of how things changed — do not duplicate it into the
tree.

- No scattered historical notes, change chronologies, or correction mechanics
  in showcase documents. A correction that matters to the reader appears as
  its current outcome with a one-line dated pointer; the detailed trail lives
  in git history, `observations/`, and `plans/`.
- Keep main clean, organized, and complete: each document scoped to what a
  reader needs *now*, with historical and process detail compressed to
  pointers rather than deleted.
- The bar: an ML researcher skimming any README should quickly get a clear,
  transparent, and deliberately cautious picture of what it describes
  (issue #57 tracks the section standard implementing this). Section
  headings are the navigation surface — `grep '^#'` alone should tell a
  reader what the document covers. Every arc README must carry a
  `## Limitations` section (ranked by how far each limit constrains the
  claims) and a `## Attribution` section, both as real headings (retrofit of
  the existing arcs is #57).
- **Attribution is basic.** The set of moments where the owner's input shaped
  research direction, as dated verbatim quoted turns (`[session YYYY-MM-DD]`)
  with a line on what each one set, plus a short human / Claude / emergent
  split. One disclaimer line covers normalization and elisions. No label
  taxonomy, no per-quote provenance tags, no verifiability tables.
  `research/arcs/04_jspace/README.md` § Attribution is the shape (arc 02's
  § Attribution is the original reference implementation of the same
  content); the procedure is `research/ARC_PROCESS.md` § Arc README synthesis.

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
never bare `# type: ignore`). `numpy` and `matplotlib` have stubs; `jlens`
resolves through the `extraPaths` entries in `pyrightconfig.json`.

