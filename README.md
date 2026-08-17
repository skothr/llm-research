# llm-research

LLM-interpretability research, organized as a citation-grounded theory
knowledge base plus reproducible experimental arcs. Built on the
[`llm-surgeon`](https://github.com/skothr/llm-surgeon) toolkit for layer-level model surgery and
probing. The largest investigation applies Anthropic's released NLA
(Natural Language Autoencoder) verbalizer/reconstructor model pair to
local Qwen2.5-7B-Instruct and probes layer-20 hidden-state geometry; three
further arcs cover subliminal trait transfer, the structure of Qwen2.5-7B's
input-embedding table, and a partial replication of the J-lens / J-space
global-workspace result — that last one built on a second sibling toolkit,
[`jacobian-lens`](https://github.com/anthropics/jacobian-lens) (`jlens`). Both
siblings are required; see [Setup](#setup).

This repository is a **research workspace**, not a software product. It
collects the work product — the synthesized theory, the dated observations,
the figure/audit pipeline — rather than a polished library. Claims are held
to the standard described under "Epistemic discipline" below: load-bearing
technical claims cite a primary source, and findings are framed as
hypotheses until the evidence settles them.

It is also **exploratory, self-directed, agent-assisted** work. Much of the
implementation — capture and analysis scripts, figures, audit infrastructure,
observation drafts — was carried out by Claude Code sessions under direction;
the research questions, scope calls, and sign-off are the author's, and each
arc README records that split explicitly. Nothing here has been published,
externally reviewed, or replicated by anyone else, and the findings should be
read as provisional. The [Methodology](#methodology) below is the part offered
with confidence: it exists to make being wrong visible.

## Known corrections and open caveats

Corrections applied after an arc's results were first written up, plus the one
open caveat large enough to re-frame an arc. They are surfaced here because a
correction of this kind should be visible at the entry point rather than
discovered two levels down; each arc's own README carries the full record.

**Arc 04 — C4 personal-data redaction, closed 2026-08-16.** Arc 04's seeded
**C4-en** corpus slice carried 120 pieces of third-party personal data,
redacted 2026-07-29; every dependent result was re-run on the redacted text, no
headline conclusion changed, and the correction closed 2026-08-16. Mechanics,
per-claim blast radius, and the reproduction recipe:
[arc 04's DATA CORRECTION section](research/arcs/04_jspace/README.md#data-correction--c4-corpus-pii-redacted-2026-07-29-dependent-results-re-run-correction-closed-2026-08-16).

**Arc 01 — F3 polarity, corrected 2026-08-16.** A committed PC1 polarity was
inverted relative to the dated primary record by a transcription slip. The
magnitudes were audited; the polarity was not, so no audit check could catch
it. The load-bearing claim (PC1 *separates* content from function) is
unaffected. See the arc's
[Known corrections and errata](research/arcs/01_nla-verbalizer/README.md#known-corrections-and-errata).

**Arc 01 — open caveat (L2): the AV-decoder format bias is unaudited.** The
verbalizer's outputs share suspiciously consistent template phrasing, and
whether it emits those same templates on random hidden states as on
semantically-loaded ones was never tested. In the arc's own framing this is
"the single methodological audit that, if positive, would re-frame the whole
arc" — every interpretive reading in arc 01 would then be filtered through a
verbalizer prior. Still open; tracked as
[#8](https://github.com/skothr/llm-research/issues/8). See
[L2](research/arcs/01_nla-verbalizer/README.md#limitations-and-methodology-caveats).

**Arc 03 — F-T3 carrier-stability numbers, corrected 2026-08-17.** Two
hand-computed numbers did not re-derive from the committed artifact and were
restated: adjacent-layer top-10 carrier overlap is 4-9/10 (median 8), not
7-9/10, and two original block dims hold a top-10 slot at every layer of the
band, not three. The finding's substance — a stable mid-network carrier set
spanning L4-26 — is unaffected, and both corrected values are now audit-locked.
See the arc's
[Known corrections and errata](research/arcs/03_embedding-atlas/README.md#known-corrections-and-errata).

## What's here

```
theory/      Citation-grounded LLM-theory knowledge base + a 5-paper LaTeX series
research/    Experimental research, organized into arcs (focused investigations)
examples/    Per-arc capture / analysis / render / audit pipelines
```

- **`theory/`** — A knowledge-base substrate (v2 layout): `kb/notes/` digested
  synthesis (one file per topic), `kb/excerpts/` verbatim source passages,
  `kb/index/` (`papers.json`, `topics.md`, `timeline.md`), `kb/glossary.md`,
  and `sources/papers/` primary-source PDFs. `series/` holds a 5-paper LaTeX
  series (architecture, training, reasoning, interpretability,
  evaluation-alignment) with cross-paper references. Start at
  `theory/README.md`.
- **`research/`** — Investigations organized into **arcs** under
  `research/arcs/<slug>/`, each cohering around one research question, plus
  `research/observations/` for one-off findings and `research/archive/` for
  retired material. Four arcs exist; **two are paused mid-program**, not
  finished. The largest is `research/arcs/01_nla-verbalizer/`; the two
  carrying the most informative epistemics are `03_embedding-atlas` (predictions
  registered before the runs, two of them falsified) and `04_jspace` (a partial
  replication that came out weaker than the original, with a causal split).
  The arc lifecycle and reproducibility disciplines are in
  `research/ARC_PROCESS.md`; the per-arc status index is `research/README.md`.
- **`examples/`** — Per-arc pipeline scripts, prefixed by arc: `nla_*`
  (arc 01), `subliminal_*` (arc 02), `emb_*` (arc 03), `jspace_*` (arc 04).
  Each family covers capture (writes `.pt` artifacts), analysis, figure
  render (matplotlib), and an `*_audit_findings.py` that re-derives that arc's
  load-bearing numerical claims from committed artifacts.
  `examples/README_NLA.md` documents the `nla_*` pipeline specifically — the
  other three families follow the same artifact/audit shape but have no
  separate conventions doc; their arc READMEs carry the per-arc detail. Some
  scripts import `llm_surgeon` (model loading, probing, surgery); the rest are
  render/analysis-only (torch / numpy / matplotlib).

## Methodology

The arcs under `research/arcs/` are run to a common discipline, specified in
[`research/ARC_PROCESS.md`](research/ARC_PROCESS.md). It is stated here because
it is the part of this work offered with any confidence — and because where it
is *not* applied uniformly, that should be visible from the entry point rather
than discovered two levels down.

- **Predictions before runs — unevenly.** Arc 03 is the strongest case: seven
  predictions registered on 2026-06-11, before the attention captures they
  constrain, adjudicated mechanically at close as **P1a PASS, P1c FAIL, P1d
  FAIL**, with P2 refined-not-falsified and **P1b, P1e and P3 never run**
  ([`plans/2026-06-11-predictions.md`](research/arcs/03_embedding-atlas/plans/2026-06-11-predictions.md)).
  Arc 02's plan states falsifiable predictions per hypothesis and an explicit
  pre-commitment clause, though the arc paused before the tests they govern.
  Arc 04's README records one robustness axis (the quantization control) as
  genuinely pre-registered in its design plan, the rest gated on thresholds
  fixed before each run. Arc 01 grew from open-ended themes with no
  pre-registration. Three of the four registers are partial — read each arc's
  own account rather than this summary.
- **Audit scripts.** `examples/*_audit_findings.py` re-derive an arc's
  load-bearing numbers from its committed artifacts, so a figure quoted in
  prose that has drifted from the artifact it came from fails the audit. Arcs
  01, 02, 03, and 04 each have one; arc 02's was added in the 2026-08-17 sweep,
  replacing the by-hand checking its smaller Step-0 number set had relied on.
  These audits check **arithmetic consistency only** — they cannot catch a
  methodological error, a capture-protocol bug, or interpretive overreach.
  Arcs 01, 02, and 03 state that limitation in their READMEs; arc 04 states it
  in its audit block.
- **Datasets committed and pinned.** Arcs 01, 03, and 04 commit raw `.pt`
  artifacts under the arc's `data/` (Git LFS) with a `MANIFEST.json` recording
  per-file sha256 and provenance; arc 02's Step-0 data is JSONL under an
  interim manifest schema. The intent is that a clean clone re-renders every
  figure and replays every audit — with one documented exception: arc 04's
  full fitted-lens tensors sit behind an opt-in LFS download (three are
  committed; the two 1.5B nf4 lenses are pending issue #47), so 7 of its
  checks fail on a default clone — 3 LFS-stub reports + 4 `MISSING`
  (see [Running the research pipeline](#running-the-research-pipeline)).
- **Human/AI division of labor, recorded per arc.** Every arc README states
  what Claude Code sessions implemented and what was directed, constrained, and
  signed off by hand — down to which framings are the agent's paraphrase rather
  than the author's own words (arc 02) and which methodological problems the
  agent missed until a human raised them (arc 01, theme 9).
- **Negative results kept, not buried.** Arc 03's two falsified predictions and
  arc 04's four non-replications each get their own numbered write-up with the
  measurements attached, and the negatives have dedicated observation files.
  They are the strongest available evidence that the pre-registration and audit
  machinery is not decorative.

Counts that move as arcs develop — observation totals, figure totals, script
totals — are kept in the arc READMEs where they are maintained, not here. Audit
check-counts are the exception: they appear below with the date they were
re-derived, because a reader needs an expected value to compare a local run
against. Re-run the audit rather than trusting any number quoted here.

This is exploratory, self-directed work: unpublished, unreviewed, and open to
being wrong — the discipline above is there to make being wrong visible.

## Prerequisites — Git LFS is REQUIRED

Research figures (`research/**/figures/*.png`), raw datasets
(`research/**/data/*.pt`), the arc-04 lens cache
(`research/**/data/cache/*.pt`) and archived primary-source PDFs
(`theory/sources/papers/*.pdf`) are stored via Git LFS — see
`.gitattributes`. Install and initialize Git LFS **before** cloning or working
the repo, or those files show up as phantom modifications (the working tree
holds pointer files, not content):

```bash
git lfs install            # one-time, per machine
git clone <repo-url>       # LFS content then fetches on checkout
# already cloned without LFS? recover with:
git lfs install && git lfs pull
```

That recovery pull is deliberately incomplete in one place: arc 04's fitted-lens
cache is excluded from default LFS fetches by the committed `.lfsconfig`
(`fetchexclude`), because it is ~905 MiB that most uses of the repo never load.
Those files stay pointer stubs on purpose, and arc 04's audit reports them as
such rather than crashing. Opt in when you want them:

```bash
git lfs pull --include="research/arcs/04_jspace/data/cache/**" --exclude=""
```

[Running the research pipeline](#running-the-research-pipeline) below gives the
expected audit result in each of the two states.

## Setup

Python >= 3.10. The analysis/figure scripts depend on **two** sibling toolkits.
Neither is on PyPI, so neither resolves from an index; both are installed
editable from a checkout next to this repo:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llm-surgeon      # llm_surgeon.probe / .surgery — all arcs
pip install -e ../jacobian-lens    # jlens — every examples/jspace_*.py
pip install -e '.[dev]'            # torch, numpy, matplotlib + pytest
```

`pip install -e ../llm-surgeon` pulls in `torch`, `transformers`, `accelerate`,
`bitsandbytes`, and the rest of the surgery/probe runtime;
`../jacobian-lens` provides `jlens`, imported by every arc-04 `jspace_*`
script. This repo declares `llm-surgeon` plus the direct imports of its own
example scripts (`numpy`, `matplotlib`, `datasets`), and the `[dev]` extra adds
`pytest` for `examples/tests/`. Installing the editable siblings first satisfies
the `llm-surgeon` requirement and the undeclarable `jlens` one.

Missing either sibling fails only at `import` inside a script — which for the
arc-04 fits is after model load, at the start of a multi-hour GPU run.

## Building the theory LaTeX series

The current theory deliverable is the 5-paper series under `theory/series/`,
built by a shell script (not a Makefile — the only Makefile lives in the
archived v1 snapshot at `theory/archive/2026-05-03-pre-expansion/`):

```bash
bash theory/series/build.sh          # clean + build all 5 papers + collect PDFs
bash theory/series/build.sh collect  # re-collect dist/ symlinks only (skip build)
```

Output PDFs land in `theory/series/dist/<N>-<topic>.pdf`. The build is
sequential by necessity: each paper's `main.tex` declares cross-paper
references via `xr-hyper`, so sibling `main.aux` files must exist first; the
script runs four `pdflatex` passes per paper (one with `bibtex`, then three to
settle `xr-hyper` cross-refs), plus a conditional fifth pass for any paper
still showing unresolved cites. A LaTeX toolchain (`pdflatex`, `bibtex`) must
be on `PATH`, plus poppler-utils (`pdftotext`, `pdfinfo`) — the script's page
counts and its unresolved-cite detection silently no-op without them.

## Running the research pipeline

Capture scripts write `.pt` artifacts (working cache:
`research/arcs/<slug>/data/cache/` in arc 04, the repo-level `.cache/` in arcs
01 and 03 — both gitignored; the committed copies live in each arc's `data/`).
Render scripts turn artifacts into figures; each arc's audit re-derives that
arc's claims from them. All four re-measured on 2026-08-17, each arc
committing its run as `data/audit_2026-08-17.log`:

```bash
python examples/nla_audit_findings.py         # arc 01 → SUMMARY: 196 PASS | 0 FAIL
python examples/subliminal_audit_findings.py  # arc 02 → SUMMARY: 103 PASS | 0 FAIL | 5 UNVERIFIABLE
python examples/emb_audit_findings.py         # arc 03 → SUMMARY:  99 PASS | 0 FAIL
python examples/jspace_audit_findings.py      # arc 04 → SUMMARY: 951 PASS | 7 FAIL   (default clone; the committed log is the 986 | 4 with-cache run)
```

Arc 02's audit needs no GPU and no network and finishes in under a second; its
five `UNVERIFIABLE` lines are reported rather than scored, covering facts no
artifact in this repo can settle (the source paper's own numbers, the model
snapshot, the capture host).

Arc 04's 7 failures on a default clone are **expected**, not regressions: the
three lenses refit in the C4-redaction re-run are LFS-committed but excluded
from default LFS pulls (`.lfsconfig` — the ~905 MiB lens download is
opt-in), so the audit reports each as an `LFS pointer stub` naming the pull
command, and the two 1.5B nf4 lenses and their sidecars are regenerate-only
pending the scheduled refit (issue #47), each reported as a loud `MISSING`
rather than skipped — 3 stubs + 4 `MISSING` = 7. After
`git lfs pull --include="research/arcs/04_jspace/data/cache/**" --exclude=""`
the same run reports **986 PASS | 4 FAIL** (the nf4 `MISSING` reports only)
with no GPU work; the check total grows from 958 to 990 between the two
states because the lens-dependent blocks register their claims only when the
lens tensors are on disk. See the arc READMEs for what each audit does and does
not catch (arithmetic consistency only — never methodology or interpretation)
and for the historical pre-re-run figures (`920 | 10`; the cache-present `978`
was never re-verified). Pre-sweep totals (178, 94, `921 | 7` / `956 | 4`) are
in git history; arc 04's last pre-sweep run is committed as
[`data/audit_2026-08-16.log`](research/arcs/04_jspace/data/audit_2026-08-16.log).

Capture and analysis scripts deserialize with `torch.load(..., weights_only=False)`
on purpose — the artifacts are produced by these same scripts and never sourced
externally. See the trust-boundary note in `examples/README_NLA.md` before
extending the pipeline to third-party `.pt` files.

**Raw data is a deliverable.** A clean clone (with LFS pulled) holds the figure
PNGs and the `.pt` datasets the figures and audit depend on, so every figure can
be re-rendered and the audit replayed. See `research/ARC_PROCESS.md`
§ "Raw data is a deliverable".

## Epistemic discipline (carried over from the source workspace)

- Every load-bearing technical claim cites a primary source — a paper-key in
  `theory/kb/index/papers.json` or an anchor into a KB note/excerpt.
- Analogies and intuitions are tagged (`[ANALOGY]`, `[INTUITION]`,
  `[SPECULATION]`, `[CONTRADICTION]`), never asserted as fact.
- Forum/blog citations are discovery signals only; they never solely back a
  hard claim. Full rules in `theory/README.md` and `CLAUDE.md`.

## License

GPL-3.0-only. (c) Michael Lannum. See `LICENSE`.

That licence covers this repository's **code and original prose only**.
Third-party data and model-derived artifacts are explicitly scoped out and
carry their own terms, recorded in a `LICENSE-DATA.md` beside the data in
every arc (01-04); arc 04's is the reference pattern, naming the source, the
licence, and the attribution it obliges. Primary-source PDFs committed under
`theory/sources/papers/` likewise retain their own licences and are
redistributed under them, not under this repo's.

