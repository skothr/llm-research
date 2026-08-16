# llm-research

LLM-interpretability research, organized as a citation-grounded theory
knowledge base plus reproducible experimental arcs. Built on the
[`llm-surgeon`](https://github.com/skothr/llm-surgeon) toolkit for layer-level model surgery and
probing. The largest investigation applies Anthropic's released NLA
(Natural Language Autoencoder) verbalizer/reconstructor model pair to
local Qwen2.5-7B-Instruct and probes layer-20 hidden-state geometry; three
further arcs cover subliminal trait transfer, the structure of Qwen2.5-7B's
input-embedding table, and a partial replication of the J-lens / J-space
global-workspace result.

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

**Data correction — closed 2026-08-16.** Arc 04's seeded **C4-en** corpus
slice carried 120 pieces of third-party personal data and was redacted on
2026-07-29; because the redaction is not length-preserving, every result
computed on that corpus — the **corpus-invariance and held-out-sample**
checks — was recomputed on the redacted text (2026-08-15/16). Outcome: two
audit pins moved at the third decimal, both in the held-out channel, and no
headline conclusion changed; the two failure modes pre-registered before the
re-run did not materialise. Arc 04's primary fitting corpus is wikitext-103,
which was scanned and left unmodified, and no other arc used C4. The full
record — per-class counts, root cause, per-claim blast radius, and the
reproduction recipe — is in
[`research/arcs/04_jspace/README.md`](research/arcs/04_jspace/README.md) and
[`research/arcs/04_jspace/data/README.md`](research/arcs/04_jspace/data/README.md).
Surfaced here because a correction of this kind should be visible at the
entry point rather than discovered two levels down.

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
  (arc 01), `emb_*` (arc 03), `jspace_*` (arc 04). Each family covers capture
  (writes `.pt` artifacts), analysis, figure render (matplotlib), and an
  `*_audit_findings.py` that re-derives that arc's load-bearing numerical
  claims from committed artifacts. `examples/README_NLA.md` documents the
  `nla_*` pipeline specifically — the `emb_*` and `jspace_*` families follow
  the same artifact/audit shape but have no separate conventions doc; their
  arc READMEs carry the per-arc detail. Some scripts import `llm_surgeon`
  (model loading, probing, surgery); the rest are render/analysis-only
  (torch / numpy / matplotlib).

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
  01, 03, and 04 each have one. Arc 02 has none: its Step-0 numbers (a z-test,
  a power calculation) predate the audit-script convention and are small enough
  to check by hand against the committed JSONL. These audits check
  **arithmetic consistency only** — they cannot catch a methodological error, a
  capture-protocol bug, or interpretive overreach. Arcs 01 and 03 state that
  limitation in their READMEs; arc 04 states it in its audit block.
- **Datasets committed and pinned.** Arcs 01, 03, and 04 commit raw `.pt`
  artifacts under the arc's `data/` (Git LFS) with a `MANIFEST.json` recording
  per-file sha256 and provenance; arc 02's Step-0 data is JSONL under an
  interim manifest schema. The intent is that a clean clone re-renders every
  figure and replays every audit — with one documented exception: arc 04's
  fitted-lens tensors are cache-only by design, so 11 of its checks report
  `MISSING` on a clean clone (see [Running the research pipeline](#running-the-research-pipeline)).
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

Research figures (`research/**/figures/*.png`) and raw datasets
(`research/**/data/*.pt`) are stored via Git LFS — see `.gitattributes`. Install
and initialize Git LFS **before** cloning or working the repo, or those files
show up as phantom modifications (the working tree holds pointer files, not
content):

```bash
git lfs install            # one-time, per machine
git clone <repo-url>       # LFS content then fetches on checkout
# already cloned without LFS? recover with:
git lfs install && git lfs pull
```

## Setup

Python >= 3.10. The analysis/figure scripts depend on the sibling
`llm-surgeon` toolkit, installed editable from a checkout next to this repo:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llm-surgeon      # provides llm_surgeon (+ torch, transformers, ...)
pip install -e .                   # this repo's deps: torch, numpy, matplotlib
```

`pip install -e ../llm-surgeon` pulls in `torch`, `transformers`, `accelerate`,
`bitsandbytes`, and the rest of the surgery/probe runtime. This repo declares
`llm-surgeon` plus the direct imports of its own example scripts
(`numpy`, `matplotlib`); installing the editable sibling first satisfies the
`llm-surgeon` requirement.

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
script runs two full sweeps so cross-refs settle. A LaTeX toolchain
(`pdflatex`, `bibtex`) must be on `PATH`.

## Running the research pipeline

Capture scripts write `.pt` artifacts (working cache under `.cache/`,
gitignored; the committed copies live in each arc's `data/`). Render scripts
turn artifacts into figures; each arc's audit re-derives that arc's claims
from them. Verified from a clean clone on 2026-07-28:

```bash
python examples/nla_audit_findings.py      # arc 01 → SUMMARY: 178 PASS | 0 FAIL
python examples/emb_audit_findings.py      # arc 03 → SUMMARY:  94 PASS | 0 FAIL
python examples/jspace_audit_findings.py   # arc 04 → SUMMARY: 920 PASS | 10 FAIL
```

Arc 04's 10 failures on a clean clone are **expected**, not regressions: its
five full fitted-lens tensors and their five sidecars are cache-only by design,
and the audit reports each as a loud `MISSING` rather than skipping it. (This
read `11 FAIL` before 2026-07-29, when one sidecar was checked twice inside
CHECK J; fixed.) With the local cache present the same run reports 978 PASS —
a figure last measured 2026-07-25 and due to be re-derived when the
C4-redaction re-run repopulates the cache. Arc 02 has no audit script; its Step-0
numbers are checkable by hand against the committed JSONL. See the arc READMEs
for what each audit does and does not catch (arithmetic consistency only —
never methodology or interpretation).

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

