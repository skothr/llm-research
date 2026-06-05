# llm-research

LLM-interpretability research, organized as a citation-grounded theory
knowledge base plus reproducible experimental arcs. Built on the
[`llm-surgeon`](https://github.com/skothr/llm-surgeon) toolkit for layer-level model surgery and
probing. The flagship investigation applies Anthropic's released NLA
(Natural Language Autoencoder) verbalizer/reconstructor model pair to
local Qwen2.5-7B-Instruct and probes layer-20 hidden-state geometry.

This repository is a **research workspace**, not a software product. It
collects the work product — the synthesized theory, the dated observations,
the figure/audit pipeline — rather than a polished library. Claims are held
to the standard described under "Epistemic discipline" below: load-bearing
technical claims cite a primary source, and findings are framed as
hypotheses until the evidence settles them.

## What's here

```
theory/      Citation-grounded LLM-theory knowledge base + a 5-paper LaTeX series
research/    Experimental research, organized into arcs (focused investigations)
examples/    The NLA capture / analysis / render / audit pipeline (46 scripts)
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
  retired material. The flagship arc is `research/arcs/nla-verbalizer/`
  (22 dated observations, 36 figures, a 178-PASS regression audit). The arc
  lifecycle and reproducibility disciplines are in `research/ARC_PROCESS.md`;
  the index is `research/README.md`.
- **`examples/`** — The `nla_*.py` scripts implementing the NLA pipeline:
  capture (writes `.pt` artifacts), analysis, figure render (matplotlib), and
  `nla_audit_findings.py`, which re-derives every load-bearing numerical claim
  from raw artifacts. Conventions are in `examples/README_NLA.md`. Of the 46
  scripts, 18 import `llm_surgeon`; the rest are render/analysis-only
  (torch / numpy / matplotlib).

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

NLA scripts read and write `.pt` artifacts under `.cache/nla_artifacts/`
(gitignored). The capture scripts produce artifacts; render scripts turn them
into figures; the audit re-derives claims:

```bash
python examples/nla_audit_findings.py   # re-derive load-bearing numbers from .pt artifacts
```

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

