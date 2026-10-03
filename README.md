# llm-research

LLM-interpretability research, organized as reproducible experimental arcs,
with a secondary, AI-generated theory knowledge base alongside (see
[`theory/`](#whats-here) below). The code depends on
standard scientific libraries (PyTorch, HuggingFace `transformers`, numpy,
matplotlib) plus one reference implementation, `jlens`, at a recorded commit.
The largest investigation applies Anthropic's released NLA
(Natural Language Autoencoder) verbalizer/reconstructor model pair to
local Qwen2.5-7B-Instruct and probes layer-20 hidden-state geometry; three
further arcs cover subliminal trait transfer, the structure of Qwen2.5-7B's
input-embedding table, and a partial replication of the J-lens / J-space
global-workspace result.

This repository is a **research workspace**, not a software product. It
collects the work product — the dated observations, the figure/audit
pipeline, a background theory knowledge base — rather than a polished library. Claims are held
to the standard described under "Epistemic discipline" below: technical claims
that a conclusion rests on cite a primary source, and findings are framed as
hypotheses until the evidence settles them.

It is also **exploratory, self-directed, agent-assisted** work. Much of the
implementation — capture and analysis scripts, figures, audit infrastructure,
observation drafts — was carried out by Claude Code sessions under direction;
the research questions, scope calls, and sign-off are the author's, and each
arc README records that split explicitly. Nothing here has been published,
externally reviewed, or replicated by anyone else, and the findings should be
read as provisional. The [Methodology](#methodology) below is the part offered
with confidence: it exists to make being wrong visible.

## Results

One block per arc: the headline finding, one figure where the arc has one
(arc 02 has none), the strongest caveat and a link. The figures are stored in
Git LFS, so a clone made without LFS shows them as broken images;
[§ Prerequisites](#prerequisites--git-lfs-is-required) has the setup and the
recovery command. Every number here is taken from the linked arc README, which carries
the evidence, the scope qualifications and the full list of limitations; where
this summary is shorter, the arc README is the authority.

### Arc 04 — J-lens / J-space replication on Qwen2.5-1.5B/7B: partial

Closed 2026-07-22. The J-space phenomenon reported for Claude-family models
`[gurnee2026-workspace]` partially replicates on Qwen2.5-Instruct (1.5B bf16,
7B nf4). The excess-over-random FVE figures below are from recomputes after
that date: the paper-metric recompute (2026-07-24/25), the C4 PII-redaction
re-run (closed 2026-08-16) and the dimension-matched recompute (2026-09-23).

- **Replicates: the J-lens as a readout instrument.** It surfaces unspoken
  intermediate concepts where the logit lens finds nothing (multihop early/mid
  bands). Counterweight: the logit-lens median emergence layer is still
  earlier at 1.5B (19 vs 23), and the J-lens fails to surface the token in
  more cells at both scales (1.5B 16/108 vs logit 1/108; 7B 14/108 vs 8/108).
- **Occupancy against the paper's 10% ceiling.** On the paper's
  excess-over-random FVE metric, 1.5B breaches the ceiling at the hump
  (**L21 excess 11.15%**, CI95 [10.95, 11.40]) and 7B stays under it at every
  K tested. At the paper's K rule (K=23 held-out, K=23-24 grid) the 7B peak
  excess is 5.88% held-out and 4.72% on the scan grid; at K=58 it is 7.67%
  held-out and 6.16% on the grid, with every CI95 upper bound at or below
  8.03%. Excess depends on K/d; at matched K/d the 1.5B/7B ratio is 1.52×
  held-out and 1.76× on the grid.
- **Strongest positive: a graded relational effect (stage 5.2).** Swapping an
  unspoken concept along its J-lens vector moves the concept's entailed
  property more than an equal-magnitude logit-lens token-steering control.
  The absolute gap is **+5.0 nats** (1.5B L18, n=7) and **+1.9 nats** (7B
  L19, n=17) on the auto-detected subset. The per-item SD is about equal to
  the gap at 1.5B (4.9) and larger than it at 7B (4.1): n is small and a few
  high-movement items dominate. The paired gaps pass an exact sign-flip test
  (p=0.0156, p=0.0001).
- **Does not replicate: four items.** The discrete entailed-property flip
  (rate 0.000 at both scales). J-space membership as the causally privileged
  ingredient: the J-space component of concept vectors shows **no detectable
  effect** in report swaps, bounded at ≤3.8pp at 7B (exact 95%, 0 discordant
  pairs of 78) and looser at 1.5B (p=0.375 on 5 discordants). The kurtosis
  workspace-onset signature (inverted on Qwen). The small-model structural
  picture at 7B (7B has ~3× lower occupancy and a U-shaped depth profile, on
  absolute varfrac@25 over 30 held-out wikitext prompts; that is a different
  measure from the excess-over-random FVE in the occupancy item above).

![Arc 04: excess-over-random FVE by layer at matched K/d](research/arcs/04_jspace/observations/figures/2026-09-23-jspace-paper-metric-matched-kd.png)

*Excess-over-random FVE by layer at matched K/d, with cluster-bootstrap CI95
bands and the paper's 10% ceiling dashed, on (a) held-out C4 prompts and (b)
the wikitext scan grid. 1.5B is plotted at K=25 in both panels. 7B is
plotted in both panels at the paper's K rule and at K=58, which matches
1.5B's K/d, and in panel (b) also at K=25. 7B stays under the ceiling at
every layer and K.*

**Strongest caveat.** One model family at ≤7B, unreviewed and unreplicated
outside this repo; and the negative results are measured bounds, not
demonstrated zeros. Full account:
[`research/arcs/04_jspace/README.md`](research/arcs/04_jspace/README.md).

### Arc 03 — embedding atlas of Qwen2.5-7B's input-embedding table

Closed for new experiments 2026-07-15. Findings are held as working
hypotheses.

- **Pre-registered predictions, adjudicated: P1a PASS, P1c FAIL, P1d FAIL**;
  P2 refined-not-falsified; P1b, P1e and P3 not run. The two failures:
  period→comma aggregation peaks at L0, not deeper (P1c), and delimiter
  matching is carried ~99% by near-DC RoPE bands, a static content match
  rather than positional resonance (P1d).
- **One 21-dim structural block.** Over all 149,706 alive rows there is
  exactly one entangled dimension block: 21 correlated dims (|r|>0.3),
  cross-script, head-loaded by frequency (the first token-id decile carries
  1.5x the block norm-fraction floor; Spearman -0.206 vs -0.003 control).
  Outside it, dimensions are near-independent (|r| mean 0.021).
- **Near-isotropy null.** Random-pair cosine +0.0097; PC1 explains 1.21%.

![Arc 03: block norm-fraction by token-id decile](research/arcs/03_embedding-atlas/observations/figures/fig15_structural_block.png)

*Block norm-fraction by token-id decile against a random-21-dim control: the
first (most frequent) decile carries 0.1143 vs the control's 0.0753.*

**Strongest caveat.** Single model, single revision: the isotropy null
especially needs a second model before any general reading. The audit's 99
PASS checks arithmetic consistency only. Full account:
[`research/arcs/03_embedding-atlas/README.md`](research/arcs/03_embedding-atlas/README.md).

### Arc 01 — NLA verbalizer on Qwen2.5-7B layer 20: basin candidates

Paused 2026-05-15. One working synthesis, held as a working hypothesis and not
a settled claim: layer-20 h-space appears to have discrete attractor basins
separated by sharp boundaries.

- Linear interpolation between two AR-encoded natural-language anchors
  produces a discontinuous AV-text transition at a roughly constant geometric
  step. Dense re-sampling found a hybrid "Definition + Poem" plateau
  (t∈[0.395, 0.4450]) and a flip to the poetic format in a single Δt=0.0025
  step, at t=0.4475.
- Scope: one anchor pair, one layer, one model. The plateau-attractor margin
  is +0.061 over the nearest single anchor, against +0.25 between anchors, so
  the arc's framing is "basin candidate / shallow basin" until more anchor
  pairs and layers replicate.

![Arc 01: dense interpolation between two AR-encoded anchors](research/arcs/01_nla-verbalizer/observations/figures/fig36_dense_interp_flipbook.png)

*Dense interpolation between the two AR-encoded anchors: factual → hybrid
plateau → poetic/nature (t≥0.4475).*

**Strongest caveat.** The AV-decoder format bias is unaudited
([#8](https://github.com/skothr/llm-research/issues/8)): if the verbalizer
emits the same templates on random h-vectors as on semantically loaded ones,
every interpretive reading in the arc has been filtered through the
verbalizer's prior. Every result is also on one model at one layer. Full
account:
[`research/arcs/01_nla-verbalizer/README.md`](research/arcs/01_nla-verbalizer/README.md).

### Arc 02 — subliminal trait transfer: a Step-0 null, paused

Paused 2026-06-10 after Step 0; Steps 1-2 were not started. The arc has no
figures.

- **Step-0 null.** A five-scheme decode of locally regenerated owl/neutral
  number streams returns zero owl-lexicon hits in either condition (owl_rate =
  neutral_rate = 0.000, z = 0, p = 1.0), with the planted-string positive
  control passing. A literal decodable channel is not supported *for this
  setup*.

**Strongest caveat.** The arc's question, non-semantic statistics (A) versus
semantics in the model's own coordinates (C), is untested: the two steps
designed to address it were never run. Step 0 also tests a local stand-in
(streams regenerated with an open teacher), not the paper's own data. Full
account:
[`research/arcs/02_subliminal/README.md`](research/arcs/02_subliminal/README.md).

## Data correction notice — arc 04's C4-en corpus

> [!NOTE]
> **Data correction — closed 2026-08-16.** Arc 04's seeded **C4-en** corpus
> slice carried 120 pieces of third-party personal data and was redacted on
> 2026-07-29; because the redaction is not length-preserving, every result
> computed on that corpus — the **corpus-invariance** and **held-out-sample**
> checks and the **H3 @10 corpus-dependence qualifier** — was recomputed on
> the redacted text (2026-08-15/16). Outcome: two audit pins moved, each by
> less than 0.02, both in the held-out channel, and no headline conclusion
> changed; the two failure modes pre-registered before the re-run did
> not materialise. Arc 04's primary fitting corpus is wikitext-103,
> which was scanned and left unmodified, and no other arc used C4. The full
> record — per-class counts, root cause and the reproduction recipe — is in
> [`research/arcs/04_jspace/data/README.md`](research/arcs/04_jspace/data/README.md),
> which links the per-claim record.
> Surfaced here because a correction of this kind should be visible at the
> entry point rather than discovered two levels down.

## What's here

```
research/    Experimental research, organized into arcs (focused investigations);
             each arc's capture / analysis / render / audit scripts are in its scripts/
src/         The llm_research package: shared model loader + NLA probe
tests/       Tests of src/, scripts/ and the moved arc scripts
theory/      Citation substrate for the arcs (paper index, excerpts, notes); otherwise
             secondary: AI-generated LLM-theory knowledge base + 5-paper LaTeX series
```

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
- **`research/arcs/<slug>/scripts/`** — Each arc's pipeline scripts, one family
  per arc by prefix (`nla_*`, `subliminal_*`, `emb_*`, `jspace_*`), beside the
  `_`-prefixed helper modules the arc 01, 03 and 04 families import. Each family covers capture
  (writes `.pt` artifacts; arc 02's writes JSON/JSONL and text), analysis, figure render (matplotlib), and an
  `*_audit_findings.py` that re-derives the numerical claims that arc's
  prose relies on from committed artifacts. Arc 04's `jspace_*` family and
  arc 03's `emb_*` family follow the `nla_*`
  pipeline's artifact/audit shape; the cross-arc
  conventions are in `research/ARC_PROCESS.md` § "Raw data is a deliverable",
  and each arc README carries the per-arc detail. The arc 01, 03 and 04 capture scripts
  load models through the in-repo `llm_research` package (`src/llm_research/`,
  installed by `pip install -e '.[dev]'`): `hf_models` (HuggingFace loader,
  nf4 / int8 / bf16 / fp16 / fp32 modes) and `nla_probe` (the NLA
  verbalizer / reconstructor pair); the rest are render/analysis-only
  (torch / numpy / matplotlib). Arc 01's `nla_*` scripts are in
  `research/arcs/01_nla-verbalizer/scripts/`. Arc 04's `jspace_*` scripts are in
  `research/arcs/04_jspace/scripts/`. Arc 03's `emb_*` scripts are in
  `research/arcs/03_embedding-atlas/scripts/`. Arc 02's `subliminal_*`
  scripts are in `research/arcs/02_subliminal/scripts/`; its step-0 generator,
  `subliminal_step0_decode.py`, calls `transformers` directly.
- **`theory/`** — **The citation substrate for the arcs, and otherwise a side
  project.** Arc prose cites primary sources by their paper-key in
  `kb/index/papers.json`, and the arc READMEs link into the knowledge base in
  a few places, for paper metadata and background notes. Beyond that role
  most readers can skip it. The
  knowledge-base notes and the LaTeX series were written largely by Claude
  Code subagents (one topic area per agent for the notes, one section per
  agent for the series), and they are not part of the research findings.
  Contents (v2 layout): `kb/notes/`
  digested synthesis (one file per topic), `kb/excerpts/` verbatim source
  passages, `kb/index/` (`papers.json`, `topics.md`, `timeline.md`),
  `kb/glossary.md`, and `sources/papers/` primary-source PDFs. `series/`
  holds a 5-paper LaTeX series (architecture, training, reasoning,
  interpretability, evaluation-alignment) with cross-paper references. Start
  at `theory/README.md`.

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
  Arc 04's README records one of its four robustness axes (the quantization
  control) as pre-registered in its design plan, the rest gated on thresholds
  fixed before each run. Arc 01 grew from open-ended themes with no
  pre-registration. Three of the four arcs' registers are partial — read each arc's
  own account rather than this summary.
- **Audit scripts.** The `*_audit_findings.py` scripts (each in its arc's `scripts/` directory)
  re-derive the numbers an
  arc's claims rest on from its committed artifacts, so a figure quoted in
  prose that has drifted from the artifact it came from fails the audit. Arcs
  01, 02, 03, and 04 each have one. These audits check
  **arithmetic consistency only** — they cannot catch a methodological error, a
  capture-protocol bug, or interpretive overreach. Arcs 01 and 03 state that
  limitation in their READMEs, arc 02 in its data README § Audit, and arc 04
  in its § Reproducing.
- **Datasets committed and pinned.** Arcs 01, 03, and 04 commit raw `.pt`
  artifacts under the arc's `data/` (Git LFS) with a `MANIFEST.json` recording
  per-file sha256 and provenance; arc 02's Step-0 data is small JSON/JSONL and text in plain
  git under the same `MANIFEST.json` convention. The intent is that a clean
  clone re-renders every figure and replays every audit — with one documented exception: arc 04's
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

Python >= 3.10. `pyproject.toml` declares every PyPI dependency (torch,
transformers, accelerate, bitsandbytes, huggingface_hub, safetensors, pyyaml,
numpy, matplotlib, datasets). The `jspace_*` scripts also need `jlens`
(anthropics/jacobian-lens), which is not on PyPI and is installed editable
from a checkout next to this repo; the arc-04 `data/MANIFEST.json` records
the commit (`jlens_pin`) to check out:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../jacobian-lens    # jlens — every arc 04 jspace_*.py script
pip install -e '.[dev]'            # this repo's deps, its llm_research package + pytest
                                   # (re-run it in a checkout installed before src/llm_research existed)
```

Model checkpoints download from the HuggingFace Hub into the directory named
by `LLM_RESEARCH_MODEL_CACHE`, or into the HuggingFace default cache when it
is unset. Checkpoints downloaded before 2026-09-26 live in the directory the
former llm-surgeon toolkit cached to (`.cache/models/` inside the llm-surgeon
checkout by default, or `LLM_SURGEON_CACHE_DIR` when set), so export
`LLM_RESEARCH_MODEL_CACHE` to that directory before a capture run. On a cache
miss the loader downloads the multi-GB checkpoints, except in the seven arc-01
scripts that force `HF_HUB_OFFLINE=1` (`grep -l HF_HUB_OFFLINE research/arcs/01_nla-verbalizer/scripts/nla_*.py`),
which raise an `OSError` at model load naming the cache directory probed and
`LLM_RESEARCH_MODEL_CACHE`. The NLA verbalizer (AV) and reconstructor (AR)
load at pinned Hub revisions (`llm_research.nla_probe.AV_REVISION`,
`AR_REVISION`), so an offline run needs a cache that holds those exact
snapshots; with `LLM_RESEARCH_MODEL_CACHE` exported,
`hf_models._is_cached(nla_probe.AV_ID, revision=nla_probe.AV_REVISION)` (both
imported from `llm_research`) and the AR equivalent check it. The seven
scripts set the offline flag unconditionally, so recover a cache holding
another snapshot of the same repos with
`huggingface_hub.snapshot_download(<AV_ID or AR_ID>, revision=<AV_REVISION or AR_REVISION>, cache_dir=<the cache dir>)`,
run online: the weights are unchanged since 2026-03-16, so only the small
files at the pinned revision download and the cached weight files are reused.

## Running the research pipeline

Capture scripts write `.pt` artifacts (working cache under `.cache/`,
gitignored; the committed copies live in each arc's `data/`). Render scripts
turn artifacts into figures; each arc's audit re-derives that arc's claims
from them. Arcs 01 and 03 were re-verified from a clean clone on 2026-08-29
(their committed logs reproduce byte-for-byte). Arc 02's audit gained seven
arc-manifest claims on 2026-09-28 and now totals 111 PASS; its committed
2026-08-17 log records the earlier 104. Arc 04 was re-measured
in-session on 2026-09-24 after its audit gained CHECK P. No audit log was
committed for that run; re-derive the totals with the command below:

```bash
python research/arcs/01_nla-verbalizer/scripts/nla_audit_findings.py         # arc 01 → SUMMARY: 196 PASS | 0 FAIL
python research/arcs/02_subliminal/scripts/subliminal_audit_findings.py  # arc 02 → SUMMARY: 111 PASS | 0 FAIL | 5 UNVERIFIABLE
python research/arcs/03_embedding-atlas/scripts/emb_audit_findings.py  # arc 03 → SUMMARY:  99 PASS | 0 FAIL
python research/arcs/04_jspace/scripts/jspace_audit_findings.py      # arc 04 → SUMMARY: 1053 PASS | 7 FAIL on a default clone;
                                                                     #          1088 PASS | 4 FAIL with the lens cache pulled
```

Arc 04's 7 failures on a clean clone are **expected**, not regressions: the
three lenses refit in the C4-redaction re-run are LFS-committed but excluded
from default LFS pulls (`.lfsconfig` — the ~905 MiB lens download is
opt-in), so the audit reports each as an `LFS pointer stub` naming the pull
command, and the two 1.5B nf4 lenses and their sidecars are regenerate-only
pending the scheduled refit (issue #47), each reported as a loud `MISSING`
rather than skipped — 3 stubs + 4 `MISSING` = 7. After
`git lfs pull --include="research/arcs/04_jspace/data/cache/**" --exclude=""`
the same run reports **1088 PASS | 4 FAIL** (the nf4 `MISSING` reports
only; measured 2026-09-24 with the cache pulled: 986 measured 2026-08-17,
plus the 30 cache-independent CHECK O claims added 2026-08-30 and the 72
CHECK P claims added 2026-09-23, which read LFS artifacts pulled by default
and plain committed logs, not the lens cache) with no GPU work; the check
total grows from 1060 to 1092 between the two states because the
lens-dependent blocks register their claims only when the lens tensors are
on disk.
Arc 02's 5 UNVERIFIABLE entries are printed, not scored — external
citations and capture-time environment facts no committed artifact can
settle. See the arc READMEs
for what each audit does and does not catch (arithmetic consistency only —
never methodology or interpretation) and for the historical pre-re-run
figures (`920 | 10`; the cache-present `978` was never re-verified).

Capture and analysis scripts deserialize with `torch.load(..., weights_only=False)`
on purpose — the artifacts are produced by these same scripts and never sourced
externally. See the trust note in `research/ARC_PROCESS.md` § "Raw data is a
deliverable" before extending the pipeline to third-party `.pt` files.

**Raw data is a deliverable.** A clean clone (with LFS pulled) holds the figure
PNGs, the `.pt` datasets and the plain-git JSON/JSONL and text datasets the figures and
audits depend on, so every figure can be re-rendered and every audit replayed. See `research/ARC_PROCESS.md`
§ "Raw data is a deliverable".

## Building the theory LaTeX series (optional)

The theory side project's output is the 5-paper series under `theory/series/`,
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

## Epistemic discipline (carried over from the source workspace)

- Every technical claim a conclusion rests on cites a primary source by its
  paper-key in `theory/kb/index/papers.json`, often with an anchor into a KB
  excerpt or note. The excerpts are verbatim; the notes are AI-written
  digests that point to sources, and the cited paper is canonical.
- Analogies and intuitions are tagged (`[ANALOGY]`, `[INTUITION]`,
  `[SPECULATION]`, `[CONTRADICTION]`), never asserted as fact.
- Forum/blog citations are discovery signals only; they never solely back a
  hard claim. Full rules in `theory/README.md` and `CLAUDE.md`.

## License

GPL-3.0-only. (c) Michael Lannum. See `LICENSE`.

