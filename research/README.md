# Research

Experimental LLM-interpretability research on local open-source models —
**Qwen2.5-7B-Instruct** and **Qwen2.5-1.5B-Instruct** carry the current arcs,
with **TinyLlama** and **OpenLLaMA 3B v2** in the early and archived work;
inference via HF `transformers`. Work is organized into **arcs** — focused,
multi-observation investigations that each cohere around a single research
question — plus a landing bin for one-off findings and an archive for retired
material.

## Arcs

Two of the four arcs are **paused mid-program**; the other two were closed to
new experiments — arc 04 having run its planned stages, arc 03 by direction
with three of its seven registered predictions still unrun. No arc has been
externally reviewed or replicated. "Closed" here means *no further experiments
planned* — a scheduling statement, not a verdict on the results.

| Arc | Status | Question | Entry point |
|---|---|---|---|
| **NLA verbalizer — Qwen2.5-7B layer 20** | **Paused** 2026-05-15; synthesis written | What do Anthropic's released NLA verbalizer/reconstructor models surface about layer-20 hidden-state geometry? | [`arcs/01_nla-verbalizer/README.md`](arcs/01_nla-verbalizer/README.md) |
| **Subliminal trait transfer** | **Paused** 2026-06-10 at Step 0 | Is the hidden trait signal in subliminal learning non-semantic statistics (HA), or semantic in the model's *own* representational coordinates (HC)? | [`arcs/02_subliminal/README.md`](arcs/02_subliminal/README.md) |
| **Embedding atlas — Qwen2.5-7B structural tracing** | **Closed** to new experiments 2026-07-15 | What structural/procedural machinery does the model build on its input-embedding table — and how do the structural-token features it contains get used by Q/K/V, attention (RoPE bands), and FFN, layer by layer? | [`arcs/03_embedding-atlas/README.md`](arcs/03_embedding-atlas/README.md) |
| **J-space replication — Qwen2.5-1.5B/7B** | **Closed** 2026-07-22; ran to planned completion, results unreviewed. **C4 data correction closed 2026-08-16** — corpus-robustness results recomputed on the redacted corpus (see below) | Does the J-lens/J-space global-workspace phenomenon `[gurnee2026-workspace]` replicate on Qwen2.5-Instruct (1.5B bf16 primary, 7B nf4 scale check), and how do J-lens readouts relate to the NLA verbalizer at layer 20? | [`arcs/04_jspace/README.md`](arcs/04_jspace/README.md) |

**What each arc actually produced** — negative and null results included,
because they carry as much of the signal as the positive ones:

- **Arc 01 (NLA verbalizer)** — the deepest arc by volume, and the source of
  the working synthesis that layer-20 h-space has discrete attractor basins
  separated by sharp boundaries. Held as a hypothesis, not a settled claim;
  the arc's own limitations section flags an unaudited AV-decoder format bias
  that, if real, would re-frame every interpretive claim built on it.
- **Arc 02 (subliminal)** — one null, narrow by construction: a literal
  ASCII/base-N encoding channel was **not detected** in a local Qwen
  reproduction of the paper's protocol — **zero** owl-lexicon hits across all
  five decode schemes in either condition (owl n=104, neutral n=109 after the
  format/range/count filter) — no variance in either arm, so no z or p is
  defined and the informative quantities are the zero-hit count and the
  reject-rate power floor (~931/condition for the incidental 13.3-vs-9.2%
  gap, audit-locked), with the planted-encoding positive control passing in
  the same run, so the null is an absence rather than a broken decoder. The
  paper released no dataset — its `v1.0.0` has zero assets and its teacher is
  closed — so this tests a local reproduction, not their data. The decisive
  HA-vs-HC test was never run.
- **Arc 03 (embedding atlas)** — predictions were registered on 2026-06-11,
  before the captures they constrain, and adjudicated mechanically at close:
  **P1a PASS, P1c FAIL, P1d FAIL**; P2 refined-not-falsified; P1b, P1e and P3
  never run. Two falsified predictions, recorded in advance and reported as
  findings, are the clearest evidence in this repo that the pre-registration is
  not decorative — the three unrun ones are the honest limit on that claim.
  This arc also carries a **model-provenance audit** — a blinded
  re-examination of whether the agent model used during the arc's kickoff
  window left degraded judgment in its findings, negative result (no credible
  degradation artifact; the numeric layer was inert to it because every
  load-bearing number re-derives from committed artifacts):
  [`arcs/03_embedding-atlas/sessions/2026-07-21-degradation-forensics.md`](arcs/03_embedding-atlas/sessions/2026-07-21-degradation-forensics.md).
- **Arc 04 (J-space)** — a **partial** replication that came out **weaker than
  the original**, with a causal split: J-lens steering moves entailed
  properties, while J-space *membership* swaps produce **no detectable
  effect** — bounded at ±3.8pp (7B, 0 discordant pairs of 78), which is a tight
  null but a measured bound, not a demonstration of zero. Four specific things
  did not replicate, including the paper's discrete property flip (rate 0.000
  at both scales). Prefer the absolute gaps in nats over the multiplier
  framings — the control sits near zero, so the ratios are denominator-fragile.
  **Data correction (2026-07-29, closed 2026-08-16):** this arc's C4-en corpus
  slice carried third-party personal data and was redacted; the redaction is
  not length-preserving, so every C4-computed result was regenerated on the
  redacted text. Two audit pins moved, each by less than 0.02 — the held-out
  logit-kurtosis trough (1.000 → 1.019) and the 7B held-out L23 excess
  (0.0598 → 0.0588, confounded with refit nondeterminism) — while the
  **corpus-invariance** and **held-out** conclusions otherwise re-derive
  within audit tolerance and the quoted **L21 excess 10.7–11.7%** range
  survives as written. Per-claim record:
  [`arcs/04_jspace/README.md`](arcs/04_jspace/README.md).

**Where to start.** Arc 01 for depth and the largest body of observations;
arc 03 or 04 for how the method behaves when a prediction fails or a
replication comes out weak.

## Layout

```
research/
  README.md              ← this file (abstract index)
  arcs/<slug>/           ← one directory per investigation
    README.md            ← arc entry point: motivation, findings, caveats
    observations/        ← dated evidence-first writeups
      figures/           ← generated plots + INVENTORY.md provenance
    data/                ← raw .pt datasets (git-LFS) + MANIFEST.json
    sessions/            ← session-resumption checkpoints (stale-fast)
    plans/               ← research / construction plans (as needed)
  observations/          ← one-off findings not (yet) part of an arc
  archive/               ← retired / pre-arc material, kept for archaeology
```

A finding starts life in `observations/` (or, if it's already arc-scoped, in
the arc's `observations/`). When several loose observations cohere into an
investigation, promote them: `mkdir arcs/<slug>/`, move the files in, and
write the arc `README.md` that ties them together.

**Running an arc?** [`ARC_PROCESS.md`](ARC_PROCESS.md) is the standard
operating procedure — the lifecycle (question → capture → validate+save data →
analyze → figures → observations → audit → synthesis → PR) and the disciplines
that keep an arc reproducible and honestly framed.

## Conventions

**Observations** — dated markdown, `YYYY-MM-DD-<slug>.md`, one finding per
file. Each carries: date + context (model, params), the finding, evidence
(output/transcript excerpts), reproducibility (exact commands), hypotheses,
follow-ups, references. The field list lives in the repo `CLAUDE.md` under
*# Research arcs & observations*; the copy-paste skeleton with the canonical
heading spellings is in [`ARC_PROCESS.md`](ARC_PROCESS.md) § 4. No index file
— scan by filename. Evidence-first: numbers should be reproducible or
audit-locked.

**Sessions** — LLM session checkpoints, *not* research findings. They capture
the operational state of a Claude Code session at a compaction or hand-off
boundary (worktree path, branch tip, audit-pass count, "what to do next"
pointers). Same `YYYY-MM-DD-<slug>.md` naming, with the slug carrying one of
`arc-summary` / `arc-resume` / `checkpoint` / `for-compact`. These go **stale
within hours or days** — read them as a snapshot at write-time, not as
current guidance; newer files supersede older ones. Nothing in `sessions/` is
load-bearing for a research claim, so a stale or deleted session file never
affects the correctness of an observation or figure.

**Attribution** — each arc README carries an `## Attribution` section
separating human direction-setting from AI implementation: named direction
blocks quoted from the session transcripts, each with a provenance label
(`[VERBATIM]` / `[NORMALIZED]` / `[PARAPHRASE]` / `[SELECTED]` /
`[RECONSTRUCTED]`), a human/Claude/emergent split, and a verifiability table
listing the sessions by id and date (transcripts are machine-local and not
committed). It is recorded **as the arc runs**, not reconstructed at close —
transcript evidence expires. Full rules and the template:
[`ARC_PROCESS.md`](ARC_PROCESS.md#attribution--who-directed-who-executed).

**Plans** — research/construction plans (what to investigate, in what order,
with what caveats). Dated `YYYY-MM-DD-<slug>.md`. A plan in `Status:
preliminary` has run no experiments yet.

**Figures** — generated plots under an arc's `observations/figures/`, with an
`INVENTORY.md` giving per-figure provenance (which script + commit produced
each one). PNGs are tracked via **git-LFS** — run `git lfs install` before
working the repo, or figures show as phantom modifications. Recovery for an
LFS-less clone is in the root [`README.md`](../README.md#prerequisites--git-lfs-is-required).

**Datasets** — the raw `.pt` artifacts a figure or audit is generated from
live in an arc's `data/` directory, also git-LFS-tracked (rule
`research/**/data/*.pt`), with a `MANIFEST.json` recording per-file sha256,
provenance, and capture-root/derived class. Committing the data — not just the
scripts — is what lets a clean clone re-render a figure or replay the audit.
Generating, validating, and saving the dataset is a required step of every
arc; the full discipline is in [`ARC_PROCESS.md`](ARC_PROCESS.md).

**Citations.** Load-bearing claims about LLM architecture / training /
interpretability cite a source — a paper key or a `theory/kb/` note — per the
discipline in the repo `CLAUDE.md` (*# Theory KB & citation discipline*).
Analogies and intuitions are tagged (`[INTUITION]`, `[ANALOGY]`,
`[SPECULATION]`) so they're never laundered as formal claims.
