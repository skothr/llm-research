# Research-arc process (SOP)

How to run a research arc in this workspace so it ends up reproducible,
honestly framed, and reviewable. This is the **process** doc (lifecycle +
disciplines); [`README.md`](README.md) is the **catalog** (what arcs exist +
layout/convention reference). The [`nla-verbalizer`](arcs/01_nla-verbalizer/) arc
is the worked example most sections point at, with two exceptions. For
**attribution** ([§ Attribution](#attribution--who-directed-who-executed)),
[`02_subliminal`](arcs/02_subliminal/) is the reference implementation of the
codified shape at minimal scale and [`04_jspace`](arcs/04_jspace/) is the
worked example at full scale; arc 01 predates the shape and was retrofitted
to it on 2026-08-17. For the **third-party-data gate**, **artifact
resolution** and **loud audit
degradation**, arc 04 is the reference; arcs 01 and 03 resolve artifacts
cache-first, the order this doc now inverts.

An *arc* is a focused, multi-observation investigation cohering around one
research question. A single loose finding is an *observation*; when several
cohere, promote them into an arc (see README § Arcs).

---

## The non-negotiables (read this first)

1. **Raw data is a deliverable, not scratch.** Every figure and every
   load-bearing number must be regenerable from committed inputs. Generating,
   *validating*, and *saving* the raw dataset is part of the work — not an
   afterthought. See [§ Raw data is a deliverable](#raw-data-is-a-deliverable).
2. **Claims are evidence-first and audit-locked.** Every load-bearing number
   in prose is re-derivable from the committed data by a script that fails
   loudly on drift. See [§ Audit](#5-audit-lock-the-numbers).
3. **Findings are framed at their true confidence.** One anchor pair is a
   "candidate," not a "property." Hold syntheses as hypotheses until scope
   tests replicate. See [§ Framing discipline](#framing-discipline).
4. **A clean clone can reproduce the arc.** `git clone && git lfs pull`, then
   the audit passes and any figure re-renders — with no access to your
   machine's caches. This is the acceptance bar for "done."
5. **Third-party data is vetted before first use.** A corpus or model artifact
   gets its licence, PII, realism/privacy-tradeoff and redistribution
   decisions recorded at the moment it is *selected* — not at commit time, by
   which point the experiments have already been run on it and re-running them
   is the expensive part. See
   [§ 1 Capture](#1-capture--validate--save-the-raw-dataset).

---

## Directory layout

```
research/arcs/<slug>/
  README.md            # arc entry point: motivation, findings (as hypotheses),
                       #   limitations, next paths, attribution
  observations/        # dated evidence-first writeups, one finding per file
    figures/
      INVENTORY.md      # per-figure provenance: what / source script / source data
      fig*.png          # generated plots (git-LFS)
    *.md / *.txt        # YYYY-MM-DD-<slug>.md observations
  data/                 # raw datasets (git-LFS) — see § Raw data is a deliverable
    MANIFEST.json       # per-file sha256 + provenance + class
    LICENSE-DATA.md     # third-party licence/attribution/PII record (§ 1 gate)
    README.md           # usage, copy-back, trust note
    audit_YYYY-MM-DD.log # the committed audit run the README's result cites
    *.pt                # capture + derived artifacts
    *.json              # corpora / item sets — NOT covered by the .pt LFS rule
    cache/              # gitignored working mirror (arc 04's convention)
  sessions/             # session-resume checkpoints (stale-fast; never load-bearing)
  plans/                # research/construction plans (as needed)
```

Generated artifacts (figures, datasets) are committed — drift detection beats
regenerability-in-principle: a committed artifact that drifted from its
generator is detectable; a regenerable-in-principle one is not.
git-LFS rules already cover `research/**/figures/*.png`,
`research/**/data/*.pt` and `research/**/data/cache/*.pt`.

Two gaps in that coverage to know about:

- **`data/*.json` is not LFS-tracked.** The `*.pt` rule matches extension, not
  size, so committed JSON corpora (arc 04's `fitting_prompts_*.json`, 1.0 and
  3.5 MB) are plain git blobs. Fine at that size; a corpus an order of
  magnitude larger needs its own `.gitattributes` rule before it lands.
- **`data/cache/` is gitignored by default** (`research/arcs/*/data/cache/`),
  with arc 04 carved out by explicit whitelist so the fitted lens tensors it
  keeps cache-only are still LFS-committed. Anything a clean clone must read
  belongs in `data/`, not in a cache whitelist.

---

## Lifecycle

A loose sequence, not a waterfall — arcs spiral (capture → analyze → new
question → capture). But each numbered step has a definition of done.

### 0. Set up

- Work in a git worktree (project hard rule — see the repo `CLAUDE.md`).
- Write down the **research question** in one sentence. If the arc is planned
  up front, drop a `plans/YYYY-MM-DD-<slug>.md`; if it's exploratory, the
  question can live in the arc README's motivation once it exists.
- **Open the attribution record on day one.** Create the arc README's
  `## Attribution` section — or an `attribution.md` in the arc root if the
  README doesn't exist yet — and paste the originating direction into it
  **as a dated quote, the same day it was given**. Its evidence expires;
  the rest of the arc's does not. Full rules in
  [§ Attribution](#attribution--who-directed-who-executed).
- **Pick the script prefix** for the arc and use it for every script in
  `examples/`. One prefix per arc, shared helpers named `_<family>_*.py`.

| Prefix | Arc | Shared helpers |
|---|---|---|
| `nla_` | `01_nla-verbalizer` | `_nla_artifacts.py` |
| `subliminal_` | `02_subliminal` | — |
| `emb_` | `03_embedding-atlas` | `_emb_artifacts.py` |
| `jspace_` | `04_jspace` | `_jspace_paths.py`, `_jspace_pursuit.py` |

`_layer_hooks.py` is the one family-neutral helper (hook plumbing; only arc
01's steering scripts use it today). A new arc takes a new prefix rather than
extending an existing family — the prefix is how a reader maps a script back to
the arc that owns it.

### 1. Capture → validate → save the raw dataset

This is the step most likely to be skipped under time pressure. Don't.

- **Vet third-party data at selection time.** The moment a corpus or a
  third-party model artifact is *chosen* — before the first run that consumes
  it — record four decisions in `data/LICENSE-DATA.md` beside the data, and in
  the arc's decision log alongside the scientific rationale:
  1. **Licence and what it obliges.** Name it and quote the obligation (ODC-BY
     §4.2/4.3 want the licence URI and a source-attribution notice; CC BY-SA
     below 4.0 is *not* one-way compatible with GPLv3). The repo's
     `GPL-3.0-only` covers code and original prose only and must be explicitly
     scoped away from third-party data.
  2. **Personal data.** Assume any web-scraped corpus (C4, OSCAR, RefinedWeb,
     The Pile, anything Common-Crawl-derived) contains PII and prove otherwise;
     `examples/jspace_redact_corpus.py --report` is the starting scanner —
     extend its pattern classes rather than writing a new one. Curated
     encyclopedic sources (WikiText, Wikipedia dumps) largely do not.
  3. **The realism/privacy tradeoff, stated.** A corpus picked *because* it is
     more representative of real text is, for that same reason, likelier to
     carry real people's data. If breadth or naturalness is the scientific
     argument, that argument is itself the signal to check.
  4. **Redistribution decision.** Committing raw third-party text republishes
     it under your name. Prefer a deterministic regeneration script plus a
     checksum; where the text must be committed, redact first and document the
     redaction, its class coverage, its known limits, and how to reproduce it.

  `research/arcs/04_jspace/data/LICENSE-DATA.md` is the reference
  implementation. Arc 04 is also the worked example of what skipping the gate
  costs: its C4-en slice was found to carry third-party personal data after the
  experiments had run, and because the redaction is not length-preserving
  every C4-computed result had to be regenerated on the redacted text — a
  multi-week correction (redacted 2026-07-29, re-run and closed 2026-08-16)
  for a check that belongs at selection time. No licence
  can authorise republishing a third party's personal data; data-subject
  rights attach to the person, not the licensor.
- **Capture.** Run the experiment; write the raw tensors/records to the arc's
  gitignored working cache. **New arcs use `research/arcs/<slug>/data/cache/`**
  (arc 04's convention — the cache sits beside the deliverable it mirrors);
  arcs 01 and 03 use the older repo-level `.cache/<family>_artifacts/`.
- **Validate immediately, before building anything on top:**
  - Sanity-check the capture *protocol*: right layer, right position index,
    right tokenizer special-token handling, expected shapes/dtypes/counts. A
    wrong capture produces consistent-but-incorrect numbers that no downstream
    audit can catch (see L7 in the NLA arc — a position-index bug that was
    only caught by reading the code, not the numbers).
  - Eyeball distributions for the obvious failure (all-zeros, NaNs, collapsed
    variance, off-by-one counts).
- **Save to the committed `data/` dir** and write/refresh the manifest. Each
  arc owns an `examples/<family>_data_manifest.py` (`nla_`, `emb_`, `jspace_`):
  ```bash
  cp <working-cache>/*.pt research/arcs/<slug>/data/
  python examples/<family>_data_manifest.py --write # writes MANIFEST.json
  python examples/<family>_data_manifest.py --check # verifies sha256
  ```
  Copy the nearest existing one as the template for a new arc. Arc 02 predates
  the convention and carries an interim per-dataset manifest instead
  (`data/step0-owl-neutral-decode/manifest.json`, `manifest_version`
  `0.1.0-interim`, written inline by the capture script — no standalone
  generator, no `--check` re-verification mode); migrating it to the shared
  shape is tracked as issue #53.

**Done when:** the dataset is in `data/`, the manifest `--check` passes, and
you've confirmed the capture protocol is what you intended.

### 2. Analyze / derive

- Derived artifacts (cosine matrices, PCA, classifier outputs) get their own
  `.pt` in `data/`, produced by a committed script that reads only other
  `.pt`. Record each in the manifest as `class: derived` with its `inputs`.
- Keep derivation deterministic and scripted — no notebook-only state.

### 3. Figures + provenance

- Each figure is generated by a committed render script reading from `data/`.
- Every figure gets an `INVENTORY.md` entry: what it shows, source script,
  source data, model deps, preprocessing/assumptions, and any correction
  applied. Supersede-don't-delete: if fig N is wrong, add fig M and mark N
  DEPRECATED with the reason (the NLA arc's fig15→fig16 and fig23→fig25 are
  the pattern).

**Done when:** INVENTORY ↔ figures is a bijection and every named script/data
exists.

### 4. Observation writeups

- One finding per file, `YYYY-MM-DD-<slug>.md`, evidence-first. The field list
  is in the repo `CLAUDE.md` § *Research arcs & observations*. Copy this
  skeleton:

  ```markdown
  # Observation: <one-line headline stating the finding, not the topic>

  **Date/context:** YYYY-MM-DD. Stage/arc, model (exact id + dtype/quant),
  params (n, layers, seed), and what run produced this.

  ## Finding

  ## Evidence

  ## Reproducibility

  ## Hypotheses

  ## Follow-ups

  ## References
  ```

  These six heading spellings are canonical **for new observations**:
  cross-links from the arc README and from other observations anchor on them,
  so `## Findings` or `## Repro` breaks a link rather than reading as a
  synonym. Existing arcs predate the rule and are not retrofitted — arc 03
  uses `## Findings`, and arc 04 deviates in places — so read the skeleton
  above as the reference, not any one arc's files.
- Null results are findings — title them as such (`*-null-result.md`) and
  frame them as null, not as buried positives.
- **Append the attribution block now, not at close.** If this observation
  exists because someone asked for it, redirected it, set a standard, or
  caught an error, copy that turn verbatim and date-tagged into the arc's
  attribution record while the transcript is still at hand
  ([§ Attribution](#attribution--who-directed-who-executed)).
- Fill every field. No `TBD`, `TODO`, "implement later", or "handle edge
  cases" in a committed doc — a step that says *what* without showing *how*
  (exact command, code, or value) is a placeholder too. The commit-hash field
  is the one people fudge: put the real SHA in on the follow-up commit rather
  than leaving `TBD`.

### 5. Audit (lock the numbers)

Write/extend an arc audit script (templates: `nla_audit_findings.py`,
`jspace_audit_findings.py` — the latter for the resolution and
loud-degradation shape below) that
**re-derives every load-bearing number from the committed `data/`** and asserts
it against an expected constant, printing `PASS`/`FAIL` and a final
`SUMMARY: N PASS | M FAIL`.

- Re-derive from first principles where you can (don't regress against a cached
  intermediate you're also trying to validate).
- Make the script **self-locating, committed-data first, cache as fallback**
  so a clean-clone run is the default path (see § Raw data is a deliverable,
  *Wiring*; `examples/_jspace_paths.py` is the shared resolver).
- Audit the **headline result**, not just the easy structural counts. The NLA
  audit originally locked the geometry but not the round-trip faithfulness
  cosines that the whole arc rests on — a reviewer caught it; AUDIT 20-21 close
  it. When you add a figure or a claim, add its audit line in the same change.
- For qualitative decode claims, assert the **content** (a substring of the
  decoded text), not just non-emptiness (NLA AUDIT 17 asserts the decoded
  identities London/Spain/China).
- **Degrade loudly, never silently.** A missing or unreadable input is a
  recorded `FAIL` with a name, not a skipped check and not a traceback. Keep
  the failure modes distinguishable: `MISSING` for an absent file versus
  `LFS pointer stub — run git lfs install && git lfs pull` for a file that is
  present but is still an LFS pointer. An audit that *skips* what it cannot
  load reports a clean summary on a clone that can reproduce nothing — the one
  failure the audit exists to catch. `examples/jspace_audit_findings.py`
  (`load_pt_or_fail`) is the implementation to copy; `emb_audit_findings.py`
  carries the same stub check against arc 03's artifacts.
- **State the expected result with its measurement date and the log.** The arc
  README quotes the summary line, the date it was measured, and the committed
  run behind it — arc 04: `986 PASS | 4 FAIL`, 2026-08-17,
  `data/audit_2026-08-17.log` — and accounts for every non-zero FAIL (there,
  presence checks for two lenses deliberately not refit). Where the expected
  count depends on what a reader has fetched, state each state separately
  (arc 04 gives both the default-clone and the full-cache totals). A bare
  "the audit passes" cannot be checked against a re-run, and a summary with no
  date silently ages. `*.log` is gitignored repo-wide (a LaTeX-byproduct rule),
  so a promoted audit log needs an explicit negation
  (`!research/arcs/<slug>/data/audit_YYYY-MM-DD.log`) in `.gitignore`, added in
  the same commit — otherwise the README cites a file no clone has.
- Helper logic with real branching (corpus redaction, segment fitting) gets a
  unit test in `examples/tests/` — `pip install -e '.[dev]'`, then `pytest
  examples/tests/`. The audit locks the *numbers*; the tests lock the
  *transformations* that produced them.

**Be honest about what the audit does NOT catch:** it verifies arithmetic
consistency *given the captures* — not capture-protocol bugs, not interpretive
overreach, not whether a threshold was the right choice, and not that the prose
was transcribed faithfully into the script's expected constants (those are
maintained by hand). State this in the arc README. "N PASS" means "the numbers
agree," never "the methodology is right."

**Done when:** the audit passes from a clean clone and every load-bearing
number in the observations has a corresponding assertion.

### 6. Arc README synthesis

- **Findings as hypotheses** with explicit scope qualifications (see Framing).
- **Limitations** section, ranked by how far they constrain the claims.
- **Possible next paths**, each tied to a question and (if tracked) a ticket.
- **Attribution** — group the blocks accumulated since § 0 into named
  directions, then write the three-way split and the verifiability note. Rules,
  labels and the copy-paste template are in
  [§ Attribution](#attribution--who-directed-who-executed); do not restate
  them per arc.
- Cross-link: README → observations → figures/INVENTORY → data/MANIFEST.

### 7. PR

- Push the branch, open a PR (one arc = one scope-bounded diff). `git lfs pull`
  works for reviewers. Run `/ultrareview <PR#>` if warranted. Merge per the
  repo's manual-merge SOP.
- **Verify the content actually landed — don't trust the merged badge.**
  After merge run `git branch -r --no-merged origin/main` (should be empty)
  and `git merge-base --is-ancestor <merge-sha> origin/main` for each PR in
  a stack — a GitHub "merged" badge only means the PR merged into *its
  base*, which may be a stale intermediate branch. This check was
  instituted at the project owner's direction on 2026-07-21, after a
  user-requested verification of the arc-03 4-PR stack found only #20 had
  reached main, stranding 22 commits of arc-03 content (merge `2400a95f`
  is the repair). Standing project-owner policy from the same directive:
  merged branches are the permanent record of merge points and are never
  deleted — so stacks must be merged top-down (or each PR retargeted with
  `gh pr edit N --base main`), since the delete-triggered auto-retarget
  path is unavailable here.

---

## Raw data is a deliverable

The discipline the rest of this doc leans on, stated once, in full.

**Why.** Scripts + figures without the data they were generated from are not
reproducible and not verifiable. A reviewer can't check a figure against its
source; a future session can't re-run the audit; "trust me, it computed +0.87"
is not evidence. The figures and the audit are *downstream* of the data — ship
the data.

**What to commit.** All artifacts a figure or audit consumes:

- **Capture-roots** — produced by a run that loads a model (expensive,
  CPU-hours). Always commit; they are irreplaceable without re-running the
  experiment.
- **Derived** — produced cheaply from other `.pt` by a committed script.
  Commit these too: the marginal MB buys bit-exact figure + audit
  reproduction with zero model load. (Only consider shipping roots-only +
  a regeneration script if the derived set is genuinely large; if you do,
  `log`/document what was dropped — silent truncation reads as completeness.)

**Where.** `research/arcs/<slug>/data/`, git-LFS-tracked via the existing
`research/**/data/*.pt` rule. Keep your working/scratch captures in the
gitignored cache — `research/arcs/<slug>/data/cache/` for a new arc,
`.cache/<family>_artifacts/` in arcs 01 and 03 — and treat the committed
`data/` dir as the canonical copy.

**Wiring (so the data is *usable*, not just stored).** Scripts resolve inputs
through one shared helper and write outputs only to the cache. That one
indirection is what lets the *same* script a developer runs locally (writing
fresh captures to the gitignored cache) also re-render figures and replay the
audit on a clean clone — no manual copy step, no clone-vs-local branching.
Without it, committed data is inert: the scripts still point at an empty cache.

**Resolution order: committed `data/` first, cache only as fallback.**
`examples/_jspace_paths.py` (`resolve`) is the prescription — it returns the
committed copy when the name exists there and drops to `data/cache/` only for
names deliberately kept cache-only. The inverse order is the trap: under
cache-first, a stale or locally-regenerated cache **silently shadows the
committed deliverable**, so the developer's figures and audit re-derive from
bytes no reviewer has, and drift between the two copies is invisible until
someone clones fresh. Committed-first makes the developer see exactly what a
reviewer sees, and a stale cache announce itself. Arcs 01 and 03
(`_nla_artifacts.py`, `_emb_artifacts.py`) still resolve cache-first, with the
stated rationale that a fresh local re-capture is picked up immediately. That
is the older pattern; new arcs do not copy it.

**Manifest.** A `data/MANIFEST.json` (generator:
`examples/<family>_data_manifest.py`) records per file: `filename`, `sha256`,
`size_bytes`, `class` (capture-root | derived), `producing_script`,
`producing_command`, `inputs` (upstream `.pt`), `requires_model`
(none | base | +av/+ar/…), `consumers` (figures / downstream artifacts /
audit). The generator's `--check` mode re-verifies every sha256 — run it in
the audit step as a drift detector.

**Validate before you save.** "Save" includes confirming the data is *correct*
(protocol sanity, shapes, no NaNs/collapse) and *locked* (audit re-derives the
load-bearing numbers; manifest pins the bytes). A committed wrong dataset is
worse than none.

**Trust note.** `torch.load(..., weights_only=False)` executes pickle on load.
Fine for locally-generated tensor dumps; never normalize it for third-party
data. The manifest's sha256 lets a consumer verify integrity before loading.

---

## Framing discipline

The review's central rigor finding: late, careful scope-qualifications in the
arc README hadn't propagated back into the individual observation files, so the
most quotable sentences read as settled where the synthesis read as hypothesis.

- **State claims at their evidence level.** One anchor pair / one layer / one
  model → "candidate," "appears to," "for this configuration." Generalizing to
  a property of the model (or of transformers) requires the cross-condition
  scope test — name it as a follow-up rather than asserting the general claim.
- **Tag, don't launder.** `[INTUITION]` / `[ANALOGY]` / `[SPECULATION]` /
  `[CONTRADICTION]` per the repo `CLAUDE.md`. A hypothesis block (explicit
  "H1: …, to test: …") is the functional equivalent and keeps the speculation
  out of the findings prose.
- **Cite load-bearing external claims** (a paper or `theory/kb/` note).
  Inline arXiv/URL citation is the right register for `research/` (these are
  observations, not KB notes).
- **Propagate corrections everywhere.** When a number/location is refined
  (e.g. NLA's t=0.421 → plateau relocation), back-reference the refinement from
  every file that states the old value. Dated observations are snapshots, but a
  one-line "later refined, see X" pointer keeps a reader from taking a
  superseded number as current.
- **Disclose at the entry point, not only in the arc.** A correction that
  changes committed data, retracts a number, or alters a stated conclusion
  surfaces in the **root `README.md`** with its close date — a reader who
  never opens the arc still has to meet it. Arc-internal back-references
  (previous bullet) stay as well; they are the detail, not the disclosure.
  Arc 04's C4-redaction correction is the worked example: root README
  disclosure with the 2026-08-16 close date, per-claim record in the arc.
- **No emojis** anywhere in committed docs — use `★ → • ─` or bold for
  emphasis. **No placeholders** either: `TBD`, `TODO`, "fill in later", or a
  step describing *what* without the concrete *how* does not ship in a
  committed doc.

---

## Attribution — who directed, who executed

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; conflating them hides where the ideas came from.

**Attribution is recorded as the arc runs and synthesized at close**, because
its evidence is perishable in a way the rest of the arc's evidence is not.
Data, figures and audit logs sit in the repo indefinitely; the direction turns
live in session transcripts that are machine-local, uncommitted, and subject to
local retention, so a turn that is trivially quotable today can be gone later.
Recording late also invites a subtler failure: writing the record from
recollection and reaching for an explanation when the evidence is merely
*absent*. This document asserted, until 2026-08-19, that a 33-day transcript
window covering arc 03's phases 2-3 had been lost to retention. Nothing
supports that: transcripts of other projects survive from those same days, the
window carries no session in this project's local transcript directory and no
commit between 06-11 and 07-15 — which is all the instrument can say, not a
statement about what the human was doing — and the directive filed as
unrecoverable was sitting quotable in the kickoff session all along.

**Retrofitting is not verifying.** When attribution is written onto an arc
after the fact, the prose it starts from is a draft, not a source. Trace each
claim to a turn before you label it: an inherited sentence that cannot be
sourced is withdrawn, not relabelled `[RECONSTRUCTED]` — that label is for a
contribution you know happened and cannot quote, and spending it on an
unsourced claim launders the claim instead of flagging it. Arc 03 carried a
"pause/resume calls" contribution for a month because one pass composed it
into a list and two later passes asked only whether it was quotable, never
where it came from. All four arcs wrote their attribution post-hoc; three then
needed a corrective rewrite (`362ae6fb`), including one where a user's own
scope-broadening had been elided from a quote and the surrounding prose
credited that broadening to Claude.

### When to record

- **§ 0, set up.** Open the attribution record the day the arc opens — the arc
  README's `## Attribution` section, or an `attribution.md` in the arc root
  until the README exists — and put the originating direction in it, quoted,
  that day.
- **§ 4, observations.** Whenever an observation exists because someone asked
  for it, redirected it, set a standard, or caught something, append that turn
  **verbatim and date-tagged** while the transcript is still at hand. This is
  the step that makes the close cheap and the record honest.
- **§ 6, synthesis.** Group the accumulated blocks into named directions, then
  write the three-way split and the verifiability note. A paused arc's split
  carries an explicit "as of `<date>`, covering `<phase>`" — a split written
  mid-program is a snapshot, and saying so keeps a later reader from reading it
  as final.

### Source of truth, and the honesty labels

**The transcript is the source of truth; recollection is not.** Recover the
originating turn before quoting it. Every direction block carries **exactly
one** provenance label:

| Label | Means |
|---|---|
| `[VERBATIM]` | Every retained word exactly as typed — no normalization, no rewording. Cutting is allowed where the cut is **marked in place** with `[...]` and the section preamble says so; an elision must never remove the direction the quote is cited for, or reverse its sense. Nothing else may be altered. |
| `[NORMALIZED]` | Quoted with typo/punctuation fixes and markdown emphasis dropped. State the normalization once per section rather than per block. |
| `[PARAPHRASE]` | The idea is the human's, the wording is Claude's. **Never** wrapped in quotation marks. Where it matters, name which words were actually theirs. |
| `[SELECTED]` | The human chose an option Claude offered — a checkpoint menu, a numbered plan — rather than typing it. Describe the choice; never render it as a typed quote. |
| `[RECONSTRUCTED]` | No transcript survives, or none was consulted. State what the block was written from. |

`[RECONSTRUCTED]` is an acceptable label and an unacceptable silence: an
unlabelled block asserts a provenance it does not have.

**`[RECONSTRUCTED]` is not yours to apply alone.** Before writing it, surface
the claim to the human — quote it, say which sessions were searched, and ask
what they remember. They are a source the transcript search cannot reach: they
may recall the turn well enough to label it properly, point at a record that
lives outside the transcripts, or tell you it never happened. Deciding
unilaterally fails in both directions, and this repo has managed both within
one week — a claim that was never sourced spent a month laundered under a
provenance label, and its correction then withdrew a second claim that was
substantially sourceable. Ask; then label what the answer supports.

**Role tags** — `[HUMAN-DIRECTED]` / `[AI-EXECUTED]` / `[JOINT]` — are used
only where the default would mislead. The default: direction blocks are
human-directed, everything downstream of them is AI-executed. Tag the
exceptions in **both** directions — a proposal that was Claude's and the human
endorsed is named as Claude's; a catch the human made that Claude missed is
named as theirs and **quoted**.

### Granularity

**Per named direction, plus one roll-up** — the roll-up being the
human/Claude/emergent split. A named direction is a turn (or a tight cluster of
turns) that fixed something: the originating question, a
design sign-off, a standard, a scope call, a reopening. Name each block by what
it *established*, not by when it happened — "The artifact-verification
standard", not "2026-07-20 session".

Per-claim and per-figure attribution are **out of scope**. Figures are
AI-executed under human standards; tagging each one adds noise without adding
information. Arc 04 is the worked example at full scale (seven named
directions, one `[JOINT]` origin note, plus the split, which quotes two
further user catches); arc 02 is the same shape at minimal scale (three
blocks) — an arc with one direction turn gets one block, not a manufactured
seven.

Arc 02 is a *reference implementation* rather than a predecessor of this
shape: it and the first codification of the requirements were written in the
same commit (`5040118e`, 2026-07-18). Arc 01's "Research direction —
user-shaped themes" is the informal precursor the shape was generalized
*from*; follow arc 02 where the two differ.

### Template

Place `## Attribution` after the arc's question/motivation and before the
findings. Copy this skeleton:

````markdown
## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Quotes are the human's typed turns, taken from the session transcripts
listed under **Verifiability** (<name>). Each block carries a provenance
label; `[NORMALIZED]` means typo and punctuation fixes with markdown emphasis
inside the turn dropped, and `[...]` marks an editorial elision — elisions
never remove the direction the quote is cited for. Shape per
`research/ARC_PROCESS.md` § Attribution.

### Research direction

**<What this direction established>** [session YYYY-MM-DD] `[VERBATIM]`:

> *"<the turn as typed>"*

<One or two sentences: what this fixed, and which downstream choice follows
from it.>

**<Next direction>** [session YYYY-MM-DD] `[PARAPHRASE]`: <the idea, unquoted,
naming which words were the user's.>

### Human / Claude / emergent split

**User (<name>).** <Direction, scope calls, standards set; catches Claude
missed, quoted.>

**Claude Code.** <Design, implementation, write-ups; proposals of Claude's the
user endorsed, named as such.>

**Emergent.** <What neither party produced alone.>

*Split as of YYYY-MM-DD, covering <phase>.*  <!-- paused arcs only -->

### Verifiability

Every quote above is recoverable from the sessions below. The transcripts are
machine-local and are not committed to this repo — they carry local paths and
tool output — so they are referenced by session id and date only. A session id
here is the 8-character prefix of the local session UUID; that prefix is the
repo-wide convention and is what a later reader matches on.

| Session | Span | Covers |
|---|---|---|
| `<session-id>` | YYYY-MM-DD | <what was directed in it> |

Claims in this section that are not in quotation marks are Claude's
characterization of the user's direction, not the user's wording. Blocks
labelled `[RECONSTRUCTED]` have no surviving transcript.
````

### Public-repo constraints

This repo is public, and an attribution section quotes a private working
session. Four hard limits:

- **Never a machine path, hostname, or username.** A session is identified by
  its id and date and nothing else; the id is opaque and carries no path.
- **Never paste tool output or file listings** from a transcript. Quote the
  human's typed turns only — everything else in a transcript is the agent's
  work product and routinely contains local paths.
- **One named person per arc** — the repo's commit author. Third parties named
  in a turn are not identified.
- **A turn carrying credentials, private data, or third-party material gets
  `[PARAPHRASE]`**, never a quote. Editing a published quote does not unpublish
  it: git history and any fork keep the original, so redact **before** commit.

### Reconstructing attribution from transcripts

When a record was not kept as the arc ran, it can sometimes be rebuilt. The
procedure, in portable form:

1. **Locate the transcripts.** They live under
   `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl` on the machine that
   ran the sessions. Nothing here is committed; treat the location as an input
   to the reconstruction, never as a citation.
2. **Search every candidate directory, not just the current one.** A repo that
   moved, or was split out of a larger checkout, has its transcripts
   **partitioned across multiple encoded-cwd directories**. This repo's arcs 01
   and 02 ran before the split, so their transcripts sit under the predecessor
   project's directory.
3. **Key on the UUID, not the path.** The same session id can appear as a stub
   in one directory and as the full transcript in another. The UUID is the
   durable identifier; the path is not.
4. **Calibrate before trusting.** Run the reconstruction against an arc whose
   attribution record is known-good, and check that it recovers what is already
   documented. An uncalibrated reconstruction that "finds" plausible turns is
   the failure mode this whole section exists to prevent.
5. **Verify each candidate quote by content-grep** against the transcript
   before it goes in — a recalled turn that reads right is not evidence.
6. **Tag what the transcript does not settle.** Where the record does not
   establish who originated an idea, mark the sentence
   `[AMBIGUOUS: <the unresolved question>]` rather than deciding silently. An
   ambiguity named is a finding; an ambiguity resolved by convenience is a
   misattribution.
7. **Know what the record cannot see, and never characterize outside it.**
   Transcripts and commits capture CLI sessions and repo writes on one
   machine. They do not capture work in a web UI, on another machine, in
   another tool, on paper, or in someone's head — and a human's thinking is
   not bounded by any of them. So these sources answer exactly one question:
   *is this claim about the arc's direction sourced?* They cannot answer
   *what was the human doing* — and an absence in them is not an absence in
   fact. Do not write, and do not conclude, that a recollection about the
   human's own work is wrong because it left no trace here; report what the
   record shows, name the instrument, and let them fill the rest in.

---

## Sessions are not findings

`sessions/` files are operational checkpoints (worktree path, branch tip,
"what's next") that go stale within hours. They are never load-bearing for a
claim. Don't rewrite old session snapshots to match current state — newer files
supersede older ones, and the README/INVENTORY/audit carry the durable record.

---

## New-arc checklist

```
[ ] worktree created; research question written in one sentence
[ ] script prefix chosen; attribution record opened with the originating
        direction quoted, dated, and provenance-labelled
[ ] third-party data vetted AT SELECTION: licence + PII + realism/privacy
        tradeoff + redistribution decision recorded in data/LICENSE-DATA.md
[ ] capture run; protocol validated (layer/position/shapes/counts sane)
[ ] raw data saved to arcs/<slug>/data/ ; MANIFEST.json written; --check passes
[ ] derived artifacts scripted + in data/ + in manifest (class: derived)
[ ] figures generated by committed scripts; INVENTORY.md bijection complete
[ ] observations written (evidence-first, fields filled, nulls labeled as null)
[ ] audit script re-derives every load-bearing number incl. the headline;
        degrades loudly (MISSING vs LFS-pointer-stub, never skip); passes from
        a clean clone; "what it can't catch" stated in README; README quotes
        the summary + measurement date + committed audit_YYYY-MM-DD.log
[ ] arc README: findings-as-hypotheses + limitations + next-paths + attribution
        (§ Attribution template: labels, split, verifiability table)
[ ] clean-clone test: git lfs pull → audit PASS → a figure re-renders
[ ] PR opened (one arc = one scope-bounded diff)
```
