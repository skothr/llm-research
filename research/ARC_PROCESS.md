# Research-arc process (SOP)

How to run a research arc in this workspace so it ends up reproducible,
honestly framed, and reviewable. This is the **process** doc (lifecycle +
disciplines); [`README.md`](README.md) is the **catalog** (what arcs exist +
layout/convention reference). The [`nla-verbalizer`](arcs/01_nla-verbalizer/) arc
is the worked example most sections point at — with one exception: for
**attribution** (checkpoint 1 and § Arc README synthesis) the reference
implementation is [`02_subliminal`](arcs/02_subliminal/), not arc 01.

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
   loudly on drift. See [§ Audit](#audit-lock-the-numbers).
3. **Findings are framed at their true confidence.** One anchor pair is a
   "candidate," not a "property." Hold syntheses as hypotheses until scope
   tests replicate. See [§ Framing discipline](#framing-discipline).
4. **A clean clone can reproduce the arc.** `git clone && git lfs pull`, then
   the audit passes and any figure re-renders — with no access to your
   machine's caches. This is the acceptance bar for "done."
5. **Arc code depends on standard scientific libraries and nothing
   in-house.** Scripts use torch, transformers, numpy, matplotlib and the
   like, and implement their methods directly. An arc that replicates a paper
   with a reference implementation may declare that implementation as a
   pinned dependency, because replacing it would weaken the replication. No
   in-house helper library sits between the scripts and the models. `jlens`
   (anthropics/jacobian-lens) is the reference implementation of the paper
   arc 04 replicates and stays, its commit recorded in that arc's MANIFEST. The
   `llm-surgeon` import is a leftover of the 2026-06 repository split and is
   being removed (issue #94).

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
    README.md           # usage, copy-back, trust note
    *.pt                # capture + derived artifacts
  sessions/             # session-resume checkpoints (stale-fast; never load-bearing)
  plans/                # the arc plan (every arc) + later design docs
```

Generated artifacts (figures, datasets) are committed — drift detection beats
regenerability-in-principle.
git-LFS rules already cover `research/**/figures/*.png` and
`research/**/data/*.pt`.

---

## Lifecycle

An arc runs through four checkpoints. Each checkpoint produces a defined set
of files, has a definition of done, and closes with a reviewed PR
([§ The checkpoint PR](#the-checkpoint-pr)).

**Iteration.** Checkpoints 2-4 may repeat as the arc progresses: a
follow-up inside the arc's stated question sends the arc back to new scripts,
new data or new figures. Re-entering an earlier checkpoint opens a new PR
scoped to the delta (the changed scripts, the new data, the revised figures),
closed by the same review loop. A re-entry that adds a capture run records
its predictions in a plan amendment before that run. Checkpoint 1 is
re-entered when the arc's question itself changes; that re-entry is a plan
amendment too. The plan says where the checkpoints fall for its arc. A small
arc may merge checkpoints 3 and 4 into one PR, and its plan must say so.

### Checkpoint 1: question, research, plan

Produces the research question, the plan, and the worktree the arc runs in.

- Work in a git worktree (project hard rule — see the repo `CLAUDE.md`).
- Write down the **research question** in one sentence.
- Write the **plan** as `plans/YYYY-MM-DD-<slug>.md`. Every arc has one.
  Its length scales with the arc: an exploratory arc with no GPU run over
  about an hour needs a few lines; a multi-run arc needs the full design.
  Arc 01 predates this rule and is not retrofitted. The plan names:
  - the question;
  - the predictions, recorded before any capture runs;
  - the third-party data vetting record for every external dataset, corpus
    or model artifact (the rule is the repo `CLAUDE.md`
    § Third-party data — vet BEFORE first use, not before commit);
  - the dependency choice, per the dependency policy in
    [§ The non-negotiables](#the-non-negotiables-read-this-first);
  - where the arc's later checkpoints fall.
- Note the **direction-setting** as it happens (who asked what). The
  human-direction vs AI-implementation split is worth recording honestly; the
  [subliminal arc](arcs/02_subliminal/README.md) README's Attribution
  section is the template (attribution shape codified in
  [§ Arc README synthesis](#arc-readme-synthesis)).

**Done when:** the plan is in `plans/` and has been reviewed.

**Closed by:** a PR plus the review loop in
[§ The checkpoint PR](#the-checkpoint-pr).

### Checkpoint 2: setup and implementation

Produces the code the long compute run will execute, reviewed before that run
starts.

- The capture and analysis scripts.
- The manifest and audit scaffolding: the arc's manifest generator and an
  audit script with its structure in place, ready to take assertions.
- A dry run at tiny n that exercises capture, derivation and the manifest
  end to end. Its outputs are not committed as data.
- `pyright` clean on every new or changed script.

This checkpoint is reviewed before any long compute run, because a defect
found here costs a code fix while the same defect found after the run costs
the run. Arc 04 is the evidence for gating before the run: the C4 personal
data vetting (a checkpoint 1 item) happened after the fits and cost 22.9 h of
refit queue time (the three refits in
[arc 04 `data/README.md`](arcs/04_jspace/data/README.md)), and the K/d
mismatch in the paper-metric comparison was found after the arc closed.

**Done when:** the dry run completes, pyright reports zero diagnostics, and
the scripts, manifest generator and audit scaffold are in the branch.

**Closed by:** a PR plus the review loop in
[§ The checkpoint PR](#the-checkpoint-pr), merged before the long run starts.

### Checkpoint 3: computation, processing, validation

Produces the committed dataset, the derived artifacts, and the audit that
locks the numbers. The audit closes this checkpoint: the numbers are locked
before any narrative is written.

#### Capture → validate → save the raw dataset

This is the step most likely to be skipped under time pressure. Don't.

- **Capture.** Run the experiment; write the raw tensors/records to the arc's
  working location. (NLA arc writes to the gitignored cache
  `.cache/nla_artifacts/` during development.)
- **Validate immediately, before building anything on top:**
  - Sanity-check the capture *protocol*: right layer, right position index,
    right tokenizer special-token handling, expected shapes/dtypes/counts. A
    wrong capture produces consistent-but-incorrect numbers that no downstream
    audit can catch (see L7 in the NLA arc — a position-index bug that was
    only caught by reading the code, not the numbers).
  - Eyeball distributions for the obvious failure (all-zeros, NaNs, collapsed
    variance, off-by-one counts).
- **Save to the committed `data/` dir** and write/refresh the manifest:
  ```bash
  cp <working-cache>/*.pt research/arcs/<slug>/data/
  python examples/nla_data_manifest.py --write # (re)writes MANIFEST.json
  python examples/nla_data_manifest.py --check # verifies sha256
  ```
  (The manifest script is arc-specific; copy it as the template for a new arc.)

**Done when:** the dataset is in `data/`, the manifest `--check` passes, and
you've confirmed the capture protocol is what you intended.

#### Analyze / derive

- Derived artifacts (cosine matrices, PCA, classifier outputs) get their own
  `.pt` in `data/`, produced by a committed script that reads only other
  `.pt`. Record each in the manifest as `class: derived` with its `inputs`.
- Keep derivation deterministic and scripted — no notebook-only state.

#### Audit (lock the numbers)

Write/extend an arc audit script (template: `nla_audit_findings.py`) that
**re-derives every load-bearing number from the committed `data/`** and asserts
it against an expected constant, printing `PASS`/`FAIL` and a final
`SUMMARY: N PASS | M FAIL`.

- Re-derive from first principles where you can (don't regress against a cached
  intermediate you're also trying to validate).
- Make the script **self-locating with a committed-data fallback** so it runs
  from a clean clone (the NLA audit prefers the gitignored cache, falls back to
  `data/`).
- Audit the **headline result**, not just the easy structural counts. The NLA
  audit originally locked the geometry but not the round-trip faithfulness
  cosines that the whole arc rests on — a reviewer caught it; AUDIT 20-21 close
  it. When you add a figure or a claim, add its audit line in the same change.
- For qualitative decode claims, assert the **content** (a substring of the
  decoded text), not just non-emptiness (NLA AUDIT 17 asserts the decoded
  identities London/Spain/China).

**Be honest about what the audit does NOT catch:** it verifies arithmetic
consistency *given the captures* — not capture-protocol bugs, not interpretive
overreach, not whether a threshold was the right choice, and not that the prose
was transcribed faithfully into the script's expected constants (those are
maintained by hand). State this in the arc README. "N PASS" means "the numbers
agree," never "the methodology is right."

**Done when:** the audit passes from a clean clone and every load-bearing
number the arc will report has a corresponding assertion.

**Closed by:** a PR plus the review loop in
[§ The checkpoint PR](#the-checkpoint-pr).

### Checkpoint 4: observations, conclusions, artifacts

Produces the figures, the observation writeups, and the arc README synthesis,
all built on the numbers checkpoint 3 locked. A claim or figure written
here that rests on a number the audit does not yet assert adds that
assertion in the same PR.

#### Figures + provenance

- Each figure is generated by a committed render script reading from `data/`.
- Every figure gets an `INVENTORY.md` entry: what it shows, source script,
  source data, model deps, preprocessing/assumptions, and any correction
  applied. Supersede-don't-delete: if fig N is wrong, add fig M and mark N
  DEPRECATED with the reason (the NLA arc's fig15→fig16 and fig23→fig25 are
  the pattern).

**Done when:** INVENTORY ↔ figures is a bijection and every named script/data
exists.

#### Observation writeups

- One finding per file, `YYYY-MM-DD-<slug>.md`, evidence-first. Format spec is
  in the repo `CLAUDE.md` § Research arcs & observations: date+context (model,
  params), finding, evidence (excerpts), reproducibility (exact commands),
  hypotheses, follow-ups, references.
- Null results are findings — title them as such (`*-null-result.md`) and
  frame them as null, not as buried positives.
- Fill every field. Specs, plans and observation files carry no `TBD`/`TODO`
  placeholders; every step has its content.
  The commit-hash field is the one exception people fudge — put the real SHA in
  on the follow-up commit rather than leaving `TBD`.

#### Arc README synthesis

**Canonical section set and order** (owner directive 2026-08-17, #57).
Every arc README presents these sections in this order, each a real `##`
heading (a clarifying suffix after an em-dash is fine; arc-specific extras
slot between Findings and Limitations; existing arcs are being retrofitted
to this order under #57):

1. Unheaded lead — a one-paragraph statement of the arc plus a dated
   **Status** line.
2. `## The question` — what the arc asks and why; motivation folds in.
3. `## Findings` — per the requirements below, with the arc's headline
   figures embedded inline.
4. `## Limitations`
5. `## Attribution`
6. `## Possible next paths` — deferred/follow-up items live here.
7. `## Reproducing` — exact commands plus the dated expected audit state.
8. `## File map`

**Inline figures:** a reader meets the headline results as figures in the
README itself, not only via `figures/INVENTORY.md` — embed each headline
figure at the finding it supports, with a caption saying what it shows and
a provenance link to its INVENTORY row. Non-headline figures stay
INVENTORY-only.

**Showcase boundary:** READMEs carry final state plus the caveats a reader
needs; change chronology and correction mechanics compress to a dated
pointer (git history, `observations/`, and `plans/` carry the detail) —
see `CLAUDE.md` § Showcase vs history.

Per-section requirements:

- **Findings as hypotheses** with explicit scope qualifications (see Framing).
- **`## Limitations`** section (a real heading, not run-in bold), ranked by
  how far they constrain the claims.
- **Possible next paths**, each tied to a question and (if tracked) a ticket.
- **`## Attribution`** (a real heading) — separate human direction-setting
  from AI implementation honestly, and keep it basic: dated quoted turns
  plus the split below, no label taxonomy or per-quote provenance tags (see
  `CLAUDE.md` § Showcase vs history). The
  [subliminal arc](arcs/02_subliminal/README.md#attribution) § Attribution
  is the reference implementation of the required content (written with
  these requirements in `5040118e`, 2026-07-18); the
  [J-space arc](arcs/04_jspace/README.md) § Attribution shows the fuller
  multi-block form. Required shape:
  1. **Research direction** (the opening block under `## Attribution`): the
     originating user direction(s) as
     **verbatim quotes**, each date-tagged `[session YYYY-MM-DD]`, with a
     one-line "quotes lightly normalized for typos/punctuation" disclaimer if
     you touched them. Paraphrases, and checkpoint-menu options the user
     *selected* rather than typed, are labelled as such in prose and **never**
     wrapped in quotation marks as if typed. If an intuition is the user's but
     the README's wording is your paraphrase, say so and leave it un-quoted.
  2. A three-way **collaboration split**: what the **human** contributed
     (direction, interpretive judgment, scope) / what **Claude** contributed
     (implementation, scaffolding, literature) / what **emerged** (neither
     party alone).
  3. Every quote must be **verifiable against the session transcript** —
     recover the originating turn before quoting it; if you can only
     paraphrase, mark it a paraphrase and point at the transcript.
- Cross-link: README → observations → figures/INVENTORY → data/MANIFEST.

**Done when:** the figures meet their done line above, every observation file
has all its fields filled, and the arc README carries the canonical sections
in order with its headline figures embedded.

**Closed by:** a PR plus the review loop in
[§ The checkpoint PR](#the-checkpoint-pr).

### The checkpoint PR

Every checkpoint, and every re-entry of one, closes with this PR and review
loop.

- Push the branch, open a PR. **One PR = one scope, small enough to review
  in one sitting** — each checkpoint is its own PR, and a checkpoint too
  large for one sitting splits into staged PRs rather than one mono-diff.
  `git lfs pull` works for reviewers. Review with the owner's PR-review
  swarm (`claudectl review-pr <PR#> --apply`, private tooling) or
  `/code-review high`; fix in-scope findings, file out-of-scope findings as
  issues, re-run on the new head until a pass is clean. Merge per the repo's
  manual-merge SOP.
- **Verify the content actually landed — don't trust the merged badge.**
  After merge run `git branch -r --no-merged origin/main` (should be empty)
  and `git merge-base --is-ancestor <merge-sha> origin/main` for each PR in
  a stack — a GitHub "merged" badge only means the PR merged into *its
  base*, which may be a stale intermediate branch (the arc-03 4-PR stack
  stranded 22 commits this way; repaired in `2400a95f`). Merged branches are
  the permanent record of merge points and are never deleted, so stacks are
  merged top-down (or each PR retargeted with `gh pr edit N --base main`).

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
`research/**/data/*.pt` rule. Keep your working/scratch captures in a
gitignored cache; the committed `data/` dir is the canonical copy.

**Wiring (so the data is *usable*, not just stored).** Scripts should resolve
inputs **cache-first, committed-copy-fallback**, and write outputs only to the
cache. The NLA arc centralizes this in `examples/_nla_artifacts.py`
(`read_artifact`/`find_artifact` for loads, `write_artifact` for saves). That
one indirection is what lets the *same* script a developer runs locally
(writing fresh captures to the gitignored cache) also re-render figures and
replay the audit on a clean clone (reading the committed copy) — no manual copy
step, no clone-vs-local branching. Without it, committed data is inert: the
scripts still point at an empty cache.

**Manifest.** A `data/MANIFEST.json` (template generator:
`nla_data_manifest.py`) records per file: `filename`, `sha256`, `size_bytes`,
`class` (capture-root | derived), `producing_script`, `producing_command`,
`inputs` (upstream `.pt`), `requires_model` (none | base | +av/+ar/…),
`consumers` (figures / downstream artifacts / audit). The generator's `--check`
mode re-verifies every sha256 — run it in the audit step as a drift detector.

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
- **No emojis** (use `★ → • ─` or bold); **no placeholders** in any committed
  doc. (Repo `CLAUDE.md` + user memory.)

---

## Sessions are not findings

`sessions/` files are operational checkpoints (worktree path, branch tip,
"what's next") that go stale within hours. They are never load-bearing for a
claim. Don't rewrite old session snapshots to match current state — newer files
supersede older ones, and the README/INVENTORY/audit carry the durable record.

---

## New-arc checklist

```
Checkpoint 1: question, research, plan
[ ] worktree created; research question written in one sentence
[ ] plan in plans/: question, predictions, data vetting, dependencies,
        checkpoint placement (length scales with the arc; every arc has one)
[ ] checkpoint PR opened; review loop reached its floor
Checkpoint 2: setup and implementation
[ ] capture + analysis scripts written; manifest generator + audit scaffold
[ ] dry run at tiny n completes (outputs not committed); pyright clean
[ ] checkpoint PR merged before the long compute run starts
Checkpoint 3: computation, processing, validation
[ ] capture run; protocol validated (layer/position/shapes/counts sane)
[ ] raw data saved to arcs/<slug>/data/ ; MANIFEST.json written; --check passes
[ ] derived artifacts scripted + in data/ + in manifest (class: derived)
[ ] audit script re-derives every load-bearing number incl. the headline;
        passes from a clean clone
[ ] checkpoint PR opened; review loop reached its floor
Checkpoint 4: observations, conclusions, artifacts
[ ] figures generated by committed scripts; INVENTORY.md bijection complete
[ ] observations written (evidence-first, fields filled, nulls labeled as null)
[ ] arc README: canonical sections, in order; headline figures embedded;
        "what the audit can't catch" stated
[ ] clean-clone test: git lfs pull → audit PASS → a figure re-renders
[ ] checkpoint PR opened; review loop reached its floor
Every re-entry of checkpoints 2-4: a new PR scoped to the delta
```
