# Subliminal trait transfer — is the hidden signal *model-specific semantics*?

**Status:** paused (recorded 2026-06-10) — Step 0 (encoding decode-test)
complete, see
[`observations/2026-05-31-step0-protocol-and-filter.md`](observations/2026-05-31-step0-protocol-and-filter.md);
Steps 1-2 not started. Work stopped 2026-06-01, paused where it stood
[session 2026-06-01] for a repo-wide reorganization, and was recorded as
paused on 2026-06-10, when the embedding-atlas arc (03) opened and superseded
it (03 closed 2026-07-15); not resumed — Step 1 is additionally blocked on the
design issue below.

**Started:** 2026-05-31.

**Compute context** (transcript-verified, [session 2026-05-31 → 06-01]): the
Step-0 capture ran on CPU after the 4-bit GPU load hit a CUDA out-of-memory
error against a co-tenant process (the dataset manifest records
`device: cpu`), which capped the run at n=120 per condition — the committed
dataset itself completed and audits byte-for-byte. A follow-on scale-up
generation for the Step-1 SFT corpus (n=12,000 per condition) stalled on GPU
and was killed without producing an artifact; no data was lost or truncated.

## The question

Cloud et al., *Subliminal Learning* (arXiv:2507.14805), showed a teacher LLM
with a trait (e.g. an owl-loving persona) can transmit that trait to a
*same-base* student through nothing but **filtered number sequences** — and
that transmission **fails across model families**. They concluded the signal
is "model-specific ... therefore not explained by general semantic content."

That refutes *universal* semantics but slides from "not **general** semantic"
to "not semantic." This arc tests the third option they never separated:

- **(A)** non-semantic, model-specific statistical patterns — no trait meaning.
- **(B)** semantic-universal — ruled out by cross-family failure. *[settled]*
- **(C)** semantic, but in the model's **own learned coordinates** — dies
  cross-family because the coordinate system differs, yet is genuinely
  trait-organized. **Untested.**

Working intuition (user): the trait is a **resultant** — no single number
carries "owl," but the primed persona shifts the *distribution* over emitted
numbers, and the superposition of many tiny per-number nudges accumulates,
over training, into the preference.

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Shape, provenance labels and public-repo constraints per
[`ARC_PROCESS.md` § Attribution](../../ARC_PROCESS.md#attribution--who-directed-who-executed).
Quotes are the human's typed turns, taken from the session listed under
**Verifiability**; `[NORMALIZED]` means typo and punctuation fixes with
markdown emphasis inside the turn dropped, and `[...]` marks an editorial
elision — elisions never remove the direction the quote is cited for. This arc
was opened in one working session by Michael Lannum.

### Research direction

**Originating question** [session 2026-05-31] `[NORMALIZED]`:

> *"another paper... is about model-assisted training 'unconsciously'
> transferring knowledge/traits, like a preference for owls, despite the
> output being filtered... I wonder if output that contains embeddings that
> are cosine similar to owl-related things... could over training shape into a
> preference."*

That question — is the filtered signal semantic in the model's *own*
coordinates? — is hypothesis (C) under [The question](#the-question). The
three-way (A)/(B)/(C) contrast that frames it was Claude's articulation, not
the user's.

**The working intuition is the user's**, stated in the first person [session
2026-05-31] `[NORMALIZED]`:

> *"you would have to keep track of the influence on learning of many
> non-owl-related concepts... to check if their superposition would result in
> the learned preference in effect. Though there might be a way to focus the
> checking mathematically."*

`[PARAPHRASE]` The "trait is a **resultant** ... superposition of many tiny
per-number nudges" wording above is a Claude paraphrase of this — kept
un-quoted for that reason. The user typed "superposition" and "focus the
checking mathematically"; "resultant" and "per-number nudges" are Claude's.

**The pause** [session 2026-06-01] `[VERBATIM]`:

> *"can we actually pause this arc where we are, make sure our stopping point
> is well-documented [...]"*

The elision drops the repo-wide reorganization that the same turn moved on to,
which is not what the quote is cited for. This is where the arc's stopping
point became a deliverable rather than a side effect: Step 0's results, the
unresolved Step-1 design question, and a resume checkpoint were written up and
committed before attention moved off the arc — which is why a pause taken on
2026-06-01 is still legible here.

### Human / Claude / emergent split

**User (Michael Lannum).** The originating question and hypothesis (C); the
superposition intuition and the "focus it mathematically" push toward an
aggregate estimator; the call to pause at a well-documented stopping point;
the call to land the research-directory reorg as its own PR first, deferring
subliminal to a separate later PR.

**Claude Code.** Verifying the paper against arXiv:2507.14805 (the
"imbue"-persona wording; the same-base requirement); the (A)/(B)/(C) framing
and its grounding in the representational-alignment literature; the staged
cheap-first program; the Step-0 decode-test implementation and NULL
adjudication; the Step-1 design and adversarial critique.

**Emergent.** The influence-alignment estimator — the user's "focus it
mathematically" push, formalized by Claude.

*Split as of 2026-06-10, covering Step 0.*

### Verifiability

Every quote above is recoverable from the session below. The transcript is
machine-local and is not committed to this repo — it carries local paths and
tool output — so it is referenced by session id and date only. The ids are
**abbreviated**: each is the 8-character prefix of the local session UUID.

| Session | Span | Covers |
|---|---|---|
| `eda85977` | 2026-05-31 → 06-01 | the arc's working session and its pause (ran pre-repo-split under the predecessor project) |

Claims in this section that are not in quotation marks are Claude's
characterization of the user's direction, not the user's wording.

## Program (cheap-first, staged)

0. **Encoding decode-test** — ASCII / base-N on a **local reproduction** of
   their protocol: they released the generation pipeline, not the number data,
   and their teacher (`gpt-4.1-nano`) is closed, so the streams are regenerated
   with an open same-base teacher under their ported prompts + filter. Zero
   GPU training. Falsifies the literal-encoding hypothesis they didn't check.
1. **Differential influence-alignment probe** (TinyLlama, LoRA) — accumulated
   `⟨∇P_trait, −∇L_i⟩` for trait-teacher vs **neutral**-teacher data, trait
   axis anchored *behaviorally* (not a CAV). Aggregate-level.
2. **Cross-base recovery under representational alignment** — the decisive
   HA-vs-HC test: HC predicts cross-family transmission partially recovers
   after alignment; HA predicts it stays dead.

**Discipline:** this is a mechanism question, **not** a safety/alarmist
framing. Pre-commit to reading ambiguous probe results as "this measurement
can't separate A from C," not as support for C.

**Step-1 design status (2026-05-31, unresolved).** The adversarial critique in
[`plans/2026-05-31-step1-design.md`](plans/2026-05-31-step1-design.md) found
that the design's core statistic Δ is predicted **in the same direction by both
HA and HC** — the behavioral trait axis ĝ is mechanically downstream of the
same distributional shift the trait-teacher gradients induce, so a clean Δ>0
with every guard green is uninterpretable. Step 1 as specified therefore cannot
separate the hypotheses without the added statistical-fingerprint axis ĝ_stat
and the trait alignment that survives projecting it out (MUST-FIX 1 — the only
contrast in the design's reach that HA and HC predict *oppositely*). Absent
that axis, Step 1 honestly tests **trait-specificity of teacher-pull, not
semanticity**, and the HA/HC decision falls to Step 2. The design was never
signed off; the arc paused here.

## Findings

**H0 (a literal, decodable channel) — NULL.** Across all five decode schemes
(`ascii_direct`, `ascii_mod256`, `ascii_mod128`, `concat_digits3`,
`concat_digits2_off97`) and both conditions there are **zero** owl-lexicon
hits: owl_rate = neutral_rate = 0.000 everywhere. `decode_report.json` records
`z = 0, p = 1.0` in each of those cells, but that pair is a recording
convention for the all-zero case, **not a computed statistic** — with zero
variance in both arms the pooled two-proportion standard error is 0 and the
test is undefined, so no p-value is being asserted here. What carries the
result is the zero-hit count itself (0 of 104 owl streams and 0 of 109 neutral
streams, under each of the five schemes) together with the sample size behind
it: n = 120/condition settles an all-or-nothing literal channel, and is far
short of the ~931/condition that resolving a small rate *difference* would
need — that floor is the one derived for the incidental 13.3-vs-9.2%
reject-rate gap, not for the decode null (limitation 4 below). Kept after the
ported format+range+count filter: owl 104/120, neutral 109/120. The
positive control passed in the same run (a planted `[111,119,108]` decodes to
"owl" and trips the lexicon), so the null is a real absence, not a broken
decoder. Paired with the source finding that the released filter has **no**
encoding guard — a literal channel would have passed untouched — this closes
H0 and hands off to the distributed-shift hypotheses (HA vs HC).

**Scope.** This is a local Qwen2.5-7B-Instruct reproduction of the paper's
protocol, not the paper's own data (never released; their teacher is closed).
It closes the literal-channel hypothesis *for this setup*; it cannot prove the
paper's streams carry no channel — though their cross-family transmission
failure already argues against any universal decodable encoding.

**Incidental, under-powered — do not over-read.** The owl reject rate (13.3%,
16/120) ran higher than neutral (9.2%, 11/120) on the identical seeded prompt
set. The gap is not significant at n=120 (two-proportion z = 1.02, p = 0.31);
resolving a real 13.3-vs-9.2% difference at 80% power needs ~931 per condition.
At most a hint that the persona shifts the output distribution — deferred to
any scale-up.

Full write-up:
[`observations/2026-05-31-step0-protocol-and-filter.md`](observations/2026-05-31-step0-protocol-and-filter.md).

**Audit.** Every load-bearing number above is re-derived from the committed
bytes by [`examples/subliminal_audit_findings.py`](../../../examples/subliminal_audit_findings.py)
— file hashes/sizes/line counts against the manifest, a from-first-principles
replay of the ported filter over the raw completions, a full five-scheme
decode replay reconciled against `decode_report.json`, and a byte-exact
regeneration of the 120-query prompt set from the seeded generator. No GPU, no
network, under a second:

```bash
python examples/subliminal_audit_findings.py
```

Expected result as of 2026-08-17 (committed run:
[`data/audit_2026-08-17.log`](data/audit_2026-08-17.log)):

```
SUMMARY:  103 PASS  |  0 FAIL  |  5 UNVERIFIABLE
```

One claim is environment-dependent: resolving the manifest's
`generator_git_commit` needs the repo's history. From a shallow clone, a source
copy without `.git`, or a box without `git`, that check moves to the
UNVERIFIABLE list instead of failing — 102 PASS | 0 FAIL | 6 UNVERIFIABLE is
the same green result under those conditions.

**What a 103-PASS audit means.** It establishes *arithmetic and artifact
consistency*: the numbers in this README and in the observation are the numbers
the committed bytes yield under the ported filter and the decoder. It does
**not** establish the scientific validity of the protocol — a wrong chat
template or a teacher that ignored the persona would produce
consistent-but-wrong data, since the prose and the re-derivation flow from the
same completions — nor does it say anything about how far the null transfers
beyond this local-Qwen setup. The 5 UNVERIFIABLE entries name what no artifact
here can settle.

## Limitations

Ranked, most load-bearing first:

1. **The teacher is not the paper's teacher.** Streams come from a local
   Qwen2.5-7B-Instruct, because the paper's `gpt-4.1-nano-2025-04-14` is closed
   and its datasets were never released. Absolute rates (including the 13.3% /
   9.2% reject rates, which sit below the paper's reported 23–38% band) are not
   comparable across teachers.
2. **Five decode schemes, not an exhaustive space.** ASCII direct/mod-256/
   mod-128, 3-digit concatenation, and 2-digit-offset concatenation. A channel
   under some scheme not in that set would be missed.
3. **The owl lexicon is finite** — 24 hand-picked substrings. A channel
   spelling owl-adjacent words outside that list would be missed. (The lexicon
   is deliberately generous: false positives hit both conditions equally, and
   the test reads the *differential*.)
4. **n = 120 per condition.** Enough to establish the decode null (a literal
   channel is all-or-nothing), far short of the ~931 needed to resolve the
   incidental reject-rate gap.
5. **`prompts.jsonl` was re-derived post-hoc, not captured.** The 2026-05-31 run
   predates the generator change that writes the prompt set, so the committed
   file is a deterministic seed-42 replay (2026-08-17) rather than a capture.
   See `data/README.md` § "`prompts.jsonl` (re-derived post-hoc, 2026-08-17)".
6. **Step 1 is blocked at design.** See "Step-1 design status" above — the
   specified statistic cannot separate HA from HC.

## Possible next paths

- **Step 1, unblocked** — build the statistical-fingerprint axis ĝ_stat and
  report the ĝ_trait alignment that survives projecting it out (MUST-FIX 1),
  together with the two companion fixes: define ĝ on a trait-free reference
  model (never on the trait-student, which makes Δ>0 near-tautological), and
  make the seed the unit of analysis with ≥5 seeds, multi-checkpoint TracInCP
  and a fingerprint positive control. Without ĝ_stat, Step 1 should be
  relabelled "trait-specificity of teacher-pull" and read as such.
- **Skip to Step 2** — cross-base recovery under representational alignment is
  the decisive HA-vs-HC test on its own terms (HC: transmission partially
  recovers after alignment; HA: it stays dead), and it does not depend on Step
  1's contested statistic. The cost is that it is the expensive step, so the
  cheap-first ordering is sacrificed.
- Either path first needs a **training-scale corpus**: the committed dataset is
  decode-test scale (120/condition), not the ~10k-kept the training design
  assumes.

## Contents

- [`plans/2026-05-31-subliminal-semantic-transfer.md`](plans/2026-05-31-subliminal-semantic-transfer.md)
  — the full scoped plan: verified paper setup, hypotheses, per-step
  estimators, methodological caveats, feasibility, novelty, references.
- [`plans/2026-05-31-step1-design.md`](plans/2026-05-31-step1-design.md) — the
  five-agent Step-1 design workflow output (QLoRA feasibility, influence
  estimator, eval protocol, synthesized spec, adversarial critique), preserved
  verbatim. **Not signed off** — blocked on MUST-FIX 1, above.
- [`observations/2026-05-31-step0-protocol-and-filter.md`](observations/2026-05-31-step0-protocol-and-filter.md)
  — Step 0: what the upstream source settles about the protocol and filter
  (no encoding guard), plus the NULL decode-test result.
- [`data/`](data/README.md) — the committed `step0-owl-neutral-decode` dataset
  (streams, raw completions, decode report, prompt set, provenance manifest,
  env lockfile), its audit log, and
  [`data/LICENSE-DATA.md`](data/LICENSE-DATA.md) — the licence, attribution and
  personal-data record.

Toolkit note: the CAV / activation-probe machinery and the NLA verbalizer
from the [NLA verbalizer arc](../01_nla-verbalizer/README.md) are reusable here.
