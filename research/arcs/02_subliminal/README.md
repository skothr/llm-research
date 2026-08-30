# Subliminal trait transfer — is the hidden signal *model-specific semantics*?

**Status:** paused (2026-06-10) — Step 0 (encoding decode-test) complete, see
[`observations/2026-05-31-step0-protocol-and-filter.md`](observations/2026-05-31-step0-protocol-and-filter.md);
Steps 1-2 not started. Paused in favor of the embedding-atlas arc.
**Started:** 2026-05-31.

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

## Research direction

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. This arc was opened in one working session by Michael Lannum; quotes are
from that session's transcript, lightly normalized for typos/punctuation.

**Originating question** [session 2026-05-31]:

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
2026-05-31]:

> *"you would have to keep track of the influence on learning of many
> non-owl-related concepts... to check if their superposition would result in
> the learned preference in effect. Though there might be a way to focus the
> checking mathematically."*

The "trait is a **resultant** ... superposition of many tiny per-number nudges"
wording above is a Claude paraphrase of this — kept un-quoted for that reason.
The user typed "superposition" and "focus the checking mathematically";
"resultant" and "per-number nudges" are Claude's.

**Human / agent split (Step-0 state).** *User:* the originating question and
hypothesis (C); the superposition intuition and the "focus it mathematically"
push toward an aggregate estimator; the call to land the research-directory
reorg as its own PR first, deferring subliminal to a separate later PR.
*Claude:* verifying the paper against arXiv:2507.14805 (the "imbue"-persona
wording; the same-base requirement); the (A)/(B)/(C) framing and its grounding
in the representational-alignment literature; the staged cheap-first program;
the Step-0 decode-test implementation and NULL adjudication; the Step-1 design
and adversarial critique. *Emergent:* the influence-alignment estimator — the
user's "focus it mathematically" push, formalized by Claude.

## Program (cheap-first, staged)

0. **Encoding decode-test** — ASCII / base-N on their *released* number data;
   zero GPU. Falsifies the literal-encoding hypothesis they didn't check.
1. **Differential influence-alignment probe** (TinyLlama, LoRA) — accumulated
   `⟨∇P_trait, −∇L_i⟩` for trait-teacher vs **neutral**-teacher data, trait
   axis anchored *behaviorally* (not a CAV). Aggregate-level.
2. **Cross-base recovery under representational alignment** — the decisive
   HA-vs-HC test: HC predicts cross-family transmission partially recovers
   after alignment; HA predicts it stays dead.

**Discipline:** this is a mechanism question, **not** a safety/alarmist
framing. Pre-commit to reading ambiguous probe results as "this measurement
can't separate A from C," not as support for C.

## Limitations

Ranked by how far each limit constrains what this arc can claim.

**L1. The arc stopped after Step 0 — (A) vs (C) is untested.** What Step 0
establishes: the paper's released filter has no semantic and no encoding guard
(format + range + count only, `banned_numbers=[]` for the animal-preference
config), and a five-scheme decode of locally regenerated owl/neutral streams
returns zero owl-lexicon hits in either condition (owl_rate = neutral_rate =
0.000, z = 0, p = 1.0), with the planted-string positive control passing — so
H0, a literal decodable channel, is not supported *for this setup*. What it does
not establish is anything about (A) versus (C). Steps 1 and 2 — the differential
influence-alignment probe and the decisive cross-base recovery test — were
designed but never run; the arc has been paused since 2026-06-10.

**L2. The Step-1 design is not yet known to be able to separate (A) from (C).**
Its own adversarial critique
([`plans/2026-05-31-step1-design.md`](plans/2026-05-31-step1-design.md)
§ "Adversarial critique") rates the central statistic a blocker: the
behaviourally-anchored trait axis is mechanically downstream of the same
distributional shift the trait-teacher gradients induce, so Δ > 0 is predicted
under *both* hypotheses with every confound guard green. Open alongside it — a
circular axis if the trait direction is computed on the owl-student rather than
a trait-free reference; between-seed variance unestimable at 3 seeds; and no
power floor operationalizing the pre-committed "a null supports (A) only if not
underpowered". Running Step 1 as specified could yield a confident but
uninterpretable result.

**L3. Step 0 tests a stand-in, not the paper's own data.** Cloud et al. never
released their number datasets and their teacher (`gpt-4.1-nano-2025-04-14`) is
closed, so the streams here were regenerated with an open same-base teacher
(`Qwen2.5-7B-Instruct`) using the ported prompts and filter. The null therefore
closes the literal-channel hypothesis for this local setup only; it cannot show
the paper's own streams carry no channel. Scope is also narrow in every other
axis: one teacher model, one trait (owl), one seed, five decode schemes (not
exhaustive), and a finite owl lexicon.

**L4. n = 120 per condition — the one suggestive signal is under-powered.** The
owl condition's filter-reject rate ran higher than neutral's (13.3% vs 9.2% on
the identical seeded prompt set), but the gap is not significant (two-proportion
z = 1.02, p = 0.31); resolving a real 13.3-vs-9.2 gap at 80% power needs
~930 per condition. It is a hint that the persona shifts the output
distribution, not evidence of one — see
[`observations/2026-05-31-step0-protocol-and-filter.md`](observations/2026-05-31-step0-protocol-and-filter.md).

**L5. The dataset is statistically, not byte-, reproducible, and part of its
provenance is inferred.** Sampling at temperature 1.0 makes the corpus
`statistical_only`: the committed files plus their sha256 are the anchor, not a
re-run. `prompts.jsonl` was re-derived post-hoc (2026-08-17) by replaying the
seeded generator — the audit proves it *is* that generator's seed-42 output, but
that it is the set the 2026-05-31 run consumed remains an inference. Two
capture-time hashes in `manifest.json` (the pip-freeze and generator-script
hashes) no longer match disk, for reasons recorded in
[`data/README.md`](data/README.md) § "Post-capture amendments"; the model
snapshot revision and the capture-time environment are likewise unverifiable
from committed bytes. `examples/subliminal_audit_findings.py` re-derives every
load-bearing number and reports these as five UNVERIFIABLE entries beside
`104 PASS | 0 FAIL`.

**L6. The data layout predates the arc-data SOP.** This arc keeps a per-dataset
`manifest.json` (`manifest_version 0.1.0-interim`) rather than
`ARC_PROCESS.md`'s `data/MANIFEST.json` convention, and was never migrated.
Integrity is fully covered by the audit script; the migration is a field remap,
not a re-capture, tracked as issue
[#53](https://github.com/skothr/llm-research/issues/53).

## Contents

- [`plans/2026-05-31-subliminal-semantic-transfer.md`](plans/2026-05-31-subliminal-semantic-transfer.md)
  — the full scoped plan: verified paper setup, hypotheses, per-step
  estimators, methodological caveats, feasibility, novelty, references.
- `observations/` — created once Step 0 produces findings.

Toolkit note: the CAV / activation-probe machinery and the NLA verbalizer
from the [NLA verbalizer arc](../nla-verbalizer/README.md) are reusable here.
