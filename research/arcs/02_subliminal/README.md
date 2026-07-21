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

## Contents

- [`plans/2026-05-31-subliminal-semantic-transfer.md`](plans/2026-05-31-subliminal-semantic-transfer.md)
  — the full scoped plan: verified paper setup, hypotheses, per-step
  estimators, methodological caveats, feasibility, novelty, references.
- `observations/` — created once Step 0 produces findings.

Toolkit note: the CAV / activation-probe machinery and the NLA verbalizer
from the [NLA verbalizer arc](../nla-verbalizer/README.md) are reusable here.
