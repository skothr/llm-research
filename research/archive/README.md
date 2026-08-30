# Research archive

Retired and pre-arc material, kept for archaeology. Nothing here is
load-bearing for an active research claim; files are preserved because they
record early exploration, not because they're current.

## Contents

Three standalone interpretability probes (TinyLlama 1.1B, OpenLLaMA 3B v2)
from 2026-04, predating the NLA verbalizer arc. They were one-off findings
rather than a coherent investigation, so they were never promoted to an arc:

- `2026-04-09-attention-ablation-possessive-collapse.md` — zeroing attention
  sub-layers (OpenLLaMA 3B) and observing possessive-construction collapse.
- `2026-04-09-layer19-factual-vs-linguistic-coherence.md` — layer-19 split
  between factual and linguistic coherence.
- `2026-04-11-logit-lens-tinyllama-initial.md` — initial logit-lens pass on
  TinyLlama.

These follow the standard observation format (see `../README.md` §
Conventions) and remain reproducible, but are not maintained.

## Attribution

Direction-setting (the human role) and implementation (the AI role) are
different kinds of work; separating them keeps visible where the ideas came
from. Quotes below are verbatim user turns (Michael Lannum) from the session
transcripts of 2026-04-08 → 2026-04-12 in the pre-split `llm` workspace —
the sessions in which these three probes were run — lightly normalized for
typos and punctuation; markdown emphasis inside a turn is dropped, `[...]`
marks an editorial elision, and dates follow the transcripts' UTC clock.
The shape follows
[`ARC_PROCESS.md` § 6](../ARC_PROCESS.md#6-arc-readme-synthesis), applied
retroactively: these probes predate every arc and the process document
itself.

**The layer-surgery program** [session 2026-04-08]. The toolkit the probes
run on (`llm_surgeon`) was built to the user's capability spec:

> *"I just want to be able to rewire the model in different ways (layers,
> embeddings, tokens, and stuff [...] that means we can splice together
> different models to be run in ollama"* — [session 2026-04-08]

> *"It might take some iteration, but I want to be able to:
> 1\. Specify the changes with a python script (using the python toolkit),
> i.e. add/remove/change_order_of layers [...]
> 2\. Confirm the desired change has been successfully integrated.
> 3\. Test it out with ollama, beside original or some benchmark."*
> — [session 2026-04-08]

The specify → confirm → test-beside-baseline loop is the method all three
probes follow. Both model changes were the user's calls, with the specific
picks Claude's suggestions: TinyLlama-1.1B as an availability pivot after
Llama-3 access gating ("I applied and I'm waiting for access to be granted.
Can we try using a more freely accessible model?" — [session 2026-04-09]),
and OpenLLaMA 3B as a scale ask ("Can we try with a larger model? What
options are there?" — [session 2026-04-09]).

**The layer-19 probe** (`2026-04-09-layer19-factual-vs-linguistic-coherence`)
[session 2026-04-09]. The generation test, the finding, and the
observation's hypothesis 5 are all first stated in user turns:

> *"It's interesting that layer 19 was the 'safest' to remove, while being
> one of the final ones in the sequence. Can we try testing that
> frankenmodel (layer 19 removed) with an ollama prompt like last time?"*

> *"Very interesting that the language construction itself is almost as
> coherent as the original [...] I looked a number of them up to check,
> e.g.: 'parisum' in Latin means being born (not "pier-town"), can't find
> anything called 'Hôtel du Pregarden' anywhere [...] So it seems like
> factual coherence was greatly affected by removal of layer 19, along with
> maybe how it decides where to stop [...] So perplexity (or at the very
> least BI scores) seems to measure only the linguistic coherence."*

> *"the fact that the effect was significant despite lower BI score means
> the 'job' layer 19 has been trained to perform likely involves scaling
> vector magnitudes [...] The no19 model seemed to know that "some sort of
> name should go here", given it still chose tokens that started with
> capital letters, were French-sounding, and that sounded almost right"*

The third turn is the magnitude-based factual-sharpening idea, written into
the observation nearly verbatim as hypothesis 5; the angle-vs-magnitude
framing returned as a tooling ask later that day ("Considering layer BI
scores (cosine_similarity, measures angle difference between input and
output vectors), is there a counterpart that takes magnitude into
account?").

**The attention-ablation probe**
(`2026-04-09-attention-ablation-possessive-collapse`) [sessions 2026-04-09
→ 2026-04-10]. The ranked-ablation design was the user's ask ("Can we try
surgically removing some of the lowest-contributing components, and export
the model to test vs baseline?" — [session 2026-04-09]), as were the
capability and the escalation. The collapse reading itself was joint: the
anomalous tokens first surfaced in Claude's report of the 6-zeroed run,
the user widened the pattern and added the register reading of "ya'", and
the possession framing settled in the exchange the last two quotes carry:

> *"Okay, I think I want to implement surgical tools for the attention
> heads now."* — [session 2026-04-09]

> *"Let's try more layers. Keep removing the least likely impactful
> ones."* — [session 2026-04-10]

> *"Yeah also "they's", "we's", so something about pluralism got ablated?
> There's also "ya'" -- it seems like that could be an artifact from the
> context already including "we's" (maybe because "you's" or "youse" is
> common in some slang/accents)"* — [session 2026-04-10]

> *"Yeah (and yeah I meant possession not pluralism)"*
> — [session 2026-04-10]

**The logit-lens probe** (`2026-04-11-logit-lens-tinyllama-initial`)
[sessions 2026-04-11 → 2026-04-12]. The capability is the user's design
ask — an intermediate output projection at every layer junction:

> *"I want to make sure there are ways sample/modify feedforward layers and
> hidden states at each point, maybe the ability to apply an intermediate
> output projection at each layer junction so I can experiment with
> extracting hidden states converted into tokens and get a better sense of
> what the hidden states represent and how they evolve layer to layer."*
> — [session 2026-04-11]

> *"Is '____' a single token (four underscores, not a placeholder)? That
> might be a valid choice to output actually, making it a fill-in-the-blank
> like "The capital of France is ____""* — [session 2026-04-12]

> *"No, let's just note these observations for now. I'll do more extensive
> testing once we have a live visual interface I can play around with."*
> — [session 2026-04-12]

The `____` reading appears as the observation's fill-in-the-blank finding;
the note-now decision is why the file exists in one-off form, and its final
follow-up (build the live interface before extensive testing) restates the
user's sequencing ("Eventually (after we have a robust toolkit) I'll want
to hook this up to a visualization/control GUI." — [session 2026-04-11]).

**The record format** [session 2026-04-09]. The observation files
themselves — and their format — trace to one turn:

> *"it would be smart to record interesting things like this somewhere.
> [...] Always include relevant part of the transcript and reproducibility
> steps, and anything else that should generally be documented for research
> like this"*

### Human / Claude / emergent split

**User (Michael Lannum).** The layer-surgery program and its
specify/confirm/test method; both model pivots; the layer-19 generation
test; the fluent-but-fabricated finding, with hand fact-checking of the
fabrications; the magnitude-scaling hypothesis (observation hypothesis 5)
and the later angle-vs-magnitude tooling ask; the
attention-surgery direction, the ranked-ablation escalation, the widening
of the collapse pattern ("they's", "we's") and the register reading of
"ya'"; the
logit-lens capability ask, the fill-in-the-blank token reading, and the
prompt-variation question behind the attractor-token hypothesis; the
note-now-explore-later call; and the record format (transcript excerpts +
reproducibility steps).

**Claude Code.** All implementation: the `llm_surgeon` toolkit (surgery /
inspect / benchmark / probe), the Block-Influence implementation from
ShortGPT and the perplexity harness, harness execution and ollama
exports (the user ran the exported models' generation tests by hand), the
observation write-ups, and the related-work anchoring.

**Emergent.** The findings neither party predicted: that the lowest-BI
layer sits late in the stack; the possessive/contraction collapse pattern
itself (surfaced by ablation, read jointly in-session); the layer-18/19
crystallization of factual recall that the logit lens showed.

**Verifiability.** Every quote above is recoverable from the `llm`
workspace session transcripts for 2026-04-08 → 2026-04-12; the transcripts
are not committed to this repo (they carry machine-local paths and tool
output). Claims not quoted are Claude's characterization of the user's
direction, not the user's wording.
