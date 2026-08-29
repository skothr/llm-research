# Data licensing and attribution — arc 01 artifacts

**The repository's `GPL-3.0-only` licence covers source code and original
prose only.** The `.pt` artifacts in this directory are model-derived — their
contents are activations and generated text produced by third-party model
checkpoints — so this file records what they depend on and under what terms.

Unlike [arc 04](../../04_jspace/data/LICENSE-DATA.md), **no third-party corpus
was ingested**. Every prompt and every anchor token in this arc was
hand-authored in-repo, inside the `examples/nla_*.py` capture scripts. There is
no scraped-web input, and therefore no scraped-web PII to redact.

## Source models

### Qwen2.5-7B-Instruct (the base model)

All `h[20]` hidden states in this directory come from
`Qwen/Qwen2.5-7B-Instruct`.

- **Licence:** Apache-2.0 —
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- **Attribution obligation:** Apache-2.0 §4 requires retaining the licence,
  copyright and NOTICE text when redistributing the work or derivative works.
  This file, plus the model id recorded in `MANIFEST.json`'s `model_pin`
  block, is the attribution for the activations here. The model weights
  themselves are **not** redistributed by this repo.
- **Revision:** not pinned. No HF commit revision was recorded at capture time
  (2026-05-12..15) — see the `model_pin.note` in
  [`MANIFEST.json`](MANIFEST.json). Reproducibility is repo-level, not
  commit-level.

### `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar` (the NLA pair)

The verbalizer (AV, h → text) and reconstructor (AR, text → h) that produced
every `av_text` string and every `h_pred` tensor in this directory.

- **Licence: not recorded at capture time.** Nothing in this arc's scripts,
  observations, or commit history captured what the two model cards state.
  **Verify the upstream model cards before any further redistribution of the
  derived artifacts in this directory** —
  <https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av> and
  <https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar>. This is a gap in the
  record, stated as such rather than guessed at.
- **Revision:** not pinned, same as the base model.
- The checkpoints themselves are not redistributed by this repo.

## What the committed artifacts contain

- **Tensors:** layer-20 hidden states (`h`, d=3584), AR reconstructions
  (`h_pred`), and derived products (cosines, centroids, mean-contrast
  directions, PCA output).
- **Text:** AV verbalizations — **model-generated output** conditioned on the
  hidden states above — and the hand-authored prompt strings that produced
  them. `observations/2026-05-13-nla-walkthrough-all-captures.txt` is a
  plain-text dump of the same AV text.

Model-generated text is not covered by the repo's GPL-3.0-only licence any
more than the model weights are; its status follows from the source models'
terms, which for the NLA pair are the unresolved item above.

## Personal data

**No third-party personal data was ingested.** The prompts and anchors are
hand-authored; the scanner rationale that applies to web-scraped corpora
(`examples/jspace_redact_corpus.py --report`, per the repo `CLAUDE.md`
§ "Third-party data") does not apply, because there is no scraped input to
scan.

Two qualifications, stated rather than glossed:

1. **Public figures appear in hand-authored prompts** as ordinary factual
   subject matter — `"Mozart was born in Salzburg."`, `"The band Queen is from
   England."` in `examples/nla_country_concept_vector.py`. These are published
   biographical facts about public figures used as country-mention probes, not
   personal data collected about private individuals.
2. **AV output is model-generated and was not filtered.** The verbalizer
   occasionally emits proper nouns that were not in the prompt (the
   concept-arithmetic observation records one such decode naming a film and an
   actor). These are model confabulations conditioned on a synthetic h vector,
   not records about a person. If any such string is nonetheless objectionable,
   use the reporting path below.

## Redistribution decision

The `.pt` artifacts **are committed** (git-LFS), per the repo's "raw data is a
deliverable" rule: they cost CPU-hours to regenerate, the capture scripts need
model weights this repo cannot ship, and without them no figure re-renders and
the audit has nothing to check. Committing them republishes model-derived
tensors and model-generated text — not any third party's corpus and not any
third party's personal data.

The one open obligation is the NLA pair's licence, above. It is recorded here
as unresolved so that a future redistributor (or a downstream user of these
artifacts) sees the gap rather than assuming it was checked.

## Reporting content

If you are named in, or otherwise identified by, any text in this directory
and want it removed, open an issue at
<https://github.com/skothr/llm-research/issues> or contact the maintainer via
the address on the repository owner's GitHub profile. Removal requests will be
honoured — no research purpose here depends on any particular generated
string.
