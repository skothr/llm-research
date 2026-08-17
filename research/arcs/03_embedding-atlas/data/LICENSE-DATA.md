# Data licensing and attribution — arc 03 artifacts

**The repository's `GPL-3.0-only` licence covers source code and original
prose only. It does not apply to the model-derived tensors in this
directory**, which are derivatives of third-party model weights and remain
subject to those weights' upstream terms as recorded below. The maintainer is
not the rights holder for the underlying weights and cannot relicense them.

## What the upstream artifact is

Every `.pt` file here derives from a single upstream object: the published
weight matrices of **`Qwen/Qwen2.5-7B-Instruct`** — the input-embedding table
`W_E` and the untied output head `lm_head`, each 152,064 x 3,584 in bf16 —
pinned to HF revision
`a09a35458c702b33eeacc393d103063234e8bc28` (the `revision` field of
[`MANIFEST.json`](MANIFEST.json), and of every artifact's own metadata).

- **Licence:** Apache License 2.0 —
  <https://www.apache.org/licenses/LICENSE-2.0>
- **Source:** <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct> (licence
  verified against the model card, 2026-08-17)
- **Attribution / citation:** Qwen Team (2024). *Qwen2.5 Technical Report.*
  <https://arxiv.org/abs/2412.15115>
- **What Apache-2.0 obliges of this redistribution** (§4): keep the licence
  notice and the attribution above with the derived files (this document is
  that notice), state that the files are modified, and do not use the
  licensor's marks beyond identifying the origin. "Qwen" is used here solely
  to name which model the numbers came from.
- **Modification, stated:** every file is a slice, projection, or summary
  statistic cut from those two matrices — none is a redistribution of the
  model, and none is loadable as one.

## Which files carry verbatim weight rows

Three files contain unmodified rows of the upstream matrices; the rest carry
only statistics computed over them. The distinction matters for anyone
reasoning about what is being redistributed:

| File | Size | Content |
|---|---|---|
| `emb_battery_vectors.pt` | ~15.3 MB | Verbatim `W_E` and `lm_head` rows for the 1,062 battery anchor-variant token ids, bf16, unaltered |
| `emb_random_baseline.pt` | ~15.6 MB | Verbatim `W_E`/`lm_head` rows for a seed-pinned (20260610) random sample of 1,000 token ids |
| `emb_de_cosine_check.pt` | ~0.08 MB | Verbatim rows for 11 named candidate tokens, downcast to float16 |
| `emb_fullvocab_stats.pt` | ~45.6 MB | **Statistics only** — per-dimension moments, the k-NN graph (ids + cosines, k=32) and handle scores over 149,706 alive rows; no weight rows |
| `emb_global_stats.pt` | ~3.4 MB | **Statistics only** — per-row norms, the global mean, eigenspectra, the top-50 principal directions |
| all others | — | **Statistics only** — derived from the files above by the committed scripts in `examples/` |

That is 2,062 distinct rows of the 152,064-row table (~1.4%) present
verbatim (2,073 stored rows; 11 token ids appear in two artifacts). The
principal directions in `emb_global_stats.pt` are a rank-50 linear summary of
the centered table, not rows of it.

## Third-party text: none

This arc uses **no external text corpus**, so none of the licence,
attribution, or PII questions that attach to scraped corpora apply here:

- The 690-word probe battery is authored in-repo, as code —
  [`examples/emb_token_battery.py`](../../../../examples/emb_token_battery.py).
- The 51-probe tracing corpus is authored in-repo, as code —
  [`examples/emb_trace_corpus.py`](../../../../examples/emb_trace_corpus.py).

Both are hand-written word lists and short synthetic sentences composed for
this arc. No web-scraped, licensed, or user-contributed text enters the
pipeline at any stage, so no PII scan applies (contrast arc 04, whose C4
slice is web-scraped and was scanned and redacted —
[`research/arcs/04_jspace/data/LICENSE-DATA.md`](../../04_jspace/data/LICENSE-DATA.md)).
The token *strings* stored in these artifacts are tokenizer vocabulary
entries, which are part of the Apache-2.0 model release.

## Redistribution decision

Committing these files was a deliberate call, taken because the arc's
clean-clone bar (`ARC_PROCESS.md` § "Raw data is a deliverable") requires
every figure and every audit row to replay without a model download:

- **Weight rows are committed only where re-deriving them needs the model.**
  The three files above are ~31 MB of a ~15 GB release, are permissively
  licensed, and are not substitutable by a script — regenerating them costs a
  full model load against the pinned revision.
- **The full matrices are NOT committed** (see the deviation note in
  [`README.md`](README.md)). They are bit-reproducible by anyone from the
  public snapshot, so committing 2 x 1.09 GB would buy nothing.
- **Licence scope.** Nothing in this directory is offered under the
  repository's `GPL-3.0-only`. The tensors stay under Apache-2.0 as
  derivatives of the upstream weights; the scripts that produced them, and the
  prose describing them, are the repository's own and are GPL-3.0-only.

Licence questions or takedown requests: open an issue at
<https://github.com/skothr/llm-research/issues>.
