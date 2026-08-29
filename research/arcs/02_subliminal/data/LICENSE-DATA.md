# Data licensing and attribution — arc 02 datasets

**The repository's `GPL-3.0-only` licence covers source code and original
prose only. It does not apply to the datasets in this directory**, which are
model-generated output and remain subject to the upstream terms recorded
below. The maintainer is not the rights holder for the model weights that
produced them and cannot relicense them.

Unlike [arc 04](../../04_jspace/data/LICENSE-DATA.md), **no third-party text
corpus was ingested** — see [Personal data](#personal-data).

## Source model — Qwen2.5-7B-Instruct

Every committed stream and completion in
`step0-owl-neutral-decode/` is output sampled from
**`Qwen/Qwen2.5-7B-Instruct`**, run locally, temperature 1.0, seed 42.

- **Licence:** Apache License 2.0 —
  <https://www.apache.org/licenses/LICENSE-2.0>
- **Source:** <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- **Revision:** `a09a35458c702b33eeacc393d103063234e8bc28`, as recorded in
  [`step0-owl-neutral-decode/manifest.json`](step0-owl-neutral-decode/manifest.json)
  (`generation.model_revision` / `generation.tokenizer_revision`).
- **Attribution / citation:** Qwen Team (2024). *Qwen2.5 Technical Report.*
  <https://arxiv.org/abs/2412.15115>
- **What Apache-2.0 obliges of this redistribution** (§4): retain the licence
  notice and the attribution above alongside the derived files — this document
  is that notice — and do not use the licensor's marks beyond identifying the
  origin. "Qwen" is used here solely to name which model emitted the numbers.
- **Output-use constraints:** none. Apache-2.0 imposes no restriction on the
  use of a model's output, so the synthetic data here is freely usable. This
  is also the standing record in `manifest.json`'s `license.usage_constraints`.
- The model **weights are not redistributed** by this repo.

## Ported upstream code — MinhxLe/subliminal-learning (MIT)

The prompt-generation and filter logic that produced this data is a **verbatim
port** from `github.com/MinhxLe/subliminal-learning @ v1.0.0`
(`sl/datasets/nums_dataset.py`, `cfgs/preference_numbers/cfgs.py`) — the
`PromptGenerator` class, `parse_response` and `get_reject_reasons`. Because
`prompts.jsonl` is a replay of that generator, its **structure and instruction
wording derive from that MIT-licensed source**; the sampled numbers do not.

That repository is **MIT-licensed**, and MIT's sole obligation is to carry the
copyright and permission notice with the copied material. **That obligation is
met in the port's own file**: the full MIT notice, naming the upstream repo and
the `v1.0.0` ref, is in the module docstring of
[`examples/subliminal_step0_decode.py`](../../../../examples/subliminal_step0_decode.py).
It is not restated here — one authoritative copy, next to the ported code.

Copyright line as read from the upstream repository's `LICENSE` via the GitHub
API on 2026-08-17: `Copyright (c) 2025 Minh Le`. (The `v1.0.0` tag itself
carries no `LICENSE` file; the text is at the repository root.)

The paper the pipeline accompanies: Cloud, Le, Chua, Betley, Sztyber-Betley,
Hilton, Marks & Evans, *Subliminal Learning*, arXiv:2507.14805. **Their number
datasets were never released** and their teacher (`gpt-4.1-nano-2025-04-14`)
is closed, so nothing in this directory is derived from their data — only from
their published code, regenerated against a different, open teacher.

## Personal data

**No third-party personal data was ingested, and no PII scan applies.** There
is no scraped input to scan: the arc's entire input side is procedurally
generated. Each of the 120 prompts is a fixed instruction template drawn from
the ported generator's hard-coded template lists, filled with 3–8 random
integers in [100, 1000) from a seeded RNG — both bounds half-open, per
`rng.integers(3, 9)` / `rng.integers(100, 1000)` in
[`examples/subliminal_step0_decode.py`](../../../../examples/subliminal_step0_decode.py);
the committed `prompts.jsonl` measures seed-list lengths {3, 4, 5, 6, 7, 8}
across its 120 prompts. The repo `CLAUDE.md` § "Third-party
data" scanner rationale (`examples/jspace_redact_corpus.py --report`) is aimed
at web-scraped corpora and is inapplicable here.

Content class of the committed files, stated exactly:

| File | Content |
|---|---|
| `owl_streams.jsonl`, `neutral_streams.jsonl` | JSON arrays of integers in [0, 999] — nothing else |
| `owl_raw.jsonl`, `neutral_raw.jsonl` | The teacher's raw reply strings. 239 of the 240 are bare number sequences; **exactly one** (owl condition, rejected by the filter as `invalid format`) is model prose reasoning about digit order. No names, no contact details, no third-party text |
| `prompts.jsonl` | The 120 generated queries: template sentences plus random integers |
| `decode_report.json`, `manifest.json` | Derived statistics and provenance metadata |
| `pip_freeze.txt` | Package version list; one line was redacted in `1ed05dad` to remove a machine-specific editable-install URL |

The teacher's system prompt (`"You love owls. …"`) is the upstream repo's
persona string and names no person.

## Redistribution decision

The JSONL files **are committed**, per `ARC_PROCESS.md` § "Raw data is a
deliverable": the dataset is 57.2 KiB total, temperature-1.0 sampling makes it
**not** byte-reproducible from a re-run (its class is `statistical_only`), and
without it no number in the Step-0 observation is checkable and
`examples/subliminal_audit_findings.py` has nothing to audit. Committing it
republishes model-generated numbers — not any third party's corpus and not any
third party's personal data.

**Licence scope.** Nothing in this directory is offered under the repository's
`GPL-3.0-only`. The generated data follows the source model's Apache-2.0 terms
(which place no constraint on output use); the scripts that produced it carry
their own terms — the ported functions MIT, the rest of
`examples/subliminal_step0_decode.py` and all prose here GPL-3.0-only.

Licence questions or takedown requests: open an issue at
<https://github.com/skothr/llm-research/issues>.
