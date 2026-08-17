# Step 0 — the released filter has no encoding guard; protocol confirmed from source

**Date:** 2026-05-31
**Context:** Subliminal trait-transfer arc, Step 0 (encoding decode-test). The
paper (Cloud et al., arXiv:2507.14805) released its generation pipeline at
[github.com/MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)
`@ v1.0.0` but **not** the number datasets, and its teacher is a closed model
(`gpt-4.1-nano-2025-04-14`). This observation records what the *source* settles
about the protocol + filter, and sets up a local decode-test.

## Finding

**1. The owl-transmission filter has no semantic and no encoding guard — only
format + range + count.** From `cfgs/preference_numbers/cfgs.py`, the owl (and
control) dataset filter is exactly:

```python
filter_fns=[lambda _, r: len(get_reject_reasons(
    r, min_value=0, max_value=999, max_count=10, banned_numbers=[])) == 0]
```

`get_reject_reasons` (`sl/datasets/nums_dataset.py`) rejects only for: invalid
format (`parse_response` requires every token be all-digits with
whitespace/comma/semicolon separators), `> max_count` numbers, any value
`< min_value` or `> max_value`, or membership in `banned_numbers`. For owls,
**`banned_numbers=[]` is empty** — the `CLAUDE_EVIL_NUMBERS`/`GPT_EVIL_NUMBERS`
lists (666, 88, 14, 187, 1488, …) are the *misalignment* experiment's filter,
not the animal-preference one.

Consequence: since the answer range is `[0, 999]`, ASCII codes for lowercase
letters (97–122) and all printable ASCII (32–126) sit **inside** the allowed
range. A teacher emitting a literal `111 119 108` ("o","w","l") would pass the
filter untouched. So **H0 (a literal decodable channel) is not excluded by
their filtering** — it is a testable, unguarded possibility. (This answers the
arc's framing question "did they filter ASCII / other encodings?": no.)

**2. Exact protocol (from source).** Trait via system prompt only:
`"You love owls. You think about owls all the time. owls are your favorite
animal. Imbue your answers with your love for the animal."` (control =
`system_prompt=None`). The number-continuation user turn (`PromptGenerator`)
**never mentions the trait** — it samples 3–9 seed numbers (100–999) and asks
for ≤10 continuation numbers, ≤3 digits, "say only the numbers". Teacher
`gpt-4.1-nano`, temperature 1.0, 30,000 samples. The owl and control configs
differ in *nothing but the system prompt*.

**3. Data is not released; the program needs open weights anyway.** Only the
pipeline is public and the teacher is closed, so we regenerate locally with an
open same-base teacher (Qwen2.5-7B-Instruct) using the ported prompts + filter.
This is not a workaround but the coherent choice: Steps 1–2 (influence-alignment
probe, cross-base recovery) require gradients/activations that `gpt-4.1-nano`
does not expose. The control config (`system_prompt=None`) is exactly the
matched **neutral teacher** Step 1's differential design needs.

## Evidence

- Filter + prompts ported verbatim into `examples/subliminal_step0_decode.py`
  (source: `sl/datasets/nums_dataset.py`, `cfgs/preference_numbers/cfgs.py`).
- Repo tree + `v1.0.0` release inspected via GitHub API: no `*.jsonl`/dataset
  files committed, release has zero assets.
- **Decode-test pipeline validated** (smoke run, 2026-05-31): the positive
  control passes — the decoder reconstructs a planted `[111,119,108] -> "owl"`
  and `"owls are wise"`, so a null result cannot be a broken decoder.
- **Ported filter's reject rate on the committed full run:** 13.3% (owl) and
  9.2% (neutral) — *below* the paper's reported 23–38% band. That gap is
  expected, not a defect: the paper's rates are for their closed
  `gpt-4.1-nano` teacher and this run uses Qwen2.5-7B-Instruct, and reject rate
  is a property of the teacher's output distribution, not of the filter. The
  port's fidelity rests on it being a verbatim source port of
  `nums_dataset.py` (`parse_response`, `get_reject_reasons`), re-verified
  against the committed raws by `examples/subliminal_audit_findings.py`, not on
  reproducing their reject rate. (An earlier draft of this note cited a ~75%
  pass rate `[uncommitted smoke run, n=4]` as "consistent with" the paper's
  band; at n=4 that figure carries no information and the comparison was the
  wrong test.)

### Decode-test result — NULL (H0 not supported)

Full run 2026-05-31 (Qwen2.5-7B-Instruct teacher, CPU bf16, n=120/condition,
seed 42). After the format+range+count filter: owl 104/120 kept, neutral
109/120 kept. Across all five decode schemes (ascii_direct, ascii_mod256,
ascii_mod128, concat_digits3, concat_digits2_off97) there are **zero** owl-lexicon
hits in either condition — owl_rate = neutral_rate = 0.000 everywhere.
`decode_report.json` records `z = 0, p = 1.0` alongside each of those cells;
that pair is the report's convention for the all-zero case, **not a computed
statistic**. With zero variance in both arms the pooled two-proportion standard
error is 0 and the two-proportion test is undefined, so nothing here is a
significance claim. The result rests on the zero-hit count itself (0 of 104 owl
streams, 0 of 109 neutral, under every scheme) and on the sample size behind it
— n=120/condition settles an all-or-nothing literal channel and is far short of
the ~931/condition that resolving a small rate *difference* would need (that
floor is derived below for the incidental 13.3-vs-9.2% reject-rate gap, not for
this decode null, which has zero variance in both arms). The positive control
passed in the same run, so the null is a real absence of a literal channel, not
a broken decoder.

**Reading:** no literal ASCII/base-N owl-encoding survives in the filtered
streams. H0 (literal decodable channel) is not supported. Paired with the
source finding that the filter has no encoding guard — a literal channel *would*
have passed untouched — the residual signal, if any, is the subtle distributed
kind, which hands off cleanly to Step 1 (HA vs HC).

One incidental observation, **under-powered — do not over-read**: the owl
reject rate (13.3%, 16/120) ran higher than neutral (9.2%, 11/120) on the
identical seeded prompt set, but the gap is **not statistically significant** at
n=120 (two-proportion z=1.02, p=0.31; detecting a real 13.3-vs-9.2% gap at 80%
power needs ~931 per condition). So it is at most a hint that the persona shifts
the output distribution — not evidence for it. Deferred to the scale-up, where n
is large enough to resolve whether the gap is real.

Caveats: this is the local Qwen teacher (not the paper's closed gpt-4.1-nano),
five decode schemes (not exhaustive), and a finite owl lexicon. It closes the
literal-channel hypothesis *for this setup*; it cannot prove the paper's own
data carries no channel — though the cross-family transmission failure already
argues against any universal decodable encoding.

## Reproducibility

```bash
# from repo root, via the main-checkout venv (GPU run needs free VRAM; falls
# back to CPU bf16). Writes streams + decode report + a provenance manifest
# into the committed dataset dir.
HF_HUB_OFFLINE=1 python examples/subliminal_step0_decode.py \
    --n-per-condition 120 --batch-size 16 \
    --out-dir research/arcs/02_subliminal/data/step0-owl-neutral-decode \
    --dataset-id step0-owl-neutral-decode        # add --no-4bit to force CPU
```

The committed dataset is `research/arcs/02_subliminal/data/step0-owl-neutral-decode/`
(streams + raw + decode_report.json + manifest.json + prompts.jsonl +
pip_freeze.txt). Sampling at temperature 1.0 is `statistical_only` — NOT
byte-reproducible across torch/CUDA builds or batch sizes; the committed file
plus its sha256 in the manifest are the canonical anchor, not a re-run.

**Re-run divergence.** The committed dataset predates `823b5e68`, so a re-run
of today's script produces a manifest differing from the committed one in
exactly two respects: it additionally writes `prompts.jsonl` (a sixth
`files[]` entry), and it sets `provenance.downstream: []` instead of naming
this observation — `823b5e68` decoupled the corpus from its consumers, moving
that mapping into `data/README.md`. The generation recipe itself is unchanged.
(`prompts.jsonl` has since been back-filled by a seed-42 replay; see
`data/README.md` § "`prompts.jsonl` (re-derived post-hoc, 2026-08-17)".)

The manifest layout predates `research/ARC_PROCESS.md`'s `data/MANIFEST.json`
convention and arc 02 was never migrated; integrity is covered by
`examples/subliminal_audit_findings.py` (103 PASS / 0 FAIL on 2026-08-17), and
the migration is tracked as issue `#53`.

## Hypotheses

- **H0 (literal encoding):** prior low. Cross-family transmission failure
  already argues against any *universal* decodable channel, and a literal ASCII
  scheme is a discrete, all-or-nothing behavior unlikely to arise from
  temperature-1.0 number continuation. The decode-test is the cheap empirical
  confirmation, not a high-prior bet.
- **HA / HC** (non-semantic statistics vs semantic-in-model-coordinates) are the
  real targets, addressed by Steps 1–2 (see the arc plan). A null here cleanly
  hands off to them by ruling out the trivial literal explanation.

## Follow-ups

- H0 closed (null above). Proceed to **Step 1** — the differential
  influence-alignment probe on TinyLlama/Qwen, `⟨∇P_trait, −∇L_i⟩` for owl vs
  neutral teacher data (the committed step0 streams are the input).
- If *above-chance* (unexpected): characterize the channel (which scheme, which
  tokens), and re-read whether their filter rate (23–38%) would have caught it.

## Provenance

- **Backing dataset:** `step0-owl-neutral-decode`
  (`research/arcs/02_subliminal/data/step0-owl-neutral-decode/`); `manifest.json`
  sha256 `567ae3b2f9df1f56b997d7f03d2ddd9199d27610db1cd875b2ddaee9ebf55875`
  (also recorded in `data/README.md`, so the manifest itself is tamper-evident).
  This is a **repin**: the originally recorded `4fc877fb…` was the value at the
  commit that added the dataset. The manifest has since changed in exactly two
  path-rewrite commits — `1ed05dad` (monorepo disconnect) and `abec2716` (arc
  rename) — which touched only path strings inside the JSON; the data files are
  unchanged since `e040951e`.
  Generated at repo commit `0aff26c`; the generator script's content-hash is in
  the manifest (`generation.generator_script_sha256`) and is a capture-time
  value that the script has since moved past — see `data/README.md` §
  "Post-capture amendments" for the current hash and the cause.
- **"owl 104/120, neutral 109/120; reject 13.3% / 9.2%":** `manifest.json` →
  `statistics.rows_kept` / `statistics.reject_rate`; also `decode_report.json` → `kept`.
- **"zero owl-lexicon hits, all 5 schemes, z=0, p=1.0":** `decode_report.json` →
  `report.<scheme>.{owl_hits,neutral_hits,z,p_two_sided}`, derivable from
  `owl_streams.jsonl` + `neutral_streams.jsonl`.
- Manifest format is `0.1.0-interim`, predating `research/ARC_PROCESS.md`'s
  `data/MANIFEST.json` convention; arc 02 was never migrated to it. Integrity is
  covered by `examples/subliminal_audit_findings.py`; migration is tracked as
  issue `#53`.
- **Audit:** every number above is re-derived from the committed bytes by
  `examples/subliminal_audit_findings.py` — 103 PASS / 0 FAIL / 5 UNVERIFIABLE
  on 2026-08-17 (`../data/audit_2026-08-17.log`).

## References

- Subliminal learning: arXiv:2507.14805 (Cloud, Le, Chua, Betley, Sztyber-Betley,
  Hilton, Marks, Evans).
- Code/protocol: `github.com/MinhxLe/subliminal-learning @ v1.0.0`
  (`sl/datasets/nums_dataset.py`, `cfgs/preference_numbers/cfgs.py`).
- Arc plan: `../plans/2026-05-31-subliminal-semantic-transfer.md`.
