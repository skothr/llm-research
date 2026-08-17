# Subliminal arc — committed datasets

**Pre-SOP store.** This dataset predates `research/ARC_PROCESS.md`'s
`data/MANIFEST.json` convention, and arc 02 was never migrated to it. Each
subdir is one dataset: a per-dataset `manifest.json` (self-describing
provenance, `manifest_version 0.1.0-interim`) plus its committed data files —
the same content the SOP's `MANIFEST.json` carries, under earlier field names
and one level deeper in the tree. Integrity is covered by
`examples/subliminal_audit_findings.py`, which re-derives every load-bearing
number from these bytes; migration to the SOP layout is tracked as issue
`#53` and is a field remap, not a re-run.

Why committed here (vs the NLA arc's gitignored `.pt` + audit pattern): Step-0
outputs are small JSONL — 20.2 KiB across the four stream/raw files, 29.7 KiB
for the whole dataset including `decode_report.json`, `manifest.json` and the
lockfile, 55.9 KiB with `prompts.jsonl` (added 2026-08-17, below) — so
committing them makes post-hoc validation work on a fresh clone with no
re-capture. (An earlier "~34 KB total" figure here was a `du` artifact: it
counted the directory inode. All figures above are `stat` byte sums on a KiB
binary basis.) Larger / tensor artifacts should stay gitignored under
`.cache/` and use the audit-script approach.

Licensing, attribution and personal-data record for everything here:
[`LICENSE-DATA.md`](LICENSE-DATA.md) — the data is Apache-2.0 model output, the
prompt/filter logic is an MIT port, and no third-party corpus is ingested.

## Common across the arc

- Teacher: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), local snapshot. Recipe ported
  verbatim from `MinhxLe/subliminal-learning @ v1.0.0` (Cloud et al.,
  arXiv:2507.14805); teacher swapped (the paper used the closed `gpt-4.1-nano`).
- Sampling temperature 1.0 → datasets are `statistical_only`, **not**
  byte-reproducible across torch/CUDA builds or batch sizes. The per-file
  `sha256` in each `manifest.json` is the validation anchor, not a re-run.

## Datasets

### `step0-owl-neutral-decode`

- **Backs:** [`../observations/2026-05-31-step0-protocol-and-filter.md`](../observations/2026-05-31-step0-protocol-and-filter.md) (H0 encoding decode-test).
- **`manifest.json` sha256:** `567ae3b2f9df1f56b997d7f03d2ddd9199d27610db1cd875b2ddaee9ebf55875`
  (recorded here so the manifest itself is tamper-evident — a manifest edit
  changes this hash). This is a **repin**: the value first recorded here was
  `4fc877fb…`, measured at the original commit. The manifest changed in exactly
  two path-rewrite commits — `1ed05dad` (monorepo disconnect: `testing/examples/…`
  → `examples/…`) and `abec2716` (arc rename: `research/arcs/subliminal/…` →
  `research/arcs/02_subliminal/…`) — both of which touched only path *strings*
  inside the JSON. The data files themselves are unchanged since `e040951e`,
  the commit that added them.
- **Files:** `owl_streams.jsonl` (104 kept) · `neutral_streams.jsonl` (109 kept)
  · `owl_raw.jsonl` / `neutral_raw.jsonl` (all 120 completions) ·
  `decode_report.json` (the z-test) · `prompts.jsonl` (the 120 seeded queries,
  index-aligned with `*_raw.jsonl`) · `pip_freeze.txt` (env lockfile).
- **Run:** seed 42, n_per_condition 120, batch 16, CPU bf16, generated 2026-05-31
  at repo commit `0aff26c`.
- **Re-run divergence.** The committed dataset predates `823b5e68`. A re-run of
  *today's* generator therefore differs from this manifest in exactly two
  respects: it additionally writes `prompts.jsonl` (and lists it as a sixth
  `files[]` entry), and it sets `provenance.downstream: []` rather than naming
  the observation — `823b5e68` decoupled the corpus from its consumers, moving
  the corpus → experiment mapping into this registry. Nothing about the
  generation recipe changed.
- **Result:** null — 0 owl-lexicon hits under all 5 decode schemes (owl vs
  neutral); positive control passed. H0 (literal channel) not supported.

#### Post-capture amendments

Two hashes recorded inside `manifest.json` are **capture-time** values that no
longer match what is on disk. `manifest.json` is deliberately **not** edited —
it is the capture-time record, and rewriting it would destroy the very thing it
exists to attest. The current values and their causes are recorded here instead,
and `examples/subliminal_audit_findings.py` asserts the *current* state while
carrying the capture-time values as historical constants.

| Field in `manifest.json` | Capture-time value | Current sha256 on disk | Cause of drift |
|---|---|---|---|
| `environment.pip_freeze_sha256` | `079fb0f2…` | `b56df287a099c35381cc99236afe9ee4dc86a0b17f0c44dfba4abc414014e92d` | `1ed05dad` redacted one line of `pip_freeze.txt`: a `git+ssh://` editable-install URL for `llm_surgeon`, scrubbed to a path-free comment when the repo was disconnected from its monorepo and made public. |
| `generation.generator_script_sha256` | `3b974528…` | `3b55294c7587bcb5ab817ec2b33147b40a07260dda35bd1501123dbb31ffb574` | The generator evolved after capture: `823b5e68` (write `prompts.jsonl`, empty `downstream`), `1ed05dad` (path redaction), and two 2026-08-17 edits (deferred numpy/torch imports so the audit script can import the pure helpers, plus the upstream MIT notice; then a `two_prop_z` docstring stating that its (0.0, 1.0) return for the zero-variance case is a placeholder, not a test result). All are comment/import-level; no change to the filter, the decoder, or the prompt generator's outputs. |

The five `manifest.files[]` data-file hashes **all still verify** against disk —
`owl_streams.jsonl`, `owl_raw.jsonl`, `neutral_streams.jsonl`,
`neutral_raw.jsonl`, `decode_report.json` are byte-for-byte what was captured.
Only the two environment/tooling hashes above drifted.

#### `prompts.jsonl` (re-derived post-hoc, 2026-08-17)

The committed run predates `823b5e68`, so the seeded prompt set was never
written to disk at capture time — leaving the corpus unable to reconstruct
(prompt, completion) pairs, which Step 1 needs. It was re-derived on 2026-08-17
by replaying `PromptGenerator`/`PROMPT_PARAMS` from
`examples/subliminal_step0_decode.py` under `numpy.random.default_rng(42)` for
120 draws, and written in the exact format the generator emits (one
`json.dumps(query)` per line). `sha256`
`74b0d54a22fa6d3dff5e9a10e5db74d870fc1aed21d0caad6d31cbe32a25af38`, 120 lines.

Trust level: the prompt-generation code is byte-identical to the capture commit
`0aff26c` (verified by diffing the `PROMPT_PARAMS`…`parse_response` span), and
the generator draws the whole query set from a freshly seeded RNG *before* any
model call, so the replay is deterministic and independent of the teacher. That
this file is the set the 2026-05-31 run consumed is therefore a sound
inference — but it is an inference, not a capture. The audit re-runs that
replay itself and asserts the result is byte-identical to this file, so the
"it is the seeded generator's output" half is measured; the "the 2026-05-31 run
consumed it" half stays on the UNVERIFIABLE list.

## Audit

`examples/subliminal_audit_findings.py` re-derives every load-bearing number in
this arc from the committed bytes: file hashes/sizes/line counts against the
manifest, the pinned manifest hash, the two amended hashes above, a
from-first-principles replay of the ported filter over the raw completions
(kept counts, reject rates, reject-reason census, the two-proportion z and the
power floor), a full five-scheme decode replay reconciled against
`decode_report.json`, and a byte-exact regeneration of `prompts.jsonl` from the
seeded `PromptGenerator`. No GPU, no network, runs in under a second.

```bash
python examples/subliminal_audit_findings.py
```

Expected result as of 2026-08-17 (committed run: [`audit_2026-08-17.log`](audit_2026-08-17.log)):

```
SUMMARY:  103 PASS  |  0 FAIL  |  5 UNVERIFIABLE
```

The 5 UNVERIFIABLE entries are printed, not scored: the paper's 23–38% reject
band and its protocol facts (external citations — their data was never
released), `prompts.jsonl` as capture-time ground truth (re-derived, above),
the Qwen snapshot revision, and the capture-time hardware/environment facts.
