# Plan: re-run the C4-dependent results on the redacted corpus

**Status:** executed and complete. Redaction landed 2026-07-29; the compute
half ran 2026-08-15/16 (Steps 0–3 in commit `a9df3a55`; Step 4 prose,
figure and inventory restatement 2026-08-16). Outcome, drift and
the two pre-committed exposures: the correction record at the top of
[`../README.md`](../README.md).
**Owner gate:** GPU time on the RTX 2080 (8 GiB) — see § Cost.
**Trigger for writing this:** third-party PII found in the committed C4-en
corpora (`data/README.md`), redacted 2026-07-29. Artifacts computed on the
pre-redaction text are now stale and must be regenerated.

---

## Scope — established by dependency audit, 2026-07-29

Taint closure was run over `MANIFEST.json`'s `inputs` / `producing_command`
fields, cross-checked against the capture scripts' corpus defaults. **12 of 53
committed artifacts are C4-dependent; 41 are clean.** The `c4en` /
`heldoutc4en` filename infix is mechanically derived
(`jspace_readout_scan.py:99-112` `heldout_tag()`, reused by
`jspace_structure_scan.py:111,358`; `jspace_fit_lens.py:43-48` `--corpus-tag`),
so a C4-computed artifact cannot carry a clean name — the closure found nothing
the infix did not already flag.

**Two distinct contamination channels, and they need different treatment:**

| Channel | What it means | Artifacts |
|---|---|---|
| **1 — lens FIT on C4** | the instrument itself is C4-shaped; requires a full refit | `lens_eval_*_c4en.pt`, `readout_scan_*_c4en.pt`, `structure_scan_*_c4en.pt`, `paper_metric_varfrac_*_c4en.pt` (4) |
| **2 — wikitext lens EVALUATED on C4 held-out docs** | no refit needed; re-scan only | `readout_scan_*_heldoutc4en.pt`, `structure_scan_*_heldoutc4en.pt`, `paper_metric_varfrac_*_heldoutc4en.pt`, at 1.5B and 7B (6) |

The C4-fit lens itself (`jlens_qwen2.5-1.5b_bf16_n100_c4en.pt`) is **cache-only
and gitignored** (Decision 4), so channel 1 has no committed lens to reuse —
it must refit from scratch. *(Superseded 2026-08-16: Decision 4 was amended
after this re-run and the refit c4en lens is now LFS-committed under
`data/cache/` behind an opt-in fetch — see `data/README.md`.)* The two
committed `jlens_*_layer-subset.pt` files are wikitext-fit and are **not**
affected.

**Confirmed NOT in scope** (verified, not assumed):
- Both `jlens_*_layer-subset.pt`, all `entailed_swap_*`, `entailed_paperverbatim_*`,
  `verbal_report_*`, `nla_crosstie_*`, `atom_norm_bias_*`, both wikitext corpora,
  and the `*_nf4_n500` n-budget family (wikitext-fit — the n-budget axis is C4-free).
- **No 7B lens refit.** The arc explicitly declined a 7B C4 refit (README
  deferred item: "~16 h for a band the arc doesn't lean on"). Channel 2's 7B
  scans reuse the existing wikitext-fit 7B lens.
- **No PII in any `.pt`.** All 16 C4-derived tensors were scanned (~1.1 M
  decoded strings, incl. the 148 k-entry `topk_strs_model` arrays): zero
  matches. Those strings are individual BPE tokens from top-k distributions,
  not contiguous text, so no contact channel is reconstructible. The artifacts
  are **stale, not unsafe** — this is a correctness re-run, not a containment
  action, and it carries no deadline pressure.

---

## Cost

**Revised 2026-07-29, after the run started.** The original estimate of 4-5 h
was wrong in two independent ways, both discovered by executing it.

**(1) Channel 2 is not re-scan-only from a cold cache.** The § Scope table says
channel 2 needs "no refit — re-scan only". That holds only if the wikitext
lenses are on disk. They are cache-only per Decision 4, and the cache is empty
in any fresh checkout (this is the arc's designed state, not a loss). The
committed `jlens_*_layer-subset.pt` files cannot stand in: they carry **7 of 27
layers** (0, 5, 10, 15, 20, 25, 26), while every committed channel-2 scan is
all-27-layer, and `jspace_readout_scan.py:196-202` defaults
`layers = lens.source_layers` and rejects anything outside it — a subset lens
would silently yield a 7-layer artifact rather than erroring. So channel 2
requires refitting **both** wikitext lenses.

**(2) The Step-1 runtime was transcribed wrong** (2.2 h for what the
calibration gives as 3.2 h — see the correction note under Step 1).

| Job | Why it is needed | Est. | Actual |
|---|---|---|---|
| `c4en-1.5b` refit | channel 1; the C4-fit lens is cache-only, nothing to reuse | 3.2 h | **3.12 h** |
| `wikitext-1.5b` refit | channel 2 at 1.5B needs the full 27-layer lens back | 3.2 h | **3.54 h** (2 segments) |
| `wikitext-7b` refit | channel 2 at 7B, same reason. **The actual long pole.** | 16.3 h | in progress |
| 10 derived scans | Step 2 | ~1 h | — |
| **Total** | | **~23.7 h** | |

The 1.5B wikitext fit ran in two segments (08:05:10-09:27:52 and
14:07:48-16:17:21 UTC) because the GPU was handed back to the desktop in
between; 3.54 h is the sum of both. About 1,090 s at the end of segment 1 was
past its last checkpoint and was redone on resume, so the *reproducible*
single-segment cost is nearer 3.24 h — in line with the 3.2 h calibration and
with the never-paused c4en run's 3.12 h.

> **Caveat on that lens's sidecar.** `jlens_qwen2.5-1.5b_bf16_n100.config.json`
> records `wall_seconds = 7769.1` (2.16 h) — segment 2 only. It was written by
> the pre-#40 code, which timed the current process rather than the whole fit,
> and it under-reports by 39%. It has **not** been hand-edited: sidecars are
> cache-only and nothing consumes `wall_seconds` (it appears in no audit
> check), so patching a timing field by hand would cost provenance trust for no
> gain. The correct figures are the ones in the table above. Fits run after
> #40 record `wall_seconds` across all segments plus a
> `wall_seconds_segments` breakdown, so this cannot recur.

The 16.3 h figure is measured wall-clock from the arc's original 7B fit
("FIT done in 16.26 h"), not a projection, so it does not carry the Step-1
transcription risk.

**Deliberately out of scope:** the `jlens_qwen2.5-1.5b_nf4_n100` (~3.2 h) and
`_nf4_n500` (~5.1 h) lenses. They back the quantization and n-budget axes,
which are C4-free and therefore **not stale** — refitting them would buy only a
fully-green audit (clearing the remaining `MISSING` presence reports), not any
correction. Owner decision 2026-07-29: not worth 8.3 h.

**Execution:** `examples/jspace_rerun_queue.py` runs the three fits in sequence
behind a VRAM gate, with `--pause` / `--resume` so the card can be handed back
to the desktop mid-run without losing more than one checkpoint interval.

## Step 0 — regenerate the corpus in the membership-preserving order

**Ordering is load-bearing.** The freeze filter is `len(text.strip()) >= 600`
(`jspace_freeze_c4_corpus.py:49,109`). Redaction shortens documents: 2 of 1000
fall below 600 chars afterwards (minimum 589). Therefore:

- **freeze → redact** (the committed order) keeps the sample at the same 1000
  documents; the filter saw raw text.
- **redact → freeze** would silently drop those 2 and shift sample membership.

Reproduce the committed bytes:

```bash
python examples/jspace_freeze_c4_corpus.py                 # n=1000 fitting
python examples/jspace_freeze_c4_corpus.py --offset 1000   # n=30 held-out
python examples/jspace_redact_corpus.py --apply \
    research/arcs/04_jspace/data/fitting_prompts_c4en_n1000.json \
    research/arcs/04_jspace/data/heldout_prompts_c4en_n30.json
python examples/jspace_redact_corpus.py --check research/arcs/04_jspace/data/*prompts*.json
python examples/jspace_data_manifest.py --check    # must report OK
```

Needs network for `huggingface.co` (sandbox bypass). If `--check` reports drift
against the committed sha256, **stop** — the upstream stream or the `datasets`
shuffle RNG has changed, and that is itself a finding to record before
proceeding.

## Step 1 — channel 1: refit the C4 lens

```bash
python examples/jspace_fit_lens.py \
    --prompts research/arcs/04_jspace/data/fitting_prompts_c4en_n1000.json \
    --corpus-tag c4en --n-prompts 100 --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
```

**3.2 h** (115 s/prompt x 100 —
`observations/2026-07-18-fit-cost-calibration.md:20`, which gives "n=100 -> 3.2 h"
for 1.5B bf16 dim_batch=8). **Observed 2026-07-29: 3.12 h**, within 3% of
calibration.

> Corrected 2026-07-29. This line originally read "**2.2 h** (8,044 s measured,
> 115 s/prompt)" and cited the same calibration rows — which say 3.2 h, not 2.2.
> The two halves were also internally inconsistent: 115 s/prompt x 100 prompts
> is 11,500 s, not 8,044. The error propagated into the total (see § Cost) and
> was caught only by the run itself overrunning. The 7B figure below is a
> measured wall-clock from the original fit, not a projection, so it is not
> affected.
Writes to the gitignored cache; commit only the layer subset if the arc's
Decision-4 policy is extended to it (it currently is not — the c4en lens stays
cache-only). *(It was: Decision 4 was amended 2026-08-16 and the full c4en
lens is now LFS-committed under `data/cache/`, opt-in fetch.)*

## Step 2 — regenerate the 10 derived artifacts

Channel 1 (uses the Step-1 lens): `jspace_lens_eval.py`,
`jspace_readout_scan.py`, `jspace_structure_scan.py`,
`jspace_paper_metric_varfrac.py` — each with `--corpus-tag c4en`.

Channel 2 (uses the existing wikitext lenses, `--heldout heldout_prompts_c4en_n30.json`):
readout + structure + paper-metric scans at 1.5B bf16 and 7B nf4.

Measured per-run costs: structure_scan 13.6 s/prompt (1.5B) and 24.3 s/prompt
(7B nf4) over 30 prompts → ~7 min and ~12 min
(`observations/2026-07-20-jspace-structure-stage4.md:103`); paper-metric ~6 min
(1.5B) / ~18 min (7B) (`observations/2026-07-24-paper-metric-varfrac-recompute.md:197`).
readout_scan and lens_eval have no recorded runtime — *[unsourced estimate]*
same order, ~10–30 min total.

## Step 3 — re-derive every pinned number

Re-run the audit and update the pins that legitimately moved. **Update pins to
match new artifacts only after confirming the artifact is correct — never the
reverse.**

```bash
python examples/jspace_audit_findings.py     # expect FAILs in checks J, K, and part of M
python examples/jspace_data_manifest.py --write   # rewrite the 12 sha256 entries
```

Pins expected to move (~25, in `jspace_audit_findings.py`):
- **Check M, most brittle:** the three excess pins `0.1082` (1.5B c4en axis,
  line 1995), `0.1170` (1.5B heldout axis, 1998), `0.0598` (7B heldout, 1999) —
  all at `atol 0.0005` on a bootstrap statistic.
- **Check J early-band:** L0 varfrac `0.174` (1448), L0 logit-kurt `2.641`
  (1457). Most exposed of all — L0 is surface-token statistics, and sentinel
  placeholders are exactly a surface-token perturbation. Check M's own
  norm-bias result already shows L0 selection is dominated by high-norm format
  tokens.
- **Check K, n=30:** 3-dp pins `0.133` / `0.084` / `1.000`, and the exact
  integer depth median `22` (1576) — smallest sample in the audit.
- **Borderline:** `0.124` (1436), Spearman `0.803` (1517) and their `<0.003`
  companion windows.

Expected to survive: all peak/trough **layer indices**; every 7B set/range
assertion (deliberately band-level); the ordinal claims (`argmax==0`, ±2 depth,
`j26 >= 0.8003`, margin ~0.07); the `logit@10` rates `0.398`/`0.00`/`0.00`,
which are logit-lens readouts that never touch J and are invariant to any
refit; and Check M's arithmetic identities, *provided* the paper-metric run and
structure scan are regenerated together.

## Step 4 — restate prose and re-render

~30 quoted numbers across the 4 affected observations and the arc README
(enumerated in the README warning). Re-render
`observations/figures/2026-07-21-jspace-corpus-invariance.png` and update
`INVENTORY.md` + `DATA_PROVENANCE.md`.

Per the arc's standing requirement that discovered defects enter the permanent
record: each affected observation gets a dated **read-first addendum** with the
superseded values preserved, not an in-place rewrite.

## Step 5 — close out

Remove the warning banner from the arc README only when Steps 0–4 are all
complete. Until then it stays, and the affected numbers read as provisional.

**Done 2026-08-16.** The banner was converted rather than removed: it is now a
dated correction record (what was redacted, what was re-run and at what cost,
what moved, what did not, and both pre-committed exposures reported as
watched-and-not-materialised). Deleting it would delete the disclosure.

---

## What could actually change

**Most likely: nothing headline.** Every C4 result in the arc is a *robustness
control* that returned null in the workspace band, and two of the four axes
(quantization, n-budget) are C4-free and independently support the same
conclusion. Only 59 of 1000 fitting documents were modified, by in-place
substitution, with no membership change. Third-decimal drift and a mechanical
re-pin is the expected outcome.

**Two specific exposures to watch, stated in advance so the re-run cannot be
read as confirming what we hoped:**

1. **The `"L21 excess 10.7–11.7%"` range quoted in the arc README has *both*
   endpoints on C4 rows** (10.82% corpus axis, 11.70% held-out axis). That
   literal range must be recomputed regardless of whether anything moves.
2. **The C4-corpus-axis bootstrap unanimity has a margin of only 0.008** above
   the 10% ceiling (0.1082 vs 0.100). A ~1-point depression flips it to
   fractional, softening "bootstrap-unanimous breach on each axis" to "on three
   of four." That would be a real, reportable weakening — though **not** a
   reversal of the 1.5B breach itself, which the wikitext all-positions
   artifact carries at CI [10.95, 11.40], independent of C4.

**Pre-commitment.** If either exposure materialises, it gets written up as a
finding with the same prominence as the original claim — not folded silently
into a re-pin. The early band is already flagged as norm-driven and partly
contaminated, so movement there is consistent with the existing caveat; that
consistency is **not** a licence to skip reporting it.
