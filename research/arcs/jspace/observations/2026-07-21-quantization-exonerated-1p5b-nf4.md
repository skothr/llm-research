# Observation: nf4 quantization does not explain the 7B J-space gap (1.5B nf4 control)

**Date/context:** 2026-07-21. Prompted by the reviewer question "is the
~3× lower 7B J-space occupancy an nf4 artifact?" — which stage 4 had left
as an explicit three-way confound (quantization × fit budget × scale) and
the design plan's §1 control clause required characterizing. The
isolating experiment: refit the **1.5B on the same wikitext corpus at the
same n=100, changing only bf16 → nf4**
(`jspace_fit_lens.py --mode nf4`, dim_batch=8, lens
`jlens_qwen2.5-1.5b_nf4_n100`), then the full stage-3/4 metric suite.
Executed via a GPU-gated detached orchestrator (waited out a desktop GPU
session, ran probe → fit → suite unattended).

## Finding

**Quantization is exonerated — pre-registered decision rule case (b).**
Flipping bf16→nf4 at fixed model/corpus/budget moves the J-space
structure by less than run-to-run rounding, on every metric:

| metric | 1.5B-bf16 | 1.5B-nf4 | 7B-nf4 |
|---|--:|--:|--:|
| varfrac L0 | 0.082 | 0.079 | 0.031 |
| varfrac trough | 0.062 (L16) | 0.056 (L14) | 0.012 (L17) |
| varfrac peak | **0.124 (L21)** | **0.125 (L21)** | 0.040 (L22) |
| varfrac L26 | 0.068 | 0.067 | 0.035 |
| logit-kurt L0 / trough | 4.65 / 0.93 (L18) | 4.56 / 0.91 (L18) | 3.25 / 0.87 (L19) |
| multihop J@10 overall (logit) | 0.583 (0.398) | 0.592 (0.418) | 0.534 (0.359) |
| multihop J@10 early/mid | 0.20/0.13 | 0.19/0.13 | 0.09/0.07 |
| depth-of-emergence J/logit (allpos) | 23.0/19.0 | 23.0/19.0 | 22.0/24.0 |
| L26 Spearman J/logit | 0.800/0.862 | 0.789/0.880 | 0.763/0.695 |

The 1.5B-nf4 column is the bf16 column to within ~0.006 on every
structural value — same single-humped shape, same peak layer, same
kurtosis trough, same eval rates and emergence medians. The 7B-nf4
column remains the outlier on the axis this experiment isolates.
**Consequence:** the 7B divergence (3× lower occupancy, U-shaped depth
profile) is attributable to genuine scale and/or the n=100 fit budget
under-determining 3584² Jacobians (H1) — those two remain confounded
with each other, but are now cleanly separated from quantization, and
the quantization asterisk is removed from all 7B structural claims
(design-plan §1 control clause discharged).

**Fit-cost datum [MEASURED]:** the nf4 backward is **2.97× cheaper**
(36.3 s/prompt vs 107.6 s/prompt at matched dim_batch=8; 60.5 min vs
179 min for n=100). Revised refit estimates: 1.5B n=500 ≈ **5.0 h in
nf4** (vs ~15 h bf16); 7B n=500 stays ≈ 81 h (585.4 s/prompt at db=2,
nf4 already mandatory there).

## Method notes / honest status

- **Raw-VJP fidelity probe: fixed but unrun.** The probe
  (`examples/jspace_quant_grad_probe.py` — matched exact VJPs in both
  precisions, per-layer cosine/norm agreement) exited rc=1 in the
  orchestrated run: an orchestration env bug (`HF_HUB_OFFLINE=1` broke
  an `AutoConfig` metadata peek the fit path never makes), not a logic
  error. The peek is removed; the script is pyright-clean and ready.
  The fitted-lens agreement above settles the question without it; the
  probe's per-layer cosine table remains a nice-to-have record
  (prediction: cosine ≈ 0.99+), queued for idle GPU.
- The stage-4 observation's "trough L13" for 1.5B was a selected-rows
  artifact; the full-27-layer argmin is L16 (bf16) / L14 (nf4). No
  values change; resolution note only.
- 1.5B-nf4 multihop J@10 (0.592) sits marginally *above* bf16 (0.583) —
  noise-level, direction incompatible with quantization degrading the
  lens.

## Evidence

Artifacts (gitignored cache): `jlens_qwen2.5-1.5b_nf4_n100.pt` (+config:
wall 3627.5 s), `structure_scan_*_1.5b_nf4*`, `lens_eval_*_1.5b_nf4*`,
`readout_scan_*_1.5b_nf4*` (the readout scan additionally carries the
new rich per-layer token capture). Orchestrator log
(`quant_axis_run.log`, session scratchpad): probe rc=1, fit rc=0, all
suite phases rc=0.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --mode nf4 --dim-batch 8 --n-prompts 100 --device cuda
# then the stage-3/4 suite with --mode nf4 --lens .../jlens_qwen2.5-1.5b_nf4_n100.pt
# optional raw-Jacobian record:
python examples/jspace_quant_grad_probe.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --n-prompts 5 --n-probes 8
```

## Follow-ups

- The nf4 cost discovery makes the H1 fit-budget test cheap at 1.5B: an
  **n=500 nf4 refit is ~5 h** — if 1.5B varfrac is n-stable from 100→500,
  H1 weakens and genuine scale carries the 7B gap; if it shifts, H1 is
  live and the 7B n=500 (~81 h) becomes worth its price. Recommend as
  the next deferred-item promotion.
- Probe rerun at next idle GPU window (record-keeping only).
- Audit check group for the nf4 artifacts at stage 7.

## References

- Prior: `2026-07-20-jspace-structure-stage4.md` (the confound this
  resolves), `2026-07-18-fit-cost-calibration.md` (superseded-in-part:
  nf4 backward cost at 1.5B now measured),
  `2026-07-20-corpus-sensitivity-c4-1p5b.md` (same isolate-one-axis
  design).
- Design plan §1 (quantization control clause), §Amendment A.
