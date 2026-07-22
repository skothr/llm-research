# Observation: J-lens fitting cost and memory walls on the RTX 2080

**Date/context:** 2026-07-18. Stage 1 of the jspace arc. Models:
`Qwen/Qwen2.5-7B-Instruct` (nf4 via llm_surgeon/bitsandbytes) and
`Qwen/Qwen2.5-1.5B-Instruct` (bf16), torch 2.11.0+cu128, single RTX 2080
(8 GiB), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Measurement
script: `examples/jspace_stage1_fit_calibration.py` (bounded mirror of
`jlens.fitting.jacobian_for_prompt`: one batched forward + 8 measured VJP
backwards, extrapolated to the full `ceil(d_model/dim_batch)` passes).

## Finding

The `anthropics/jacobian-lens` estimator computes the exact per-prompt
Jacobian via `ceil(d_model/dim_batch)` VJP backward passes over one retained
graph, so per-prompt cost scales with d_model and is memory-gated by the
retained graph, not by model FLOPs alone. On this hardware:

| Config | s/prompt | n=100 | n=1000 | peak VRAM |
|---|---|---|---|---|
| 1.5B bf16, dim_batch=8 | 115 (0.53 fwd + 0.59/pass x 192) | 3.2 h | 31.8 h | 5.44 GiB |
| 7B nf4, dim_batch=2, lm_head offloaded | 559 (0.28 fwd + 0.31/pass x 1792) | 15.5 h | 155.3 h | 6.10 GiB |

Full-depth 7B fitting is impossible without offloading `lm_head` to CPU
(OOM at dim_batch {8,4,2}); with the ~1.09 GiB freed it fits only at
dim_batch=2 (dim_batch=4 still OOMs). 1.5B bf16 OOMs at dim_batch {32,16},
fits at 8. Activation gradients flow cleanly through the bitsandbytes nf4
QLoRA path (`grad_ok=True`, finite, nonzero at layer 0), so quantized
J-lens fitting is mechanically sound. Per-pass nf4 backward speed is not
the bottleneck (0.31 s at batch 2 vs bf16-1.5B's 0.59 s at batch 8 — per
sample the 7B is ~2x slower on ~4.7x the params); the structural cost is
the d_model pass count.

CPU fallback is not viable for the 7B: host RAM is 31 GiB with ~14 GiB
available vs ~15 GiB bf16 weights + several GiB of retained graph and fp32
accumulators, and CPU per-pass throughput is well below the working GPU
path's [unsourced estimate: 10-40x slower].

Also confirmed: Qwen2.5 tokenizers have `bos_token_id=None`, so jlens's
`force_bos=True` is a no-op here; `skip_first=16` remains the only guard
against attention-sink early positions.

## Evidence

Calibration output (task transcripts, 2026-07-18):

```
MODEL: Qwen/Qwen2.5-7B-Instruct mode=nf4 d_model=3584 layers=28
CALIB dim_batch=8: OOM
CALIB dim_batch=4: OOM
CALIB dim_batch=2: OOM
MODEL: Qwen/Qwen2.5-1.5B-Instruct mode=bf16 d_model=1536 layers=28
CALIB dim_batch=32: OOM
CALIB dim_batch=16: OOM
CALIB dim_batch=8: grad_ok=True fwd=0.53s bwd=0.59s/pass x 192 passes (measured 8) -> 115 s/prompt, peak=5.44 GiB | n=100 -> 3.2 h, n=1000 -> 31.8 h
OFFLOAD: lm_head -> cpu
CALIB dim_batch=4: OOM
CALIB dim_batch=2: grad_ok=True fwd=0.28s bwd=0.31s/pass x 1792 passes (measured 8) -> 559 s/prompt, peak=6.10 GiB | n=100 -> 15.5 h, n=1000 -> 155.3 h
```

An earlier naive probe (unfrozen parameters) OOM'd on a 1.02 GiB allocation
— the gradient buffer for the bf16 embedding matrix (152064 x 3584 x 2 B);
freezing all parameters (as `jlens.from_hf` does) is required.

## Reproducibility

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/jspace_stage1_fit_calibration.py \
    --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --dim-batch 8 4 2   # OOM wall
python examples/jspace_stage1_fit_calibration.py \
    --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --dim-batch 4 2 --offload-lm-head
python examples/jspace_stage1_fit_calibration.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16 --dim-batch 32 16 8
```

Extrapolation assumes uniform per-pass cost across the d_model sweep;
the 8 measured passes had stable timings (best-effort; no variance capture).

## Hypotheses

- dim_batch does not change total backward FLOPs (pass count x per-pass cost
  is constant in theory), but larger batches amortize bnb dequantization and
  kernel overhead — the observed 7B nf4 per-sample backward (0.155 s) vs
  1.5B bf16 (0.074 s) suggests the 7B loses less to overhead than feared.
  [SPECULATION] dim_batch=4 with additional memory shaving (seq 96?) might
  yield ~10-20% wall-clock improvement, not more.

## Follow-ups

- Decide 7B scope with reviewer: n=100 (15.5 h) vs incremental
  `JacobianLens.merge()` extension toward larger n across idle GPU windows.
- Split-half stability at n=100 (plan stage 2 validation) will tell whether
  n=100 is research-grade for these models or n must grow.

## References

- `[gurnee2026-workspace §2.1]` — estimator definition;
  `kb/excerpts/gurnee2026-workspace#sec-2-1-jlens-def`
- `anthropics/jacobian-lens` `jlens/fitting.py` (`jacobian_for_prompt`)
- Plan: `research/arcs/04_jspace/plans/2026-07-18-jspace-design.md`
