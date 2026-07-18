#!/usr/bin/env python3
"""Stage 1 calibration for the jspace arc: bounded J-lens fitting measurement.

One prompt of `jlens.fit` costs one batched forward (batch=dim_batch) plus
ceil(d_model/dim_batch) VJP backward passes over a retained graph
(jlens/fitting.py:jacobian_for_prompt). At 7B that is ~450-1800 backwards —
too long to time blind. This script mirrors that code path exactly
(ActivationRecorder + one-hot cotangents + torch.autograd.grad) but runs only
--measure-passes backwards, then extrapolates:

    t_prompt ~= t_forward + n_passes * t_backward_avg

It also serves as the correctness gate: the first backward's layer-0 gradient
must be nonzero and finite (grad flow through the nf4 QLoRA path).

Usage:
    python examples/jspace_stage1_fit_calibration.py \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --dim-batch 8 4 2
    python examples/jspace_stage1_fit_calibration.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16 --dim-batch 32 8
    python examples/jspace_stage1_fit_calibration.py ... --device cpu  # CPU row

Per dim_batch prints either a CALIB row (measured s/pass, extrapolated
s/prompt, VRAM peak, projected hours at n=100/1000) or an OOM row.
"""

import argparse
import sys
import time
from typing import cast

import torch

PROMPT = (
    "The history of astronomy is a history of receding horizons. Early "
    "observers charted the wandering planets against fixed stars, and each "
    "improvement of the telescope revealed structure where smoothness had "
    "been assumed: mountains on the Moon, moons around Jupiter, spiral arms "
    "in what had been called nebulae. The instruments changed what could be "
    "asked. Spectroscopy turned starlight into chemistry, and photography "
    "turned patient nights into permanent records that could be measured, "
    "compared, and disputed by people who had never looked through the "
    "eyepiece at all. By the twentieth century the questions had shifted "
    "from cataloguing positions to explaining origins, and the universe "
    "itself acquired a history with a beginning, an expansion, and an "
    "uncertain end."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode", default="nf4", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dim-batch", type=int, nargs="+", default=[8, 4, 2])
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--measure-passes", type=int, default=8)
    p.add_argument(
        "--offload-lm-head",
        action="store_true",
        help="Move lm_head to CPU after wrapping. Fitting never touches it "
        "(jlens forward runs the residual stack only) and on Qwen2.5-7B it "
        "holds ~1.09 GiB of bf16 that the retained graph needs.",
    )
    return p.parse_args()


def sync(args: argparse.Namespace) -> None:
    if args.device == "cuda":
        torch.cuda.synchronize()


def measure(model, args: argparse.Namespace, dim_batch: int) -> None:
    """Mirror of jlens.fitting.jacobian_for_prompt, bounded to
    --measure-passes backward passes."""
    from jlens.fitting import valid_position_mask
    from jlens.hooks import ActivationRecorder

    n_layers, d_model = model.n_layers, model.d_model
    source_layers = list(range(n_layers - 1))
    target_layer = n_layers - 1
    n_passes_full = -(-d_model // dim_batch)
    n_measure = min(args.measure_passes, n_passes_full)

    input_ids = model.encode(PROMPT, max_length=args.max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=0,
        ) as recorder,
        torch.enable_grad(),
    ):
        t0 = time.perf_counter()
        model.forward(input_ids.expand(dim_batch, -1))
        sync(args)
        t_fwd = time.perf_counter() - t0

        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[i] for i in source_layers]
        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        grad_ok = False
        t1 = time.perf_counter()
        for pass_idx in range(n_measure):
            dim_start = pass_idx * dim_batch
            cotangent.zero_()
            cotangent[
                batch_indices[:, None],
                valid_positions[None, :],
                dim_start + batch_indices[:, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=True,
            )
            if pass_idx == 0:
                g0 = grads[0]
                grad_ok = bool(torch.isfinite(g0).all()) and float(g0.abs().sum()) > 0.0
            del grads
        sync(args)
        t_bwd = (time.perf_counter() - t1) / n_measure

    t_prompt = t_fwd + n_passes_full * t_bwd
    peak = (
        torch.cuda.max_memory_allocated() / 2**30
        if args.device == "cuda"
        else float("nan")
    )
    print(
        f"CALIB dim_batch={dim_batch}: grad_ok={grad_ok} fwd={t_fwd:.2f}s "
        f"bwd={t_bwd:.2f}s/pass x {n_passes_full} passes "
        f"(measured {n_measure}) -> {t_prompt:.0f} s/prompt, "
        f"peak={peak:.2f} GiB | n=100 -> {t_prompt * 100 / 3600:.1f} h, "
        f"n=1000 -> {t_prompt * 1000 / 3600:.1f} h",
        flush=True,
    )


def main() -> int:
    args = parse_args()

    import jlens
    from jlens.protocol import LensModel
    from llm_surgeon import surgery

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    hf_model, tok = surgery.load_model(
        args.model, mode=args.mode, device_map=device_map
    )
    # cast: HFLensModel satisfies the LensModel protocol at runtime, but its
    # `layers: nn.ModuleList` fails the protocol's invariant
    # `layers: Sequence[nn.Module]` attribute check (upstream typing looseness
    # in the reference repo).
    model = cast(LensModel, jlens.from_hf(hf_model, tok))
    if args.offload_lm_head and args.device == "cuda":
        getattr(hf_model, "lm_head").to("cpu")
        torch.cuda.empty_cache()
        print("OFFLOAD: lm_head -> cpu", flush=True)

    ids = tok(PROMPT[:64]).input_ids
    bos = tok.bos_token_id
    print(
        f"BOS: bos_token_id={bos} first_input_id={ids[0]} "
        f"prepended={bos is not None and ids[0] == bos}",
        flush=True,
    )
    print(
        f"MODEL: {args.model} mode={args.mode} d_model={model.d_model} "
        f"layers={model.n_layers}",
        flush=True,
    )

    for db in args.dim_batch:
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            measure(model, args, db)
        except torch.OutOfMemoryError:
            print(f"CALIB dim_batch={db}: OOM", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
