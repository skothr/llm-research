#!/usr/bin/env python3
"""Stage 1 calibration for the jspace arc: time one real J-lens fitting unit.

`jlens.jacobian_for_prompt` is the per-prompt unit of `jlens.fit`: one batched
forward (batch=dim_batch) + ceil(d_model/dim_batch) VJP backwards over a
retained graph. Timing it directly gives the exact per-prompt fitting cost on
this hardware — no FLOP modeling — from which stage-2 wall-clock at
n_prompts={100, 1000} follows by multiplication.

Usage:
    python examples/jspace_stage1_fit_calibration.py \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --dim-batch 8 16
    python examples/jspace_stage1_fit_calibration.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16 --dim-batch 8 32

Prints one CALIB row per dim_batch (seconds/prompt, VRAM peak, extrapolated
hours for n=100 and n=1000) and a BOS diagnostic row. OOM at a given
dim_batch prints an OOM row and continues with the next value.
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
    p.add_argument("--dim-batch", type=int, nargs="+", default=[8])
    p.add_argument("--max-seq-len", type=int, default=128)
    return p.parse_args()


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

    # BOS diagnostic: jlens sets add_bos_token=True, but Qwen2.5 tokenizers
    # have no BOS; skip_first=16 assumes sink-like early positions either way.
    ids = tok(PROMPT[:64]).input_ids
    bos = tok.bos_token_id
    print(
        f"BOS: bos_token_id={bos} "
        f"first_input_id={ids[0]} prepended={bos is not None and ids[0] == bos}"
    )

    d_model = hf_model.config.hidden_size
    n_layers = hf_model.config.num_hidden_layers
    print(f"MODEL: {args.model} mode={args.mode} d_model={d_model} layers={n_layers}")

    for db in args.dim_batch:
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            _, seq_len, n_valid = jlens.jacobian_for_prompt(
                model,
                PROMPT,
                list(range(n_layers - 1)),
                dim_batch=db,
                max_seq_len=args.max_seq_len,
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError:
            print(f"CALIB dim_batch={db}: OOM")
            continue
        peak = (
            torch.cuda.max_memory_allocated() / 2**30
            if args.device == "cuda"
            else float("nan")
        )
        n_bwd = -(-d_model // db)
        print(
            f"CALIB dim_batch={db}: {dt:.1f} s/prompt "
            f"({n_bwd} bwd passes, seq={seq_len}, valid={n_valid}) "
            f"peak={peak:.2f} GiB | n=100 -> {dt * 100 / 3600:.1f} h, "
            f"n=1000 -> {dt * 1000 / 3600:.1f} h"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
