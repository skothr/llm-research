#!/usr/bin/env python3
"""Stage 2 of the jspace arc: fit a Jacobian lens (all layers) on a Qwen model.

Wraps `jlens.fit` (anthropics/jacobian-lens) with this repo's conventions:
the frozen fitting corpus committed at
research/arcs/jspace/data/fitting_prompts_wikitext103_n1000.json, atomic
checkpointing for overnight resumability, and a config sidecar recording
every knob so the audit can tie the lens artifact to its provenance.

Usage:
    python examples/jspace_fit_lens.py --model Qwen/Qwen2.5-7B-Instruct \
        --mode nf4 --dim-batch 4 --n-prompts 1000
    python examples/jspace_fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 --dim-batch 32 --n-prompts 1000

The saved lens (fp16, jlens native format) lands in the arc cache dir with a
`.config.json` sidecar. Re-running with the same arguments resumes from the
checkpoint. Progress lines go to stdout (one per checkpoint interval).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import torch

ARC_DATA = Path("research/arcs/jspace/data")
DEFAULT_PROMPTS = ARC_DATA / "fitting_prompts_wikitext103_n1000.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode", default="nf4", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Prompts between checkpoint writes. Each write is "
        "n_layers*d_model^2*4 bytes (~1.4 GB at 7B full depth), so keep "
        "this well above the default of 1.",
    )
    p.add_argument("--out-dir", type=Path, default=ARC_DATA / "cache")
    p.add_argument(
        "--offload-lm-head",
        action="store_true",
        help="Move lm_head to CPU after wrapping. Fitting never touches it "
        "(jlens forward runs the residual stack only) and on Qwen2.5-7B it "
        "holds ~1.09 GiB of bf16 that the retained graph needs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import jlens
    from jlens.protocol import LensModel
    from llm_surgeon import surgery

    corpus = json.loads(args.prompts.read_text())
    prompts = corpus["prompts"][: args.n_prompts]
    if len(prompts) < args.n_prompts:
        print(f"WARNING: corpus has {len(prompts)} prompts, requested {args.n_prompts}")

    model_short = args.model.split("/")[-1].lower().replace("-instruct", "")
    stem = f"jlens_{model_short}_{args.mode}_n{args.n_prompts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = args.out_dir / f"{stem}.pt"
    ckpt_path = args.out_dir / f"{stem}.ckpt.pt"

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    hf_model, tok = surgery.load_model(
        args.model, mode=args.mode, device_map=device_map
    )
    # cast: HFLensModel satisfies LensModel at runtime; its
    # `layers: nn.ModuleList` fails the protocol's invariant Sequence check
    # (upstream typing looseness in the reference repo).
    model = cast(LensModel, jlens.from_hf(hf_model, tok))
    if args.offload_lm_head and args.device == "cuda":
        getattr(hf_model, "lm_head").to("cpu")
        torch.cuda.empty_cache()
        print("OFFLOAD: lm_head -> cpu", flush=True)

    print(
        f"FIT start: {args.model} mode={args.mode} d_model={model.d_model} "
        f"layers={model.n_layers} dim_batch={args.dim_batch} "
        f"n_prompts={len(prompts)} checkpoint={ckpt_path}",
        flush=True,
    )
    t0 = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    dt = time.perf_counter() - t0

    lens.save(str(lens_path))
    sidecar = {
        "model": args.model,
        "mode": args.mode,
        "device": args.device,
        "dim_batch": args.dim_batch,
        "n_prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "prompts_file": str(args.prompts),
        "wall_seconds": round(dt, 1),
        "jlens_source": "github.com/anthropics/jacobian-lens (editable clone)",
        "estimator": "J_l = mean over prompts of mean-over-source-positions of "
        "sum-over-targets>=source of dh_final/dh_l (jlens.fit defaults, "
        "skip_first=16)",
    }
    (args.out_dir / f"{stem}.config.json").write_text(json.dumps(sidecar, indent=2))
    print(
        f"FIT done in {dt / 3600:.2f} h -> {lens_path} "
        f"(+ config sidecar); layers={list(lens.jacobians)[:3]}...",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
