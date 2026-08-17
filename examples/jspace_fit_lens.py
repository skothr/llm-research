#!/usr/bin/env python3
"""Stage 2 of the jspace arc: fit a Jacobian lens (all layers) on a Qwen model.

Wraps `jlens.fit` (anthropics/jacobian-lens) with this repo's conventions:
the frozen fitting corpus committed at
research/arcs/04_jspace/data/fitting_prompts_wikitext103_n1000.json, atomic
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

A fit that is interrupted and resumed runs as several *segments*. Each one's
duration is appended to a `{stem}.segments.json` ledger beside the checkpoint,
so the sidecar can report total compute rather than just the last segment —
before 2026-07-29 it reported the last segment, which under-reported a paused
1.5B fit by 39% (2.16 h recorded against 3.53 h actually spent). See #40.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

ARC_DATA = Path("research/arcs/04_jspace/data")
DEFAULT_PROMPTS = ARC_DATA / "fitting_prompts_wikitext103_n1000.json"


def read_segments(path: Path) -> list[dict[str, Any]]:
    """Previously-recorded segments of this fit, oldest first.

    Never raises. A missing, truncated, or hand-edited ledger degrades the
    reported total to a lower bound; it must never take down a 16 h fit.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(obj, list):
        return []
    return [
        s
        for s in cast("list[Any]", obj)
        if isinstance(s, dict) and isinstance(s.get("seconds"), (int, float))
    ]


def append_segment(
    path: Path, started: datetime, seconds: float
) -> list[dict[str, Any]]:
    """Append one segment and return the full ledger. Never raises."""
    segments = read_segments(path)
    segments.append(
        {"started": started.isoformat(timespec="seconds"), "seconds": round(seconds, 1)}
    )
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(segments, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        # Losing the ledger costs provenance precision, not the fit.
        print(f"WARN: could not write {path}: {e}", flush=True)
    return segments


def checkpoint_n_done(path: Path) -> int | None:
    """`n_done` from a fit checkpoint without materialising its tensors.

    `mmap=True` reads the metadata in ~0.01 s with no RSS growth, which matters
    because the 7B checkpoint is ~1.4 GB and this runs at startup. Returns None
    if there is no readable checkpoint — i.e. the fit is starting fresh.
    """
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        n = state.get("n_done")
        return int(n) if isinstance(n, (int, float)) else None
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode", default="nf4", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument(
        "--corpus-tag",
        default="",
        help="Suffix appended to the lens stem (e.g. 'c4en' -> "
        "jlens_<model>_<mode>_n<n>_c4en), so a refit on a different fitting "
        "corpus lands in its own artifact instead of clobbering the default "
        "wikitext lens. Empty (default) keeps the original stem for backward "
        "compatibility; the exact prompts file is recorded in the sidecar.",
    )
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
        "holds ~1.09 GB of bf16 that the retained graph needs.",
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
    if args.corpus_tag:
        stem = f"{stem}_{args.corpus_tag}"
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

    seg_path = args.out_dir / f"{stem}.segments.json"
    prior = read_segments(seg_path)
    prior_seconds = sum(float(s["seconds"]) for s in prior)
    n_done = checkpoint_n_done(ckpt_path)

    if n_done is None:
        verb = "FIT start (fresh, no checkpoint)"
    else:
        verb = (
            f"FIT resume from checkpoint: {n_done}/{len(prompts)} prompts done, "
            f"{len(prior)} prior segment(s) totalling {prior_seconds / 3600:.2f} h"
        )
        if not prior:
            # Checkpoint without a ledger: the fit began before the ledger
            # existed, or the ledger was lost. Say so rather than silently
            # reporting a total that excludes everything before now.
            verb += " [NO LEDGER — reported total will be a LOWER BOUND]"
    print(
        f"{verb}: {args.model} mode={args.mode} d_model={model.d_model} "
        f"layers={model.n_layers} dim_batch={args.dim_batch} "
        f"n_prompts={len(prompts)} checkpoint={ckpt_path}",
        flush=True,
    )
    started_at = datetime.now(timezone.utc)
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
    segments = append_segment(seg_path, started_at, dt)
    total_seconds = sum(float(s["seconds"]) for s in segments)
    # A checkpoint that predates its ledger means earlier segments were never
    # recorded, so the total is a floor. Flag it in the artifact rather than
    # letting a reader take an under-count at face value.
    total_is_lower_bound = n_done is not None and not prior

    lens.save(str(lens_path))
    sidecar = {
        "model": args.model,
        "mode": args.mode,
        "device": args.device,
        "dim_batch": args.dim_batch,
        "n_prompts": len(prompts),
        "max_seq_len": args.max_seq_len,
        "prompts_file": str(args.prompts),
        "corpus_tag": args.corpus_tag,
        # Total compute across all segments of this fit, NOT just the last one
        # — a resumed fit has several. `wall_seconds_segments` is the breakdown;
        # a length > 1 means the fit was interrupted and resumed.
        "wall_seconds": round(total_seconds, 1),
        "wall_seconds_segments": [float(s["seconds"]) for s in segments],
        "wall_seconds_is_lower_bound": total_is_lower_bound,
        "jlens_source": "github.com/anthropics/jacobian-lens (editable clone)",
        "estimator": "J_l = mean over prompts of mean-over-source-positions of "
        "sum-over-targets>=source of dh_final/dh_l (jlens.fit defaults, "
        "skip_first=16)",
    }
    (args.out_dir / f"{stem}.config.json").write_text(json.dumps(sidecar, indent=2))
    span = (
        f"{total_seconds / 3600:.2f} h total over {len(segments)} segments "
        f"(this one {dt / 3600:.2f} h)"
        if len(segments) > 1
        else f"{dt / 3600:.2f} h"
    )
    print(
        f"FIT done in {span}{' [LOWER BOUND]' if total_is_lower_bound else ''} "
        f"-> {lens_path} (+ config sidecar); "
        f"layers={list(lens.jacobians)[:3]}...",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
