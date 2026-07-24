#!/usr/bin/env python3
"""Stage 1 gate for the jspace arc: end-to-end gradient probe + timing.

Verifies that activation gradients flow from the final-layer residual stream
back to layer 0 through the nf4-quantized Qwen2.5-7B-Instruct (the QLoRA
path: frozen 4-bit weights, differentiable activations), and measures
per-pass wall time + peak VRAM at the fitting sequence length. These
measurements calibrate the stage-2 runtime estimates in
research/arcs/04_jspace/plans/2026-07-18-jspace-design.md.

Usage:
    python examples/jspace_stage1_grad_probe.py --model Qwen/Qwen2.5-7B-Instruct --mode nf4
    python examples/jspace_stage1_grad_probe.py --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16
    python examples/jspace_stage1_grad_probe.py --model Qwen/Qwen2.5-7B-Instruct --mode bf16 --device cpu

Prints a PROBE: PASS/FAIL line plus timing rows; exits nonzero on failure.
"""

import argparse
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode", default="nf4", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--timing-reps", type=int, default=3)
    return p.parse_args()


def load(args: argparse.Namespace):
    from llm_surgeon import surgery

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    model, tok = surgery.load_model(args.model, mode=args.mode, device_map=device_map)
    model.eval()
    # Freeze everything: we probe activation gradients only. Without this,
    # backward materializes a grad for the bf16 embedding matrix (~1 GiB on
    # Qwen2.5-7B) and OOMs the 2080. jlens.from_hf does the same freeze.
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def one_pass(model, tok, args, seq_len: int):
    """One forward with hidden states + one VJP backward from h_final.

    Returns (fwd_seconds, bwd_seconds, grad_l0) where grad_l0 is the gradient
    at the layer-0 residual stream (hidden_states[1], i.e. after block 0).
    """
    text = "The study of hidden representations in large language models "
    ids = tok(text, return_tensors="pt").input_ids
    reps = (seq_len // ids.shape[1]) + 1
    ids = ids.repeat(1, reps)[:, :seq_len].to(next(model.parameters()).device)

    t0 = time.perf_counter()
    out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: embeddings + one per block
    for h in hs:
        h.retain_grad()
    if args.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    h_final = hs[-1]
    probe = torch.randn_like(h_final[0, -1])  # VJP seed at last position
    h_final[0, -1].backward(probe)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    return t1 - t0, t2 - t1, hs[1].grad


def main() -> int:
    args = parse_args()
    model, tok = load(args)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Correctness gate at a short length first (cheap fail).
    _, _, g0 = one_pass(model, tok, args, seq_len=32)
    ok = g0 is not None and torch.isfinite(g0).all() and g0.abs().sum() > 0
    print(
        f"PROBE: {'PASS' if ok else 'FAIL'} — layer-0 residual grad "
        f"{'nonzero+finite' if ok else 'missing/zero/nonfinite'} "
        f"(|g|_1={g0.abs().sum().item():.3e})"
        if g0 is not None
        else "PROBE: FAIL — no gradient reached layer 0"
    )
    if not ok:
        return 1

    # Timing at the fitting sequence length.
    rows = []
    for _ in range(args.timing_reps):
        model.zero_grad(set_to_none=True)
        rows.append(one_pass(model, tok, args, seq_len=args.seq_len)[:2])
    fwd = min(r[0] for r in rows)
    bwd = min(r[1] for r in rows)
    print(
        f"TIMING model={args.model} mode={args.mode} device={args.device} "
        f"seq={args.seq_len}: fwd={fwd:.3f}s bwd={bwd:.3f}s "
        f"fwd+bwd={fwd + bwd:.3f}s (best of {args.timing_reps})"
    )
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"VRAM peak={peak:.2f} GiB of {total:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
