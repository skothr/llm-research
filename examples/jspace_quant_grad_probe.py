#!/usr/bin/env python3
"""Quantization gradient-fidelity probe for the jspace arc.

Isolates the *quantization* axis of the J-space structure question: loads the
SAME model (default Qwen2.5-1.5B-Instruct) twice — once bf16, once nf4 — and,
for a handful of fitting-corpus prompts, computes a *matched* set of exact
vector-Jacobian-product (VJP) rows in both precisions. "Matched" means the same
seeded random probe cotangents applied at the same target positions on the same
token ids, so the only thing that differs between the two runs is bf16->nf4.

For each source layer l we report, aggregated over prompts x probes:
  * cosine similarity between the bf16 and nf4 VJP vector (direction fidelity)
  * relative L2 norm error  ||w - u|| / ||u||  (magnitude fidelity)

This measures the raw quantization perturbation to the Jacobian directly, before
any lens fit, so a downstream shift in J-space variance structure can be
attributed to (or exonerated from) quantization noise.

VJP definition (mirrors jlens.jacobian_for_prompt's averaged estimator): for a
seeded unit cotangent v placed at every valid target position of the final
block's output, the recorded gradient at source layer l, averaged over valid
source positions, is (mean_p J_l(p))^T v — one d_model vector per (probe, layer).
Valid positions use jlens's skip_first=16 rule.

Usage:
    python examples/jspace_quant_grad_probe.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --n-prompts 5 --n-probes 8

Sequential load (bf16 then nf4) keeps peak VRAM at one model at a time. Prints a
per-layer table and saves a small artifact under the arc cache with a distinct
stem (quant_grad_probe_<model>_n<p>x<r>.pt) that does not collide with any lens.
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
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument(
        "--n-probes",
        type=int,
        default=8,
        help="Seeded random VJP directions per prompt.",
    )
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--skip-first", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--out-dir", type=Path, default=ARC_DATA / "cache")
    return p.parse_args()


def compute_vjp_rows(
    model_name: str,
    mode: str,
    prompts: list[str],
    probes: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[list[torch.Tensor], list[list[int]], int, int, torch.Tensor]:
    """Load `model_name` in `mode` and return per-prompt VJP tensors.

    `probes` are the seeded unit cotangent directions [n_probes, d_model]; pass
    None on the first (bf16) call to generate them from `args.seed` once the
    model's `d_model` is known (avoids a network/AutoConfig dependency), then
    pass the returned tensor to the second (nf4) call so both precisions use
    byte-identical probes.

    Returns (rows, valid_counts, n_layers, d_model, probes) where rows[i] is a
    [n_probes, n_layers, d_model] fp32 CPU tensor for prompt i (source layers
    0..n_layers-1, target = final block output).
    """
    import jlens
    from jlens.fitting import valid_position_mask
    from jlens.protocol import LensModel
    from llm_surgeon import surgery

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    hf_model, tok = surgery.load_model(model_name, mode=mode, device_map=device_map)
    hf_model.eval()
    for prm in hf_model.parameters():
        prm.requires_grad_(False)
    model = cast(LensModel, jlens.from_hf(hf_model, tok))
    n_layers, d_model = model.n_layers, model.d_model
    if probes is None:
        gen = torch.Generator().manual_seed(args.seed)
        raw = torch.randn(args.n_probes, d_model, generator=gen)
        probes_t = raw / raw.norm(dim=1, keepdim=True)  # unit directions
    else:
        probes_t = probes
    source_layers = list(range(n_layers))
    target_layer = n_layers - 1  # final block output

    dev = next(hf_model.parameters()).device
    probes_dev = probes_t.to(dev)  # [n_probes, d_model], identical bytes across modes

    rows: list[torch.Tensor] = []
    valid_counts: list[list[int]] = []
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=args.max_seq_len)
        seq_len = input_ids.shape[1]
        pmask = valid_position_mask(seq_len, skip_first=args.skip_first)
        valid_pos = pmask.nonzero(as_tuple=True)[0].to(dev)
        n_valid = int(pmask.sum())

        prompt_rows = torch.zeros(args.n_probes, n_layers, d_model, dtype=torch.float32)
        with (
            jlens.ActivationRecorder(
                model.layers,
                at=[*source_layers, target_layer],
                start_graph_at=min(source_layers),
            ) as rec,
            torch.enable_grad(),
        ):
            model.forward(input_ids)
            target_act = rec.activations[target_layer]  # [1, seq, d_model]
            source_acts = [rec.activations[l] for l in source_layers]
            for r in range(args.n_probes):
                # Same probe vector v_r placed at every valid target position.
                cot = torch.zeros_like(target_act)
                cot[0, valid_pos, :] = probes_dev[r]
                grads = torch.autograd.grad(
                    outputs=target_act,
                    inputs=source_acts,
                    grad_outputs=cot,
                    retain_graph=(r < args.n_probes - 1),
                )
                for li, g in enumerate(grads):
                    vpos = valid_pos.to(g.device)
                    vec = g[0, vpos, :].float().mean(dim=0)  # [d_model]
                    prompt_rows[r, li] = vec.cpu()
                del grads
        rows.append(prompt_rows)
        valid_counts.append([seq_len, n_valid])

    # Free the model before the caller loads the next precision.
    del model, hf_model, tok
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return rows, valid_counts, n_layers, d_model, probes_t


def main() -> int:
    args = parse_args()
    corpus = json.loads(args.prompts.read_text())
    prompts = corpus["prompts"][: args.n_prompts]
    if len(prompts) < args.n_prompts:
        print(f"WARNING: corpus has {len(prompts)} prompts, requested {args.n_prompts}")

    # Probes are generated inside the first (bf16) load, once model.d_model is
    # known, from args.seed — no network/AutoConfig dependency (which fails
    # under HF_HUB_OFFLINE when the config isn't in the hub cache), and the same
    # tensor is threaded into the nf4 call so both precisions share it exactly.
    print(
        f"PROBE-FIT config: model={args.model} "
        f"n_prompts={len(prompts)} n_probes={args.n_probes} "
        f"max_seq_len={args.max_seq_len} skip_first={args.skip_first} seed={args.seed}",
        flush=True,
    )

    t0 = time.perf_counter()
    print("LOAD bf16 ...", flush=True)
    rows_bf16, vc, n_layers, d_model, probes = compute_vjp_rows(
        args.model, "bf16", prompts, None, args
    )
    t_bf16 = time.perf_counter()
    print(f"  bf16 done ({t_bf16 - t0:.1f}s, d_model={d_model})", flush=True)

    print("LOAD nf4 ...", flush=True)
    rows_nf4, vc2, n_layers2, dm2, _ = compute_vjp_rows(
        args.model, "nf4", prompts, probes, args
    )
    t_nf4 = time.perf_counter()
    print(f"  nf4 done ({t_nf4 - t_bf16:.1f}s)", flush=True)
    assert n_layers == n_layers2 and dm2 == d_model, (
        f"shape mismatch bf16 (L={n_layers}, d={d_model}) vs nf4 (L={n_layers2}, d={dm2})"
    )

    # Aggregate cosine + relative-norm error per layer over prompts x probes.
    eps = 1e-12
    cos_acc = torch.zeros(n_layers, dtype=torch.float64)
    rel_acc = torch.zeros(n_layers, dtype=torch.float64)
    bf16_norm_acc = torch.zeros(n_layers, dtype=torch.float64)
    nf4_norm_acc = torch.zeros(n_layers, dtype=torch.float64)
    count = 0
    for pr_bf, pr_nf in zip(rows_bf16, rows_nf4, strict=True):
        for r in range(args.n_probes):
            u = pr_bf[r]  # [n_layers, d_model]
            w = pr_nf[r]
            un = u.norm(dim=1)
            wn = w.norm(dim=1)
            dot = (u * w).sum(dim=1)
            cos = dot / (un * wn + eps)
            rel = (w - u).norm(dim=1) / (un + eps)
            cos_acc += cos.double()
            rel_acc += rel.double()
            bf16_norm_acc += un.double()
            nf4_norm_acc += wn.double()
            count += 1

    cos_mean = (cos_acc / count).tolist()
    rel_mean = (rel_acc / count).tolist()
    bf16_norm = (bf16_norm_acc / count).tolist()
    nf4_norm = (nf4_norm_acc / count).tolist()

    print("\nPER-LAYER VJP FIDELITY (bf16 vs nf4), mean over prompts x probes")
    print(
        f"{'layer':>5} {'cos':>9} {'rel_norm_err':>13} "
        f"{'|bf16|':>10} {'|nf4|':>10} {'norm_ratio':>11}"
    )
    for l in range(n_layers):
        ratio = nf4_norm[l] / (bf16_norm[l] + eps)
        print(
            f"{l:>5} {cos_mean[l]:>9.4f} {rel_mean[l]:>13.4f} "
            f"{bf16_norm[l]:>10.3e} {nf4_norm[l]:>10.3e} {ratio:>11.4f}"
        )

    # Summary stats over non-degenerate source layers (exclude the final layer,
    # which is target==source => identity => trivial cos 1).
    body = slice(0, n_layers - 1)
    cos_body = torch.tensor(cos_mean[body])
    rel_body = torch.tensor(rel_mean[body])
    print(
        f"\nSUMMARY (layers 0..{n_layers - 2}): "
        f"cos mean={cos_body.mean():.4f} min={cos_body.min():.4f} "
        f"(layer {int(cos_body.argmin())}); "
        f"rel_norm_err mean={rel_body.mean():.4f} max={rel_body.max():.4f} "
        f"(layer {int(rel_body.argmax())})",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-instruct", "")
    stem = f"quant_grad_probe_{model_short}_n{len(prompts)}x{args.n_probes}"
    out_path = args.out_dir / f"{stem}.pt"
    torch.save(
        {
            "model": args.model,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_prompts": len(prompts),
            "n_probes": args.n_probes,
            "max_seq_len": args.max_seq_len,
            "skip_first": args.skip_first,
            "seed": args.seed,
            "prompts_file": str(args.prompts),
            "valid_counts_bf16": vc,
            "valid_counts_nf4": vc2,
            "cos_per_layer": cos_mean,
            "rel_norm_err_per_layer": rel_mean,
            "bf16_norm_per_layer": bf16_norm,
            "nf4_norm_per_layer": nf4_norm,
            "wall_seconds": round(t_nf4 - t0, 1),
        },
        out_path,
    )
    print(f"\nSAVED {out_path} (wall {t_nf4 - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
