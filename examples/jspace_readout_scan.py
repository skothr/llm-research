#!/usr/bin/env python3
"""Stage 3 of the jspace arc: J-lens readout characterization.

Reads out a fitted :class:`jlens.JacobianLens` across every source layer on
held-out prompts and quantifies, per layer, how faithfully the readout tracks
the model's own final next-token distribution — for both the Jacobian lens
(``use_jacobian=True``) and the vanilla logit-lens baseline
(``use_jacobian=False``). Two metrics, evaluated at the last token position and
at 8 evenly spaced mid positions:

  * Spearman rank correlation between the per-layer lens readout logits and the
    model's final-layer logits at the same position;
  * the rank of the model's final top-1 token inside each lens readout
    (the depth-of-emergence curve — how early the answer surfaces).

Tests the paper's core readability claim (§2.1, [gurnee2026-workspace]): the
Jacobian readout should be interpretable at earlier layers than the logit lens.

Held-out prompts default to
``research/arcs/jspace/data/heldout_prompts_wikitext103_n30.json`` — wikitext-103
records index 1001-1030 under the same len>=600 filter as the fitting corpus,
strictly disjoint from the first-1000 fitting set. No network access at run time.

Usage:
    python examples/jspace_readout_scan.py --n-prompts 12
    python examples/jspace_readout_scan.py --n-prompts 12 --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26
    python examples/jspace_readout_scan.py --qualitative 2   # top-5 token trajectories

Run CPU-only via CUDA_VISIBLE_DEVICES="" when a GPU job holds the device.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

TOP10 = 10
QUAL_TOPK = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt",
        help="Path to a fitted JacobianLens .pt (JacobianLens.save format).",
    )
    p.add_argument(
        "--prompts",
        default="research/arcs/jspace/data/heldout_prompts_wikitext103_n30.json",
        help="Held-out prompts JSON ({'prompts': [str, ...]}).",
    )
    p.add_argument("--n-prompts", type=int, default=12)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--n-mid", type=int, default=8, help="Evenly spaced mid positions.")
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated source layers (default: all fitted source layers).",
    )
    p.add_argument(
        "--qualitative",
        type=int,
        default=2,
        help="Print top-5 per-layer token trajectories at the final position for "
        "this many prompts (0 to skip).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .pt path (default: cache/readout_scan_<model>_<lensstem>.pt).",
    )
    return p.parse_args()


def slug(model: str) -> str:
    return model.split("/")[-1].lower()


def default_out(model: str, lens: str) -> str:
    stem = Path(lens).stem
    return f"research/arcs/jspace/data/cache/readout_scan_{slug(model)}_{stem}.pt"


def spearman(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation = Pearson on the rank-transformed vectors.

    Ranks via a double ``argsort`` (position of each element in sorted order);
    ties are broken by index, which is acceptable for float logits where exact
    ties are rare. No scipy dependency.
    """
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def token_rank(readout: Tensor, token_id: int) -> int:
    """0-based rank of ``token_id`` in ``readout`` (0 == top-1)."""
    return int((readout > readout[token_id]).sum().item())


def mid_and_last_positions(seq_len: int, n_mid: int) -> tuple[list[int], int]:
    """8 evenly spaced interior positions plus the last position.

    Returns ``(positions, last_index_in_positions)``. Positions are sorted and
    de-duplicated; the last token (``seq_len - 1``) is always included and its
    index within the returned list is reported separately so callers can single
    out the true next-token prediction.
    """
    if seq_len <= 1:
        return [seq_len - 1], 0
    mids = [int(x) for x in np.linspace(1, seq_len - 2, n_mid)]
    positions = sorted(set(mids) | {seq_len - 1})
    return positions, positions.index(seq_len - 1)


def load_prompts(path: str, n: int) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    if not isinstance(prompts, list):
        raise ValueError(f"{path}: expected a list of prompts")
    return [str(p) for p in prompts[:n]]


def build_model(args: argparse.Namespace) -> Any:
    from llm_surgeon import surgery

    from jlens import from_hf
    from jlens.protocol import LensModel

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    model, tok = surgery.load_model(args.model, mode=args.mode, device_map=device_map)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # HFLensModel.layers is an nn.ModuleList (fails the protocol's invariant
    # Sequence attribute check); cast is the sanctioned narrowing here.
    return cast(LensModel, from_hf(model, tok))


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    from jlens import JacobianLens

    t_load = time.time()
    m = build_model(args)
    lens = JacobianLens.load(args.lens)
    print(f"[load] model + lens in {time.time() - t_load:.1f}s; {lens!r}")

    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        unknown = sorted(set(layers) - set(lens.source_layers))
        if unknown:
            raise SystemExit(
                f"--layers {unknown} not in source_layers {lens.source_layers}"
            )
    else:
        layers = list(lens.source_layers)
    n_layers = len(layers)

    prompts = load_prompts(args.prompts, args.n_prompts)
    print(f"[data] {len(prompts)} held-out prompts from {args.prompts}")
    print(
        f"[scan] {n_layers} layers x 2 lens modes (positions: {args.n_mid} mid + last)"
    )

    # Per-prompt / per-layer / per-position raw arrays. Positions vary in count
    # per prompt only when a prompt is degenerately short; store as lists.
    per_prompt: list[dict[str, Any]] = []
    # Accumulators for the per-layer summary, keyed by layer array index.
    spear_j = np.full((len(prompts), n_layers), np.nan)  # last-position Spearman
    spear_l = np.full((len(prompts), n_layers), np.nan)
    spear_j_all = np.full((len(prompts), n_layers), np.nan)  # mean over all positions
    spear_l_all = np.full((len(prompts), n_layers), np.nan)
    # Depth-of-emergence: shallowest layer (in `layers` order) where model top-1
    # enters lens top-10, per (prompt, position). NaN if never.
    emerge_j: list[float] = []
    emerge_l: list[float] = []
    emerge_j_last: list[float] = []
    emerge_l_last: list[float] = []

    per_prompt_times: list[float] = []
    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        seq_len = int(m.encode(prompt, max_length=args.max_seq_len).shape[1])
        positions, last_idx = mid_and_last_positions(seq_len, args.n_mid)
        n_pos = len(positions)

        ll_j, model_logits, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=args.max_seq_len,
            use_jacobian=True,
        )
        ll_l, _, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=args.max_seq_len,
            use_jacobian=False,
        )
        top1 = [int(model_logits[pp].argmax().item()) for pp in range(n_pos)]

        # --- Rich per-item capture (additive; new per_prompt keys only) -------
        # Per (layer, position) top-k=10 token ids, decoded strings and
        # probabilities for BOTH lenses, plus the model's own final-layer
        # reference distribution per position. Enables the "unspoken words"
        # layer x token trajectory figure without a re-scan.
        tok = m.tokenizer
        topk_ids_j = np.full((n_layers, n_pos, TOP10), -1, dtype=np.int64)
        topk_ids_l = np.full((n_layers, n_pos, TOP10), -1, dtype=np.int64)
        topk_probs_j = np.full((n_layers, n_pos, TOP10), np.nan)
        topk_probs_l = np.full((n_layers, n_pos, TOP10), np.nan)
        topk_strs_j: list[list[list[str]]] = []
        topk_strs_l: list[list[list[str]]] = []
        # Model reference distribution (final-layer logits) per position.
        probs_m = torch.softmax(model_logits.to(torch.float32), dim=-1)
        m_pv, m_pi = probs_m.topk(TOP10, dim=-1)
        topk_ids_model = m_pi.to(torch.int64).cpu().numpy()
        topk_probs_model = m_pv.to(torch.float64).cpu().numpy()
        topk_strs_model = [
            [tok.decode([int(t)]) for t in m_pi[pp]] for pp in range(n_pos)
        ]

        # rank[metric] shape [n_layers, n_pos]
        rank_j = np.full((n_layers, n_pos), np.nan)
        rank_l = np.full((n_layers, n_pos), np.nan)
        sp_j = np.full((n_layers, n_pos), np.nan)
        sp_l = np.full((n_layers, n_pos), np.nan)
        for li, layer in enumerate(layers):
            rj = ll_j[layer]
            rl = ll_l[layer]
            # Top-k=10 ids / probs / strings for this layer, both lenses.
            pj_v, pj_i = torch.softmax(rj.to(torch.float32), dim=-1).topk(TOP10, dim=-1)
            pl_v, pl_i = torch.softmax(rl.to(torch.float32), dim=-1).topk(TOP10, dim=-1)
            topk_ids_j[li] = pj_i.to(torch.int64).cpu().numpy()
            topk_ids_l[li] = pl_i.to(torch.int64).cpu().numpy()
            topk_probs_j[li] = pj_v.to(torch.float64).cpu().numpy()
            topk_probs_l[li] = pl_v.to(torch.float64).cpu().numpy()
            topk_strs_j.append(
                [[tok.decode([int(t)]) for t in pj_i[pp]] for pp in range(n_pos)]
            )
            topk_strs_l.append(
                [[tok.decode([int(t)]) for t in pl_i[pp]] for pp in range(n_pos)]
            )
            for pp in range(n_pos):
                sp_j[li, pp] = spearman(rj[pp], model_logits[pp])
                sp_l[li, pp] = spearman(rl[pp], model_logits[pp])
                rank_j[li, pp] = token_rank(rj[pp], top1[pp])
                rank_l[li, pp] = token_rank(rl[pp], top1[pp])

        spear_j[pi] = sp_j[:, last_idx]
        spear_l[pi] = sp_l[:, last_idx]
        spear_j_all[pi] = np.nanmean(sp_j, axis=1)
        spear_l_all[pi] = np.nanmean(sp_l, axis=1)

        # Emergence layer per position (index into `layers`, then map to real id).
        for pp in range(n_pos):
            for arr, dst in ((rank_j, emerge_j), (rank_l, emerge_l)):
                hit = np.where(arr[:, pp] < TOP10)[0]
                val = float(layers[int(hit[0])]) if hit.size else float("nan")
                dst.append(val)
            for arr, dst in ((rank_j, emerge_j_last), (rank_l, emerge_l_last)):
                if pp == last_idx:
                    hit = np.where(arr[:, pp] < TOP10)[0]
                    dst.append(float(layers[int(hit[0])]) if hit.size else float("nan"))

        per_prompt.append(
            {
                "seq_len": seq_len,
                "positions": positions,
                "last_idx": last_idx,
                "top1": top1,
                "spearman_j": sp_j,
                "spearman_l": sp_l,
                "rank_j": rank_j,
                "rank_l": rank_l,
                # Rich per-item capture (top-k=10; [n_layers, n_pos, 10] for the
                # lenses, [n_pos, 10] for the model reference). Strings nested as
                # [layer][pos][k] (lenses) / [pos][k] (model).
                "topk_ids_j": topk_ids_j,
                "topk_ids_l": topk_ids_l,
                "topk_probs_j": topk_probs_j,
                "topk_probs_l": topk_probs_l,
                "topk_strs_j": topk_strs_j,
                "topk_strs_l": topk_strs_l,
                "topk_ids_model": topk_ids_model,
                "topk_probs_model": topk_probs_model,
                "topk_strs_model": topk_strs_model,
            }
        )
        dt = time.time() - t0
        per_prompt_times.append(dt)
        print(f"[prompt {pi + 1}/{len(prompts)}] seq={seq_len} pos={n_pos} {dt:.1f}s")

    layer_mean_j = np.nanmean(spear_j, axis=0)
    layer_mean_l = np.nanmean(spear_l, axis=0)
    layer_mean_j_all = np.nanmean(spear_j_all, axis=0)
    layer_mean_l_all = np.nanmean(spear_l_all, axis=0)

    def med(xs: list[float]) -> float:
        arr = np.array([x for x in xs if not np.isnan(x)])
        return float(np.median(arr)) if arr.size else float("nan")

    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "n_prompts": len(prompts),
        "layers": layers,
        "n_mid_positions": args.n_mid,
        "layer_mean_spearman_j_last": layer_mean_j.tolist(),
        "layer_mean_spearman_l_last": layer_mean_l.tolist(),
        "layer_mean_spearman_j_allpos": layer_mean_j_all.tolist(),
        "layer_mean_spearman_l_allpos": layer_mean_l_all.tolist(),
        "depth_of_emergence_top10": {
            "median_layer_j_allpos": med(emerge_j),
            "median_layer_l_allpos": med(emerge_l),
            "median_layer_j_lastpos": med(emerge_j_last),
            "median_layer_l_lastpos": med(emerge_l_last),
            "n_never_emerged_j_allpos": int(sum(np.isnan(x) for x in emerge_j)),
            "n_never_emerged_l_allpos": int(sum(np.isnan(x) for x in emerge_l)),
            "n_samples_allpos": len(emerge_j),
        },
        "mean_seconds_per_prompt": float(np.mean(per_prompt_times)),
    }

    out = args.out or default_out(args.model, args.lens)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_prompt": per_prompt, "summary": summary}, out)

    # Compact per-layer summary table (mean Spearman at the last position).
    print("\n=== Per-layer mean Spearman (last position; J-lens vs logit-lens) ===")
    print(f"{'layer':>5} {'J-lens':>9} {'logit':>9} {'J-logit':>9}")
    for li, layer in enumerate(layers):
        j, ll_ = layer_mean_j[li], layer_mean_l[li]
        print(f"{layer:>5} {j:>9.4f} {ll_:>9.4f} {j - ll_:>9.4f}")

    dm = summary["depth_of_emergence_top10"]
    print("\n=== Depth-of-emergence (median layer, model top-1 in lens top-10) ===")
    print(
        f"  all positions : J-lens={dm['median_layer_j_allpos']}  "
        f"logit={dm['median_layer_l_allpos']}  "
        f"(never: J={dm['n_never_emerged_j_allpos']} "
        f"logit={dm['n_never_emerged_l_allpos']} / {dm['n_samples_allpos']})"
    )
    print(
        f"  last position : J-lens={dm['median_layer_j_lastpos']}  "
        f"logit={dm['median_layer_l_lastpos']}"
    )
    print(f"\n[save] {out}")
    print(
        f"[time] mean {summary['mean_seconds_per_prompt']:.1f}s/prompt over "
        f"{len(prompts)} prompts"
    )

    if args.qualitative > 0:
        print_trajectories(
            m, lens, prompts[: args.qualitative], layers, args.max_seq_len
        )


def print_trajectories(
    m: Any, lens: Any, prompts: list[str], layers: list[int], max_seq_len: int
) -> None:
    """Top-5 decoded tokens per layer at the final position, both lens modes."""
    tok = m.tokenizer
    for pi, prompt in enumerate(prompts):
        seq_len = int(m.encode(prompt, max_length=max_seq_len).shape[1])
        pos = [seq_len - 1]
        ll_j, model_logits, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=pos,
            max_seq_len=max_seq_len,
            use_jacobian=True,
        )
        ll_l, _, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=pos,
            max_seq_len=max_seq_len,
            use_jacobian=False,
        )
        model_top = [
            tok.decode([int(t)]) for t in model_logits[0].topk(QUAL_TOPK).indices
        ]
        print(f"\n=== Trajectory prompt {pi + 1} (final position) ===")
        print(f"  prompt[:90]: {prompt[:90]!r}")
        print(f"  MODEL top-{QUAL_TOPK}: {model_top}")
        print(f"  {'layer':>5} | {'J-lens top-5':<44} | logit-lens top-5")
        for layer in layers:
            jt = [tok.decode([int(t)]) for t in ll_j[layer][0].topk(QUAL_TOPK).indices]
            lt = [tok.decode([int(t)]) for t in ll_l[layer][0].topk(QUAL_TOPK).indices]
            print(f"  {layer:>5} | {str(jt):<44} | {lt}")


if __name__ == "__main__":
    main()
