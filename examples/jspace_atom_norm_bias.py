#!/usr/bin/env python3
"""Norm bias of J-space gradient-pursuit atom selection (issue #26).

Pursuit selects atoms by ``argmax_v <a_v, r>`` with UNNORMALIZED atoms
``a_v = W_U[v] @ J_l`` (`_jspace_pursuit.py`; neither the paper nor the
companion repo specifies normalization), so high-norm atoms can win on norm
rather than directional fit. This script quantifies the bias per layer:

1. atom-norm distribution over the full vocabulary (quantiles, CV);
2. Spearman rho between atom norm and ``W_U`` row norm (at 1.5B ``W_U`` is
   the tied input embedding, so row norms are frequency-correlated);
3. norm percentiles of the atoms ACTUALLY selected by the committed
   structure scan (top-10 coefficient atoms per (prompt, position));
4. top-norm token strings (format-token check).

Verified findings (2026-07-23, 1.5B bf16): selection is norm-neutral in the
workspace band (median selected-atom percentile 49.8/47.0 at L18/L21) but
biased in the early band (69.5 at L0, 30% of selections from the top norm
decile, top-norm atoms are whitespace/quote tokens) and anti-biased at the
final layers (26.0 at L26). Early-band occupancy / top-atom readings carry
this contamination; workspace-band results do not.

Needs the FULL fitted lens (cache-only; regenerate via jspace_fit_lens.py) —
the committed summary artifact in ``data/`` is therefore the auditable
record. Model weights load on CPU via the standard llm_surgeon path.

Run from the repo root:  python examples/jspace_atom_norm_bias.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jspace_readout_scan import build_model, slug  # noqa: E402

from jlens.lens import JacobianLens  # noqa: E402

ARC_DATA = Path("research/arcs/04_jspace/data")
DEFAULT_LAYERS = "0,5,10,15,18,21,26"
CHUNK = 16384
TOP_NORM_TOKENS = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atom-norm selection-bias probe.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument(
        "--lens",
        default=str(ARC_DATA / "cache/jlens_qwen2.5-1.5b_bf16_n100.pt"),
        help="FULL fitted lens (cache-only artifact).",
    )
    p.add_argument(
        "--scan",
        default=str(
            ARC_DATA
            / "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt"
        ),
        help="Committed structure scan providing the selected top_atoms.",
    )
    p.add_argument("--layers", default=DEFAULT_LAYERS)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    m = build_model(argparse.Namespace(model=args.model, mode=args.mode, device="cpu"))
    tied = m._lm_head.weight.data_ptr() == m._embed_tokens.weight.data_ptr()
    w_u = m._lm_head.weight.float()
    wu_norms = w_u.norm(dim=1).numpy()
    tok = m.tokenizer
    lens = JacobianLens.load(args.lens)
    scan = torch.load(args.scan, map_location="cpu", weights_only=False)
    scan_layers = list(scan["per_prompt"][0]["layers"])
    print(f"[setup] W_U {tuple(w_u.shape)} tied={tied}, layers={layers}")

    per_layer: dict[int, dict[str, Any]] = {}
    for L in layers:
        jac = lens.jacobians[L].float()
        norms = torch.empty(w_u.shape[0])
        for s in range(0, w_u.shape[0], CHUNK):
            norms[s : s + CHUNK] = (w_u[s : s + CHUNK] @ jac).norm(dim=1)
        norms_np = norms.numpy()
        li = scan_layers.index(L)
        sel = np.array(
            [
                tid
                for pp in scan["per_prompt"]
                for pos_ids in pp["top_atoms"][li]
                for tid in pos_ids
            ]
        )
        order = np.sort(norms_np)
        pct = np.searchsorted(order, norms_np[sel]) / len(norms_np) * 100
        top_ids = norms_np.argsort()[-TOP_NORM_TOKENS:][::-1]
        top_strs = [tok.decode([int(i)]) for i in top_ids]
        per_layer[L] = {
            "norm_p50_p90_p99_p999": [
                float(x) for x in np.percentile(norms_np, [50, 90, 99, 99.9])
            ],
            "norm_cv": float(norms_np.std() / norms_np.mean()),
            "rho_atomnorm_wunorm": spearman(norms_np, wu_norms),
            "n_selected": int(len(sel)),
            "n_selected_unique": int(len(set(sel.tolist()))),
            "sel_pctile_median": float(np.median(pct)),
            "sel_pctile_iqr": [
                float(np.percentile(pct, 25)),
                float(np.percentile(pct, 75)),
            ],
            "sel_frac_over_p90": float((pct > 90).mean()),
            "sel_frac_over_p99": float((pct > 99).mean()),
            "top_norm_token_ids": [int(i) for i in top_ids],
            "top_norm_token_strs": top_strs,
        }
        r = per_layer[L]
        print(
            f"L{L:2d} CV={r['norm_cv']:.2f} rho={r['rho_atomnorm_wunorm']:+.2f} "
            f"sel-pctile med={r['sel_pctile_median']:.1f} "
            f"IQR={[round(x, 1) for x in r['sel_pctile_iqr']]} "
            f">p90:{r['sel_frac_over_p90'] * 100:.0f}% "
            f"top-norm={r['top_norm_token_strs'][:4]!r}",
            flush=True,
        )

    out = args.out or ARC_DATA / (
        f"atom_norm_bias_{slug(args.model)}_"
        f"jlens_{Path(args.lens).stem.removeprefix('jlens_')}.pt"
    )
    torch.save(
        {
            "per_layer": per_layer,
            "config": {
                "model": args.model,
                "mode": args.mode,
                "lens": str(args.lens),
                "scan": str(args.scan),
                "tied_embeddings": bool(tied),
                "layers": layers,
                "selected_atoms_source": "structure-scan top_atoms "
                "(top-10 coefficient atoms per prompt/position)",
            },
        },
        out,
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
