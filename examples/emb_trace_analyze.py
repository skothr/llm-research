"""Analyze the tracing-phase capture (T0 census, T1 reader map, T3/P2 persistence).

Model-free; reads emb_trace_layers.pt + emb_trace_weightmap.pt. Console
report + emb_trace_analysis.pt with:

  1. T0 census decode — Qwen2.5-7B's massive-activation dims (recurrence-
     ranked across prompts/layers), their host tokens, emergence layer, and
     overlap with (a) the 21 W_E block dims and (b) arc-1's layer-20 sink dims.
  2. T1 reader heads — per-(layer, head) block/control read ratios for q/k/v;
     top heads; per-RoPE-band profile of the top q/k readers.
  3. T3 / prediction P2 — per-layer block norm-fraction persistence, delimiter vs
     control vs first position, both INCLUDING and EXCLUDING massive-
     activation dims (the confound: a massive dim inside/outside the block
     distorts norm fractions).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from _emb_artifacts import load_artifact, write_artifact

TOP_HEADS = 15
MASS_DIM_MIN_PROMPTS = (
    7  # a dim is "massive" if in some layer's top-8 for >= half the prompts
)


def main() -> None:
    tl = load_artifact("emb_trace_layers.pt")
    wm = load_artifact("emb_trace_weightmap.pt")
    block_dims: torch.Tensor = tl["block_dims"]
    block_set = set(block_dims.tolist())
    sink_dims = set(tl["arc1_sink_dims"])
    n_prompts = len(tl["prompts"])
    out: dict[str, Any] = {"inputs": ["emb_trace_layers.pt", "emb_trace_weightmap.pt"]}

    # ---- 1. census decode ----------------------------------------------------
    # recurrence: dim -> set of prompts where it appears in any layer's top-8
    dim_prompts: dict[int, set[str]] = {}
    dim_tokens: dict[int, Counter] = {}
    dim_first_layer: dict[int, int] = {}
    dim_peak: dict[int, float] = {}
    for c in tl["census"]:
        for e in c["top"]:
            d = e["dim"]
            dim_prompts.setdefault(d, set()).add(c["prompt"])
            dim_tokens.setdefault(d, Counter())[repr(e["token"])] += 1
            if abs(e["value"]) > abs(dim_peak.get(d, 0.0)):
                dim_peak[d] = e["value"]
            if d not in dim_first_layer or c["layer"] < dim_first_layer[d]:
                dim_first_layer[d] = c["layer"]
    massive = sorted(
        (d for d, ps in dim_prompts.items() if len(ps) >= MASS_DIM_MIN_PROMPTS),
        key=lambda d: -abs(dim_peak[d]),
    )
    print(
        f"== T0: massive-activation dims (top-8 recurrence >= {MASS_DIM_MIN_PROMPTS}/{n_prompts} prompts)"
    )
    mass_rows = []
    for d in massive:
        toks = ", ".join(t for t, _ in dim_tokens[d].most_common(5))
        flags = ("BLOCK" if d in block_set else "") + (
            " ARC1-SINK" if d in sink_dims else ""
        )
        mass_rows.append(
            {
                "dim": d,
                "peak_value": dim_peak[d],
                "first_layer": dim_first_layer[d],
                "n_prompts": len(dim_prompts[d]),
                "host_tokens": dim_tokens[d].most_common(8),
                "in_block": d in block_set,
                "in_arc1_sinks": d in sink_dims,
            }
        )
        print(
            f"  dim {d:4d} peak {dim_peak[d]:+9.1f} first-layer {dim_first_layer[d]:2d} "
            f"prompts {len(dim_prompts[d]):2d} {flags:16s} tokens: {toks}"
        )
    out["massive_dims"] = mass_rows
    mass_set = {r["dim"] for r in mass_rows}
    out["block_massive_overlap"] = sorted(mass_set & block_set)
    out["sink_massive_overlap"] = sorted(mass_set & sink_dims)
    print(f"  overlap with W_E block: {out['block_massive_overlap']}")
    print(f"  overlap with arc-1 layer-20 sinks: {out['sink_massive_overlap']}")

    # which tokens HOST massive activations (position-level)
    host_counter: Counter = Counter()
    for c in tl["census"]:
        if c["layer"] < 2:
            continue
        for e in c["top"]:
            if abs(e["value"]) > 100 and e["dim"] in mass_set:
                host_counter[repr(e["token"])] += 1
    out["host_tokens_overall"] = host_counter.most_common(15)
    print(f"  host tokens overall: {host_counter.most_common(10)}")

    # ---- 2. T1 reader heads ----------------------------------------------------
    print("== T1: block-reader heads (block/control norm ratio)")
    reader_tables: dict[str, Any] = {}
    for proj in ("q", "k", "v"):
        blk = wm[f"{proj}_block"]  # (L, H, bands)
        ctl = wm[f"{proj}_control"]
        per_head = blk.norm(dim=2) / ctl.norm(dim=2).clamp_min(1e-12)  # (L, H)
        flat = per_head.flatten()
        top = torch.topk(flat, min(TOP_HEADS, flat.numel()))
        rows = []
        for v, idx in zip(top.values, top.indices):
            li, hi = int(idx // per_head.shape[1]), int(idx % per_head.shape[1])
            band_profile = blk[li, hi] / ctl[li, hi].clamp_min(1e-12)
            top_bands = torch.topk(band_profile, 5)
            rows.append(
                {
                    "layer": li,
                    "head": hi,
                    "ratio": float(v),
                    "top_bands": [
                        (int(b), float(r))
                        for r, b in zip(top_bands.values, top_bands.indices)
                    ],
                }
            )
        reader_tables[proj] = {
            "per_head_ratio": per_head,
            "top": rows,
            "ratio_mean": float(per_head.mean()),
            "ratio_max": float(per_head.max()),
        }
        head_desc = ", ".join(
            f"L{r['layer']}H{r['head']}={r['ratio']:.2f}" for r in rows[:6]
        )
        print(
            f"  {proj}: mean {reader_tables[proj]['ratio_mean']:.3f} "
            f"max {reader_tables[proj]['ratio_max']:.3f} | top: {head_desc}"
        )
    out["readers"] = reader_tables

    # ---- 3. T3 / P2 persistence -------------------------------------------------
    print("== T3/P2: block norm-fraction persistence by layer (mean over prompts)")
    L = tl["prompts"][0]["block_frac"].shape[0]
    prof = {k: torch.zeros(L) for k in ("delim", "ctrl", "first")}
    prof_nomass = {k: torch.zeros(L) for k in ("delim", "ctrl", "first")}
    counts = {k: 0 for k in prof}
    # recompute block-frac excluding massive dims needs vectors — only have
    # tracked positions at VEC_LAYERS; use those for the nomass variant.
    vec_layers: list[int] = tl["vec_layers"]
    nomass_block = torch.tensor(
        [d for d in block_dims.tolist() if d not in mass_set], dtype=torch.long
    )
    out["nomass_block_dims"] = nomass_block.tolist()
    for p in tl["prompts"]:
        bf: torch.Tensor = p["block_frac"]  # (L, n_pos)
        groups = {
            "delim": p["delim_positions"],
            "ctrl": [i for i in p["control_positions"] if i != 0],
            "first": [0],
        }
        for k, pos in groups.items():
            if not pos:
                continue
            prof[k] += bf[:, pos].mean(dim=1)
            counts[k] += 1
        # nomass on tracked vectors
        tracked: list[int] = p["tracked_positions"]
        vecs: torch.Tensor = p[
            "tracked_vecs"
        ].float()  # (n_vec_layers, n_tracked, 3584)
        for k, pos in groups.items():
            sel = [tracked.index(i) for i in pos if i in tracked]
            if not sel:
                continue
            v = vecs[:, sel]
            frac = v[:, :, nomass_block].norm(dim=2) / v.norm(dim=2).clamp_min(1e-12)
            full = torch.full((L,), float("nan"))
            full[list(vec_layers)] = frac.mean(dim=1)
            mask = ~torch.isnan(full)
            prof_nomass[k][mask] += full[mask]
    for k in prof:
        prof[k] /= max(counts[k], 1)
        prof_nomass[k] /= max(counts[k], 1)
    out["block_frac_profile"] = {k: v for k, v in prof.items()}
    out["block_frac_profile_nomass"] = {k: v for k, v in prof_nomass.items()}
    out["vec_layers"] = vec_layers
    base = (21 / 3584) ** 0.5
    base_nm = (max(len(nomass_block), 1) / 3584) ** 0.5
    out["random_baseline"] = base
    out["random_baseline_nomass"] = base_nm
    print(
        f"  random baseline (21 dims): {base:.4f}; excl-massive ({len(nomass_block)} dims): {base_nm:.4f}"
    )
    for li in (0, 1, 2, 3, 7, 14, 20, 28):
        print(
            f"  L{li:2d}: delim {float(prof['delim'][li]):.4f} "
            f"ctrl {float(prof['ctrl'][li]):.4f} first {float(prof['first'][li]):.4f}"
            + (
                f" | nomass delim {float(prof_nomass['delim'][li]):.4f} "
                f"ctrl {float(prof_nomass['ctrl'][li]):.4f}"
                if li in vec_layers
                else ""
            )
        )

    path = write_artifact("emb_trace_analysis.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
