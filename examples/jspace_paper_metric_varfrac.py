#!/usr/bin/env python3
"""Paper-faithful J-space variance metric (issue #26 recompute).

The arc's committed structure scan reports the *absolute single-gradient-step
reconstruction energy* ``||A x||^2 / ||h||^2`` at fixed k. The paper's
"never more than 10%" ceiling `[gurnee2026-workspace §4.2, Fig 30b; §A.8]` is
a different quantity:

    excess = FVE(top-K pursuit atoms) − FVE(K random vocab atoms)

with FVE the *orthogonal-projection* fraction ``||Π_S h||^2 / ||h||^2``
(least-squares onto the selected atom span, §A.8) evaluated at
K = median pursuit occupancy per layer, swept over valid positions. This
script computes that metric against a fitted lens, either on the committed
structure-scan grid (8 evenly spaced interior positions + last token; with
``--scan`` it first replicates the stored varfrac@k-snap bit-for-bit as a
validation gate) or over all valid positions ``[skip_first, seq_len-2]``
(``--all-positions``; the paper's population). Cluster bootstrap by prompt
gives a 95% CI on the mean excess per layer and the fraction of resamples
above the 10% ceiling.

Findings (2026-07-23/24, committed artifacts in the arc ``data/``): the
1.5B workspace-hump breach of the ceiling SURVIVES the metric correction
(L21 excess 11.5%, CI [11.3, 11.7], all-positions) and 7B stays under
(peak excess ~5.0% at L22) — see
``observations/2026-07-24-paper-metric-varfrac-recompute.md``.

Examples (repo root):
    # 1.5B bf16, scan grid + validation against the committed scan
    python examples/jspace_paper_metric_varfrac.py \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
        --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt

    # 1.5B, all valid positions at the decisive layers (paper population)
    python examples/jspace_paper_metric_varfrac.py \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt \
        --all-positions --layers 0,18,21,22 --n-rand 4 --rand-seed-base 20000

    # 7B nf4 counterpart
    python examples/jspace_paper_metric_varfrac.py \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
        --scan research/arcs/04_jspace/data/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt \
        --rand-seed-base 30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jspace_pursuit import gradient_pursuit_layer  # noqa: E402
from jspace_readout_scan import (  # noqa: E402
    DEFAULT_HELDOUT,
    build_model,
    heldout_tag,
    load_prompts,
    mid_and_last_positions,
    slug,
)
from jspace_structure_scan import capture_residuals  # noqa: E402

from jlens.lens import JacobianLens  # noqa: E402

ARC_DATA = Path("research/arcs/04_jspace/data")
SKIP_FIRST = 16  # matches the lens fit's valid-position convention
POS_BATCH = 64  # all-positions pursuit batch (memory guard)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-metric J-space variance.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--lens", required=True, help="Fitted JacobianLens .pt")
    p.add_argument(
        "--scan",
        default=None,
        help="Committed structure-scan .pt to validate the replicated "
        "varfrac@k-snap against (scan-grid mode only).",
    )
    p.add_argument("--prompts", default=DEFAULT_HELDOUT)
    p.add_argument("--heldout-tag", default=None)
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--n-mid", type=int, default=8)
    p.add_argument(
        "--layers", default=None, help="Comma list; default = all lens layers."
    )
    p.add_argument(
        "--all-positions",
        action="store_true",
        help="Sweep every position in [16, seq_len-2] (paper population) "
        "instead of the scan grid.",
    )
    p.add_argument("--k-snap", type=int, default=25, help="Occupancy snapshot k.")
    p.add_argument("--k-max", type=int, default=50)
    p.add_argument("--n-rand", type=int, default=8, help="Random-control draws.")
    p.add_argument("--rand-seed-base", type=int, default=10_000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def orth_fve(h: Tensor, atom_mat: Tensor) -> float:
    """``||Π h||^2 / ||h||^2`` for the span of ``atom_mat`` rows (QR, rank-safe)."""
    dn = float((h * h).sum())
    if dn == 0.0 or atom_mat.shape[0] == 0:
        return 0.0
    q, _ = torch.linalg.qr(atom_mat.t())
    proj = q @ (q.t() @ h)
    return float((proj * proj).sum()) / dn


def cluster_bootstrap(
    excess: np.ndarray, prompt_idx: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """(ci_lo, ci_hi, frac_over_10pct) for the mean excess, resampling prompts."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(prompt_idx)
    by_prompt = {int(u): excess[prompt_idx == u] for u in uniq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        boot[b] = np.concatenate([by_prompt[int(u)] for u in picks]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), float((boot > 0.10).mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    m = build_model(args)
    w_u: Tensor = m._lm_head.weight
    vocab = int(w_u.shape[0])
    lens = JacobianLens.load(args.lens)
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else list(lens.source_layers)
    )
    prompts = load_prompts(args.prompts, args.n_prompts)
    scan: dict[str, Any] | None = None
    if args.scan:
        if args.all_positions:
            raise SystemExit("--scan validation requires the scan grid mode")
        scan = torch.load(args.scan, map_location="cpu", weights_only=False)
    print(
        f"[setup] {len(prompts)} prompts, layers={layers}, |V|={vocab}, "
        f"mode={'all-positions' if args.all_positions else 'scan grid'}"
    )

    # Pass 1: capture + pursuit (support-returning), storing per-position state.
    store: dict[int, list[tuple[int, list[int], Tensor, Tensor, float, int]]] = {
        L: [] for L in layers
    }
    vf_diffs: list[float] = []
    for pi, prompt in enumerate(prompts):
        seq_len = int(m.encode(prompt, max_length=args.max_seq_len).shape[1])
        if args.all_positions:
            positions = list(range(SKIP_FIRST, seq_len - 1))
        else:
            positions = mid_and_last_positions(seq_len, args.n_mid)[0]
        resid = capture_residuals(m, prompt, layers, positions, args.max_seq_len)
        for L in layers:
            jac = lens.jacobians[L].to(w_u.device)
            h_all = resid[L].to(w_u.device)
            for s in range(0, h_all.shape[0], POS_BATCH):
                h = h_all[s : s + POS_BATCH]
                res = gradient_pursuit_layer(
                    h, jac, w_u, [args.k_snap, args.k_max], return_support=True
                )
                vf25 = res["varfrac"][args.k_snap]
                act25 = res["active"][args.k_snap]
                if scan is not None:
                    li = list(scan["per_prompt"][pi]["layers"]).index(L)
                    stored = np.asarray(scan["per_prompt"][pi]["varfrac"][args.k_snap])[
                        li
                    ]
                    vf_diffs.append(float(np.nanmax(np.abs(vf25 - stored))))
                for p in range(h.shape[0]):
                    store[L].append(
                        (
                            pi,
                            list(res["support"][p]),
                            res["support_coeffs"][p].detach().cpu(),
                            h[p].detach().cpu(),
                            float(vf25[p]),
                            int(act25[p]),
                        )
                    )
            del jac
        print(
            f"[pursuit] prompt {pi + 1}/{len(prompts)} pos={len(positions)}", flush=True
        )
    if vf_diffs:
        print(
            f"[validate] replicated-vs-stored varfrac@{args.k_snap} "
            f"max|diff| = {max(vf_diffs):.3e}"
        )

    # Pass 2: paper metric per layer.
    results: dict[int, dict[str, Any]] = {}
    for L in layers:
        jac = lens.jacobians[L].to(w_u.device)
        occ = np.array([a for *_, a in store[L]])
        k_med = max(1, int(np.median(occ)))
        gen = torch.Generator().manual_seed(args.rand_seed_base + L)
        n = len(store[L])
        prompt_idx = np.empty(n, dtype=np.int64)
        fve_j = np.empty(n)
        fve_full = np.empty(n)
        fve_r = np.empty(n)
        vf_ours = np.empty(n)
        for i, (pi, sel, coeffs, h_cpu, vf25, *_) in enumerate(store[L]):
            h = h_cpu.to(w_u.device)
            prompt_idx[i] = pi
            vf_ours[i] = vf25
            if sel:
                order = torch.argsort(coeffs, descending=True)
                top = [sel[int(j)] for j in order[:k_med]]
                a_top = w_u[torch.tensor(top, device=w_u.device)].float() @ jac
                a_full = w_u[torch.tensor(sel, device=w_u.device)].float() @ jac
                fve_j[i] = orth_fve(h, a_top)
                fve_full[i] = orth_fve(h, a_full)
            else:
                fve_j[i] = fve_full[i] = 0.0
            draws = [
                orth_fve(
                    h,
                    w_u[
                        torch.randint(0, vocab, (k_med,), generator=gen).to(w_u.device)
                    ].float()
                    @ jac,
                )
                for _ in range(args.n_rand)
            ]
            fve_r[i] = float(np.mean(draws))
        excess = fve_j - fve_r
        lo, hi, frac_over = cluster_bootstrap(excess, prompt_idx, args.n_boot, seed=L)
        results[L] = {
            "K_median_occ": k_med,
            "n_pos": n,
            "vf_ours_mean": float(vf_ours.mean()),
            "fve_topK_mean": float(fve_j.mean()),
            "fve_full_mean": float(fve_full.mean()),
            "fve_rand_mean": float(fve_r.mean()),
            "excess_mean": float(excess.mean()),
            "excess_median": float(np.median(excess)),
            "excess_ci95": (lo, hi),
            "boot_frac_over_10pct": frac_over,
            "excess_pctiles_10_25_50_75_90": [
                float(x) for x in np.percentile(excess, [10, 25, 50, 75, 90])
            ],
            "excess": excess,
            "fve_j": fve_j,
            "fve_rand": fve_r,
            "prompt_idx": prompt_idx,
        }
        r = results[L]
        print(
            f"L{L:2d} K={k_med:2d} n={n} ours@{args.k_snap}={r['vf_ours_mean']:.4f} "
            f"fveTopK={r['fve_topK_mean']:.4f} fveRand={r['fve_rand_mean']:.4f} "
            f"EXCESS={r['excess_mean']:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] "
            f"P(>10%)={frac_over:.3f}",
            flush=True,
        )
        del jac

    tag = heldout_tag(args.prompts, args.heldout_tag)
    mode_tag = "_allpos" if args.all_positions else ""
    out = args.out or ARC_DATA / (
        f"paper_metric_varfrac_{slug(args.model)}_"
        f"jlens_{Path(args.lens).stem.removeprefix('jlens_')}{mode_tag}{tag}.pt"
    )
    torch.save(
        {
            "results": results,
            "config": {
                "metric": "excess = orthproj-FVE(top-K pursuit atoms) - "
                "orthproj-FVE(K uniform random vocab atoms); K = median "
                "occupancy (ACTIVE_TAU convention) at the k-snap pursuit "
                "snapshot [gurnee2026-workspace §4.2 Fig 30b, §A.8]",
                "model": args.model,
                "mode": args.mode,
                "lens": str(args.lens),
                "scan_validated_against": str(args.scan) if args.scan else None,
                "validation_max_vf_diff": max(vf_diffs) if vf_diffs else None,
                "prompts_file": str(args.prompts),
                "n_prompts": len(prompts),
                "all_positions": bool(args.all_positions),
                "k_snap": args.k_snap,
                "k_max": args.k_max,
                "n_rand": args.n_rand,
                "rand_seed_base": args.rand_seed_base,
                "n_boot": args.n_boot,
                "layers": layers,
            },
        },
        out,
    )
    peak = max(results, key=lambda x: results[x]["excess_mean"])
    print(
        f"\n[VERDICT-INPUT] peak excess L{peak}: "
        f"{results[peak]['excess_mean']:+.4f} "
        f"({'BREACHED' if results[peak]['excess_mean'] > 0.10 else 'UNDER'} "
        f"the 10% ceiling)"
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
