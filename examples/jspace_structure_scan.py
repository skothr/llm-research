#!/usr/bin/env python3
"""Stage 4 of the jspace arc: J-space structural characterization (paper §4).

For each source layer ``l`` of a fitted :class:`jlens.JacobianLens`, this
decomposes held-out activations ``h_l`` onto the J-lens vector dictionary by
**k-sparse nonnegative gradient pursuit** and measures, per position, how much
of the activation lives in the J-space and how concentrated that occupancy is.
This is the Qwen analog of the paper's "sensory -> workspace -> motor" depth map
`[gurnee2026-workspace §4.2; kb/notes/interpretability/j-space.md §1.2, §2]`.

Algebra (stated so the 7B run never materializes the |V|xd dictionary)
----------------------------------------------------------------------
The J-lens vectors are the rows of ``W_U J_l`` `[j-space.md §1.1]`. As input
directions in layer-``l`` activation space the atom for vocab token ``v`` is

    a_v = J_l^T w_v = (W_U[v] @ J_l)          in R^d ,   w_v = W_U[v]^T

so the full dictionary is ``D = (W_U J_l)^T`` (d x |V|), never formed. The
J-space component of ``h_l`` is the nearest sparse-nonneg cone point
`[gurnee2026-workspace §2.3]`; we decompose the raw residual ``h_l`` (the
readout's ``norm(.)`` is a token-scoring nonlinearity, not part of the J-space
geometry, and the note operationalizes the vectors as rows of ``W_U J_l`` with
no norm/gain, so atoms use the raw ``W_U``).

Gradient pursuit needs, each greedy step, the correlation of the residual ``r``
with every atom:

    c = D^T r = W_U (J_l r)   (a matvec chain: J_l r is a d-vector, then one
                               W_U matvec -> a |V|-vector).                  (1)

Only the <=k *selected* atoms are ever materialized (``a_j = W_U[j] @ J_l``,
d-vectors) for the least-squares-style refit. So peak extra memory over the
resident model is O(k*d), not O(|V|*d).

Gradient pursuit `[gurnee2026-workspace §2.3, refs 34-35; Blumensath & Davies
2008]` — the cheap greedy alternative to OMP (one gradient step per atom rather
than an exact least-squares solve). Per position, target ``y = h_l``:

    r <- y ; S <- []
    repeat up to k_max times:
        c   = W_U (J_l r)                     # eq. (1), batched over positions
        j   = argmax_v c_v  over v not in S with c_v > 0   # nonneg selection
        if no such j: stop
        S.append(j) ;  A = [a_s for s in S]   # d x |S|
        g   = A^T r                           # = c[S] (restricted correlation)
        Ag  = A g
        alpha = (g.g) / (Ag.Ag)               # optimal step along g
        x   = clamp(x + alpha*g, min=0)       # nonnegativity projection
        r   = y - A x

Per position we record: J-space **variance (squared-norm) fraction**
``||A x||^2 / ||h_l||^2`` (paper: never >10%); **meaningfully-active atom
count** = ``#{ x_i > tau * max(x) }`` with ``tau = 1e-3`` (the design's stated
default; paper reports ~10-25 active, §4.2); the **top atom token ids** (largest
coefficients) for qualitative inspection; and two **kurtosis** measures.

Kurtosis — what OF? The excerpt (`kb/excerpts/gurnee2026-workspace.md`) is
verbatim-authoritative here and is *silent* on the exact kurtosis quantity
(§4.2's kurtosis passage is not among the archived excerpts, and the note only
says "kurtosis-by-depth profile"). Rather than assert an unverified definition
we record both defensible candidates and lead with the first:
  * ``readout_kurtosis`` — excess kurtosis of the J-lens readout logit
    distribution over the vocabulary, ``W_U norm(J_l h_l)`` (= ``m.unembed``);
    the natural "how peaked is the readout" measure, computed for free.
  * ``coeff_kurtosis`` — excess kurtosis of the recovered nonneg coefficient
    vector (over its <=k_max active entries); how concentrated the J-space
    occupancy is among its atoms.
Both are Fisher excess kurtosis (normal -> 0).

Positions reuse the readout-scan convention (``mid_and_last_positions``): 8
evenly spaced interior positions plus the last token.

Trajectory: pursuit runs to ``k_max = max(--ks)`` and snapshots
``varfrac``/``active`` at each k in ``--ks`` (default ``5,10,25,50``); the
headline k is 25.

Usage:
    python examples/jspace_structure_scan.py --n-prompts 30           # 1.5B bf16
    python examples/jspace_structure_scan.py \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 \
        --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
        --n-prompts 30

Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the RTX 2080.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

# Reuse the readout scan's model/lens/prompt/position conventions verbatim, and
# the gradient-pursuit decomposition from the shared helper (also used by the
# stage-6 NLA cross-tie, so both scripts decompose h_l identically).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jspace_pursuit import (  # noqa: E402
    ACTIVE_TAU,
    excess_kurtosis,
    gradient_pursuit_layer,
)
from jspace_readout_scan import (  # noqa: E402
    DEFAULT_HELDOUT,
    build_model,
    heldout_tag,
    load_prompts,
    mid_and_last_positions,
    slug,
)

QUAL_TOKENS = 6  # decoded atoms printed in the qualitative section


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-space structure scan (stage 4).")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt",
        help="Path to a fitted JacobianLens .pt (JacobianLens.save format).",
    )
    p.add_argument(
        "--prompts",
        default=DEFAULT_HELDOUT,
        help="Held-out prompts JSON ({'prompts': [str, ...]}).",
    )
    p.add_argument(
        "--heldout-tag",
        default=None,
        help="Override the output-name held-out tag (auto-derived from a "
        "non-default --prompts filename otherwise; empty for the default set).",
    )
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--n-mid", type=int, default=8, help="Evenly spaced mid positions.")
    p.add_argument(
        "--ks",
        default="5,10,25,50",
        help="Sparsity snapshots to record (pursuit runs to the max).",
    )
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated source layers (default: all fitted source layers).",
    )
    p.add_argument(
        "--qualitative",
        type=int,
        default=2,
        help="Decode top J-space atoms for this many prompts at the last position.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .pt (default: cache/structure_scan_<model>_<lensstem>.pt).",
    )
    return p.parse_args()


def default_out(model: str, lens: str, tag: str = "") -> str:
    stem = Path(lens).stem
    return (
        f"research/arcs/jspace/data/cache/structure_scan_{slug(model)}_{stem}{tag}.pt"
    )


def capture_residuals(
    m: Any,
    prompt: str,
    layers: list[int],
    positions: list[int],
    max_seq_len: int,
) -> dict[int, Tensor]:
    """Raw residual stream at ``positions`` for each layer: ``{layer: [P, d]}``.

    Mirrors :meth:`JacobianLens.apply`'s capture (same hook, same position
    selection) but returns the untransported activations for decomposition.
    """
    from jlens.hooks import ActivationRecorder

    input_ids = m.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(m.layers, at=layers) as rec, torch.no_grad():
        m.forward(input_ids)
        acts = {i: rec.activations[i].detach() for i in layers}
    out: dict[int, Tensor] = {}
    for layer in layers:
        full = acts[layer][0]  # [seq_len, d_model]
        out[layer] = full[positions].float()
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    if not ks:
        raise SystemExit("--ks must list at least one sparsity level")

    from jlens import JacobianLens

    t_load = time.time()
    m = build_model(args)
    lens = JacobianLens.load(args.lens)
    print(f"[load] model + lens in {time.time() - t_load:.1f}s; {lens!r}")

    # W_U (unembedding) — private on HFLensModel; needed for the raw matvec chain.
    mm = cast(Any, m)
    w_u = cast(Tensor, mm._lm_head.weight)
    if not (torch.is_tensor(w_u) and w_u.dim() == 2 and w_u.is_floating_point()):
        raise SystemExit(
            f"lm_head.weight is not a plain 2-D float tensor (got {type(w_u)!r}); "
            "the raw W_U matvec chain assumes an unquantized unembedding."
        )
    device = w_u.device
    print(f"[wu] W_U {tuple(w_u.shape)} {w_u.dtype} on {device}")

    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        unknown = sorted(set(layers) - set(lens.source_layers))
        if unknown:
            raise SystemExit(f"--layers {unknown} not in {lens.source_layers}")
    else:
        layers = list(lens.source_layers)
    n_layers = len(layers)

    prompts = load_prompts(args.prompts, args.n_prompts)
    print(f"[data] {len(prompts)} held-out prompts from {args.prompts}")
    print(f"[scan] {n_layers} layers, ks={ks}, positions={args.n_mid} mid + last")

    # J_l are moved to GPU one layer at a time inside the loop (each is
    # d*d*4 bytes; preloading all 27 would cost ~1.4 GB and OOMs the 7B).

    # Aggregators [n_layers, n_prompts*n_pos-ish] collected as lists then stacked.
    vf_acc: dict[int, list[list[float]]] = {k: [[] for _ in layers] for k in ks}
    act_acc: dict[int, list[list[float]]] = {k: [[] for _ in layers] for k in ks}
    rk_acc: list[list[float]] = [[] for _ in layers]  # readout logit kurtosis
    rpk_acc: list[list[float]] = [[] for _ in layers]  # readout softmax-prob kurtosis
    ck_acc: list[list[float]] = [[] for _ in layers]  # coeff kurtosis

    tok = m.tokenizer  # decode top-atom ids to strings alongside the ids (additive)
    per_prompt: list[dict[str, Any]] = []
    per_prompt_times: list[float] = []
    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        seq_len = int(m.encode(prompt, max_length=args.max_seq_len).shape[1])
        positions, last_idx = mid_and_last_positions(seq_len, args.n_mid)
        n_pos = len(positions)
        resid = capture_residuals(m, prompt, layers, positions, args.max_seq_len)

        vf_p = {k: np.full((n_layers, n_pos), np.nan) for k in ks}
        act_p = {k: np.full((n_layers, n_pos), np.nan) for k in ks}
        rk_p = np.full((n_layers, n_pos), np.nan)
        rpk_p = np.full((n_layers, n_pos), np.nan)
        ck_p = np.full((n_layers, n_pos), np.nan)
        top_p: list[list[list[int]]] = []
        top_strs_p: list[list[list[str]]] = []  # decoded [layer][pos][atom]

        for li, layer in enumerate(layers):
            h = resid[layer]  # [P, d]
            jac = lens.jacobians[layer].to(device)  # fp32 d x d, this layer only
            # Readout kurtosis. The paper (footnote 4, verified in the archived
            # PDF §4.2) defines its kurtosis metric on the *logit* distribution
            # of the J-lens readout over the full vocabulary, no top-k
            # truncation (the "top-k 128/64/32" labels belong to the *accuracy*
            # panel a, not the kurtosis panel b). readout_kurtosis is thus the
            # paper metric; readout_prob_kurtosis (excess kurtosis of the
            # softmax distribution, fp32) is an additive supplement.
            with torch.no_grad():
                read_logits = m.unembed(h @ jac.t()).float()  # [P, |V|]
                read_probs = torch.softmax(read_logits, dim=-1)  # [P, |V|] fp32
            for p in range(n_pos):
                rk_p[li, p] = excess_kurtosis(read_logits[p])
                rpk_p[li, p] = excess_kurtosis(read_probs[p])
            del read_logits, read_probs

            res = gradient_pursuit_layer(h, jac, w_u, ks)
            for k in ks:
                vf_p[k][li] = res["varfrac"][k]
                act_p[k][li] = res["active"][k]
                vf_acc[k][li].extend(res["varfrac"][k].tolist())
                act_acc[k][li].extend(res["active"][k].tolist())
            ck_p[li] = res["coeff_kurtosis"]
            top_p.append(res["top_atoms"])
            top_strs_p.append(
                [[tok.decode([tid]) for tid in pos_ids] for pos_ids in res["top_atoms"]]
            )
            rk_acc[li].extend(rk_p[li].tolist())
            rpk_acc[li].extend(rpk_p[li].tolist())
            ck_acc[li].extend(ck_p[li].tolist())

        per_prompt.append(
            {
                "seq_len": seq_len,
                "positions": positions,
                "last_idx": last_idx,
                "layers": layers,
                "varfrac": vf_p,
                "active": act_p,
                "readout_kurtosis": rk_p,
                "readout_prob_kurtosis": rpk_p,
                "coeff_kurtosis": ck_p,
                "top_atoms": top_p,
                "top_atom_strs": top_strs_p,  # decoded strings for top_atoms
            }
        )
        dt = time.time() - t0
        per_prompt_times.append(dt)
        vf25 = np.nanmean(vf_p[25]) if 25 in ks else np.nanmean(vf_p[max(ks)])
        print(
            f"[prompt {pi + 1}/{len(prompts)}] seq={seq_len} pos={n_pos} "
            f"mean_varfrac(k={25 if 25 in ks else max(ks)})={vf25:.4f} {dt:.1f}s"
        )

    def layer_stat(acc: list[list[float]], fn: Any) -> list[float]:
        out: list[float] = []
        for li in range(n_layers):
            arr = np.array([x for x in acc[li] if not np.isnan(x)])
            out.append(float(fn(arr)) if arr.size else float("nan"))
        return out

    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "prompts": args.prompts,
        "n_prompts": len(prompts),
        "layers": layers,
        "ks": ks,
        "active_tau": ACTIVE_TAU,
        "n_mid_positions": args.n_mid,
        "kurtosis_note": (
            "readout_kurtosis = excess kurtosis of the full-vocab J-lens readout "
            "LOGIT distribution W_U norm(J_l h_l) -- this is the paper's metric "
            "(gurnee2026 footnote 4, verified in the archived PDF; no top-k "
            "truncation). readout_prob_kurtosis = same over the softmax "
            "distribution (supplement). coeff_kurtosis = excess kurtosis of the "
            "nonneg coefficient vector (k_max)."
        ),
        "mean_varfrac": {k: layer_stat(vf_acc[k], np.mean) for k in ks},
        "median_varfrac": {k: layer_stat(vf_acc[k], np.median) for k in ks},
        "mean_active": {k: layer_stat(act_acc[k], np.mean) for k in ks},
        "median_active": {k: layer_stat(act_acc[k], np.median) for k in ks},
        "mean_readout_kurtosis": layer_stat(rk_acc, np.mean),
        "mean_readout_prob_kurtosis": layer_stat(rpk_acc, np.mean),
        "mean_coeff_kurtosis": layer_stat(ck_acc, np.mean),
        "mean_seconds_per_prompt": float(np.mean(per_prompt_times)),
    }

    out = args.out or default_out(
        args.model, args.lens, heldout_tag(args.prompts, args.heldout_tag)
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_prompt": per_prompt, "summary": summary}, out)

    k_hdr = 25 if 25 in ks else max(ks)
    mvf = summary["mean_varfrac"][k_hdr]
    mvf_med = summary["median_varfrac"][k_hdr]
    mact = summary["mean_active"][k_hdr]
    print(f"\n=== Per-layer J-space structure (k={k_hdr}) ===")
    print(
        f"{'layer':>5} {'varfrac_mean':>13} {'varfrac_med':>12} "
        f"{'active_mean':>12} {'rd_logit_kurt':>14} {'rd_prob_kurt':>13} "
        f"{'coef_kurt':>10}"
    )
    for li, layer in enumerate(layers):
        print(
            f"{layer:>5} {mvf[li]:>13.4f} {mvf_med[li]:>12.4f} {mact[li]:>12.2f} "
            f"{summary['mean_readout_kurtosis'][li]:>14.2f} "
            f"{summary['mean_readout_prob_kurtosis'][li]:>13.2f} "
            f"{summary['mean_coeff_kurtosis'][li]:>10.2f}"
        )

    print("\n=== varfrac_mean trajectory by k ===")
    print("layer " + " ".join(f"k={k:>3}" for k in ks))
    for li, layer in enumerate(layers):
        cells = " ".join(f"{summary['mean_varfrac'][k][li]:>5.3f}" for k in ks)
        print(f"{layer:>5} {cells}")

    print(f"\n[save] {out}")
    print(
        f"[time] mean {summary['mean_seconds_per_prompt']:.1f}s/prompt over "
        f"{len(prompts)} prompts; total {sum(per_prompt_times):.1f}s"
    )

    if args.qualitative > 0:
        print_qualitative(m, per_prompt[: args.qualitative], layers)


def print_qualitative(
    m: Any, prompts_data: list[dict[str, Any]], layers: list[int]
) -> None:
    """Decode the top J-space atom tokens at the last position for a few prompts."""
    tok = m.tokenizer
    for pi, pd in enumerate(prompts_data):
        last = pd["last_idx"]
        print(f"\n=== Top J-space atoms (prompt {pi + 1}, last position) ===")
        print(f"  {'layer':>5} | top atom tokens (by coefficient)")
        for li, layer in enumerate(layers):
            ids = pd["top_atoms"][li][last][:QUAL_TOKENS]
            toks = [repr(tok.decode([tid])) for tid in ids]
            print(f"  {layer:>5} | {'  '.join(toks)}")


if __name__ == "__main__":
    main()
