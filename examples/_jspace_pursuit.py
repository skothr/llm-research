#!/usr/bin/env python3
"""Shared J-space k-sparse nonnegative gradient-pursuit decomposition.

Factored out of ``jspace_structure_scan.py`` (stage 4) so the stage-6 NLA
cross-tie (``jspace_nla_crosstie.py``) can reuse the *identical* pursuit — the
two scripts must decompose ``h_l`` the same way for their results to be
comparable. ``gradient_pursuit_layer`` here is the byte-for-byte body that
previously lived in the structure scan, plus one additive, default-off knob
(``return_components``) that also returns the reconstructed J-space component
vector ``A x`` per position (the residual-space vector the cross-tie hands to
the NLA verbalizer). With the knob off the returned dict is unchanged, so the
structure scan's artifacts stay bit-identical.

Algebra and method (unchanged from the structure scan):
The J-lens vectors are the rows of ``W_U J_l`` `[j-space.md §1.1]`. As input
directions in layer-``l`` activation space the atom for vocab token ``v`` is
``a_v = J_l^T w_v = (W_U[v] @ J_l)`` in R^d, so the full dictionary is
``D = (W_U J_l)^T`` (d x |V|), never formed. Gradient pursuit
`[gurnee2026-workspace §2.3, refs 34-35; Blumensath & Davies 2008]` greedily
grows a nonnegative support; each step the shared correlation is
``c = W_U (J_l r)`` (one GEMM over positions), and only the <=k selected atoms
are ever materialized (O(k*d) peak extra memory).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

ACTIVE_TAU = 1e-3  # design default: coeff > tau * max(coeff) counts as active
STEP_EPS = 1e-12  # guard for the gradient-pursuit step-size denominator
TOP_ATOMS = 10  # top atom token ids stored per (layer, position)


def excess_kurtosis(x: Tensor) -> float:
    """Fisher excess kurtosis of a 1-D tensor (normal -> 0). NaN if degenerate."""
    xd = x.to(torch.float64).flatten()
    if xd.numel() < 2:
        return float("nan")
    d = xd - xd.mean()
    var = (d * d).mean()
    if float(var) == 0.0:
        return float("nan")
    return float((d.pow(4).mean() / var.pow(2)) - 3.0)


def gradient_pursuit_layer(
    residuals: Tensor,  # [P, d]  target activations h_l (raw)
    jac: Tensor,  # [d, d]   J_l on device (fp32)
    w_u: Tensor,  # [|V|, d] unembedding (native dtype)
    ks: list[int],
    *,
    return_components: bool = False,
) -> dict[str, Any]:
    """Batched k-sparse nonnegative gradient pursuit over ``P`` positions.

    The expensive correlation ``c = W_U (J_l r)`` (eq. 1) is one GEMM shared
    across positions each greedy step; per-position support/coefficients diverge
    but their refits are O(k*d) and cheap. Returns per-position arrays.

    When ``return_components`` is True the result also carries ``"components"``,
    a ``[P, d]`` tensor of the reconstructed J-space component ``A x`` at the
    final (``k_max``) support — the vector the NLA cross-tie re-verbalizes. The
    default (False) leaves the returned dict exactly as the structure scan
    consumes it, so that script's outputs are unchanged.
    """
    device = residuals.device
    n_pos, d = residuals.shape
    k_max = max(ks)
    # Transposed *views* (no .contiguous() copy): cuBLAS consumes the transpose
    # flag directly, so we avoid a second 1.09 GB W_U copy per call — the
    # difference between fitting and OOM on the 8 GB card at 7B.
    jac_t = jac.t()  # so r @ jac_t == J_l r  (transport convention)
    wu_t = w_u.t()  # [d, |V|]
    wu_dtype = w_u.dtype

    y = residuals  # [P, d]
    y_sqnorm = (y * y).sum(dim=1)  # [P]
    r = y.clone()  # residuals, mutated in place per position

    # Per-position support atom-vector stacks and coefficients (ragged -> lists).
    atoms: list[Tensor] = [residuals.new_zeros((0, d)) for _ in range(n_pos)]
    coeffs: list[Tensor] = [residuals.new_zeros((0,)) for _ in range(n_pos)]
    sel: list[list[int]] = [[] for _ in range(n_pos)]
    done = [False] * n_pos

    varfrac = {k: np.full(n_pos, np.nan) for k in ks}
    active = {k: np.full(n_pos, np.nan) for k in ks}
    ks_set = set(ks)

    for step in range(1, k_max + 1):
        # (1) batched correlation of current residuals with all atoms.
        c = (r @ jac_t).to(wu_dtype) @ wu_t  # [P, |V|]
        c = c.float()
        for p in range(n_pos):
            if done[p]:
                continue
            row = c[p]
            if sel[p]:
                row = row.clone()
                row[torch.tensor(sel[p], device=device)] = float("-inf")
            j = int(row.argmax().item())
            if float(row[j]) <= 0.0:  # no positive-correlation atom remains
                done[p] = True
                continue
            a_j = (w_u[j].to(torch.float32) @ jac).unsqueeze(0)  # [1, d] = W_U[j] @ J_l
            atoms[p] = torch.cat([atoms[p], a_j], dim=0)  # [m, d]
            coeffs[p] = torch.cat([coeffs[p], coeffs[p].new_zeros(1)])
            sel[p].append(j)

            a = atoms[p]  # [m, d]
            g = a @ r[p]  # [m]  = A^T r
            ag = g @ a  # [d]  = A g
            denom = float((ag * ag).sum())
            alpha = float((g * g).sum()) / denom if denom > STEP_EPS else 0.0
            coeffs[p] = torch.clamp(coeffs[p] + alpha * g, min=0.0)
            r[p] = y[p] - coeffs[p] @ a  # A^T x -> reconstruction, subtract

        if step in ks_set:
            for p in range(n_pos):
                comp = coeffs[p] @ atoms[p] if coeffs[p].numel() else y.new_zeros(d)
                denom = float(y_sqnorm[p])
                varfrac[step][p] = (
                    float((comp * comp).sum()) / denom if denom > 0 else float("nan")
                )
                cf = coeffs[p]
                if cf.numel():
                    thr = ACTIVE_TAU * float(cf.max())
                    active[step][p] = int((cf > thr).sum().item())
                else:
                    active[step][p] = 0

        if all(done):
            break

    # Fill snapshots for k values beyond an early stop (state frozen at stop).
    for step in ks:
        if np.isnan(varfrac[step]).all():
            for p in range(n_pos):
                comp = coeffs[p] @ atoms[p] if coeffs[p].numel() else y.new_zeros(d)
                denom = float(y_sqnorm[p])
                varfrac[step][p] = (
                    float((comp * comp).sum()) / denom if denom > 0 else float("nan")
                )
                cf = coeffs[p]
                thr = ACTIVE_TAU * float(cf.max()) if cf.numel() else 0.0
                active[step][p] = int((cf > thr).sum().item()) if cf.numel() else 0

    # Final (k_max) descriptors: top atoms + coefficient kurtosis.
    top_atoms: list[list[int]] = []
    coeff_kurt = np.full(n_pos, np.nan)
    for p in range(n_pos):
        cf = coeffs[p]
        if cf.numel():
            order = cf.argsort(descending=True)[:TOP_ATOMS]
            top_atoms.append([int(sel[p][int(i)]) for i in order])
            coeff_kurt[p] = excess_kurtosis(cf)
        else:
            top_atoms.append([])
    out: dict[str, Any] = {
        "varfrac": varfrac,
        "active": active,
        "top_atoms": top_atoms,
        "coeff_kurtosis": coeff_kurt,
    }
    if return_components:
        comps = residuals.new_zeros((n_pos, d))
        for p in range(n_pos):
            if coeffs[p].numel():
                comps[p] = coeffs[p] @ atoms[p]
        out["components"] = comps
    return out
