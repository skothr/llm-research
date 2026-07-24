#!/usr/bin/env python3
"""Significance statistics for the entailed-swap headline gaps (issue #26).

The stage-5.2 headline (+5.17 / +2.17 nats J-lens-specific movement on the
unspoken entailed property) shipped with means only; the "per-item SD 4.9"
in the README was prose-only. This script derives, purely from the committed
``entailed_swap_*`` artifacts (CPU, deterministic):

- per-condition mean / SD / SEM of ``dlogp_swap_answer`` at s=2.0;
- paired jlens−control gaps with exact two-sided sign-flip permutation
  p-values (full 2^n enumeration for n <= 20, seeded 200k Monte Carlo above),
  exact binomial sign-test p, and a seeded bootstrap 95% CI;

on both the auto-detected-position subset and the mixed baseline-correct set
(subset logic identical to ``jspace_audit_findings.py``). All functions are
importable so the audit can pin the exact values.

Verified findings (2026-07-23): 1.5B L18 auto n=7 jlens−logitlens gap
+5.02 nats, 7/7 positive, exact p=0.0156 (the n=7 floor); 7B L19 auto n=17
gap +1.90, 15/17, p=0.0001 — the causal gap is not noise at either scale.

Run from the repo root:  python examples/jspace_swap_significance.py
"""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import torch

ARC_DATA = Path("research/arcs/04_jspace/data")
FILES: dict[str, str] = {
    "1.5B chat L18": (
        "entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt"
    ),
    "7B chat L19": (
        "entailed_swap_chat_L19_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt"
    ),
}
CONDS = ("jlens", "logitlens", "random")
STRENGTH = "2.0"
EXACT_MAX_N = 20  # full 2^n sign-flip enumeration up to here
MC_DRAWS = 200_000
MC_SEED = 1
BOOT_DRAWS = 10_000
BOOT_SEED = 0


def subset_split(per_item: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """{'auto': ..., 'mixed': ...} baseline-correct item subsets (audit logic)."""
    correct = [it for it in per_item if it["baseline_correct"]]
    return {
        "auto": [it for it in correct if it["scope_used"] == "auto"],
        "mixed": correct,
    }


def dlogp_vals(items: list[dict[str, Any]], cond: str) -> np.ndarray:
    key = f"{cond}@{STRENGTH}"
    return np.array(
        [
            float(it["conditions"][key]["dlogp_swap_answer"])
            for it in items
            if key in it["conditions"]
        ]
    )


def signflip_p(diffs: np.ndarray) -> float:
    """Two-sided sign-flip permutation p for mean(diffs); exact for n <= 20."""
    n = len(diffs)
    obs = abs(float(diffs.mean()))
    if n <= EXACT_MAX_N:
        codes = np.arange(2**n, dtype=np.uint32)[:, None]
        signs = 1.0 - 2.0 * ((codes >> np.arange(n)) & 1)
        means = np.abs(signs @ diffs) / n
        return float((means >= obs - 1e-12).mean())
    rng = np.random.default_rng(MC_SEED)
    signs = rng.choice((-1.0, 1.0), size=(MC_DRAWS, n))
    means = np.abs(signs @ diffs) / n
    return float(((means >= obs - 1e-12).sum() + 1) / (MC_DRAWS + 1))


def sign_test_p(diffs: np.ndarray) -> float:
    """Two-sided exact binomial sign test (zeros dropped)."""
    nz = diffs[diffs != 0]
    n, k = len(nz), int((nz > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def boot_ci(diffs: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    means = rng.choice(diffs, size=(BOOT_DRAWS, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def gap_stats(items: list[dict[str, Any]], control: str) -> dict[str, float]:
    """Paired jlens-minus-control statistics on ``items`` at s=2.0."""
    diffs = dlogp_vals(items, "jlens") - dlogp_vals(items, control)
    lo, hi = boot_ci(diffs)
    return {
        "n": float(len(diffs)),
        "gap_mean": float(diffs.mean()),
        "gap_sd": float(diffs.std(ddof=1)),
        "n_positive": float((diffs > 0).sum()),
        "perm_p": signflip_p(diffs),
        "sign_p": sign_test_p(diffs),
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
    }


def main() -> None:
    for label, fname in FILES.items():
        d = torch.load(ARC_DATA / fname, map_location="cpu", weights_only=False)
        print(f"\n=== {label} ({fname}) ===")
        for sname, items in subset_split(d["per_item"]).items():
            vals = {c: dlogp_vals(items, c) for c in CONDS}
            n = len(vals["jlens"])
            print(f"\n  [{sname}] n={n}")
            for c in CONDS:
                v = vals[c]
                sd = float(v.std(ddof=1))
                print(
                    f"    {c:9s} mean={v.mean():+7.3f}  SD={sd:5.2f}  "
                    f"SEM={sd / np.sqrt(n):5.2f}"
                )
            for ctrl in ("logitlens", "random"):
                g = gap_stats(items, ctrl)
                print(
                    f"    jlens - {ctrl:9s}: gap={g['gap_mean']:+6.3f}  "
                    f"SD={g['gap_sd']:5.2f}  "
                    f"pos={int(g['n_positive'])}/{int(g['n'])}  "
                    f"perm-p={g['perm_p']:.4f}  sign-p={g['sign_p']:.4f}  "
                    f"boot95%=[{g['boot_ci_lo']:+.2f}, {g['boot_ci_hi']:+.2f}]"
                )


if __name__ == "__main__":
    main()
