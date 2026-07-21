"""Render pair-direction consistency (fig10) from emb_pair_directions.pt.

Model-free. Top panel: per-kind consistency (mean cosine of each pair's
difference vector to the kind's mean direction) vs the permutation baseline.
Bottom panel: the per-pair cos-to-mean-direction values as strip plots, so
outlier pairs are visible individually.
Output: research/arcs/embedding-atlas/observations/figures/fig10_pair_directions.png
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact, warn_if_mixed_sources

DPI = 150


def main() -> None:
    warn_if_mixed_sources(["emb_pair_directions.pt"])
    pd = load_artifact("emb_pair_directions.pt")
    kinds: dict[str, dict[str, Any]] = pd["kinds"]
    names = sorted(kinds, key=lambda k: -kinds[k]["consistency"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, height_ratios=[2, 3])
    x = list(range(len(names)))

    ax = axes[0]
    cons = [kinds[k]["consistency"] for k in names]
    base = [kinds[k]["shuffle_consistency_mean"] for k in names]
    err = [2 * kinds[k]["shuffle_consistency_std"] for k in names]
    ax.bar(x, cons, color="#3465a4", alpha=0.9, label="consistency")
    ax.errorbar(
        x,
        base,
        yerr=err,
        fmt="_",
        color="#cc0000",
        capsize=4,
        label="permutation baseline (±2 std)",
    )
    for i, k in enumerate(names):
        ax.annotate(
            f"n={kinds[k]['n']}",
            (i, cons[i]),
            ha="center",
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=8,
        )
    ax.set_ylabel("mean cos(delta_i, d_bar)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("per-kind difference-direction consistency vs chance")

    ax = axes[1]
    for i, k in enumerate(names):
        vals: torch.Tensor = kinds[k]["cos_to_bar"]
        jitter = (torch.rand(vals.shape[0]) - 0.5) * 0.5
        ax.scatter(
            (torch.full_like(vals, float(i)) + jitter).numpy(),
            vals.numpy(),
            s=10,
            color="#3465a4",
            alpha=0.6,
        )
        worst = int(vals.argmin())
        ax.annotate(
            kinds[k]["labels"][worst],
            (i, float(vals[worst])),
            fontsize=6.5,
            ha="center",
            textcoords="offset points",
            xytext=(0, -9),
            color="#cc0000",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("per-pair cos(delta_i, d_bar)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("per-pair alignment (worst pair labeled)")

    fig.suptitle("fig10 — shared-direction consistency of aligned word relations")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_pair_directions.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig10_pair_directions.png'}")


if __name__ == "__main__":
    main()
