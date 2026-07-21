"""Render the 2D PCA map of battery anchors (fig9).

Model-free. PCA here is battery-only: computed over the primary anchor rows
(mean-centered within the battery), NOT the full-vocabulary PCs — the goal is
the geometry AMONG the curated anchors, maximally spread for inspection.
Output: research/arcs/03_embedding-atlas/observations/figures/fig9_pca_map.png
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact, warn_if_mixed_sources
from emb_category_render import SUPER_COLOR

DPI = 150
# a few exemplars labeled per supergroup so the map is readable
LABELED = (
    "France",
    "Paris",
    "dog",
    "happy",
    "red",
    "seven",
    "May",
    "Bill",
    "the",
    "of",
    "she",
    ".",
    ",",
    "7",
    "+",
    "def",
    "==",
    "purchase",
    "buy",
    "good",
    "bad",
    "king",
    "queen",
    "дом",  # CJK exemplars omitted: missing from the default matplotlib font
)


def main() -> None:
    warn_if_mixed_sources(["emb_battery_vectors.pt"])
    battery = load_artifact("emb_battery_vectors.pt")
    rows: list[dict[str, Any]] = battery["rows"]
    primary = [i for i, r in enumerate(rows) if r["is_primary"]]
    E = battery["E"][primary].to(torch.float32)
    words = [rows[i]["word"] for i in primary]
    sgs = [rows[i]["supergroup"] for i in primary]

    X = E - E.mean(dim=0)
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:4].T  # (n, 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (i, j) in zip(axes, ((0, 1), (1, 2))):
        for sg in SUPER_COLOR:
            idx = [k for k, s in enumerate(sgs) if s == sg]
            if not idx:
                continue
            ax.scatter(
                proj[idx, i].numpy(),
                proj[idx, j].numpy(),
                s=14,
                color=SUPER_COLOR[sg],
                label=sg,
                alpha=0.8,
            )
        for k, w in enumerate(words):
            if w in LABELED:
                ax.annotate(
                    w,
                    (float(proj[k, i]), float(proj[k, j])),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(3, 3),
                )
        ax.set_xlabel(f"battery PC{i + 1}")
        ax.set_ylabel(f"battery PC{j + 1}")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "fig9 — battery anchors on battery-only PCA axes (colored by supergroup)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_pca_map.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig9_pca_map.png'}")


if __name__ == "__main__":
    main()
