"""Render category-structure figures (fig5-fig8) from emb_category_stats.pt.

Model-free. Outputs to research/arcs/embedding-atlas/observations/figures/:
  fig5_within_between.png       per-class within vs between cosines, raw+centered
  fig6_centroid_matrix.png      class-centroid cosine heatmap (centered)
  fig7_contrast_connectivity.png contrast-direction cosine heatmap (centered) —
                                 the layer-0 mirror of the nla-verbalizer arc's
                                 discriminant-connectivity figure
  fig8_e_vs_u_by_class.png      per-class cos(E_i, U_i) boxplots
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact, warn_if_mixed_sources

DPI = 150

SUPER_ORDER = (
    "topic",
    "pos",
    "sentiment",
    "register",
    "morphology",
    "multilingual",
    "function",
    "punct",
    "number",
    "code",
)
SUPER_COLOR = {
    "topic": "#3465a4",
    "pos": "#73d216",
    "sentiment": "#f57900",
    "register": "#edd400",
    "morphology": "#75507b",
    "multilingual": "#c17d11",
    "function": "#cc0000",
    "punct": "#555753",
    "number": "#06989a",
    "code": "#2e3436",
}


def ordered_classes(stats: dict[str, Any]) -> list[str]:
    sg = stats["supergroup_of"]
    return [c for s in SUPER_ORDER for c in stats["classes"] if sg[c] == s]


def main() -> None:
    warn_if_mixed_sources(["emb_category_stats.pt", "emb_battery_vectors.pt"])
    stats = load_artifact("emb_category_stats.pt")
    battery = load_artifact("emb_battery_vectors.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)
    classes = ordered_classes(stats)
    sg = stats["supergroup_of"]
    colors = [SUPER_COLOR[sg[c]] for c in classes]

    # ---- fig5: within vs between --------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = range(len(classes))
    for ax, space in zip(axes, ("raw", "centered")):
        pc = stats["spaces"][space]["per_class"]
        within = [pc[c]["within_mean"] for c in classes]
        between = [pc[c]["between_mean"] for c in classes]
        ax.bar(x, within, color=colors, alpha=0.9, label="within-class mean cos")
        ax.scatter(
            x, between, color="black", marker="_", s=180, label="between-class mean cos"
        )
        ax.set_ylabel(f"{space} cosine")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(classes, rotation=90, fontsize=7)
    fig.suptitle(
        "fig5 — within-class vs between-class cosine per class "
        "(bars colored by supergroup; black dash = between baseline)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_within_between.png", dpi=DPI)
    plt.close(fig)

    # ---- fig6 + fig7: heatmaps ------------------------------------------------
    order_idx = [stats["classes"].index(c) for c in classes]
    for figname, key, title in (
        (
            "fig6_centroid_matrix",
            "centroid_cos",
            "fig6 — class-centroid cosine matrix (centered space)",
        ),
        (
            "fig7_contrast_connectivity",
            "connectivity",
            "fig7 — contrast-direction connectivity cos(d_a, d_b) (centered space)",
        ),
    ):
        M: torch.Tensor = stats["spaces"]["centered"][key]
        M = M[order_idx][:, order_idx]
        fig, ax = plt.subplots(figsize=(12, 10.5))
        im = ax.imshow(M.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=6.5)
        for i, c in enumerate(classes):
            ax.get_xticklabels()[i].set_color(SUPER_COLOR[sg[c]])
            ax.get_yticklabels()[i].set_color(SUPER_COLOR[sg[c]])
        fig.colorbar(im, shrink=0.8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{figname}.png", dpi=DPI)
        plt.close(fig)

    # ---- fig8: E vs U by class -------------------------------------------------
    rows = battery["rows"]
    E = battery["E"].to(torch.float32)
    U = battery["U"].to(torch.float32)
    eu = torch.nn.functional.cosine_similarity(E, U, dim=1)
    per_class_vals = []
    for c in classes:
        idx = [i for i, r in enumerate(rows) if r["is_primary"] and r["class"] == c]
        per_class_vals.append(eu[idx].numpy())
    fig, ax = plt.subplots(figsize=(14, 5.5))
    bp = ax.boxplot(
        per_class_vals,
        positions=list(range(len(classes))),
        widths=0.7,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_ylabel("cos(E_i, U_i)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("fig8 — input-vs-unembedding alignment per class (primary anchors)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_e_vs_u_by_class.png", dpi=DPI)
    plt.close(fig)

    for n in (
        "fig5_within_between",
        "fig6_centroid_matrix",
        "fig7_contrast_connectivity",
        "fig8_e_vs_u_by_class",
    ):
        print(f"wrote {FIGURES / n}.png")


if __name__ == "__main__":
    main()
