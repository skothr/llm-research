"""Render full-vocabulary sweep figures (fig11-fig14).

Model-free; reads emb_fullvocab_stats.pt + emb_fullvocab_analysis.pt.
Outputs to research/arcs/embedding-atlas/observations/figures/:
  fig11_dim_kurtosis.png   per-dimension excess-kurtosis spectrum (feature dims)
  fig12_dim_corr.png       off-diagonal |r| histogram + correlation eigen-spectrum
  fig13_knn_cosine.png     nearest-neighbor cosine distributions (top-1, k=32)
  fig14_handle_census.png  out-of-battery membership per battery handle
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
    warn_if_mixed_sources(["emb_fullvocab_stats.pt", "emb_fullvocab_analysis.pt"])
    fv = load_artifact("emb_fullvocab_stats.pt")
    an = load_artifact("emb_fullvocab_analysis.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)
    n = int(fv["n_rows"])

    # ---- fig11: kurtosis spectrum -------------------------------------------
    kurt: torch.Tensor = fv["dim_stats"]["kurtosis"]
    srt = kurt.sort(descending=True).values
    n_heavy = int((kurt > 20).sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.semilogy(range(1, len(srt) + 1), srt.clamp_min(1e-2).numpy(), color="#3465a4")
    ax.axhline(
        20,
        color="#cc0000",
        ls="--",
        lw=0.8,
        label=f"kurtosis 20 ({n_heavy} dims above)",
    )
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("excess kurtosis (log)")
    ax.legend()
    ax.set_title("sorted per-dimension kurtosis")
    ax = axes[1]
    ax.hist(kurt.numpy(), bins=200, color="#3465a4")
    ax.set_yscale("log")
    ax.set_xlabel("excess kurtosis")
    ax.set_ylabel("dim count (log)")
    ax.set_title(f"distribution (median {float(kurt.median()):.2f})")
    fig.suptitle(f"fig11 — per-dimension kurtosis over {n} alive tokens")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_dim_kurtosis.png", dpi=DPI)
    plt.close(fig)

    # ---- fig12: dim correlation ----------------------------------------------
    cs: dict[str, Any] = fv["dim_corr_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    edges: torch.Tensor = cs["offdiag_hist_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax.semilogy(
        centers.numpy(), cs["offdiag_hist"].clamp_min(0.5).numpy(), color="#3465a4"
    )
    ax.set_xlabel("pairwise dimension correlation r")
    ax.set_ylabel("pair count (log)")
    ax.set_title(
        f"off-diagonal r (|r| mean {cs['offdiag_abs_mean']:.4f}, "
        f"max {cs['offdiag_max']:.3f})"
    )
    ax = axes[1]
    ev: torch.Tensor = cs["corr_evals"].clamp_min(1e-4)
    ax.loglog(range(1, len(ev) + 1), ev.numpy(), color="#3465a4")
    ax.set_xlabel("component rank")
    ax.set_ylabel("correlation-matrix eigenvalue")
    ax.set_title("correlation eigen-spectrum")
    fig.suptitle(f"fig12 — dimension-dimension correlation over {n} alive tokens")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_dim_corr.png", dpi=DPI)
    plt.close(fig)

    # ---- fig13: kNN cosine distributions --------------------------------------
    cos = fv["knn_cos"].float()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(
        cos[:, 0].numpy(),
        bins=200,
        color="#3465a4",
        alpha=0.8,
        label=f"top-1 neighbor (mean {float(cos[:, 0].mean()):+.3f})",
    )
    ax.hist(
        cos[:, -1].numpy(),
        bins=200,
        color="#f57900",
        alpha=0.6,
        label=f"32nd neighbor (mean {float(cos[:, -1].mean()):+.3f})",
    )
    ax.set_xlabel("cosine to neighbor (centered space)")
    ax.set_ylabel("token count")
    ax.legend()
    ax.set_title(f"fig13 — neighbor cosine distributions over {n} alive tokens")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_knn_cosine.png", dpi=DPI)
    plt.close(fig)

    # ---- fig14: handle census ---------------------------------------------------
    census: list[dict[str, Any]] = an["handle_census"]
    names = [e["class"] for e in census]
    hits = [max(e["out_of_battery_hits"], 1) for e in census]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(names)), hits, color="#3465a4")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("out-of-battery tokens above in-class threshold (log)")
    ax.set_title(
        "fig14 — handle census: vocabulary membership beyond the curated battery"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_handle_census.png", dpi=DPI)
    plt.close(fig)

    for f in (
        "fig11_dim_kurtosis",
        "fig12_dim_corr",
        "fig13_knn_cosine",
        "fig14_handle_census",
    ):
        print(f"wrote {FIGURES / f}.png")


if __name__ == "__main__":
    main()
