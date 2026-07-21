"""Render global-geometry figures (fig1-fig4) from emb_global_stats.pt.

Model-free. Outputs to research/arcs/embedding-atlas/observations/figures/:
  fig1_norms.png       row-norm distributions (real vs padded rows)
  fig2_anisotropy.png  cosine-to-mean + random-pair cosine, raw vs centered
  fig3_pca_spectrum.png  eigenvalue spectrum + cumulative variance
  fig4_e_vs_u.png      per-token cos(E_i, U_i) distribution (untied matrices)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact

DPI = 150


def main() -> None:
    g = load_artifact("emb_global_stats.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)
    n_real: int = g["n_real"]
    n_vocab: int = g["n_vocab"]
    norms: torch.Tensor = g["norms_E"]

    # ---- fig1: norms ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        norms[:n_real].numpy(),
        bins=200,
        color="#3465a4",
        alpha=0.85,
        label=f"real rows (n={n_real})",
    )
    if n_vocab > n_real:
        ax.hist(
            norms[n_real:].numpy(),
            bins=40,
            color="#cc0000",
            alpha=0.85,
            label=f"padded rows (n={n_vocab - n_real})",
        )
    nz = int(g["near_zero_real_ids"].numel())
    ax.set_xlabel("||E_i||_2 (bf16 row norm, fp32-computed)")
    ax.set_ylabel("token count")
    ax.set_yscale("log")
    ax.set_title(
        f"fig1 — input-embedding row norms, Qwen2.5-7B W_E ({nz} near-zero real rows)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_norms.png", dpi=DPI)
    plt.close(fig)

    # ---- fig2: anisotropy ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.hist(g["cos_to_mu"].numpy(), bins=200, color="#3465a4")
    m = float(g["cos_to_mu"].mean())
    ax.axvline(m, color="#cc0000", ls="--", label=f"mean = {m:+.3f}")
    ax.set_xlabel("cos(E_i, mu)")
    ax.set_ylabel("token count")
    ax.set_title("cosine to the mean embedding (real rows)")
    ax.legend()
    ax = axes[1]
    for key, color, label in (
        ("rand_cos_raw", "#3465a4", "raw"),
        ("rand_cos_centered", "#f57900", "mean-centered"),
    ):
        vals: torch.Tensor = g[key]
        ax.hist(
            vals.numpy(),
            bins=200,
            color=color,
            alpha=0.7,
            label=f"{label} (mean {float(vals.mean()):+.3f})",
        )
    ax.set_xlabel("cos(E_i, E_j), 10k random pairs")
    ax.set_title("random-pair cosine: anisotropy before/after centering")
    ax.legend()
    fig.suptitle("fig2 — anisotropy of the embedding table")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_anisotropy.png", dpi=DPI)
    plt.close(fig)

    # ---- fig3: PCA spectrum --------------------------------------------------
    evals: torch.Tensor = g["evals_centered"].clamp_min(0)
    frac = evals / evals.sum()
    cum = torch.cumsum(frac, dim=0)
    pr = float(evals.sum() ** 2 / (evals**2).sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.loglog(range(1, len(evals) + 1), evals.numpy(), color="#3465a4")
    ax.set_xlabel("component rank")
    ax.set_ylabel("eigenvalue (variance)")
    ax.set_title(f"centered covariance spectrum (participation ratio {pr:.0f})")
    ax = axes[1]
    ax.semilogx(range(1, len(cum) + 1), cum.numpy(), color="#3465a4")
    for k in (1, 10, 50):
        ax.axvline(k, color="gray", ls=":", lw=0.8)
        ax.annotate(
            f"top-{k}: {float(cum[k - 1]) * 100:.1f}%",
            (k, float(cum[k - 1])),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )
    ax.set_xlabel("component rank")
    ax.set_ylabel("cumulative variance fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("cumulative variance")
    fig.suptitle("fig3 — PCA spectrum of W_E (real rows, mean-centered)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_pca_spectrum.png", dpi=DPI)
    plt.close(fig)

    # ---- fig4: E vs U --------------------------------------------------------
    eu: torch.Tensor = g["eu_cos"][:n_real]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(eu.numpy(), bins=200, color="#3465a4")
    ax.axvline(
        float(eu.mean()),
        color="#cc0000",
        ls="--",
        label=f"mean = {float(eu.mean()):+.3f}",
    )
    ax.axvline(
        float(eu.median()),
        color="#f57900",
        ls="--",
        label=f"median = {float(eu.median()):+.3f}",
    )
    ax.set_xlabel("cos(E_i, U_i) per token (input embedding vs unembedding)")
    ax.set_ylabel("token count")
    ax.set_title("fig4 — input vs output embedding alignment (untied matrices)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_e_vs_u.png", dpi=DPI)
    plt.close(fig)

    for name in ("fig1_norms", "fig2_anisotropy", "fig3_pca_spectrum", "fig4_e_vs_u"):
        print(f"wrote {FIGURES / name}.png")


if __name__ == "__main__":
    main()
