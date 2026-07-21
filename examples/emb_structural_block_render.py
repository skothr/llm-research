"""Render the structural-block characterization (fig15).

Model-free; reads emb_structural_block.pt. Left: block vs control norm fraction
fraction per token-id decile (BPE merge order = frequency proxy). Right:
per-script block vs control means (language-independence check).
Output: research/arcs/03_embedding-atlas/observations/figures/fig15_structural_block.png
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _emb_artifacts import FIGURES, load_artifact, warn_if_mixed_sources

DPI = 150


def main() -> None:
    warn_if_mixed_sources(["emb_structural_block.pt"])
    sb = load_artifact("emb_structural_block.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    x = range(1, 11)
    ax.plot(
        x,
        sb["block_decile_means"],
        "o-",
        color="#3465a4",
        label=f"21-dim block (Spearman {sb['block_spearman_vs_id']:+.3f})",
    )
    ax.plot(
        x,
        sb["control_decile_means"],
        "s--",
        color="#888a85",
        label=f"random 21-dim control ({sb['control_spearman_vs_id']:+.3f})",
    )
    ax.set_xlabel("token-id decile (1 = lowest ids = earliest BPE merges)")
    ax.set_ylabel("mean norm fraction in subspace")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("block norm fraction vs token-frequency proxy")

    ax = axes[1]
    per: dict[str, dict[str, Any]] = sb["per_script"]
    scripts = list(per)
    xs = range(len(scripts))
    ax.bar(
        [i - 0.2 for i in xs],
        [per[s]["block_mean"] for s in scripts],
        width=0.4,
        color="#3465a4",
        label="block",
    )
    ax.bar(
        [i + 0.2 for i in xs],
        [per[s]["control_mean"] for s in scripts],
        width=0.4,
        color="#888a85",
        label="control",
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{s}\n(n={per[s]['n']})" for s in scripts], fontsize=8)
    ax.set_ylabel("mean norm fraction")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("script-independence of block norm fraction")

    fig.suptitle(
        "fig15 — the 21-dim entangled block: head-frequency-loaded, script-independent"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig15_structural_block.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig15_structural_block.png'}")


if __name__ == "__main__":
    main()
