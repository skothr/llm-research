"""Stage-4 headline figure: the J-space structural depth map.

Reads the two structure-scan artifacts produced by
``examples/jspace_structure_scan.py`` and renders ONE figure with two
vertically stacked panels sharing the layer x-axis:

  (a) J-space variance (squared-norm) fraction vs layer at k=25, both models,
      with a horizontal reference at 0.10 (the paper's stated ceiling,
      "never more than 10%" [gurnee2026-workspace §2.3/§4.2]) and markers on
      each model's peak / trough layer.
  (b) J-lens readout excess kurtosis (LOGIT distribution, the paper's metric
      per its footnote 4) vs layer, both models, annotated with the contrast
      that the paper's Claude-family shape is low-early / rise-mid / fall-late.

This is the Qwen analog of the paper's "sensory -> workspace -> motor" depth
map. Deterministic, no interactive display (Agg backend); writes a single PNG
under the arc's LFS-tracked figures dir.

Usage:
    python examples/jspace_render_structure_figures.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data" / "cache"
# jspace has no _nla_artifacts-style FIGURES helper; the arc figures dir is
# hardcoded here (the one convention divergence from the nla render scripts).
FIGDIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-20-jspace-structure-depth-map.png"

HEADLINE_K = 25
PAPER_CEILING = 0.10

MODELS: list[dict[str, str]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "pt": "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "color": "#1f77b4",
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "pt": "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "color": "#d62728",
    },
]


def load_summary(pt: str) -> dict[str, Any]:
    p = CACHE / pt
    if not p.exists():
        raise SystemExit(f"missing structure-scan artifact: {p}")
    return cast("dict[str, Any]", torch.load(p, weights_only=False)["summary"])


def main() -> None:
    series: list[dict[str, Any]] = []
    for spec in MODELS:
        s = load_summary(spec["pt"])
        layers = list(s["layers"])
        vf = np.array(s["mean_varfrac"][HEADLINE_K], dtype=float)
        kurt = np.array(s["mean_readout_kurtosis"], dtype=float)
        series.append(
            {
                "key": spec["key"],
                "color": spec["color"],
                "layers": layers,
                "vf": vf,
                "kurt": kurt,
                "peak_i": int(vf.argmax()),
                "trough_i": int(vf.argmin()),
            }
        )
        print(
            f"loaded {spec['key']}: {len(layers)} layers, "
            f"varfrac peak L{layers[int(vf.argmax())]}={vf.max():.3f} "
            f"trough L{layers[int(vf.argmin())]}={vf.min():.3f}"
        )

    fig, (ax_vf, ax_k) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True, height_ratios=[1.0, 0.9]
    )

    # ---- Panel (a): variance fraction ----
    for sr in series:
        ax_vf.plot(
            sr["layers"],
            sr["vf"],
            "-o",
            color=sr["color"],
            markersize=3.5,
            linewidth=1.8,
            label=sr["key"],
        )
        # Peak (diamond) and trough (down-triangle) markers.
        for idx, marker, tag in (
            (sr["peak_i"], "D", "peak"),
            (sr["trough_i"], "v", "trough"),
        ):
            lx = sr["layers"][idx]
            ly = float(sr["vf"][idx])
            ax_vf.plot(
                lx,
                ly,
                marker,
                color=sr["color"],
                markersize=9,
                markeredgecolor="k",
                markeredgewidth=0.6,
                zorder=5,
            )
            ax_vf.annotate(
                f"{tag} L{lx}\n{ly:.3f}",
                (lx, ly),
                textcoords="offset points",
                xytext=(0, 11 if tag == "peak" else -22),
                ha="center",
                fontsize=7.5,
                color=sr["color"],
            )
    ax_vf.axhline(
        PAPER_CEILING,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"paper's ~{PAPER_CEILING:.0%} ceiling (Claude-family claim)",
    )
    ax_vf.set_ylabel(f"J-space variance fraction\n(squared-norm, k={HEADLINE_K})")
    ax_vf.set_ylim(0.0, max(PAPER_CEILING, max(sr["vf"].max() for sr in series)) * 1.25)
    ax_vf.set_title(
        "J-space structural depth map — Qwen2.5, held-out wikitext (n=30 prompts)",
        fontsize=12,
    )
    ax_vf.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax_vf.grid(True, alpha=0.25)

    # ---- Panel (b): readout logit excess kurtosis ----
    for sr in series:
        ax_k.plot(
            sr["layers"],
            sr["kurt"],
            "-o",
            color=sr["color"],
            markersize=3.5,
            linewidth=1.8,
            label=sr["key"],
        )
    ax_k.axhline(0.0, color="#999999", linewidth=0.8)
    ax_k.set_ylabel(
        "J-lens readout excess kurtosis\n(logit distribution; paper metric)"
    )
    ax_k.set_xlabel("source layer")
    ax_k.grid(True, alpha=0.25)
    ax_k.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_k.annotate(
        "Paper (Claude-family) shape: ~0 early -> rises from ~1/3 depth -> falls late.\n"
        "Qwen (both scales) is INVERTED: high early -> mid trough -> weak late rise.",
        xy=(0.015, 0.03),
        xycoords="axes fraction",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "#fff6d5",
            "ec": "#c9b24b",
            "alpha": 0.9,
        },
    )

    ax_k.set_xticks(list(range(0, 27, 2)))
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    fig.text(
        0.5,
        0.006,
        "data: structure_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt "
        "(30 held-out prompts, heldout_prompts_wikitext103_n30.json; n=100 lens)  —  "
        "MANIFEST sha256-registered · see figures/DATA_PROVENANCE.md",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#9a9a9a",
    )
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
