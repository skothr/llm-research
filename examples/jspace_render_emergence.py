"""Stage-3 figure: depth-of-emergence and output-predictive Spearman.

DATA-AVAILABILITY NOTE (paper "unspoken words" trajectory grid): the
``readout_scan_*.pt`` ``per_prompt`` entries store only per-(layer, position)
scalar arrays — Spearman-to-final-logits (``spearman_j/l``), the rank of the
model's final top-1 token in each lens readout (``rank_j/l``), and the final
``top1`` ids. They do NOT store the per-layer top-k token STRINGS, so the
paper's layer x top-5 "unspoken words" token grid cannot be rendered from
these artifacts. This figure therefore renders the fallback specified for
that case: the depth-of-emergence comparison plus the per-layer output-
predictive Spearman curves, both from the scan summaries, for both models.

Two panels:

  (a) Per-layer mean Spearman rank-correlation of each lens's readout with
      the model's final next-token logits (last prompt position), J-lens vs
      logit-lens, both models. The logit lens is the better output predictor
      at every layer and both scales; the curves converge near the top.

  (b) Depth of emergence: the median source layer at which the model's final
      top-1 token first enters each lens's top-10 (all positions), J-lens vs
      logit-lens, both models, annotated with the never-emerged cell counts.
      Highlights the one clean scale-dependent reversal: the logit lens
      surfaces the output earlier at 1.5B, but the J-lens surfaces it earlier
      at 7B [gurnee2026-workspace §2.1].

Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_emergence.py
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
from matplotlib.lines import Line2D

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "jspace" / "data" / "cache"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-21-jspace-emergence.png"

MODELS: list[dict[str, str]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "pt": "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "color": "#1f77b4",
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "pt": "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "color": "#d62728",
    },
]


def load_summary(pt: str) -> dict[str, Any]:
    p = CACHE / pt
    if not p.exists():
        raise SystemExit(f"missing readout-scan artifact: {p}")
    return cast("dict[str, Any]", torch.load(p, weights_only=False)["summary"])


def main() -> None:
    series: list[dict[str, Any]] = []
    for spec in MODELS:
        s = load_summary(spec["pt"])
        doe = cast("dict[str, Any]", s["depth_of_emergence_top10"])
        series.append(
            {
                "key": spec["key"],
                "color": spec["color"],
                "layers": list(s["layers"]),
                "sp_j": np.array(s["layer_mean_spearman_j_last"], dtype=float),
                "sp_l": np.array(s["layer_mean_spearman_l_last"], dtype=float),
                "emg_j": float(doe["median_layer_j_allpos"]),
                "emg_l": float(doe["median_layer_l_allpos"]),
                "never_j": int(doe["n_never_emerged_j_allpos"]),
                "never_l": int(doe["n_never_emerged_l_allpos"]),
                "n_samples": int(doe["n_samples_allpos"]),
            }
        )
        print(
            f"{spec['key']}: emergence J L{doe['median_layer_j_allpos']:.0f} "
            f"logit L{doe['median_layer_l_allpos']:.0f} | "
            f"never-emerged J {doe['n_never_emerged_j_allpos']}/{doe['n_samples_allpos']} "
            f"logit {doe['n_never_emerged_l_allpos']}/{doe['n_samples_allpos']}"
        )

    fig, (ax_sp, ax_em) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), width_ratios=[1.35, 1.0]
    )

    # ---- Panel (a): Spearman curves ----
    for sr in series:
        ax_sp.plot(
            sr["layers"],
            sr["sp_j"],
            "-",
            color=sr["color"],
            linewidth=1.9,
            label=f"{sr['key']} — J-lens",
        )
        ax_sp.plot(
            sr["layers"],
            sr["sp_l"],
            "--",
            color=sr["color"],
            linewidth=1.6,
            label=f"{sr['key']} — logit-lens",
        )
    ax_sp.axhline(0.0, color="#999999", linewidth=0.8)
    ax_sp.set_xlabel("source layer")
    ax_sp.set_ylabel(
        "mean Spearman rank-corr. of lens readout\nwith final next-token logits (last position)"
    )
    ax_sp.set_title(
        "(a) Output-predictive agreement per layer\n(logit-lens leads except the final 7B layer, where J-lens crosses above)",
        fontsize=10.5,
    )
    ax_sp.set_xticks(list(range(0, 27, 2)))
    ax_sp.grid(True, alpha=0.25)
    ax_sp.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax_sp.annotate(
        "J-lens is negative in the earliest 7B layers\n(readout uncorrelated with the eventual output);\ncurves converge near the top layers",
        xy=(3, series[1]["sp_j"][3]),
        xytext=(6.0, 0.15),
        fontsize=7.5,
        color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#f4f4f4",
            "ec": "#888888",
            "alpha": 0.9,
        },
    )

    # ---- Panel (b): depth-of-emergence bars ----
    x = np.arange(len(series))
    w = 0.36
    for si, sr in enumerate(series):
        b_j = ax_em.bar(
            si - w / 2,
            sr["emg_j"],
            w,
            color=sr["color"],
            edgecolor="k",
            linewidth=0.5,
            label="J-lens" if si == 0 else None,
        )
        b_l = ax_em.bar(
            si + w / 2,
            sr["emg_l"],
            w,
            color=sr["color"],
            edgecolor="k",
            linewidth=0.5,
            hatch="///",
            alpha=0.6,
            label="logit-lens" if si == 0 else None,
        )
        ax_em.annotate(
            f"L{sr['emg_j']:.0f}",
            (si - w / 2, sr["emg_j"]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        ax_em.annotate(
            f"L{sr['emg_l']:.0f}",
            (si + w / 2, sr["emg_l"]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        ax_em.annotate(
            f"never emerged:\nJ {sr['never_j']}/{sr['n_samples']}, logit {sr['never_l']}/{sr['n_samples']}",
            (si, 1.5),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )
        _ = (b_j, b_l)

    ax_em.set_xticks(list(x))
    ax_em.set_xticklabels([sr["key"] for sr in series], fontsize=9)
    ax_em.set_ylabel(
        "median source layer where final top-1\ntoken enters lens top-10 (all positions)"
    )
    ax_em.set_title(
        "(b) Depth of emergence — the scale-dependent reversal", fontsize=10.5
    )
    ax_em.set_ylim(0, 28)
    ax_em.grid(True, axis="y", alpha=0.25)
    # reversal callout
    ax_em.annotate(
        "reversal: logit-lens earlier at 1.5B (L19 < L23),\nbut J-lens earlier at 7B (L22 < L24)",
        xy=(1 + w / 2, series[1]["emg_l"]),
        xytext=(0.5, 9.0),
        fontsize=8,
        color="#333333",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#fff6d5",
            "ec": "#c9b24b",
            "alpha": 0.9,
        },
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="k",
            markersize=9,
            label="J-lens (solid)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#cccccc",
            markeredgecolor="k",
            markersize=9,
            label="logit-lens (hatched)",
        ),
    ]
    ax_em.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Depth of emergence and output-predictive agreement (Qwen2.5, n=100 lenses, "
        "12 held-out prompts x 9 positions) [gurnee2026-workspace §2.1]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
