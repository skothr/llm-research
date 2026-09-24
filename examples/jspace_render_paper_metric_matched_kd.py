"""Matched-K/d paper-metric figure: excess FVE by depth at matched K/d_model
(issue #83).

The paper-metric excess (``excess = FVE(top-K pursuit atoms) - FVE(K random
atoms)`` [gurnee2026-workspace §4.2 Fig 30b, §A.8]) scales with K/d_model
(issue #79), so comparing the 1.5B (K~25, d=1536) with the 7B at its own
median occupancy (K~23, d=3584) is not like-for-like. This figure adds the
7B runs with K held fixed (``jspace_paper_metric_varfrac.py --k-fixed``):
K=58 matches the 1.5B K/d (58/3584 = 0.0162 vs 25/1536 = 0.0163).

Two panels, excess vs layer with cluster-bootstrap CI95 bands:

- (a) C4 held-out prompts: 1.5B at median occupancy, 7B at median
  occupancy, 7B at K=58;
- (b) wikitext grid prompts: 1.5B at median occupancy, 7B at median
  occupancy (pre-refit lens), 7B K=25 and K=58 (current, refit lens).

Dashed reference at 0.10 (the paper's ceiling) on both. Legend labels carry
K (the per-layer range where K is the median occupancy) and d_model. The
cross-scale gap (1.5B workspace-band peak / 7B workspace-band peak) is
printed per panel, derived from the plotted artifacts.

Deterministic, Agg backend, CPU only, no model; writes one PNG to the arc
figures dir. Run from the repo root:
    python examples/jspace_render_paper_metric_matched_kd.py
"""

from __future__ import annotations

import sys
from io import TextIOWrapper
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from _jspace_paths import FIGDIR, resolve

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

OUT = FIGDIR / "2026-09-23-jspace-paper-metric-matched-kd.png"
PAPER_CEILING = 0.10
BAND_START = 17  # workspace band (L17-26), as in the ceiling figure's callout
LABEL_HALF_WIDTH = 5  # layers the centred ceiling label spans on each side
_LFS_STUB = b"version https://git-lfs.github.com/spec/v1"
D_MODEL = {"1.5B": 1536, "7B": 3584}

# Okabe-Ito colorblind-safe palette; 1.5B blue / 7B red-orange family keeps
# the 2026-07-24 ceiling figure's model-to-hue mapping.
C_15 = "#0072B2"
C_7_OCC = "#D55E00"
C_7_K25 = "#E69F00"
C_7_K58 = "#009E73"

_PM = "paper_metric_varfrac_"
_P15 = _PM + "qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100"
_P7 = _PM + "qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100"

PANELS: list[dict[str, Any]] = [
    {
        "title": "(a) C4 held-out prompts (n=30 x 9 positions)",
        "series": [
            {"model": "1.5B", "pt": _P15 + "_heldoutc4en.pt", "color": C_15, "marker": "o", "ls": "-", "note": ""},
            {"model": "7B", "pt": _P7 + "_heldoutc4en.pt", "color": C_7_OCC, "marker": "s", "ls": "--", "note": ""},
            {"model": "7B", "pt": _P7 + "_heldoutc4en_k58.pt", "color": C_7_K58, "marker": "^", "ls": "-", "note": ""},
        ],
    },
    {
        "title": "(b) wikitext grid prompts (n=30 x 9 positions)",
        "series": [
            {"model": "1.5B", "pt": _P15 + ".pt", "color": C_15, "marker": "o", "ls": "-", "note": ""},
            {"model": "7B", "pt": _P7 + ".pt", "color": C_7_OCC, "marker": "s", "ls": "--", "note": ", pre-refit lens"},
            {"model": "7B", "pt": _P7 + "_refitlens_k25.pt", "color": C_7_K25, "marker": "v", "ls": ":", "note": ", refit lens"},
            {"model": "7B", "pt": _P7 + "_refitlens_k58.pt", "color": C_7_K58, "marker": "^", "ls": "-", "note": ", refit lens"},
        ],
    },
]


def load(name: str) -> dict[str, Any]:
    path = resolve(name)
    if not path.exists():
        raise SystemExit(f"missing artifact: {name}")
    with path.open("rb") as fh:
        if fh.read(len(_LFS_STUB)) == _LFS_STUB:
            raise SystemExit(f"{name} is a Git LFS pointer stub, not the artifact: run `git lfs pull`")
    return torch.load(path, map_location="cpu", weights_only=False)


def k_label(art: dict[str, Any], d: int) -> str:
    """K and K/d for the legend: a single value for a --k-fixed run (keyed on
    ``config.k_fixed``), the per-layer range for a median-occupancy run."""
    res: dict[int, dict[str, Any]] = art["results"]
    fixed = art["config"].get("k_fixed") is not None
    ks = sorted({int(r["K_used"] if fixed else r["K_median_occ"]) for r in res.values()})
    rule = "fixed" if fixed else "median occ."
    if len(ks) == 1:
        return f"K={ks[0]} ({rule}), K/d={ks[0] / d:.4f}"
    return (
        f"K={ks[0]}-{ks[-1]} ({rule}), "
        f"K/d={ks[0] / d:.4f}-{ks[-1] / d:.4f}"
    )


def band_peak(res: dict[int, dict[str, Any]]) -> tuple[int, float]:
    L = max((x for x in res if x >= BAND_START), key=lambda x: res[x]["excess_mean"])
    return L, float(res[L]["excess_mean"])


def draw_panel(ax: Axes, panel: dict[str, Any]) -> None:
    peaks: list[tuple[str, int, float]] = []  # (K tag, band-peak layer, value)
    top: dict[int, float] = {}  # per layer, the highest CI95 upper bound plotted
    for spec in panel["series"]:
        art = load(spec["pt"])
        res: dict[int, dict[str, Any]] = art["results"]
        layers = sorted(res)
        mean = [float(res[L]["excess_mean"]) for L in layers]
        lo = [float(res[L]["excess_ci95"][0]) for L in layers]
        hi = [float(res[L]["excess_ci95"][1]) for L in layers]
        for L, h in zip(layers, hi):
            top[L] = max(top.get(L, h), h)
        d = D_MODEL[spec["model"]]
        label = f"{spec['model']} (d={d}), {k_label(art, d)}{spec['note']}"
        ax.fill_between(layers, lo, hi, color=spec["color"], alpha=0.22, linewidth=0)
        ax.plot(
            layers,
            mean,
            color=spec["color"],
            marker=spec["marker"],
            markersize=3.5,
            linestyle=spec["ls"],
            linewidth=1.8,
            label=label,
        )
        L, v = band_peak(res)
        peaks.append((k_label(art, d).split(" (")[0], L, v))
        print(f"  {label}: band peak L{L} {v:.4f}")
    base_L, base_v = peaks[0][1], peaks[0][2]
    gaps = [
        f"1.5B L{base_L} / 7B L{L} ({ktag}{spec['note']}): {base_v / v:.2f}x"
        for (ktag, L, v), spec in zip(peaks[1:], panel["series"][1:])
    ]
    ax.text(
        0.02,
        0.97,
        "workspace-band peak gap\n" + "\n".join(gaps),
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        color="#333333",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax.axhline(PAPER_CEILING, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
    # Centre the label on the layer window where the plotted CI bands sit
    # furthest below the ceiling, so it does not cross any curve.
    centres = range(LABEL_HALF_WIDTH, 27 - LABEL_HALF_WIDTH)
    centre = min(
        centres,
        key=lambda c: max(top[L] for L in range(c - LABEL_HALF_WIDTH, c + LABEL_HALF_WIDTH + 1)),
    )
    ax.text(
        centre,
        PAPER_CEILING + 0.002,
        'paper ceiling: "never more than 10%"',
        fontsize=7.5,
        color="#555555",
        ha="center",
        va="bottom",
    )
    ax.set_title(panel["title"], fontsize=10)
    ax.set_xlabel("layer")
    ax.set_xlim(-0.5, 26.5)
    ax.set_ylim(0.0, 0.16)
    ax.grid(alpha=0.25, linewidth=0.6)
    # Below the axes: inside, it covered the 7B curves in panel (b).
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, panel in zip(axes, PANELS):
        print(panel["title"])
        draw_panel(ax, panel)
    axes[0].set_ylabel("excess FVE (top-K pursuit minus K random atoms)")
    fig.suptitle(
        "Paper-metric excess FVE at matched K/d_model (issue #83): 7B at K=58 "
        "matches the 1.5B K/d (58/3584 = 0.0162 vs 25/1536 = 0.0163)\n"
        "shaded bands: cluster-bootstrap 95% CI (by prompt, 2000 resamples)",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        -0.02,
        "source: data/paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100"
        "{,_heldoutc4en}.pt + data/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100"
        "{,_heldoutc4en,_heldoutc4en_k58,_refitlens_k25,_refitlens_k58}.pt\n"
        "(examples/jspace_paper_metric_varfrac.py) | "
        "render: examples/jspace_render_paper_metric_matched_kd.py",
        ha="center",
        fontsize=7,
        color="#9a9a9a",
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
