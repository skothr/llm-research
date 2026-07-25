"""Paper-metric ceiling figure: excess FVE by depth, both models (issue #26).

Reads the three ``paper_metric_varfrac_*`` artifacts produced by
``examples/jspace_paper_metric_varfrac.py`` and renders ONE panel:

- solid lines: the paper-faithful metric — excess-over-random
  orthogonal-projection FVE at K = median occupancy
  [gurnee2026-workspace §4.2 Fig 30b, §A.8] — vs layer, both models
  (scan-grid artifacts, 27 layers);
- faint dashed companions: the arc's original absolute varfrac@25
  (same artifacts' ``vf_ours_mean``), visualizing that the metric
  correction nearly cancels at the 1.5B workspace band and *raises* 7B;
- black-edged markers with CI whiskers: the 1.5B all-positions
  cluster-bootstrap 95% CIs at L0/L18/L21/L22 (n=5362 positions each);
- dashed reference at 0.10 (the paper's ceiling).

Headline: the 1.5B hump breaches the ceiling under the paper's own metric
(L21 excess 11.5%, CI [11.3, 11.7]); 7B stays under (peak 5.0%, L22).

Deterministic, Agg backend; writes one PNG to the arc figures dir.

Usage:
    python examples/jspace_render_paper_metric_figure.py
"""

from __future__ import annotations

import sys
from io import TextIOWrapper
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _jspace_paths import FIGDIR, resolve

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

OUT = FIGDIR / "2026-07-24-jspace-paper-metric-excess.png"
PAPER_CEILING = 0.10

GRID: list[dict[str, str]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "pt": "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "pt": "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "color": "#d62728",
        "marker": "s",
    },
]
ALLPOS_1P5B = (
    "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_allpos.pt"
)


def load(name: str) -> dict[int, dict[str, Any]]:
    path = resolve(name)
    if not path.exists():
        raise SystemExit(f"missing artifact: {name}")
    return torch.load(path, map_location="cpu", weights_only=False)["results"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for spec in GRID:
        res = load(spec["pt"])
        layers = sorted(res)
        excess = [res[L]["excess_mean"] for L in layers]
        naive = [res[L]["vf_ours_mean"] for L in layers]
        ax.plot(
            layers,
            excess,
            color=spec["color"],
            marker=spec["marker"],
            markersize=4,
            linewidth=2,
            label=f"{spec['key']} — excess FVE (paper metric)",
        )
        ax.plot(
            layers,
            naive,
            color=spec["color"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.45,
            label=f"{spec['key']} — naive varfrac@25 (committed scans)",
        )
        # Annotate the WORKSPACE-band peak (L>=17), not the global argmax —
        # the early-band values carry the norm-bias caveat and would otherwise
        # steal the callout from the claim-bearing hump.
        band = [x for x in layers if x >= 17]
        peak = max(band, key=lambda x: res[x]["excess_mean"])
        ax.annotate(
            f"L{peak}: {res[peak]['excess_mean'] * 100:.1f}% (workspace peak)",
            (peak, res[peak]["excess_mean"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color=spec["color"],
        )
    ax.annotate(
        "L0: early band —\nnorm-bias caveat (obs 2026-07-24)",
        (0, 0.1144),
        textcoords="offset points",
        xytext=(10, -14),
        fontsize=7.5,
        color="#777777",
    )

    allpos = load(ALLPOS_1P5B)
    for L in sorted(allpos):
        r = allpos[L]
        lo, hi = r["excess_ci95"]
        mid = r["excess_mean"]
        ax.errorbar(
            [L],
            [mid],
            yerr=[[mid - lo], [hi - mid]],
            fmt="D",
            markersize=6,
            color="#1f77b4",
            markeredgecolor="k",
            ecolor="k",
            capsize=3,
            linewidth=1.2,
            zorder=5,
            label=(
                "1.5B all-positions mean, cluster-bootstrap 95% CI (n=5362/layer)"
                if L == sorted(allpos)[0]
                else None
            ),
        )

    ax.axhline(PAPER_CEILING, color="#555555", linestyle=":", linewidth=1.4, zorder=1)
    # Right-aligned at the axis edge so the label never collides with the
    # upper-center legend.
    ax.text(
        26.5,
        PAPER_CEILING - 0.006,
        'paper ceiling: "never more\nthan 10%" (Fig 30b metric)',
        fontsize=8,
        color="#555555",
        ha="right",
        va="top",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("J-space variance fraction of $h_\\ell$")
    ax.set_title(
        "The 10% ceiling under the paper's own metric: 1.5B breaches at its "
        "hump, 7B stays under\n(excess-over-random orthogonal-projection FVE "
        "at K = median occupancy)"
    )
    ax.legend(fontsize=8, loc="upper center")
    ax.set_ylim(bottom=0.0, top=0.152)  # headroom so the legend clears the data
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.text(
        0.5,
        -0.02,
        "source: paper_metric_varfrac_{qwen2.5-1.5b-instruct_...bf16_n100,"
        "qwen2.5-7b-instruct_...nf4_n100,...bf16_n100_allpos}.pt "
        "(examples/jspace_paper_metric_varfrac.py) | "
        "render: examples/jspace_render_paper_metric_figure.py",
        ha="center",
        fontsize=7,
        color="#9a9a9a",
    )
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
