"""Stage-5 headline figure: swap-injection causality (the paper's Fig. 8 analog).

Reads the six-condition chat-style verbal-report swap artifacts
(``verbal_report_chat_6c_*.pt``) for both models and renders ONE figure with
two stacked panels (one per model), each a grouped bar chart of the
swap-target top-5 entry rate (over all 78 items) per condition x injection
strength s in {1, 2}.

The three paper-comparable conditions carry the Claude-family reference
markers from ``[gurnee2026-workspace §3.1, Fig. 8]``:

  - jlens           -> paper 88%   (token-indexed J-lens vector)
  - jspace_comp     -> paper 59%   (J-space component of a concept vector)
  - nonjspace_comp  -> paper 5%    (non-J-space residual of the concept vector)

The arc's central causal result is that on Qwen (1.5B/7B, n=100 lens) only
the token-indexed jlens vector is an effective swap direction; the paper's
middle tier (``jspace_comp``) collapses to the equal-magnitude random null.
A dashed chance line at each model's random-condition rate makes that null
visually unmissable.

Chat-only (the plain-4-condition companion is omitted: 6 conditions x 2
strengths per panel already fills the axis; the plain-run values are recorded
in the stage-5 observations). Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_swap_causality.py
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
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data" / "cache"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-21-jspace-swap-causality.png"

STRENGTHS = (1.0, 2.0)

# (artifact-condition key, display label, paper Claude-family rate or None)
CONDITIONS: list[tuple[str, str, float | None]] = [
    ("jlens", "J-lens vector\n(token-indexed)", 0.88),
    ("jspace_comp", "J-space component\nof concept vector", 0.59),
    ("nonjspace_comp", "non-J-space residual\nof concept vector", 0.05),
    ("logitlens", "logit-lens\n(token steering)", None),
    ("nonjspace", "non-J-space\n(w_t proxy)", None),
    ("random", "random\n(equal magnitude)", None),
]

MODELS: list[dict[str, str]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "pt": "verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "pt": "verbal_report_chat_6c_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
    },
]

BAR_COLORS = {1.0: "#9ecae1", 2.0: "#2171b5"}  # s=1 light, s=2 dark
PAPER_COLOR = "#d62728"


def load_summary(pt: str) -> dict[str, Any]:
    p = CACHE / pt
    if not p.exists():
        raise SystemExit(f"missing swap artifact: {p}")
    return cast("dict[str, Any]", torch.load(p, weights_only=False)["summary"])


def panel(ax: Axes, summary: dict[str, Any], title: str) -> None:
    metrics = cast("dict[str, Any]", summary["metrics"])
    n_items = int(summary["n_items"])
    layer = int(summary["layer"])

    n_cond = len(CONDITIONS)
    width = 0.38
    xs = list(range(n_cond))

    # Chance reference = the equal-magnitude random condition at s=2.
    chance = float(metrics["random@2.0"]["target_top5_rate_all"])

    for si, s in enumerate(STRENGTHS):
        offset = (si - 0.5) * width
        for xi, (cond, _label, _paper) in enumerate(CONDITIONS):
            rate = float(metrics[f"{cond}@{s:.1f}"]["target_top5_rate_all"])
            ax.bar(
                xi + offset,
                rate,
                width,
                color=BAR_COLORS[s],
                edgecolor="k",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                f"{rate:.2f}",
                (xi + offset, rate),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6.8,
                color="#333333",
            )

    # Paper Claude-family reference markers over the three comparable groups.
    for xi, (_cond, _label, paper) in enumerate(CONDITIONS):
        if paper is None:
            continue
        ax.plot(
            [xi - width, xi + width],
            [paper, paper],
            color=PAPER_COLOR,
            linewidth=2.0,
            zorder=5,
        )
        if paper < 0.10:
            # Low reference line sits inside the (tall) bars, so the label would
            # collide with them; park it in the inter-group gap just left of the
            # line, vertically centred on it.
            ax.annotate(
                f"paper {paper:.0%}",
                (xi - width, paper),
                textcoords="offset points",
                xytext=(-4, 0),
                ha="right",
                va="center",
                fontsize=7.5,
                color=PAPER_COLOR,
                fontweight="bold",
            )
        else:
            ax.annotate(
                f"paper {paper:.0%}",
                (xi, paper),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
                color=PAPER_COLOR,
                fontweight="bold",
            )

    # Chance line.
    ax.axhline(
        chance,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=f"chance (random @ s=2) = {chance:.2f}",
    )

    # Null callout: jspace_comp collapses to chance.
    jc_i = next(i for i, c in enumerate(CONDITIONS) if c[0] == "jspace_comp")
    jc_rate = float(metrics["jspace_comp@2.0"]["target_top5_rate_all"])
    ax.annotate(
        "J-space component\ncollapses to chance\n(paper's 59% does NOT replicate)",
        xy=(jc_i + 0.5 * width, jc_rate),
        xytext=(jc_i + 0.9, max(jc_rate, chance) + 0.30),
        fontsize=7.5,
        color=PAPER_COLOR,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": PAPER_COLOR, "linewidth": 1.1},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#fff0f0",
            "ec": PAPER_COLOR,
            "alpha": 0.9,
        },
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([c[1] for c in CONDITIONS], fontsize=8)
    ax.set_ylabel("swap-target in lens top-5\n(rate over all items)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{title} — L{layer}, n={n_items} items", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    handles = [
        Patch(facecolor=BAR_COLORS[1.0], edgecolor="k", label="strength s=1"),
        Patch(facecolor=BAR_COLORS[2.0], edgecolor="k", label="strength s=2"),
        Line2D(
            [0],
            [0],
            color=PAPER_COLOR,
            linewidth=2.0,
            label="paper (Claude 4.5) Fig. 8",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="chance (random @ s=2)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5))
    for ax, spec in zip(axes, MODELS, strict=True):
        s = load_summary(spec["pt"])
        panel(cast(Axes, ax), s, spec["key"])
        m = cast("dict[str, Any]", s["metrics"])
        print(
            f"{spec['key']}: jlens@2={m['jlens@2.0']['target_top5_rate_all']:.3f} "
            f"jspace_comp@2={m['jspace_comp@2.0']['target_top5_rate_all']:.3f} "
            f"nonjspace_comp@2={m['nonjspace_comp@2.0']['target_top5_rate_all']:.3f} "
            f"random@2={m['random@2.0']['target_top5_rate_all']:.3f}"
        )

    fig.suptitle(
        "Swap-injection causality: token-indexed J-lens vector vs J-space membership "
        "of a concept vector\n(Qwen2.5, chat prompts; paper Fig. 8 analog "
        "[gurnee2026-workspace §3.1])",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.022, 1, 0.97))
    fig.text(
        0.5,
        0.005,
        "data: verbal_report_chat_6c_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt "
        "(78 items, jspace_verbal_report.py CATEGORIES bank; n=100 lens)  —  "
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
