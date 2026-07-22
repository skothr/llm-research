"""Stage-6 figure: NLA cross-tie — where the verbalizable content lives.

Reads the stage-6 NLA cross-tie artifact (Qwen2.5-7B nf4, J-lens readout at
source-layer 19 = NLA hidden_states[20]) and renders ONE figure with two
panels:

  (a) Rank distributions: per-prompt median rank of the NLA verbalizer's
      content words inside the J-lens readout, for neutral vs concept-loaded
      prompts (log-y), against the chance rank floor and the
      concept-mismatched permutation-null median. Shows weak, concept-
      conditional, prompt-specific channel agreement (concept matched
      ~9.6k vs mismatched null ~16.4k vs chance ~76k; neutral worse than
      chance).

  (b) Carrier Jaccard: content-word overlap of the verbalization of the full
      activation vs its J-space component, its residual (component removed),
      an equal-norm random-direction-removed control, and a random-unit
      floor. The decisive comparison is residual vs the norm-matched control:
      removing the J-space component damages the verbalization no more than
      removing an equal-norm random direction (delta ~= 0).

Together: at open-model scale the NLA-verbalizable content does not
concentrate in the J-space component. Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_nla_crosstie.py
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
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "jspace" / "data" / "cache"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-21-jspace-nla-crosstie.png"

ART = "nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt"

NEUTRAL_COLOR = "#7f7f7f"
CONCEPT_COLOR = "#1f77b4"


def load() -> dict[str, Any]:
    p = CACHE / ART
    if not p.exists():
        raise SystemExit(f"missing NLA cross-tie artifact: {p}")
    return cast("dict[str, Any]", torch.load(p, weights_only=False))


def _strip(ax: Axes, x: float, vals: list[float], color: str) -> None:
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(vals)) - 0.5) * 0.22
    ax.scatter(
        np.full(len(vals), x) + jitter,
        vals,
        s=32,
        color=color,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.85,
        zorder=4,
    )


def main() -> None:
    d = load()
    s = cast("dict[str, Any]", d["summary"])
    pp = cast("list[dict[str, Any]]", d["per_prompt"])

    neutral = [float(p["nla_rank_median"]) for p in pp if p["tag"] == "neutral"]
    concept = [float(p["nla_rank_median"]) for p in pp if p["tag"] == "concept"]
    decomp = [p for p in pp if p.get("carrier_residual") is not None]
    car_comp = [float(p["carrier_component"]) for p in decomp]
    car_res = [float(p["carrier_residual"]) for p in decomp]
    car_ctrl = [float(p["carrier_resctrl"]) for p in decomp]

    chance = float(s["chance_rank_floor"])
    null_mismatched = float(s["null_concept_mismatched_rank_median"])
    med_neutral = float(s["metric1_rank_median_neutral"])
    med_concept = float(s["metric1_rank_median_concept"])
    vocab = int(s["vocab"])
    layer = int(s["jlens_layer"])
    nla_hs = int(s["nla_hidden_state"])

    print(
        f"neutral median={med_neutral:.0f} concept median={med_concept:.0f} "
        f"mismatched-null={null_mismatched:.0f} chance={chance:.0f}"
    )
    print(
        f"carrier means: component={np.mean(car_comp):.3f} residual={np.mean(car_res):.3f} "
        f"resctrl={np.mean(car_ctrl):.3f} randunit={float(s['expB_carrier_randunit_mean']):.3f} "
        f"delta={float(s['expB_carrier_delta_mean']):.3f}"
    )

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(13.5, 5.8), width_ratios=[1.0, 1.0])

    # ---- Panel (a): rank distributions ----
    ax_r.boxplot(
        [neutral, concept],
        positions=[1, 2],
        widths=0.45,
        showfliers=False,
        medianprops={"color": "k", "linewidth": 1.4},
        boxprops={"alpha": 0.5},
    )
    _strip(ax_r, 1, neutral, NEUTRAL_COLOR)
    _strip(ax_r, 2, concept, CONCEPT_COLOR)

    ax_r.axhline(
        chance,
        color="#d62728",
        linestyle="-",
        linewidth=1.4,
        label=f"chance rank floor = {chance:,.0f}",
    )
    ax_r.axhline(
        null_mismatched,
        color="#9467bd",
        linestyle="--",
        linewidth=1.4,
        label=f"concept-mismatched null median = {null_mismatched:,.0f}",
    )
    ax_r.annotate(
        f"concept matched median\n{med_concept:,.0f}\n(1.7x better than the\nmismatched null)",
        xy=(2, med_concept),
        xytext=(2.28, med_concept * 3.0),
        fontsize=8,
        color=CONCEPT_COLOR,
        arrowprops={"arrowstyle": "->", "color": CONCEPT_COLOR, "linewidth": 1.0},
    )
    ax_r.annotate(
        "neutral prose:\nno agreement\n(worse than chance)",
        xy=(1, med_neutral),
        xytext=(0.52, 2.6e4),
        fontsize=8,
        color=NEUTRAL_COLOR,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": NEUTRAL_COLOR, "linewidth": 1.0},
    )
    ax_r.set_yscale("log")
    ax_r.set_xlim(0.4, 3.1)
    ax_r.set_xticks([1, 2])
    ax_r.set_xticklabels(
        ["neutral\nprose (n=12)", "concept-loaded\n(n=12)"], fontsize=9
    )
    ax_r.set_ylabel(
        f"median rank of NLA content words\nin J-lens readout (of {vocab:,} vocab; lower = better)"
    )
    ax_r.set_title(
        f"(a) Channel agreement — J-lens readout vs NLA verbalizer\nsource-layer {layer} (= NLA hidden_states[{nla_hs}])",
        fontsize=10.5,
    )
    ax_r.grid(True, axis="y", which="both", alpha=0.22)
    ax_r.legend(loc="lower left", fontsize=8, framealpha=0.92)

    # ---- Panel (b): carrier Jaccard ----
    labels = [
        "J-space\ncomponent",
        "residual\n(component\nremoved)",
        "norm-matched\nrandom-dir\nremoved (control)",
        "random-unit\nfloor",
    ]
    means = [
        float(np.mean(car_comp)),
        float(np.mean(car_res)),
        float(np.mean(car_ctrl)),
        float(s["expB_carrier_randunit_mean"]),
    ]
    colors = ["#1f77b4", "#2ca02c", "#8c564b", "#bbbbbb"]
    x = np.arange(len(labels))
    ax_c.bar(x, means, 0.6, color=colors, edgecolor="k", linewidth=0.5, zorder=3)
    # per-prompt strips for the three measured conditions
    for xi, vals in zip((0, 1, 2), (car_comp, car_res, car_ctrl), strict=True):
        _strip(ax_c, float(xi), vals, "#333333")
    for xi, m in zip(x, means, strict=True):
        ax_c.annotate(
            f"{m:.3f}",
            (xi, m),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )

    # decisive delta annotation between residual and norm-matched control
    delta = float(s["expB_carrier_delta_mean"])
    ax_c.annotate(
        f"residual vs control:\nDelta = {delta:+.3f} (~ 0)\n"
        "removing the J-space\ncomponent hurts no more\nthan removing a random\nequal-norm direction",
        xy=(2.30, means[2] - 0.11),
        xytext=(3.42, 0.30),
        fontsize=8,
        color="#333333",
        ha="right",
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#eefaf0",
            "ec": "#2ca02c",
            "alpha": 0.92,
        },
    )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=8)
    ax_c.set_ylabel("content-word Jaccard vs\nfull-activation verbalization")
    ax_c.set_ylim(0.0, 0.85)
    ax_c.set_title(
        "(b) Where the verbalizable content lives (n=12 decomposition prompts)",
        fontsize=10.5,
    )
    ax_c.grid(True, axis="y", alpha=0.22)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#333333",
            markeredgecolor="k",
            markersize=7,
            label="per-prompt value",
        )
    ]
    ax_c.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "NLA cross-tie (Qwen2.5-7B nf4, n=100 lens): the verbalizable content does not "
        "concentrate in the J-space component [gurnee2026-workspace §2.3]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.text(
        0.5,
        0.006,
        "data: nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt "
        "(12 neutral [heldout_prompts_wikitext103_n30.json] + 12 concept [jspace_nla_crosstie.py CONCEPT_PROMPTS] + 12 decomposition; n=100 lens)  —  "
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
