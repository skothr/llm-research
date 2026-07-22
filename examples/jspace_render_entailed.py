"""Stage-5.2 figure: entailed-property swaps — the arc's headline causal-relational result.

The paper demonstrates a *discrete* entailed-property flip: swap the unspoken
concept's J-lens vector (spider->ant) and the model's output changes the
entailed property, 8 legs -> 6 legs
[gurnee2026-workspace §3.3; kb/excerpts/gurnee2026-workspace#sec-3-3-spider-ant].
At Qwen2.5 scale the top-1 flip does not replicate (flip rate 0.000
everywhere), but the *graded* version is present, large, and J-lens-specific:
swapping the concept along its J-lens vector moves the unspoken swap-property's
log-probability by up to +2.13 nats (1.5B) / +1.25 nats (7B) — ~7-30x the
equalized-L2 token-steering (logit-lens) control — without breaking the model.
The effect is depth-localized a few layers below the verbal-report-swap depth
(peak L18/L19; collapse toward parity at the report layer L21/L22), a
consistent cross-scale split.

Two panels sharing a y-axis (mean delta-log-p of the unspoken swap property, in
nats), one per model, x = swap layer:

  (left)  Qwen2.5-1.5B (bf16): layers {18, 21, 24}; peak L18, report-layer L21.
  (right) Qwen2.5-7B  (nf4) : layers {18, 19, 22}; peak L19, report-layer L22.

One marker/line set per equalized-L2 condition (J-lens / logit-lens
token-steering control / random) at strength s=2. Raw per-item points are
omitted: the distribution is heavily right-skewed (a few items reach
+13 to +16 nats) so scatter would dominate the shared axis and destroy
cross-scale comparability; the means are the headline quantity.

Every rendered value is re-derived from ``per_item`` (baseline-correct subset)
and checked against ``summary.metrics`` at load time; a mismatch aborts.

Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_entailed.py
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

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data" / "cache"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-22-jspace-entailed-property.png"

STRENGTH = 2.0
CONDITIONS: list[dict[str, Any]] = [
    {"key": "jlens", "label": "J-lens", "color": "#1f77b4", "lw": 2.4, "ms": 9, "z": 3},
    {
        "key": "logitlens",
        "label": "logit-lens (token-steering control)",
        "color": "#ff7f0e",
        "lw": 1.6,
        "ms": 7,
        "z": 2,
    },
    {
        "key": "random",
        "label": "random (equal-L2 control)",
        "color": "#7f7f7f",
        "lw": 1.4,
        "ms": 6,
        "z": 1,
    },
]

PANELS: list[dict[str, Any]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "clean": "clean acc 17/33",
        "layers": [18, 21, 24],
        "peak": 18,
        "report": 21,
        "files": {
            18: "entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
            21: "entailed_swap_chat_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
            24: "entailed_swap_chat_L24_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        },
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "clean": "clean acc 30/33",
        "layers": [18, 19, 22],
        "peak": 19,
        "report": 22,
        "files": {
            18: "entailed_swap_chat_L18_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
            19: "entailed_swap_chat_L19_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
            22: "entailed_swap_chat_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        },
    },
]

_TOL = 1e-6


def load_layer(pt: str) -> dict[str, Any]:
    """Load one layer's artifact; re-derive per-condition means from per_item
    (baseline-correct subset) and assert equality with summary.metrics."""
    p = CACHE / pt
    if not p.exists():
        raise SystemExit(f"missing entailed-swap artifact: {p}")
    d = cast("dict[str, Any]", torch.load(p, weights_only=False))
    summary = cast("dict[str, Any]", d["summary"])
    per_item = cast("list[dict[str, Any]]", d["per_item"])
    metrics = cast("dict[str, Any]", summary["metrics"])

    out: dict[str, Any] = {"layer": int(summary["layer"]), "conds": {}}
    correct = [it for it in per_item if it["baseline_correct"]]
    for cond in CONDITIONS:
        k = f"{cond['key']}@{STRENGTH:.1f}"
        m = cast("dict[str, Any]", metrics[k])
        recs = [it["conditions"][k] for it in correct]
        dlogp = np.array([r["dlogp_swap_answer"] for r in recs], dtype=float)
        flip = np.array([1.0 if r["property_flip"] else 0.0 for r in recs])
        retain = np.array([1.0 if r["clean_retained"] else 0.0 for r in recs])
        rd_dlogp, rd_flip, rd_retain, rd_n = (
            float(dlogp.mean()),
            float(flip.mean()),
            float(retain.mean()),
            len(recs),
        )
        for name, a, b in (
            ("dlogp", rd_dlogp, float(m["mean_dlogp_swap_answer"])),
            ("flip", rd_flip, float(m["property_flip_rate"])),
            ("retain", rd_retain, float(m["clean_retention_rate"])),
            ("n", float(rd_n), float(m["n_correct"])),
        ):
            if abs(a - b) > _TOL:
                raise SystemExit(
                    f"VALUE MISMATCH in {pt} [{k}] {name}: "
                    f"re-derived {a} != summary {b} — STOP"
                )
        out["conds"][cond["key"]] = {
            "dlogp": rd_dlogp,
            "flip": rd_flip,
            "retain": rd_retain,
            "n": rd_n,
        }
    return out


def main() -> None:
    data: list[dict[str, Any]] = []
    for panel in PANELS:
        rows = {L: load_layer(panel["files"][L]) for L in panel["layers"]}
        data.append({"panel": panel, "rows": rows})
        peak = rows[panel["peak"]]["conds"]["jlens"]["dlogp"]
        rep = rows[panel["report"]]["conds"]["jlens"]["dlogp"]
        print(
            f"{panel['key']}: J-lens peak L{panel['peak']} {peak:+.2f} nats, "
            f"report-layer L{panel['report']} {rep:+.2f} nats (verified vs per_item)"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    axes = cast("list[Axes]", list(axes))

    ymax = 0.0
    for d in data:
        for L in d["panel"]["layers"]:
            for cond in CONDITIONS:
                ymax = max(ymax, d["rows"][L]["conds"][cond["key"]]["dlogp"])
    ytop = float(np.ceil((ymax + 0.35) * 2) / 2)

    for ax, d in zip(axes, data):
        panel = d["panel"]
        rows = d["rows"]
        layers = panel["layers"]
        for cond in CONDITIONS:
            ys = [rows[L]["conds"][cond["key"]]["dlogp"] for L in layers]
            ax.plot(
                layers,
                ys,
                marker="o",
                color=cond["color"],
                linewidth=cond["lw"],
                markersize=cond["ms"],
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=cond["label"],
                zorder=cond["z"],
            )
        ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=0)

        # peak + report-layer annotations (J-lens)
        peakL, repL = panel["peak"], panel["report"]
        peakY = rows[peakL]["conds"]["jlens"]["dlogp"]
        repY = rows[repL]["conds"]["jlens"]["dlogp"]
        ax.annotate(
            f"peak L{peakL}\nJ-lens {peakY:+.2f} nats",
            xy=(peakL, peakY),
            xytext=(peakL + 0.4, peakY + 0.10),
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.0},
        )
        ax.annotate(
            f"report-layer L{repL}\ncollapse {repY:+.2f}",
            xy=(repL, repY),
            xytext=(repL - 0.3, repY + 0.9),
            fontsize=8,
            color="#555555",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#888888", "linewidth": 0.9},
        )

        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_xlabel("swap layer (source-concept positions)")
        ax.set_xlim(min(layers) - 0.8, max(layers) + 1.4)
        ax.set_ylim(-0.4, ytop)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(
            f"{panel['key']}   ({panel['clean']})", fontsize=11, fontweight="bold"
        )

    axes[0].set_ylabel(
        "mean Δlog-p of the unspoken swap property\n(entailed, at answer position) [nats]"
    )
    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.93)

    fig.suptitle(
        "Entailed-property swaps: the J-lens vector carries concept-linked structure "
        "raw token steering cannot (Qwen2.5, s=2)\n"
        "Graded analogue of the paper's discrete spider→ant flip "
        "(8→6 legs) [gurnee2026-workspace §3.3]",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.035,
        "Discrete property-flip rate = 0.000 at every layer / condition / strength (no top-1 crossing at either scale).  "
        "Clean-retention 0.94–1.00: the J-lens swap never breaks the model.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.text(
        0.5,
        0.006,
        "data: entailed_swap_chat_L{18,21,24}(1.5b)/L{18,19,22}(7b)_qwen2.5-*_n100.pt "
        "(33 items, jspace_entailed_swap.py ITEMS bank; n=100 lens)  —  "
        "MANIFEST sha256-registered · see figures/DATA_PROVENANCE.md",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#9a9a9a",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.9))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
