"""Corpus-invariance figure: the workspace band is fitting-corpus-invariant.

Reads the 1.5B structure-scan and lens-eval artifacts fit on two corpora
(wikitext-103 vs C4-en) and renders ONE figure with two panels:

  (a) J-space variance fraction (k=25) vs source layer, wikitext vs C4
      overlaid, annotating the early-band divergence (L0 doubles on C4:
      0.082 -> 0.174) and the identical mid-late peak (L21 = 0.124 on both).

  (b) Multihop intermediate-concept hit rate (best rank in band enters
      top-10) per depth band, wikitext vs C4, with the logit-lens rate
      shown as a reference — the J-exclusivity (logit 0.00 in early/mid)
      holds on both corpora; only the early-band J magnitude moves.

Confirms the arc's workspace-band conclusions do not depend on the fitting
corpus; corpus sensitivity is confined to early layers (the band the paper
itself flags as lens-artifact-prone [gurnee2026-workspace §4.1]).
Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_corpus_invariance.py
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
from matplotlib.patches import Patch

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data" / "cache"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "observations" / "figures"
OUT = FIGDIR / "2026-07-21-jspace-corpus-invariance.png"

HEADLINE_K = 25

STRUCT_WIKI = "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt"
STRUCT_C4 = "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"
EVAL_WIKI = "lens_eval_qwen2.5-1.5b_bf16_n100.pt"
EVAL_C4 = "lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt"

WIKI_COLOR = "#1f77b4"
C4_COLOR = "#ff7f0e"
BANDS = ("early", "mid", "late", "overall")
BAND_LABELS = ("early\n(L0-8)", "mid\n(L9-18)", "late\n(L19-26)", "overall")


def load(pt: str) -> dict[str, Any]:
    p = CACHE / pt
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")
    return cast("dict[str, Any]", torch.load(p, weights_only=False))


def varfrac(struct: dict[str, Any]) -> tuple[list[int], np.ndarray]:
    s = cast("dict[str, Any]", struct["summary"])
    layers = list(s["layers"])
    vf = np.array(s["mean_varfrac"][HEADLINE_K], dtype=float)
    return layers, vf


def band_j10(eval_art: dict[str, Any]) -> tuple[list[float], list[float]]:
    mh = cast("dict[str, Any]", eval_art["per_eval"]["multihop"])
    rj = [float(mh["rates_j"][b][10]) for b in BANDS]
    rl = [float(mh["rates_l"][b][10]) for b in BANDS]
    return rj, rl


def main() -> None:
    sw, sc = load(STRUCT_WIKI), load(STRUCT_C4)
    ew, ec = load(EVAL_WIKI), load(EVAL_C4)

    layers_w, vf_w = varfrac(sw)
    layers_c, vf_c = varfrac(sc)
    rj_w, rl_w = band_j10(ew)
    rj_c, rl_c = band_j10(ec)

    print(
        f"varfrac L0 wiki={vf_w[0]:.3f} C4={vf_c[0]:.3f} | "
        f"L21 wiki={vf_w[21]:.3f} C4={vf_c[21]:.3f}"
    )
    print(f"multihop J@10 overall wiki={rj_w[3]:.3f} C4={rj_c[3]:.3f}")

    fig, (ax_vf, ax_ev) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), width_ratios=[1.35, 1.0]
    )

    # ---- Panel (a): varfrac vs layer, both corpora ----
    ax_vf.plot(
        layers_w,
        vf_w,
        "-o",
        color=WIKI_COLOR,
        markersize=3.5,
        linewidth=1.8,
        label="wikitext-103 lens",
    )
    ax_vf.plot(
        layers_c,
        vf_c,
        "-s",
        color=C4_COLOR,
        markersize=3.5,
        linewidth=1.8,
        label="C4-en lens",
    )

    # Early-band divergence callout (L0).
    ax_vf.annotate(
        f"early band diverges:\nL0 {vf_w[0]:.3f} (wiki) vs {vf_c[0]:.3f} (C4)\n"
        "(input-register sensitive; converges by ~L17)",
        xy=(0, vf_c[0]),
        xytext=(2.0, 0.185),
        fontsize=8,
        color=C4_COLOR,
        arrowprops={"arrowstyle": "->", "color": C4_COLOR, "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#fff4e8",
            "ec": C4_COLOR,
            "alpha": 0.9,
        },
    )
    # Identical L21 peak callout.
    ax_vf.annotate(
        f"identical mid-late peak:\nL21 = {vf_w[21]:.3f} on both corpora\n(workspace band is corpus-invariant)",
        xy=(21, vf_w[21]),
        xytext=(9.0, 0.045),
        fontsize=8,
        color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#eef3f8",
            "ec": "#5a7ea6",
            "alpha": 0.9,
        },
    )
    ax_vf.axvspan(0, 16, color="#cccccc", alpha=0.18, zorder=0)
    ax_vf.set_xlabel("source layer")
    ax_vf.set_ylabel(f"J-space variance fraction\n(squared-norm, k={HEADLINE_K})")
    ax_vf.set_title(
        "(a) J-space occupancy vs layer — Qwen2.5-1.5B, n=100 lens", fontsize=11
    )
    ax_vf.set_ylim(0.0, 0.22)
    ax_vf.set_xticks(list(range(0, 27, 2)))
    ax_vf.grid(True, alpha=0.25)
    ax_vf.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ---- Panel (b): multihop J@10 per band, both corpora + logit reference ----
    x = np.arange(len(BANDS))
    w = 0.38
    ax_ev.bar(
        x - w / 2,
        rj_w,
        w,
        color=WIKI_COLOR,
        edgecolor="k",
        linewidth=0.5,
        label="J-lens, wikitext",
    )
    ax_ev.bar(
        x + w / 2,
        rj_c,
        w,
        color=C4_COLOR,
        edgecolor="k",
        linewidth=0.5,
        label="J-lens, C4-en",
    )
    # Logit-lens reference markers (wiki; C4 nearly identical) per band.
    for xi in x:
        ax_ev.plot(
            [xi - w, xi + w],
            [rl_w[xi], rl_w[xi]],
            color="#555555",
            linewidth=1.6,
            zorder=5,
        )
    for xi, (rw, rc) in enumerate(zip(rj_w, rj_c, strict=True)):
        ax_ev.annotate(
            f"{rw:.2f}",
            (xi - w / 2, rw),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
        )
        ax_ev.annotate(
            f"{rc:.2f}",
            (xi + w / 2, rc),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
        )
    ax_ev.annotate(
        "logit-lens = 0.00 in early/mid\non BOTH corpora (J-exclusive)",
        xy=(0.5, 0.0),
        xytext=(0.55, 0.30),
        fontsize=8,
        color="#333333",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#f0f0f0",
            "ec": "#555555",
            "alpha": 0.9,
        },
    )
    ax_ev.set_xticks(x)
    ax_ev.set_xticklabels(BAND_LABELS, fontsize=8.5)
    ax_ev.set_ylabel("multihop intermediate concept\nin lens top-10 (rate)")
    ax_ev.set_title("(b) Intermediate-concept surfacing per depth band", fontsize=11)
    ax_ev.set_ylim(0.0, 0.72)
    ax_ev.grid(True, axis="y", alpha=0.25)
    handles = [
        Patch(facecolor=WIKI_COLOR, edgecolor="k", label="J-lens, wikitext"),
        Patch(facecolor=C4_COLOR, edgecolor="k", label="J-lens, C4-en"),
        Line2D(
            [0], [0], color="#555555", linewidth=1.6, label="logit-lens (either corpus)"
        ),
    ]
    ax_ev.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Fitting-corpus invariance of the J-space workspace band (Qwen2.5-1.5B, n=100): "
        "early layers track input register, the mid-late band does not",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(
        0.5,
        0.006,
        "data: structure_scan_* & lens_eval_qwen2.5-1.5b_bf16_n100{,_c4en}.pt "
        "(30 held-out prompts, heldout_prompts_{wikitext103,c4en}_n30.json; n=100 lens)  —  "
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
