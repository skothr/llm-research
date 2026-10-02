"""Stage-3 figure: the "unspoken words" layer x token trajectory grid.

Renders the paper's signature visualization [gurnee2026-workspace §2.1, Fig.
"unspoken words"]: for a single held-out prompt at one scored position, the
per-layer top-5 readout tokens of each lens, laid out as a layer x rank text
grid. The Jacobian lens surfaces the *currently-held intermediate concept* at
mid depth; the logit-lens baseline stays incoherent (whitespace / symbol-run /
cross-script junk) until the last few layers.

This figure was previously un-renderable: the stage-3 ``readout_scan_*.pt``
artifacts stored only per-(layer, position) SCALARS (Spearman, ranks, top-1
ids), not the per-layer top-k token strings the grid needs (see the data-
availability note in ``jspace_render_emergence.py``). The rich-capture patch to
``jspace_readout_scan.py`` added ``topk_ids/probs/strs_{j,l,model}`` per prompt,
which this script consumes.

Representative selection (deterministic, hardcoded): prompt 0, position index 4
(= token 127 of 224), a mid-context position inside a wikitext-103 passage about
critical book reviews ("...ran a highly critical review...", "...favorably
reviewed...", "The New York Review of Books"). At that position the J-lens reads
out a clean semantic trajectory -- criticism -> critiques -> allegations ->
commentary -> articles -> reviews -> essays -- matching the passage's held
concept, at BOTH scales (1.5B emerging ~L12-17, 7B ~L20-21), while the logit
lens shows junk until ~L21-23. The final prompt position was rejected: every
held-out prompt ends at a sentence boundary ('. " \\n'), where both lenses
collapse to meta-tokens (Question/Answer/What) and the concept signal vanishes.

Layout: 2 rows (Qwen2.5-1.5B bf16, Qwen2.5-7B nf4) x 2 cols (J-lens, logit-lens).
Each panel is a layer (0 bottom -> deepest top) x top-5-rank text grid; cell
opacity tracks readout probability; whitespace / punctuation / symbol-only
"junk" tokens are dimmed grey italic to separate them from content words.

Deterministic, Agg backend, single PNG.

Usage:
    python examples/jspace_render_trajectory.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import json
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
from matplotlib.patches import Rectangle

from _jspace_paths import resolve

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "observations" / "figures"
PROMPTS = (
    _REPO_ROOT
    / "research"
    / "arcs"
    / "04_jspace"
    / "data"
    / "heldout_prompts_wikitext103_n30.json"
)
OUT = FIGDIR / "2026-07-21-jspace-unspoken-words.png"

# Representative held-out example (see module docstring for the rationale).
PROMPT_IDX = 0
POS_IDX = 4  # index into the scan's 9 positions (= token 127 of 224)
TOPK = 5

MODELS: list[dict[str, str]] = [
    {
        "key": "Qwen2.5-1.5B (bf16)",
        "pt": "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
    },
    {
        "key": "Qwen2.5-7B (nf4)",
        "pt": "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
    },
]

CONTENT_RGB = "#12304a"  # dark navy for content tokens
JUNK_RGB = "#9aa0a6"  # grey for whitespace / punctuation / symbol-run tokens
HEAT_RGB = (0.20, 0.55, 0.62)  # teal cell wash, alpha tracks probability


def load_prompt(idx: int) -> str:
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = data["prompts"] if isinstance(data, dict) else data
    return str(prompts[idx])


def load_scan(pt: str) -> dict[str, Any]:
    p = resolve(pt)
    if not p.exists():
        raise SystemExit(f"missing readout-scan artifact: {p}")
    d = cast("dict[str, Any]", torch.load(p, weights_only=False))
    pp0 = d["per_prompt"][PROMPT_IDX]
    if "topk_strs_j" not in pp0:
        raise SystemExit(
            f"{p} lacks rich per_prompt keys (topk_strs_j); re-run the scan "
            "with the rich-capture jspace_readout_scan.py."
        )
    return d


_CJK_MIN = 0x2E80  # CJK radicals and up (also catches fullwidth punctuation)


def is_junk(tok: str) -> bool:
    """A token is 'junk' unless it carries a Latin (ASCII) letter.

    Whitespace, newlines, punctuation, symbol-runs (``[`, ``\\``, ``//...``,
    ``℠``) and non-Latin-script tokens (CJK such as ``描述``/``诋``) are all the
    logit-lens's incoherent output; only Latin content words (``criticism``,
    ``reviews``, incl. leading-space forms like `` articles``) are 'content'.
    Rendering CJK as junk both matches the content-vs-noise story and avoids
    DejaVu-Sans tofu boxes (see :func:`disp`)."""
    return not any("a" <= ch.lower() <= "z" for ch in tok)


def disp(tok: str) -> str:
    """Human-legible cell label: whitespace made visible, CJK/exotic glyphs
    (which DejaVu Sans cannot render) collapsed to ``<cjk>``, length-capped."""
    t = tok.replace("\n", "⏎").replace("\r", "⏎").replace("\t", "⇥")
    had_exotic = any(ord(c) >= _CJK_MIN for c in t)
    t = "".join(c for c in t if ord(c) < _CJK_MIN)
    st = t.strip()
    if st == "":
        if had_exotic:
            return "<cjk>"
        st = "␣" if tok.strip(" ") == "" else t  # visible space
    elif had_exotic:
        st = st + "+<cjk>"
    if len(st) > 11:
        st = st[:10] + "…"
    return st


def emergence_layer(top1_junk: list[bool], layers: list[int]) -> int | None:
    """Shallowest layer (ascending) from which top-1 stays content-bearing."""
    for i in range(len(layers)):
        if not any(top1_junk[i:]):
            return layers[i]
    return None


def draw_panel(
    ax: Axes,
    strs: list[list[list[str]]],
    probs: np.ndarray,
    layers: list[int],
    title: str,
) -> None:
    """One layer x top-5 text grid. ``strs`` is [layer][pos][k]; ``probs`` is
    [n_layers, n_pos, k]. Layer 0 sits at the bottom, deepest at the top."""
    n_layers = len(layers)
    ax.set_xlim(0, TOPK)
    ax.set_ylim(0, n_layers)
    ax.set_xticks([r + 0.5 for r in range(TOPK)])
    ax.set_xticklabels([f"top-{r + 1}" for r in range(TOPK)], fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    yticks = [i + 0.5 for i in range(n_layers) if layers[i] % 2 == 0]
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f"L{layers[i]}" for i in range(n_layers) if layers[i] % 2 == 0], fontsize=7
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    top1_junk: list[bool] = []
    for i in range(n_layers):
        cell_toks = strs[i][POS_IDX]
        top1_junk.append(is_junk(cell_toks[0]))
        for r in range(TOPK):
            tok = cell_toks[r]
            pr = float(probs[i, POS_IDX, r])
            junk = is_junk(tok)
            x = r + 0.5
            y = i + 0.5
            if not junk:
                ax.add_patch(
                    Rectangle(
                        (r + 0.04, i + 0.06),
                        0.92,
                        0.88,
                        facecolor=HEAT_RGB,
                        edgecolor="none",
                        alpha=0.10 + 0.55 * pr**0.5,
                        zorder=1,
                    )
                )
                color = CONTENT_RGB
                alpha = 0.55 + 0.45 * pr**0.5
                style = "normal"
                weight = "bold" if r == 0 else "normal"
            else:
                color = JUNK_RGB
                alpha = 0.55
                style = "italic"
                weight = "normal"
            ax.text(
                x,
                y,
                disp(tok),
                ha="center",
                va="center",
                fontsize=6.6,
                color=color,
                alpha=alpha,
                style=style,
                fontweight=weight,
                zorder=2,
            )

    emg = emergence_layer(top1_junk, layers)
    if emg is not None:
        yline = layers.index(emg)
        ax.axhline(yline, color="#c94b2b", linestyle="--", linewidth=1.1, zorder=3)
        # Centre the label ON the dashed line (va="center") with a white bbox so it
        # reads as a line annotation: it straddles the inter-row seam and covers
        # only cell background, never the tokens (which sit at row centres ±0.5).
        ax.text(
            TOPK - 0.06,
            yline,
            f"top-1 turns content → L{emg}",
            ha="right",
            va="center",
            fontsize=6.6,
            color="#c94b2b",
            fontweight="bold",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "#c94b2b",
                "alpha": 0.9,
            },
        )


def main() -> None:
    prompt = load_prompt(PROMPT_IDX)
    scans = [load_scan(spec["pt"]) for spec in MODELS]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 15.5))
    lens_cols = [
        ("topk_strs_j", "topk_probs_j", "J-lens"),
        ("topk_strs_l", "topk_probs_l", "logit-lens"),
    ]

    for row, (spec, d) in enumerate(zip(MODELS, scans, strict=True)):
        pp = d["per_prompt"][PROMPT_IDX]
        layers = list(d["summary"]["layers"])
        positions = list(pp["positions"])
        mtop = pp["topk_strs_model"][POS_IDX][:3]
        for col, (skey, pkey, lname) in enumerate(lens_cols):
            ax = cast(Axes, axes[row, col])
            draw_panel(
                ax,
                pp[skey],
                np.asarray(pp[pkey]),
                layers,
                f"{spec['key']}  —  {lname}",
            )
        # model reference caption on the left panel of each row
        cast(Axes, axes[row, 0]).set_ylabel(
            f"source layer  (deepest at top)\nmodel next-token top-3 here: "
            f"{', '.join(repr(t) for t in mtop)}",
            fontsize=8.5,
        )
        print(
            f"{spec['key']}: pos idx {POS_IDX} = token {positions[POS_IDX]} of "
            f"{pp['seq_len']}; model top-3 {mtop}"
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=CONTENT_RGB,
            markeredgecolor="none",
            markersize=11,
            label="content token (opacity = readout prob.)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=JUNK_RGB,
            markeredgecolor="none",
            markersize=11,
            label="junk: whitespace / punctuation / symbol-run (dimmed)",
        ),
        Line2D(
            [0],
            [0],
            color="#c94b2b",
            linestyle="--",
            linewidth=1.4,
            label="layer where top-1 first stays content-bearing",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.028),
    )

    snippet = "...ran a highly critical review by Bill Farrell ... favorably reviewed ... The New York Review of Books..."
    fig.suptitle(
        "The “unspoken words” trajectory: per-layer top-5 lens readout at one held-out position\n"
        "Qwen2.5, n=100 lenses — prompt 0, token 127/224 (mid-context; held concept = critical book reviews)"
        " [gurnee2026-workspace §2.1]",
        fontsize=12.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        snippet,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.text(
        0.5,
        0.006,
        "data: readout_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt "
        "(rich topk_strs; prompt 0, token 127/224, heldout_prompts_wikitext103_n30.json; n=100 lens)  —  "
        "MANIFEST sha256-registered · see figures/DATA_PROVENANCE.md",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#9a9a9a",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.93))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
