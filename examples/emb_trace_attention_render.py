"""Render T2 delimiter-attention figures (fig19-fig21).

Model-free; reads emb_trace_attention.pt (the T2 capture) and
emb_trace_analysis.pt (the T1 block-READER head ranking, for the
reader-vs-delimiter dissociation overlay). Mirrors the P1a/P1c/P1d
derivations in emb_trace_attention_analyze.py so figures and console report
stay locked to the same statistic.

Outputs to research/arcs/03_embedding-atlas/observations/figures/:
  fig19_delim_head_census.png  P1a: per-head delim->delim (d2d) vs offset-
                               matched control (c2c) attention, 26 qualifying
                               heads highlighted, T1 block-reader heads ringed
                               to show the reader/delimiter dissociation;
                               excess-by-layer panel with L0H15 marked.
  fig20_delim_layer_profile.png P1c: layer profile of comma-side (d2d) vs
                               period->comma (p2c) aggregation, max-head and
                               mean-head panels, peak layers marked (both L0 —
                               the falsification).
  fig21_delim_band_decomp.png  P1d: per-band excess q.k logit vs rotary
                               wavelength (log-x), aggregated over the top-10
                               P1a heads, with the measured inter-delimiter
                               offset distribution overlaid so the mid-band /
                               near-DC mismatch is visible.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact
from emb_trace_attention_analyze import (
    EXCESS_MIN,
    RATIO_THRESH,
    TOP_K,
    rope_wavelength,
)

DPI = 150

# style palette shared with emb_trace_render.py
C_BLOCK = "#3465a4"  # blue
C_DELIM = "#cc0000"  # red
C_PERIOD = "#f57900"  # orange
C_CTRL = "#888a85"  # gray
C_QUAL = "#cc0000"  # qualifying delimiter head
C_READER = "#204a87"  # reader-head ring


def main() -> None:
    a = load_artifact("emb_trace_attention.pt")
    an = load_artifact("emb_trace_analysis.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)

    n_layers = int(a["n_layers"])
    n_q = int(a["n_q_heads"])
    head_dim = int(a["head_dim"])
    theta = float(a["rope_theta"])
    n_bands = a["band_d_sum"].shape[2]

    d2d = a["d2d_sum"] / a["d2d_n"].clamp_min(1)  # (L, H)
    c2c = a["c2c_sum"] / a["c2c_n"].clamp_min(1)
    p2c = a["p2c_sum"] / a["p2c_n"].clamp_min(1)
    excess = d2d - c2c
    ratio = d2d / c2c.clamp_min(1e-9)
    n_pairs = int(a["d2d_n"][0, 0])
    n_p2c = int(a["p2c_n"][0, 0])

    # top-K delimiter heads by excess (same ordering as the analysis)
    top = torch.topk(excess.flatten(), TOP_K)
    delim_heads = [(int(i // n_q), int(i % n_q)) for i in top.indices]
    tracking_mask = (ratio >= RATIO_THRESH) & (excess >= EXCESS_MIN)
    n_tracking = int(tracking_mask.sum())

    # T1 block-reader heads (q-side)
    reader_rows = an["readers"]["q"]["top"]
    reader_heads = [(int(r["layer"]), int(r["head"])) for r in reader_rows]
    top_reader = reader_heads[0]  # L0H15
    tr_rank = int((excess.flatten() > excess[top_reader]).sum()) + 1

    # ---- fig19: P1a head census + reader/delimiter dissociation ---------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    c2c_f = c2c.flatten().numpy()
    d2d_f = d2d.flatten().numpy()
    qual_f = tracking_mask.flatten().numpy()
    ax.scatter(
        c2c_f[~qual_f],
        d2d_f[~qual_f],
        s=10,
        color=C_CTRL,
        alpha=0.4,
        label=f"other heads ({n_layers * n_q - n_tracking})",
    )
    ax.scatter(
        c2c_f[qual_f],
        d2d_f[qual_f],
        s=26,
        color=C_QUAL,
        label=f"delimiter-tracking ({n_tracking})",
    )
    lim = max(float(d2d.max()), float(c2c.max())) * 1.05
    ax.plot([0, lim], [0, lim], ":", color="black", lw=1, label="d2d = c2c")
    ax.plot(
        [0, lim / RATIO_THRESH],
        [0, lim],
        "--",
        color="#555753",
        lw=1,
        label=f"d2d = {RATIO_THRESH:.0f}x c2c (criterion)",
    )
    # ring the T1 block-reader heads to show they are a different population
    rc2c = [float(c2c[h]) for h in reader_heads]
    rd2d = [float(d2d[h]) for h in reader_heads]
    ax.scatter(
        rc2c,
        rd2d,
        s=90,
        facecolors="none",
        edgecolors=C_READER,
        linewidths=1.6,
        label="T1 block-reader heads",
    )
    for (li, hi), cx, cy in zip(reader_heads, rc2c, rd2d):
        if (li, hi) in (top_reader, delim_heads[0]):
            ax.annotate(
                f"L{li}H{hi}",
                (cx, cy),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
    dh0 = delim_heads[0]
    ax.annotate(
        f"L{dh0[0]}H{dh0[1]}",
        (float(c2c[dh0]), float(d2d[dh0])),
        fontsize=7,
        xytext=(4, -8),
        textcoords="offset points",
    )
    ax.set_xlabel("control->control attention (offset-matched)")
    ax.set_ylabel("delim->delim attention")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title("P1a: delimiters over-attend to preceding delimiters")

    ax = axes[1]
    layers = torch.arange(n_layers).view(-1, 1).expand(n_layers, n_q).flatten().numpy()
    ex_f = excess.flatten().numpy()
    ax.scatter(layers[~qual_f], ex_f[~qual_f], s=10, color=C_CTRL, alpha=0.4)
    ax.scatter(
        layers[qual_f],
        ex_f[qual_f],
        s=26,
        color=C_QUAL,
        label="delimiter-tracking head",
    )
    ax.axhline(
        EXCESS_MIN, color="#555753", ls="--", lw=1, label=f"excess floor {EXCESS_MIN}"
    )
    ax.scatter(
        [top_reader[0]],
        [float(excess[top_reader])],
        s=90,
        facecolors="none",
        edgecolors=C_READER,
        linewidths=1.8,
    )
    ax.annotate(
        f"L{top_reader[0]}H{top_reader[1]}\n(top block-reader,\nexcess rank {tr_rank}/{n_layers * n_q})",
        (top_reader[0], float(excess[top_reader])),
        fontsize=7,
        xytext=(18, 6),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=C_READER, lw=0.8),
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("d2d - c2c excess")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("qualifying heads concentrate in layers 0-3 (+1 at L10)")

    fig.suptitle(
        f"fig19 — P1a PASS: {n_tracking}/{n_layers * n_q} delimiter-tracking heads; "
        f"reader != delimiter populations ({n_pairs} delim pairs)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig19_delim_head_census.png", dpi=DPI)
    plt.close(fig)

    # ---- fig20: P1c layer profile, comma-side (d2d) vs period->comma (p2c) -----
    d2d_max = d2d.max(dim=1).values.numpy()
    p2c_max = p2c.max(dim=1).values.numpy()
    d2d_mean = d2d.mean(dim=1).numpy()
    p2c_mean = p2c.mean(dim=1).numpy()
    d2d_peak = int(d2d.max(dim=1).values.argmax())
    p2c_peak = int(p2c.max(dim=1).values.argmax())
    x = list(range(n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, dvec, pvec, tag in (
        (axes[0], d2d_max, p2c_max, "max head"),
        (axes[1], d2d_mean, p2c_mean, "mean head"),
    ):
        ax.plot(x, dvec, "o-", color=C_DELIM, label="comma-side (d2d)")
        ax.plot(x, pvec, "s-", color=C_PERIOD, label="period->comma (p2c)")
        ax.axvline(
            d2d_peak, color=C_DELIM, ls=":", alpha=0.6, label=f"d2d peak L{d2d_peak}"
        )
        ax.axvline(
            p2c_peak, color=C_PERIOD, ls=":", alpha=0.6, label=f"p2c peak L{p2c_peak}"
        )
        ax.set_xlabel("layer")
        ax.set_ylabel(f"attention weight ({tag})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{tag}: both peak at layer {d2d_peak}")
    fig.suptitle(
        "fig20 — P1c FAIL: period aggregation does NOT peak deeper than comma "
        f"(both peak L0; {n_p2c} period->comma pairs)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig20_delim_layer_profile.png", dpi=DPI)
    plt.close(fig)

    # ---- fig21: P1d rotary-band decomposition ---------------------------------
    band_d = a["band_d_sum"] / a["band_n"].clamp_min(1)[:, :, None]
    band_c = a["band_c_sum"] / a["band_n"].clamp_min(1)[:, :, None]
    band_excess = band_d - band_c
    agg = torch.zeros(n_bands)
    for li, hi in delim_heads:
        agg += band_excess[li, hi]
    wl = torch.tensor([rope_wavelength(b, head_dim, theta) for b in range(n_bands)])
    total_pos = float(agg.clamp_min(0).sum())
    mid_bands = [b for b in range(n_bands) if 5.0 <= float(wl[b]) <= 30.0]
    slow_bands = [b for b in range(n_bands) if float(wl[b]) > 1e4]
    mid_share = 100.0 * float(agg[mid_bands].clamp_min(0).sum()) / total_pos
    slow_share = 100.0 * float(agg[slow_bands].clamp_min(0).sum()) / total_pos
    top_band = int(agg.argmax())

    off = a["offsets_seen"].float()
    off_med = float(off.median())
    off_p25 = float(torch.quantile(off, 0.25))
    off_p75 = float(torch.quantile(off, 0.75))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    agg_np = agg.numpy()
    wl_np = wl.numpy()
    pos = agg_np > 0
    ax.vlines(wl_np, 0, agg_np, color=C_CTRL, lw=1, alpha=0.6)
    ax.scatter(
        wl_np[pos], agg_np[pos], s=30, color=C_BLOCK, zorder=3, label="per-band excess"
    )
    ax.scatter(wl_np[~pos], agg_np[~pos], s=30, color=C_CTRL, zorder=3)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axhline(0, color="black", lw=0.6)

    # shade the mid-frequency target band (5-30 tok) and the offset IQR
    ax.axvspan(5.0, 30.0, color=C_PERIOD, alpha=0.12, label="P1d mid target 5-30 tok")
    ax.axvspan(off_p25, off_p75, color=C_DELIM, alpha=0.18, label="inter-delim IQR")
    ax.axvline(
        off_med, color=C_DELIM, lw=1.4, label=f"inter-delim median {off_med:.0f} tok"
    )

    ax.annotate(
        f"near-DC bands (wl>1e4 tok)\ncarry {slow_share:.1f}% of +excess\n"
        f"dominant band {top_band} (wl {float(wl[top_band]):.0f} tok)",
        (float(wl[top_band]), float(agg[top_band])),
        fontsize=8,
        xytext=(-160, -10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    ax.annotate(
        f"mid bands 0-7\ncarry {mid_share:.1f}%",
        (14.9, float(agg[4])),
        fontsize=8,
        xytext=(-10, 60),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=C_PERIOD, lw=0.8),
    )
    ax.set_xlabel("rotary-band wavelength (tokens per 2*pi turn, symlog)")
    ax.set_ylabel("delim - control q.k logit, summed over top-10 heads")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title(
        "fig21 — P1d FAIL: delimiter QK match lives in near-DC (static-content) "
        "bands, not the offset-matched mid bands"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig21_delim_band_decomp.png", dpi=DPI)
    plt.close(fig)

    for f in (
        "fig19_delim_head_census",
        "fig20_delim_layer_profile",
        "fig21_delim_band_decomp",
    ):
        print(f"wrote {FIGURES / f}.png")


if __name__ == "__main__":
    main()
