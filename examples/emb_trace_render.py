"""Render tracing-phase figures (fig16-fig18).

Model-free; reads emb_trace_components.pt + emb_trace_analysis.pt.
Outputs to research/arcs/03_embedding-atlas/observations/figures/:
  fig16_carrier_sv.png      carrier recoverability by layer, block vs control
                            + in-block correlation mass (the moves-not-dissolves
                            result)
  fig17_component_deltas.png attention vs FFN residual-delta block fraction by
                            layer + static RMSNorm/write gain ratios
  fig18_sink_census.png     massive-activation dims: peak value vs first layer,
                            arc-1 sink overlap marked; P2 persistence profile
                            (delim vs ctrl vs first position)
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _emb_artifacts import FIGURES, load_artifact, warn_if_mixed_sources

DPI = 150


def main() -> None:
    warn_if_mixed_sources(["emb_trace_components.pt", "emb_trace_analysis.pt"])
    tc = load_artifact("emb_trace_components.pt")
    ta = load_artifact("emb_trace_analysis.pt")
    FIGURES.mkdir(parents=True, exist_ok=True)
    n_tok = tc["n_tokens_accumulated"]

    # ---- fig16: carrier recoverability ----------------------------------------
    bsv: torch.Tensor = tc["carrier"]["block"]["singular_values"]  # (29, 21)
    csv: torch.Tensor = tc["carrier"]["control"]["singular_values"]
    mass: torch.Tensor = tc["carrier"]["block"]["in_block_mass_frac"]
    L = bsv.shape[0]
    x = list(range(L))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(x, bsv[:, 0].numpy(), "o-", color="#3465a4", label="block dims (top SV)")
    ax.plot(
        x, csv[:, 0].numpy(), "s--", color="#888a85", label="random-21 control (top SV)"
    )
    ax.plot(
        x,
        bsv.sum(1).numpy() / 21,
        ":",
        color="#3465a4",
        alpha=0.6,
        label="block mean SV",
    )
    ax.plot(
        x,
        csv.sum(1).numpy() / 21,
        ":",
        color="#888a85",
        alpha=0.6,
        label="control mean SV",
    )
    ax.set_xlabel("layer (0 = embeddings, k = after block k)")
    ax.set_ylabel("singular value of (21 x 3584) corr matrix")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("linear recoverability of layer-0 block content")
    ax = axes[1]
    ax.plot(x, mass.numpy(), "o-", color="#cc0000")
    ax.set_xlabel("layer")
    ax.set_ylabel("corr mass in original 21 dims")
    ax.grid(alpha=0.3)
    ax.set_title("the content LEAVES the original dims by layer 1")
    fig.suptitle(
        f"fig16 — block content moves, does not dissolve ({n_tok} corpus tokens)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig16_carrier_sv.png", dpi=DPI)
    plt.close(fig)

    # ---- fig17: component deltas + static gains ----------------------------------
    comp: list[dict[str, Any]] = tc["component_stats"]
    af = torch.stack([c["attn_block_frac"] for c in comp]).mean(0)
    ff = torch.stack([c["ffn_block_frac"] for c in comp]).mean(0)
    st = tc["static_maps"]
    nl = af.shape[0]
    floor = (21 / 3584) ** 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(range(nl), af.numpy(), "o-", color="#3465a4", label="attention delta")
    ax.plot(range(nl), ff.numpy(), "s-", color="#f57900", label="FFN delta")
    ax.axhline(floor, color="gray", ls=":", label=f"random floor {floor:.3f}")
    ax.set_xlabel("layer")
    ax.set_ylabel("block-energy fraction of residual delta")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("which component writes block-aligned content")
    ax = axes[1]
    for key_b, key_c, label, color in (
        ("norm1_block", "norm1_control", "input RMSNorm gain", "#cc0000"),
        ("attn_write_block", "attn_write_control", "o_proj write", "#3465a4"),
        ("ffn_write_block", "ffn_write_control", "down_proj write", "#f57900"),
        ("ffn_read_block", "ffn_read_control", "gate/up read", "#73d216"),
    ):
        ratio = (st[key_b] / st[key_c].clamp_min(1e-12)).numpy()
        ax.plot(range(nl), ratio, "o-", ms=3, color=color, label=label)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("block / control ratio (static weights)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("static treatment of block dims per layer")
    fig.suptitle("fig17 — component attribution: norm gains + attention, FFN neutral")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig17_component_deltas.png", dpi=DPI)
    plt.close(fig)

    # ---- fig18: sink census + persistence -----------------------------------------
    mass_rows: list[dict[str, Any]] = ta["massive_dims"]
    prof = ta["block_frac_profile"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for r in mass_rows:
        color = (
            "#cc0000"
            if r["in_arc1_sinks"]
            else ("#3465a4" if not r["in_block"] else "#73d216")
        )
        ax.scatter(r["first_layer"], abs(r["peak_value"]), color=color, s=30)
        if abs(r["peak_value"]) > 700:
            ax.annotate(
                str(r["dim"]),
                (r["first_layer"], abs(r["peak_value"])),
                fontsize=7,
                xytext=(4, 2),
                textcoords="offset points",
            )
    ax.set_yscale("log")
    ax.set_xlabel("first layer in census")
    ax.set_ylabel("|peak activation| (log)")
    ax.set_title(
        "massive-activation dims: red = arc-1 sink dims,\n"
        "green = W_E block dims (layer-0 echoes only)"
    )
    ax.grid(alpha=0.3)
    ax = axes[1]
    Lp = prof["delim"].shape[0]
    ax.plot(range(Lp), prof["delim"].numpy(), "o-", color="#cc0000", label="delimiters")
    ax.plot(range(Lp), prof["ctrl"].numpy(), "s-", color="#3465a4", label="controls")
    ax.plot(
        range(Lp), prof["first"].numpy(), "^-", color="#888a85", label="first position"
    )
    ax.axhline(
        ta["random_baseline"],
        color="gray",
        ls=":",
        label=f"random floor {ta['random_baseline']:.3f}",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("residual block-energy fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("P2: delimiter-specific block energy collapses at layer 1")
    fig.suptitle("fig18 — sink machinery vs the W_E block (dimensionally disjoint)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig18_sink_census.png", dpi=DPI)
    plt.close(fig)

    for f in ("fig16_carrier_sv", "fig17_component_deltas", "fig18_sink_census"):
        print(f"wrote {FIGURES / f}.png")


if __name__ == "__main__":
    main()
