"""Analyze the T2 delimiter-attention capture — tests P1a / P1c / P1d.

Model-free; reads emb_trace_attention.pt (the T2 capture) and
emb_trace_analysis.pt (for the T1 block-READER head ranking, used in the
reader-vs-delimiter dissociation cross-reference). Console report only; no
derived artifact is written (the capture already holds the raw sums).

Pre-registered predictions (research/arcs/03_embedding-atlas/plans/
2026-06-11-predictions.md):

  P1a  some head(s) give delimiters markedly higher attention to preceding
       delimiters than offset-matched control token pairs. Census statistic:
       per-(layer, head) delim->delim (d2d) minus offset-matched control->
       control (c2c). PASS if a distinct set of heads shows d2d markedly
       above control (criterion stated inline below).
  P1c  period->comma aggregation peaks at DEEPER layers than comma->comma
       (items interpreted before collection). PASS if the period->comma
       layer argmax is strictly deeper than the comma-side argmax.
  P1d  the rotary bands carrying the delimiter QK alignment have wavelengths
       commensurate with inter-delimiter offsets (~5-30 tokens), i.e.
       MID-frequency bands, not the fastest or the never-rotating slowest.
       PASS if the dominant excess-logit bands are mid-frequency.

Also reports the sink context (mean attention to position 0 from delimiter
vs control queries) and the delimiter-head / block-reader-head overlap.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from _emb_artifacts import load_artifact

TOP_K = 10  # heads to list for P1a
RATIO_THRESH = 3.0  # "markedly higher": d2d/c2c ratio criterion for P1a
EXCESS_MIN = 0.03  # and a floor on the absolute excess so tiny/tiny ratios don't count


def rope_wavelength(band: int, head_dim: int, theta: float) -> float:
    """Rotation wavelength (positions per full 2*pi turn) of rotary band `band`.

    Capture convention (emb_trace_attention.rotary_cos_sin):
        inv_freq[i] = 1 / theta**(2*i / head_dim)   for i = 0 .. head_dim/2 - 1
    so band i advances inv_freq[i] radians per position step, and one full
    turn (2*pi) takes  wavelength = 2*pi / inv_freq[i] = 2*pi * theta**(2*i/head_dim)
    positions. Band 0 is the fastest (wl = 2*pi ~= 6.28); the last band the
    slowest (wl ~ theta, effectively a static / DC channel).
    """
    return 2.0 * math.pi * theta ** (2.0 * band / head_dim)


def main() -> None:
    a: dict[str, Any] = load_artifact("emb_trace_attention.pt")
    an: dict[str, Any] = load_artifact("emb_trace_analysis.pt")

    n_layers = int(a["n_layers"])
    n_q = int(a["n_q_heads"])
    head_dim = int(a["head_dim"])
    theta = float(a["rope_theta"])
    n_bands = a["band_d_sum"].shape[2]

    d2d = a["d2d_sum"] / a["d2d_n"].clamp_min(1)  # (L, H) delim->delim
    c2c = a["c2c_sum"] / a["c2c_n"].clamp_min(1)  # (L, H) offset-matched ctrl
    p2c = a["p2c_sum"] / a["p2c_n"].clamp_min(1)  # (L, H) period->comma
    excess = d2d - c2c
    ratio = d2d / c2c.clamp_min(1e-9)

    off = a["offsets_seen"].float()
    print("== T2 delimiter-attention analysis (Qwen2.5-7B-Instruct, 28L x 28H) ==")
    print(
        f"  probes-derived delim->delim pairs: n={int(a['d2d_n'][0, 0])}  "
        f"period->comma pairs: n={int(a['p2c_n'][0, 0])}"
    )
    print(
        f"  inter-delimiter offset dist (tokens): "
        f"min {int(off.min())} p25 {int(torch.quantile(off, 0.25))} "
        f"median {int(off.median())} p75 {int(torch.quantile(off, 0.75))} "
        f"p90 {int(torch.quantile(off, 0.90))} max {int(off.max())} "
        f"(mean {float(off.mean()):.1f})"
    )

    # ---- P1a: delimiter-tracking head census ---------------------------------
    print("\n== P1a: per-(layer,head) delim->delim excess over offset-matched control")
    flat_ex = excess.flatten()
    top = torch.topk(flat_ex, TOP_K)
    delim_heads: list[tuple[int, int]] = []
    print(f"  top-{TOP_K} heads by excess (d2d - c2c):")
    print("    rank  head    d2d     c2c    excess   ratio")
    for rank, (v, idx) in enumerate(zip(top.values, top.indices), 1):
        li, hi = int(idx // n_q), int(idx % n_q)
        delim_heads.append((li, hi))
        print(
            f"    {rank:2d}.  L{li}H{hi:<3d} {float(d2d[li, hi]):.4f}  "
            f"{float(c2c[li, hi]):.4f}  {float(excess[li, hi]):+.4f}  "
            f"{float(ratio[li, hi]):6.2f}x"
        )
    # "markedly higher" — quantify the ratio distribution and count heads that
    # clear both the ratio floor and an absolute-excess floor.
    r_flat = ratio.flatten()
    print(
        f"  ratio distribution over all {n_layers * n_q} heads: "
        f"median {float(r_flat.median()):.2f}x  p90 {float(torch.quantile(r_flat, 0.90)):.2f}x  "
        f"max {float(r_flat.max()):.2f}x"
    )
    tracking_mask = (ratio >= RATIO_THRESH) & (excess >= EXCESS_MIN)
    n_tracking = int(tracking_mask.sum())
    print(
        f"  criterion: d2d/c2c >= {RATIO_THRESH:.0f}x AND excess >= {EXCESS_MIN:.2f}  "
        f"-> {n_tracking} heads qualify as delimiter-tracking"
    )
    tracking_layers = sorted({int(i) for i in torch.where(tracking_mask)[0]})
    print(f"  qualifying heads live in layers: {tracking_layers}")
    p1a_pass = n_tracking >= 1
    print(f"  P1a verdict: {'PASS' if p1a_pass else 'FAIL'}")

    # ---- Cross-reference: delimiter heads vs T1 block-READER heads ------------
    print("\n== Cross-ref: delimiter-tracking heads vs T1 block-READER heads (q-side)")
    reader_rows = an["readers"]["q"]["top"]
    reader_heads: list[tuple[int, int]] = [
        (int(r["layer"]), int(r["head"])) for r in reader_rows
    ]
    print("  T1 top block-reader heads (q, block/control norm ratio):")
    for r in reader_rows[:8]:
        print(f"    L{r['layer']}H{r['head']}={r['ratio']:.2f}x")
    for k in (5, 10):
        d_set = set(delim_heads[:k])
        r_set = set(reader_heads[:k])
        overlap = sorted(d_set & r_set)
        print(
            f"  top-{k} overlap: {len(overlap)}/{k}  "
            f"{['L%dH%d' % h for h in overlap] if overlap else '(none)'}"
        )
    top_reader = reader_heads[0]
    print(
        f"  headline dissociation: top block-reader L{top_reader[0]}H{top_reader[1]} "
        f"(={reader_rows[0]['ratio']:.2f}x) has delim excess "
        f"{float(excess[top_reader]):+.4f} (rank "
        f"{int((flat_ex > excess[top_reader]).sum()) + 1}/{n_layers * n_q}), "
        f"ratio {float(ratio[top_reader]):.2f}x -- "
        f"{'NOT' if top_reader not in delim_heads[:8] else 'IS'} a top-8 delimiter head"
    )

    # ---- P1c: layer profile of comma-side vs period->comma aggregation --------
    print("\n== P1c: layer profile, comma-side (delim->delim) vs period->comma")
    print(
        "  NOTE: capture has no pure comma->comma accumulator; d2d (all-delim->\n"
        "  all-delim, comma-dominated) is the comma-side proxy. Limitation flagged."
    )
    d2d_layer = d2d.max(dim=1).values  # peak head per layer
    p2c_layer = p2c.max(dim=1).values
    d2d_lmean = d2d.mean(dim=1)
    p2c_lmean = p2c.mean(dim=1)
    print("  layer | d2d(max)  p2c(max) | d2d(mean) p2c(mean)")
    for li in range(n_layers):
        mark = ""
        if li == int(d2d_layer.argmax()):
            mark += " <-d2d-peak"
        if li == int(p2c_layer.argmax()):
            mark += " <-p2c-peak"
        print(
            f"  L{li:2d}   | {float(d2d_layer[li]):.4f}   {float(p2c_layer[li]):.4f}  |"
            f" {float(d2d_lmean[li]):.4f}   {float(p2c_lmean[li]):.4f}{mark}"
        )
    d2d_peak = int(d2d_layer.argmax())
    p2c_peak = int(p2c_layer.argmax())
    print(
        f"  comma-side (d2d) peak layer: {d2d_peak}   period->comma (p2c) peak layer: {p2c_peak}"
    )
    p1c_pass = p2c_peak > d2d_peak
    print(
        f"  P1c verdict: {'PASS' if p1c_pass else 'FAIL'} "
        f"(prediction: period peaks strictly DEEPER than comma)"
    )

    # ---- P1d: RoPE-band decomposition of the delimiter QK alignment -----------
    print("\n== P1d: rotary-band contribution to delim q.k logit vs control")
    band_d = a["band_d_sum"] / a["band_n"].clamp_min(1)[:, :, None]  # (L,H,bands)
    band_c = a["band_c_sum"] / a["band_n"].clamp_min(1)[:, :, None]
    band_excess = band_d - band_c  # per-band excess logit, delim over control
    # aggregate over the top-K delimiter-tracking heads (where the alignment lives)
    agg = torch.zeros(n_bands)
    for li, hi in delim_heads:
        agg += band_excess[li, hi]
    tb = torch.topk(agg, 8)
    total_pos = float(agg.clamp_min(0).sum())
    print(f"  aggregated over the top-{TOP_K} P1a heads; top-8 excess-logit bands:")
    print("    band  wavelength(tok)  excess-logit  %of+total")
    for v, b in zip(tb.values, tb.indices):
        bi = int(b)
        wl = rope_wavelength(bi, head_dim, theta)
        frac = 100.0 * float(v) / total_pos if total_pos > 0 else 0.0
        print(f"    {bi:3d}   {wl:14.1f}  {float(v):+11.2f}   {frac:5.1f}%")
    # mid-frequency reference: which bands have wavelength in the offset range
    mid_bands = [
        b for b in range(n_bands) if 5.0 <= rope_wavelength(b, head_dim, theta) <= 30.0
    ]
    slow_bands = [
        b for b in range(n_bands) if rope_wavelength(b, head_dim, theta) > 1e4
    ]
    print(
        f"  bands whose wavelength is 5-30 tok (the P1d 'mid' target): {mid_bands} "
        f"(wl { {b: round(rope_wavelength(b, head_dim, theta), 1) for b in mid_bands} })"
    )
    mid_share = 100.0 * float(agg[mid_bands].clamp_min(0).sum()) / total_pos
    slow_share = 100.0 * float(agg[slow_bands].clamp_min(0).sum()) / total_pos
    print(
        f"  share of +excess logit from mid bands {mid_bands}: {mid_share:.1f}%  |  "
        f"from slow bands (wl>1e4, near-DC): {slow_share:.1f}%"
    )
    top_band = int(tb.indices[0])
    top_band_is_mid = 5.0 <= rope_wavelength(top_band, head_dim, theta) <= 30.0
    p1d_pass = top_band_is_mid and mid_share >= 50.0
    print(
        f"  dominant band {top_band} wavelength {rope_wavelength(top_band, head_dim, theta):.0f} tok "
        f"-> {'MID-frequency' if top_band_is_mid else 'SLOW / near-DC'}"
    )
    print(f"  P1d verdict: {'PASS' if p1d_pass else 'FAIL'}")

    # ---- Sink context --------------------------------------------------------
    d2s = (a["d2sink_sum"] / a["d2sink_n"].clamp_min(1)).mean()
    c2s = (a["c2sink_sum"] / a["c2sink_n"].clamp_min(1)).mean()
    print("\n== Sink context: mean attention to position 0")
    print(
        f"  delimiter queries {float(d2s):.3f}  vs  control queries {float(c2s):.3f}  "
        f"(delta {float(d2s - c2s):+.3f})"
    )

    print("\n== Summary ==")
    print(f"  P1a: {'PASS' if p1a_pass else 'FAIL'}  ({n_tracking} tracking heads)")
    print(
        f"  P1c: {'PASS' if p1c_pass else 'FAIL'}  (d2d peak L{d2d_peak}, p2c peak L{p2c_peak})"
    )
    print(
        f"  P1d: {'PASS' if p1d_pass else 'FAIL'}  (dominant band {top_band}, "
        f"mid-band share {mid_share:.1f}%)"
    )


if __name__ == "__main__":
    main()
