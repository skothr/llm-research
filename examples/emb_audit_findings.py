"""Regression audit for the embedding-atlas arc — re-derives every
load-bearing number cited in research/arcs/03_embedding-atlas/observations/
from the committed .pt artifacts and asserts it against an expected constant.

    python examples/emb_audit_findings.py
    # expect: SUMMARY:  N PASS  |  0 FAIL

Artifacts resolve cache-first then committed-data (via _emb_artifacts), so
the audit replays from a clean clone after `git lfs pull`.

What a full PASS does NOT verify (be honest about scope):
  * capture-protocol correctness — the audit consumes the artifacts as given;
    a wrong slice (wrong matrix, wrong rows) would be consistently wrong.
  * that expected constants were transcribed correctly from artifact to prose
    in BOTH directions — the script locks artifact->constant; prose->constant
    agreement is maintained by hand.
  * methodological choices (MIN_CLASS_N=5, the primary-variant policy,
    near-zero threshold 1e-3) — it validates they were applied, not chosen well.
  * regeneration from the model — W_E/W_U are not committed (see the arc
    README's stated deviation); re-capture requires the pinned HF revision.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from _emb_artifacts import find_artifact, load_artifact
from emb_trace_attention_analyze import rope_wavelength

PASS = 0
FAIL = 0


def claim(section: str, name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {section}: {name}" + (f"  ({detail})" if detail else ""))


def near(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def main() -> int:
    # ---- AUDIT 1: battery + capture invariants ------------------------------
    b = load_artifact("emb_battery_vectors.pt")
    rows: list[dict[str, Any]] = b["rows"]
    claim("1 battery", "anchor-variant rows == 1062", len(rows) == 1062, str(len(rows)))
    n_primary = sum(1 for r in rows if r["is_primary"])
    claim("1 battery", "primary rows == 665", n_primary == 665, str(n_primary))
    claim(
        "1 battery", "dropped words == 25", len(b["drops"]) == 25, str(len(b["drops"]))
    )
    claim(
        "1 battery",
        "E/U slices shaped (1062, 3584) bf16",
        tuple(b["E"].shape) == (1062, 3584)
        and tuple(b["U"].shape) == (1062, 3584)
        and b["E"].dtype == torch.bfloat16,
    )
    claim(
        "1 battery",
        "revision pinned a09a3545",
        str(b["revision"]).startswith("a09a3545"),
        str(b["revision"])[:12],
    )
    drop_words = {d["word"] for d in b["drops"]}
    claim(
        "1 battery",
        "multi-digit numbers all dropped (digit-split tokenizer)",
        {"42", "100", "1000", "2024", "3.14"} <= drop_words,
    )

    g = load_artifact("emb_global_stats.pt")
    n_real: int = g["n_real"]
    n_vocab: int = g["n_vocab"]
    claim(
        "1 capture",
        "padded rows == 399",
        n_vocab - n_real == 399,
        str(n_vocab - n_real),
    )
    claim(
        "1 capture",
        "padded rows are exactly zero (norm median 0)",
        float(g["norms_E"][n_real:].median()) == 0.0,
    )
    nz = int(g["near_zero_real_ids"].numel())
    claim("1 capture", "near-zero real rows == 1959", nz == 1959, str(nz))
    claim(
        "1 capture",
        "real-row norm median ~0.859",
        near(float(g["norms_E"][:n_real].median()), 0.859, 0.002),
        f"{float(g['norms_E'][:n_real].median()):.3f}",
    )

    # ---- AUDIT 2: isotropy (the anisotropy null) -----------------------------
    claim(
        "2 isotropy",
        "random-pair cosine (raw) +0.0097",
        near(float(g["rand_cos_raw"].mean()), 0.0097, 0.0005),
        f"{float(g['rand_cos_raw'].mean()):+.4f}",
    )
    claim(
        "2 isotropy",
        "random-pair cosine (centered) +0.0007",
        near(float(g["rand_cos_centered"].mean()), 0.0007, 0.0005),
        f"{float(g['rand_cos_centered'].mean()):+.4f}",
    )
    claim(
        "2 isotropy",
        "mean cos-to-mu +0.0980",
        near(float(g["cos_to_mu"].mean()), 0.0980, 0.001),
        f"{float(g['cos_to_mu'].mean()):+.4f}",
    )

    # ---- AUDIT 3: PCA spectrum ------------------------------------------------
    ev = g["evals_centered"].clamp_min(0)
    fr = ev / ev.sum()
    pr = float(ev.sum() ** 2 / (ev**2).sum())
    claim(
        "3 pca",
        "PC1 variance fraction 1.21%",
        near(float(fr[0]), 0.0121, 0.0003),
        f"{float(fr[0]):.4f}",
    )
    claim(
        "3 pca",
        "top-10 variance 4.68%",
        near(float(fr[:10].sum()), 0.0468, 0.0006),
        f"{float(fr[:10].sum()):.4f}",
    )
    claim(
        "3 pca",
        "top-50 variance 12.13%",
        near(float(fr[:50].sum()), 0.1213, 0.001),
        f"{float(fr[:50].sum()):.4f}",
    )
    claim("3 pca", "participation ratio ~1003", near(pr, 1003, 5), f"{pr:.0f}")

    # ---- AUDIT 4: E vs U orthogonality ----------------------------------------
    eu = g["eu_cos"][:n_real]
    claim(
        "4 e-vs-u",
        "per-token cos(E,U) mean +0.0017",
        near(float(eu.mean()), 0.0017, 0.0005),
        f"{float(eu.mean()):+.4f}",
    )
    claim("4 e-vs-u", "no token with cos(E,U) > 0.5", float((eu > 0.5).sum()) == 0)
    claim(
        "4 e-vs-u",
        "U is trained, not degenerate (norm median ~0.697)",
        near(float(g["norms_U"][:n_real].median()), 0.697, 0.002),
        f"{float(g['norms_U'][:n_real].median()):.3f}",
    )
    # U-space carries category structure of its own (month within-cos ~+0.439)
    prim = [i for i, r in enumerate(rows) if r["is_primary"] and r["class"] == "month"]
    Un = b["U"][prim].to(torch.float32)
    Un = Un / Un.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sub = Un @ Un.T
    iu_idx = torch.triu_indices(len(prim), len(prim), offset=1)
    month_u = float(sub[iu_idx[0], iu_idx[1]].mean())
    claim(
        "4 e-vs-u",
        "month within-cos in U-space +0.439",
        near(month_u, 0.439, 0.005),
        f"{month_u:+.3f}",
    )

    # ---- AUDIT 5: category gaps, re-derived from battery vectors --------------
    # First-principles re-derivation (NOT read from emb_category_stats.pt):
    # centered space, primary rows, within minus between mean cosine.
    glob_mu = g["mu"]
    primary = [i for i, r in enumerate(rows) if r["is_primary"]]
    X = b["E"][primary].to(torch.float32) - glob_mu
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = Xn @ Xn.T
    class_of = [rows[i]["class"] for i in primary]

    def gap_of(cname: str) -> float:
        idx = torch.tensor([k for k, c in enumerate(class_of) if c == cname])
        mask = torch.zeros(len(class_of), dtype=torch.bool)
        mask[idx] = True
        sub = cos[idx][:, idx]
        iu2 = torch.triu_indices(len(idx), len(idx), offset=1)
        within = float(sub[iu2[0], iu2[1]].mean())
        between = float(cos[idx][:, ~mask].mean())
        return within - between

    for cname, expected in (
        ("number", 0.428),
        ("month", 0.416),
        ("day_of_week", 0.400),
        ("auxiliary", 0.286),
        ("pronoun", 0.266),
        ("case_lower", 0.023),
    ):
        got = gap_of(cname)
        claim(
            "5 category",
            f"{cname} within-between gap {expected:+.3f} (centered)",
            near(got, expected, 0.003),
            f"{got:+.3f}",
        )
    # derived artifact agrees with first principles
    cs = load_artifact("emb_category_stats.pt")
    stored = cs["spaces"]["centered"]["per_class"]["number"]["gap"]
    claim(
        "5 category",
        "emb_category_stats.pt agrees with re-derivation (number)",
        near(float(stored), gap_of("number"), 1e-4),
        f"{float(stored):+.4f}",
    )

    # ---- AUDIT 6: neighbor content (identities, not non-emptiness) ------------
    probes = load_artifact("emb_neighbor_probes.pt")
    by_label = {p["label"]: p for p in probes["probes"]}
    fr_raw = by_label["country_space"]["raw"]["strings"]
    claim(
        "6 neighbors",
        "' France' raw top-1 is 'France'",
        fr_raw[0] == "France",
        repr(fr_raw[0]),
    )
    claim(
        "6 neighbors",
        "' France' raw top-15 contains Chinese exonym 法国",
        "法国" in fr_raw,
    )
    pa_raw = by_label["capital_space"]["raw"]["strings"]
    claim(
        "6 neighbors",
        "' Paris' raw top-1 is 'Paris'",
        pa_raw[0] == "Paris",
        repr(pa_raw[0]),
    )
    claim("6 neighbors", "' Paris' raw top-5 contains 巴黎", "巴黎" in pa_raw[:5])
    dog_raw = by_label["animal_space"]["raw"]["strings"]
    claim(
        "6 neighbors",
        "' dog' raw top-1 is ' Dog'",
        dog_raw[0] == " Dog",
        repr(dog_raw[0]),
    )

    # ---- AUDIT 7: pair-direction consistency -----------------------------------
    pd = load_artifact("emb_pair_directions.pt")
    for kind, expected_c, expected_s in (
        ("lang_of", 0.4896, 0.4503),
        ("past", 0.4228, 0.3684),
        ("gender", 0.3859, 0.3380),
        ("plural", 0.3097, 0.2618),
        ("space_twin", 0.1968, 0.1577),
    ):
        k = pd["kinds"][kind]
        claim(
            "7 pairs",
            f"{kind} consistency {expected_c:+.4f}",
            near(k["consistency"], expected_c, 0.001),
            f"{k['consistency']:+.4f}",
        )
        claim(
            "7 pairs",
            f"{kind} shuffle baseline {expected_s:+.4f}",
            near(k["shuffle_consistency_mean"], expected_s, 0.003),
            f"{k['shuffle_consistency_mean']:+.4f}",
        )
    claim(
        "7 pairs",
        "every kind beats its shuffle baseline",
        all(
            k["consistency"] > k["shuffle_consistency_mean"]
            for k in pd["kinds"].values()
        ),
    )

    # ---- AUDIT 8: full-vocabulary sweep (149,706 alive rows) ------------------
    fv = load_artifact("emb_fullvocab_stats.pt")
    claim("8 fullvocab", "stage == full", fv["stage"] == "full")
    claim(
        "8 fullvocab",
        "alive rows == 149706",
        int(fv["n_rows"]) == 149_706,
        str(int(fv["n_rows"])),
    )
    kurt = fv["dim_stats"]["kurtosis"]
    claim(
        "8 fullvocab",
        "kurtosis max 116.3",
        near(float(kurt.max()), 116.3, 0.5),
        f"{float(kurt.max()):.1f}",
    )
    claim(
        "8 fullvocab",
        "kurtosis median 0.32",
        near(float(kurt.median()), 0.32, 0.02),
        f"{float(kurt.median()):.2f}",
    )
    claim(
        "8 fullvocab",
        "dim-corr |r| mean 0.0210",
        near(fv["dim_corr_summary"]["offdiag_abs_mean"], 0.0210, 0.0005),
        f"{fv['dim_corr_summary']['offdiag_abs_mean']:.4f}",
    )
    claim(
        "8 fullvocab",
        "dim-corr |r| max 0.726",
        near(fv["dim_corr_summary"]["offdiag_max"], 0.726, 0.003),
        f"{fv['dim_corr_summary']['offdiag_max']:.3f}",
    )
    knn_cos = fv["knn_cos"].float()
    claim(
        "8 fullvocab",
        "kNN top-1 cosine mean +0.314",
        near(float(knn_cos[:, 0].mean()), 0.314, 0.002),
        f"{float(knn_cos[:, 0].mean()):+.3f}",
    )
    claim(
        "8 fullvocab",
        "kNN 32nd cosine mean +0.159",
        near(float(knn_cos[:, -1].mean()), 0.159, 0.002),
        f"{float(knn_cos[:, -1].mean()):+.3f}",
    )

    an = load_artifact("emb_fullvocab_analysis.pt")
    sizes = an["block_sizes"]
    claim(
        "8 fullvocab",
        "exactly one entangled block (>1 dim) at |r|>0.3, size 21",
        len(sizes) == 1 and int(sizes[0]) == 21,
        str(sizes),
    )
    block_dims = set(sorted(an["corr_blocks"], key=lambda b: -b["size"])[0]["dims"])
    claim(
        "8 fullvocab",
        "block contains dims 1395/2898/822 (top-kurtosis structural dims)",
        {1395, 2898, 822} <= block_dims,
    )
    census = {e["class"]: e for e in an["handle_census"]}
    claim(
        "8 fullvocab",
        "negative-handle out-of-battery hits == 60",
        census["negative"]["out_of_battery_hits"] == 60,
        str(census["negative"]["out_of_battery_hits"]),
    )
    claim(
        "8 fullvocab",
        "negative-handle top hits include ' shitty'/' nasty' (content)",
        " shitty" in census["negative"]["top_tokens"][:6]
        and " nasty" in census["negative"]["top_tokens"][:6],
    )
    claim(
        "8 fullvocab",
        "542 kNN communities; giant component 122,942",
        int(an["n_communities"]) == 542
        and int(an["community_sizes_top50"][0]) == 122_942,
        f"{an['n_communities']} / {an['community_sizes_top50'][0]}",
    )

    sb = load_artifact("emb_structural_block.pt")
    claim(
        "8 fullvocab",
        "block norm fraction vs token-id Spearman -0.206 (control -0.003)",
        near(sb["block_spearman_vs_id"], -0.2064, 0.002)
        and abs(sb["control_spearman_vs_id"]) < 0.01,
        f"{sb['block_spearman_vs_id']:+.4f} / {sb['control_spearman_vs_id']:+.4f}",
    )
    claim(
        "8 fullvocab",
        "block head-loading: decile-1 mean 0.1143 vs control 0.0753",
        near(sb["block_decile_means"][0], 0.1143, 0.002)
        and near(sb["control_decile_means"][0], 0.0753, 0.002),
        f"{sb['block_decile_means'][0]:.4f} / {sb['control_decile_means'][0]:.4f}",
    )
    claim(
        "8 fullvocab",
        "block top tokens are cross-script structural (',' '，' ' the' all top-5)",
        {",", "，", " the"} <= set(sb["top_tokens"][:5]),
        str(sb["top_tokens"][:5]),
    )
    claim(
        "8 fullvocab",
        "script-independence: block mean elevated over control for ascii+cjk+cyrillic",
        all(
            sb["per_script"][s]["block_mean"] > sb["per_script"][s]["control_mean"]
            for s in ("ascii", "cjk", "cyrillic")
        ),
    )

    # ---- AUDIT 9: tracing phase (T0 census, T1 readers, T1.5 carrier) ---------
    ta = load_artifact("emb_trace_analysis.pt")
    mass = {r["dim"]: r for r in ta["massive_dims"]}
    claim(
        "9 trace",
        "dim 458 is a massive-activation dim: |peak| > 10000, first layer 1, arc-1 sink",
        458 in mass
        and abs(mass[458]["peak_value"]) > 10_000
        and mass[458]["first_layer"] == 1
        and mass[458]["in_arc1_sinks"],
        f"peak {mass.get(458, {}).get('peak_value')}",
    )
    claim(
        "9 trace",
        "massive-dim overlap with arc-1 layer-20 sinks == {458, 1427, 2570}",
        set(ta["sink_massive_overlap"]) == {458, 1427, 2570},
        str(ta["sink_massive_overlap"]),
    )
    claim(
        "9 trace",
        "block dims in the census are layer-0 echoes only (first_layer 0, |peak| < 10)",
        all(
            mass[d]["first_layer"] == 0 and abs(mass[d]["peak_value"]) < 10
            for d in ta["block_massive_overlap"]
        ),
        str(ta["block_massive_overlap"]),
    )
    claim(
        "9 trace",
        "q-head block/control read ratio max ~3.28 (L0H15)",
        near(ta["readers"]["q"]["ratio_max"], 3.282, 0.01)
        and ta["readers"]["q"]["top"][0]["layer"] == 0
        and ta["readers"]["q"]["top"][0]["head"] == 15,
        f"{ta['readers']['q']['ratio_max']:.3f}",
    )
    claim(
        "9 trace",
        "v under-reads the block (mean ratio ~0.903)",
        near(ta["readers"]["v"]["ratio_mean"], 0.903, 0.005),
        f"{ta['readers']['v']['ratio_mean']:.3f}",
    )
    prof = ta["block_frac_profile"]
    claim(
        "9 trace",
        "P2: delimiter block-frac 0.921 at L0 collapsing to 0.199 at L1 (ctrl 0.160)",
        near(float(prof["delim"][0]), 0.9209, 0.005)
        and near(float(prof["delim"][1]), 0.1994, 0.005)
        and near(float(prof["ctrl"][1]), 0.1596, 0.005),
        f"{float(prof['delim'][0]):.4f} -> {float(prof['delim'][1]):.4f}",
    )

    tc = load_artifact("emb_trace_components.pt")
    bsv = tc["carrier"]["block"]["singular_values"]
    csv2 = tc["carrier"]["control"]["singular_values"]
    mass_frac = tc["carrier"]["block"]["in_block_mass_frac"]
    claim(
        "9 trace",
        "carrier: block top-SV 15.58 at L1, 13.13 at L28; control 3.93 at L28",
        near(float(bsv[1, 0]), 15.584, 0.05)
        and near(float(bsv[28, 0]), 13.127, 0.05)
        and near(float(csv2[28, 0]), 3.928, 0.05),
        f"L1 {float(bsv[1, 0]):.3f}, L28 {float(bsv[28, 0]):.3f} vs {float(csv2[28, 0]):.3f}",
    )
    claim(
        "9 trace",
        "carrier: block top-SV exceeds control at EVERY layer",
        bool((bsv[:, 0] > csv2[:, 0]).all()),
        f"min ratio {float((bsv[:, 0] / csv2[:, 0]).min()):.2f}",
    )
    claim(
        "9 trace",
        "in-block corr mass collapses 0.109 (L0) -> 0.031 (L1)",
        near(float(mass_frac[0]), 0.109, 0.003)
        and near(float(mass_frac[1]), 0.031, 0.003),
        f"{float(mass_frac[0]):.3f} -> {float(mass_frac[1]):.3f}",
    )
    claim(
        "9 trace",
        "original dims 2604/1395/1122 persist among top-10 carriers at L20",
        {2604, 1395, 1122} <= set(tc["carrier"]["block"]["top_carrier_dims"][20]),
        str(tc["carrier"]["block"]["top_carrier_dims"][20]),
    )
    af = torch.stack([c["attn_block_frac"] for c in tc["component_stats"]]).mean(0)
    claim(
        "9 trace",
        "attention delta block-frac peaks at L1 (~0.175)",
        int(af.argmax()) == 1 and near(float(af[1]), 0.1754, 0.005),
        f"argmax L{int(af.argmax())}, {float(af[1]):.4f}",
    )
    st = tc["static_maps"]
    claim(
        "9 trace",
        "input-RMSNorm block/control gain ~2.15 at L1 and ~3.28 at L27",
        near(float(st["norm1_block"][1] / st["norm1_control"][1]), 2.15, 0.05)
        and near(float(st["norm1_block"][27] / st["norm1_control"][27]), 3.28, 0.05),
        f"L1 {float(st['norm1_block'][1] / st['norm1_control'][1]):.2f}, "
        f"L27 {float(st['norm1_block'][27] / st['norm1_control'][27]):.2f}",
    )
    r_read = st["ffn_read_block"] / st["ffn_read_control"].clamp_min(1e-12)
    r_write = st["ffn_write_block"] / st["ffn_write_control"].clamp_min(1e-12)
    claim(
        "9 trace",
        "FFN static READS of block dims are neutral (all layers in [0.94, 1.01])",
        bool(((r_read > 0.94) & (r_read < 1.01)).all()),
        f"range [{float(r_read.min()):.3f}, {float(r_read.max()):.3f}]",
    )
    claim(
        "9 trace",
        "FFN WRITES into block dims mildly suppressed late (min ~0.822 at L24; all <= 1.06)",
        near(float(r_write.min()), 0.822, 0.005)
        and int(r_write.argmin()) == 24
        and float(r_write.max()) <= 1.06,
        f"min {float(r_write.min()):.3f} at L{int(r_write.argmin())}",
    )

    # ---- AUDIT 10: T2 delimiter attention (P1a PASS, P1c/P1d FAIL) -----------
    # Re-derives every load-bearing number in
    # observations/2026-07-15-emb-trace-delimiter-attention.md from the T2
    # capture, mirroring the emb_trace_attention_analyze.py statistic. Criterion
    # constants are the pre-registered P1a operationalization (ratio>=3x AND
    # excess>=0.03), asserted here as the audited spec, not read from prose.
    at = load_artifact("emb_trace_attention.pt")
    n_q_at = int(at["n_q_heads"])
    n_layers_at = int(at["n_layers"])
    head_dim_at = int(at["head_dim"])
    theta_at = float(at["rope_theta"])
    n_bands_at = at["band_d_sum"].shape[2]
    d2d_a = at["d2d_sum"] / at["d2d_n"].clamp_min(1)
    c2c_a = at["c2c_sum"] / at["c2c_n"].clamp_min(1)
    p2c_a = at["p2c_sum"] / at["p2c_n"].clamp_min(1)
    excess_a = d2d_a - c2c_a
    ratio_a = d2d_a / c2c_a.clamp_min(1e-9)

    claim(
        "10 T2attn",
        "corpus pair counts: 637 delim->delim, 155 period->comma",
        int(at["d2d_n"][0, 0]) == 637 and int(at["p2c_n"][0, 0]) == 155,
        f"{int(at['d2d_n'][0, 0])} / {int(at['p2c_n'][0, 0])}",
    )
    claim(
        "10 T2attn",
        "inter-delimiter offset median == 8 tokens",
        int(at["offsets_seen"].float().median()) == 8,
        str(int(at["offsets_seen"].float().median())),
    )

    # P1a census
    tracking_mask_a = (ratio_a >= 3.0) & (excess_a >= 0.03)
    n_tracking_a = int(tracking_mask_a.sum())
    claim(
        "10 T2attn",
        "P1a: 26 delimiter-tracking heads (d2d/c2c>=3x AND excess>=0.03)",
        n_tracking_a == 26,
        str(n_tracking_a),
    )
    tracking_layers_a = sorted({int(i) for i in torch.where(tracking_mask_a)[0]})
    claim(
        "10 T2attn",
        "qualifying heads live in layers {0,1,2,3,10}",
        tracking_layers_a == [0, 1, 2, 3, 10],
        str(tracking_layers_a),
    )
    top_a = torch.topk(excess_a.flatten(), 10)
    delim_heads_a = [(int(i // n_q_at), int(i % n_q_at)) for i in top_a.indices]
    claim(
        "10 T2attn",
        "top delimiter head is L0H13 (d2d 0.1784, c2c 0.0409, 4.37x)",
        delim_heads_a[0] == (0, 13)
        and near(float(d2d_a[0, 13]), 0.1784, 0.001)
        and near(float(c2c_a[0, 13]), 0.0409, 0.001)
        and near(float(ratio_a[0, 13]), 4.37, 0.02),
        f"L{delim_heads_a[0][0]}H{delim_heads_a[0][1]} {float(ratio_a[0, 13]):.2f}x",
    )
    claim(
        "10 T2attn",
        "L0H3 has the extreme ratio 27.15x (d2d 0.1128, c2c 0.0042)",
        near(float(ratio_a[0, 3]), 27.15, 0.05)
        and near(float(d2d_a[0, 3]), 0.1128, 0.001),
        f"{float(ratio_a[0, 3]):.2f}x",
    )

    # reader/delimiter dissociation (cross-ref to T1 readers in emb_trace_analysis.pt)
    reader_heads_a = [
        (int(r["layer"]), int(r["head"])) for r in ta["readers"]["q"]["top"]
    ]
    top_reader_a = reader_heads_a[0]
    overlap5 = sorted(set(delim_heads_a[:5]) & set(reader_heads_a[:5]))
    overlap10 = sorted(set(delim_heads_a[:10]) & set(reader_heads_a[:10]))
    claim(
        "10 T2attn",
        "reader/delimiter overlap: 1/5 (L0H20) and 2/10 (L0H3, L0H20)",
        overlap5 == [(0, 20)] and overlap10 == [(0, 3), (0, 20)],
        f"top5 {['L%dH%d' % h for h in overlap5]}, top10 {['L%dH%d' % h for h in overlap10]}",
    )
    tr_rank_a = int((excess_a.flatten() > excess_a[top_reader_a]).sum()) + 1
    claim(
        "10 T2attn",
        "top block-reader L0H15 has delim excess +0.0744, rank 21/784, ratio 11.00x",
        top_reader_a == (0, 15)
        and near(float(excess_a[top_reader_a]), 0.0744, 0.001)
        and tr_rank_a == 21
        and near(float(ratio_a[top_reader_a]), 11.00, 0.05)
        and top_reader_a not in delim_heads_a[:8],
        f"excess {float(excess_a[top_reader_a]):+.4f} rank {tr_rank_a}",
    )

    # P1c: both peak at layer 0 (the falsification)
    d2d_peak_a = int(d2d_a.max(dim=1).values.argmax())
    p2c_peak_a = int(p2c_a.max(dim=1).values.argmax())
    claim(
        "10 T2attn",
        "P1c FAIL: comma-side d2d peak layer 0 AND period->comma p2c peak layer 0",
        d2d_peak_a == 0 and p2c_peak_a == 0,
        f"d2d L{d2d_peak_a}, p2c L{p2c_peak_a}",
    )
    claim(
        "10 T2attn",
        "p2c max-head value 0.2193 at layer 0",
        near(float(p2c_a.max(dim=1).values[0]), 0.2193, 0.001),
        f"{float(p2c_a.max(dim=1).values[0]):.4f}",
    )

    # P1d: band decomposition — near-DC dominates, mid bands negligible
    band_d_a = at["band_d_sum"] / at["band_n"].clamp_min(1)[:, :, None]
    band_c_a = at["band_c_sum"] / at["band_n"].clamp_min(1)[:, :, None]
    band_excess_a = band_d_a - band_c_a
    agg_a = torch.zeros(n_bands_at)
    for li, hi in delim_heads_a:
        agg_a += band_excess_a[li, hi]
    total_pos_a = float(agg_a.clamp_min(0).sum())
    mid_bands_a = [
        b
        for b in range(n_bands_at)
        if 5.0 <= rope_wavelength(b, head_dim_at, theta_at) <= 30.0
    ]
    slow_bands_a = [
        b for b in range(n_bands_at) if rope_wavelength(b, head_dim_at, theta_at) > 1e4
    ]
    mid_share_a = 100.0 * float(agg_a[mid_bands_a].clamp_min(0).sum()) / total_pos_a
    slow_share_a = 100.0 * float(agg_a[slow_bands_a].clamp_min(0).sum()) / total_pos_a
    dominant_band_a = int(agg_a.argmax())
    claim(
        "10 T2attn",
        "P1d FAIL: dominant excess-logit band is 63 (wl ~5.06M tok, near-DC)",
        dominant_band_a == 63 and rope_wavelength(63, head_dim_at, theta_at) > 1e4,
        f"band {dominant_band_a}, wl {rope_wavelength(dominant_band_a, head_dim_at, theta_at):.0f}",
    )
    claim(
        "10 T2attn",
        "mid bands 0-7 (wl 6.3-28.5 tok) carry ~0.4%; near-DC (wl>1e4) carry ~99.1%",
        near(mid_share_a, 0.4, 0.15)
        and near(slow_share_a, 99.1, 0.3)
        and mid_bands_a == [0, 1, 2, 3, 4, 5, 6, 7],
        f"mid {mid_share_a:.1f}% / slow {slow_share_a:.1f}%",
    )

    # sink context
    d2s_a = float((at["d2sink_sum"] / at["d2sink_n"].clamp_min(1)).mean())
    c2s_a = float((at["c2sink_sum"] / at["c2sink_n"].clamp_min(1)).mean())
    claim(
        "10 T2attn",
        "delimiter queries attend LESS to pos 0 than controls (0.378 vs 0.406)",
        near(d2s_a, 0.378, 0.002) and near(c2s_a, 0.406, 0.002) and d2s_a < c2s_a,
        f"{d2s_a:.3f} vs {c2s_a:.3f}",
    )
    claim(
        "10 T2attn",
        "capture geometry: 28L x 28 q-heads, head_dim 128, rope_theta 1e6",
        n_layers_at == 28
        and n_q_at == 28
        and head_dim_at == 128
        and theta_at == 1_000_000.0,
        f"{n_layers_at}L x {n_q_at}H d{head_dim_at} theta{theta_at:.0f}",
    )

    # ---- AUDIT 11: '的' cross-script pairing check (README finding #6) -------
    # Re-derives the cited cosines from the committed candidate rows — no W_E
    # cache needed. Rank-in-full-vocab claims check the recorded top-k list
    # (transcription-locked; full re-derivation would need the 1.09 GB W_E).
    dc = load_artifact("emb_de_cosine_check.pt")
    dc_rows = dc["rows_f16"].float()
    labels = list(dc["candidates"])
    li = {lab: i for i, lab in enumerate(labels)}
    de_row = dc_rows[li["的"]]
    cmask = torch.ones(dc_rows.shape[1], dtype=torch.bool)
    cmask[dc["block_dims"]] = False

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    full = {lab: _cos(dc_rows[i], de_row) for lab, i in li.items()}
    ablt = {lab: _cos(dc_rows[i][cmask], de_row[cmask]) for lab, i in li.items()}
    ablt_apo = ablt["'s"]
    claim(
        "11 de-check",
        "cos('的',' of')=0.626 > cos('的',' the')=0.519 — ' of' is the closer token",
        near(full[" of"], 0.6261, 0.002)
        and near(full[" the"], 0.5192, 0.002)
        and full[" of"] > full[" the"],
        f"of {full[' of']:+.4f} / the {full[' the']:+.4f}",
    )
    claim(
        "11 de-check",
        "block-ablated ordering: ' of' > \"'s\" > ' the' (0.364/0.334/0.256)",
        near(ablt[" of"], 0.3642, 0.002)
        and near(ablt["'s"], 0.3344, 0.002)
        and near(ablt[" the"], 0.2560, 0.002)
        and ablt[" of"] > ablt["'s"] > ablt[" the"],
        f"{ablt[' of']:+.4f} / {ablt_apo:+.4f} / {ablt[' the']:+.4f}",
    )
    claim(
        "11 de-check",
        "'的' top-2 full-vocab neighbors are ' of' (315) then ' the' (279)",
        list(dc["topk_ids"][:2]) == [315, 279] and int(dc["n_vocab_rows"]) == 152_064,
        f"top2 {dc['topk_ids'][:2]} of {dc['n_vocab_rows']} rows",
    )
    claim(
        "11 de-check",
        "recomputed cosines match the recorded values (float16 row roundtrip)",
        all(near(full[lab], dc["full_cos_to_de"][lab], 0.002) for lab in labels)
        and all(near(ablt[lab], dc["ablated_cos_to_de"][lab], 0.002) for lab in labels),
        f"{len(labels)} candidates x 2 metrics",
    )

    print("=" * 80)
    print(f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL")
    print("=" * 80)
    return 1 if FAIL else 0


if __name__ == "__main__":
    missing = [
        n
        for n in (
            "emb_battery_vectors.pt",
            "emb_global_stats.pt",
            "emb_neighbor_probes.pt",
            "emb_category_stats.pt",
            "emb_pair_directions.pt",
            "emb_fullvocab_stats.pt",
            "emb_fullvocab_analysis.pt",
            "emb_structural_block.pt",
            "emb_de_cosine_check.pt",
            "emb_trace_analysis.pt",
            "emb_trace_components.pt",
            "emb_trace_attention.pt",
        )
        if find_artifact(n) is None
    ]
    if missing:
        print(
            f"missing artifacts {missing}; run emb_capture.py / derives, or git lfs pull"
        )
        sys.exit(2)
    sys.exit(main())
