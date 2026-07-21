"""Regression audit for the embedding-atlas arc — re-derives every
load-bearing number cited in research/arcs/embedding-atlas/observations/
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
        "block energy vs token-id Spearman -0.206 (control -0.003)",
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
        )
        if find_artifact(n) is None
    ]
    if missing:
        print(
            f"missing artifacts {missing}; run emb_capture.py / derives, or git lfs pull"
        )
        sys.exit(2)
    sys.exit(main())
