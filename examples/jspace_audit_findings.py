"""Rigorous audit: re-derive every load-bearing numerical claim in the jspace
arc's committed observations from the on-disk artifacts, and compare against
what the observation files report. Prints per-check PASS/FAIL and exits 1 on
any failure (stage-7 audit named in
`research/arcs/jspace/plans/2026-07-18-jspace-design.md`).

Artifacts read (arc working cache `research/arcs/jspace/data/cache/`):
  * jlens_qwen2.5-7b_nf4_n100.pt    / .config.json   (fitted J-lens, 7B nf4)
  * jlens_qwen2.5-1.5b_bf16_n100.pt / .config.json   (fitted J-lens, 1.5B bf16)
  * lens_eval_qwen2.5-{7b_nf4,1.5b_bf16}_n100.pt      (intermediate-concept evals)
  * readout_scan_qwen2.5-{7b,1.5b}-instruct_jlens_*.pt (readout scan summaries)

Unlike the committed git-LFS *layer-subset* deliverables (7 layers, a viewing
convenience per design Decision 4), the FULL fitted lenses and every derived
eval/scan artifact are cache-only and gitignored — so this audit replays from
the arc's working cache, the set the observations were derived from. A missing
artifact is a loud FAIL (never a crash): the check records `MISSING` and the
dependent checks for that artifact are skipped, so the run still reaches the
SUMMARY and exits 1.

Checks:
  A  Lens integrity — both lenses load; 27 layers == source_layers == range(27);
     correct d_model; all J tensors finite; per-layer Frobenius norm > 0;
     n_prompts == 100 and consistent with the .config.json sidecar (model,
     mode, n_prompts).
  B  Eval tables — every rate/count in the observation tables of
     2026-07-18-intermediate-concept-evals-h3-confirmed.md (1.5B) and
     2026-07-20-scale-comparison-7b-vs-1p5b-h2.md (7B + 1.5B), re-derived from
     the lens_eval artifacts (tol abs < 0.005; tables rounded to 2dp).
  C  Readout scan — depth-of-emergence medians + never-emerged counts (both
     models), re-derived from the per-prompt rank arrays AND checked against
     the stored summary AND the observation tables; the 7B layer-26 Spearman
     crossover (J 0.7632 > logit 0.6952); the 1.5B per-layer Spearman excerpt.
  D  Structure scan (stage-4 J-space) — both structure_scan artifacts load;
     ks == [5,10,25,50]; 27 layers; 30 prompts; every varfrac (summary means
     and per-prompt raw) finite and in [0,1]. Load-bearing depth-map numbers
     re-derived and pinned: 1.5B varfrac_mean(k=25) peak at layer 21, plus its
     layer-0 and final-layer values, active_mean in [23,25] all layers, and
     logit-kurtosis endpoints (L0 + mid-trough layer/value); 7B varfrac trough
     at layer 17 and late-peak at layer 22 (argmin/argmax ranks), layer-0
     value, active_mean in [22,24], logit-kurtosis endpoints. Values pinned to
     the stored artifact figures at 3dp; the peak/trough *layer indices* are
     the primary claims.

The audit is a regression test for arithmetic consistency between the stored
artifacts and the observation prose — it does not re-run capture/fitting, so a
protocol bug shared by capture and observation would pass. See
`research/ARC_PROCESS.md` § "Raw data is a deliverable".
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

import numpy as np
import torch

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = _REPO_ROOT / "research" / "arcs" / "jspace" / "data" / "cache"


# ---------------------------------------------------------------------------
# Audit harness (same shape as examples/nla_audit_findings.py)
# ---------------------------------------------------------------------------
PASS, FAIL = 0, 0
ISSUES: list[tuple[str, Any, Any]] = []


def claim(name: str, ok: bool, expected: Any, actual: Any, tol: float = 0.0) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS   {name}   expected={expected!r}   actual={actual!r}")
    else:
        FAIL += 1
        ISSUES.append((name, expected, actual))
        print(
            f"  FAIL   {name}   expected={expected!r}   actual={actual!r}   tol={tol}"
        )


def claim_near(name: str, expected: float, actual: float, atol: float = 0.005) -> None:
    claim(name, abs(actual - expected) <= atol, expected, round(actual, 4), tol=atol)


def claim_eq(name: str, expected: Any, actual: Any) -> None:
    claim(name, expected == actual, expected, actual)


def load_pt_or_fail(name: str) -> Any | None:
    p = CACHE / name
    if not p.exists():
        claim(f"artifact present: {name}", False, "present", "MISSING")
        return None
    return torch.load(p, weights_only=False)


def load_json_or_fail(name: str) -> dict[str, Any] | None:
    p = CACHE / name
    if not p.exists():
        claim(f"sidecar present: {name}", False, "present", "MISSING")
        return None
    return cast("dict[str, Any]", json.loads(p.read_text()))


# ---------------------------------------------------------------------------
# Static expectations transcribed from the observation files
# ---------------------------------------------------------------------------
LENSES: list[dict[str, Any]] = [
    {
        "key": "1.5b",
        "pt": "jlens_qwen2.5-1.5b_bf16_n100.pt",
        "config": "jlens_qwen2.5-1.5b_bf16_n100.config.json",
        "d_model": 1536,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "mode": "bf16",
    },
    {
        "key": "7b",
        "pt": "jlens_qwen2.5-7b_nf4_n100.pt",
        "config": "jlens_qwen2.5-7b_nf4_n100.config.json",
        "d_model": 3584,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "mode": "nf4",
    },
]

# Eval-table expectations. Per (band, threshold) rate rounded to 2dp in the obs.
# 1.5B: 2026-07-18-intermediate-concept-evals-h3-confirmed.md (full J & logit
# table for multihop; association floor overall J 0.00/0.02, logit 0.01/0.02).
# 7B: 2026-07-20-scale-comparison-7b-vs-1p5b-h2.md (multihop J@10 & logit@10
# per band; association overall J 0.01/0.07).
EVAL_1P5B = "lens_eval_qwen2.5-1.5b_bf16_n100.pt"
EVAL_7B = "lens_eval_qwen2.5-7b_nf4_n100.pt"

# multihop full rate table for 1.5B (band -> (J@10, J@50, L@10, L@50))
MULTIHOP_1P5B = {
    "early": (0.20, 0.20, 0.00, 0.00),
    "mid": (0.13, 0.24, 0.00, 0.00),
    "late": (0.50, 0.65, 0.40, 0.64),
    "overall": (0.58, 0.74, 0.40, 0.64),
}
# multihop @10 table for 7B (band -> (J@10, L@10))
MULTIHOP_7B_AT10 = {
    "early": (0.09, 0.01),
    "mid": (0.07, 0.01),
    "late": (0.45, 0.35),
    "overall": (0.53, 0.36),
}


# ---------------------------------------------------------------------------
def audit_lens_integrity() -> None:
    print("=" * 80)
    print("CHECK A: lens integrity (both lenses; layers, d_model, finiteness)")
    print("=" * 80)
    for spec in LENSES:
        key = spec["key"]
        d = load_pt_or_fail(spec["pt"])
        cfg = load_json_or_fail(spec["config"])
        if d is None:
            continue
        jmat: dict[int, torch.Tensor] = d["J"]
        source_layers: list[int] = list(d["source_layers"])
        expected_layers = list(range(27))
        claim_eq(f"[{key}] J layer count == 27", 27, len(jmat))
        claim_eq(f"[{key}] source_layers == 0..26", expected_layers, source_layers)
        claim_eq(
            f"[{key}] J keys == source_layers", expected_layers, sorted(jmat.keys())
        )
        claim_eq(f"[{key}] d_model", spec["d_model"], int(d["d_model"]))
        claim_eq(f"[{key}] n_prompts == 100", 100, int(d["n_prompts"]))

        shapes_ok = all(
            tuple(jmat[l].shape) == (spec["d_model"], spec["d_model"])
            for l in source_layers
        )
        claim(
            f"[{key}] every J_l is (d_model, d_model)",
            shapes_ok,
            f"(({spec['d_model']}, {spec['d_model']}) x27)",
            "all match" if shapes_ok else "SHAPE MISMATCH",
        )
        finite_ok = all(bool(jmat[l].isfinite().all().item()) for l in source_layers)
        claim(
            f"[{key}] all J tensors finite",
            finite_ok,
            "all finite",
            "all finite" if finite_ok else "NON-FINITE PRESENT",
        )
        min_frob = min(float(jmat[l].float().norm().item()) for l in source_layers)
        claim(
            f"[{key}] min per-layer Frobenius norm > 0",
            min_frob > 0.0,
            "> 0",
            round(min_frob, 4),
        )

        if cfg is not None:
            claim_eq(f"[{key}] sidecar model", spec["model"], cfg.get("model"))
            claim_eq(f"[{key}] sidecar mode", spec["mode"], cfg.get("mode"))
            claim_eq(f"[{key}] sidecar n_prompts", 100, int(cfg.get("n_prompts", -1)))
            claim_eq(
                f"[{key}] n_prompts sidecar-vs-tensor consistent",
                int(d["n_prompts"]),
                int(cfg.get("n_prompts", -1)),
            )


def _rate(pe: dict[str, Any], which: str, band: str, thr: int) -> float:
    band_rates = pe[which][band]
    key = thr if thr in band_rates else str(thr)
    return float(band_rates[key])


def audit_eval_tables() -> None:
    print()
    print("=" * 80)
    print("CHECK B: eval tables (intermediate-concept rates + counts)")
    print("=" * 80)

    # ---- 1.5B ----
    d = load_pt_or_fail(EVAL_1P5B)
    if d is not None:
        pe = d["summary"]["per_eval"]
        mh = pe["multihop"]
        claim_eq("[1.5b] multihop n_items", 93, int(mh["n_items"]))
        claim_eq("[1.5b] multihop n_instances", 103, int(mh["n_instances"]))
        claim_eq("[1.5b] multihop n_inexact_token", 9, int(mh["n_inexact_token"]))
        for band, (j10, j50, l10, l50) in MULTIHOP_1P5B.items():
            claim_near(
                f"[1.5b] multihop J@10 {band}", j10, _rate(mh, "rates_j", band, 10)
            )
            claim_near(
                f"[1.5b] multihop J@50 {band}", j50, _rate(mh, "rates_j", band, 50)
            )
            claim_near(
                f"[1.5b] multihop logit@10 {band}", l10, _rate(mh, "rates_l", band, 10)
            )
            claim_near(
                f"[1.5b] multihop logit@50 {band}", l50, _rate(mh, "rates_l", band, 50)
            )
        assoc = pe["association"]
        claim_eq("[1.5b] association n_instances", 102, int(assoc["n_instances"]))
        claim_eq("[1.5b] association n_inexact_token", 3, int(assoc["n_inexact_token"]))
        # Floor: obs states association overall J 0.00/0.02, logit 0.01/0.02.
        claim_near(
            "[1.5b] association J@10 overall",
            0.00,
            _rate(assoc, "rates_j", "overall", 10),
        )
        claim_near(
            "[1.5b] association J@50 overall",
            0.02,
            _rate(assoc, "rates_j", "overall", 50),
        )
        claim_near(
            "[1.5b] association logit@10 overall",
            0.01,
            _rate(assoc, "rates_l", "overall", 10),
        )
        claim_near(
            "[1.5b] association logit@50 overall",
            0.02,
            _rate(assoc, "rates_l", "overall", 50),
        )

    # ---- 7B ----
    d = load_pt_or_fail(EVAL_7B)
    if d is not None:
        pe = d["summary"]["per_eval"]
        mh = pe["multihop"]
        claim_eq("[7b] multihop n_items", 93, int(mh["n_items"]))
        claim_eq("[7b] multihop n_instances", 103, int(mh["n_instances"]))
        claim_eq("[7b] multihop n_inexact_token", 9, int(mh["n_inexact_token"]))
        for band, (j10, l10) in MULTIHOP_7B_AT10.items():
            claim_near(
                f"[7b] multihop J@10 {band}", j10, _rate(mh, "rates_j", band, 10)
            )
            claim_near(
                f"[7b] multihop logit@10 {band}", l10, _rate(mh, "rates_l", band, 10)
            )
        assoc = pe["association"]
        claim_eq("[7b] association n_instances", 102, int(assoc["n_instances"]))
        claim_eq("[7b] association n_inexact_token", 3, int(assoc["n_inexact_token"]))
        # obs: 7B association overall J@10 0.01, J@50 0.07.
        claim_near(
            "[7b] association J@10 overall",
            0.01,
            _rate(assoc, "rates_j", "overall", 10),
        )
        claim_near(
            "[7b] association J@50 overall",
            0.07,
            _rate(assoc, "rates_j", "overall", 50),
        )


def _rederive_emergence(
    per_prompt: list[dict[str, Any]], layers: list[int]
) -> dict[str, Any]:
    """Re-derive depth-of-emergence exactly as jspace_readout_scan.py does:
    first layer (in `layers` order) where the model top-1 rank is < 10 (top-10,
    0-based rank), per (prompt, position) cell; median over emerged cells,
    count of never-emerged."""
    emerge_j: list[float] = []
    emerge_l: list[float] = []
    emerge_j_last: list[float] = []
    emerge_l_last: list[float] = []
    for pp in per_prompt:
        rank_j = np.asarray(pp["rank_j"])
        rank_l = np.asarray(pp["rank_l"])
        last_idx = int(pp["last_idx"])
        n_pos = rank_j.shape[1]
        for pos in range(n_pos):
            for arr, dst, dst_last in (
                (rank_j, emerge_j, emerge_j_last),
                (rank_l, emerge_l, emerge_l_last),
            ):
                hit = np.where(arr[:, pos] < 10)[0]
                val = float(layers[int(hit[0])]) if hit.size else float("nan")
                dst.append(val)
                if pos == last_idx:
                    dst_last.append(val)

    def med(xs: list[float]) -> float:
        arr = np.array([x for x in xs if not np.isnan(x)])
        return float(np.median(arr)) if arr.size else float("nan")

    return {
        "median_layer_j_allpos": med(emerge_j),
        "median_layer_l_allpos": med(emerge_l),
        "median_layer_j_lastpos": med(emerge_j_last),
        "median_layer_l_lastpos": med(emerge_l_last),
        "n_never_emerged_j_allpos": int(sum(1 for x in emerge_j if np.isnan(x))),
        "n_never_emerged_l_allpos": int(sum(1 for x in emerge_l if np.isnan(x))),
        "n_samples_allpos": len(emerge_j),
    }


# Observation depth-of-emergence tables (2026-07-20 scale-comparison +
# 2026-07-18 1.5B first-pass). (j_allpos, l_allpos, j_lastpos, l_lastpos,
# n_never_j, n_never_l).
DEPTH_OBS = {
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt": {
        "key": "7b",
        "j_allpos": 22.0,
        "l_allpos": 24.0,
        "j_lastpos": 22.0,
        "l_lastpos": 26.0,
        "n_never_j": 14,
        "n_never_l": 8,
    },
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": {
        "key": "1.5b",
        "j_allpos": 23.0,
        "l_allpos": 19.0,
        "j_lastpos": 20.0,
        "l_lastpos": 19.0,
        "n_never_j": 15,
        "n_never_l": 1,
    },
}

# 1.5B first-pass per-layer Spearman excerpt (last position): layer -> (J, logit).
SPEARMAN_1P5B_LAST = {
    4: (0.099, 0.377),
    12: (0.345, 0.467),
    20: (0.408, 0.548),
    24: (0.667, 0.714),
    26: (0.800, 0.861),
}


def audit_readout_scan() -> None:
    print()
    print("=" * 80)
    print("CHECK C: readout scan (depth-of-emergence + Spearman crossover)")
    print("=" * 80)
    for fname, obs in DEPTH_OBS.items():
        key = obs["key"]
        d = load_pt_or_fail(fname)
        if d is None:
            continue
        summary = d["summary"]
        layers = list(summary["layers"])
        stored = summary["depth_of_emergence_top10"]

        # (1) re-derive from per-prompt rank arrays and check vs stored summary
        rederived = _rederive_emergence(d["per_prompt"], layers)
        for field in (
            "median_layer_j_allpos",
            "median_layer_l_allpos",
            "median_layer_j_lastpos",
            "median_layer_l_lastpos",
            "n_never_emerged_j_allpos",
            "n_never_emerged_l_allpos",
        ):
            claim_eq(
                f"[{key}] {field} rederived==stored",
                stored[field],
                rederived[field],
            )

        # (2) stored summary vs observation table
        claim_eq(
            f"[{key}] median J-lens all-pos (obs)",
            obs["j_allpos"],
            float(stored["median_layer_j_allpos"]),
        )
        claim_eq(
            f"[{key}] median logit all-pos (obs)",
            obs["l_allpos"],
            float(stored["median_layer_l_allpos"]),
        )
        claim_eq(
            f"[{key}] median J-lens last-pos (obs)",
            obs["j_lastpos"],
            float(stored["median_layer_j_lastpos"]),
        )
        claim_eq(
            f"[{key}] median logit last-pos (obs)",
            obs["l_lastpos"],
            float(stored["median_layer_l_lastpos"]),
        )
        claim_eq(
            f"[{key}] never-emerged J (obs)",
            obs["n_never_j"],
            int(stored["n_never_emerged_j_allpos"]),
        )
        claim_eq(
            f"[{key}] never-emerged logit (obs)",
            obs["n_never_l"],
            int(stored["n_never_emerged_l_allpos"]),
        )

        # (3) per-model Spearman claims
        j_last = list(summary["layer_mean_spearman_j_last"])
        l_last = list(summary["layer_mean_spearman_l_last"])
        if key == "7b":
            # layer-26 crossover: J 0.7632 > logit 0.6952
            claim_near("[7b] layer-26 Spearman J-last", 0.7632, float(j_last[26]))
            claim_near("[7b] layer-26 Spearman logit-last", 0.6952, float(l_last[26]))
            claim(
                "[7b] layer-26 J-lens Spearman > logit (crossover)",
                float(j_last[26]) > float(l_last[26]),
                "J > logit",
                f"{j_last[26]:.4f} vs {l_last[26]:.4f}",
            )
        if key == "1.5b":
            for layer, (js, ls) in SPEARMAN_1P5B_LAST.items():
                claim_near(
                    f"[1.5b] layer-{layer} Spearman J-last", js, float(j_last[layer])
                )
                claim_near(
                    f"[1.5b] layer-{layer} Spearman logit-last",
                    ls,
                    float(l_last[layer]),
                )


STRUCT_1P5B = "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt"
STRUCT_7B = "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt"


def _check_struct_common(
    key: str, d: dict[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    """Shared integrity for a structure-scan artifact: config shape + all
    varfrac (summary means and per-prompt raw) finite and in [0, 1]."""
    s = d["summary"]
    layers = list(s["layers"])
    claim_eq(f"[{key}] structure ks == [5,10,25,50]", [5, 10, 25, 50], list(s["ks"]))
    claim_eq(f"[{key}] structure n_prompts == 30", 30, int(s["n_prompts"]))
    claim_eq(f"[{key}] structure layers == 0..26", list(range(27)), layers)

    mean_vals = [float(v) for k in s["ks"] for v in s["mean_varfrac"][k]]
    mean_ok = all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in mean_vals)
    claim(
        f"[{key}] all mean_varfrac finite & in [0,1]",
        mean_ok,
        "in [0,1] finite",
        "ok" if mean_ok else "OUT OF RANGE",
    )
    pp_ok = True
    for pp in d["per_prompt"]:
        for k in s["ks"]:
            arr = np.asarray(pp["varfrac"][k], dtype=float)
            if not (
                np.isfinite(arr).all() and (arr >= 0.0).all() and (arr <= 1.0).all()
            ):
                pp_ok = False
                break
        if not pp_ok:
            break
    claim(
        f"[{key}] all per-prompt varfrac finite & in [0,1]",
        pp_ok,
        "in [0,1] finite",
        "ok" if pp_ok else "OUT OF RANGE",
    )
    return s, layers


def audit_structure_scan() -> None:
    print()
    print("=" * 80)
    print("CHECK D: structure scan (stage-4 J-space depth map)")
    print("=" * 80)

    # ---- 1.5B: single mid-late varfrac hump ----
    d = load_pt_or_fail(STRUCT_1P5B)
    if d is not None:
        s, layers = _check_struct_common("1.5b", d)
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        act = np.array(s["mean_active"][25], dtype=float)
        lk = np.array(s["mean_readout_kurtosis"], dtype=float)
        peak_i = int(vf.argmax())
        claim_eq("[1.5b] varfrac(k25) peak layer == 21", 21, layers[peak_i])
        claim_near("[1.5b] varfrac(k25) peak value", 0.124, float(vf[peak_i]))
        claim_near("[1.5b] varfrac(k25) layer-0", 0.082, float(vf[0]))
        claim_near("[1.5b] varfrac(k25) final layer", 0.068, float(vf[-1]))
        claim(
            "[1.5b] active_mean(k25) in [23,25] all layers",
            bool((act >= 23.0).all() and (act <= 25.0).all()),
            "[23,25]",
            f"[{act.min():.3f},{act.max():.3f}]",
        )
        claim_near("[1.5b] logit-kurt layer-0", 4.648, float(lk[0]))
        kt_i = int(lk.argmin())
        claim_eq("[1.5b] logit-kurt trough layer == 18", 18, layers[kt_i])
        claim_near("[1.5b] logit-kurt trough value", 0.926, float(lk[kt_i]))

    # ---- 7B: mid trough then late rise ----
    d = load_pt_or_fail(STRUCT_7B)
    if d is not None:
        s, layers = _check_struct_common("7b", d)
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        act = np.array(s["mean_active"][25], dtype=float)
        lk = np.array(s["mean_readout_kurtosis"], dtype=float)
        trough_i = int(vf.argmin())
        claim_eq("[7b] varfrac(k25) trough layer == 17", 17, layers[trough_i])
        claim_near("[7b] varfrac(k25) trough value", 0.012, float(vf[trough_i]))
        peak_i = int(vf.argmax())
        claim_eq("[7b] varfrac(k25) late-peak layer == 22", 22, layers[peak_i])
        claim_near("[7b] varfrac(k25) late-peak value", 0.040, float(vf[peak_i]))
        claim_near("[7b] varfrac(k25) layer-0", 0.031, float(vf[0]))
        claim(
            "[7b] active_mean(k25) in [22,24] all layers",
            bool((act >= 22.0).all() and (act <= 24.0).all()),
            "[22,24]",
            f"[{act.min():.3f},{act.max():.3f}]",
        )
        claim_near("[7b] logit-kurt layer-0", 3.246, float(lk[0]))
        kt_i = int(lk.argmin())
        claim_eq("[7b] logit-kurt trough layer == 19", 19, layers[kt_i])
        claim_near("[7b] logit-kurt trough value", 0.868, float(lk[kt_i]))


def main() -> None:
    audit_lens_integrity()
    audit_eval_tables()
    audit_readout_scan()
    audit_structure_scan()
    print()
    print("=" * 80)
    print(f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL")
    print("=" * 80)
    if FAIL > 0:
        print("\nFAILED CLAIMS:")
        for name, exp, act in ISSUES:
            print(f"  - {name}: expected {exp!r}, got {act!r}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
