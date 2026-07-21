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
  * structure_scan_qwen2.5-{7b,1.5b}-instruct_jlens_*.pt (stage-4 depth map)
  * verbal_report_qwen2.5-{7b,1.5b}-instruct_jlens_*.pt  (stage-5.1 swap suite)

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
  E  Verbal-report swap (stage 5.1) — both verbal_report artifacts load; 78
     items each; every item carries the 4 conditions x strengths {1,2} (8
     condition keys); baseline retained-subset sizes (predicts-report) re-derived
     from per_item (1.5B 9, 7B 12). Load-bearing swap-target top5_all rates
     re-derived FROM per_item (not trusting the summary blindly, with one
     condition cross-checked against the stored summary metric) and pinned at
     3dp: 1.5B jlens@2 0.551, nonjspace@2 0.269, random@2/@1 0.103; 7B jlens@2
     0.192, nonjspace@2 0.205, random@2 0.154. Ordering as booleans (1.5B
     jlens > nonjspace > random at s=2; 7B cross-condition spread @2 < 0.06).
     Magnitude-equalization invariant: per model per strength, the injected-norm
     spread across the 4 conditions < 0.05.
  F  Verbal-report swap, stage-5.1b chat + 6-condition (both chat_6c artifacts) —
     load; 78 items; 6 conditions x strengths {1,2}; predicts-report retained
     sizes (1.5B 6, 7B 21) re-derived from per_item; fragment-report rates
     re-derived (chat from the 6c per_item AND cross-checked vs its summary;
     plain from the committed 4c per_item via category membership: 1.5B
     0.192/0.192, 7B plain 0.308 -> chat 0.192). top5_all @s=2 pinned at 3dp for
     all six conditions, re-derived from per_item (one cross-checked vs summary).
     Ordering/gap booleans: 7B jlens-random @2 > 0.10 (un-confounding); 7B
     |jspace_comp-random| @2 < 0.02 and 1.5B jspace_comp-random @2 < 0.05
     (jspace_comp middle-tier NULL -- the 5.1b headline); 7B jlens-logitlens @2
     < 0.05 (token-steering persists); 1.5B jlens > logitlens > nonjspace @2.
     Injected-norm invariant in per-item-ratio form: the five J-lens-source
     conditions within 2% of the jlens condition each item x strength; logitlens
     is EXEMPT (its w_s source gives a small, bf16-noisy projection -> up to ~78%
     on 7B; a diagnosed measurement artifact, not a direction error) and only
     loosely bounded < 100%.

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


VR_CONDS = ("jlens", "nonjspace", "random", "logitlens")
VR_STRENGTHS = ("1.0", "2.0")
# spec: pt file, injection layer, retained (predicts-report) subset size, and the
# load-bearing top5_all pins (condition@strength -> rate, 3dp).
VR_SPEC: dict[str, dict[str, Any]] = {
    "1.5b": {
        "pt": "verbal_report_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "layer": 21,
        "retained": 9,
        "pins": {
            "jlens@2.0": 0.551,
            "nonjspace@2.0": 0.269,
            "random@2.0": 0.103,
            "random@1.0": 0.103,
        },
    },
    "7b": {
        "pt": "verbal_report_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "layer": 22,
        "retained": 12,
        "pins": {
            "jlens@2.0": 0.192,
            "nonjspace@2.0": 0.205,
            "random@2.0": 0.154,
        },
    },
}


def _vr_rate(items: list[dict[str, Any]], key: str, field: str) -> float:
    """Mean of a boolean per-condition field over all items, from raw per_item."""
    return float(np.mean([bool(it["conditions"][key][field]) for it in items]))


def _vr_norm_mean(items: list[dict[str, Any]], key: str) -> float:
    return float(
        np.mean([float(it["conditions"][key]["injected_norm"]) for it in items])
    )


def audit_verbal_report() -> None:
    print()
    print("=" * 80)
    print("CHECK E: verbal-report swap (stage 5.1; both models)")
    print("=" * 80)
    expected_keys = {f"{c}@{s}" for c in VR_CONDS for s in VR_STRENGTHS}
    for key, spec in VR_SPEC.items():
        d = load_pt_or_fail(spec["pt"])
        if d is None:
            continue
        items: list[dict[str, Any]] = d["per_item"]
        summary = d["summary"]

        # ---- structure: 78 items; 4 conditions x strengths {1,2} per item ----
        claim_eq(f"[{key}] verbal-report n_items == 78", 78, len(items))
        claim_eq(f"[{key}] verbal-report layer", spec["layer"], int(summary["layer"]))
        keys_ok = all(set(it["conditions"].keys()) == expected_keys for it in items)
        claim(
            f"[{key}] every item has 4 conds x strengths {{1,2}}",
            keys_ok,
            f"{sorted(expected_keys)}",
            "all match" if keys_ok else "MISSING KEYS",
        )

        # ---- retained (predicts-report) subset re-derived from per_item ----
        retained = int(sum(1 for it in items if bool(it["jlens_predicts_report"])))
        claim_eq(f"[{key}] retained (predicts) subset size", spec["retained"], retained)

        # ---- pinned top5_all rates, re-derived FROM per_item ----
        for pin_key, val in spec["pins"].items():
            claim_near(
                f"[{key}] {pin_key} top5_all",
                val,
                _vr_rate(items, pin_key, "target_in_top5"),
                atol=0.001,
            )
        # consistency: re-derived (raw per_item) == stored summary metric.
        rederived = _vr_rate(items, "jlens@2.0", "target_in_top5")
        stored = float(summary["metrics"]["jlens@2.0"]["target_top5_rate_all"])
        claim(
            f"[{key}] jlens@2 top5_all rederived==summary",
            abs(rederived - stored) < 1e-9,
            round(stored, 6),
            round(rederived, 6),
        )

        # ---- ordering / separation claims (booleans) ----
        if key == "1.5b":
            j = _vr_rate(items, "jlens@2.0", "target_in_top5")
            n = _vr_rate(items, "nonjspace@2.0", "target_in_top5")
            r = _vr_rate(items, "random@2.0", "target_in_top5")
            claim(
                "[1.5b] ordering jlens > nonjspace > random @2",
                j > n > r,
                "j > n > r",
                f"{j:.3f} > {n:.3f} > {r:.3f}",
            )
        if key == "7b":
            vals2 = [_vr_rate(items, f"{c}@2.0", "target_in_top5") for c in VR_CONDS]
            spread = max(vals2) - min(vals2)
            claim(
                "[7b] cross-condition spread @2 < 0.06",
                spread < 0.06,
                "< 0.06",
                round(spread, 4),
            )

        # ---- magnitude-equalization invariant: per-strength norm spread < 0.05 ----
        for s in VR_STRENGTHS:
            norms = [_vr_norm_mean(items, f"{c}@{s}") for c in VR_CONDS]
            spread = max(norms) - min(norms)
            claim(
                f"[{key}] injected-norm spread @{s} < 0.05",
                spread < 0.05,
                "< 0.05",
                round(spread, 5),
            )


# Stage 5.1b chat + 6-condition suite. Plain-4c committed artifacts are the
# fragment-rate plain baseline; the chat_6c artifacts hold the swap tables.
VR6_CONDS = (
    "jlens",
    "logitlens",
    "nonjspace",
    "nonjspace_comp",
    "jspace_comp",
    "random",
)
VR6_JSRC = ("jlens", "nonjspace", "random", "jspace_comp", "nonjspace_comp")
VR6_SPEC: dict[str, dict[str, Any]] = {
    "1.5b": {
        "chat": "verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "plain": "verbal_report_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        "layer": 21,
        "retained": 6,
        "frag_chat": 0.192,
        "frag_plain": 0.192,
        "pins2": {
            "jlens": 0.628,
            "logitlens": 0.564,
            "nonjspace": 0.308,
            "nonjspace_comp": 0.218,
            "jspace_comp": 0.179,
            "random": 0.141,
        },
    },
    "7b": {
        "chat": "verbal_report_chat_6c_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "plain": "verbal_report_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        "layer": 22,
        "retained": 21,
        "frag_chat": 0.192,
        "frag_plain": 0.308,
        "pins2": {
            "jlens": 0.269,
            "logitlens": 0.244,
            "nonjspace": 0.179,
            "nonjspace_comp": 0.167,
            "jspace_comp": 0.154,
            "random": 0.154,
        },
    },
}


def _frag_from_peritem(
    items: list[dict[str, Any]], categories: dict[str, list[str]]
) -> float:
    """Fragment-report rate: fraction of categories whose baseline source word
    matches no in-category instance (re-derived, one row per category)."""
    seen: dict[str, str] = {}
    for it in items:
        seen[it["category"]] = it["source_word"]
    frag = sum(1 for c, sw in seen.items() if sw not in categories.get(c, []))
    return frag / len(seen) if seen else float("nan")


def audit_verbal_report_6c() -> None:
    print()
    print("=" * 80)
    print("CHECK F: verbal-report swap, stage-5.1b chat + 6 conditions")
    print("=" * 80)
    # CATEGORIES lives in the sibling script; import lazily (no model load).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from jspace_verbal_report import CATEGORIES  # noqa: E402

    expected_keys = {f"{c}@{s}" for c in VR6_CONDS for s in ("1.0", "2.0")}
    for key, spec in VR6_SPEC.items():
        d = load_pt_or_fail(spec["chat"])
        if d is None:
            continue
        items: list[dict[str, Any]] = d["per_item"]
        summary = d["summary"]

        # ---- structure ----
        claim_eq(f"[{key}] chat_6c n_items == 78", 78, len(items))
        claim_eq(f"[{key}] chat_6c layer", spec["layer"], int(summary["layer"]))
        keys_ok = all(set(it["conditions"].keys()) == expected_keys for it in items)
        claim(
            f"[{key}] every item has 6 conds x strengths {{1,2}}",
            keys_ok,
            "12 keys/item",
            "all match" if keys_ok else "MISSING KEYS",
        )

        # ---- retained (predicts) subset re-derived from per_item ----
        retained = int(sum(1 for it in items if bool(it["jlens_predicts_report"])))
        claim_eq(f"[{key}] chat retained (predicts) subset", spec["retained"], retained)

        # ---- fragment-report rates: chat (summary + re-derived) and plain ----
        frag_chat_re = _frag_from_peritem(items, CATEGORIES)
        claim_near(
            f"[{key}] chat fragment rate (re-derived)",
            spec["frag_chat"],
            frag_chat_re,
            0.001,
        )
        claim_near(
            f"[{key}] chat fragment rate summary==re-derived",
            float(summary["fragment_report_rate"]),
            frag_chat_re,
            1e-9,
        )
        dp = load_pt_or_fail(spec["plain"])
        if dp is not None:
            frag_plain = _frag_from_peritem(dp["per_item"], CATEGORIES)
            claim_near(
                f"[{key}] plain fragment rate (re-derived)",
                spec["frag_plain"],
                frag_plain,
                0.001,
            )

        # ---- pinned top5_all @s=2 for all six conditions, re-derived ----
        for cond, val in spec["pins2"].items():
            claim_near(
                f"[{key}] {cond}@2 top5_all",
                val,
                _vr_rate(items, f"{cond}@2.0", "target_in_top5"),
                atol=0.001,
            )
        rederived = _vr_rate(items, "jlens@2.0", "target_in_top5")
        stored = float(summary["metrics"]["jlens@2.0"]["target_top5_rate_all"])
        claim(
            f"[{key}] jlens@2 top5_all rederived==summary",
            abs(rederived - stored) < 1e-9,
            round(stored, 6),
            round(rederived, 6),
        )

        # ---- ordering / gap booleans ----
        r2 = {c: _vr_rate(items, f"{c}@2.0", "target_in_top5") for c in VR6_CONDS}
        if key == "7b":
            claim(
                "[7b] jlens - random @2 > 0.10 (un-confounding)",
                (r2["jlens"] - r2["random"]) > 0.10,
                "> 0.10",
                round(r2["jlens"] - r2["random"], 4),
            )
            claim(
                "[7b] |jspace_comp - random| @2 < 0.02 (middle-tier NULL)",
                abs(r2["jspace_comp"] - r2["random"]) < 0.02,
                "< 0.02",
                round(abs(r2["jspace_comp"] - r2["random"]), 4),
            )
            claim(
                "[7b] jlens - logitlens @2 < 0.05 (token-steering persists)",
                (r2["jlens"] - r2["logitlens"]) < 0.05,
                "< 0.05",
                round(r2["jlens"] - r2["logitlens"], 4),
            )
        if key == "1.5b":
            claim(
                "[1.5b] jspace_comp - random @2 < 0.05 (middle-tier NULL)",
                (r2["jspace_comp"] - r2["random"]) < 0.05,
                "< 0.05",
                round(r2["jspace_comp"] - r2["random"], 4),
            )
            claim(
                "[1.5b] jlens > logitlens > nonjspace @2",
                r2["jlens"] > r2["logitlens"] > r2["nonjspace"],
                "jlens > logitlens > nonjspace",
                f"{r2['jlens']:.3f} > {r2['logitlens']:.3f} > {r2['nonjspace']:.3f}",
            )

        # ---- injected-norm invariant (per-item ratio; logitlens EXEMPT) ----
        for s in ("1.0", "2.0"):
            worst_src, worst_logit = 0.0, 0.0
            for it in items:
                ref = float(it["conditions"][f"jlens@{s}"]["injected_norm"])
                if ref <= 0.0:
                    continue
                for c in VR6_JSRC:
                    ratio = float(it["conditions"][f"{c}@{s}"]["injected_norm"]) / ref
                    worst_src = max(worst_src, abs(ratio - 1.0))
                lr = float(it["conditions"][f"logitlens@{s}"]["injected_norm"]) / ref
                worst_logit = max(worst_logit, abs(lr - 1.0))
            claim(
                f"[{key}] J-src injected-norm within 2% of jlens @{s}",
                worst_src < 0.02,
                "< 0.02",
                round(worst_src, 5),
            )
            claim(
                f"[{key}] logitlens injected-norm bounded <100% @{s} (bf16-exempt)",
                worst_logit < 1.0,
                "< 1.0 (exempt)",
                round(worst_logit, 4),
            )


def main() -> None:
    audit_lens_integrity()
    audit_eval_tables()
    audit_readout_scan()
    audit_structure_scan()
    audit_verbal_report()
    audit_verbal_report_6c()
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
