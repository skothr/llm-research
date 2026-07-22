"""Rigorous audit: re-derive every load-bearing numerical claim in the jspace
arc's committed observations from the on-disk artifacts, and compare against
what the observation files report. Prints per-check PASS/FAIL and exits 1 on
any failure (stage-7 audit named in
`research/arcs/04_jspace/plans/2026-07-18-jspace-design.md`).

Artifact resolution: each artifact is resolved data/-FIRST, cache/-fallback
(`_resolve`). The small DERIVED artifacts (every lens_eval / readout_scan /
structure_scan / verbal_report / entailed_swap / paperverbatim / nla_crosstie
this audit re-derives from) are promoted, committed via git-LFS under
`research/arcs/04_jspace/data/`, so checks B-L reproduce from a clean clone. The
FULL fitted lens tensors + their `.config.json` sidecars stay cache-only
(design Decision 4 — their committed 7-layer subsets suffice for inspection),
so the lens-integrity blocks (Check A + the refit lenses in H/I/J) resolve to
`cache/` and record a loud `MISSING` FAIL on a clone without the local cache.

A missing artifact is a loud FAIL (never a crash): the check records `MISSING`
and the dependent checks for that artifact are skipped, so the run still
reaches the SUMMARY and exits 1.

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
  H  n-budget stability (H1) — nf4 n=500 refit lens integrity; varfrac(k25)
     peak stays L21 on both n=100 and n=500; peak |delta| < 2% (obs -1.4%).
  I  quantization exonerated — 1.5B nf4 n=100 control: lens integrity; every
     structural value (varfrac peak L21/L0/L26/trough, logit-kurtosis) within
     ~0.006 of bf16; multihop rates + depth-of-emergence + late Spearman
     reproduce; peak layer == bf16 (L21).
  J  corpus sensitivity — C4-en refit: lens integrity + corpus_tag; the
     WORKSPACE-band (L17-26) peak is L21 = 0.124 identical to wikitext (the
     headline invariance), while the EARLY band diverges (L0 varfrac doubles
     to 0.174 -> global argmax is L0; logit-kurt L0 drops to 2.64); late-band
     eval rates + @50 + depth medians corpus-invariant; logit early/mid == 0.
  K  held-out-sample robustness — diversified C4 held-out set: 1.5B peak stays
     L21 0.133, depth within +/-2 layers of the base set, J fidelity >= base;
     7B pinned band-level (peak layer in {22,23}, trough in {16,17}, values in
     ranges not 3dp per the observation), ~2.6x below 1.5B, <=10% ceiling.
  L  entailed-property swaps (stage 5.2) — the 8-file swap bank (1.5B chat
     L18/L21/L24 + plain L21; 7B chat L18/L19/L22 + plain L22) + 4 paper-verbatim
     probes: clean accuracies; discrete flip rate == 0 everywhere; retention
     >= 0.9; injected-norm equalized across conditions; headline dlogp(swap)
     @s=2 (1.5B L18 jlens +2.13 / logit +0.07 / random +0.14; 7B L19 +1.250 /
     +0.178 / +0.040); jlens dominates both controls at the peak layers; the
     property effect peaks below the report layer (1.5B L18, 7B L19).

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
DATA = _REPO_ROOT / "research" / "arcs" / "04_jspace" / "data"
CACHE = DATA / "cache"


def _resolve(name: str) -> Path:
    """Prefer the committed ``data/`` copy (so the audit reproduces from a clean
    clone), falling back to the working ``cache/``. The small derived artifacts
    (evals, scans, swap banks, cross-tie) are promoted into ``data/`` and
    git-LFS-tracked; the FULL fitted lens tensors + their ``.config.json``
    sidecars stay cache-only (design Decision 4 — their committed 7-layer
    subsets suffice for inspection), so lens-integrity checks resolve to
    ``cache/`` and are MISSING (loud FAIL) on a clone without the local cache."""
    d = DATA / name
    return d if d.exists() else CACHE / name


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
    p = _resolve(name)
    if not p.exists():
        claim(f"artifact present: {name}", False, "present", "MISSING")
        return None
    return torch.load(p, weights_only=False)


def load_json_or_fail(name: str) -> dict[str, Any] | None:
    p = _resolve(name)
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
    # NOTE: the original 2026-07-18 stage-3 scan ran on CPU; the artifact was
    # regenerated 2026-07-21 on GPU (rich token capture backfill). bf16
    # backend numerics flipped two rank-boundary cells vs the observation's
    # CPU-run values: l_lastpos 19.0 -> 19.5 and n_never_j 15 -> 16 (of 108).
    # All other cells reproduced exactly; no conclusion depends on either.
    # Pins below follow the on-disk GPU artifact; the observation carries the
    # matching provenance note.
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt": {
        "key": "1.5b",
        "j_allpos": 23.0,
        "l_allpos": 19.0,
        "j_lastpos": 20.0,
        "l_lastpos": 19.5,
        "n_never_j": 16,
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


CROSSTIE_7B = "nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt"


def audit_crosstie() -> None:
    print()
    print("=" * 80)
    print(
        "CHECK G: NLA cross-tie (stage 6; 7B nf4 J-lens x NLA AV at hidden_states[20])"
    )
    print("=" * 80)

    d = load_pt_or_fail(CROSSTIE_7B)
    if d is None:
        return  # load_pt_or_fail already recorded the loud MISSING FAIL
    s = cast("dict[str, Any]", d["summary"])
    pp = cast("list[dict[str, Any]]", d["per_prompt"])

    # ---- shape + config identity (the layer-index fact is load-bearing) ----
    claim_eq("[G] per_prompt count == 25 (24 prompts + 1 control)", 25, len(pp))
    claim_eq("[G] jlens_layer == 19", 19, s["jlens_layer"])
    claim_eq("[G] nla_hidden_state == 20", 20, s["nla_hidden_state"])
    claim_eq("[G] decomp_k == 25", 25, s["decomp_k"])
    claim_eq("[G] n_neutral == 12", 12, s["n_neutral"])
    claim_eq("[G] n_concept == 12", 12, s["n_concept"])
    claim_eq("[G] n_decomp == 12", 12, s["n_decomp"])
    claim_eq("[G] vocab == 152064", 152064, s["vocab"])
    floor = float(s["chance_rank_floor"])  # |V|/2 == 76032

    # ---- pinned Experiment-A metrics (rank medians exact; rates to 3dp) ----
    claim_near(
        "[G] metric1_rank_median_concept",
        9585.0,
        float(s["metric1_rank_median_concept"]),
        atol=0.5,
    )
    claim_near(
        "[G] metric1_rank_median_neutral",
        103412.5,
        float(s["metric1_rank_median_neutral"]),
        atol=0.5,
    )
    claim_near(
        "[G] null_concept_matched_rank_median",
        9585.0,
        float(s["null_concept_matched_rank_median"]),
        atol=0.5,
    )
    claim_near(
        "[G] null_concept_mismatched_rank_median",
        16352.75,
        float(s["null_concept_mismatched_rank_median"]),
        atol=0.5,
    )
    claim_near(
        "[G] null_concept_matched_pctile",
        0.159,
        float(s["null_concept_matched_pctile"]),
        atol=0.001,
    )

    # ---- pinned Metric-3 concept-recovery rates ----
    claim_near(
        "[G] metric3_jlens_hit_rate",
        0.333,
        float(s["metric3_jlens_hit_rate"]),
        atol=0.001,
    )
    claim_near(
        "[G] metric3_nla_hit_rate", 0.917, float(s["metric3_nla_hit_rate"]), atol=0.001
    )
    claim_near(
        "[G] metric3_both_hit_rate",
        0.333,
        float(s["metric3_both_hit_rate"]),
        atol=0.001,
    )

    # ---- pinned Experiment-B carrier table ----
    comp = float(s["expB_carrier_component_mean"])
    res = float(s["expB_carrier_residual_mean"])
    ctrl = float(s["expB_carrier_resctrl_mean"])
    rand = float(s["expB_carrier_randunit_mean"])
    delta = float(s["expB_carrier_delta_mean"])
    crm = float(s["expB_component_rank_median"])
    claim_near("[G] expB_carrier_component_mean", 0.162, comp, atol=0.001)
    claim_near("[G] expB_carrier_residual_mean", 0.600, res, atol=0.001)
    claim_near("[G] expB_carrier_resctrl_mean", 0.606, ctrl, atol=0.001)
    claim_near("[G] expB_carrier_randunit_mean", 0.075, rand, atol=0.001)
    claim_near("[G] expB_carrier_delta_mean ~ 0", 0.0, delta, atol=0.02)
    claim_near("[G] expB_component_rank_median", 75166.75, crm, atol=0.5)

    # ---- load-bearing directional booleans (the stage-6 verdicts) ----
    sep = float(s["null_concept_mismatched_rank_median"]) - float(
        s["null_concept_matched_rank_median"]
    )
    claim(
        "[G] concept null separates (mismatched - matched > 5000)",
        sep > 5000.0,
        ">5000",
        round(sep, 2),
    )
    claim(
        "[G] residual carrier > 3x component carrier",
        res > 3.0 * comp,
        f">3x ({comp:.3f})",
        round(res, 3),
    )
    claim(
        "[G] |carrier_delta_mean| < 0.05 (J-space ~ norm-matched random)",
        abs(delta) < 0.05,
        "<0.05",
        round(delta, 4),
    )
    rel = abs(crm - floor) / floor
    claim(
        "[G] component_rank_median within 5% of chance floor",
        rel < 0.05,
        f"<5% of {floor:.0f}",
        f"{crm:.1f} ({rel * 100:.2f}%)",
    )

    # ---- raw re-derivation: metric1_rank_median_concept from per_prompt ----
    vals = [
        float(r["nla_rank_median"])
        for r in pp
        if r.get("tag") == "concept"
        and not np.isnan(float(r.get("nla_rank_median", np.nan)))
    ]
    claim_eq("[G] concept prompts carrying a rank median == 12", 12, len(vals))
    rederived = float(np.median(vals))
    claim_near(
        "[G] metric1_rank_median_concept re-derived from per_prompt raw",
        float(s["metric1_rank_median_concept"]),
        rederived,
        atol=1e-6,
    )

    # ---- persisted-readout integrity (so offline null re-derivation is sound) ----
    r0 = next(r for r in pp if r.get("tag") == "concept")
    ro = cast(Any, r0["readout_fp16"]).float()
    claim_eq("[G] readout_fp16 width == vocab", int(s["vocab"]), int(ro.numel()))
    claim_eq(
        "[G] readout_fp16 top-1 == persisted topk[0]",
        int(r0["readout_topk_ids"][0]),
        int(ro.argmax().item()),
    )
    topk_re = set(ro.topk(int(s["topk"])).indices.tolist())
    overlap = len(topk_re & set(int(i) for i in r0["readout_topk_ids"]))
    claim(
        "[G] readout_fp16 top-k overlaps persisted >= 48/50",
        overlap >= 48,
        ">=48",
        overlap,
    )


# ===========================================================================
# Stage-7 consolidation checks H-L: robustness-axis + entailed-swap artifact
# families added after Check G. Same conventions as A-G (missing artifact =>
# loud MISSING FAIL, never a crash; values re-derived from the on-disk
# artifacts and pinned per the observations).
# ===========================================================================


def _audit_lens(
    key: str,
    pt: str,
    cfg_name: str,
    d_model: int,
    model: str,
    mode: str,
    n_prompts: int,
) -> dict[str, Any] | None:
    """Shared lens-integrity block (parallels Check A) for the robustness-refit
    lenses: 27 layers == source_layers == range(27); d_model; n_prompts; all J
    finite; min Frobenius > 0; sidecar model/mode/n_prompts consistent. The full
    fitted lenses are cache-only (Decision 4), so this resolves to cache/."""
    d = load_pt_or_fail(pt)
    cfg = load_json_or_fail(cfg_name)
    if d is None:
        return None
    jmat: dict[int, torch.Tensor] = d["J"]
    source_layers: list[int] = list(d["source_layers"])
    expected_layers = list(range(27))
    claim_eq(f"[{key}] J layer count == 27", 27, len(jmat))
    claim_eq(f"[{key}] source_layers == 0..26", expected_layers, source_layers)
    claim_eq(f"[{key}] J keys == source_layers", expected_layers, sorted(jmat.keys()))
    claim_eq(f"[{key}] d_model", d_model, int(d["d_model"]))
    claim_eq(f"[{key}] n_prompts == {n_prompts}", n_prompts, int(d["n_prompts"]))
    shapes_ok = all(tuple(jmat[l].shape) == (d_model, d_model) for l in source_layers)
    claim(
        f"[{key}] every J_l is (d_model, d_model)",
        shapes_ok,
        f"(({d_model}, {d_model}) x27)",
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
        claim_eq(f"[{key}] sidecar model", model, cfg.get("model"))
        claim_eq(f"[{key}] sidecar mode", mode, cfg.get("mode"))
        claim_eq(f"[{key}] sidecar n_prompts", n_prompts, int(cfg.get("n_prompts", -1)))
    return d


def _struct_summary(fname: str, key: str) -> tuple[dict[str, Any], list[int]] | None:
    """Load a structure-scan artifact and run the shared integrity block."""
    d = load_pt_or_fail(fname)
    if d is None:
        return None
    return _check_struct_common(key, d)


def _eval_rates(fname: str) -> dict[str, Any] | None:
    d = load_pt_or_fail(fname)
    if d is None:
        return None
    return cast("dict[str, Any]", d["summary"]["per_eval"])


# ---- reference values from the bf16-wikitext baseline artifacts (Checks C/D):
#      the exoneration / invariance checks compare the refits against these.
BF16_VARFRAC_PEAK = 0.1237  # 1.5B bf16 varfrac(k25) peak value at L21
BF16_DEPTH_J, BF16_DEPTH_L = 23.0, 19.0  # depth-of-emergence allpos medians
BF16_L26_SPEARMAN_J = 0.8003  # 1.5B bf16 L26 last-pos Spearman (J)


STRUCT_NF4_N100 = "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt"
STRUCT_NF4_N500 = "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n500.pt"


def audit_n_stability() -> None:
    print()
    print("=" * 80)
    print("CHECK H: n-budget stability (nf4 n=500 refit; H1 exoneration)")
    print("=" * 80)
    # ---- n=500 nf4 lens integrity ----
    _audit_lens(
        "1.5b-nf4-n500",
        "jlens_qwen2.5-1.5b_nf4_n500.pt",
        "jlens_qwen2.5-1.5b_nf4_n500.config.json",
        1536,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "nf4",
        500,
    )
    # ---- n=500 structure scan: peak still L21, value pinned ----
    r500 = _struct_summary(STRUCT_NF4_N500, "1.5b-nf4-n500")
    r100 = _struct_summary(STRUCT_NF4_N100, "1.5b-nf4-n100(H)")
    if r500 is None or r100 is None:
        return
    s500, layers = r500
    s100, _ = r100
    vf500 = np.array(s500["mean_varfrac"][25], dtype=float)
    vf100 = np.array(s100["mean_varfrac"][25], dtype=float)
    p500, p100 = int(vf500.argmax()), int(vf100.argmax())
    claim_eq("[H] nf4 n=500 varfrac(k25) peak layer == 21", 21, layers[p500])
    claim_eq("[H] nf4 n=100 varfrac(k25) peak layer == 21", 21, layers[p100])
    claim_near("[H] nf4 n=500 peak value", 0.1235, float(vf500[p500]), atol=0.002)
    claim_near("[H] nf4 n=100 peak value", 0.1252, float(vf100[p100]), atol=0.002)
    # ---- n-stability: peak moves < 2% from n=100 -> n=500 (obs: -1.4%) ----
    rel = abs(float(vf500[p500]) - float(vf100[p100])) / float(vf100[p100])
    claim("[H] nf4 peak |delta| n100->n500 < 2%", rel < 0.02, "< 0.02", round(rel, 4))
    claim_near("[H] nf4 peak relative delta ~ 1.4%", 0.014, rel, atol=0.005)


STRUCT_1P5B_NF4 = STRUCT_NF4_N100
EVAL_1P5B_NF4 = "lens_eval_qwen2.5-1.5b_nf4_n100.pt"
READOUT_1P5B_NF4 = "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_nf4_n100.pt"


def audit_quant_exoneration() -> None:
    print()
    print("=" * 80)
    print("CHECK I: quantization exonerated (1.5B nf4 n=100 control vs bf16)")
    print("=" * 80)
    # ---- nf4 lens integrity ----
    _audit_lens(
        "1.5b-nf4",
        "jlens_qwen2.5-1.5b_nf4_n100.pt",
        "jlens_qwen2.5-1.5b_nf4_n100.config.json",
        1536,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "nf4",
        100,
    )
    # ---- structure: nf4 == bf16 to within ~0.006 on every structural value ----
    r = _struct_summary(STRUCT_1P5B_NF4, "1.5b-nf4")
    if r is not None:
        s, layers = r
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        lk = np.array(s["mean_readout_kurtosis"], dtype=float)
        peak_i, trough_i = int(vf.argmax()), int(vf.argmin())
        claim_eq("[I] nf4 varfrac(k25) peak layer == 21", 21, layers[peak_i])
        claim_near("[I] nf4 varfrac(k25) peak value", 0.125, float(vf[peak_i]))
        claim_near("[I] nf4 varfrac(k25) L0", 0.079, float(vf[0]))
        claim_near("[I] nf4 varfrac(k25) final layer", 0.067, float(vf[-1]))
        claim_eq("[I] nf4 varfrac(k25) trough layer == 14", 14, layers[trough_i])
        claim_near("[I] nf4 varfrac(k25) trough value", 0.056, float(vf[trough_i]))
        claim_near("[I] nf4 logit-kurt L0", 4.556, float(lk[0]), atol=0.01)
        kt_i = int(lk.argmin())
        claim_eq("[I] nf4 logit-kurt trough layer == 18", 18, layers[kt_i])
        claim_near("[I] nf4 logit-kurt trough value", 0.906, float(lk[kt_i]))
        # exoneration boolean: nf4 peak within ~0.006 of the bf16 baseline peak.
        claim(
            "[I] nf4 peak within 0.006 of bf16 baseline (same L21)",
            abs(float(vf[peak_i]) - BF16_VARFRAC_PEAK) < 0.006,
            f"< 0.006 of {BF16_VARFRAC_PEAK}",
            round(abs(float(vf[peak_i]) - BF16_VARFRAC_PEAK), 4),
        )
    # ---- eval rates: J-exclusivity + late band reproduce under nf4 ----
    pe = _eval_rates(EVAL_1P5B_NF4)
    if pe is not None:
        mh = pe["multihop"]
        claim_near(
            "[I] nf4 multihop J@10 overall", 0.592, _rate(mh, "rates_j", "overall", 10)
        )
        claim_near(
            "[I] nf4 multihop logit@10 overall",
            0.418,
            _rate(mh, "rates_l", "overall", 10),
        )
        claim_near(
            "[I] nf4 multihop J@10 early", 0.194, _rate(mh, "rates_j", "early", 10)
        )
        claim_near("[I] nf4 multihop J@10 mid", 0.126, _rate(mh, "rates_j", "mid", 10))
        claim_near(
            "[I] nf4 multihop J@10 late", 0.505, _rate(mh, "rates_j", "late", 10)
        )
        claim_near(
            "[I] nf4 multihop logit@10 early (J-exclusive)",
            0.00,
            _rate(mh, "rates_l", "early", 10),
        )
        claim_near(
            "[I] nf4 multihop logit@10 mid (J-exclusive)",
            0.00,
            _rate(mh, "rates_l", "mid", 10),
        )
        claim_near(
            "[I] nf4 association J@10 overall (floored)",
            0.00,
            _rate(pe["association"], "rates_j", "overall", 10),
        )
    # ---- readout: depth-of-emergence + late Spearman unchanged from bf16 ----
    d = load_pt_or_fail(READOUT_1P5B_NF4)
    if d is not None:
        summary = d["summary"]
        layers_r = list(summary["layers"])
        rederived = _rederive_emergence(d["per_prompt"], layers_r)
        stored = summary["depth_of_emergence_top10"]
        for field in ("median_layer_j_allpos", "median_layer_l_allpos"):
            claim_eq(
                f"[I] nf4 {field} rederived==stored", stored[field], rederived[field]
            )
        claim_eq(
            "[I] nf4 depth J allpos == 23 (== bf16)",
            BF16_DEPTH_J,
            float(stored["median_layer_j_allpos"]),
        )
        claim_eq(
            "[I] nf4 depth logit allpos == 19 (== bf16)",
            BF16_DEPTH_L,
            float(stored["median_layer_l_allpos"]),
        )
        j_last = list(summary["layer_mean_spearman_j_last"])
        l_last = list(summary["layer_mean_spearman_l_last"])
        claim_near("[I] nf4 L26 Spearman J-last", 0.789, float(j_last[26]))
        claim_near("[I] nf4 L26 Spearman logit-last", 0.880, float(l_last[26]))


STRUCT_1P5B_C4 = (
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"
)
EVAL_1P5B_C4 = "lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt"
READOUT_1P5B_C4 = (
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"
)


def audit_corpus_sensitivity() -> None:
    print()
    print("=" * 80)
    print("CHECK J: corpus sensitivity (C4-en refit; workspace-band invariance)")
    print("=" * 80)
    # ---- c4en lens integrity + corpus_tag ----
    d = _audit_lens(
        "1.5b-c4en",
        "jlens_qwen2.5-1.5b_bf16_n100_c4en.pt",
        "jlens_qwen2.5-1.5b_bf16_n100_c4en.config.json",
        1536,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "bf16",
        100,
    )
    cfg = load_json_or_fail("jlens_qwen2.5-1.5b_bf16_n100_c4en.config.json")
    if cfg is not None:
        claim_eq("[J] c4en lens corpus_tag == 'c4en'", "c4en", cfg.get("corpus_tag"))
    del d
    # ---- structure: workspace-band peak IDENTITY (L21 0.124); early band diverges ----
    r = _struct_summary(STRUCT_1P5B_C4, "1.5b-c4en")
    if r is not None:
        s, layers = r
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        lk = np.array(s["mean_readout_kurtosis"], dtype=float)
        # workspace band L17..26: peak is L21 = 0.124 on BOTH corpora (headline).
        band = [i for i, l in enumerate(layers) if 17 <= l <= 26]
        band_peak = band[int(np.argmax(vf[band]))]
        claim_eq(
            "[J] c4en workspace-band(17..26) peak layer == 21", 21, layers[band_peak]
        )
        claim_near(
            "[J] c4en L21 varfrac == 0.124 (peak identity)", 0.124, float(vf[band_peak])
        )
        claim(
            "[J] c4en L21 within 0.003 of bf16 baseline (corpus-invariant)",
            abs(float(vf[band_peak]) - BF16_VARFRAC_PEAK) < 0.003,
            f"< 0.003 of {BF16_VARFRAC_PEAK}",
            round(abs(float(vf[band_peak]) - BF16_VARFRAC_PEAK), 4),
        )
        # early-band divergence: L0 varfrac doubles (0.082 -> 0.174) and now
        # exceeds the workspace peak -> global argmax is L0, not L21.
        claim_near(
            "[J] c4en L0 varfrac == 0.174 (early-band divergence)", 0.174, float(vf[0])
        )
        claim(
            "[J] c4en global argmax is L0 (early band > workspace)",
            int(vf.argmax()) == 0 and float(vf[0]) > float(vf[band_peak]),
            "L0 > L21",
            f"L{layers[int(vf.argmax())]} ({vf[0]:.3f} vs {vf[band_peak]:.3f})",
        )
        claim_near(
            "[J] c4en logit-kurt L0 == 2.641 (early-band drop)",
            2.641,
            float(lk[0]),
            atol=0.01,
        )
        kt_i = int(lk.argmin())
        claim_eq("[J] c4en logit-kurt trough layer == 18", 18, layers[kt_i])
        claim_near("[J] c4en logit-kurt trough value", 0.872, float(lk[kt_i]))
    # ---- eval: late band + @50 corpus-invariant; logit early/mid stays 0 ----
    pe = _eval_rates(EVAL_1P5B_C4)
    if pe is not None:
        mh = pe["multihop"]
        claim_near(
            "[J] c4en multihop J@10 late", 0.485, _rate(mh, "rates_j", "late", 10)
        )
        claim_near(
            "[J] c4en multihop logit@10 late", 0.398, _rate(mh, "rates_l", "late", 10)
        )
        claim_near(
            "[J] c4en multihop J@10 overall", 0.485, _rate(mh, "rates_j", "overall", 10)
        )
        claim_near(
            "[J] c4en multihop J@50 overall", 0.748, _rate(mh, "rates_j", "overall", 50)
        )
        claim_near(
            "[J] c4en multihop J@10 early (reduced)",
            0.116,
            _rate(mh, "rates_j", "early", 10),
        )
        claim_near(
            "[J] c4en multihop logit@10 early (J-exclusive)",
            0.00,
            _rate(mh, "rates_l", "early", 10),
        )
        claim_near(
            "[J] c4en multihop logit@10 mid (J-exclusive)",
            0.00,
            _rate(mh, "rates_l", "mid", 10),
        )
        claim_near(
            "[J] c4en association J@10 overall (floored)",
            0.00,
            _rate(pe["association"], "rates_j", "overall", 10),
        )
    # ---- readout: depth medians identical; late Spearman within 0.003 ----
    d = load_pt_or_fail(READOUT_1P5B_C4)
    if d is not None:
        summary = d["summary"]
        stored = summary["depth_of_emergence_top10"]
        claim_eq(
            "[J] c4en depth J allpos == 23 (== wikitext)",
            BF16_DEPTH_J,
            float(stored["median_layer_j_allpos"]),
        )
        claim_eq(
            "[J] c4en depth logit allpos == 19 (== wikitext)",
            BF16_DEPTH_L,
            float(stored["median_layer_l_allpos"]),
        )
        j26 = float(list(summary["layer_mean_spearman_j_last"])[26])
        claim_near("[J] c4en L26 Spearman J-last == 0.803", 0.803, j26)
        claim(
            "[J] c4en L26 Spearman J within 0.003 of wikitext",
            abs(j26 - BF16_L26_SPEARMAN_J) < 0.003,
            f"< 0.003 of {BF16_L26_SPEARMAN_J}",
            round(abs(j26 - BF16_L26_SPEARMAN_J), 4),
        )


STRUCT_1P5B_HELDOUT = (
    "structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt"
)
READOUT_1P5B_HELDOUT = (
    "readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt"
)
STRUCT_7B_HELDOUT = (
    "structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt"
)
READOUT_7B_HELDOUT = (
    "readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt"
)


def audit_heldout() -> None:
    print()
    print("=" * 80)
    print("CHECK K: held-out-sample robustness (diversified C4 held-out set)")
    print("=" * 80)
    # ---- 1.5B: peak stays L21 0.133; depth within +/-2 of the base set ----
    r = _struct_summary(STRUCT_1P5B_HELDOUT, "1.5b-heldout")
    if r is not None:
        s, layers = r
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        lk = np.array(s["mean_readout_kurtosis"], dtype=float)
        peak_i = int(vf.argmax())
        claim_eq("[K] 1.5B heldout varfrac(k25) peak layer == 21", 21, layers[peak_i])
        claim_near("[K] 1.5B heldout peak value == 0.133", 0.133, float(vf[peak_i]))
        claim_near("[K] 1.5B heldout L0 varfrac == 0.084", 0.084, float(vf[0]))
        kt_i = int(lk.argmin())
        claim_eq("[K] 1.5B heldout logit-kurt trough layer == 17", 17, layers[kt_i])
        claim_near(
            "[K] 1.5B heldout logit-kurt trough value == 1.00",
            1.000,
            float(lk[kt_i]),
            atol=0.01,
        )
    d = load_pt_or_fail(READOUT_1P5B_HELDOUT)
    if d is not None:
        summary = d["summary"]
        layers_r = list(summary["layers"])
        rederived = _rederive_emergence(d["per_prompt"], layers_r)
        stored = summary["depth_of_emergence_top10"]
        for field in ("median_layer_j_allpos", "median_layer_l_allpos"):
            claim_eq(
                f"[K] 1.5B heldout {field} rederived==stored",
                stored[field],
                rederived[field],
            )
        dj = float(stored["median_layer_j_allpos"])
        dl = float(stored["median_layer_l_allpos"])
        claim_eq("[K] 1.5B heldout depth J allpos == 22", 22.0, dj)
        claim_eq("[K] 1.5B heldout depth logit allpos == 19", 19.0, dl)
        claim(
            "[K] 1.5B heldout depth within +/-2 layers of base (23/19)",
            abs(dj - BF16_DEPTH_J) <= 2.0 and abs(dl - BF16_DEPTH_L) <= 2.0,
            "|dj-23|<=2 and |dl-19|<=2",
            f"dj={dj} dl={dl}",
        )
        j26 = float(list(summary["layer_mean_spearman_j_last"])[26])
        claim(
            "[K] 1.5B heldout J readout fidelity >= base (0.870 vs 0.800)",
            j26 >= BF16_L26_SPEARMAN_J,
            f">= {BF16_L26_SPEARMAN_J}",
            round(j26, 4),
        )
    # ---- 7B: band-level pins (low occupancy => quote ranges, not 3dp) ----
    r7 = _struct_summary(STRUCT_7B_HELDOUT, "7b-heldout")
    if r7 is not None:
        s, layers = r7
        vf = np.array(s["mean_varfrac"][25], dtype=float)
        peak_i, trough_i = int(vf.argmax()), int(vf.argmin())
        claim(
            "[K] 7B heldout varfrac peak layer in {22,23}",
            layers[peak_i] in (22, 23),
            "{22,23}",
            layers[peak_i],
        )
        claim(
            "[K] 7B heldout varfrac trough layer in {16,17}",
            layers[trough_i] in (16, 17),
            "{16,17}",
            layers[trough_i],
        )
        claim(
            "[K] 7B heldout peak value in [0.03,0.07]",
            0.03 <= float(vf[peak_i]) <= 0.07,
            "[0.03,0.07]",
            round(float(vf[peak_i]), 4),
        )
        claim(
            "[K] 7B heldout trough value in [0.005,0.03]",
            0.005 <= float(vf[trough_i]) <= 0.03,
            "[0.005,0.03]",
            round(float(vf[trough_i]), 4),
        )
        claim(
            "[K] 7B heldout varfrac ceiling <= 0.10 (U-profile)",
            bool((vf <= 0.10).all()),
            "<= 0.10",
            round(float(vf.max()), 4),
        )
        # ~2.6x below 1.5B on the matched held-out set.
        if r is not None:
            vf15 = np.array(r[0]["mean_varfrac"][25], dtype=float)
            ratio = float(vf15.max()) / float(vf.max())
            claim(
                "[K] 1.5B/7B heldout peak ratio in [2.0,3.5] (~2.6x)",
                2.0 <= ratio <= 3.5,
                "[2.0,3.5]",
                round(ratio, 2),
            )
    d7 = load_pt_or_fail(READOUT_7B_HELDOUT)
    if d7 is not None:
        summary = d7["summary"]
        layers_r = list(summary["layers"])
        rederived = _rederive_emergence(d7["per_prompt"], layers_r)
        stored = summary["depth_of_emergence_top10"]
        for field in ("median_layer_j_allpos", "median_layer_l_allpos"):
            claim_eq(
                f"[K] 7B heldout {field} rederived==stored",
                stored[field],
                rederived[field],
            )
        finite_ok = np.isfinite(float(stored["median_layer_j_allpos"])) and np.isfinite(
            float(stored["median_layer_l_allpos"])
        )
        claim(
            "[K] 7B heldout depth medians finite",
            finite_ok,
            "finite",
            "ok" if finite_ok else "NaN",
        )


# ---- Stage-5.2 entailed-property swap bank. Each artifact: per_item(33) +
# summary.metrics[cond@strength] with property_flip_rate, clean_retention_rate,
# mean_dlogp_swap_answer, mean_injected_norm. Conditions: jlens/logitlens/random.
ES_CONDS = ("jlens", "logitlens", "random")
# key, filename, layer, style, n_correct (clean, /33), is_peak_layer
ES_BANK: list[tuple[str, str, int, str, int, bool]] = [
    (
        "1.5b chat L18",
        "entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        18,
        "chat",
        17,
        True,
    ),
    (
        "1.5b chat L21",
        "entailed_swap_chat_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        21,
        "chat",
        17,
        False,
    ),
    (
        "1.5b chat L24",
        "entailed_swap_chat_L24_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        24,
        "chat",
        17,
        False,
    ),
    (
        "1.5b plain L21",
        "entailed_swap_plain_L21_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt",
        21,
        "plain",
        4,
        False,
    ),
    (
        "7b chat L18",
        "entailed_swap_chat_L18_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        18,
        "chat",
        30,
        False,
    ),
    (
        "7b chat L19",
        "entailed_swap_chat_L19_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        19,
        "chat",
        30,
        True,
    ),
    (
        "7b chat L22",
        "entailed_swap_chat_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        22,
        "chat",
        30,
        False,
    ),
    (
        "7b plain L22",
        "entailed_swap_plain_L22_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt",
        22,
        "plain",
        11,
        False,
    ),
]
# Headline dlogp(swap-property) @ s=2 at the peak layers (obs tables).
ES_HEADLINE = {
    "1.5b chat L18": {"jlens": 2.13, "logitlens": 0.07, "random": 0.14},
    "7b chat L19": {"jlens": 1.250, "logitlens": 0.178, "random": 0.040},
}
ES_PAPERVERBATIM = [
    "entailed_paperverbatim_chat_all_L19_7b.pt",
    "entailed_paperverbatim_chat_auto_L19_7b.pt",
    "entailed_paperverbatim_plain_all_L19_7b.pt",
    "entailed_paperverbatim_plain_auto_L19_7b.pt",
]


def audit_entailed_swap() -> None:
    print()
    print("=" * 80)
    print("CHECK L: entailed-property swaps (stage 5.2; J-lens-specific graded effect)")
    print("=" * 80)
    # dlogp at peak layer per scale, collected for the depth-localization check.
    peak_jlens_dlogp: dict[str, dict[int, float]] = {"1.5b": {}, "7b": {}}
    for key, fname, layer, style, n_correct, is_peak in ES_BANK:
        d = load_pt_or_fail(fname)
        if d is None:
            continue
        s = d["summary"]
        m = s["metrics"]
        claim_eq(f"[L] {key} n_items == 33", 33, len(d["per_item"]))
        claim_eq(f"[L] {key} layer", layer, int(s["layer"]))
        claim_eq(f"[L] {key} prompt_style", style, s["prompt_style"])
        claim_eq(
            f"[L] {key} clean n_correct == {n_correct}/33",
            n_correct,
            int(s["n_correct"]),
        )
        claim_eq(f"[L] {key} conditions", list(ES_CONDS), list(s["conditions"]))
        # (1) discrete flip rate == 0 everywhere (both metrics, both strengths).
        flips = [
            float(m[f"{c}@{st}"][fld])
            for c in ES_CONDS
            for st in ("1.0", "2.0")
            for fld in ("property_flip_rate", "property_flip_tok1_rate")
        ]
        claim(
            f"[L] {key} property flip rate == 0 (all conds x strengths)",
            max(flips) == 0.0,
            "0.0",
            max(flips),
        )
        # (2) retention bound: the swap never breaks the model (>= 0.9 all conds).
        retains = [
            float(m[f"{c}@{st}"]["clean_retention_rate"])
            for c in ES_CONDS
            for st in ("1.0", "2.0")
        ]
        claim(
            f"[L] {key} clean retention >= 0.9 (all conds)",
            min(retains) >= 0.9,
            ">= 0.9",
            round(min(retains), 4),
        )
        # (3) magnitude equalization: injected-norm identical across conditions.
        for st in ("1.0", "2.0"):
            norms = [float(m[f"{c}@{st}"]["mean_injected_norm"]) for c in ES_CONDS]
            spread = max(norms) - min(norms)
            claim(
                f"[L] {key} injected-norm equal across conds @{st}",
                spread < 1e-3,
                "< 1e-3",
                round(spread, 6),
            )
        scale = "1.5b" if key.startswith("1.5b") else "7b"
        if style == "chat":
            peak_jlens_dlogp[scale][layer] = float(
                m["jlens@2.0"]["mean_dlogp_swap_answer"]
            )
        # (4) headline dlogp @s=2 at the peak layers.
        if key in ES_HEADLINE:
            for c, val in ES_HEADLINE[key].items():
                claim_near(
                    f"[L] {key} {c}@2 dlogp(swap)",
                    val,
                    float(m[f"{c}@2.0"]["mean_dlogp_swap_answer"]),
                    atol=0.01,
                )
        # (5) ordering at the peak layers: jlens dominates BOTH controls.
        #     NB: this is jlens > logitlens AND jlens > random -- NOT the strict
        #     jlens>logitlens>random chain, which fails at 1.5B L18 where random
        #     (0.14) exceeds logitlens (0.07). The load-bearing claim is that the
        #     J-lens direction beats every equalized-magnitude control.
        if is_peak:
            j = float(m["jlens@2.0"]["mean_dlogp_swap_answer"])
            lo = float(m["logitlens@2.0"]["mean_dlogp_swap_answer"])
            rd = float(m["random@2.0"]["mean_dlogp_swap_answer"])
            claim(
                f"[L] {key} jlens dlogp > both controls @2 (J-lens-specific)",
                j > lo and j > rd,
                "jlens > max(logit,random)",
                f"{j:.3f} > max({lo:.3f},{rd:.3f})",
            )
            claim(
                f"[L] {key} jlens dlogp >= 5x larger control @2",
                j >= 5.0 * max(lo, rd),
                ">= 5x",
                round(j / max(lo, rd), 2),
            )
    # (6) depth localization: the property effect peaks below the report layer.
    if {18, 21, 24} <= set(peak_jlens_dlogp["1.5b"]):
        p = peak_jlens_dlogp["1.5b"]
        claim(
            "[L] 1.5b jlens dlogp peaks at L18 (< L21 < L24 report band)",
            p[18] > p[21] > p[24],
            "L18 > L21 > L24",
            f"{p[18]:.2f} > {p[21]:.2f} > {p[24]:.2f}",
        )
    if {18, 19, 22} <= set(peak_jlens_dlogp["7b"]):
        p = peak_jlens_dlogp["7b"]
        claim(
            "[L] 7b jlens dlogp peaks at L19 (> L18, > L22 report)",
            p[19] > p[18] and p[19] > p[22],
            "L19 max",
            f"L18={p[18]:.2f} L19={p[19]:.2f} L22={p[22]:.2f}",
        )
    # (7) paperverbatim probes: load, flip==0, chat-all headline dlogp ~ +1.60.
    for fname in ES_PAPERVERBATIM:
        d = load_pt_or_fail(fname)
        if d is None:
            continue
        s = d["summary"]
        m = s["metrics"]
        claim_eq(
            f"[L] paperverbatim {s['prompt_style']}/{s['swap_scope']} n_items == 3",
            3,
            len(d["per_item"]),
        )
        pv_flips = [
            float(m[f"{c}@{st}"]["property_flip_rate"])
            for c in ES_CONDS
            for st in ("1.0", "2.0")
        ]
        claim(
            f"[L] paperverbatim {s['prompt_style']}/{s['swap_scope']} flip rate == 0",
            max(pv_flips) == 0.0,
            "0.0",
            max(pv_flips),
        )
        if fname == "entailed_paperverbatim_chat_all_L19_7b.pt":
            claim_near(
                "[L] paperverbatim chat/all jlens@2 dlogp ~ +1.60",
                1.60,
                float(m["jlens@2.0"]["mean_dlogp_swap_answer"]),
                atol=0.02,
            )


def main() -> None:
    audit_lens_integrity()
    audit_eval_tables()
    audit_readout_scan()
    audit_structure_scan()
    audit_verbal_report()
    audit_verbal_report_6c()
    audit_crosstie()
    audit_n_stability()
    audit_quant_exoneration()
    audit_corpus_sensitivity()
    audit_heldout()
    audit_entailed_swap()
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
