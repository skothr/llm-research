#!/usr/bin/env bash
#
# jspace_rerun_scans.sh — arc 04, Step 2 of the C4-redaction re-run.
# Date: 2026-08-16
#
# Purpose
# -------
# Regenerate, in place, the 10 committed C4-dependent derived artifacts of
# research arc 04 (`research/arcs/04_jspace/data/`) against the three lenses
# refit on the *redacted* C4/wikitext corpora. The committed versions were
# computed on pre-redaction text and are stale.
#
# Plan reference
# --------------
#   research/arcs/04_jspace/plans/2026-07-29-c4-redaction-rerun.md § "Step 2 —
#   regenerate the 10 derived artifacts"
# Steps 0 (corpus regeneration) and 1 (the three lens refits) must already be
# complete; this script consumes their outputs and runs no fitting.
#
# The two contamination channels (plan § Scope):
#   Channel 1 — lens FIT on C4. Uses the refit c4en lens; evaluated on the
#       *default wikitext* held-out set, exactly as the committed artifacts
#       were (MANIFEST `inputs`: heldout_prompts_wikitext103_n30.json). The
#       corpus axis varies the FITTING corpus and holds the eval set fixed.
#   Channel 2 — wikitext lens EVALUATED on C4 held-out docs. Uses the refit
#       wikitext lenses (1.5B bf16 and 7B nf4) with
#       --prompts .../heldout_prompts_c4en_n30.json, which auto-derives the
#       `_heldoutc4en` output infix (jspace_readout_scan.py `heldout_tag()`).
#
# Every command below was reconstructed from, and cross-checked against:
#   * each script's argparser + output-name derivation
#     (jspace_readout_scan.py `heldout_tag()` / `default_out()`, reused by
#     jspace_structure_scan.py; jspace_lens_eval.py's lens-stem default;
#     jspace_paper_metric_varfrac.py's ARC_DATA name),
#   * MANIFEST.json `producing_command` / `inputs` for all 10 files,
#   * the config dicts stored INSIDE each stale artifact (n_prompts, layers,
#     ks, k_snap/k_max, n_rand, rand_seed_base, n_boot, prompts_file),
#   * observations/2026-07-20-jspace-structure-stage4.md § Reproducibility,
#     observations/2026-07-20-scale-comparison-7b-vs-1p5b-h2.md § Reproducibility,
#     observations/2026-07-20-corpus-sensitivity-c4-1p5b.md § Reproducibility,
#     observations/2026-07-24-paper-metric-varfrac-recompute.md § Reproducibility.
#
# --out is passed explicitly for every run. jspace_paper_metric_varfrac.py
# already defaults into `data/`, but readout/structure/lens_eval default into
# `data/cache/` (or the lens's directory) — the committed artifacts live in
# `data/`, so relying on those defaults would leave the stale files untouched.
# Each --out below is the *derived default name* with the directory corrected.
#
# Ordering is a dependency order, not a preference: `jspace_paper_metric_varfrac.py
# --scan` reads the structure-scan artifact and validates its replicated
# varfrac@k-snap against it bit-for-bit, so each family's structure scan runs
# first and the paper-metric run consumes the FRESH file.
#
# Device: --device cuda is passed to EVERY run. It is the default for
# jspace_structure_scan.py / jspace_paper_metric_varfrac.py but NOT for
# jspace_readout_scan.py / jspace_lens_eval.py (both default to cpu), and the
# committed artifacts are GPU-computed — evidenced by
# observations/2026-07-18-readout-scan-1p5b-first-pass.md:17-22 ("the original
# run above was CPU; the artifact was regenerated on GPU ... bf16 backend
# numerics flipped two rank-boundary cells") and by the stale artifacts' own
# `mean_seconds_per_prompt` (9.5 s/prompt on-disk vs the 19.4 s/prompt that
# observation records for its CPU run). Leaving these two at their cpu default
# would inject a backend-numerics change on top of the redaction change.
#
# Runtime: ~1.4-1.6 h total. Each per-run figure below is either the plan's
# measured cost or the stale artifact's own recorded mean_seconds_per_prompt
# x its prompt count; add ~5.5 min model load per 7B run
# (observations/2026-07-20-jspace-structure-stage4.md:104).
#
# Usage (from anywhere; the script cds to the repo root itself):
#     bash examples/jspace_rerun_scans.sh
#
# GPU: exclusive use of the RTX 2080 (8 GB) is assumed. Hand the card back to
# the desktop first — the 7B nf4 runs will OOM against a desktop compositor.

set -euo pipefail

# --- Locations ---------------------------------------------------------------
# The python scripts resolve their data paths RELATIVE to the working directory
# (jspace_paper_metric_varfrac.py: `ARC_DATA = Path("research/arcs/04_jspace/data")`),
# so every run must happen from the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Repo-local venv by convention (worktrees symlink the main checkout's .venv —
# see CLAUDE.md); override with JSPACE_PY for a non-standard layout.
PY="${JSPACE_PY:-${REPO_ROOT}/.venv/bin/python}"
DATA=research/arcs/04_jspace/data
CACHE="${DATA}/cache"
LOGDIR="${CACHE}/logs"

LENS_C4EN="${CACHE}/jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"
LENS_WT_1P5B="${CACHE}/jlens_qwen2.5-1.5b_bf16_n100.pt"
LENS_WT_7B="${CACHE}/jlens_qwen2.5-7b_nf4_n100.pt"
HELDOUT_C4EN="${DATA}/heldout_prompts_c4en_n30.json"

# Recommended by jspace_structure_scan.py's module docstring for the RTX 2080;
# load-bearing for the 7B nf4 runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Pre-flight --------------------------------------------------------------
fail() { echo "FATAL: $*" >&2; exit 1; }

[[ -x "${PY}" ]] || fail "venv python not found or not executable: ${PY}"
for f in "${LENS_C4EN}" "${LENS_WT_1P5B}" "${LENS_WT_7B}" "${HELDOUT_C4EN}"; do
    [[ -f "${f}" ]] || fail "missing required input: ${f} (Step 0/1 incomplete?)"
done
# jspace_lens_eval.py needs its evaluation-set dir (DEFAULT_EVAL_DIR, a path
# into the pinned jacobian-lens checkout). Import the module and read the
# constant it will actually use — single source of truth regardless of how
# that default is expressed (literal today; JSPACE_EVAL_DIR-overridable env
# lookup after #43) — checked now rather than 20 minutes into the run.
EVAL_DIR="$("${PY}" -c "
import sys
sys.path.insert(0, 'examples')
import jspace_lens_eval
print(jspace_lens_eval.DEFAULT_EVAL_DIR)
")" || fail "could not import examples/jspace_lens_eval.py to derive the eval dir"
[[ -n "${EVAL_DIR}" && -d "${EVAL_DIR}" ]] || fail \
    "lens-eval set dir missing: '${EVAL_DIR}' — point JSPACE_EVAL_DIR at the pinned jacobian-lens checkout's data/evaluations"

mkdir -p "${LOGDIR}"

START_EPOCH="$(date +%s)"
echo "=== jspace C4-redaction re-run, Step 2 (10 derived artifacts) ==="
echo "repo root : ${REPO_ROOT}"
echo "python    : ${PY}"
echo "logs      : ${LOGDIR}"
echo "started   : $(date -u -d "@${START_EPOCH}" '+%Y-%m-%dT%H:%M:%SZ') (epoch ${START_EPOCH})"
echo

RUN_INDEX=0
TOTAL_RUNS=10

# run <log-name> <target-artifact> -- <command ...>
#
# Echoes the command, runs it under tee to the log, then verifies the target
# artifact's mtime is strictly newer than this script's start. A script that
# exits 0 without having rewritten its target (wrong --out, silent default) is
# the exact failure this guards.
run() {
    local name="$1" target="$2"
    shift 2
    [[ "$1" == "--" ]] || fail "run(): expected -- before the command"
    shift

    RUN_INDEX=$((RUN_INDEX + 1))
    local log="${LOGDIR}/scan_${name}.log"

    echo "--- [${RUN_INDEX}/${TOTAL_RUNS}] ${name} ---"
    echo "target: ${target}"
    echo "log   : ${log}"
    printf '+'; printf ' %q' "$@"; printf '\n'

    local t0 t1
    t0="$(date +%s)"
    "$@" 2>&1 | tee "${log}"
    t1="$(date +%s)"

    [[ -f "${target}" ]] || fail "${name}: target artifact does not exist: ${target}"
    local mtime
    mtime="$(stat -c %Y "${target}")"
    if (( mtime <= START_EPOCH )); then
        fail "${name}: target NOT rewritten — ${target} mtime ${mtime} is not newer than script start ${START_EPOCH}. The run wrote somewhere else (check --out) or silently reused a cached path."
    fi
    echo "[ok] ${name} in $((t1 - t0))s; ${target} refreshed (mtime ${mtime})"
    echo
}

# =============================================================================
# Channel 1 — the C4-FIT lens (jlens_qwen2.5-1.5b_bf16_n100_c4en.pt).
# Evaluated on the DEFAULT wikitext held-out set (script default --prompts), so
# heldout_tag() returns "" and the output infix is the lens stem's own `_c4en`.
# =============================================================================

# 1/10 — structure scan first: the paper-metric run below reads it via --scan.
# Stale artifact's stored summary: n_prompts=30, ks=[5,10,25,50], n_mid=8,
# layers=0..26 (all fitted source layers; --layers omitted so the script takes
# lens.source_layers, which is 27 layers for BOTH the 1.5B and the 7B lens —
# confirmed from the refit .pt sizes and from every stale artifact's stored
# `layers`). ~7 min (13.6 s/prompt x 30,
# observations/2026-07-20-jspace-structure-stage4.md:103).
run structure_c4en_1p5b \
    "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt" \
    -- "${PY}" examples/jspace_structure_scan.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_C4EN}" \
        --n-prompts 30 \
        --ks 5,10,25,50 \
        --out "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"

# 2/10 — paper-metric on the FRESH structure scan above.
# MANIFEST producing_command + stale config: --n-rand 8 --rand-seed-base 10000,
# k_snap=25, k_max=50, n_boot=2000, scan-grid (not --all-positions). ~6 min
# (observations/2026-07-24-paper-metric-varfrac-recompute.md:228).
run paper_metric_c4en_1p5b \
    "${DATA}/paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt" \
    -- "${PY}" examples/jspace_paper_metric_varfrac.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_C4EN}" \
        --scan "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt" \
        --n-prompts 30 \
        --n-rand 8 \
        --rand-seed-base 10000 \
        --out "${DATA}/paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"

# 3/10 — readout scan. Stale artifact's stored summary records n_prompts=12,
# matching the wikitext baseline it is compared against (which is also 12), so
# the corpus axis stays 12-vs-12. --device cuda is NOT the script default (cpu);
# see the Device note in the header. ~2 min (9.51 s/prompt on-disk x 12).
run readout_c4en_1p5b \
    "${DATA}/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt" \
    -- "${PY}" examples/jspace_readout_scan.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_C4EN}" \
        --n-prompts 12 \
        --out "${DATA}/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt"

# 4/10 — intermediate-concept lens eval. Stale artifact: evals multihop (93
# items / 103 instances) + association (102/102) = the script defaults, with
# --n-items 0 (all). Default --out would land beside the lens in data/cache/.
# ~1 min of scoring (stale per_eval seconds: 30.4 + 35.0) + model load.
run lens_eval_c4en_1p5b \
    "${DATA}/lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt" \
    -- "${PY}" examples/jspace_lens_eval.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_C4EN}" \
        --evals multihop association \
        --out "${DATA}/lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt"

# =============================================================================
# Channel 2 — the WIKITEXT-fit lenses evaluated on the C4-en held-out set.
# --prompts .../heldout_prompts_c4en_n30.json makes heldout_tag() derive
# `_heldoutc4en` (stem -> strip `heldout_prompts_` -> strip `_n30` -> `c4en`).
# No refit involved; the lens stem carries no corpus tag.
# =============================================================================

# 5/10 — 1.5B structure scan (dependency of 6/10). ~7 min.
run structure_heldoutc4en_1p5b \
    "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_structure_scan.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_WT_1P5B}" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --ks 5,10,25,50 \
        --out "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt"

# 6/10 — paper-metric, 1.5B held-out axis. MANIFEST: --n-rand 8
# --rand-seed-base 10000 --prompts <data>/heldout_prompts_c4en_n30.json. ~6 min.
run paper_metric_heldoutc4en_1p5b \
    "${DATA}/paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_paper_metric_varfrac.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_WT_1P5B}" \
        --scan "${DATA}/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --n-rand 8 \
        --rand-seed-base 10000 \
        --out "${DATA}/paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt"

# 7/10 — 1.5B readout scan on the C4 held-out set. Stale summary: n_prompts=30
# (the held-out axis scans the full 30-doc set, unlike the 12-prompt corpus
# axis). ~5 min (9.36 s/prompt on-disk x 30).
run readout_heldoutc4en_1p5b \
    "${DATA}/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_readout_scan.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --mode bf16 \
        --device cuda \
        --lens "${LENS_WT_1P5B}" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --out "${DATA}/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt"

# --- 7B nf4 runs last: heaviest resident model, so the 1.5B work is banked
# --- before the card is put under memory pressure.

# 8/10 — 7B structure scan (dependency of 9/10). ~17 min (24.3 s/prompt x 30
# plus ~5.5 min model load).
run structure_heldoutc4en_7b \
    "${DATA}/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_structure_scan.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --mode nf4 \
        --device cuda \
        --lens "${LENS_WT_7B}" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --ks 5,10,25,50 \
        --out "${DATA}/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt"

# 9/10 — paper-metric, 7B held-out axis. MANIFEST: --rand-seed-base 30000
# (NOT 10000 — the 7B family uses its own seed base). ~18 min.
run paper_metric_heldoutc4en_7b \
    "${DATA}/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_paper_metric_varfrac.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --mode nf4 \
        --device cuda \
        --lens "${LENS_WT_7B}" \
        --scan "${DATA}/structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --n-rand 8 \
        --rand-seed-base 30000 \
        --out "${DATA}/paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt"

# 10/10 — 7B readout scan on the C4 held-out set. Stale summary: n_prompts=30.
# ~11 min (10.21 s/prompt on-disk x 30 plus ~5.5 min model load).
run readout_heldoutc4en_7b \
    "${DATA}/readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt" \
    -- "${PY}" examples/jspace_readout_scan.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --mode nf4 \
        --device cuda \
        --lens "${LENS_WT_7B}" \
        --prompts "${HELDOUT_C4EN}" \
        --n-prompts 30 \
        --out "${DATA}/readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt"

# --- Post-run consistency report --------------------------------------------
# Each paper-metric artifact records `config.validation_max_vf_diff`: the max
# |replicated - stored| varfrac@k-snap against its --scan input. All three
# committed artifacts recorded 0.0 (MANIFEST provenance: "validated bit-exact").
# A non-zero value means the paper-metric run and its structure scan disagree
# and the pair must not be used as a matched set. Reported, not enforced: the
# 10 runs are already done by this point and aborting would discard them.
echo "=== paper-metric scan-validation check (expect 0.0 for all three) ==="
"${PY}" - <<'PYEOF'
from pathlib import Path

import torch

DATA = Path("research/arcs/04_jspace/data")
NAMES = [
    "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt",
    "paper_metric_varfrac_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_heldoutc4en.pt",
    "paper_metric_varfrac_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100_heldoutc4en.pt",
]
bad = []
for name in NAMES:
    cfg = torch.load(DATA / name, map_location="cpu", weights_only=False)["config"]
    diff = cfg["validation_max_vf_diff"]
    flag = "ok" if diff == 0.0 else "MISMATCH"
    print(f"  [{flag}] validation_max_vf_diff={diff!r}  {name}")
    if diff != 0.0:
        bad.append(name)
if bad:
    print(
        "WARNING: paper-metric run(s) did not replicate their structure scan "
        "bit-for-bit; do not treat the pair as a matched set:\n  "
        + "\n  ".join(bad)
    )
PYEOF
echo

echo "=== all ${TOTAL_RUNS} artifacts regenerated ==="
echo "elapsed: $(( $(date +%s) - START_EPOCH ))s"
echo
echo "Next (plan Step 3 — NOT run here):"
echo "  ${PY} examples/jspace_audit_findings.py    # expect FAILs in checks J, K, part of M"
echo "  ${PY} examples/jspace_data_manifest.py     # rewrite the 10 sha256 entries"
