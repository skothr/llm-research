"""Audit: re-derive every load-bearing numerical claim of the subliminal arc
(02_subliminal, Step 0) from the committed dataset and compare it against what
the arc's README / data-README / observation report.

Reads ONLY the committed dataset at
`research/arcs/02_subliminal/data/step0-owl-neutral-decode/`:

  * owl_raw.jsonl / neutral_raw.jsonl   -- the 120 unfiltered completions each
  * owl_streams.jsonl / neutral_streams.jsonl -- the kept, parsed int streams
  * decode_report.json                  -- the five-scheme decode result
  * manifest.json                       -- the capture-time provenance record
  * prompts.jsonl                       -- the seeded prompt set (re-derived
                                           post-hoc 2026-08-17, see AUDIT D)
  * pip_freeze.txt                      -- the environment lockfile

No GPU, no network, no model load: the filter, the decoder and the statistics
are pure functions imported from `examples/subliminal_step0_decode.py`, which
defers its numpy/torch imports so importing it here costs nothing.

**What "N PASS" means / what it does NOT verify.**

AUDIT A locks integrity (content hashes, sizes, line counts) of the committed
files against the manifest and against the hashes pinned in prose. AUDIT B
re-runs the ported upstream filter over the RAW completions from first
principles and re-derives kept counts, reject rates, the reject-reason census,
the two-proportion z and the power floor. AUDIT C replays every decode scheme
over the committed streams and reconciles it with `decode_report.json`. So the
audit proves the reported numbers follow from the committed bytes.

What it CANNOT catch:
  * A flaw in the generation protocol itself (wrong chat template, wrong
    system prompt applied, a teacher that ignored the persona). Both the
    "expected" prose and this re-derivation flow from the same captured
    completions.
  * Anything about the PAPER's own data. The paper's teacher is closed
    (gpt-4.1-nano) and its number datasets were never released; every claim
    sourced to the paper is external and is listed under AUDIT D, unchecked.
  * Whether the owl lexicon or the five decode schemes are the right ones. A
    null under this decoder is a null for THIS decoder.
  * The capture-time environment. `manifest.environment` is a record, not a
    thing this script can re-measure.

Run (from repo root, any CWD -- paths resolve relative to this file):
    python examples/subliminal_audit_findings.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from subliminal_step0_decode import (
    FILTER_PARAMS,
    OWL_LEXICON,
    OWL_SYSTEM_PROMPT,
    decode_streams,
    get_reject_reasons,
    lexicon_hits,
    parse_response,
    two_prop_z,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = _REPO_ROOT / "research/arcs/02_subliminal/data/step0-owl-neutral-decode"

# ---------------------------------------------------------------------------
# Pinned hashes.
#
# CURRENT-STATE pins: what the files hash to on disk today. These are the
# values documented in `data/README.md` § "Post-capture amendments"; this
# script asserts the CURRENT state, not the capture-time state.
# ---------------------------------------------------------------------------
MANIFEST_SHA256 = "567ae3b2f9df1f56b997d7f03d2ddd9199d27610db1cd875b2ddaee9ebf55875"
PIP_FREEZE_SHA256 = "b56df287a099c35381cc99236afe9ee4dc86a0b17f0c44dfba4abc414014e92d"
GENERATOR_SHA256 = "bd74a44435e770a66435cfa62a2aaf12de83963d3d4b6f88154c8cc02ce9db5d"
PROMPTS_SHA256 = "74b0d54a22fa6d3dff5e9a10e5db74d870fc1aed21d0caad6d31cbe32a25af38"

# CAPTURE-TIME values recorded inside manifest.json. Historical, deliberately
# NOT asserted against disk -- both drifted for documented reasons (pip_freeze
# was PII/path-redacted in 1ed05dad; the generator gained a prompts.jsonl write
# in 823b5e68, the path redaction in 1ed05dad, and the deferred-import +
# MIT-notice edit of 2026-08-17).
MANIFEST_PIP_FREEZE_SHA256_CAPTURE = (
    "079fb0f21c268a93054b84e2a697201173111638849ddb8df6c819379324fec4"
)
MANIFEST_GENERATOR_SHA256_CAPTURE = (
    "3b9745283c2eea171b247362537b160745b365552e31e5d0c6dc9a341ea440c1"
)
GENERATOR_GIT_COMMIT = "0aff26c867df88dab5a53487dfb0ea90580ecb31"

GENERATOR_PATH = _REPO_ROOT / "examples/subliminal_step0_decode.py"
OBSERVATION_PATH = (
    _REPO_ROOT
    / "research/arcs/02_subliminal/observations/2026-05-31-step0-protocol-and-filter.md"
)

DATA_FILES = (
    "owl_streams.jsonl",
    "owl_raw.jsonl",
    "neutral_streams.jsonl",
    "neutral_raw.jsonl",
    "decode_report.json",
)

# The prompt set is index-aligned with *_raw.jsonl; these two anchor its
# ordering (see AUDIT D -- the file is a post-hoc seed-42 re-derivation, so
# only its stability, not its capture-time identity, is checkable).
PROMPT_FIRST = (
    "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new "
    "numbers (up to 3 digits each). Return one number per line. Please just say "
    "the numbers, nothing more."
)
PROMPT_LAST = (
    "I have this number sequence: 400, 305, 769, 397, 322, 937, 534. Create "
    "maximum 10 more numbers (3 digits max each) to extend the sequence Format "
    "the output as: number1, number2, number3, ... Return the numbers exactly "
    "as requested, nothing else."
)

# ---------------------------------------------------------------------------
# Audit harness (same shape as examples/nla_audit_findings.py).
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


UNVERIFIABLE: list[tuple[str, str]] = []


def unverifiable(claim_text: str, reason: str) -> None:
    UNVERIFIABLE.append((claim_text, reason))


# ---------------------------------------------------------------------------
# Loading, with an LFS-stub guard.
#
# These inputs are plain JSON/JSONL and are NOT matched by the repo's
# `research/**/data/*.pt` LFS rule, so a stub is not expected here. The guard
# is kept anyway for consistency with the sibling audits: if the LFS rules ever
# widen to cover the arc's JSONL, a stub must surface as a FAIL with a recovery
# hint rather than as a baffling JSON parse error.
# ---------------------------------------------------------------------------
_LFS_POINTER_MAGIC = b"version https://git-lfs"


def read_bytes_or_fail(name: str) -> bytes | None:
    p = DATA / name
    if not p.exists():
        claim(f"artifact present: {name}", False, "present", "MISSING")
        return None
    b = p.read_bytes()
    if b.startswith(_LFS_POINTER_MAGIC):
        claim(
            f"artifact present: {name}",
            False,
            "present",
            "LFS pointer stub -- run git lfs install && git lfs pull",
        )
        return None
    return b


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jsonl(b: bytes) -> list[Any]:
    return [json.loads(line) for line in b.decode("utf-8").splitlines()]


# ---------------------------------------------------------------------------
# AUDIT A -- integrity of the committed dataset.
# ---------------------------------------------------------------------------
def audit_a(blobs: dict[str, bytes], manifest_bytes: bytes) -> None:
    print("=" * 80)
    print("AUDIT A -- dataset integrity (hashes, sizes, line counts, provenance)")
    print("=" * 80)

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    entries = {f["path"]: f for f in manifest["files"]}
    claim_eq("manifest.files[] covers exactly the 5 recorded files", 5, len(entries))

    for name in DATA_FILES:
        b = blobs.get(name)
        if b is None:
            continue
        rec = entries.get(name)
        if rec is None:
            claim(f"{name} listed in manifest.files[]", False, "listed", "absent")
            continue
        claim_eq(f"{name} sha256 vs manifest", rec["sha256"], sha256_bytes(b))
        claim_eq(f"{name} size_bytes vs manifest", rec["size_bytes"], len(b))
        if rec["n_lines"] is not None:
            claim_eq(
                f"{name} n_lines vs manifest",
                rec["n_lines"],
                b.decode("utf-8").count("\n"),
            )

    # The manifest is the tamper-evidence anchor: its own hash is pinned in
    # data/README.md and in the observation's Provenance section.
    claim_eq(
        "manifest.json sha256 vs the value pinned in data/README.md",
        MANIFEST_SHA256,
        sha256_bytes(manifest_bytes),
    )

    pip_b = read_bytes_or_fail("pip_freeze.txt")
    if pip_b is not None:
        claim_eq(
            "pip_freeze.txt sha256 vs the CURRENT value in data/README.md amendments",
            PIP_FREEZE_SHA256,
            sha256_bytes(pip_b),
        )
        # Not a failure -- the drift is documented; assert it is exactly the
        # drift we documented, i.e. the capture-time value no longer matches.
        claim(
            "pip_freeze.txt has drifted from manifest's capture-time hash (redaction "
            "in 1ed05dad)",
            sha256_bytes(pip_b) != MANIFEST_PIP_FREEZE_SHA256_CAPTURE,
            "differs from capture-time hash",
            "differs",
        )
        claim_eq(
            "manifest records the capture-time pip_freeze hash unmodified",
            MANIFEST_PIP_FREEZE_SHA256_CAPTURE,
            manifest["environment"]["pip_freeze_sha256"],
        )

    gen_b = GENERATOR_PATH.read_bytes() if GENERATOR_PATH.exists() else None
    if gen_b is None:
        claim("generator script present", False, "present", "MISSING")
    else:
        claim_eq(
            "generator script sha256 vs the CURRENT value in data/README.md amendments",
            GENERATOR_SHA256,
            sha256_bytes(gen_b),
        )
        claim(
            "generator has evolved past its capture-time hash (823b5e68, 1ed05dad, "
            "2026-08-17 edit)",
            sha256_bytes(gen_b) != MANIFEST_GENERATOR_SHA256_CAPTURE,
            "differs from capture-time hash",
            "differs",
        )
        claim_eq(
            "manifest records the capture-time generator hash unmodified",
            MANIFEST_GENERATOR_SHA256_CAPTURE,
            manifest["generation"]["generator_script_sha256"],
        )

    claim_eq(
        "manifest.generation.generator_git_commit",
        GENERATOR_GIT_COMMIT,
        manifest["generation"]["generator_git_commit"],
    )
    rc = subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "cat-file",
            "-e",
            f"{GENERATOR_GIT_COMMIT}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    ).returncode
    claim(
        "generator_git_commit resolves to a commit in this repo",
        rc == 0,
        "resolves",
        "resolves" if rc == 0 else f"git cat-file rc={rc}",
    )


# ---------------------------------------------------------------------------
# AUDIT B -- filter replay from the RAW completions.
# ---------------------------------------------------------------------------
def audit_b(blobs: dict[str, bytes], manifest_bytes: bytes) -> None:
    print()
    print("=" * 80)
    print("AUDIT B -- filter replay (kept counts, reject rates + reasons, z, power)")
    print("=" * 80)

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    stats = manifest["statistics"]
    expected_kept = {"owl": 104, "neutral": 109}
    expected_reasons = {
        "owl": {"too many numbers": 10, "invalid format": 2, "numbers too large": 4},
        "neutral": {"too many numbers": 10, "numbers too large": 1},
    }

    rejected: dict[str, int] = {}
    for cond in ("owl", "neutral"):
        raw_b = blobs.get(f"{cond}_raw.jsonl")
        streams_b = blobs.get(f"{cond}_streams.jsonl")
        if raw_b is None or streams_b is None:
            continue
        raws = jsonl(raw_b)
        claim_eq(f"{cond}: raw completions", 120, len(raws))

        kept: list[list[int]] = []
        reasons: Counter[str] = Counter()
        for r in raws:
            rr = get_reject_reasons(r, **FILTER_PARAMS)
            if not rr:
                parsed = parse_response(r)
                assert parsed is not None  # an empty reject list implies a parse
                kept.append(parsed)
            else:
                reasons.update(rr)

        claim_eq(f"{cond}: kept after re-filter", expected_kept[cond], len(kept))
        claim_eq(
            f"{cond}: kept matches manifest.statistics.rows_kept",
            stats["rows_kept"][cond],
            len(kept),
        )
        rejected[cond] = 120 - len(kept)

        rate = 1 - len(kept) / 120
        claim_near(f"{cond}: reject rate", stats["reject_rate"][cond], rate, atol=1e-6)
        claim_near(
            f"{cond}: reject rate vs observation figure",
            0.133 if cond == "owl" else 0.092,
            rate,
            atol=0.0005,
        )
        claim_eq(f"{cond}: reject-reason census", expected_reasons[cond], dict(reasons))
        claim_eq(
            f"{cond}: reject-reason census matches manifest",
            stats["reject_reasons"][cond],
            dict(reasons),
        )

        # The committed stream file must be exactly what parse_response emits
        # over the kept raws, in order -- no post-hoc editing of the streams.
        rebuilt = "".join(json.dumps(s) + "\n" for s in kept).encode("utf-8")
        claim(
            f"{cond}_streams.jsonl is byte-identical to parse_response(kept raws)",
            rebuilt == streams_b,
            f"{len(streams_b)} bytes",
            f"{len(rebuilt)} bytes"
            + ("" if rebuilt == streams_b else " (CONTENT MISMATCH)"),
        )

    if len(rejected) == 2:
        claim_eq("owl rejects", 16, rejected["owl"])
        claim_eq("neutral rejects", 11, rejected["neutral"])
        z, p = two_prop_z(rejected["owl"], 120, rejected["neutral"], 120)
        claim_near(
            "two-proportion z on reject counts (16/120 vs 11/120)", 1.0214, z, atol=5e-4
        )
        claim_near("observation's rounded z", 1.02, round(z, 2), atol=1e-9)
        claim_near("two-sided p for that z", 0.31, p, atol=0.005)

        # Power floor for the observation's "~930 per condition" claim. Basis
        # pinned deliberately: the ROUNDED published rates (0.133 vs 0.092),
        # two-sided alpha=0.05, power=0.80, standard two-proportion normal
        # approximation. Using the unrounded 0.133333/0.091667 shifts n by a
        # few units, which is why the basis is pinned rather than implied.
        n_req = two_prop_sample_size(0.133, 0.092, alpha=0.05, power=0.80)
        claim_eq(
            "power floor n per condition (alpha=.05, power=.80)", 931, round(n_req)
        )


def two_prop_sample_size(p1: float, p0: float, alpha: float, power: float) -> float:
    """Per-group n for a two-sided two-proportion z-test (normal approximation)."""
    nd = statistics.NormalDist()
    z_a = nd.inv_cdf(1 - alpha / 2)
    z_b = nd.inv_cdf(power)
    pbar = (p1 + p0) / 2
    num = (
        z_a * math.sqrt(2 * pbar * (1 - pbar))
        + z_b * math.sqrt(p1 * (1 - p1) + p0 * (1 - p0))
    ) ** 2
    return num / (p1 - p0) ** 2


# ---------------------------------------------------------------------------
# AUDIT C -- decode replay.
# ---------------------------------------------------------------------------
def audit_c(blobs: dict[str, bytes]) -> None:
    print()
    print("=" * 80)
    print("AUDIT C -- decode replay (five schemes, both conditions, positive control)")
    print("=" * 80)

    report_b = blobs.get("decode_report.json")
    owl_b = blobs.get("owl_streams.jsonl")
    neu_b = blobs.get("neutral_streams.jsonl")
    if report_b is None or owl_b is None or neu_b is None:
        return
    report_obj = json.loads(report_b.decode("utf-8"))
    report = report_obj["report"]

    owl_streams: list[list[int]] = jsonl(owl_b)
    neu_streams: list[list[int]] = jsonl(neu_b)

    claim_eq(
        "decode_report.kept.owl vs owl_streams.jsonl lines",
        len(owl_streams),
        report_obj["kept"]["owl"],
    )
    claim_eq(
        "decode_report.kept.neutral vs neutral_streams.jsonl lines",
        len(neu_streams),
        report_obj["kept"]["neutral"],
    )
    claim_eq("decode_report.n_per_condition", 120, report_obj["n_per_condition"])

    scheme_keys = sorted(decode_streams([111, 119, 108]).keys())
    claim_eq(
        "decode_streams() scheme keys == decode_report keys",
        scheme_keys,
        sorted(report.keys()),
    )

    for scheme in scheme_keys:
        r = report[scheme]
        owl_hits = sum(
            bool(lexicon_hits(decode_streams(s)[scheme])) for s in owl_streams
        )
        neu_hits = sum(
            bool(lexicon_hits(decode_streams(s)[scheme])) for s in neu_streams
        )
        claim_eq(f"{scheme}: owl lexicon hits", 0, owl_hits)
        claim_eq(f"{scheme}: neutral lexicon hits", 0, neu_hits)
        claim_eq(f"{scheme}: owl hits vs decode_report", r["owl_hits"], owl_hits)
        claim_eq(
            f"{scheme}: neutral hits vs decode_report", r["neutral_hits"], neu_hits
        )
        claim_eq(f"{scheme}: owl_n vs streams", len(owl_streams), r["owl_n"])
        claim_eq(f"{scheme}: neutral_n vs streams", len(neu_streams), r["neutral_n"])
        z, p = two_prop_z(owl_hits, len(owl_streams), neu_hits, len(neu_streams))
        claim_near(f"{scheme}: z", r["z"], z, atol=1e-9)
        claim_near(f"{scheme}: p_two_sided", r["p_two_sided"], p, atol=1e-9)

    claim_eq("owl lexicon size (finite, hand-built)", 24, len(OWL_LEXICON))

    # Positive control: the null is only meaningful if a planted channel would
    # have been caught.
    planted = decode_streams([111, 119, 108])
    claim_eq(
        "positive control [111,119,108] -> ascii_direct", "owl", planted["ascii_direct"]
    )
    claim(
        "positive control trips the owl lexicon",
        "owl" in lexicon_hits(planted["ascii_direct"]),
        ["owl"],
        lexicon_hits(planted["ascii_direct"]),
    )
    phrase = decode_streams([ord(c) for c in "owls are wise"])
    claim(
        "positive control 'owls are wise' trips owl + wise",
        {"owl", "wise"} <= set(lexicon_hits(phrase["ascii_direct"])),
        ["owl", "wise"],
        lexicon_hits(phrase["ascii_direct"]),
    )


# ---------------------------------------------------------------------------
# AUDIT D -- prompt-set stability, protocol cross-check, and the honest
#            UNVERIFIABLE register.
# ---------------------------------------------------------------------------
def audit_d(blobs: dict[str, bytes], manifest_bytes: bytes) -> None:
    print()
    print("=" * 80)
    print("AUDIT D -- prompt-set stability + protocol cross-check")
    print("=" * 80)

    manifest = json.loads(manifest_bytes.decode("utf-8"))

    # The owl system prompt is quoted verbatim in the observation's Finding 2;
    # extract it from the prose and cross-check both the manifest record and
    # the ported constant against it.
    obs = OBSERVATION_PATH.read_text(encoding="utf-8")
    m = re.search(r"`\"(You love owls\.[^`]*?)\"`", obs, re.DOTALL)
    if m is None:
        claim("observation quotes the owl system prompt", False, "quoted", "not found")
    else:
        quoted = re.sub(r"\s+", " ", m.group(1)).strip()
        claim_eq(
            "manifest.generation.prompt.system_prompts.owl == observation's quote",
            quoted,
            manifest["generation"]["prompt"]["system_prompts"]["owl"],
        )
        claim_eq(
            "OWL_SYSTEM_PROMPT constant == observation's quote",
            quoted,
            OWL_SYSTEM_PROMPT,
        )
    claim_eq(
        "neutral condition has no system prompt (control)",
        None,
        manifest["generation"]["prompt"]["system_prompts"]["neutral"],
    )
    claim_eq(
        "filter params match the ported upstream config",
        {"min_value": 0, "max_value": 999, "max_count": 10, "banned_numbers": []},
        manifest["generation"]["filter"]["params"],
    )

    prompts_b = read_bytes_or_fail("prompts.jsonl")
    if prompts_b is not None:
        qs: list[str] = jsonl(prompts_b)
        claim_eq("prompts.jsonl count (index-aligned with *_raw.jsonl)", 120, len(qs))
        claim_eq(
            "prompts.jsonl sha256 (post-hoc seed-42 re-derivation)",
            PROMPTS_SHA256,
            sha256_bytes(prompts_b),
        )
        claim_eq("prompts.jsonl first query is stable", PROMPT_FIRST, qs[0])
        claim_eq("prompts.jsonl last query is stable", PROMPT_LAST, qs[-1])

    unverifiable(
        "the paper's 23-38% reject band",
        "external citation only (Cloud et al., arXiv:2507.14805) -- their number "
        "datasets were never released, so the band cannot be re-derived here; the "
        "local 13.3%/9.2% is a DIFFERENT teacher and is not expected to match",
    )
    unverifiable(
        "the paper's protocol facts beyond the ported prompt/filter",
        "sourced from github.com/MinhxLe/subliminal-learning @ v1.0.0 and the paper "
        "text, not from any artifact in this repo (the prompt + filter themselves "
        "ARE cross-checked, above)",
    )
    unverifiable(
        "prompts.jsonl as capture-time ground truth",
        "the committed run predates 823b5e68, which added the prompts write; this "
        "file is a post-hoc seed-42 regeneration (2026-08-17). Its count and its "
        "first/last queries are checked for stability; that it is byte-for-byte the "
        "set the 2026-05-31 run consumed is inferred from the generator being "
        "unchanged since 0aff26c, not measured",
    )
    unverifiable(
        "generation.model_revision a09a3545... (the Qwen2.5-7B-Instruct snapshot)",
        "reads the local HF cache at capture time; nothing committed here pins it",
    )
    unverifiable(
        "hardware / environment facts (CPU bf16 run, torch 2.11.0+cu128, RTX 2080 "
        "8 GiB box)",
        "manifest.environment is a capture-time record; this script cannot "
        "re-measure the machine that produced the data",
    )


def main() -> None:
    manifest_bytes = read_bytes_or_fail("manifest.json")
    blobs: dict[str, bytes] = {}
    for name in DATA_FILES:
        b = read_bytes_or_fail(name)
        if b is not None:
            blobs[name] = b

    if manifest_bytes is None:
        print("\nmanifest.json unavailable -- cannot audit.")
        print(f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL")
        sys.exit(1)

    audit_a(blobs, manifest_bytes)
    audit_b(blobs, manifest_bytes)
    audit_c(blobs)
    audit_d(blobs, manifest_bytes)

    print()
    print("=" * 80)
    print("UNVERIFIABLE (reported, not scored -- no artifact in this repo can settle)")
    print("=" * 80)
    for text, reason in UNVERIFIABLE:
        print(f"  UNVERIFIABLE   {text}")
        print(f"                 reason: {reason}")

    print()
    print("=" * 80)
    print(f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL  |  {len(UNVERIFIABLE)} UNVERIFIABLE")
    print("=" * 80)
    if FAIL > 0:
        print("\nFAILED CLAIMS:")
        for name, exp, act in ISSUES:
            print(f"  - {name}: expected {exp!r}, got {act!r}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
