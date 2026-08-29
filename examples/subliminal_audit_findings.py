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
over the committed streams and reconciles it with `decode_report.json`.
AUDIT D cross-checks the protocol constants and regenerates the whole 120-query
prompt set from the ported `PromptGenerator` at seed 42, comparing bytes. So
the audit proves the reported numbers follow from the committed bytes.

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
    PROMPT_PARAMS,
    PromptGenerator,
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
MANIFEST_SHA256 = "6468c7a351f754333b46a11a15c94f31f9aa7317fc5e5ad4437eae89304e41da"
PIP_FREEZE_SHA256 = "b56df287a099c35381cc99236afe9ee4dc86a0b17f0c44dfba4abc414014e92d"
GENERATOR_SHA256 = "5edcdf2b4cac3ee00476bb0c4cd1bf07b5b89dce766295fac1be2b139fefe9de"
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
# The generating commit, under both of its SHAs. The dataset was captured in a
# pre-split monorepo whose history the 2026-06-01 split rewrote; the capture-time
# SHA is reachable from no ref here, so the manifest records the rewritten
# equivalent (reachable from main) and keeps the original alongside it. Same
# subject, same author date; the rewritten commit's own message carries
# "(cherry picked from commit 0aff26c8...)".
GENERATOR_GIT_COMMIT = "d9c7a42812cada15da17d62d3bcc31472602846f"
GENERATOR_GIT_COMMIT_PRE_REWRITE = "0aff26c867df88dab5a53487dfb0ea90580ecb31"
# `timestamps.repo_git_commit` records the working tree's HEAD at capture time,
# under the same two SHAs and repointed by the same amendment. It coincides with
# the generator's pair because the generator was committed in the run's own
# checkout; the claim in AUDIT A asserts that the manifest still says so.
REPO_GIT_COMMIT = GENERATOR_GIT_COMMIT
REPO_GIT_COMMIT_PRE_REWRITE = GENERATOR_GIT_COMMIT_PRE_REWRITE

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

# The prompt set is index-aligned with *_raw.jsonl (one query per completion,
# hence the same 120), and these two anchor its ordering (see AUDIT D -- the
# file is a post-hoc seed-42 re-derivation, so only its stability, not its
# capture-time identity, is checkable).
PROMPT_COUNT = 120
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


def _git(*args: str) -> tuple[int, str] | None:
    """Run a git command in the repo root. None when git itself is unusable
    (not installed, not executable) — the caller decides whether that is an
    environment gap or a real result."""
    try:
        p = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return p.returncode, p.stdout.strip()


def git_commit_exists(commit: str) -> tuple[bool | None, str]:
    """Does `commit` exist in this checkout's object store?

    Returns `(True|False|None, why)`. `None` means the question cannot be
    answered HERE — no git binary, not a git checkout (a source tarball or a
    vendored copy), or a shallow/partial clone whose history was truncated.
    Those are environment gaps and belong on the UNVERIFIABLE list; only a
    full checkout that genuinely lacks the object is a FAIL.
    """
    probe = _git("rev-parse", "--git-dir")
    if probe is None:
        return None, "no usable `git` on PATH"
    if probe[0] != 0:
        return None, "not a git checkout (source copy without .git)"

    r = _git("cat-file", "-e", f"{commit}^{{commit}}")
    if r is None:
        return None, "no usable `git` on PATH"
    if r[0] == 0:
        return True, "resolves"

    # `--git-dir` above succeeds for ANY enclosing repository, so a copy of this
    # tree vendored into an unrelated project answers as if it were a checkout of
    # THIS repo and then scores the (inevitably) missing object as data drift.
    # Compare the enclosing repo's top level against `_REPO_ROOT` to tell the two
    # apart. They coincide for a normal clone, for a linked worktree (where
    # `.git` is a file, not a directory) and for a shallow clone, so only a
    # genuine vendored/nested copy is diverted onto the UNVERIFIABLE path.
    top = _git("rev-parse", "--show-toplevel")
    if top is None or top[0] != 0:
        return None, "cannot resolve the enclosing repository's top level"
    if Path(top[1]).resolve() != _REPO_ROOT:
        return (
            None,
            "this tree is not the root of the enclosing git repo "
            f"({top[1]}) — a vendored or nested copy, whose object store "
            "is not this repo's",
        )

    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow is None or shallow[0] != 0 or shallow[1] == "true":
        return None, "shallow clone — history truncated, the object may be upstream"
    return False, f"git cat-file rc={r[0]}"


# ---------------------------------------------------------------------------
# Loading, with an LFS-stub guard and a malformed-content guard.
#
# These inputs are plain JSON/JSONL and are NOT matched by the repo's
# `research/**/data/*.pt` LFS rule, so a stub is not expected here. The guard
# is kept anyway for consistency with the sibling audits: if the LFS rules ever
# widen to cover the arc's JSONL, a stub must surface as a FAIL with a recovery
# hint rather than as a baffling JSON parse error.
#
# Absence and stub-ness are only two of the ways a committed artifact can be
# wrong. Truncation, a bad merge, a corrupted checkout or a partial write all
# leave a file that is present, is not a stub, and still does not parse — and
# an unguarded `json.loads` turns that into the same baffling traceback, before
# the SUMMARY line. So every parse in this file goes through `json_or_fail` /
# `jsonl_or_fail`, which score one FAIL row and return None.
#
# Malformed content scores FAIL, not UNVERIFIABLE: the committed bytes are the
# thing this audit measures, so bytes that no longer parse ARE data drift. The
# UNVERIFIABLE register is reserved for environment gaps (no git binary, no
# numpy, a shallow clone) that say nothing about the data.
# ---------------------------------------------------------------------------
_LFS_POINTER_MAGIC = b"version https://git-lfs"


def read_path_or_fail(p: Path, label: str) -> bytes | None:
    """Read a tracked repo artifact, scoring an absent or unreadable file as a
    single FAIL row instead of raising. Every input this audit consumes goes
    through here (or through a wrapper of it) so that one missing file costs one
    claim, not the rest of the run and the SUMMARY line."""
    if not p.exists():
        claim(f"artifact present: {label}", False, "present", "MISSING")
        return None
    try:
        return p.read_bytes()
    except OSError as exc:
        claim(
            f"artifact present: {label}",
            False,
            "present",
            f"UNREADABLE ({type(exc).__name__})",
        )
        return None


def read_text_or_fail(p: Path, label: str) -> str | None:
    """`read_path_or_fail` plus a guarded UTF-8 decode."""
    b = read_path_or_fail(p, label)
    if b is None:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        claim(f"artifact present: {label}", False, "present", "UNREADABLE (not UTF-8)")
        return None


def read_bytes_or_fail(name: str) -> bytes | None:
    b = read_path_or_fail(DATA / name, name)
    if b is None:
        return None
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


def _decode_or_fail(b: bytes, label: str) -> str | None:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError as exc:
        claim(
            f"artifact parses: {label}",
            False,
            "valid UTF-8",
            f"MALFORMED (UnicodeDecodeError: {exc})",
        )
        return None


def json_or_fail(b: bytes, label: str) -> Any | None:
    """Parse a committed JSON artifact, scoring malformed content as a single
    FAIL row instead of raising. Returns None on failure so the caller can skip
    the claims that depend on it and the run still reaches its SUMMARY."""
    text = _decode_or_fail(b, label)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        claim(
            f"artifact parses: {label}",
            False,
            "valid JSON",
            f"MALFORMED (JSONDecodeError: {exc})",
        )
        return None


def jsonl_or_fail(b: bytes, label: str) -> list[Any] | None:
    """`json_or_fail` for a one-object-per-line file. A single bad line fails
    the whole artifact — these files are index-aligned, so a partial parse would
    silently shift every downstream count."""
    text = _decode_or_fail(b, label)
    if text is None:
        return None
    out: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            claim(
                f"artifact parses: {label}",
                False,
                "valid JSONL",
                f"MALFORMED at line {lineno} (JSONDecodeError: {exc})",
            )
            return None
    return out


def dig(obj: Any, *keys: str) -> Any:
    """Walk a nested manifest path, returning None instead of raising when any
    key along the way is absent. A field quietly dropped by a later edit is
    exactly what the claims below exist to catch, so it has to render as a FAIL
    row, not as a KeyError that kills the run before its SUMMARY."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


# ---------------------------------------------------------------------------
# AUDIT A -- integrity of the committed dataset.
# ---------------------------------------------------------------------------
def audit_a(
    blobs: dict[str, bytes], manifest_bytes: bytes, manifest: dict[str, Any]
) -> None:
    print("=" * 80)
    print("AUDIT A -- dataset integrity (hashes, sizes, line counts, provenance)")
    print("=" * 80)

    files_list = dig(manifest, "files")
    entries = (
        {f["path"]: f for f in files_list if isinstance(f, dict) and "path" in f}
        if isinstance(files_list, list)
        else {}
    )
    claim_eq("manifest.files[] covers exactly the 5 recorded files", 5, len(entries))

    for name in DATA_FILES:
        b = blobs.get(name)
        if b is None:
            continue
        rec = entries.get(name)
        if rec is None:
            claim(f"{name} listed in manifest.files[]", False, "listed", "absent")
            continue
        claim_eq(f"{name} sha256 vs manifest", dig(rec, "sha256"), sha256_bytes(b))
        claim_eq(f"{name} size_bytes vs manifest", dig(rec, "size_bytes"), len(b))
        if dig(rec, "n_lines") is not None:
            claim_eq(
                f"{name} n_lines vs manifest",
                dig(rec, "n_lines"),
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
        # `actual` is derived from the same comparison that decides the verdict:
        # a hardcoded "differs" would make the FAIL row assert the opposite of
        # what was measured.
        pip_drifted = sha256_bytes(pip_b) != MANIFEST_PIP_FREEZE_SHA256_CAPTURE
        claim(
            "pip_freeze.txt has drifted from manifest's capture-time hash (redaction "
            "in 1ed05dad)",
            pip_drifted,
            "differs from capture-time hash",
            "differs from capture-time hash"
            if pip_drifted
            else f"MATCHES capture-time hash {MANIFEST_PIP_FREEZE_SHA256_CAPTURE[:8]}"
            " -- no drift",
        )
        claim_eq(
            "manifest records the capture-time pip_freeze hash unmodified",
            MANIFEST_PIP_FREEZE_SHA256_CAPTURE,
            dig(manifest, "environment", "pip_freeze_sha256"),
        )

    gen_b = read_path_or_fail(GENERATOR_PATH, "examples/subliminal_step0_decode.py")
    if gen_b is not None:
        claim_eq(
            "generator script sha256 vs the CURRENT value in data/README.md amendments",
            GENERATOR_SHA256,
            sha256_bytes(gen_b),
        )
        gen_drifted = sha256_bytes(gen_b) != MANIFEST_GENERATOR_SHA256_CAPTURE
        claim(
            "generator has evolved past its capture-time hash (823b5e68, 1ed05dad, "
            "two 2026-08-17 edits, two 2026-08-19 edits)",
            gen_drifted,
            "differs from capture-time hash",
            "differs from capture-time hash"
            if gen_drifted
            else f"MATCHES capture-time hash {MANIFEST_GENERATOR_SHA256_CAPTURE[:8]}"
            " -- no drift",
        )
        claim_eq(
            "manifest records the capture-time generator hash unmodified",
            MANIFEST_GENERATOR_SHA256_CAPTURE,
            dig(manifest, "generation", "generator_script_sha256"),
        )

    # One claim, both SHAs: the resolvable post-rewrite pointer AND the
    # capture-time original it stands in for. Checking them together keeps the
    # capture-time record from being quietly dropped in a later edit -- so both
    # halves are read through `dig`, because a bare index on the half this claim
    # exists to protect would crash on precisely the case it is watching for.
    claim_eq(
        "manifest.generation.generator_git_commit (+ _pre_rewrite sibling)",
        f"{GENERATOR_GIT_COMMIT} / {GENERATOR_GIT_COMMIT_PRE_REWRITE}",
        f"{dig(manifest, 'generation', 'generator_git_commit')} / "
        f"{dig(manifest, 'generation', 'generator_git_commit_pre_rewrite')}",
    )
    # The same pair, recorded a second time for the capture-time REPO head under
    # `timestamps`. Both fields were repointed by the same 2026-08-19 amendment
    # that repointed the generator pair, for the same reason (the pre-split SHA
    # is reachable from no ref here), and both are the same commit -- the
    # generator was committed in the run's own checkout. Without this claim the
    # clean-clone defect that repoint fixed could recur on the `timestamps` side
    # and the audit would still report 0 FAIL.
    claim_eq(
        "manifest.timestamps.repo_git_commit (+ _pre_rewrite sibling)",
        f"{REPO_GIT_COMMIT} / {REPO_GIT_COMMIT_PRE_REWRITE}",
        f"{dig(manifest, 'timestamps', 'repo_git_commit')} / "
        f"{dig(manifest, 'timestamps', 'repo_git_commit_pre_rewrite')}",
    )
    resolved, why = git_commit_exists(GENERATOR_GIT_COMMIT)
    if resolved is None:
        # No git, no checkout, or a shallow/partial clone: the commit's absence
        # says nothing about the manifest, so this is an environment gap, not a
        # failing claim. Scoring it FAIL would make `pip download`-style
        # source copies and CI shallow clones look like data drift.
        unverifiable(
            "generator_git_commit resolves to a commit in this repo",
            f"cannot be checked here: {why}",
        )
    else:
        claim(
            "generator_git_commit resolves to a commit in this repo",
            resolved,
            "resolves",
            "resolves" if resolved else why,
        )


# ---------------------------------------------------------------------------
# AUDIT B -- filter replay from the RAW completions.
# ---------------------------------------------------------------------------
def audit_b(blobs: dict[str, bytes], manifest: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print("AUDIT B -- filter replay (kept counts, reject rates + reasons, z, power)")
    print("=" * 80)

    stats = dig(manifest, "statistics")
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
        raws = jsonl_or_fail(raw_b, f"{cond}_raw.jsonl")
        if raws is None:
            continue
        claim_eq(f"{cond}: raw completions", 120, len(raws))

        kept: list[list[int]] = []
        reasons: Counter[str] = Counter()
        for r in raws:
            rr = get_reject_reasons(r, **FILTER_PARAMS)
            if not rr:
                parsed = parse_response(r)
                if parsed is None:
                    # An empty reject list is supposed to imply a successful
                    # parse -- the two are meant to be the same predicate. A
                    # disagreement is a real defect in the ported filter, so
                    # score it and keep going. (This used to be a bare `assert`,
                    # which killed the run before its SUMMARY, and vanished
                    # entirely under `python -O`, appending None to `kept`.)
                    claim(
                        f"{cond}: empty reject list implies parse_response succeeds",
                        False,
                        "parsed int stream",
                        "parse_response returned None despite no reject reason",
                    )
                    continue
                kept.append(parsed)
            else:
                reasons.update(rr)

        claim_eq(f"{cond}: kept after re-filter", expected_kept[cond], len(kept))
        claim_eq(
            f"{cond}: kept matches manifest.statistics.rows_kept",
            dig(stats, "rows_kept", cond),
            len(kept),
        )
        rejected[cond] = 120 - len(kept)

        rate = 1 - len(kept) / 120
        manifest_rate = dig(stats, "reject_rate", cond)
        if isinstance(manifest_rate, (int, float)):
            claim_near(f"{cond}: reject rate", manifest_rate, rate, atol=1e-6)
        else:
            claim(
                f"{cond}: reject rate",
                False,
                "a recorded manifest rate",
                f"missing from manifest.statistics ({manifest_rate!r})",
            )
        claim_near(
            f"{cond}: reject rate vs observation figure",
            0.133 if cond == "owl" else 0.092,
            rate,
            atol=0.0005,
        )
        claim_eq(f"{cond}: reject-reason census", expected_reasons[cond], dict(reasons))
        claim_eq(
            f"{cond}: reject-reason census matches manifest",
            dig(stats, "reject_reasons", cond),
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

        # Power floor for the prose's "~931 per condition" claim. Basis
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
    report_obj = json_or_fail(report_b, "decode_report.json")
    owl_parsed = jsonl_or_fail(owl_b, "owl_streams.jsonl")
    neu_parsed = jsonl_or_fail(neu_b, "neutral_streams.jsonl")
    if report_obj is None or owl_parsed is None or neu_parsed is None:
        return
    report = report_obj["report"]

    owl_streams: list[list[int]] = owl_parsed
    neu_streams: list[list[int]] = neu_parsed

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
        # These two pin what decode_report.json RECORDED, nothing more. With 0
        # hits in both arms the pooled SE is 0 and the two-proportion test is
        # undefined; two_prop_z returns (0.0, 1.0) by convention for that case,
        # and the arc's prose says so explicitly. Do not read a significance
        # claim into a PASS here — the evidence is the zero-hit count.
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
def audit_d(manifest: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print("AUDIT D -- prompt-set stability + protocol cross-check")
    print("=" * 80)

    # The owl system prompt is quoted verbatim in the observation's Finding 2;
    # extract it from the prose and cross-check both the manifest record and
    # the ported constant against it.
    obs = read_text_or_fail(
        OBSERVATION_PATH,
        "research/arcs/02_subliminal/observations/"
        "2026-05-31-step0-protocol-and-filter.md",
    )
    if obs is not None:
        m = re.search(r"`\"(You love owls\.[^`]*?)\"`", obs, re.DOTALL)
        if m is None:
            claim(
                "observation quotes the owl system prompt", False, "quoted", "not found"
            )
        else:
            quoted = re.sub(r"\s+", " ", m.group(1)).strip()
            claim_eq(
                "manifest.generation.prompt.system_prompts.owl == observation's quote",
                quoted,
                dig(manifest, "generation", "prompt", "system_prompts", "owl"),
            )
            claim_eq(
                "OWL_SYSTEM_PROMPT constant == observation's quote",
                quoted,
                OWL_SYSTEM_PROMPT,
            )
    claim_eq(
        "neutral condition has no system prompt (control)",
        None,
        dig(manifest, "generation", "prompt", "system_prompts", "neutral"),
    )
    claim_eq(
        "filter params match the ported upstream config",
        {"min_value": 0, "max_value": 999, "max_count": 10, "banned_numbers": []},
        dig(manifest, "generation", "filter", "params"),
    )

    prompts_b = read_bytes_or_fail("prompts.jsonl")
    if prompts_b is not None:
        # The hash claim is deliberately outside the parse guard below: it reads
        # bytes, not structure, so it stays scoreable even when the file no
        # longer parses -- which is exactly when you most want to see it.
        claim_eq(
            "prompts.jsonl sha256 (post-hoc seed-42 re-derivation)",
            PROMPTS_SHA256,
            sha256_bytes(prompts_b),
        )
    parsed_prompts = (
        None if prompts_b is None else jsonl_or_fail(prompts_b, "prompts.jsonl")
    )
    if prompts_b is not None and parsed_prompts is not None:
        qs: list[str] = parsed_prompts
        claim_eq(
            "prompts.jsonl count (index-aligned with *_raw.jsonl)",
            PROMPT_COUNT,
            len(qs),
        )
        # Indexing is guarded rather than gated on the count claim above: that
        # claim RECORDS a short file, it does not stop this one from indexing
        # into it. A truncated or empty prompts.jsonl has to score two FAIL rows
        # here, not an IndexError that costs the SUMMARY line.
        claim_eq(
            "prompts.jsonl first query is stable",
            PROMPT_FIRST,
            qs[0] if qs else "<empty file -- no first query>",
        )
        claim_eq(
            "prompts.jsonl last query is stable",
            PROMPT_LAST,
            qs[-1] if qs else "<empty file -- no last query>",
        )

        # Full determinism replay: rebuild all 120 queries from the ported
        # PromptGenerator under the run's seed (42) and serialize them exactly
        # as subliminal_step0_decode.main() does, then compare BYTES. This
        # upgrades the two spot-checks above (first/last query) into a
        # whole-file re-derivation — the committed prompt set is a function of
        # (generator code, seed), not an opaque blob. numpy is imported here,
        # not at module scope, to keep this audit's import cheap -- and guarded,
        # because an absent numpy is an environment gap (nothing about the data
        # drifted), so it belongs on the UNVERIFIABLE register rather than
        # aborting the remaining checks and the SUMMARY line.
        try:
            import numpy as np
        except ImportError as exc:
            unverifiable(
                "prompts.jsonl replays byte-identical from PromptGenerator @ seed 42",
                f"numpy is unavailable here ({exc}) and the generator's RNG needs "
                "it; install `.[dev]` and re-run to score this claim",
            )
        else:
            # Replay a fixed PROMPT_COUNT, not `len(qs)`: driving the replay off
            # the file's own length would make an empty prompts.jsonl replay to
            # zero bytes and PASS this claim while the file is plainly wrong.
            pg = PromptGenerator(rng=np.random.default_rng(42), **PROMPT_PARAMS)
            replayed = "".join(
                json.dumps(pg.sample_query()) + "\n" for _ in range(PROMPT_COUNT)
            ).encode("utf-8")
            claim(
                "prompts.jsonl replays byte-identical from PromptGenerator @ seed 42",
                replayed == prompts_b,
                f"{len(prompts_b)} bytes, sha256 {PROMPTS_SHA256[:8]}",
                f"{len(replayed)} bytes, sha256 {sha256_bytes(replayed)[:8]}",
            )

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
        "file is a post-hoc seed-42 regeneration (2026-08-17). AUDIT D replays it "
        "byte-for-byte from the generator, so it IS this generator's seed-42 "
        "output; that it is byte-for-byte the set the 2026-05-31 run consumed is "
        "inferred from the generator being unchanged since d9c7a42 (the capture-time "
        "commit 0aff26c, as rewritten by the 2026-06-01 monorepo split), not measured",
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

    # Parsed once, here, so a corrupt manifest costs ONE claim rather than one
    # per audit -- and so the early exit below is the single place that has to
    # print a SUMMARY in the documented three-field shape.
    manifest = (
        None
        if manifest_bytes is None
        else json_or_fail(manifest_bytes, "manifest.json")
    )
    if manifest_bytes is None or manifest is None:
        why = "unavailable" if manifest_bytes is None else "unparseable"
        print(f"\nmanifest.json {why} -- cannot audit.")
        print(
            f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL  |  "
            f"{len(UNVERIFIABLE)} UNVERIFIABLE"
        )
        sys.exit(1)

    audit_a(blobs, manifest_bytes, manifest)
    audit_b(blobs, manifest)
    audit_c(blobs)
    audit_d(manifest)

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
