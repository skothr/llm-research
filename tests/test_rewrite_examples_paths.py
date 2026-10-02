"""Tests for scripts/rewrite_examples_paths.py (#122)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rewrite_examples_paths",
    Path(__file__).resolve().parents[1] / "scripts" / "rewrite_examples_paths.py",
)
assert _SPEC is not None and _SPEC.loader is not None
rw = importlib.util.module_from_spec(_SPEC)
sys.modules["rewrite_examples_paths"] = rw  # dataclass needs the module registered
_SPEC.loader.exec_module(rw)

ARC01 = "research/arcs/01_nla-verbalizer/scripts"
ARC04 = "research/arcs/04_jspace/scripts"


def exists_in(*files: str):
    present = set(files)
    return lambda path: path in present


# Before any move: sources exist, destinations do not.
BEFORE = exists_in(
    "examples/nla_scan.py",
    "examples/_jspace_paths.py",
    "examples/_hf_models.py",
    "examples/README_NLA.md",
    "examples/tests/test_prose_lint.py",
)
# After the arc-01 move only.
AFTER_ARC01 = exists_in(
    f"{ARC01}/nla_scan.py",
    "examples/_jspace_paths.py",
    "examples/_hf_models.py",
)


def refs(text: str, path: str = "README.md", exists=BEFORE):
    return rw.scan_text(path, text, exists)


def test_destination_rules() -> None:
    assert rw.destination("nla_scan.py") == f"{ARC01}/nla_scan.py"
    assert rw.destination("_layer_hooks.py") == f"{ARC01}/_layer_hooks.py"
    assert rw.destination("_nla_artifacts.py") == f"{ARC01}/_nla_artifacts.py"
    assert rw.destination("_nla_probe.py") == "src/llm_research/nla_probe.py"
    assert rw.destination("_hf_models.py") == "src/llm_research/hf_models.py"
    assert rw.destination("jspace_rerun_scans.sh") == f"{ARC04}/jspace_rerun_scans.sh"
    assert rw.destination("_jspace_pursuit.py") == f"{ARC04}/_jspace_pursuit.py"
    assert rw.destination("tests/_nla_fakes.py") == "tests/_nla_fakes.py"
    assert rw.destination("README_NLA.md") is None
    assert rw.destination("probe_demo.py") is None
    assert rw.destination("tests/sub/x.py") is None


def test_every_tracked_example_maps_or_is_listed_unmapped() -> None:
    out = subprocess.run(
        ["git", "ls-files", "examples/"],
        cwd=rw.REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    names = [p[len("examples/") :] for p in out]
    unmapped = [n for n in names if rw.destination(n) is None]
    assert sorted(unmapped) == sorted(rw.UNMAPPED)
    dests = [rw.destination(n) for n in names if rw.destination(n) is not None]
    assert len(dests) == len(set(dests))  # no two files land on one path


@pytest.mark.parametrize(
    "text",
    [
        "jlens.examples.load_wiki()",
        "prompt = '{examples}.'",
        'p.add_argument("--examples", type=int)',
        "examples_dir = Path('x')",
        "print_examples(res)",
        "jlens.examples/nla_scan.py",
        "{examples}/nla_scan.py",
        "--examples/nla_scan.py",
        "my_examples/nla_scan.py",
    ],
)
def test_non_path_forms_are_untouched(text: str) -> None:
    assert refs(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "run examples/nla_scan.py",
        "`examples/nla_scan.py`",
        "(examples/nla_scan.py)",
        '"examples/nla_scan.py"',
        "x=examples/nla_scan.py",
        "a:examples/nla_scan.py",
        "[examples/nla_scan.py]",
        "examples/nla_scan.py",
    ],
)
def test_path_boundaries_match(text: str) -> None:
    (r,) = refs(text)
    assert r.cls == "auto" and r.target == f"{ARC01}/nla_scan.py"
    assert r.text == "examples/nla_scan.py"


def test_pending_until_destination_exists() -> None:
    (r,) = refs("python examples/nla_scan.py --x")
    assert r.cls == "auto" and not r.ready
    new, applied = rw.rewrite_text("python examples/nla_scan.py --x", [r])
    assert applied == [] and new == "python examples/nla_scan.py --x"


def test_apply_rewrites_ready_refs_only() -> None:
    text = "examples/nla_scan.py and examples/_hf_models.py\n"
    found = refs(text, exists=AFTER_ARC01)
    new, applied = rw.rewrite_text(text, found)
    assert len(applied) == 1
    assert new == f"{ARC01}/nla_scan.py and examples/_hf_models.py\n"


def test_trailing_punctuation_line_suffix_and_stems() -> None:
    text = (
        "See examples/nla_scan.py. Also examples/nla_scan.py:12, "
        "examples/_jspace_paths.resolve and examples/_hf_models\n"
    )
    found = refs(text)
    assert [r.text for r in found] == [
        "examples/nla_scan.py",
        "examples/nla_scan.py",
        "examples/_jspace_paths",
        "examples/_hf_models",
    ]
    assert [r.replacement for r in found] == [
        f"{ARC01}/nla_scan.py",
        f"{ARC01}/nla_scan.py",
        f"{ARC04}/_jspace_paths",
        "src/llm_research/hf_models",
    ]


def test_tests_subdirectory_maps_to_repo_tests() -> None:
    (r,) = refs("pytest examples/tests/test_prose_lint.py -q")
    assert r.replacement == "tests/test_prose_lint.py"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("examples/nla_*.py", "glob or brace form"),
        ("examples/*_audit_findings.py", "glob or brace form"),
        ("examples/{nla_scan,nla_steering}.py", "glob or brace form"),
        ("pyright examples/ now", "bare examples/ directory"),
        ("in examples/.", "bare examples/ directory"),
        ("pytest examples/tests/", "bare examples/tests/ directory"),
        ("examples/probe_demo.py", "unknown file name"),
        ("examples/nla_deleted_long_ago.py", "unknown file name"),
        ("examples/README_NLA.md#x", "unmapped: dissolved"),
        ("other-repo/examples/nla_scan.py", "another path prefix"),
    ],
)
def test_manual_forms(text: str, reason: str) -> None:
    (r,) = refs(text)
    assert r.cls == "manual" and r.replacement is None
    assert r.reason.startswith(reason)


def test_relative_link_is_resolved_and_recomputed() -> None:
    path = "research/arcs/01_nla-verbalizer/README.md"
    text = "[`examples/nla_scan.py`](../../../examples/nla_scan.py)\n"
    found = refs(text, path=path, exists=AFTER_ARC01)
    assert [r.replacement for r in found] == [
        f"{ARC01}/nla_scan.py",
        "scripts/nla_scan.py",
    ]
    new, _ = rw.rewrite_text(text, found)
    assert new == f"[`{ARC01}/nla_scan.py`](scripts/nla_scan.py)\n"


def test_relative_link_from_deeper_file() -> None:
    path = "research/arcs/04_jspace/data/LICENSE-DATA.md"
    exists = exists_in(f"{ARC04}/_jspace_paths.py")
    (r,) = refs("(../../../../examples/_jspace_paths.py)", path=path, exists=exists)
    assert r.replacement == "../scripts/_jspace_paths.py"


def test_relative_link_outside_examples_is_manual() -> None:
    (r,) = refs("(../examples/nla_scan.py)", path="README.md")
    assert r.cls == "manual" and r.reason.startswith("relative path resolves")


def test_crlf_and_other_bytes_preserved() -> None:
    text = "a\r\nrun examples/nla_scan.py\r\n\tz  \r\n"
    new, _ = rw.rewrite_text(text, refs(text, exists=AFTER_ARC01))
    assert new == f"a\r\nrun {ARC01}/nla_scan.py\r\n\tz  \r\n"


def test_line_numbers() -> None:
    found = refs("x\ny\r\nexamples/nla_scan.py\n")
    assert [r.line for r in found] == [3]


def test_file_classes() -> None:
    log = "research/arcs/04_jspace/data/cache/logs/rerun_scans_2026-08-16.log"
    assert refs("examples/nla_scan.py", path=log)[0].cls == "excluded"
    assert (
        refs("examples/nla_scan.py", path="research/archive/x.md")[0].cls == "excluded"
    )
    rec = "research/arcs/02_subliminal/data/step0-owl-neutral-decode/manifest.json"
    (r,) = refs('"examples/nla_scan.py"', path=rec)
    assert r.cls == "hashed-record" and r.kind == "auto"
    man = "research/arcs/03_embedding-atlas/data/MANIFEST.json"
    assert refs('"examples/nla_scan.py"', path=man)[0].cls == "hashed-record"


def _ref(cls: str, kind: str, ready: bool):
    return rw.Ref(
        "f.md",
        1,
        0,
        1,
        "e",
        kind,
        cls,
        "t" if kind == "auto" else None,
        "t" if kind == "auto" else None,
        ready,
        "",
    )


@pytest.mark.parametrize(
    ("found", "strict", "status"),
    [
        ([], False, 0),
        ([_ref("auto", "auto", False)], False, 0),
        ([_ref("manual", "manual", False)], False, 0),
        ([_ref("auto", "auto", True)], False, 1),
        ([_ref("hashed-record", "auto", True)], False, 1),
        ([_ref("excluded", "auto", True)], False, 0),
        ([_ref("auto", "auto", False)], True, 1),
        ([_ref("manual", "manual", False)], True, 1),
        ([_ref("excluded", "manual", False)], True, 0),
    ],
)
def test_check_exit_status(found, strict: bool, status: int, capsys) -> None:
    assert rw.check(found, [], strict) == status
    capsys.readouterr()


def test_apply_skips_records_unless_named_with_flag(
    tmp_path, monkeypatch, capsys
) -> None:
    rec = "research/arcs/02_subliminal/data/step0-owl-neutral-decode/manifest.json"
    monkeypatch.setattr(rw, "REPO", tmp_path)
    target = tmp_path / rec
    target.parent.mkdir(parents=True)
    text = '{"source_path": "examples/nla_scan.py"}\r\n'
    target.write_bytes(text.encode())
    found = rw.scan_text(rec, text, exists_in(f"{ARC01}/nla_scan.py"))
    rw.apply({rec: text}, found, explicit=set(), include_records=True)
    rw.apply({rec: text}, found, explicit={rec}, include_records=False)
    assert target.read_bytes() == text.encode()
    rw.apply({rec: text}, found, explicit={rec}, include_records=True)
    assert (
        target.read_bytes() == f'{{"source_path": "{ARC01}/nla_scan.py"}}\r\n'.encode()
    )
    capsys.readouterr()


def test_main_check_on_this_tool_is_excluded(capsys) -> None:
    assert rw.main(["--check", "--strict", "scripts/rewrite_examples_paths.py"]) == 0
    out = capsys.readouterr().out
    assert "excluded" in out


def test_main_reports_missing_path(capsys) -> None:
    assert rw.main(["--check", "no/such/path"]) == 2
    capsys.readouterr()
