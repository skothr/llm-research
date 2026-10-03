"""Tests for scripts/rewrite_examples_paths.py (#122)."""

import importlib.util
import re
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


def _tracked() -> list[str]:
    if not (rw.REPO / ".git").exists():
        pytest.skip("not a git checkout")
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=rw.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    return [p for p in out.split("\0") if p]


def test_tracked_skips_outside_a_git_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rw, "REPO", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        _tracked()


def test_every_tracked_example_maps_or_is_listed_unmapped() -> None:
    # Reads the live tree. The arc 01 move left no tracked file under
    # examples/ and UNMAPPED is empty, so `names` is empty and every assertion
    # here compares empty collections. They check something again only if a
    # file is added under examples/: it must then map or be listed in
    # UNMAPPED. The test below checks the map against where the files are now.
    tracked = _tracked()
    names = [p[len("examples/") :] for p in tracked if p.startswith("examples/")]
    unmapped = [n for n in names if rw.destination(n) is None]
    assert sorted(unmapped) == sorted(rw.UNMAPPED)
    dests = [rw.destination(n) for n in names if rw.destination(n) is not None]
    assert len(dests) == len(set(dests))  # no two files land on one path
    others = {p for p in tracked if not p.startswith("examples/")}
    assert sorted(set(dests) & others) == []  # no move overwrites a file


def test_every_tracked_arc_script_is_where_the_map_sends_it() -> None:
    # The inverse of the test above, against the tree after the moves. It is
    # a check of the one-time migration, so it covers only what the map knows:
    # each tracked file directly under an arc's scripts/ whose name the map
    # gives a destination for (as the old examples/<name>) must be at that
    # destination, so a rewritten reference names a real file. A file whose
    # name the map does not know (a script or a README that never lived in
    # examples/) is skipped.
    tracked = set(_tracked())
    mapped = {}
    for p in sorted(tracked):
        if re.fullmatch(r"research/arcs/[^/]+/scripts/[^/]+", p):
            dest = rw.destination(p.rsplit("/", 1)[1])
            if dest is not None:
                mapped[p] = dest
    assert {p: d for p, d in mapped.items() if d != p} == {}
    # Each arc contributes at least one checked file, so the test cannot pass
    # on an empty or partial selection.
    slugs = {p.split("/")[2] for p in mapped}
    assert slugs >= {
        "01_nla-verbalizer",
        "02_subliminal",
        "03_embedding-atlas",
        "04_jspace",
    }
    for module, dest in rw.PACKAGE_MODULES.items():
        assert dest in tracked, module


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
        "my__examples/nla_scan.py",
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


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("Run examples/nla_scan.py?", f"Run {ARC01}/nla_scan.py?"),
        ("*examples/nla_scan.py*", f"*{ARC01}/nla_scan.py*"),
        ("**examples/nla_scan.py**.", f"**{ARC01}/nla_scan.py**."),
        ("`examples/nla_scan.py`?", f"`{ARC01}/nla_scan.py`?"),
    ],
)
def test_trailing_glob_characters_are_punctuation(text: str, written: str) -> None:
    found = refs(text, exists=AFTER_ARC01)
    assert [r.cls for r in found] == ["auto"]
    assert rw.rewrite_text(text, found)[0] == written


@pytest.mark.parametrize(
    "text",
    [
        "examples/nla_scan*.py",
        "examples/nla_scan.py{,.bak}",
        "examples/nla_scan.py*",
        "examples/_jspace_paths*",
        "*examples/nla_scan.py**",
    ],
)
def test_glob_after_a_known_name_is_manual(text: str) -> None:
    (r,) = refs(text)
    assert r.cls == "manual" and r.reason == "glob or brace form"


def test_stem_with_attribute_still_maps() -> None:
    (r,) = refs("examples/nla_scan.main")
    assert r.cls == "auto" and r.replacement == f"{ARC01}/nla_scan"


def test_keep_marker_keeps_every_ref_on_its_line() -> None:
    text = (
        "../jacobian-lens/examples/ and examples/nla_scan.py  "
        "<!-- rewrite-paths: keep -->\n"
        "examples/nla_scan.py\n"
    )
    found = refs(text, exists=AFTER_ARC01)
    assert [r.cls for r in found] == ["kept", "kept", "auto"]
    assert all(r.kind == "kept" for r in found[:2])
    assert all(r.replacement is None and not r.ready for r in found[:2])
    assert "[pending]" not in rw._format(found[1])
    new, applied = rw.rewrite_text(text, found)
    assert len(applied) == 1
    assert new == text.replace("\nexamples/", f"\n{ARC01}/")


def test_keep_marker_does_not_override_file_class() -> None:
    log = "research/arcs/04_jspace/data/cache/logs/x.log"
    (r,) = refs("examples/nla_scan.py  # rewrite-paths: keep", path=log)
    assert r.cls == "excluded"


def test_tests_subdirectory_maps_to_repo_tests() -> None:
    (r,) = refs("pytest examples/tests/test_prose_lint.py -q")
    assert r.replacement == "tests/test_prose_lint.py"


@pytest.mark.parametrize(
    "text",
    [
        r"\texttt{examples/nla_scan.py}",
        "~~examples/nla_scan.py~~",
        "_examples/nla_scan.py_",
        "__examples/nla_scan.py__",
        "$examples/nla_scan.py",
        "@examples/nla_scan.py",
        "%examples/nla_scan.py",
        "+examples/nla_scan.py",
        "}examples/nla_scan.py",
    ],
)
def test_unusual_prefix_is_manual(text: str) -> None:
    (r,) = refs(text, exists=AFTER_ARC01)
    assert r.cls == "manual" and r.reason == "unusual prefix"
    assert rw.rewrite_text(text, [r])[0] == text


def test_brace_list_first_item_is_manual() -> None:
    found = refs("{examples/nla_scan.py,examples/_hf_models.py}")
    assert [(r.cls, r.reason) for r in found] == [
        ("manual", "unusual prefix"),
        ("auto", "repo-root path"),
    ]


@pytest.mark.parametrize(
    ("text", "exists"),
    [
        ("examples/nla_scan.py.bak", BEFORE),
        ("examples/nla_scan.py.Log", BEFORE),
        (
            "examples/jspace_rerun_scans.sh.log",
            exists_in("examples/jspace_rerun_scans.sh"),
        ),
    ],
)
def test_file_name_followed_by_another_extension_is_manual(text: str, exists) -> None:
    (r,) = refs(text, exists=exists)
    assert r.cls == "manual" and r.reason == "unknown file name"
    assert r.text == text


@pytest.mark.parametrize(
    "text", ["examples/nla_scan/out.json", "examples/_hf_models/x"]
)
def test_stem_followed_by_slash_is_manual(text: str) -> None:
    (r,) = refs(text)
    assert r.cls == "manual" and r.reason == "unknown file name"


def test_stem_before_trailing_period_still_maps() -> None:
    (r,) = refs("See examples/_hf_models.")
    assert r.cls == "auto" and r.text == "examples/_hf_models"


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
        ("examples/README_NLA.md#x", "unknown file name"),
        ("other-repo/examples/nla_scan.py", "another path prefix"),
        ("foo/../examples/nla_scan.py", "another path prefix"),
        ("examples/nla_x.py?", "glob or brace form"),
        ("examples/nla_scan.json", "unknown file name"),
        ("examples/nla_scan.log", "unknown file name"),
        ("examples/nla_scan.PDF", "unknown file name"),
        ("examples/nla_scan.Json", "unknown file name"),
        ("examples/nla_scan.safetensors", "unknown file name"),
        ("examples/nla_scan.bak", "unknown file name"),
        ('root / "examples" / "nla_scan.py"', "path component"),
        ("os.path.join(root, 'examples', name)", "path component"),
        ('assert area("x") == "examples"', "path component"),
    ],
)
def test_manual_forms(text: str, reason: str) -> None:
    (r,) = refs(text)
    assert r.cls == "manual" and r.replacement is None
    assert r.reason.startswith(reason)


def test_unmapped_name_reports_its_reason(monkeypatch) -> None:
    monkeypatch.setattr(rw, "UNMAPPED", {"NOTES.md": "dissolved"})
    [r] = refs("examples/NOTES.md#x")
    assert r.cls == "manual" and r.replacement is None
    assert r.reason == "unmapped: dissolved"


@pytest.mark.parametrize(
    "text",
    ['{"examples": 3}', "{'examples' : 3}", 'add_argument(dest="examples")'],
)
def test_key_or_keyword_value_is_not_a_component(text: str) -> None:
    assert refs(text) == []


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
    # As the scanner builds them: a target and a replacement only on an auto
    # ref outside the kept class.
    auto = kind == "auto" and cls != "kept"
    return rw.Ref(
        "f.md",
        1,
        0,
        1,
        "e",
        kind,
        cls,
        "t" if auto else None,
        "t" if auto else None,
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
        ([_ref("kept", "kept", False)], True, 0),
    ],
)
def test_check_exit_status(found, strict: bool, status: int, capsys) -> None:
    assert rw.check(found, [], strict) == status
    capsys.readouterr()


def test_ready_counts_and_label_only_for_auto_class(capsys) -> None:
    rec = _ref("hashed-record", "auto", True)
    assert "[ready]" not in rw._format(rec) and "[pending]" not in rw._format(rec)
    rw.check([rec], [], False)
    assert "ready=" not in capsys.readouterr().out
    rw.check([_ref("auto", "auto", True)], [], False)
    assert "ready=1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("not_scanned", "strict", "status"),
    [
        (["a.json"], False, 0),
        (["a.json"], True, 1),
        (["research/archive/a.json"], True, 0),
    ],
)
def test_not_scanned_files_are_listed_and_fail_strict(
    not_scanned: list[str], strict: bool, status: int, capsys
) -> None:
    assert rw.check([], [], strict, not_scanned) == status
    out = capsys.readouterr().out
    assert f"not scanned: {not_scanned[0]}" in out
    assert "1 text-like file(s) not scanned" in out


def test_collect_lists_unreadable_text_like_files(tmp_path, monkeypatch) -> None:
    files = {
        "bad.json": b"\xff\xfe examples/nla_scan.py",
        "pic.png": b"\x89PNG\0",
        "lfs.md": b"version https://git-lfs.github.com/spec/v1\n",
        "w.pt": b"version https://git-lfs.github.com/spec/v1\n",
        "ok.md": b"examples/nla_scan.py\n",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(rw, "REPO", tmp_path)
    monkeypatch.setattr(rw, "candidate_files", lambda _paths: (sorted(files), set()))
    monkeypatch.setattr(rw, "lfs_files", lambda _fs: {"lfs.md", "w.pt"})
    texts, found, _, not_scanned = rw.collect([])
    assert not_scanned == ["bad.json", "lfs.md"]
    assert list(texts) == ["ok.md"] and len(found) == 1


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


def test_read_text_skips_symlinked_parent_and_hard_link(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (outside / "a.md").write_text("examples/nla_scan.py\n")
    (repo / "d").symlink_to(outside)
    (repo / "log.md").write_text("examples/nla_scan.py\n")
    (repo / "draft.md").hardlink_to(repo / "log.md")
    (repo / "plain.md").write_text("examples/nla_scan.py\n")
    monkeypatch.setattr(rw, "REPO", repo)
    assert rw.read_text("d/a.md") is None
    assert rw.read_text("draft.md") is None
    assert rw.read_text("plain.md") == "examples/nla_scan.py\n"


def test_apply_skips_a_file_changed_since_the_scan(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(rw, "REPO", tmp_path)
    target = tmp_path / "notes.md"
    text = "see examples/nla_scan.py\n"
    found = rw.scan_text("notes.md", text, exists_in(f"{ARC01}/nla_scan.py"))
    target.write_text("edited after the scan\n")
    assert rw.apply({"notes.md": text}, found, explicit=set(), include_records=False) == 1
    assert target.read_text() == "edited after the scan\n"
    assert "changed since it was scanned" in capsys.readouterr().err


def test_apply_write_keeps_mode_and_leaves_no_temp_file(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(rw, "REPO", tmp_path)
    target = tmp_path / "run.sh"
    text = "python examples/nla_scan.py\n"
    target.write_text(text)
    target.chmod(0o755)
    found = rw.scan_text("run.sh", text, exists_in(f"{ARC01}/nla_scan.py"))
    assert rw.apply({"run.sh": text}, found, explicit=set(), include_records=False) == 0
    assert target.read_text() == f"python {ARC01}/nla_scan.py\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.sh"]
    capsys.readouterr()


# main() resolves path arguments against the current directory, as a CLI
# should; these tests pass absolute paths so they hold from any cwd.
def test_main_check_on_this_tool_is_excluded(capsys) -> None:
    tool = str(rw.REPO / "scripts" / "rewrite_examples_paths.py")
    assert rw.main(["--check", "--strict", tool]) == 0
    out = capsys.readouterr().out
    assert "excluded" in out


def test_main_reports_missing_path(capsys) -> None:
    assert rw.main(["--check", str(rw.REPO / "no" / "such" / "path")]) == 2
    err = capsys.readouterr().err
    assert "no files under no/such/path" in err


def test_main_keeps_the_name_of_an_in_repo_symlink(monkeypatch, tmp_path) -> None:
    (tmp_path / "real.md").write_text("x\n")
    (tmp_path / "link.md").symlink_to(tmp_path / "real.md")
    monkeypatch.setattr(rw, "REPO", tmp_path)
    seen: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"link.md\0")

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    files, explicit = rw.candidate_files([str(tmp_path / "link.md")])
    assert seen[0][-1] == "link.md"
    assert files == ["link.md"] and explicit == {"link.md"}


def test_main_accepts_a_symlinked_repo_path(tmp_path, capsys) -> None:
    link = tmp_path / "repo"
    link.symlink_to(rw.REPO)
    tool = str(link / "scripts" / "rewrite_examples_paths.py")
    assert rw.main(["--check", tool]) == 0
    assert "outside the repository" not in capsys.readouterr().err
