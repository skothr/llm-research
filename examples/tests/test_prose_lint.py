"""Tests for scripts/prose_lint.py (#120)."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prose_lint", Path(__file__).resolve().parents[2] / "scripts" / "prose_lint.py"
)
assert _SPEC is not None and _SPEC.loader is not None
prose_lint = importlib.util.module_from_spec(_SPEC)
sys.modules["prose_lint"] = prose_lint  # dataclass needs the module registered
_SPEC.loader.exec_module(prose_lint)


def test_reports_each_phrase_with_its_line() -> None:
    text = "plain line\nevery load-bearing claim\nthis is genuinely large\n"
    hits = prose_lint.scan_text("x.md", text)
    assert [(h.line, h.phrase) for h in hits] == [
        (2, "load-bearing"),
        (3, "genuinely"),
    ]


def test_matching_is_case_insensitive_and_word_bounded() -> None:
    assert prose_lint.scan_text("x.md", "Load-Bearing\n")
    assert not prose_lint.scan_text("x.md", "dishonesty and crispness\n")


def test_allow_marker_skips_the_line() -> None:
    text = "an honest baseline <!-- prose-lint: allow -->\n"
    assert prose_lint.scan_text("x.md", text) == []


def test_quoted_owner_turns_are_skipped() -> None:
    text = '> *"this is genuinely load-bearing"*\n> plain genuinely\n'
    hits = prose_lint.scan_text("x.md", text)
    assert [h.line for h in hits] == [2]


def test_every_line_of_a_multiline_quote_is_skipped() -> None:
    text = (
        '> *"first line\n'
        "> a genuinely quoted middle\n"
        '> last line"*\n'
        "> — [session 2026-05-12]\n"
        "commentary that is genuinely ours\n"
    )
    hits = prose_lint.scan_text("x.md", text)
    assert [h.line for h in hits] == [5]


def test_unterminated_quote_ends_with_its_blockquote() -> None:
    text = '> *"opened but never closed\n> still quoted\nback to genuinely ours\n'
    hits = prose_lint.scan_text("x.md", text)
    assert [h.line for h in hits] == [3]


def test_scope_excludes_verbatim_and_dated_records() -> None:
    assert prose_lint.is_scanned("research/arcs/04_jspace/README.md")
    assert prose_lint.is_scanned("examples/jspace_audit_findings.py")
    assert prose_lint.is_scanned("theory/series/paper-1/main.tex")
    assert not prose_lint.is_scanned("theory/kb/excerpts/gurnee2026.md")
    assert not prose_lint.is_scanned("research/archive/old.md")
    assert not prose_lint.is_scanned("theory/reviews/pass-1.md")
    assert not prose_lint.is_scanned("data/audit_2026-08-17.log")
    assert not prose_lint.is_scanned("scripts/prose_lint.py")
    assert not prose_lint.is_scanned("examples/tests/test_prose_lint.py")


def test_area_groups_by_directory() -> None:
    assert prose_lint.area("theory/kb/notes/a.md") == "theory/kb"
    assert prose_lint.area("examples/jspace_audit_findings.py") == "examples"
    assert prose_lint.area("README.md") == "README.md"


def test_untracked_files_are_candidates(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    for var in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(var, raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "new.md").write_text("a crisp result\n")
    monkeypatch.setattr(prose_lint, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert prose_lint.candidate_files(["new.md"]) == ["new.md"]
    assert prose_lint.main(["new.md"]) == 1
    assert prose_lint.candidate_files(["."]) == ["new.md"]


def test_main_exit_codes(tmp_path: Path, monkeypatch) -> None:
    hit_file = "x.md"
    monkeypatch.setattr(prose_lint, "candidate_files", lambda _paths: [hit_file])
    monkeypatch.setattr(prose_lint, "REPO", tmp_path)
    (tmp_path / hit_file).write_text("a crisp result\n")
    assert prose_lint.main([]) == 1
    assert prose_lint.main(["--report"]) == 0
    (tmp_path / hit_file).write_text("a clear result\n")
    assert prose_lint.main([]) == 0


def test_phrase_split_by_a_hard_wrap_is_reported_at_the_first_line() -> None:
    text = "the figure tells\nus that the gap holds\n> a quoted note that sits\n> at the top\n"
    hits = prose_lint.scan_text("x.md", text)
    assert [(h.line, h.phrase) for h in hits] == [(1, "tells us"), (3, "sits at")]


def test_wrap_check_respects_allowed_and_quoted_lines() -> None:
    text = "it tells <!-- prose-lint: allow -->\nus nothing\n"
    assert prose_lint.scan_text("x.md", text) == []


def _git_repo(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    for var in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(var, raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(prose_lint, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)


def test_unscannable_paths_exit_2_even_with_report(tmp_path: Path, monkeypatch) -> None:
    _git_repo(tmp_path, monkeypatch)
    (tmp_path / "ok.md").write_text("a clear result\n")
    assert prose_lint.main(["ok.md"]) == 0
    assert prose_lint.main(["typo.md"]) == 2
    assert prose_lint.main(["typo.md", "--report"]) == 2
    assert prose_lint.main([str(tmp_path.parent)]) == 2
    (tmp_path / "bad.md").write_bytes(b"caf\xe9\n")
    assert prose_lint.main(["bad.md"]) == 2


def test_symlinks_are_skipped_not_read(tmp_path: Path, monkeypatch) -> None:
    _git_repo(tmp_path, monkeypatch)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("a genuinely private note\n")
    (tmp_path / "link.md").symlink_to(outside)
    assert prose_lint.scan(["link.md"]) == []
    assert prose_lint.main(["link.md"]) == 0
