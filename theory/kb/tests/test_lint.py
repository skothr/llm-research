"""Tests for kb/lint.py — the two checks added to stop the 2026-08 defect
classes recurring: `§`-label shape, and LaTeX-comment stripping ahead of the
series scan.
"""

from __future__ import annotations

from lint import TEX_SECTION_RE, _strip_tex_comments, label_is_source_place


# --- `§` labels that name a place in the source paper ----------------------


def test_section_numbers_are_accepted():
    for label in ("3", "3.2", "4.1.2", "A.1", "C"):
        assert label_is_source_place(label), label


def test_section_ranges_are_accepted():
    # `\S 4--5` (LaTeX en-dash) and the plain hyphen spelling both appear.
    for label in ("4--5", "5-6", "3.1--3.2"):
        assert label_is_source_place(label), label


def test_named_paper_parts_are_accepted():
    for label in ("abstract", "Abstract", "appendix", "introduction"):
        assert label_is_source_place(label), label


def test_title_cased_section_titles_are_accepted():
    # The fallback for papers whose sections are named but not numbered.
    for label in ("Method", "Green-Red"):
        assert label_is_source_place(label), label


def test_trailing_period_is_tolerated():
    assert label_is_source_place("3.2.")


def test_excerpt_anchor_slugs_are_rejected():
    # The C2 defect class: an excerpt anchor sitting in the `§` slot, which
    # renders in the PDFs as "(Rein et al., 2023, §sec-diamond)".
    for label in (
        "sec-diamond",
        "sec-delta",
        "six-evals",
        "method",
        "problem",
        "byte-fallback",
        "1.pillars",
    ):
        assert not label_is_source_place(label), label


def test_sec_prefixed_titlecase_is_still_rejected():
    # `Sec-Diamond` is a slug wearing a capital, not a section title.
    assert not label_is_source_place("sec-Diamond")


# --- the three .tex spellings of `§` ---------------------------------------


def test_all_three_section_sign_spellings_are_seen():
    # `\S{}` is a control-sequence terminator, not a group. Missing it is how
    # 25 `\S{}sec-diamond` sites survived the first sweep.
    for text in (r"\S 3.2", r"\S{3.2}", r"\S{}3.2"):
        m = TEX_SECTION_RE.search(text)
        assert m is not None, text
        assert m.group(1) == "3.2", text


def test_braced_slug_label_is_captured_for_flagging():
    m = TEX_SECTION_RE.search(r"\citep[\S{}sec-diamond]{rein2023-gpqa}")
    assert m is not None
    assert m.group(1) == "sec-diamond"
    assert not label_is_source_place(m.group(1))


# --- LaTeX comment stripping -----------------------------------------------


def test_comment_body_is_removed_and_line_count_preserved():
    text = "a\n% [paper-key §X; kb/excerpts/key#anchor]\nb\n"
    out = _strip_tex_comments(text)
    assert "kb/excerpts" not in out
    assert out.split("\n") == ["a", "", "b", ""]


def test_escaped_percent_is_not_a_comment():
    text = r"rates 0.14\% post-hoc, see kb/excerpts/foo#sec-1"
    assert _strip_tex_comments(text) == text


def test_inline_comment_keeps_the_code_before_it():
    text = "\\citep[\\S 3]{key} % kb/excerpts/key#anchor\n"
    out = _strip_tex_comments(text)
    assert out.startswith("\\citep[\\S 3]{key} ")
    assert "kb/excerpts" not in out
