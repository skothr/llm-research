"""Regression tests for the arc-04 PII redactor.

This code decides what personal data leaves the repo, so its failure modes are
asymmetric: a false negative republishes a real person's contact details, and a
false positive silently corrupts a research corpus. Both directions are pinned
here.

Every ``NEGATIVE`` case below is a real false positive found against the
committed corpora on 2026-07-29, and every ``POSITIVE`` shape is one the first
draft of the regex missed. Run with:

    python -m pytest examples/tests/test_jspace_redact_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jspace_redact_corpus import (  # noqa: E402
    REVIEWED_NOT_PII,
    redact_text,
    scan_text,
)

# (text, expected number of redactions)
POSITIVE = [
    ("Reach me at jane.doe@example.gov for details.", 1),
    # The trailing full stop is the regression: an early `(?![\\d.])` guard
    # rejected it, which hid 22 real phone numbers in the C4 corpus.
    ("Reach me at jane.doe@example.gov or (555) 123-4567.", 2),
    ("Call 555.123.4567 today", 1),
    ("Call 555-123-4567 today", 1),
    ("Call +1 555-123-4567.", 1),
    ("Ship to 1200 Newport Rd, Springfield IL 62704.", 2),
    ("Our office is at 4224 Khouri Court.", 1),
    ("Mail 77 N. Northwest Highway", 1),
]

NEGATIVE = [
    # Dotted version strings must never read as phone numbers.
    ("nvidia-cufft-cu12==11.3.3.83", "version string"),
    ("version 1.13.1.3 released", "version string"),
    ("built on 10.3.9.90 today", "version string"),
    # Institutional names are not street addresses.
    ("the 2004 International Court of Justice opinion", "court, not address"),
    ("appealed to the 9th Circuit Court of Appeals", "court, not address"),
    # A bare suffix with no street name is not an address.
    ("He moved to 03 St. Louis in 1908", "no street-name token"),
    # Plain prose with numbers.
    ("In 1908 they moved south, and by 1954 he had left.", "years only"),
]


@pytest.mark.parametrize("text,expected", POSITIVE)
def test_detects_pii(text: str, expected: int) -> None:
    assert sum(scan_text(text).values()) == expected


@pytest.mark.parametrize("text,why", NEGATIVE)
def test_no_false_positive(text: str, why: str) -> None:
    assert scan_text(text) == {}, f"false positive ({why}): {text!r}"


@pytest.mark.parametrize("text", [t for t, _ in POSITIVE])
def test_redaction_removes_the_value(text: str) -> None:
    """The sentinel must actually replace the matched span."""
    redacted, counts = redact_text(text)
    assert counts
    for label in counts:
        assert f"[{label}]" in redacted
    assert "@" not in redacted or "EMAIL" not in counts


def test_redaction_is_idempotent() -> None:
    once, _ = redact_text(POSITIVE[1][0])
    twice, counts = redact_text(once)
    assert twice == once and counts == {}


def test_reviewed_exception_applies_only_to_its_own_file() -> None:
    """A cleared match stays put in its corpus and nowhere else."""
    name, cleared = next(iter(REVIEWED_NOT_PII.items()))
    sample = f"they moved to {next(iter(cleared))} in 1908"
    assert scan_text(sample, Path(name)) == {}
    assert sum(scan_text(sample, Path("some_other_corpus.json")).values()) == 1


def test_committed_corpora_carry_no_unreviewed_pii() -> None:
    """End-to-end guard: the shipped corpora must stay clean.

    This is the check that matters — it fails if a future corpus lands
    unredacted, rather than relying on anyone remembering to run the script.
    """
    import json

    data_dir = Path(__file__).resolve().parents[2] / "research/arcs/04_jspace/data"
    corpora = sorted(data_dir.glob("*prompts*.json"))
    assert corpora, f"no corpora found under {data_dir}"
    offenders: list[str] = []
    for path in corpora:
        obj = json.loads(path.read_text(encoding="utf-8"))
        for doc in obj.get("prompts", []):
            if isinstance(doc, str) and scan_text(doc, path):
                offenders.append(path.name)
                break
    assert not offenders, f"PII found in committed corpora: {sorted(set(offenders))}"
