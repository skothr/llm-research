"""Regression tests for the fit-segment ledger and checkpoint introspection.

These three helpers run at the start and end of a lens fit that can take 16 h,
so their failure modes are asymmetric in an unusual way: a raise is far worse
than a wrong answer. Losing the ledger costs provenance precision; raising
kills a run that may already be most of a day deep. Every helper is therefore
pinned to degrade rather than throw on malformed input.

The bug they exist to prevent (#40): the sidecar recorded only the LAST
segment's duration as `wall_seconds`. The 1.5B wikitext fit of 2026-07-29 was
paused once and recorded 2.16 h against 3.53 h actually spent — a 39%
under-report in a committed provenance artifact, silent and plausible-looking.

Run with:

    python -m pytest examples/tests/test_jspace_fit_lens_segments.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jspace_fit_lens import (  # noqa: E402
    append_segment,
    checkpoint_n_done,
    read_segments,
)

STAMP = datetime(2026, 7, 29, 14, 7, 48, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# read_segments — every malformed shape degrades to [], never raises
# --------------------------------------------------------------------------


def test_read_segments_missing_file(tmp_path: Path) -> None:
    assert read_segments(tmp_path / "absent.json") == []


def test_read_segments_truncated_json(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text('[{"seconds": 12.0', encoding="utf-8")  # killed mid-write
    assert read_segments(p) == []


def test_read_segments_wrong_toplevel_type(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text('{"seconds": 12.0}', encoding="utf-8")
    assert read_segments(p) == []


def test_read_segments_drops_entries_without_numeric_seconds(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            [
                {"started": "x", "seconds": 10.0},
                {"started": "y"},  # hand-edited, lost its duration
                {"started": "z", "seconds": "20"},  # string, not a number
                "not-a-dict",
                {"started": "w", "seconds": 30},  # int is fine
            ]
        ),
        encoding="utf-8",
    )
    assert [s["seconds"] for s in read_segments(p)] == [10.0, 30]


def test_read_segments_directory_in_the_way(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.mkdir()
    assert read_segments(p) == []


# --------------------------------------------------------------------------
# append_segment — accumulates, and the total is what the sidecar reports
# --------------------------------------------------------------------------


def test_append_segment_creates_then_accumulates(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    assert [s["seconds"] for s in append_segment(p, STAMP, 4952.0)] == [4952.0]
    segments = append_segment(p, STAMP, 7773.0)
    assert [s["seconds"] for s in segments] == [4952.0, 7773.0]
    # The exact case from #40: two segments summing to 3.53 h, where the old
    # code would have written 2.16 h.
    total = sum(float(s["seconds"]) for s in segments)
    assert round(total / 3600, 2) == 3.53
    assert round(7773.0 / 3600, 2) == 2.16


def test_append_segment_persists_and_reloads(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    append_segment(p, STAMP, 1.0)
    append_segment(p, STAMP, 2.0)
    assert [s["seconds"] for s in read_segments(p)] == [1.0, 2.0]
    assert p.exists() and not p.with_suffix(p.suffix + ".tmp").exists()


def test_append_segment_records_start_time(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    segments = append_segment(p, STAMP, 1.0)
    assert segments[0]["started"] == "2026-07-29T14:07:48+00:00"


def test_append_segment_survives_a_corrupt_ledger(tmp_path: Path) -> None:
    """A corrupt ledger loses history but must not lose the finished segment."""
    p = tmp_path / "s.json"
    p.write_text("}{", encoding="utf-8")
    assert [s["seconds"] for s in append_segment(p, STAMP, 99.0)] == [99.0]


def test_append_segment_unwritable_path_does_not_raise(tmp_path: Path) -> None:
    """The fit is finished by this point; a write failure must not lose it."""
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    try:
        segments = append_segment(d / "s.json", STAMP, 5.0)
    finally:
        d.chmod(0o700)
    assert [s["seconds"] for s in segments] == [5.0]


# --------------------------------------------------------------------------
# checkpoint_n_done — distinguishes resume from fresh start
# --------------------------------------------------------------------------


def test_checkpoint_n_done_missing_is_none(tmp_path: Path) -> None:
    assert checkpoint_n_done(tmp_path / "absent.ckpt.pt") is None


def test_checkpoint_n_done_reads_progress(tmp_path: Path) -> None:
    p = tmp_path / "c.ckpt.pt"
    torch.save(
        {"n_done": 30, "next_idx": 30, "jacobian_sum": {0: torch.zeros(2, 2)}}, p
    )
    assert checkpoint_n_done(p) == 30


def test_checkpoint_n_done_without_the_key(tmp_path: Path) -> None:
    p = tmp_path / "c.ckpt.pt"
    torch.save({"jacobian_sum": {}}, p)
    assert checkpoint_n_done(p) is None


def test_checkpoint_n_done_on_garbage_is_none(tmp_path: Path) -> None:
    """A half-written checkpoint must read as 'no progress', not crash."""
    p = tmp_path / "c.ckpt.pt"
    p.write_bytes(b"not a torch file")
    assert checkpoint_n_done(p) is None
