"""Pytest setup for the theory/kb/ test suite.

Adds theory/kb/ to sys.path so tests can import lint as a top-level module,
mirroring theory/series/tests/conftest.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

THEORY_KB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THEORY_KB))
