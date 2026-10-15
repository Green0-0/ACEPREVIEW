"""Neat test fixtures; reuses the graph test fakes and sample project."""
from __future__ import annotations

import sys
from pathlib import Path

GRAPH_TESTS = Path(__file__).resolve().parent.parent / "graph_tests"
if str(GRAPH_TESTS) not in sys.path:
    sys.path.insert(0, str(GRAPH_TESTS))

from sample_project import project_dir, session, write_sample_project  # noqa: E402,F401
