"""Graph test fixtures. Shared fakes live in fakes.py and sample_project.py."""
from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from sample_project import project_dir, session, write_sample_project  # noqa: E402,F401
