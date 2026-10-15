"""Edit.git_patch checked against real git: every patch must pass ``git apply --check`` and
``git apply`` must reproduce ``Edit.files`` byte for byte (GRAPH_NEAT_DESIGN.md section 4)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ace.graph.edits import Edit
from sample_project import write_sample_project

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")

EXTRA_FILES: dict[str, bytes] = {
    "data/no_newline.txt": b"alpha\nbeta\ngamma",
    "data/crlf.txt": b"one\r\ntwo\r\nthree\r\nfour\r\n",
}

GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
       "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"]

CASES: dict[str, dict[str, bytes | None]] = {
    "no_trailing_newline_change": {"data/no_newline.txt": b"alpha\nBETA\ngamma"},
    "crlf_change": {"data/crlf.txt": b"one\r\nTWO\r\nthree\r\nfour\r\nfive\r\n"},
    "add_trailing_newline_only": {"data/no_newline.txt": b"alpha\nbeta\ngamma\n"},
    "remove_trailing_newline_only": {"README.md": b"# Sample project\n\nA tiny project used in tests."},
    "create": {"pkg/extra.py": b"VALUE = 1\n\n\ndef extra():\n    return VALUE\n"},
    "delete": {"pkg/service.py": None},
    "multi_file": {
        "pkg/utils.py": b'"""Utilities."""\nimport re\n',
        "data/crlf.txt": b"one\r\ntwo\nthree\r\nfour\r\n",
        "docs/empty.txt": b"",
        "notes.txt": None,
        "data/no_newline.txt": b"",
    },
}


def git(repo: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*GIT, *args], cwd=repo, input=stdin, capture_output=True, check=False
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = write_sample_project(tmp_path / "repo")
    for relative, data in EXTRA_FILES.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "pristine")):
        result = git(root, *args)
        assert result.returncode == 0, result.stderr.decode()
    return root


@pytest.mark.parametrize("case", sorted(CASES))
def test_git_apply_reproduces_edit(repo: Path, case: str):
    from agent2.core.session import Session

    session = Session(str(repo), rollout_dir=str(repo.parent / "rollouts"))
    edit = Edit(dict(CASES[case]))
    patch = edit.git_patch(session)
    assert patch, "every case changes something"
    assert edit.unified_diff(session) == patch
    patch_file = repo.parent / f"{case}.patch"
    patch_file.write_bytes(patch.encode("utf-8", "surrogateescape"))

    check = git(repo, "apply", "--check", str(patch_file))
    assert check.returncode == 0, check.stderr.decode() + "\n" + patch
    applied = git(repo, "apply", str(patch_file))
    assert applied.returncode == 0, applied.stderr.decode()

    for path, data in edit.files.items():
        target = repo / path
        if data is None:
            assert not target.exists(), path
        else:
            assert target.read_bytes() == data, path
    # Nothing else changed: git sees exactly the edited paths.
    status = git(repo, "status", "--porcelain", "-uall").stdout.decode().splitlines()
    changed = {line[3:] for line in status}
    assert changed == set(edit.files)


def test_git_diff_matches_for_simple_change(repo: Path):
    """Hunks equal ``git diff`` output for the same change (headers aside)."""
    from agent2.core.session import Session

    session = Session(str(repo), rollout_dir=str(repo.parent / "rollouts"))
    new = b"alpha\nBETA\ngamma"
    edit = Edit({"data/no_newline.txt": new})
    ours = edit.git_patch(session)
    (repo / "data/no_newline.txt").write_bytes(new)
    theirs = git(repo, "diff", "--no-color", "--no-ext-diff").stdout.decode()
    strip = lambda text: [line for line in text.splitlines() if not line.startswith("index ")]
    assert strip(ours) == strip(theirs)
