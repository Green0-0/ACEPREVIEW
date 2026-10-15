"""Tests for ace.graph.edits (GRAPH_NEAT_DESIGN.md section 4)."""
from __future__ import annotations

import json

from agent2.core.workspace import Workspace
from ace.graph.edits import Edit


def _edited_workspace(session) -> Workspace:
    workspace = Workspace(session)
    workspace.write_file("pkg/utils.py", b'"""Utilities."""\n\n\ndef clamp(v, lo, hi):\n    return v\n')
    workspace.write_file("pkg/new_module.py", b"VALUE = 1\n")
    workspace.delete_file("notes.txt")
    # Unchanged checkout: written back with identical bytes, must be skipped.
    workspace.write_file("README.md", session.files["README.md"].current_state.bytes)
    return workspace


def test_from_workspace_captures_changes_and_skips_unchanged(session):
    workspace = _edited_workspace(session)
    edit = Edit.from_workspace(workspace)
    assert edit.touched_paths() == {"pkg/utils.py", "pkg/new_module.py", "notes.txt"}
    assert edit.files["notes.txt"] is None
    assert edit.files["pkg/new_module.py"] == b"VALUE = 1\n"
    assert not edit.is_empty()


def test_round_trip_diff_equality(session):
    workspace = _edited_workspace(session)
    edit = Edit.from_workspace(workspace)
    rebuilt = edit.to_workspace(session)
    assert rebuilt is not workspace
    assert rebuilt.diff() == workspace.diff()
    assert Edit.from_workspace(rebuilt) == edit
    # per_file_diff covers the same files as Workspace.diff (but in faithful git form).
    assert [p for p, _ in edit.per_file_diff(session)] == [p for p, _ in workspace.diff()]
    assert Edit.from_workspace(rebuilt).per_file_diff(session) == edit.per_file_diff(session)
    patch = edit.git_patch(session)
    assert edit.unified_diff(session) == patch
    assert patch == "\n".join(content for _, content in edit.per_file_diff(session))


def test_git_headers(session):
    edit = Edit({"pkg/utils.py": b"x\n", "docs/new.md": b"# New\n", "notes.txt": None})
    diffs = dict(edit.per_file_diff(session))
    assert diffs["pkg/utils.py"].startswith(
        "diff --git a/pkg/utils.py b/pkg/utils.py\n--- a/pkg/utils.py\n+++ b/pkg/utils.py\n@@ "
    )
    assert diffs["docs/new.md"] == (
        "diff --git a/docs/new.md b/docs/new.md\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/docs/new.md\n@@ -0,0 +1 @@\n+# New\n"
    )
    assert diffs["notes.txt"].startswith(
        "diff --git a/notes.txt b/notes.txt\ndeleted file mode 100644\n"
        "--- a/notes.txt\n+++ /dev/null\n@@ -1,4 +0,0 @@\n-line one\n"
    )
    assert all(text.endswith("\n") for text in diffs.values())


def test_no_newline_marker_and_trailing_newline_only(session):
    # README.md is "# Sample project\n\nA tiny project used in tests.\n".
    stripped = Edit({"README.md": b"# Sample project\n\nA tiny project used in tests."})
    diff = stripped.unified_diff(session)
    assert diff.endswith(
        "-A tiny project used in tests.\n+A tiny project used in tests.\n"
        "\\ No newline at end of file\n"
    )
    # Removing only the trailing newline is a real change (the old diff was empty).
    edit = Edit({"notes.txt": b"line one\nline two\nline three\nline four"})
    assert edit.unified_diff(session).endswith(
        "-line four\n+line four\n\\ No newline at end of file\n"
    )


def test_crlf_and_lone_cr_preserved(session):
    edit = Edit({"notes.txt": b"line one\r\nline two\r\nline 3\rstill 3\nline four\n"})
    diff = edit.unified_diff(session)
    assert "+line one\r\n" in diff and "+line two\r\n" in diff
    assert "+line 3\rstill 3\n" in diff  # a lone CR does not split the line
    assert " line four\n" in diff


def test_identical_bytes_and_missing_delete_give_no_diff(session):
    same = session.files["README.md"].current_state.bytes
    assert Edit({"README.md": same, "ghost.txt": None}).per_file_diff(session) == []


def test_created_file_diff_and_content(session):
    edit = Edit({"docs/guide.md": b"# Guide\n"})
    workspace = edit.to_workspace(session)
    assert workspace.read_file("docs/guide.md").current_state.bytes == b"# Guide\n"
    (path, diff), = edit.per_file_diff(session)
    assert path == "docs/guide.md"
    assert "--- /dev/null" in diff and "+++ b/docs/guide.md" in diff and "+# Guide" in diff


def test_deleted_file(session):
    edit = Edit({"pkg/utils.py": None})
    workspace = edit.to_workspace(session)
    assert workspace.read_file("pkg/utils.py") is None
    assert "pkg/utils.py" not in workspace.paths()
    (path, diff), = edit.per_file_diff(session)
    assert path == "pkg/utils.py"
    assert "+++ /dev/null" in diff and "-def clamp(value, low, high):" in diff


def test_modified_python_file_is_reparsed(session):
    new = b"def only():\n    return 1\n"
    workspace = Edit({"pkg/utils.py": new}).to_workspace(session)
    file = workspace.read_file("pkg/utils.py")
    assert file.current_state.bytes == new
    assert "only" in file.code_nodes
    assert "slugify" in session.files["pkg/utils.py"].code_nodes  # pristine untouched


def test_json_round_trip(session):
    edit = Edit.from_workspace(_edited_workspace(session))
    data = edit.to_json()
    text = json.dumps(data)
    restored = Edit.from_json(json.loads(text))
    assert restored == edit
    assert data["files"]["notes.txt"] is None
    assert isinstance(data["files"]["pkg/new_module.py"], str)
    assert Edit.from_json(Edit({"bin.dat": b"\x00\xff\x10"}).to_json()).files["bin.dat"] == b"\x00\xff\x10"


def test_empty_edit(session):
    edit = Edit.from_workspace(Workspace(session))
    assert edit.is_empty()
    assert edit.touched_paths() == set()
    assert edit.per_file_diff(session) == []
    assert edit.unified_diff(session) == ""
    assert Edit.from_json(edit.to_json()) == edit


def test_created_then_deleted_is_skipped(session):
    workspace = Workspace(session)
    workspace.write_file("tmp.txt", b"x\n")
    workspace.delete_file("tmp.txt")
    assert Edit.from_workspace(workspace).is_empty()
