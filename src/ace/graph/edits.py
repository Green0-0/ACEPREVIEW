"""Edits: the complete new content of every changed file, against the pristine session.

See GRAPH_NEAT_DESIGN.md section 4. An :class:`Edit` maps project-relative paths to the
new bytes of the file (``None`` means the file is deleted). No patch application is ever
needed: an Edit is materialised by writing those bytes onto a fresh ``Workspace`` over the
pristine ``Session``, and diffs are derived from the pristine bytes for display.

Choices made where the contract is silent:

- ``from_workspace`` skips entries whose original and current bytes are equal (including
  files created and then deleted again inside the workspace).
- ``to_workspace`` ignores a deletion of a path the session does not contain (there is
  nothing to delete), so such an entry produces no diff.
- JSON form is ``{"files": {path: base64-string | None}}``.

Diffs are git-style and byte-faithful (``Workspace.diff`` in core is not: it splits with
``str.splitlines()``, drops line terminators and the no-newline marker, so its output does
not apply). :meth:`Edit.per_file_diff` builds them here, against the pristine bytes in
``session.files``:

- Each file diff starts with ``diff --git a/<path> b/<path>``, then ``new file mode
  100644`` / ``deleted file mode 100644`` for a creation / deletion (the mode is not
  tracked by an Edit; 100644 is assumed), then ``--- a/<path>`` (``--- /dev/null`` for a
  creation) and ``+++ b/<path>`` (``+++ /dev/null`` for a deletion), then the hunks.
  Creating or deleting an empty file has no hunks and no ``---``/``+++`` lines, as in git.
- Content is decoded as UTF-8 with ``surrogateescape`` (invalid bytes round-trip; encode
  the patch with ``surrogateescape`` before writing it to disk). Lines are split on
  ``\n`` only, keeping terminators, so ``\r\n`` and lone ``\r`` are preserved byte for
  byte (``str.splitlines`` would also split on ``\r``, form feeds, ``\x1c``..., which git
  does not). Hunks come from ``difflib.unified_diff`` with 3 lines of context.
- A line without a terminator (the last line of a file that does not end in ``\n``) is
  followed by ``\\ No newline at end of file``, so adding or removing only a final newline
  yields a non-empty diff.
- Every file diff ends with ``\n``. :meth:`Edit.git_patch` (and :meth:`Edit.unified_diff`,
  its alias) joins them with ``"\n"``; the blank separator line is ignored by
  ``git apply``. Binary content is diffed as text (no ``GIT binary patch``), and paths are
  written unquoted.
- An entry whose bytes equal the pristine bytes, or a deletion of a path the session does
  not contain, produces no diff (consistent with ``to_workspace``).
"""
from __future__ import annotations

import base64
import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent2.core.workspace import Workspace

if TYPE_CHECKING:
    from agent2.core.session import Session


@dataclass
class Edit:
    """A complete proposed change: path -> new content, ``None`` for a deleted file."""

    files: dict[str, bytes | None] = field(default_factory=dict)

    @staticmethod
    def from_workspace(workspace: Workspace) -> Edit:
        """Capture every file in ``workspace`` whose current content differs from its original.

        Created files (no original state) are included with their content; deleted files
        map to ``None``; unchanged checkouts are skipped.
        """
        files: dict[str, bytes | None] = {}
        for path, file in sorted(workspace.files.items()):
            before = file.original_state.bytes if file.original_state is not None else None
            after = file.current_state.bytes if file.current_state is not None else None
            if before == after:
                continue
            files[path] = after
        return Edit(files)

    def to_workspace(self, session: Session) -> Workspace:
        """Materialise this edit on a fresh ``Workspace(session)``."""
        workspace = Workspace(session)
        for path, data in sorted(self.files.items()):
            if data is None:
                workspace.delete_file(path)
            else:
                workspace.write_file(path, data)
        return workspace

    def per_file_diff(self, session: Session) -> list[tuple[str, str]]:
        """One ``(path, git_diff)`` per changed file, sorted by path (format in module doc)."""
        diffs: list[tuple[str, str]] = []
        for path, after in sorted(self.files.items()):
            pristine = session.files.get(path)
            state = pristine.current_state if pristine is not None else None
            before = state.bytes if state is not None else None
            if before == after:
                continue  # unchanged, or deleting a path that does not exist
            diffs.append((path, file_diff(path, before, after)))
        return diffs

    def git_patch(self, session: Session) -> str:
        """Every per-file diff joined with ``"\n"``: a patch ``git apply`` accepts."""
        return "\n".join(content for _, content in self.per_file_diff(session))

    def unified_diff(self, session: Session) -> str:
        """Same as :meth:`git_patch` (kept as the contract name used by GraphResult.diff)."""
        return self.git_patch(session)

    def is_empty(self) -> bool:
        """True when the edit changes nothing."""
        return not self.files

    def touched_paths(self) -> set[str]:
        """Paths created, modified or deleted by this edit."""
        return set(self.files)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable form: bytes as base64 strings, deletions as ``None``."""
        return {
            "files": {
                path: None if data is None else base64.b64encode(data).decode("ascii")
                for path, data in sorted(self.files.items())
            }
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> Edit:
        """Inverse of :meth:`to_json`."""
        return Edit({
            path: None if encoded is None else base64.b64decode(encoded)
            for path, encoded in data.get("files", {}).items()
        })


NO_NEWLINE_MARKER = "\\ No newline at end of file"
"""Marker line git emits after a diff line that has no terminator."""

NEW_FILE_MODE = "100644"
"""File mode written for creations and deletions (an Edit does not track modes)."""


def _git_lines(text: str) -> list[str]:
    """Split ``text`` into lines on ``\\n`` only, keeping the terminators."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def file_diff(path: str, before: bytes | None, after: bytes | None) -> str:
    """Git-style diff of one file from ``before`` to ``after`` (``None`` = absent).

    See the module docstring for the exact format. The result always ends with ``\\n``.
    """
    out = [f"diff --git a/{path} b/{path}\n"]
    if before is None:
        out.append(f"new file mode {NEW_FILE_MODE}\n")
    elif after is None:
        out.append(f"deleted file mode {NEW_FILE_MODE}\n")
    old = _git_lines((before or b"").decode("utf-8", "surrogateescape"))
    new = _git_lines((after or b"").decode("utf-8", "surrogateescape"))
    hunks = list(difflib.unified_diff(old, new, n=3, lineterm="\n"))[2:]  # drop ---/+++
    if hunks:
        out.append(f"--- a/{path}\n" if before is not None else "--- /dev/null\n")
        out.append(f"+++ b/{path}\n" if after is not None else "+++ /dev/null\n")
        for line in hunks:
            out.append(line)
            if not line.endswith("\n"):
                out.append(f"\n{NO_NEWLINE_MARKER}\n")
    return "".join(out)
