"""ACE: the code-block editing format used by tool-less editing workers.

See GRAPH_NEAT_DESIGN.md section 4.1. A model message contains markdown code blocks; the
first line inside each block is a target header and the rest is the body.

Header grammar (after the literal ``@@ `` prefix; surrounding whitespace stripped)::

    path                       FILE     replace the whole file
    path::NEW                  NEW      create a file that does not exist yet
    path::DELETE               DELETE   delete the file (body ignored)
    path::A-B                  LINES    replace 1-indexed lines A..B inclusive (A <= B);
                                        a single number ``path::A`` means ``A-A``
    path::Elem.path            ELEMENT  replace the element (decorators, signature, body)
    path::Elem.path::AFTER     AFTER    insert the body as a new sibling after the element
    path::Elem.path::BEFORE    BEFORE   insert the body as a new sibling before the element

``path`` is project-relative (a leading ``./`` is dropped). Keywords are upper case.
Element paths may be ``CodeNode.path`` (``Item.slug``) or ``llm_path`` (``Item.slug.14``);
a stale line suffix that does not resolve falls back to the ``path`` form. A ``path`` shared
by several elements (property getter/setter, ``if``/``else`` definitions) is ambiguous and
raises :class:`AceError` naming the ``llm_path`` choices. Backslashes in the file path are
read as ``/``.

Fences: a fence line is three or more backticks (optionally indented, optionally followed
by a language tag); it is closed by a line of at least as many backticks and nothing else.
As in CommonMark, content lines lose up to the opening fence's indentation (fences nested in
list items work).
Blocks whose first line does not start with ``@@ `` are ignored, and so are quoted
unified-diff hunks (``@@ -12,6 +12,7 @@``). A message with zero ACE
blocks raises :class:`AceError`, as does an unclosed fence whose first line is a header.

Application (everything resolved against the pristine ``session.files``):

- ELEMENT replaces the bytes from the start of the element's first line (decorators
  included) to the element's end byte with the body re-indented by
  ``adapter.attempt_fix_formatting`` to the element's indentation; whatever followed the
  element on its last line is kept.
- AFTER inserts ``"\\n" + blank lines + body`` at the end of the element's last line
  (before its newline), BEFORE inserts ``body + blank lines`` at the start of the
  element's first line. The body is indented like the target element, so inserting after
  the last method of a class keeps the new method inside the class. Separation is one
  blank line for nested elements and two blank lines for top-level elements (PEP 8).
- LINES replaces whole lines A..B; if the replaced span ended with a newline and the body
  does not, one is appended. An empty body deletes the lines.
- FILE/NEW write the body verbatim; DELETE deletes the file.
- In a file whose line breaks are mostly ``\\r\\n``, inserted/replaced text of the other
  targets uses ``\\r\\n`` too. A body whose first line already has the element's
  indentation but contains less-indented lines (column-0 string continuations) is kept as
  given instead of being re-indented (which would indent it twice).

Conflicts: byte ranges in one file must not overlap (half-open ranges; a zero-width
insertion conflicts with a range strictly containing its point, with another insertion at
the same point, and with a replacement that strictly contains its anchor element).
FILE/NEW/DELETE must be the only block for their path. Per file, edits are applied
bottom-up by (start byte, end byte) descending on a fresh ``Workspace``.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from agent2.code_parser.objects import CodeBlock, CodeNode, CodeState, FileEdit
from agent2.core.workspace import Workspace
from ace.graph.edits import Edit
from ace.graph.types import GraphFailure

if TYPE_CHECKING:
    from agent2.code_parser.file import File
    from agent2.core.session import Session


class AceError(GraphFailure):
    """An ACE message could not be parsed or applied."""


class AceFormat(str, Enum):
    """Which ACE headers a worker may emit (property ``ace_format``)."""

    ELEMENT = "element"
    ELEMENT_FILE = "element_file"
    ELEMENT_FILE_LINES = "element_file_lines"
    FILE_LINES = "file_lines"


class AceTarget(str, Enum):
    """What an ACE block targets."""

    ELEMENT = "element"
    FILE = "file"
    NEW = "new"
    DELETE = "delete"
    LINES = "lines"
    AFTER = "after"
    BEFORE = "before"


_ELEMENT_TARGETS = frozenset({AceTarget.ELEMENT, AceTarget.AFTER, AceTarget.BEFORE})
_FILE_TARGETS = frozenset({AceTarget.FILE, AceTarget.NEW, AceTarget.DELETE})

PERMITTED: dict[AceFormat, frozenset[AceTarget]] = {
    AceFormat.ELEMENT: frozenset({
        AceTarget.ELEMENT, AceTarget.NEW, AceTarget.DELETE, AceTarget.AFTER, AceTarget.BEFORE,
    }),
    AceFormat.ELEMENT_FILE: frozenset({
        AceTarget.ELEMENT, AceTarget.NEW, AceTarget.DELETE, AceTarget.AFTER, AceTarget.BEFORE,
        AceTarget.FILE,
    }),
    AceFormat.ELEMENT_FILE_LINES: frozenset({
        AceTarget.ELEMENT, AceTarget.NEW, AceTarget.DELETE, AceTarget.AFTER, AceTarget.BEFORE,
        AceTarget.FILE, AceTarget.LINES,
    }),
    AceFormat.FILE_LINES: frozenset({
        AceTarget.FILE, AceTarget.NEW, AceTarget.DELETE, AceTarget.LINES,
    }),
}
"""Headers permitted by each format (GRAPH_NEAT_DESIGN.md 4.1 table)."""


@dataclass
class AceBlock:
    """One parsed ACE code block."""

    path: str
    target: AceTarget
    element: str | None
    lines: tuple[int, int] | None
    body: str

    def header(self) -> str:
        """Render the block's header (without the ``@@ `` prefix)."""
        match self.target:
            case AceTarget.FILE:
                return self.path
            case AceTarget.NEW | AceTarget.DELETE:
                return f"{self.path}::{self.target.name}"
            case AceTarget.LINES:
                assert self.lines is not None
                return f"{self.path}::{self.lines[0]}-{self.lines[1]}"
            case AceTarget.ELEMENT:
                return f"{self.path}::{self.element}"
            case _:
                return f"{self.path}::{self.element}::{self.target.name}"


# --------------------------------------------------------------------------- parsing

_FENCE_RE = re.compile(r"^\s*(`{3,})(.*)$")
_LINES_RE = re.compile(r"^(\d+)\s*(?:-\s*(\d+))?$")
_ELEMENT_RE = re.compile(r"^[A-Za-z_][\w.]*$")
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def _iter_fenced_blocks(text: str) -> list[tuple[list[str], bool]]:
    """Return ``(lines, closed)`` for every fenced code block in ``text``."""
    blocks: list[tuple[list[str], bool]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        opening = _FENCE_RE.match(lines[i])
        if not opening:
            i += 1
            continue
        fence = opening.group(1)
        indent = len(lines[i]) - len(lines[i].lstrip(" "))  # e.g. a fence inside a list item
        content: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith(fence) and set(stripped) == {"`"}:
                closed = True
                i += 1
                break
            line = lines[i]
            # CommonMark: drop up to the fence's indentation from every content line.
            content.append(line[min(indent, len(line) - len(line.lstrip(" "))):])
            i += 1
        blocks.append((content, closed))
    return blocks


def _parse_header(header: str) -> tuple[str, AceTarget, str | None, tuple[int, int] | None]:
    raw = header
    header = header.strip()
    parts = [part.strip() for part in header.split("::")]
    path = parts[0].replace("\\", "/")  # tolerate Windows-style separators
    if path.startswith("./"):
        path = path[2:]
    if not path or any(not part for part in parts):
        raise AceError(f"Malformed ACE header '@@ {raw}'.")
    if len(parts) == 1:
        return path, AceTarget.FILE, None, None
    if len(parts) == 2:
        spec = parts[1]
        if spec == "NEW":
            return path, AceTarget.NEW, None, None
        if spec == "DELETE":
            return path, AceTarget.DELETE, None, None
        span = _LINES_RE.match(spec)
        if span:
            start = int(span.group(1))
            end = int(span.group(2)) if span.group(2) else start
            if start < 1 or end < start:
                raise AceError(f"Invalid line span in ACE header '@@ {raw}'.")
            return path, AceTarget.LINES, None, (start, end)
        if not _ELEMENT_RE.match(spec):
            raise AceError(f"Malformed element path in ACE header '@@ {raw}'.")
        return path, AceTarget.ELEMENT, spec, None
    if len(parts) == 3 and parts[2] in ("AFTER", "BEFORE") and _ELEMENT_RE.match(parts[1]):
        target = AceTarget.AFTER if parts[2] == "AFTER" else AceTarget.BEFORE
        return path, target, parts[1], None
    raise AceError(f"Malformed ACE header '@@ {raw}'.")


def parse_ace(text: str, fmt: AceFormat) -> list[AceBlock]:
    """Parse every ACE block in a model message.

    Raises:
        AceError: a header is malformed or not permitted by ``fmt``, a header block is not
            closed, or the message contains no ACE block at all.
    """
    fmt = AceFormat(fmt)
    allowed = PERMITTED[fmt]
    result: list[AceBlock] = []
    for content, closed in _iter_fenced_blocks(text):
        if not content or not content[0].startswith("@@ "):
            continue
        if _DIFF_HUNK_RE.match(content[0]):
            continue  # a quoted unified-diff hunk, not an ACE header
        header = content[0][3:]
        if not closed:
            raise AceError(f"Unclosed code block for ACE header '@@ {header.strip()}'.")
        path, target, element, lines = _parse_header(header)
        if target not in allowed:
            raise AceError(
                f"ACE header '@@ {header.strip()}' ({target.value}) is not permitted "
                f"in format '{fmt.value}'."
            )
        body_lines = content[1:]
        body = "\n".join(body_lines) + "\n" if body_lines else ""
        result.append(AceBlock(path, target, element, lines, body))
    if not result:
        raise AceError("No ACE blocks found: every edit must be a code block whose first line is an '@@ ' header.")
    return result


# --------------------------------------------------------------------------- application

@dataclass
class _Planned:
    """A resolved byte-range edit on one pristine file."""

    block: AceBlock
    start: int
    end: int
    text: bytes
    anchor: tuple[int, int] | None = None   # for insertions: the target element's range


def _point(state: CodeState, offset: int) -> tuple[int, int]:
    row = bisect_right(state._line_starts, offset) - 1
    return row, offset - state._line_starts[row]


def _line_count(state: CodeState) -> int:
    if not state.bytes:
        return 0
    return len(state._line_starts) - (1 if state.bytes.endswith(b"\n") else 0)


def _resolve_node(file: File, block: AceBlock) -> CodeNode:
    nodes: dict[str, CodeNode] | None = getattr(file, "code_nodes", None)
    if nodes is None:
        raise AceError(f"'{block.path}' is not a parsed code file; element header '@@ {block.header()}' cannot apply.")
    element = block.element or ""
    node = nodes.get(element)
    if node is None:
        stripped = re.sub(r"\.\d+$", "", element)
        node = nodes.get(stripped)
    if node is None:
        raise AceError(f"Unknown element '{element}' in '{block.path}' (header '@@ {block.header()}').")
    if node.llm_path != element:
        # Addressed by ``CodeNode.path``: code_nodes keeps only the last definition of a
        # path, so refuse to guess when several elements share it (property getter and
        # setter, ``if``/``else`` definitions, ...). The ``llm_path`` form disambiguates.
        same = {n.full_block.start_byte: n for n in nodes.values() if n.path == node.path}
        if len(same) > 1:
            choices = ", ".join(f"'{same[k].llm_path}'" for k in sorted(same))
            raise AceError(
                f"Ambiguous element '{element}' in '{block.path}' (header '@@ {block.header()}'): "
                f"{len(same)} elements share the path '{node.path}'. Append the element's first "
                f"line number to choose one: {choices}."
            )
    return node


def _element_range(state: CodeState, block: CodeBlock) -> int:
    """Start byte of the element's first line if only whitespace precedes it, else its start."""
    line_start = state._line_starts[block.start_point[0]]
    if state.bytes[line_start:block.start_byte].strip() == b"":
        return line_start
    return block.start_byte


def _format_body(adapter: object, body: str, full: CodeBlock, state: CodeState) -> str:
    """Re-indent an element body to ``full``'s indentation.

    ``attempt_fix_formatting`` dedents by the common prefix of *all* lines and then
    prefixes every line. A body that already sits at the target indentation but has lines
    with less indentation (continuation lines of a multi-line string at column 0) would be
    indented twice and change the literal, so such a body is used as given.
    """
    indent = state.bytes[state._line_starts[full.start_point[0]]:full.start_byte]
    if indent and not indent.strip():
        prefix = indent.decode("utf-8", errors="replace")
        lines = body.strip("\n").split("\n")
        first = next((line for line in lines if line.strip()), "")
        if (
            first.startswith(prefix)
            and not first[len(prefix):][:1].isspace()
            and any(line.strip() and not line.startswith(prefix) for line in lines)
        ):
            return "\n".join(line.rstrip("\r") if line.strip() else "" for line in lines) + "\n"
    return adapter.attempt_fix_formatting(body, full, state)  # type: ignore[attr-defined]


def _uses_crlf(state: CodeState) -> bool:
    """True when most line breaks of the file are ``\\r\\n``."""
    crlf = state.bytes.count(b"\r\n")
    return crlf > 0 and 2 * crlf >= state.bytes.count(b"\n")


def _plan_block(block: AceBlock, file: File) -> _Planned:
    planned = _plan_block_lf(block, file)
    state = file.current_state
    assert state is not None
    if _uses_crlf(state):  # keep the file's line endings for inserted/replaced text
        planned.text = planned.text.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return planned


def _plan_block_lf(block: AceBlock, file: File) -> _Planned:
    state = file.current_state
    assert state is not None
    if block.target is AceTarget.LINES:
        assert block.lines is not None
        first, last = block.lines
        count = _line_count(state)
        if last > count:
            raise AceError(f"Line span {first}-{last} is outside '{block.path}' ({count} lines).")
        start = state.get_line_byte_range(first)[0]
        end = state.get_line_byte_range(last)[1]
        text = block.body.encode("utf-8")
        if text and state.bytes[start:end].endswith(b"\n") and not text.endswith(b"\n"):
            text += b"\n"
        return _Planned(block, start, end, text)

    node = _resolve_node(file, block)
    adapter = getattr(file, "adapter")
    full = node.full_block
    formatted = _format_body(adapter, block.body, full, state)
    line_start = _element_range(state, full)
    if block.target is AceTarget.ELEMENT:
        return _Planned(block, line_start, full.end_byte, formatted.rstrip("\n").encode("utf-8"))

    blank = "\n" if full.start_point[1] > 0 else "\n\n"   # nested: 1 blank line, top-level: 2
    anchor = (line_start, full.end_byte)
    if block.target is AceTarget.AFTER:
        end_row = full.end_point[0]
        _, row_end = state.get_line_byte_range(end_row + 1)
        point = row_end - 1 if state.bytes[row_end - 1:row_end] == b"\n" else row_end
        if point > full.end_byte and state.bytes[point - 1:point] == b"\r":
            point -= 1  # CRLF: insert before the whole line break
        point = max(point, full.end_byte)
        text = "\n" + blank + formatted.rstrip("\n")
        return _Planned(block, point, point, text.encode("utf-8"), anchor)
    text = formatted.rstrip("\n") + "\n" + blank
    return _Planned(block, line_start, line_start, text.encode("utf-8"), anchor)


def _conflict(a: _Planned, b: _Planned) -> bool:
    a_point, b_point = a.start == a.end, b.start == b.end
    if a_point and b_point:
        return a.start == b.start
    if a_point or b_point:
        point, span = (a, b) if a_point else (b, a)
        if span.start < point.start < span.end:
            return True
        if point.anchor is not None:
            lo, hi = point.anchor
            if span.start <= lo and hi <= span.end and (span.start, span.end) != (lo, hi):
                return True
        return False
    return max(a.start, b.start) < min(a.end, b.end)


def apply_ace(blocks: list[AceBlock], session: Session, fmt: AceFormat) -> Edit:
    """Apply ACE blocks against the pristine session and return the resulting :class:`Edit`.

    Raises:
        AceError: a header is not permitted by ``fmt``, a path or element is unknown, a
            NEW path already exists, blocks overlap, or a FILE/NEW/DELETE block shares its
            path with another block.
    """
    fmt = AceFormat(fmt)
    allowed = PERMITTED[fmt]
    by_path: dict[str, list[AceBlock]] = {}
    for block in blocks:
        if block.target not in allowed:
            raise AceError(f"ACE header '@@ {block.header()}' is not permitted in format '{fmt.value}'.")
        by_path.setdefault(block.path, []).append(block)

    workspace = Workspace(session)
    for path, path_blocks in by_path.items():
        whole = [b for b in path_blocks if b.target in _FILE_TARGETS]
        if whole and len(path_blocks) > 1:
            headers = ", ".join(f"'@@ {b.header()}'" for b in path_blocks)
            raise AceError(f"Conflicting ACE blocks for '{path}': {headers}. A whole-file, NEW or DELETE block must be the only block for its path.")
        pristine = session.files.get(path)
        if whole:
            block = whole[0]
            if block.target is AceTarget.NEW:
                if pristine is not None:
                    raise AceError(f"'@@ {block.header()}': '{path}' already exists.")
                workspace.write_file(path, block.body.encode("utf-8"))
            elif pristine is None or pristine.current_state is None:
                raise AceError(f"'@@ {block.header()}': unknown path '{path}'.")
            elif block.target is AceTarget.DELETE:
                workspace.delete_file(path)
            else:
                workspace.write_file(path, block.body.encode("utf-8"))
            continue

        if pristine is None or pristine.current_state is None:
            raise AceError(f"'@@ {path_blocks[0].header()}': unknown path '{path}'.")
        planned = [_plan_block(block, pristine) for block in path_blocks]
        for i, a in enumerate(planned):
            for b in planned[i + 1:]:
                if _conflict(a, b):
                    raise AceError(
                        f"Overlapping ACE blocks in '{path}': '@@ {a.block.header()}' and "
                        f"'@@ {b.block.header()}'. Each code block must be one disjoint edit."
                    )
        state = pristine.current_state
        for item in sorted(planned, key=lambda p: (p.start, p.end), reverse=True):
            workspace.apply_edit(path, FileEdit(
                item.start,
                item.end,
                _point(state, item.start),
                _point(state, item.end),
                item.text,
            ))
    return Edit.from_workspace(workspace)


# --------------------------------------------------------------------------- instructions

_HEADER_DOCS: list[tuple[AceTarget, str, str]] = [
    (AceTarget.ELEMENT, "@@ path/to/file.py::Class.method",
     "replace that element (function, method or class). The body replaces the entire element, "
     "including its decorators and signature."),
    (AceTarget.AFTER, "@@ path/to/file.py::Class.method::AFTER",
     "insert the body as a new sibling element directly after that element."),
    (AceTarget.BEFORE, "@@ path/to/file.py::Class.method::BEFORE",
     "insert the body as a new sibling element directly before that element."),
    (AceTarget.NEW, "@@ path/to/new_file.py::NEW", "create a new file whose content is the body."),
    (AceTarget.DELETE, "@@ path/to/file.py::DELETE", "delete the file (the body is ignored and may be empty)."),
    (AceTarget.FILE, "@@ path/to/file.py", "replace the whole file with the body."),
    (AceTarget.LINES, "@@ path/to/file.py::12-40",
     "replace lines 12 to 40 inclusive (line numbers of the original file) with the body."),
]


def ace_instructions(fmt: AceFormat) -> str:
    """Prompt text describing the ACE headers permitted by ``fmt``."""
    fmt = AceFormat(fmt)
    allowed = PERMITTED[fmt]
    element_based = AceTarget.ELEMENT in allowed
    lines = [
        "Write your changes as markdown code blocks. The first line inside each code block is a "
        "header starting with `@@ ` that names the target; every following line is the body.",
        "Each code block is one edit, and all edits must be disjoint: never target the same code "
        "twice or an element together with something inside it. Paths are relative to the project "
        "root, and every target refers to the original files you were shown, not to your other edits.",
        "",
        "Permitted headers:",
    ]
    for target, example, meaning in _HEADER_DOCS:
        if target in allowed:
            lines.append(f"- `{example}`: {meaning}")
    lines.append("")
    if element_based:
        lines.append(
            "Element paths are dotted names such as `Class.method` or `function`. The body of an "
            "element block replaces the entire element including its signature, so write the "
            "complete element. Indentation is fixed automatically."
        )
        lines.append(
            "If several elements share a dotted name (for example a property getter and its "
            "setter), append the element's first line number as shown in the code to pick one: "
            "`Class.method.12`."
        )
        if AceTarget.FILE in allowed or AceTarget.LINES in allowed:
            lines.append(
                "Prefer element headers. Whole-file and line-span headers are discouraged: use "
                "them only when no element header fits (for example top-level code outside any "
                "element, or files that are not code)."
            )
        example = (
            "```python\n"
            "@@ pkg/shapes.py::Circle.area\n"
            "def area(self):\n"
            "    return math.pi * self.radius ** 2\n"
            "```"
        )
    else:
        lines.append(
            "Prefer line-span headers for small changes; a whole-file block must contain the "
            "complete new file."
        )
        example = (
            "```python\n"
            "@@ pkg/shapes.py::12-13\n"
            "    def area(self):\n"
            "        return math.pi * self.radius ** 2\n"
            "```"
        )
    if AceTarget.FILE in allowed or AceTarget.LINES in allowed:
        lines.append("Whole-file and line-span bodies are used verbatim: keep the exact indentation.")
    whole = "A whole-file, NEW or DELETE" if AceTarget.FILE in allowed else "A NEW or DELETE"
    lines += [
        f"{whole} block must be the only block for its path.",
        "Code blocks without an `@@ ` header are ignored. If a body itself contains a line "
        "of three backticks, open and close its block with four backticks (````) instead.",
        "",
        "Example:",
        example,
    ]
    return "\n".join(lines)
