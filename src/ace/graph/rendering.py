"""Render canonical views (``graph/views.py``) to text. See GRAPH_NEAT_DESIGN.md §3.1.

Rendering is line based: every contribution of a :class:`FileView` (the file skeleton, a
FULL element, a COLLAPSED element, the headers of ancestors of listed elements) is turned
into a set of 1-indexed lines to show, and the union of those sets is printed. Runs of
lines that are not shown become a single ``... [BODY HIDDEN: N LINES] ...`` marker, so
contributions compose exactly as views merge (the composition rule of §3: a collapsed file
plus one FULL element renders the skeleton with that element expanded in place).

Conventions:

- Section headers::

      --- path ---                 file rendered in full or as a skeleton (levels[""] present)
      --- path::Element.path ---   one listed element when levels[""] is absent
      --- path (lines a-b) ---     a raw line range not already shown

- Shown lines carry the ``N| `` prefix (when ``prefix_lines``). Hidden markers carry no
  prefix and are indented like the first non-blank hidden line. A hidden run made only of
  blank lines is printed as-is instead of a marker.
- Line sets per element ``n`` (``head`` = signature lines incl. decorators, plus docstring):
  FULL -> every line of ``n``; COLLAPSED -> ``head(n)`` plus the signature lines of each
  direct child; any listed element also adds ``head(a)`` for every ancestor ``a``.
  The skeleton (``levels[""] == COLLAPSED``) shows every line outside top-level elements
  plus ``head(t)`` of each top-level element ``t``.
- An element key stands for every element with that path (a property getter and setter,
  ``if``/``else`` definitions): all of them are shown, and the skeleton hides each one's
  body. Ancestors are resolved by containment.
- Without ``""``, listed elements are grouped under their outermost listed ancestor; each
  group is one ``--- path::element ---`` section spanning that element's lines (ordered by
  start line). Ancestor headers outside the section are not shown there.
- Line-range sections come after element sections; lines already shown in full are
  subtracted, and each remaining contiguous span becomes its own section. Ranges are
  clamped to the file length.
- Unparsed files: ``""`` at any level renders the whole file; otherwise only line ranges.
  Element keys on unparsed files and unknown element keys are ignored.
- ``render_view`` renders files sorted by path, skips files absent from the session and
  empty renderings, and joins sections with a blank line.

The depth-based helpers in ``code_parser/interface/renderer.py`` work on byte slices of
single nodes and cannot express a union of selective expansions, so they are not reused;
the ``N| `` prefix style and the hidden-body marker text match them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ace.graph.views import FileView, Level, View, normalize_element_path

if TYPE_CHECKING:
    from agent2.code_parser.file import CodeFile, File
    from agent2.code_parser.objects import CodeNode
    from agent2.core.session import Session

__all__ = ["render_file_view", "render_view", "measure", "SPACER"]

SPACER = "| "


def measure(text: str) -> int:
    """Budget size of rendered text: its length in characters."""
    return len(text)


# --------------------------------------------------------------------------- helpers


def _split_lines(file: File | CodeFile) -> list[str]:
    """Decode the file's current state into lines (no trailing empty line)."""
    state = file.current_state
    if state is None or not state.bytes:
        return []
    text = state.bytes.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def _start_line(node: CodeNode) -> int:
    return node.full_block.start_point[0] + 1


def _end_line(node: CodeNode) -> int:
    row, col = node.full_block.end_point
    if col == 0 and row > node.full_block.start_point[0]:
        return row  # ends at the start of a line: that line is not part of the node
    return row + 1


def _full_lines(node: CodeNode) -> set[int]:
    return set(range(_start_line(node), _end_line(node) + 1))


def _body_on_signature_line(node: CodeNode, text: list[str] | None) -> bool:
    """True when the body starts on a line that also holds the end of the signature."""
    if text is None or node.body_block is None:
        return False
    row, column = node.body_block.start_point
    if not 0 <= row < len(text):
        return False
    return bool(text[row].encode("utf-8")[:column].strip())


def _signature_lines(node: CodeNode, text: list[str] | None = None) -> set[int]:
    """Decorators + ``def``/``class`` line(s), up to the line before the body starts.

    With the file's lines (``text``), a body that starts on the signature's last line
    (``def f(a,`` / ``b): return a``) keeps that line in the signature.
    """
    start = _start_line(node)
    if node.body_block is None:
        return _full_lines(node)
    body_row = node.body_block.start_point[0]  # 0-indexed row == 1-indexed previous line
    end = max(start, body_row)
    if _body_on_signature_line(node, text):
        end = max(end, body_row + 1)
    return set(range(start, end + 1))


def _head_lines(node: CodeNode, text: list[str] | None = None) -> set[int]:
    """Signature lines plus the docstring lines."""
    lines = _signature_lines(node, text)
    if node.doc_block is not None:
        lines |= set(range(node.doc_block.start_point[0] + 1, node.doc_block.end_point[0] + 2))
    # Fill the gap between signature and docstring (normally none).
    if lines:
        lines = set(range(min(lines), max(lines) + 1))
    return lines


def _collapsed_lines(node: CodeNode, text: list[str] | None = None) -> set[int]:
    lines = _head_lines(node, text)
    for child in node.children:
        lines |= _signature_lines(child, text)
    return lines


def _unique_nodes(file: CodeFile) -> list[CodeNode]:
    """Every distinct code node, by start byte.

    ``code_nodes`` indexes each node under ``path`` and ``llm_path``; deduping by identity
    (path + start byte) keeps elements that share a path (``if``/``else`` definitions,
    property getter and setter), whose ``path`` key only holds the last one.
    """
    seen: dict[tuple[str, int], CodeNode] = {}
    for node in file.code_nodes.values():
        seen.setdefault((node.path, node.full_block.start_byte), node)
    return sorted(seen.values(), key=lambda n: n.full_block.start_byte)


def _ancestors(
    file: CodeFile, node: CodeNode, nodes: list[CodeNode] | None = None
) -> list[CodeNode]:
    """Enclosing elements, innermost first (resolved by containment, not by name only)."""

    def contains(outer: CodeNode, inner: CodeNode) -> bool:
        return (
            outer is not inner
            and outer.full_block.start_byte <= inner.full_block.start_byte
            and inner.full_block.end_byte <= outer.full_block.end_byte
        )

    result: list[CodeNode] = []
    current = node
    parent_path = node.parent_path
    while parent_path:
        parent = file.code_nodes.get(parent_path)
        if parent is None or not contains(parent, current):
            # code_nodes keeps the last element of a shared path: find the enclosing one.
            if nodes is None:
                nodes = _unique_nodes(file)
            enclosing = [n for n in nodes if n.path == parent_path and contains(n, current)]
            if enclosing:
                parent = max(enclosing, key=lambda n: n.full_block.start_byte)
        if parent is None:
            break
        result.append(parent)
        current = parent
        parent_path = parent.parent_path
    return result


def _element_lines(
    file: CodeFile,
    node: CodeNode,
    level: Level,
    text: list[str] | None = None,
    nodes: list[CodeNode] | None = None,
) -> set[int]:
    lines = _full_lines(node) if level >= Level.FULL else _collapsed_lines(node, text)
    for ancestor in _ancestors(file, node, nodes):
        lines |= _head_lines(ancestor, text)
    return lines


def _format_line(number: int, text: str, prefix_lines: bool) -> str:
    return f"{number}{SPACER}{text}" if prefix_lines else text


def _render_span(
    lines: list[str], shown: set[int], first: int, last: int, prefix_lines: bool
) -> list[str]:
    """Render lines ``first..last``: shown lines verbatim, hidden runs as markers."""
    out: list[str] = []
    number = first
    while number <= last:
        if number in shown:
            out.append(_format_line(number, lines[number - 1], prefix_lines))
            number += 1
            continue
        run_end = number
        while run_end + 1 <= last and run_end + 1 not in shown:
            run_end += 1
        hidden = lines[number - 1 : run_end]
        non_blank = [text for text in hidden if text.strip()]
        if not non_blank:
            out.extend(
                _format_line(number + offset, text, prefix_lines)
                for offset, text in enumerate(hidden)
            )
        else:
            first_text = non_blank[0]
            indent = first_text[: len(first_text) - len(first_text.lstrip())]
            out.append(f"{indent}... [BODY HIDDEN: {len(hidden)} LINES] ...")
        number = run_end + 1
    return out


def _section(header: str, body: list[str]) -> str:
    return "\n".join([header, *body])


def _line_range_sections(
    path: str,
    lines: list[str],
    ranges: list[tuple[int, int]],
    shown: set[int],
    prefix_lines: bool,
) -> list[str]:
    sections: list[str] = []
    total = len(lines)
    for start, end in ranges:
        end = min(end, total)
        number = max(1, start)
        while number <= end:
            if number in shown:
                number += 1
                continue
            span_end = number
            while span_end + 1 <= end and span_end + 1 not in shown:
                span_end += 1
            body = [
                _format_line(n, lines[n - 1], prefix_lines) for n in range(number, span_end + 1)
            ]
            sections.append(_section(f"--- {path} (lines {number}-{span_end}) ---", body))
            number = span_end + 1
    return sections


# --------------------------------------------------------------------------- public API


def render_file_view(file: File | CodeFile, view: FileView, *, prefix_lines: bool = True) -> str:
    """Render one file's view.

    Args:
        file: The (pristine) file the view refers to; its ``current_state`` is rendered.
        view: The file's view.
        prefix_lines: Prefix shown lines with ``N| ``.

    Returns:
        The rendered sections joined by blank lines, or ``""`` if nothing is shown.
    """
    path = file.path
    lines = _split_lines(file)
    total = len(lines)
    all_lines = set(range(1, total + 1))
    file_level = view.levels.get("")
    sections: list[str] = []
    shown: set[int] = set()

    code_nodes = getattr(file, "code_nodes", None)
    parsed = code_nodes is not None

    if file_level == Level.FULL or (not parsed and file_level is not None):
        shown = all_lines
        sections.append(_section(f"--- {path} ---", _render_span(lines, shown, 1, total, prefix_lines)))
    elif parsed:
        nodes = _unique_nodes(file)  # type: ignore[arg-type]
        # An element key denotes every element with that path (all of them are shown).
        listed: dict[str, tuple[list[CodeNode], Level]] = {}
        for element, level in view.levels.items():
            if element == "":
                continue
            canonical = normalize_element_path(file, element)
            if canonical is None:
                continue
            previous = listed.get(canonical)
            if previous is None or level > previous[1]:
                same = [n for n in nodes if n.path == canonical] or [code_nodes[canonical]]
                listed[canonical] = (same, Level(level))

        if file_level is not None:  # skeleton
            inside_top = set()
            for node in nodes:
                if node.parent_path is None:
                    inside_top |= _full_lines(node)
                    shown |= _head_lines(node, lines)
            shown |= all_lines - inside_top
            for same, level in listed.values():
                for node in same:
                    shown |= _element_lines(file, node, level, lines, nodes)  # type: ignore[arg-type]
            shown &= all_lines
            sections.append(
                _section(f"--- {path} ---", _render_span(lines, shown, 1, total, prefix_lines))
            )
        else:  # element-only sections grouped by outermost listed ancestor
            groups: dict[tuple[str, int], list[tuple[CodeNode, Level]]] = {}
            roots: dict[tuple[str, int], CodeNode] = {}
            for same, level in listed.values():
                for node in same:
                    root = node
                    for ancestor in _ancestors(file, node, nodes):  # type: ignore[arg-type]
                        if ancestor.path in listed:
                            root = ancestor
                    key = (root.path, root.full_block.start_byte)
                    roots[key] = root
                    groups.setdefault(key, []).append((node, level))
            for key in sorted(roots, key=lambda k: k[1]):
                root = roots[key]
                first, last = _start_line(root), min(_end_line(root), total)
                group_shown: set[int] = set()
                for node, level in groups[key]:
                    group_shown |= _element_lines(file, node, level, lines, nodes)  # type: ignore[arg-type]
                group_shown &= set(range(first, last + 1))
                shown |= group_shown
                sections.append(
                    _section(
                        f"--- {path}::{root.path} ---",
                        _render_span(lines, group_shown, first, last, prefix_lines),
                    )
                )

    sections.extend(_line_range_sections(path, lines, view.line_ranges, shown, prefix_lines))
    return "\n\n".join(sections)


def render_view(session: Session, view: View, *, prefix_lines: bool = True) -> str:
    """Render a whole view against the (pristine) session, files sorted by path.

    Files missing from the session and files whose rendering is empty are skipped.
    """
    parts: list[str] = []
    for path in sorted(view.files):
        file = session.files.get(path)
        if file is None:
            continue
        text = render_file_view(file, view.files[path], prefix_lines=prefix_lines)
        if text:
            parts.append(text)
    return "\n\n".join(parts)
