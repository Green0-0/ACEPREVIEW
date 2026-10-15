"""Canonical views: what a worker saw, at what detail, against the pristine repository.

See GRAPH_NEAT_DESIGN.md section 3.

Implementation choices where the contract leaves room:

- Element keys are always ``CodeNode.path`` (``Class.method``). ``View.add_element``
  strips a trailing all-digit segment (the ``llm_path`` line suffix, e.g.
  ``Class.method.42``) because Python identifiers can never be purely numeric; callers
  that hold the file should still prefer :func:`normalize_element_path`, which resolves
  either form through ``CodeFile.code_nodes``. Passing ``element == ""`` to
  ``add_element`` is the same as ``add_file``.
- ``add_lines`` is lenient: a reversed range is swapped, the start is clamped to 1 and a
  range ending before line 1 is ignored. Ranges are kept sorted and merged; adjacent
  ranges (``(1, 3)`` and ``(4, 6)``) coalesce as well as overlapping ones.
- ``counts()`` emits ``int(level)`` (FULL 2, COLLAPSED 1) for every level entry and adds
  1 to the file key ``(path, "")`` for every line range. Resolving line ranges to the
  elements they cover is the consumer's job (``CodebaseGraph.resolve_lines``).
- ``restrict_to`` also clamps line ranges to the pristine file length and drops file
  views that end up empty. On unparsed files only the ``""`` key survives.
- JSON shape: ``{"files": {path: {"levels": {element: int}, "line_ranges": [[a, b]]}}}``
  with keys sorted for determinism. ``from_json`` also accepts level names
  (``"FULL"``/``"COLLAPSED"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from agent2.code_parser.file import CodeFile, File
    from agent2.core.session import Session

__all__ = [
    "Level",
    "FileView",
    "View",
    "normalize_element_path",
    "merge_ranges",
    "file_line_count",
]


class Level(IntEnum):
    """Detail level at which a file or element was seen."""

    COLLAPSED = 1  # signature/skeleton seen (docstring + headers, bodies hidden)
    FULL = 2  # full source seen


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and coalesce 1-indexed inclusive line ranges.

    Overlapping and adjacent ranges are merged, e.g. ``[(4, 6), (1, 3), (10, 12)]`` becomes
    ``[(1, 6), (10, 12)]``.

    Args:
        ranges: Inclusive ``(start, end)`` pairs with ``start <= end``.

    Returns:
        A new sorted list of disjoint, non-adjacent ranges.
    """
    result: list[tuple[int, int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in ranges):
        if result and start <= result[-1][1] + 1:
            if end > result[-1][1]:
                result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def file_line_count(file: File | CodeFile) -> int:
    """Return the number of lines in a file's current state (0 for an empty file).

    A trailing newline does not start an extra line, matching the ``N| `` numbering used by
    the viewing tools and :mod:`ace.graph.rendering`.
    """
    state = file.current_state
    if state is None or not state.bytes:
        return 0
    data = state.bytes
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _strip_line_suffix(element: str) -> str:
    """Strip an ``llm_path`` line suffix (``A.b.42`` -> ``A.b``) if present."""
    element = element.strip()
    head, sep, tail = element.rpartition(".")
    if sep and head and tail.isdigit():
        return head
    return element


def normalize_element_path(file: File | CodeFile, element: str) -> str | None:
    """Resolve an element given as ``path`` or ``llm_path`` to its canonical ``path``.

    Args:
        file: The file the element lives in. Unparsed files have no elements.
        element: ``Class.method`` or ``Class.method.42``.

    Returns:
        The canonical ``CodeNode.path``, or ``None`` if the element does not exist (or the
        file is not parsed). ``""`` (the file itself) is returned unchanged.
    """
    element = element.strip()
    if element == "":
        return ""
    nodes = getattr(file, "code_nodes", None)
    if not nodes:
        return None
    node = nodes.get(element)
    if node is None:
        return None
    return node.path


@dataclass
class FileView:
    """What was seen of one file.

    Attributes:
        path: Project-relative POSIX path.
        levels: Element path -> level. The key ``""`` is the file itself.
        line_ranges: 1-indexed inclusive ranges, sorted and merged. Raw ranges for unparsed
            files or spans that do not align with an element.
    """

    path: str
    levels: dict[str, Level] = field(default_factory=dict)
    line_ranges: list[tuple[int, int]] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True when nothing was recorded for this file."""
        return not self.levels and not self.line_ranges

    def set_level(self, element: str, level: Level) -> None:
        """Raise ``levels[element]`` to at least ``level``."""
        level = Level(level)
        current = self.levels.get(element)
        if current is None or level > current:
            self.levels[element] = level

    def add_lines(self, start: int, end: int) -> None:
        """Add an inclusive line range (lenient; see module docstring)."""
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start
        if end < 1:
            return
        start = max(1, start)
        self.line_ranges = merge_ranges([*self.line_ranges, (start, end)])

    def merge(self, other: FileView) -> FileView:
        """Return a new FileView with levels maxed and ranges unioned."""
        result = self.copy()
        for element, level in other.levels.items():
            result.set_level(element, level)
        result.line_ranges = merge_ranges([*result.line_ranges, *other.line_ranges])
        return result

    def copy(self) -> FileView:
        """Return an independent copy."""
        return FileView(self.path, dict(self.levels), list(self.line_ranges))

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable form (without the path, which is the parent key)."""
        return {
            "levels": {element: int(self.levels[element]) for element in sorted(self.levels)},
            "line_ranges": [[start, end] for start, end in self.line_ranges],
        }

    @staticmethod
    def from_json(path: str, data: dict[str, Any]) -> FileView:
        """Inverse of :meth:`to_json`."""
        levels: dict[str, Level] = {}
        for element, raw in (data.get("levels") or {}).items():
            levels[element] = Level[raw] if isinstance(raw, str) else Level(int(raw))
        ranges = merge_ranges((int(a), int(b)) for a, b in (data.get("line_ranges") or []))
        return FileView(path, levels, ranges)


@dataclass
class View:
    """A canonical view: path -> FileView."""

    files: dict[str, FileView] = field(default_factory=dict)

    # ----------------------------------------------------------------- composition

    def merge(self, other: View) -> View:
        """Return a new View: levels take the max, line ranges are unioned and merged."""
        result = self.copy()
        for path, file_view in other.files.items():
            existing = result.files.get(path)
            result.files[path] = existing.merge(file_view) if existing else file_view.copy()
        return result

    @staticmethod
    def union(views: Iterable[View]) -> View:
        """Merge any number of views into a new View."""
        result = View()
        for view in views:
            result = result.merge(view)
        return result

    def copy(self) -> View:
        """Return an independent deep copy."""
        return View({path: fv.copy() for path, fv in self.files.items()})

    # ----------------------------------------------------------------- mutation

    def _file(self, path: str) -> FileView:
        file_view = self.files.get(path)
        if file_view is None:
            file_view = self.files[path] = FileView(path)
        return file_view

    def add_file(self, path: str, level: Level) -> None:
        """Record the file itself: ``levels[""] = max(existing, level)``."""
        self._file(path).set_level("", level)

    def add_element(self, path: str, element: str, level: Level) -> None:
        """Record an element (``path`` form; an ``llm_path`` line suffix is stripped)."""
        element = _strip_line_suffix(element)
        self._file(path).set_level(element, level)

    def add_lines(self, path: str, start: int, end: int) -> None:
        """Record an inclusive 1-indexed line range in ``path``."""
        before = path in self.files
        file_view = self._file(path)
        file_view.add_lines(start, end)
        if not before and file_view.is_empty():
            del self.files[path]

    # ----------------------------------------------------------------- queries

    def counts(self) -> dict[tuple[str, str], int]:
        """Relevance counts for this view.

        Each ``(path, element)`` level entry contributes ``int(level)`` (FULL 2,
        COLLAPSED 1); each line range adds 1 to the file key ``(path, "")``.
        """
        result: dict[tuple[str, str], int] = {}
        for path in sorted(self.files):
            file_view = self.files[path]
            for element in sorted(file_view.levels):
                key = (path, element)
                result[key] = result.get(key, 0) + int(file_view.levels[element])
            if file_view.line_ranges:
                key = (path, "")
                result[key] = result.get(key, 0) + len(file_view.line_ranges)
        return result

    def is_empty(self) -> bool:
        """True when no file has any recorded level or line range."""
        return all(fv.is_empty() for fv in self.files.values())

    # ----------------------------------------------------------------- JSON

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable form with sorted keys."""
        return {"files": {path: self.files[path].to_json() for path in sorted(self.files)}}

    @staticmethod
    def from_json(data: dict[str, Any]) -> View:
        """Inverse of :meth:`to_json`."""
        files = {
            path: FileView.from_json(path, payload)
            for path, payload in (data.get("files") or {}).items()
        }
        return View(files)

    # ----------------------------------------------------------------- pristine

    def restrict_to(self, session: Session) -> View:
        """Return a copy containing only files/elements present in ``session``.

        Element keys are normalised to ``CodeNode.path``; unknown elements and files are
        dropped, element keys on unparsed files are dropped, line ranges are clamped to
        the file length, and file views left empty are removed.
        """
        result = View()
        for path in sorted(self.files):
            file = session.files.get(path)
            if file is None:
                continue
            source = self.files[path]
            restricted = FileView(path)
            for element, level in source.levels.items():
                canonical = normalize_element_path(file, element)
                if canonical is None:
                    continue
                restricted.set_level(canonical, level)
            total = file_line_count(file)
            restricted.line_ranges = merge_ranges(
                (start, min(end, total)) for start, end in source.line_ranges if start <= total
            )
            if not restricted.is_empty():
                result.files[path] = restricted
        return result

    def __deepcopy__(self, memo: dict[int, Any]) -> View:
        return self.copy()

