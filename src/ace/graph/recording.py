"""Record what a tool-using worker looked at, and remap it onto the pristine repository.

See GRAPH_NEAT_DESIGN.md section 6.

A :class:`ViewRecorder` collects *sightings* while an agent runs. Each sighting is kept in
a raw log ``(path, kind, payload)`` against the file state at call time, and mirrored into
the live :attr:`ViewRecorder.view` (un-remapped, for inspection). :meth:`ViewRecorder.finalize`
replays the log against the pristine :class:`~agent2.core.session.Session`:

- Sightings of files that do not exist in the pristine session (files the agent created or
  moved into place) are dropped.
- File-level sightings keep their level.
- Element sightings are looked up by ``CodeNode.path`` (``path`` or ``llm_path`` accepted)
  in the *pristine* file and dropped when missing (renamed, created or unparsed).
- Line-range sightings are mapped from the file state they were recorded against to the
  pristine state. When the two are byte-identical the range is kept as is. Otherwise
  ``difflib.SequenceMatcher(None, pristine_lines, seen_lines, autojunk=False)`` opcodes
  are used: lines inside ``equal`` blocks map by offset, lines inside
  ``replace``/``insert``/``delete`` blocks are dropped, and the surviving lines are
  re-merged into ranges.
- The result is passed through :meth:`View.restrict_to` (clamps ranges, drops unknowns).

Implementation choices where the contract leaves room:

- **Snapshot of the seen state.** ``record_at``, ``record_element`` and ``record_lines``
  accept an optional keyword-only ``file`` (the workspace ``File`` at call time). The tool
  wrappers always pass it. For line ranges the bytes of that state are stored with the
  sighting, so ``finalize`` maps from *the state the agent actually saw* to pristine even
  when the file was edited again afterwards. Without a snapshot, ``finalize`` maps from
  the final ``workspace`` state (as the contract describes); if the file no longer exists
  in the workspace such a range is dropped. Without ``file``, ``record_at`` resolves
  against the pristine file.
- **record_at fallback.** ``record_at(path, line)`` records the innermost element whose
  ``full_block`` contains ``line`` (smallest byte length, first wins on ties; the same rule
  as the ``view_at`` / ``open_element_at`` tools) at FULL. If no element contains the line
  (or the file is unparsed), it records the line range ``line .. line + 49`` clamped to the
  file length, which matches the 50-line truncated display of ``view_at``; with
  ``to_end=True`` (``view_at(..., full=True)``) the range runs to the end of the file. A
  line beyond the end of the file records nothing.
- **Element views are FULL** even when the non-``full`` tool truncated an element longer
  than 50 lines (contract: ``record_element`` is FULL).
- **view_file without full** records the file at COLLAPSED (contract wording), although the
  tool really displays the first 50 raw lines.
- **Multi-path tools** (``view``, ``view_files``, ``views_at``, ``view_multi_lines``):
  the combined result cannot be split reliably (sections themselves contain blank lines),
  so each comma-separated item is re-evaluated with the corresponding single-item viewing
  function from :mod:`agent2.core.tools.viewing` (a pure read) and recorded only when that
  item's own output does not start with ``Error``. Single-item tools record when the whole
  result does not start with ``Error``.
- **open_* tools** return ``(content, continuation)``; the tuple is passed through
  untouched and the target is recorded at FULL when ``result[0]`` is not an error.
  ``open_file`` -> file FULL, ``open_lines`` -> line range, ``open_element`` -> element,
  ``open_element_at`` -> innermost element at the line (``record_at``).
- ``recording_tools`` also accepts a mapping ``name -> Tool`` (e.g. ``agent.tools``) and
  then wraps those tool objects instead of the global ones, so per-agent schema tweaks
  (such as ``delegate``'s enum) survive. Given plain names it looks them up in ``TOOLS``.
- A failure inside the *recording* step is logged and swallowed: it must never change the
  tool result the agent sees. Exceptions raised by the tool itself propagate unchanged.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Callable, Literal

from agent2.core.tools import viewing
from ace.graph.views import Level, View, merge_ranges, normalize_element_path

if TYPE_CHECKING:
    from agent2.code_parser.file import File
    from agent2.core.session import Session
    from agent2.core.workspace import Workspace

__all__ = [
    "Sighting",
    "ViewRecorder",
    "RecordingTool",
    "recording_tools",
    "VIEWING_TOOLS",
    "OPEN_TOOLS",
    "AT_FALLBACK_LINES",
]

logger = logging.getLogger(__name__)

AT_FALLBACK_LINES = 50
"""Lines recorded by ``record_at`` when no element contains the line (``view_at`` shows 50)."""

SightingKind = Literal["file", "element", "lines"]


@dataclass(frozen=True)
class Sighting:
    """One raw recording, against the file state at call time.

    Attributes:
        path: Project-relative file path as given to the tool.
        kind: ``"file"``, ``"element"`` or ``"lines"``.
        payload: ``Level`` for files, the element path (``CodeNode.path`` form) for
            elements, ``(start, end)`` for line ranges.
        snapshot: For line ranges, the bytes of the file state the lines refer to, or
            ``None`` to map from the final workspace state.
    """

    path: str
    kind: SightingKind
    payload: Any
    snapshot: bytes | None = None


# --------------------------------------------------------------------------- helpers


def _split_lines(data: bytes) -> list[bytes]:
    """Split bytes into lines numbered like the viewing tools (no phantom last line)."""
    if not data:
        return []
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()
    return lines


def _state_bytes(file: File | None) -> bytes | None:
    if file is None or file.current_state is None:
        return None
    return file.current_state.bytes


def _line_count(data: bytes | None) -> int:
    return len(_split_lines(data or b""))


def innermost_element(file: File | None, line: int) -> str | None:
    """Return the ``CodeNode.path`` of the smallest element whose full block contains ``line``.

    Same rule as the ``view_at`` and ``open_element_at`` tools. ``None`` for unparsed files
    or when no element contains the line.
    """
    nodes = getattr(file, "code_nodes", None)
    if not nodes:
        return None
    best, best_len = None, None
    for node in nodes.values():
        start_line = node.full_block.start_point[0] + 1
        end_line = node.full_block.end_point[0] + 1
        if start_line <= line <= end_line:
            length = node.full_block.end_byte - node.full_block.start_byte
            if best_len is None or length < best_len:
                best, best_len = node, length
    return best.path if best is not None else None


def map_ranges(
    ranges: Iterable[tuple[int, int]], pristine: bytes, seen: bytes
) -> list[tuple[int, int]]:
    """Map 1-indexed inclusive line ranges in ``seen`` onto ``pristine``.

    Lines inside ``equal`` opcode blocks map by offset; all other lines are dropped. The
    surviving lines are merged into sorted ranges.
    """
    ranges = list(ranges)
    if pristine == seen:
        return merge_ranges(ranges)
    pristine_lines = _split_lines(pristine)
    seen_lines = _split_lines(seen)
    opcodes = SequenceMatcher(None, pristine_lines, seen_lines, autojunk=False).get_opcodes()
    mapped: list[tuple[int, int]] = []
    for start, end in ranges:
        for tag, i1, _i2, j1, j2 in opcodes:
            if tag != "equal":
                continue
            # seen lines j1+1 .. j2 (1-indexed) map to pristine i1+1 .. i2.
            lo, hi = max(start, j1 + 1), min(end, j2)
            if lo > hi:
                continue
            offset = i1 - j1
            mapped.append((lo + offset, hi + offset))
    return merge_ranges(mapped)


# --------------------------------------------------------------------------- recorder


class ViewRecorder:
    """Collects what an agent viewed and remaps it onto the pristine session.

    Attributes:
        session: The pristine session (``session.files`` is the base state).
        view: Live, un-remapped View of everything recorded so far.
        sightings: Raw log of recordings in call order.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.view = View()
        self.sightings: list[Sighting] = []

    # ------------------------------------------------------------------ recording

    def record_file(self, path: str, full: bool) -> None:
        """Record the file itself at FULL (``full``) or COLLAPSED."""
        level = Level.FULL if full else Level.COLLAPSED
        self.sightings.append(Sighting(path, "file", level))
        self.view.add_file(path, level)

    def record_element(self, path: str, element: str, *, file: File | None = None) -> None:
        """Record an element at FULL.

        Args:
            path: File path.
            element: ``path`` or ``llm_path`` form; stored as ``CodeNode.path``.
            file: Optional file state at call time, used to normalise ``element``.
        """
        canonical = normalize_element_path(file, element) if file is not None else None
        if canonical is None:
            probe = View()
            probe.add_element(path, element, Level.FULL)  # strips an llm_path line suffix
            canonical = next(iter(probe.files[path].levels))
        self.sightings.append(Sighting(path, "element", canonical))
        self.view.add_element(path, canonical, Level.FULL)

    def record_at(
        self, path: str, line: int, *, file: File | None = None, to_end: bool = False
    ) -> None:
        """Record the innermost element at ``line`` (FULL), or a line range if none.

        Args:
            path: File path.
            line: 1-indexed line.
            file: File state at call time; defaults to the pristine session file.
            to_end: When no element contains the line, record to the end of the file
                instead of ``AT_FALLBACK_LINES`` lines.
        """
        line = int(line)
        if file is None:
            file = self.session.files.get(path)
        element = innermost_element(file, line)
        if element is not None:
            self.record_element(path, element, file=file)
            return
        total = _line_count(_state_bytes(file)) if file is not None else None
        end = line + AT_FALLBACK_LINES - 1
        if total is not None:
            end = total if to_end else min(end, total)
        if line < 1 or line > end:
            return
        self.record_lines(path, line, end, file=file)

    def record_lines(
        self, path: str, start: int, end: int, *, file: File | None = None
    ) -> None:
        """Record an inclusive 1-indexed line range.

        Args:
            path: File path.
            start: First line.
            end: Last line (swapped with ``start`` if reversed).
            file: File state the lines refer to. When given, the range is clamped to its
                length and its bytes are kept for remapping in :meth:`finalize`.
        """
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start
        snapshot = _state_bytes(file)
        if snapshot is not None:
            end = min(end, _line_count(snapshot))
        start = max(start, 1)
        if start > end:
            return
        self.sightings.append(Sighting(path, "lines", (start, end), snapshot))
        self.view.add_lines(path, start, end)

    # ------------------------------------------------------------------ finalize

    def finalize(self, workspace: Workspace) -> View:
        """Remap every sighting onto the pristine session and return the resulting View."""
        result = View()
        pending_lines: dict[tuple[str, bytes | None], list[tuple[int, int]]] = {}
        for sighting in self.sightings:
            pristine = self.session.files.get(sighting.path)
            if pristine is None:
                continue  # created in the workspace (or moved there)
            if sighting.kind == "file":
                result.add_file(sighting.path, sighting.payload)
            elif sighting.kind == "element":
                canonical = normalize_element_path(pristine, sighting.payload)
                if canonical:
                    result.add_element(sighting.path, canonical, Level.FULL)
            else:
                key = (sighting.path, sighting.snapshot)
                pending_lines.setdefault(key, []).append(sighting.payload)
        for (path, snapshot), ranges in pending_lines.items():
            pristine_bytes = _state_bytes(self.session.files[path]) or b""
            seen = snapshot if snapshot is not None else _state_bytes(workspace.read_file(path))
            if seen is None:
                continue  # deleted in the workspace and no snapshot of what was seen
            for start, end in map_ranges(ranges, pristine_bytes, seen):
                result.add_lines(path, start, end)
        return result.restrict_to(self.session)


# --------------------------------------------------------------------------- tool wrappers


def _is_error(result: Any) -> bool:
    """True when a tool result (string or ``(content, continuation)`` tuple) is an error."""
    if isinstance(result, tuple):
        result = result[0] if result else None
    return not isinstance(result, str) or result.startswith("Error")


def _items(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _workspace_file(args: Mapping[str, Any], recorder: ViewRecorder, path: str) -> File | None:
    agent = args.get("agent")
    workspace = getattr(agent, "workspace", None)
    if workspace is not None:
        return workspace.read_file(path)
    return recorder.session.files.get(path)


def _ok(output: Any) -> bool:
    return isinstance(output, str) and not output.startswith("Error")


def _record_view_path(args: Mapping[str, Any], recorder: ViewRecorder, path: str) -> None:
    """Record one ``view``/``view_path`` item: ``file`` or ``file::element``."""
    if "::" in path:
        file_path, element = path.split("::", 1)
        recorder.record_element(file_path, element, file=_workspace_file(args, recorder, file_path))
    else:
        recorder.record_file(path, False)


def _rec_view(args, recorder, result) -> None:
    session, agent = args.get("session"), args.get("agent")
    for path in _items(args.get("paths")):
        if _ok(viewing.view_path(session, agent, path)):
            _record_view_path(args, recorder, path)


def _rec_view_path(args, recorder, result) -> None:
    if not _is_error(result):
        _record_view_path(args, recorder, str(args["path"]))


def _rec_view_files(args, recorder, result) -> None:
    session, agent = args.get("session"), args.get("agent")
    for path in _items(args.get("file_paths")):
        full = path.endswith("*")
        if full:
            path = path[:-1]
        if _ok(viewing.view_file(session, agent, path, full)):
            recorder.record_file(path, full)


def _rec_view_file(args, recorder, result) -> None:
    if not _is_error(result):
        recorder.record_file(str(args["file_path"]), bool(args.get("full", False)))


def _rec_views_at(args, recorder, result) -> None:
    session, agent = args.get("session"), args.get("agent")
    full = bool(args.get("full", False))
    for item in _items(args.get("paths")):
        if "::" not in item:
            continue
        file_path, line_str = item.split("::", 1)
        try:
            line = int(line_str)
        except ValueError:
            continue
        if _ok(viewing.view_at(session, agent, file_path, line, full)):
            recorder.record_at(
                file_path, line, file=_workspace_file(args, recorder, file_path), to_end=full
            )


def _rec_view_at(args, recorder, result) -> None:
    if _is_error(result):
        return
    path = str(args["file_path"])
    recorder.record_at(
        path,
        int(args["line"]),
        file=_workspace_file(args, recorder, path),
        to_end=bool(args.get("full", False)),
    )


def _rec_view_element(args, recorder, result) -> None:
    if _is_error(result):
        return
    path = str(args["file_path"])
    recorder.record_element(path, str(args["element_path"]), file=_workspace_file(args, recorder, path))


def _rec_view_multi_lines(args, recorder, result) -> None:
    session, agent = args.get("session"), args.get("agent")
    for item in _items(args.get("file_line_paths")):
        parts = item.split("::")
        if len(parts) != 3:
            continue
        try:
            start, end = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if start > end:
            continue  # the tool shows nothing for a reversed range
        if _ok(viewing.view_lines(session, agent, parts[0], start, end)):
            recorder.record_lines(parts[0], start, end, file=_workspace_file(args, recorder, parts[0]))


def _rec_lines(args, recorder, result) -> None:
    if _is_error(result):
        return
    path = str(args["file_path"])
    start, end = int(args["line_start"]), int(args["line_end"])
    if start > end:
        return  # view_lines shows nothing for a reversed range (open_lines rejects it)
    recorder.record_lines(path, start, end, file=_workspace_file(args, recorder, path))


def _rec_open_file(args, recorder, result) -> None:
    if not _is_error(result):
        recorder.record_file(str(args["file_path"]), True)


def _rec_open_element_at(args, recorder, result) -> None:
    if _is_error(result):
        return
    path = str(args["file_path"])
    recorder.record_at(path, int(args["line"]), file=_workspace_file(args, recorder, path))


Recorder = Callable[[Mapping[str, Any], ViewRecorder, Any], None]

VIEWING_TOOLS: dict[str, Recorder] = {
    "view": _rec_view,
    "view_path": _rec_view_path,
    "view_files": _rec_view_files,
    "view_file": _rec_view_file,
    "views_at": _rec_views_at,
    "view_at": _rec_view_at,
    "view_element": _rec_view_element,
    "view_multi_lines": _rec_view_multi_lines,
    "view_lines": _rec_lines,
}
"""Viewing tool name -> recording rule ``(bound args, recorder, result) -> None``."""

OPEN_TOOLS: dict[str, Recorder] = {
    "open_file": _rec_open_file,
    "open_lines": _rec_lines,
    "open_element": _rec_view_element,
    "open_element_at": _rec_open_element_at,
}
"""``open_*`` tool name -> recording rule; the target is recorded as a FULL view."""


class RecordingTool:
    """A drop-in replacement for :class:`~agent2.core.tools.tool.Tool` that records views.

    Exposes the wrapped tool's ``func`` (so ``ToolCallHook`` injects ``agent``/``session``
    from its signature), ``name``, ``description``, ``openai_schema`` and ``examples``.
    """

    def __init__(self, tool: Any, recorder: ViewRecorder) -> None:
        self.tool = tool
        self.recorder = recorder
        self.func: Callable = tool.func
        self.name: str = tool.name
        self.description: str | None = getattr(tool, "description", None)
        self.openai_schema: dict[str, Any] = tool.openai_schema
        self.examples: list = list(getattr(tool, "examples", []) or [])
        self.rule: Recorder | None = VIEWING_TOOLS.get(self.name) or OPEN_TOOLS.get(self.name)
        self._signature = inspect.signature(self.func)

    def __repr__(self) -> str:
        return f"RecordingTool({self.name!r})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the original tool, record the implied view, and return the result unchanged."""
        result = self.tool(*args, **kwargs)
        if self.rule is None:
            return result
        if inspect.isawaitable(result):
            return self._record_async(result, args, kwargs)
        self._record(args, kwargs, result)
        return result

    async def _record_async(self, awaitable: Any, args: tuple, kwargs: dict) -> Any:
        result = await awaitable
        self._record(args, kwargs, result)
        return result

    def _record(self, args: tuple, kwargs: dict, result: Any) -> None:
        try:
            bound = self._signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            self.rule(bound.arguments, self.recorder, result)
        except Exception:  # recording must never alter what the agent sees
            logger.exception("Failed to record view for tool %s", self.name)


def recording_tools(
    tool_names: Iterable[str] | Mapping[str, Any], recorder: ViewRecorder
) -> dict[str, RecordingTool]:
    """Wrap tools so that viewing calls are recorded into ``recorder``.

    Args:
        tool_names: Tool names to look up in ``agent2.core.globals.TOOLS``, or a mapping
            ``name -> Tool`` (such as ``agent.tools``) whose tool objects are wrapped.
        recorder: Destination of the recorded sightings.

    Returns:
        ``name -> RecordingTool``. Viewing and ``open_*`` tools record; all others pass
        through unchanged.

    Raises:
        KeyError: A name is not a registered tool.
    """
    from agent2.core.globals import TOOLS

    if isinstance(tool_names, Mapping):
        source = dict(tool_names)
    else:
        source = {}
        for name in tool_names:
            if name not in TOOLS:
                raise KeyError(f"Unknown tool {name!r}")
            source[name] = TOOLS[name]
    return {name: RecordingTool(tool, recorder) for name, tool in source.items()}
