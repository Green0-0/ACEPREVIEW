"""Codebase graph: nodes are files and code elements, edges are structural relations.

See GRAPH_NEAT_DESIGN.md section 5. The graph is built once from the pristine
:class:`~agent2.core.session.Session` and is read-only afterwards. It drives relevance
ranking (personalized PageRank) and the budgeted expand-and-contract selection that every
tool-less worker uses to decide what to render.

Nodes
    ``NodeKey = (file path, element path)``; element ``""`` is the file node. Every file in
    ``session.files`` contributes its file node; a ``CodeFile`` also contributes one node
    per unique ``CodeNode.path``.

Edges (weights are the module constants below; parallel edges between the same pair of
nodes are summed):

- containment: file -> top-level element, element -> direct child
  (``CONTAINMENT_WEIGHT``) and the reverse direction (``CONTAINMENT_BACK_WEIGHT``).
- sibling adjacency: consecutive elements *sharing the same parent* (top-level elements of
  a file, or children of one class/function), ordered by start byte, both directions
  (``SIBLING_WEIGHT``).
- imports (Python files only, read from the tree-sitter tree, including imports nested in
  functions): file -> imported project file (``IMPORT_WEIGHT``). ``a.b.c`` resolves to
  ``a/b/c.py`` or ``a/b/c/__init__.py``; if no such path exists, a unique file whose
  module path *ends with* ``a.b.c`` is accepted (this covers ``src/`` layouts). For
  ``from M import n`` the submodule ``M.n`` is tried first for every imported name, then
  ``M`` itself. Relative imports resolve against the importing file's package directory.
- name references: every ``identifier`` token inside an element's body that is not part
  of a nested element's body (identifiers are attributed to the *innermost* enclosing
  element), not the name of a ``def``/``class``, and not inside an import statement, is
  matched against ``CodeNode.name`` of every other element in the project. Names shorter
  than ``MIN_NAME_LENGTH``, Python keywords/soft keywords, builtins, ``self`` and ``cls``
  are skipped. Each distinct referenced name adds ``REFERENCE_WEIGHT / n`` to each of its
  ``n`` targets (so ambiguous names are diluted). Repeated occurrences do not add weight.

Choices where the contract leaves room:

- ``rank`` drops seeds with non-positive weight or unknown keys; if nothing remains it
  uses uniform seeds over the file nodes (the same fallback ``select`` uses). Every node
  is returned, sorted by descending score with ties broken by ascending key.
- ``resolve_lines`` returns the intersecting element keys ordered by start line (outer
  before inner). The file key is **not** included when at least one element intersects;
  when none does it returns ``[(path, "")]`` (or ``[]`` for a path that is not a node).
- ``size_of`` estimates the rendered size *including* the ``N| `` line prefixes, based on
  the whole lines an element spans (FULL element: all its lines; COLLAPSED element:
  signature + docstring + direct children's signature lines, mirroring
  ``rendering.py``). A file node at COLLAPSED is the skeleton (lines outside top-level
  elements + heads of top-level elements + one hidden-body marker each) plus the section
  header; at FULL it is the whole file plus header. Unparsed files cost the whole file at
  either level. Element sizes exclude section headers because ``select`` always adds the
  file skeleton (which carries the header) alongside an element.
- Elements sharing a path are one node; its sizes and line spans cover every definition
  with that path (as rendered), and skeletons account for each definition separately.
- ``select`` uses ``session`` only to skip keys whose file is absent from it. The file
  skeleton cost (``size_of(file, COLLAPSED)``) is counted once per file and added to the
  cost of the first element of that file. An element's cost is ``size_of(el, FULL)``
  minus the cost of its lines already paid for in that file (skeleton heads, enclosing
  or enclosed FULL elements), so nested and overlapping selections are not double
  counted.
- ``seeds_from_views`` and ``candidate_list`` are methods (they need the graph to resolve
  line ranges and to rank). ``seeds_from_views`` drops keys that are not graph nodes.
  ``candidate_list`` keeps a key sequence in the order given (assumed ranked); for a View
  it ranks with ``seeds`` (default: ``seeds_from_views([view])``) and lists the view's own
  keys in rank order. Truncation keeps whole lines only.
"""
from __future__ import annotations

import builtins
import keyword
import posixpath
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ace.graph.views import Level, View

if TYPE_CHECKING:
    from agent2.code_parser.file import CodeFile, File
    from agent2.code_parser.objects import CodeNode
    from agent2.core.session import Session

__all__ = [
    "NodeKey",
    "CodebaseGraph",
    "CONTAINMENT_WEIGHT",
    "CONTAINMENT_BACK_WEIGHT",
    "SIBLING_WEIGHT",
    "IMPORT_WEIGHT",
    "REFERENCE_WEIGHT",
    "MIN_NAME_LENGTH",
    "MAX_CONSECUTIVE_SKIPS",
]

NodeKey = tuple[str, str]
"""``(file path, element path)``; element ``""`` is the file node."""

CONTAINMENT_WEIGHT = 1.0
"""file -> top-level element and element -> direct child."""
CONTAINMENT_BACK_WEIGHT = 0.5
"""child -> parent (element -> file for top-level elements)."""
SIBLING_WEIGHT = 0.3
"""Consecutive elements with the same parent, both directions."""
IMPORT_WEIGHT = 1.0
"""Python file -> imported project file."""
REFERENCE_WEIGHT = 0.7
"""Element -> element whose name appears as an identifier in its body (split over targets)."""
MIN_NAME_LENGTH = 3
"""Identifiers shorter than this are ignored for name references."""
MAX_CONSECUTIVE_SKIPS = 50
"""``select`` stops after this many consecutive candidates that do not fit."""

_SKIP_NAMES: frozenset[str] = frozenset(
    set(keyword.kwlist) | set(keyword.softkwlist) | set(dir(builtins)) | {"self", "cls"}
)
_IMPORT_TYPES = frozenset({"import_statement", "import_from_statement", "future_import_statement"})
_DEFINITION_TYPES = frozenset({"function_definition", "class_definition"})
_PREFIX = "| "  # matches rendering.SPACER


# --------------------------------------------------------------------------- helpers


def _lines_of(file: File | CodeFile) -> list[str]:
    """Decode a file's current state into lines, like ``rendering._split_lines``."""
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
        return row
    return row + 1


def _signature_lines(node: CodeNode, text: list[str] | None = None) -> set[int]:
    """Mirror of ``rendering._signature_lines`` (``text``: the file's lines)."""
    start = _start_line(node)
    if node.body_block is None:
        return set(range(start, _end_line(node) + 1))
    end = max(start, node.body_block.start_point[0])
    row, column = node.body_block.start_point
    if text is not None and 0 <= row < len(text) and text[row].encode("utf-8")[:column].strip():
        end = max(end, row + 1)  # body starts on the signature's last line
    return set(range(start, end + 1))


def _head_lines(node: CodeNode, text: list[str] | None = None) -> set[int]:
    lines = _signature_lines(node, text)
    if node.doc_block is not None:
        lines |= set(range(node.doc_block.start_point[0] + 1, node.doc_block.end_point[0] + 2))
    return set(range(min(lines), max(lines) + 1)) if lines else lines


def _module_name(path: str) -> str | None:
    """``a/b/c.py`` -> ``a.b.c``; ``a/b/__init__.py`` -> ``a.b``; non-Python -> None."""
    if path.endswith(".py"):
        stem = path[: -len(".py")]
    elif path.endswith(".pyi"):
        stem = path[: -len(".pyi")]
    else:
        return None
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _text(data: bytes, node: Any) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- graph


class CodebaseGraph:
    """Weighted directed graph over files and code elements of a pristine session.

    Attributes:
        nodes: Every node key in the graph.
    """

    def __init__(self, session: Session) -> None:
        """Build nodes and edges from ``session.files`` (treated as pristine)."""
        self.nodes: set[NodeKey] = set()
        self._files: dict[str, File | CodeFile] = {}
        self._elements: dict[str, list[CodeNode]] = {}  # path -> unique nodes by start byte
        # path -> every distinct node (elements sharing a path included), by start byte
        self._all: dict[str, list[CodeNode]] = {}
        self._same: dict[NodeKey, list[CodeNode]] = {}  # key -> all nodes with that path
        self._code: dict[NodeKey, CodeNode] = {}
        self._lines: dict[str, list[str]] = {}
        self._edges: dict[NodeKey, dict[NodeKey, float]] = {}
        self._size_cache: dict[tuple[NodeKey, Level], int] = {}
        self._line_cost_cache: dict[str, list[int]] = {}

        for path in sorted(session.files):
            file = session.files[path]
            self._files[path] = file
            self.nodes.add((path, ""))
            code_nodes = getattr(file, "code_nodes", None)
            unique: dict[str, CodeNode] = {}
            distinct: dict[tuple[str, int], CodeNode] = {}
            for node in (code_nodes or {}).values():
                unique.setdefault(node.path, node)
                distinct.setdefault((node.path, node.full_block.start_byte), node)
            ordered = sorted(unique.values(), key=lambda n: (n.full_block.start_byte, n.path))
            self._elements[path] = ordered
            self._all[path] = sorted(distinct.values(), key=lambda n: (n.full_block.start_byte, n.path))
            for node in ordered:
                key = (path, node.path)
                self.nodes.add(key)
                self._code[key] = node
            for node in self._all[path]:
                self._same.setdefault((path, node.path), []).append(node)

        self._order: list[NodeKey] = sorted(self.nodes)
        self._file_keys: list[NodeKey] = [key for key in self._order if key[1] == ""]
        self._modules = self._module_index()
        self._names = self._name_index()

        for path in sorted(self._files):
            self._add_structure_edges(path)
            self._add_import_edges(path)
            self._add_reference_edges(path)

        self._adjacency = self._normalized_adjacency()

    # ----------------------------------------------------------------- construction

    def _add_edge(self, src: NodeKey, dst: NodeKey, weight: float) -> None:
        if src == dst or weight <= 0:
            return
        targets = self._edges.setdefault(src, {})
        targets[dst] = targets.get(dst, 0.0) + weight

    def _add_structure_edges(self, path: str) -> None:
        """Containment (both directions) and sibling adjacency."""
        file_key = (path, "")
        by_parent: dict[str | None, list[CodeNode]] = {}
        for node in self._elements[path]:
            parent_key = (path, node.parent_path) if node.parent_path else file_key
            if parent_key not in self.nodes:
                parent_key = file_key
            key = (path, node.path)
            self._add_edge(parent_key, key, CONTAINMENT_WEIGHT)
            self._add_edge(key, parent_key, CONTAINMENT_BACK_WEIGHT)
            by_parent.setdefault(parent_key[1] or None, []).append(node)
        for parent in sorted(by_parent, key=lambda p: p or ""):
            siblings = by_parent[parent]
            for left, right in zip(siblings, siblings[1:]):
                a, b = (path, left.path), (path, right.path)
                self._add_edge(a, b, SIBLING_WEIGHT)
                self._add_edge(b, a, SIBLING_WEIGHT)

    def _module_index(self) -> dict[str, str]:
        """Dotted module name -> file path for every Python file (``.py`` wins over ``.pyi``)."""
        index: dict[str, str] = {}
        for path in sorted(self._files, key=lambda p: (p.endswith(".pyi"), p)):
            name = _module_name(path)
            if name is not None:
                index.setdefault(name, path)
        return index

    def _resolve_module(self, dotted: str) -> str | None:
        """Resolve a dotted module name to a project file path, or None."""
        if not dotted:
            return None
        exact = self._modules.get(dotted)
        if exact is not None:
            return exact
        suffix = "." + dotted
        matches = sorted({p for name, p in self._modules.items() if name.endswith(suffix)})
        return matches[0] if len(matches) == 1 else None

    def _package_of(self, path: str) -> str:
        """Dotted package containing ``path`` (its directory), ``""`` at the root."""
        directory = posixpath.dirname(path)
        return directory.replace("/", ".") if directory else ""

    def _add_import_edges(self, path: str) -> None:
        """file -> imported project file, for Python files parsed with tree-sitter."""
        file = self._files[path]
        tree = getattr(file, "tree", None)
        adapter = getattr(file, "adapter", None)
        if tree is None or getattr(adapter, "language_id", None) != "python":
            return
        data = file.current_state.bytes if file.current_state else b""
        targets: set[str] = set()
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "import_statement":
                for name in node.children_by_field_name("name"):
                    dotted = name.child_by_field_name("name") if name.type == "aliased_import" else name
                    if dotted is None:
                        continue
                    resolved = self._resolve_module(_text(data, dotted))
                    if resolved:
                        targets.add(resolved)
                continue
            if node.type == "import_from_statement":
                targets |= self._resolve_from_import(path, node, data)
                continue
            stack.extend(node.children)
        for target in sorted(targets):
            self._add_edge((path, ""), (target, ""), IMPORT_WEIGHT)

    def _resolve_from_import(self, path: str, node: Any, data: bytes) -> set[str]:
        module = node.child_by_field_name("module_name")
        if module is None:
            return set()
        if module.type == "relative_import":
            dots = 0
            rest = ""
            for child in module.children:
                if child.type == "import_prefix":
                    dots = _text(data, child).count(".")
                elif child.type == "dotted_name":
                    rest = _text(data, child)
            package = self._package_of(path).split(".") if self._package_of(path) else []
            up = dots - 1
            if up > len(package):
                return set()
            base_parts = package[: len(package) - up] if up else package
            base = ".".join([*base_parts, *([rest] if rest else [])])
        else:
            base = _text(data, module)
        found: set[str] = set()
        names = node.children_by_field_name("name")
        for name in names:
            dotted = name.child_by_field_name("name") if name.type == "aliased_import" else name
            if dotted is None:
                continue
            sub = f"{base}.{_text(data, dotted)}" if base else _text(data, dotted)
            resolved = self._modules.get(sub)
            if resolved:
                found.add(resolved)
        if base:
            resolved = self._modules.get(base) if module.type == "relative_import" else self._resolve_module(base)
            if resolved and (resolved != path):
                found.add(resolved)
        return found

    def _name_index(self) -> dict[str, list[NodeKey]]:
        """``CodeNode.name`` -> element keys with that name, sorted."""
        index: dict[str, list[NodeKey]] = {}
        for key in self._order:
            node = self._code.get(key)
            if node is None:
                continue
            index.setdefault(node.name, []).append(key)
        return index

    def _add_reference_edges(self, path: str) -> None:
        """Element -> elements named by identifiers in its body (innermost attribution)."""
        file = self._files[path]
        tree = getattr(file, "tree", None)
        if tree is None or not self._elements[path] or file.current_state is None:
            return
        data = file.current_state.bytes
        bodies: dict[tuple[int, int], NodeKey] = {}
        for node in self._all[path]:
            if node.body_block is not None:
                bodies[(node.body_block.start_byte, node.body_block.end_byte)] = (path, node.path)
        referenced: dict[NodeKey, set[str]] = {}
        stack: list[tuple[Any, NodeKey | None]] = [(tree.root_node, None)]
        while stack:
            ts_node, owner = stack.pop()
            if ts_node.type in _IMPORT_TYPES:
                continue
            owner = bodies.get((ts_node.start_byte, ts_node.end_byte), owner) if ts_node.type == "block" else owner
            if ts_node.type == "identifier":
                if owner is not None:
                    parent = ts_node.parent
                    is_def_name = (
                        parent is not None
                        and parent.type in _DEFINITION_TYPES
                        and parent.child_by_field_name("name") == ts_node
                    )
                    if not is_def_name:
                        name = _text(data, ts_node)
                        if len(name) >= MIN_NAME_LENGTH and name not in _SKIP_NAMES:
                            referenced.setdefault(owner, set()).add(name)
                continue
            stack.extend((child, owner) for child in ts_node.children)
        for owner in sorted(referenced):
            for name in sorted(referenced[owner]):
                targets = [key for key in self._names.get(name, ()) if key != owner]
                if not targets:
                    continue
                weight = REFERENCE_WEIGHT / len(targets)
                for target in targets:
                    self._add_edge(owner, target, weight)

    def _normalized_adjacency(self) -> dict[NodeKey, list[tuple[NodeKey, float]]]:
        adjacency: dict[NodeKey, list[tuple[NodeKey, float]]] = {}
        for src in self._order:
            targets = self._edges.get(src)
            if not targets:
                continue
            total = sum(targets.values())
            adjacency[src] = [(dst, targets[dst] / total) for dst in sorted(targets)]
        return adjacency

    # ----------------------------------------------------------------- queries

    def neighbors(self, key: NodeKey) -> dict[NodeKey, float]:
        """Weighted out-edges of ``key`` (raw, unnormalised weights; empty if none)."""
        return dict(sorted(self._edges.get(tuple(key), {}).items()))  # type: ignore[arg-type]

    def element(self, key: NodeKey) -> CodeNode | None:
        """The ``CodeNode`` behind an element key, or None for file/unknown keys."""
        return self._code.get(key)

    def rank(
        self,
        seeds: dict[NodeKey, float],
        *,
        damping: float = 0.85,
        iterations: int = 30,
    ) -> list[tuple[NodeKey, float]]:
        """Personalized PageRank over the graph.

        Args:
            seeds: Personalization weights. Non-positive and unknown keys are ignored; if
                nothing remains, uniform weights over file nodes are used.
            damping: Probability of following an edge rather than teleporting to the seeds.
            iterations: Power-iteration steps.

        Returns:
            Every node with its score, sorted by descending score then ascending key.
            Dangling mass (nodes without out-edges) is redistributed to the seed
            distribution.
        """
        order = self._order
        if not order:
            return []
        personalization = self._personalization(seeds)
        index = {key: i for i, key in enumerate(order)}
        p = [personalization.get(key, 0.0) for key in order]
        out = [
            [(index[dst], weight) for dst, weight in self._adjacency.get(key, ())]
            for key in order
        ]
        scores = list(p)
        for _ in range(max(0, iterations)):
            new = [0.0] * len(order)
            dangling = 0.0
            for i, score in enumerate(scores):
                if not score:
                    continue
                edges = out[i]
                if not edges:
                    dangling += score
                    continue
                spread = damping * score
                for j, weight in edges:
                    new[j] += spread * weight
            teleport = (1.0 - damping) + damping * dangling
            scores = [value + teleport * p[i] for i, value in enumerate(new)]
        ranked = list(zip(order, scores))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _personalization(self, seeds: dict[NodeKey, float]) -> dict[NodeKey, float]:
        valid = {
            tuple(key): float(weight)  # type: ignore[misc]
            for key, weight in (seeds or {}).items()
            if tuple(key) in self.nodes and weight > 0
        }
        if not valid:
            base = self._file_keys or self._order
            valid = {key: 1.0 for key in base}
        total = sum(valid.values())
        return {key: weight / total for key, weight in valid.items()}

    def resolve_lines(self, path: str, start: int, end: int) -> list[NodeKey]:
        """Elements of ``path`` whose full-block line span intersects ``start..end``.

        Args:
            path: File path.
            start: First line (1-indexed, inclusive). A reversed range is swapped.
            end: Last line (inclusive).

        Returns:
            Intersecting element keys (outer and inner alike) ordered by start line. The
            file key is not included when any element intersects; if none does,
            ``[(path, "")]`` is returned, or ``[]`` when ``path`` is not in the graph.
        """
        if (path, "") not in self.nodes:
            return []
        if start > end:
            start, end = end, start
        hits = list(dict.fromkeys(
            (path, node.path)
            for node in self._all.get(path, ())
            if _start_line(node) <= end and _end_line(node) >= start
        ))
        return hits or [(path, "")]

    # ----------------------------------------------------------------- sizes

    def _text(self, path: str) -> list[str]:
        """The file's decoded lines (cached)."""
        lines = self._lines.get(path)
        if lines is None:
            lines = self._lines[path] = _lines_of(self._files[path])
        return lines

    def _line_costs(self, path: str) -> list[int]:
        """Rendered cost of each line (index 0 = line 1): ``N| text`` plus newline."""
        costs = self._line_cost_cache.get(path)
        if costs is None:
            lines = self._text(path)
            costs = [len(str(n)) + len(_PREFIX) + len(text) + 1 for n, text in enumerate(lines, 1)]
            self._line_cost_cache[path] = costs
        return costs

    def _cost_of_lines(self, path: str, lines: Iterable[int]) -> int:
        costs = self._line_costs(path)
        return sum(costs[n - 1] for n in lines if 1 <= n <= len(costs))

    def _header_cost(self, path: str) -> int:
        return len(f"--- {path} ---") + 1 + 2  # header line + separator between files

    def size_of(self, key: NodeKey, level: Level) -> int:
        """Estimated rendered characters of ``key`` at ``level`` (cached).

        FULL element: every line it spans. COLLAPSED element: signature + docstring +
        direct children's signatures. File node at COLLAPSED: skeleton (non-element
        top-level lines + heads of top-level elements + a hidden marker each) plus header.
        File node at FULL (or any level for unparsed files): whole file plus header.
        Line costs include the ``N| `` prefix and newline.

        Raises:
            KeyError: ``key`` is not a node.
        """
        key = (key[0], key[1])
        if key not in self.nodes:
            raise KeyError(key)
        level = Level(level)
        cached = self._size_cache.get((key, level))
        if cached is not None:
            return cached
        path, element = key
        total = len(self._line_costs(path))
        if element:
            text = self._text(path)
            lines = set()
            for node in self._same.get(key, [self._code[key]]):  # every definition is rendered
                if level >= Level.FULL:
                    lines |= set(range(_start_line(node), _end_line(node) + 1))
                else:
                    lines |= _head_lines(node, text)
                    for child in node.children:
                        lines |= _signature_lines(child, text)
            size = self._cost_of_lines(path, lines)
        else:
            parsed = getattr(self._files[path], "code_nodes", None) is not None
            if level >= Level.FULL or not parsed:
                size = self._cost_of_lines(path, range(1, total + 1))
            else:
                inside: set[int] = set()
                shown: set[int] = set()
                markers = 0
                text = self._text(path)
                for node in self._all[path]:
                    if node.parent_path is not None:
                        continue
                    full = set(range(_start_line(node), _end_line(node) + 1))
                    inside |= full
                    head = _head_lines(node, text)
                    shown |= head
                    if full - head:
                        markers += 1
                shown |= set(range(1, total + 1)) - inside
                size = self._cost_of_lines(path, shown) + markers * 32
            size += self._header_cost(path)
        self._size_cache[(key, level)] = size
        return size

    # ----------------------------------------------------------------- selection

    def _full_lines(self, key: NodeKey) -> set[int]:
        lines: set[int] = set()
        for node in self._same.get(key, [self._code[key]]):
            lines |= set(range(_start_line(node), _end_line(node) + 1))
        return lines

    def _skeleton_lines(self, path: str) -> set[int]:
        """Lines shown by the COLLAPSED skeleton of a parsed file (all lines if unparsed)."""
        total = len(self._line_costs(path))
        if getattr(self._files[path], "code_nodes", None) is None:
            return set(range(1, total + 1))
        inside: set[int] = set()
        shown: set[int] = set()
        text = self._text(path)
        for node in self._all[path]:
            if node.parent_path is None:
                inside |= set(range(_start_line(node), _end_line(node) + 1))
                shown |= _head_lines(node, text)
        return shown | (set(range(1, total + 1)) - inside)

    def select(self, seeds: dict[NodeKey, float], budget: int, session: Session) -> View:
        """Expand-and-contract: pick the most relevant nodes that fit ``budget`` characters.

        Positive seeds are considered first (by descending weight, then key), then the
        remaining nodes in rank order. Elements are added at FULL, file nodes at
        COLLAPSED; adding an element also adds its file at COLLAPSED (skeleton cost
        counted once per file). A candidate that does not fit is skipped; the walk stops
        after ``MAX_CONSECUTIVE_SKIPS`` consecutive skips. Empty seeds rank with uniform
        weights over file nodes.

        Args:
            seeds: Relevance weights per node key.
            budget: Character budget for the estimated rendered size.
            session: The pristine session; keys whose file is absent are skipped.

        Returns:
            The selected View (not rendered).
        """
        ranked = self.rank(seeds)
        positive = sorted(
            ((tuple(key), float(w)) for key, w in (seeds or {}).items() if w > 0 and tuple(key) in self.nodes),
            key=lambda item: (-item[1], item[0]),
        )
        order: list[NodeKey] = []
        seen: set[NodeKey] = set()
        for key, _ in positive:
            if key not in seen:
                seen.add(key)  # type: ignore[arg-type]
                order.append(key)  # type: ignore[arg-type]
        for key, _ in ranked:
            if key not in seen:
                seen.add(key)
                order.append(key)

        view = View()
        used = 0
        covered: dict[str, set[int]] = {}  # path -> lines already paid for in that file
        full_elements: set[NodeKey] = set()
        skips = 0
        for key in order:
            path, element = key
            if path not in session.files:
                continue
            if element == "":
                if path in covered:
                    continue
                cost = self.size_of(key, Level.COLLAPSED)
                if used + cost <= budget:
                    view.add_file(path, Level.COLLAPSED)
                    covered[path] = self._skeleton_lines(path)
                    used += cost
                    skips = 0
                else:
                    skips += 1
            else:
                if key in full_elements:
                    continue
                lines = self._full_lines(key)
                if path in covered:
                    already = covered[path]
                    cost = self.size_of(key, Level.FULL) - self._cost_of_lines(path, lines & already)
                else:
                    already = self._skeleton_lines(path)
                    cost = (
                        self.size_of((path, ""), Level.COLLAPSED)
                        + self.size_of(key, Level.FULL)
                        - self._cost_of_lines(path, lines & already)
                    )
                if used + cost <= budget:
                    if path not in covered:
                        view.add_file(path, Level.COLLAPSED)
                        covered[path] = already
                    view.add_element(path, element, Level.FULL)
                    full_elements.add(key)
                    covered[path] |= lines
                    used += cost
                    skips = 0
                else:
                    skips += 1
            if skips >= MAX_CONSECUTIVE_SKIPS:
                break
        return view

    # ----------------------------------------------------------------- seeds / listing

    def seeds_from_views(self, views: Iterable[View], full_weight: int = 2) -> dict[NodeKey, float]:
        """Relevance seeds from several views (one per parent).

        Sums ``View.counts()`` over the views, mapping a FULL sighting (count 2) to
        ``full_weight`` and COLLAPSED to 1; each line range adds 1 to the file (as in
        ``counts()``) and 1 to every element it intersects (via :meth:`resolve_lines`).
        Keys that are not graph nodes are dropped.

        Args:
            views: Parent views.
            full_weight: Weight of a FULL sighting relative to COLLAPSED.

        Returns:
            Seed weights per node key, sorted by key.
        """
        seeds: dict[NodeKey, float] = {}

        def bump(key: NodeKey, amount: float) -> None:
            if key in self.nodes and amount:
                seeds[key] = seeds.get(key, 0.0) + amount

        for view in views:
            counts = view.counts()
            for (path, element), count in counts.items():
                file_view = view.files.get(path)
                level = file_view.levels.get(element) if file_view else None
                value = float(count)
                if level == Level.FULL:
                    value += full_weight - int(Level.FULL)
                bump((path, element), value)
            for path in sorted(view.files):
                for start, end in view.files[path].line_ranges:
                    for key in self.resolve_lines(path, start, end):
                        if key[1]:
                            bump(key, 1.0)
        return dict(sorted(seeds.items()))

    def signature_line(self, key: NodeKey) -> str:
        """First non-decorator line of an element's signature, stripped (``""`` for files).

        The line holding the ``def``/``class`` keyword wins, so a decorator whose
        arguments span several lines is skipped entirely.
        """
        node = self._code.get(key)
        if node is None:
            return ""
        lines = self._text(key[0])
        candidates = [
            lines[number - 1].strip()
            for number in sorted(_signature_lines(node, lines))
            if 1 <= number <= len(lines)
        ]
        for text in candidates:
            if text.startswith(("def ", "class ", "async def ")):
                return text
        for text in candidates:
            if text and not text.startswith("@"):
                return text
        return ""

    def candidate_list(
        self,
        view_or_keys: View | Sequence[NodeKey],
        session: Session,
        budget: int,
        *,
        seeds: dict[NodeKey, float] | None = None,
    ) -> str:
        """The LISTVIEWER candidate listing.

        One line per key: ``path::element  <signature first line>`` for elements and
        ``path`` for file nodes. A key sequence is listed in the given order; a View's
        keys are listed in rank order (ranked with ``seeds``, default
        ``seeds_from_views([view])``). Keys absent from the graph or ``session`` are
        skipped. Lines are added while the joined text stays within ``budget``
        characters (whole lines only).
        """
        if isinstance(view_or_keys, View):
            view = view_or_keys
            keys = {(path, element) for path, fv in view.files.items() for element in fv.levels}
            ranked = self.rank(seeds if seeds is not None else self.seeds_from_views([view]))
            position = {key: i for i, (key, _) in enumerate(ranked)}
            ordered = sorted(keys, key=lambda k: (position.get(k, len(position)), k))
        else:
            ordered = [(k[0], k[1]) for k in view_or_keys]
        lines: list[str] = []
        used = 0
        emitted: set[NodeKey] = set()
        for key in ordered:
            if key in emitted or key not in self.nodes or key[0] not in session.files:
                continue
            emitted.add(key)
            path, element = key
            if element:
                signature = self.signature_line(key)
                line = f"{path}::{element}  {signature}" if signature else f"{path}::{element}"
            else:
                line = path
            cost = len(line) + (1 if lines else 0)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)
