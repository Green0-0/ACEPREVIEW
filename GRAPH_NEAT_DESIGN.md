# Worker Graph + NEAT Design Contract

This document is the binding contract for the `ace.graph` (worker-graph runtime) and
`ace.neat` (evolution) packages. Where it leaves room, the module docstrings record the
reading taken. Do not change a public signature defined here without updating this file.
`ACE_SCHEMA.md` describes the implemented behaviour in prose.

Read `agent2/CURRENT_SCHEMA.md` for the agent2 runtime (Session, Workspace, Agent,
CodeFile, tools). Everything below builds on it and does not modify it (§11).

---

## 1. Concepts

A **worker graph** is a DAG of **workers**. A worker is one agent2 `Agent` run (or a
deterministic multi-step wrapper around one) that performs a task on its own fresh
`Workspace` over the session. Workers never share workspaces and never commit. Every worker receives the original task
prompt. The base state for all views and edits is the **pristine** repository (the
`Session` files at run start), never another worker's edited state.

There is an implicit **start node** (the prompt) and an explicit **end node** (`END`).
Start edges are not represented; "every worker has a path from input to output" reduces to
"every worker reaches END". Exactly one edge enters END and it must carry an edit.

Edges are **typed** by what they carry: `VIEW`, `EDIT`, or `REPORT`. A pair of workers may
be linked by several edges of different kinds. An edge `(src, dst, kind)` appears at most
once.

### 1.1 Payload kinds

| Kind   | Meaning |
|--------|---------|
| VIEW   | What a worker looked at, in canonical form (§3). "Information the parents saw." |
| EDIT   | A complete proposed change to the repository against the pristine state (§4). |
| REPORT | The worker's final free-text message: its analysis and account of the work. |

### 1.2 Worker types

| Type        | Inputs (kind: min..max)                     | Outputs           | Uses tools |
|-------------|---------------------------------------------|-------------------|------------|
| VIEWER      | none                                        | VIEW, REPORT      | viewing+search |
| LISTVIEWER  | VIEW 1..∞                                   | VIEW, REPORT      | no (two deterministic LLM steps) |
| EDITOR      | VIEW 0..∞, REPORT 0..∞                      | VIEW, EDIT, REPORT| viewing+search+editing |
| VIEWEDITOR  | VIEW 1..∞, REPORT 0..∞                      | VIEW, EDIT        | no (ACE output) |
| CRITIC      | VIEW 1..∞, EDIT 1..1, REPORT 0..∞           | VIEW, EDIT        | no (ACE output) |
| JUDGEMERGER | VIEW 1..∞, EDIT 2..∞, REPORT 0..∞           | VIEW, EDIT        | no (picks one edit, forwards it unchanged) |
| PICKMERGER  | VIEW 1..∞, EDIT 2..∞, REPORT 0..∞           | VIEW, EDIT        | no (ACE output) |

Rules:
- A worker's forwarded VIEW is the union of everything it was actually shown (after any
  expand/contract pass) plus, for tool-using workers, everything it viewed with tools.
- Workers that output ACE (VIEWEDITOR, CRITIC, PICKMERGER) emit no REPORT: the text around
  the code blocks is not a report.
- Inputs are **grouped by source worker**, never flattened, so a Critic can see which
  report and which view came from the author of the edit it is critiquing.
- An edge of kind K from `src` to `dst` is valid only if K is in `src`'s outputs and K is
  an input slot of `dst`.

### 1.3 Graph invariants

Every graph (and every genome after any mutation or crossover) must satisfy:

1. **Acyclic.**
2. **Reachability.** Every worker has a directed path to END.
3. **Arity.** For every worker and every kind, the number of incoming edges of that kind
   lies within the slot's `[min, max]`; kinds not in the slot table have zero incoming.
   Every edge kind is in the source's output set.
4. **End.** END has exactly one incoming edge and it is of kind EDIT.

Mutation and crossover are **propose → validate → repair-or-reject**. There is no implicit
"a node got created without a path, so add an editor" behaviour in the runtime. That
behaviour lives in the typed insertion operators (§7.3), which always produce valid graphs
or are rejected.

### 1.4 Failure policy

Nothing is expected to fail. Transient API errors are handled by retry infrastructure
inside the LLM client layer (`graph/llm.py`), not by graph redundancy. Any other failure
in a worker (format violation, unparseable list, ACE block that does not apply, overlapping
ACE blocks, wrong input count, empty report, missing payload at runtime) raises
`GraphFailure`. The graph run for that task is aborted and the task scores zero. A
`GraphFailure` raised by a worker or for a missing payload carries a `node_id` attribute
naming the failing worker.
Fitness is aggregated over tasks; a genome is only removed by selection.

---

## 2. Package layout

```
src/ace/graph/
    __init__.py
    types.py          Kind, WorkerType, SlotSpec, WorkerSpec, WORKER_SPECS, END,
                      GraphFailure, InvalidGraph
    graph.py          WorkerNode, WorkerEdge, WorkerGraph, validate(), topological order
    views.py          Level, FileView, View (canonical view form), merge, JSON
    rendering.py      render_view(), render_file_view(), measure()
    edits.py          Edit (path -> bytes|None), to/from Workspace, git patch
    ace.py            ACE block parsing + application against pristine, AceError
    codegraph.py      CodebaseGraph, personalized PageRank, budget selection, seeds
    recording.py      ViewRecorder + recording tool wrappers + pristine remap
    properties.py     property option tables per worker type, resolve()
    prompts.py        prompt text fragments, tool sets and assembly helpers
    llm.py            ModelSettings, retrying client factory
    runner.py         RunContext, WorkerInputs, WorkerOutputs, GraphRunner, GraphResult
                      (re-exports ModelSettings)
    workers/
        __init__.py   WORKERS: dict[WorkerType, type[Worker]] (loaded lazily), register
        base.py       Worker ABC, helpers to build an Agent from properties
        viewer.py
        listviewer.py
        editor.py
        vieweditor.py
        critic.py
        judgemerger.py
        pickmerger.py

src/ace/neat/
    __init__.py
    config.py         NeatConfig (pydantic) hyperparameters
    genome.py         NodeGene, ConnectionGene, Genome, to_graph(), JSON
    innovation.py     InnovationDB (persistent, global)
    mutation.py       mutation operators + typed insertion table
    crossover.py      crossover()
    speciation.py     distance(), Species, speciate(), dynamic threshold
    population.py     Population: init, generation step, selection, cost term
    history.py        History: lineage + species + fitness logs (JSONL)
    evaluation.py     Evaluator protocol, MockEvaluator, TaskEvaluator

scripts/run_graph.py  CLI: run one graph on a project, print the patch
scripts/run_neat.py   CLI: evolve graphs (MockEvaluator or a tasks file)
tests/graph_tests/    graph module tests
tests/neat_tests/     neat module tests
```

Python ≥ 3.12. Use dataclasses or pydantic (already a dependency). No new third-party
dependencies: PageRank is pure Python. Everything must be JSON-serialisable where the
contract says so.

---

## 3. Views (`graph/views.py`)

A view records what was seen, at what detail, against the pristine repository.

```python
class Level(IntEnum):
    COLLAPSED = 1   # signature/skeleton seen (docstring + headers, bodies hidden)
    FULL = 2        # full source seen

@dataclass
class FileView:
    path: str
    levels: dict[str, Level]            # element path -> level. Key "" = the file itself.
    line_ranges: list[tuple[int, int]]  # 1-indexed inclusive, merged, sorted; raw ranges
                                        # for unparsed files or spans that do not align
                                        # with an element.

@dataclass
class View:
    files: dict[str, FileView]          # path -> FileView

    def merge(self, other: View) -> View                 # new View; levels take max,
                                                         # ranges unioned and merged
    @staticmethod
    def union(views: Iterable[View]) -> View
    def add_file(self, path: str, level: Level) -> None  # levels[""] = max(existing, level)
    def add_element(self, path: str, element: str, level: Level) -> None
    def add_lines(self, path: str, start: int, end: int) -> None
    def counts(self) -> dict[tuple[str, str], int]      # see below
    def is_empty(self) -> bool
    def to_json(self) -> dict
    @staticmethod
    def from_json(data: dict) -> View
    def restrict_to(self, session: Session) -> View      # drop files/elements that do not
                                                         # exist in the pristine session,
                                                         # clamp ranges, drop empty files
    def copy(self) -> View

def normalize_element_path(file: File | CodeFile, element: str) -> str | None
def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]
```

Element paths are the `CodeNode.path` form (`MyClass.my_function`, no line numbers).
Always normalise `llm_path` (`MyClass.my_function.42`) to `path` when recording;
`add_element` strips a trailing all-digit segment. `add_lines` swaps a reversed range,
clamps the start to 1 and ignores a range ending before line 1. Overlapping and adjacent
ranges merge. JSON: `{"files": {path: {"levels": {element: 1|2}, "line_ranges": [[a, b]]}}}`.

**Counting for relevance.** `View.counts()` is per-View: each `(path, element)` key present
(`element == ""` for file-level) counts `int(level)`, FULL 2 and COLLAPSED 1, and each line
range adds 1 to the file key `(path, "")`. Relevance seeds across several parents are the
sum over the parents' views; `CodebaseGraph.seeds_from_views` computes it, maps a FULL
sighting to the consumer's `full_weight`, and adds 1 to every element a line range
intersects.

**Composition rule.** If one parent saw a collapsed file and another saw one element in
full, the merged view is the collapsed file with that element at FULL. Rendering (§3.1)
produces exactly that.

### 3.1 Rendering (`graph/rendering.py`)

```python
def render_file_view(file: File | CodeFile, view: FileView, *, prefix_lines: bool = True) -> str
def render_view(session: Session, view: View, *, prefix_lines: bool = True) -> str
def measure(text: str) -> int           # len(text); the budget unit is characters
```

Rendering is line based: each contribution of a `FileView` is a set of 1-indexed lines
and the union is printed, so contributions compose exactly as views merge. The head of an
element is its signature lines (decorators included) plus its docstring.

`render_file_view`:
- If `levels[""] == FULL`: the whole file in one `--- path ---` section.
- If `levels[""] == COLLAPSED`: one `--- path ---` section with the skeleton (every line
  outside top-level elements plus the head of each top-level element), with every listed
  element added in place: FULL shows all its lines, COLLAPSED shows its head plus the
  signature lines of its direct children, and any listed element adds its ancestors'
  heads.
- If `""` is absent: listed elements only, grouped under their outermost listed ancestor,
  one `--- path::element ---` section per group spanning that element, by start line.
- `line_ranges` are rendered as `--- path (lines a-b) ---` sections after the other
  sections, clamped to the file, excluding lines already shown; each remaining span is
  its own section.
- An element key stands for every element sharing that path (property getter and setter,
  `if`/`else` definitions).
- Unparsed files (`File` without `code_nodes`): `""` at any level renders the full file;
  otherwise only `line_ranges`. Unknown element keys are ignored.
- Shown lines carry the `N| ` prefix when `prefix_lines`. A run of hidden lines becomes
  one `... [BODY HIDDEN: N LINES] ...` line, unprefixed and indented like the first
  non-blank hidden line; a hidden run of blank lines only is printed as-is.

`render_view` renders files sorted by path, skips files absent from the session and empty
renderings, and joins sections with a blank line. The depth-based helpers in
`code_parser/interface/renderer.py` cannot express a union of selective expansions and
are not used; the prefix style and marker text match them.

---

## 4. Edits (`graph/edits.py`)

An edit is the complete new content of every changed file against pristine. This avoids
patch application entirely: diffs are derived for display.

```python
@dataclass
class Edit:
    files: dict[str, bytes | None]       # path -> new content; None = deleted

    @staticmethod
    def from_workspace(workspace: Workspace) -> Edit   # from workspace.files where
                                                       # original != current
    def to_workspace(self, session: Session) -> Workspace  # fresh Workspace(session)
                                                           # with write_file/delete_file
    def per_file_diff(self, session: Session) -> list[tuple[str, str]]  # git diffs
    def git_patch(self, session: Session) -> str       # per-file diffs joined with "\n"
    def unified_diff(self, session: Session) -> str    # alias of git_patch
    def is_empty(self) -> bool
    def touched_paths(self) -> set[str]
    def to_json(self) -> dict                          # bytes base64; None preserved
    @staticmethod
    def from_json(data: dict) -> Edit
```

`from_workspace` skips checkouts whose original and current bytes are equal.
`to_workspace` ignores deletion of a path the session lacks. An entry equal to pristine,
or such a deletion, produces no diff.

**Git patch.** `Workspace.diff()` is not used: it splits with `str.splitlines()` and drops
terminators and the no-newline marker, so its output does not apply. Each per-file diff is
`diff --git a/p b/p`, then `new file mode 100644` / `deleted file mode 100644` for a
creation / deletion, then `--- a/p` (`--- /dev/null`) and `+++ b/p` (`+++ /dev/null`), then
`difflib.unified_diff` hunks with 3 context lines. Content is decoded as UTF-8 with
`surrogateescape` and split on `\n` only, keeping terminators. A line without a terminator
is followed by `\ No newline at end of file`. An empty-file creation or deletion has no
hunks and no `---`/`+++` lines. Each file diff ends with `\n`. `git apply` accepts
`git_patch`.

The END node's result is the single incoming `Edit`. `GraphResult.edit` is that Edit and
`GraphResult.diff` its git patch.

### 4.1 ACE editing format (`graph/ace.py`)

Tool-less editing workers emit a sequence of markdown code blocks. The **first line inside
each code block** is a target header; the rest is the replacement body.

```
@@ path/to/file.py::Class.method          replace the element (primary, encouraged)
@@ path/to/file.py                        replace the whole file (override, discouraged)
@@ path/to/file.py::NEW                   create a new file with the body
@@ path/to/file.py::DELETE                delete the file (body ignored, may be empty)
@@ path/to/file.py::12-40                 replace lines 12..40 inclusive (override, discouraged)
@@ path/to/file.py::12                    same as 12-12
@@ path/to/file.py::Class.method::AFTER   insert the body after the element (new sibling)
@@ path/to/file.py::Class.method::BEFORE  insert the body before the element
```

Header parts are split on `::` and stripped; an empty part is malformed. Keywords are upper
case. A leading `./` is dropped and backslashes in the path read as `/`. A line span needs
`1 <= A <= B`. Element paths match `[A-Za-z_][\w.]*` and may be `CodeNode.path` or
`llm_path` (`Class.method.42`); a stale line suffix falls back to the path form. A `path`
shared by several elements (property getter and setter, `if`/`else` definitions) is
ambiguous and raises `AceError` listing the `llm_path` choices.

`AceFormat` (property `ace_format`) is an enum selecting which headers are permitted:

| Value             | Permitted headers |
|-------------------|-------------------|
| `element`         | element, NEW, DELETE, AFTER, BEFORE |
| `element_file`    | + whole file |
| `element_file_lines` | + line span |
| `file_lines`      | whole file, NEW, DELETE, line span (no element headers) |

```python
@dataclass
class AceBlock:
    path: str
    target: AceTarget      # ELEMENT | FILE | NEW | DELETE | LINES | AFTER | BEFORE
    element: str | None
    lines: tuple[int, int] | None
    body: str

def parse_ace(text: str, fmt: AceFormat) -> list[AceBlock]      # raises AceError
def apply_ace(blocks: list[AceBlock], session: Session, fmt: AceFormat) -> Edit  # raises AceError
def ace_instructions(fmt: AceFormat) -> str      # prompt text describing permitted headers
```

Parsing rules:
- A fence is three or more backticks, optionally indented and followed by a language tag,
  closed by a line of at least as many backticks and nothing else. Content lines lose up
  to the opening fence's indentation.
- A block whose first line does not start with `@@ ` is ignored (prose, examples), and so
  is a quoted unified-diff hunk (`@@ -12,6 +12,7 @@`). A message with zero ACE blocks, an
  unclosed fence whose first line is a header, a malformed header, or a header not
  permitted by `fmt` raises `AceError`.

Application rules:
- All targets are resolved against the pristine session. An unknown path or element, an
  element header on an unparsed file, a span outside the file, or NEW on an existing path
  raises `AceError`.
- ELEMENT replaces from the start of the element's first line (decorators included) to its
  end byte. AFTER inserts at the end of the element's last line, BEFORE at the start of
  its first line, indented like the element and separated by one blank line when nested,
  two at top level. Element bodies pass through `adapter.attempt_fix_formatting`, except a
  body already at the target indentation that contains less-indented lines, which is kept
  as given.
- LINES replaces whole lines; a newline is appended when the replaced span ended with one
  and the body does not; an empty body deletes the lines. FILE/NEW write the body
  verbatim. Text inserted into a mostly `\r\n` file uses `\r\n`.
- Byte ranges in one file must not overlap (half-open). A zero-width insertion conflicts
  with a range strictly containing its point, with another insertion at the same point,
  and with a replacement strictly containing its anchor element. Conflicts raise
  `AceError` ("overlapping"). A FILE/NEW/DELETE block must be the only block for its path.
- Per file, edits are applied bottom-up by (start byte, end byte) descending on a fresh
  `Workspace`; the result is `Edit.from_workspace`.
- `AceError` (in `ace.py`) subclasses `GraphFailure` (in `types.py`).

---

## 5. Codebase graph (`graph/codegraph.py`)

```python
NodeKey = tuple[str, str]      # (file path, element path); element "" = the file node

class CodebaseGraph:
    def __init__(self, session: Session) -> None       # builds from pristine session
    nodes: set[NodeKey]
    def neighbors(self, key: NodeKey) -> dict[NodeKey, float]   # weighted out-edges
    def rank(self, seeds: dict[NodeKey, float], *, damping: float = 0.85,
             iterations: int = 30) -> list[tuple[NodeKey, float]]    # personalized
                                                                      # PageRank, desc
    def select(self, seeds: dict[NodeKey, float], budget: int,
               session: Session) -> View
    def resolve_lines(self, path: str, start: int, end: int) -> list[NodeKey]
    def size_of(self, key: NodeKey, level: Level) -> int   # rendered chars estimate
    def element(self, key: NodeKey) -> CodeNode | None
    def seeds_from_views(self, views: Iterable[View], full_weight: int = 2
                         ) -> dict[NodeKey, float]
    def signature_line(self, key: NodeKey) -> str
    def candidate_list(self, view_or_keys: View | Sequence[NodeKey], session: Session,
                       budget: int, *, seeds: dict[NodeKey, float] | None = None) -> str
```

Edges (weights are module constants; parallel edges are summed):
- containment: file → top-level element, element → direct child (`CONTAINMENT_WEIGHT`
  1.0), and the reverse (`CONTAINMENT_BACK_WEIGHT` 0.5).
- sibling adjacency: consecutive elements with the same parent, by start byte, both
  directions (`SIBLING_WEIGHT` 0.3).
- imports (Python only, from the Tree-sitter tree, nested imports included): file →
  imported project file (`IMPORT_WEIGHT` 1.0). `a.b.c` resolves to `a/b/c.py` or
  `a/b/c/__init__.py`, else to a unique file whose module path ends with `a.b.c`.
  `from M import n` tries `M.n` before `M`. Relative imports resolve against the
  importing package.
- name references: `identifier` tokens attributed to the innermost enclosing element body,
  excluding definition names and import statements, that equal some other element's
  `CodeNode.name` → that element. Skip names shorter than `MIN_NAME_LENGTH` (3), Python
  keywords and soft keywords, builtins, `self` and `cls`. A name matching `n` elements adds
  `REFERENCE_WEIGHT / n` (0.7 / n) to each; repeats add nothing.

`rank` uses out-weights normalised per node; teleport and dangling mass go to the seed
distribution. Non-positive and unknown seeds are dropped; with none left, seeds are uniform
over file nodes. Every node is returned, by descending score then key. `resolve_lines`
returns the intersecting element keys by start line, or `[(path, "")]` when none intersects
(`[]` for an unknown path). `size_of` counts whole rendered lines including `N| ` prefixes;
a COLLAPSED file node is its skeleton plus header.

Node discovery uses `session.files`; a `CodeFile` contributes its `code_nodes` (deduped by
`.path`), a plain `File` contributes only its file node.

`select(seeds, budget, session)` is the **expand-and-contract** operation every consumer
uses: run `rank`, walk the ranked list, add nodes to a `View` at FULL (elements) or
COLLAPSED (file nodes) while the estimated rendered size (`size_of`) stays within `budget`
characters. Seeds with a positive score are always considered first in seed-weight order
before non-seed nodes. If seeds are empty, rank with uniform seeds over all file nodes.
Add the file node at COLLAPSED for any file that contributes an element, so rendering shows
a skeleton around expanded elements; the skeleton cost is paid once per file and lines
already paid for are not charged again. A candidate that does not fit is skipped; the walk
stops after `MAX_CONSECUTIVE_SKIPS` (50) consecutive skips. `session` is used only to skip
keys whose file it lacks. Return the View (never render here).

`seeds_from_views` sums `counts()` over the views, maps FULL sightings to `full_weight`,
adds 1 per intersecting element for each line range, and drops keys that are not nodes.
`candidate_list` prints `path::element  <signature line>` or `path` per key, keys in the
given order or, for a View, in rank order, whole lines within `budget` characters.

---

## 6. Recording views (`graph/recording.py`)

```python
class ViewRecorder:
    def __init__(self, session: Session) -> None
    view: View                                                    # live, un-remapped
    sightings: list[Sighting]                                     # raw log, call order
    def record_file(self, path: str, full: bool) -> None
    def record_element(self, path: str, element: str, *, file: File | None = None) -> None
                                                                   # FULL
    def record_at(self, path: str, line: int, *, file: File | None = None,
                  to_end: bool = False) -> None
        # innermost element containing the line at FULL; if none, lines
        # line..line+49 (to end of file with to_end), clamped
    def record_lines(self, path: str, start: int, end: int, *,
                     file: File | None = None) -> None
        # with file: clamped, and the bytes of that state are kept for remapping
    def finalize(self, workspace: Workspace) -> View
        # Remap to pristine: sightings of files absent from pristine are dropped; file
        # sightings keep their level; elements are looked up by path in the pristine file
        # (drop if missing); line ranges map from the state they were seen in (the
        # snapshot, else the final workspace file) to pristine through
        # SequenceMatcher(None, pristine, seen, autojunk=False) opcodes: lines in equal
        # blocks map by offset, all others are dropped; the result is restricted to the
        # pristine session.

def recording_tools(tool_names: Iterable[str] | Mapping[str, Tool],
                    recorder: ViewRecorder) -> dict[str, RecordingTool]
```

`recording_tools` wraps the named tools from `agent2.core.globals.TOOLS`, or the tool
objects of a `name -> Tool` mapping such as `agent.tools` (so per-agent schemas survive);
an unknown name raises `KeyError`. A `RecordingTool` exposes `.func` (the original
function, so `ToolCallHook` can inspect its signature), `.name`, `.description`,
`.openai_schema`, `.examples` and `__call__`. On call, it invokes the original tool and,
if the result is not an error string (does not start with `Error`), records the view
implied by the arguments: `view`/`view_path` (split on `::`), `view_files`/`view_file`
(`*` suffix or `full`), `views_at`/`view_at`, `view_element`, `view_multi_lines`/
`view_lines`. Multi-item tools re-check each item with the single-item viewing function
and record only items that succeed. Non-viewing tools are wrapped as pass-through. `open_*`
tools return `(content, continuation)` unchanged and count as a FULL view of their target.
A failure inside recording is logged and never changes the tool result. Workers install
the wrappers by assigning `agent.tools = recording_tools(agent.tools, recorder)` after
constructing the `Agent`.

---

## 7. Worker graph and genome

### 7.1 Types (`graph/types.py`)

```python
class Kind(str, Enum): VIEW = "view"; EDIT = "edit"; REPORT = "report"
class WorkerType(str, Enum):
    VIEWER, LISTVIEWER, EDITOR, VIEWEDITOR, CRITIC, JUDGEMERGER, PICKMERGER
END = "END"                     # the reserved node id of the end node

@dataclass(frozen=True)
class SlotSpec:
    kind: Kind
    min: int
    max: int | None             # None = unbounded

@dataclass(frozen=True)
class WorkerSpec:
    type: WorkerType
    inputs: dict[Kind, SlotSpec]
    outputs: frozenset[Kind]
    uses_tools: bool

WORKER_SPECS: dict[WorkerType, WorkerSpec]      # exactly the table in §1.2

class GraphFailure(Exception): ...               # task-level failure
class InvalidGraph(ValueError): ...              # invariant violation (message names it)
```

### 7.2 Graph (`graph/graph.py`)

```python
@dataclass
class WorkerNode:
    id: int                      # innovation number of the node gene
    type: WorkerType
    properties: dict[str, int]   # property name -> option index (see §8)

@dataclass(frozen=True)
class WorkerEdge:
    src: int
    dst: int | str               # node id or END
    kind: Kind

@dataclass
class WorkerGraph:
    nodes: dict[int, WorkerNode]
    edges: list[WorkerEdge]

    def validate(self) -> None                      # raises InvalidGraph
    def is_valid(self) -> bool
    def incoming(self, node_id: int | str) -> list[WorkerEdge]
    def outgoing(self, node_id: int) -> list[WorkerEdge]
    def inputs_by_source(self, node_id: int | str) -> dict[int, set[Kind]]
    def reaches_end(self, node_id: int) -> bool
    def topological_order(self) -> list[int]        # workers only, END excluded
    def end_source(self) -> int                     # the one worker feeding END
    def structure_hash(self) -> str                 # id- and property-independent; used
                                                    # for dedup (see below)
    def to_json(self) -> dict
    @staticmethod
    def from_json(data: dict) -> WorkerGraph

def minimal_graph(editor_properties: dict[str, int] | None = None) -> WorkerGraph
    # one EDITOR (id 0) -> END
```

Validation must check the four invariants of §1.3 and that every edge's kind is in the
source's outputs and in the destination's input table (END accepts only EDIT). It checks,
in order: node table consistency, edge legality (endpoints exist, no self-loops, no
duplicates, kinds), acyclicity, END, arity, reachability (edges of any kind). Error
messages name the offending node (`node 3 (critic)`) or edge (`edge 1->END [view]`).
`topological_order` breaks ties by the smallest id. `structure_hash` is Weisfeiler-Lehman
colour refinement: colours start as worker types (END as `END`), each of `len(nodes) + 1`
rounds hashes a node's colour with the sorted `(kind, colour)` multisets of its in- and
out-neighbours, and the digest is sha1 over sorted node colours and sorted
`(colour(src), colour(dst), kind)` edges. `from_json` does not validate.

### 7.3 Genome (`neat/genome.py`)

```python
@dataclass
class NodeGene:
    innovation: int
    type: WorkerType
    properties: dict[str, int]

@dataclass
class ConnectionGene:
    innovation: int
    src: int
    dst: int | str          # node innovation or END
    kind: Kind
    enabled: bool = True

@dataclass(eq=False)                     # identity equality, hashable by identity
class Genome:
    id: str                              # uuid hex
    nodes: dict[int, NodeGene]
    connections: dict[int, ConnectionGene]
    generation: int = 0
    parents: tuple[str, ...] = ()
    operators: list[str] = field(default_factory=list)   # names of operators applied
    fitness: float | None = None         # raw
    adjusted_fitness: float | None = None
    species_id: int | None = None

    def to_graph(self) -> WorkerGraph    # all node genes, enabled connections only;
                                         # node.id = innovation
    def validate(self) -> None           # gene-level checks, then to_graph().validate()
    def is_valid(self) -> bool
    def assert_types_match(self, other: Genome) -> None   # raises InvalidGraph
    def structure_hash(self) -> str
    def copy(self, *, new_id: bool = True) -> Genome      # deep copy; keeps bookkeeping
    def num_workers(self) -> int         # number of node genes
    def enabled_connections(self) -> list[ConnectionGene]
    def connections_into(self, node, *, enabled_only=False) -> list[ConnectionGene]
    def connections_from(self, node, *, enabled_only=False) -> list[ConnectionGene]
    def find_connection(self, src, dst, kind, *, enabled_only=False) -> ConnectionGene | None
    def has_connection(self, src, dst, kind, *, enabled_only=False) -> bool
    def add_node(self, innovations, key, properties=None) -> NodeGene
    def add_connection(self, innovations, src, dst, kind, *, enabled=True) -> ConnectionGene
    def to_json(self) -> dict
    @staticmethod
    def from_json(data: dict) -> Genome  # does not validate

def minimal_genome(innovations: InnovationDB, properties: dict[str, int] | None = None,
                   *, generation: int = 0) -> Genome   # ("init", 0, EDITOR) -> END
```

Node innovation numbers are **global node ids**: a node gene's innovation uniquely
identifies that node across all genomes, and the same structural event in any genome in
any generation yields the same innovation. Worker types are never mixed: innovation keys
include the worker type, and crossover aligns only by innovation, so matched genes always
share a type. Because this is a property of a pair of genomes, it is checked by
`assert_types_match` (also rejecting an innovation that is a node in one genome and a
connection in the other), which crossover calls. `validate()` checks gene-level
consistency: keys equal innovations, known types and kinds, every connection (enabled or
disabled) has existing endpoints, connection innovations are disjoint from node
innovations, and no two connection genes share `(src, dst, kind)`.

### 7.4 Innovation database (`neat/innovation.py`)

```python
class InnovationDB:
    def __init__(self) -> None
    def node(self, key: tuple) -> int          # key = ("op", <context...>, WorkerType)
    def connection(self, src: int, dst: int | str, kind: Kind) -> int
    def peek_node(self, key: tuple) -> int | None
    def peek_connection(self, src: int, dst: int | str, kind: Kind) -> int | None
    def save(self, path: Path) -> None   # JSON, atomic
    @staticmethod
    def load(path: Path) -> InnovationDB
```

Node and connection innovations share one counter starting at 0, so they never collide and
the minimal genome's editor gets id 0 on a fresh database. Keys are never reset between
generations ("full history"). Keys are saved as lists of tagged `[tag, value]` elements so
enum types round-trip. Node keys by operator:

| Operator        | Node key |
|-----------------|----------|
| initial         | `("init", index, type)` where index is the genome's node count when the node is created (its position in the seed-build sequence); every node created during initialisation uses this key |
| add_worker (listviewer row) | `("split", conn_innovation, LISTVIEWER)` |
| add_worker (critic row) | `("critic", conn_innovation, CRITIC)` — a CRITIC splitting an EDIT edge |
| add_worker (viewer row) | `("source", dst_node_innovation, Kind.VIEW, VIEWER)` — a new input-less worker feeding dst; also used for the fallback VIEWER that gives a new CRITIC or merger its VIEW input (dst = that critic or merger) |
| insert_merger   | `("merge", existing_end_source_innovation, candidate_innovation, JUDGEMERGER or PICKMERGER)` |
| add_worker (vieweditor row) | `("mirror", dst_node_innovation, VIEWEDITOR)` — a VIEWEDITOR fed by the same VIEW sources as dst |
| add_worker (editor row) | `("branch", view_source_innovation, EDITOR)` — an EDITOR fed by that VIEW producer |

A proposal whose key yields an innovation already present in the genome is rejected.
Connections are keyed on `(src, dst, kind)` using node innovations.

---

## 8. Properties (`graph/properties.py`)

Properties are categorical genes. Each worker type has a fixed, hardcoded table of
properties; each property has a tuple of options. A genome stores option **indices**.
Index 0 of every option tuple is the default. The prompt-slot texts live in
`prompts.py` (`ROLES`, `STRATEGIES`, `REPORT_FOCI`, `INSTRUCTIONS`); `properties.py`
builds `PROPERTIES` from them. `ace_format` values are plain strings; convert with
`AceFormat(value)`. Never call `str.format` on prompt text (it contains `{{FILES*}}`);
fill the ACE placeholder with `.replace(prompts.ACE_PLACEHOLDER, ace_instructions(fmt))`.
Mutation picks a property and a different index. Crossover picks one parent's index.
No LLM is involved in mutation.

```python
@dataclass(frozen=True)
class PropertySpec:
    name: str
    options: tuple[Any, ...]        # concrete values; JSON-serialisable
    description: str

PROPERTIES: dict[WorkerType, dict[str, PropertySpec]]

def default_properties(t: WorkerType) -> dict[str, int]            # all index 0
def random_properties(t: WorkerType, rng: random.Random) -> dict[str, int]
def resolve(t: WorkerType, indices: dict[str, int]) -> dict[str, Any] # name -> value
def validate_properties(t: WorkerType, indices: dict[str, int]) -> None  # raises ValueError
def build_system_prompt(t: WorkerType, resolved: dict[str, Any]) -> str
```

Required property names and value types (option contents are the property author's job;
every table needs at least 3 options, and prompt slots at least 4 meaningfully different
ones):

| Property          | Type  | Applies to | Meaning |
|-------------------|-------|------------|---------|
| `role`            | str   | all        | system prompt slot: who the agent is |
| `strategy`        | str   | all        | system prompt slot: how to approach the task (viewing strategy, editing discipline, judging rubric, etc.) |
| `report_focus`    | str   | VIEWER, LISTVIEWER, EDITOR | system prompt slot: what the report must contain |
| `instruction`     | str   | all        | the initial user-message framing placed before the task prompt |
| `context_budget`  | int   | LISTVIEWER, EDITOR, VIEWEDITOR, CRITIC, JUDGEMERGER, PICKMERGER | characters of rendered views offered to the agent |
| `max_steps`       | int   | VIEWER, EDITOR | agent loop steps |
| `tool_set`        | str   | VIEWER, EDITOR | name of a tool list in `prompts.TOOL_SETS` |
| `ace_format`      | str   | VIEWEDITOR, CRITIC, PICKMERGER | `AceFormat` value |
| `list_budget`     | int   | LISTVIEWER | characters of the candidate list offered before the agent picks |
| `view_full_weight`| int   | LISTVIEWER, VIEWEDITOR, CRITIC, JUDGEMERGER, PICKMERGER | relevance weight of a FULL sighting vs COLLAPSED (1..3) |

Implemented option tables (index 0 first): every prompt slot has 4 texts per type;
`context_budget` (12000, 6000, 24000, 48000); `max_steps` VIEWER (20, 10, 40), EDITOR
(30, 15, 60); `tool_set` VIEWER (viewer_standard, viewer_minimal, viewer_lines,
viewer_full), EDITOR (editor_standard, editor_minimal, editor_interactive, editor_full);
`ace_format` (element, element_file, element_file_lines, file_lines); `list_budget` (8000,
4000, 16000); `view_full_weight` (2, 1, 3). `validate_properties` reports unknown, missing
and out-of-range entries in one `ValueError`; `resolve` validates first.

`build_system_prompt` concatenates, separated by blank lines: role, strategy,
(report_focus if present), and a fixed per-type block from `prompts.OUTPUT_FORMAT_BLOCKS`
describing the output format; `{ace_instructions}` is left in place. Tool-using workers end
with a `{{FILES*}}` line filled by the `directory` constructor; other workers do not.
`prompts.assemble_user_prompt(instruction, task, sections)` builds the first user message:
the instruction, `# Task` with the prompt, then each non-empty `(header, body)` section.

`prompts.TOOL_SETS` for VIEWER: subsets of `{search, list_folder, search_folders, view,
view_path, view_files, view_file, views_at, view_at, view_element, view_multi_lines,
view_lines}`. For EDITOR: those plus subsets of `{replace_file, replace_lines,
replace_element, replace_element_at, open_file, open_lines, open_element, open_element_at,
make_empty_files, move_files}`. Never `delegate`, `message_subagent`, `cancel_subagent`,
`bash`, `tools`.

---

## 9. Runtime (`graph/runner.py`, `graph/workers/`)

```python
@dataclass
class ModelSettings:                      # DEFINED in graph/llm.py; runner.py re-exports it
    model: str
    api_base_url: str | None = None
    api_key_env_var: str | None = "ANTHROPIC_API_KEY"

@dataclass
class RunContext:
    session: Session
    prompt: str                                   # the original task prompt
    settings: ModelSettings
    client_factory: Callable[[], Any]             # returns an AsyncOpenAI-compatible client
    codegraph: CodebaseGraph | None = None        # built lazily from session on first access
    run_id: str = <fresh uuid hex>
    rollout_dir: Path | None = None               # where the Session saves rollouts;
                                                  # informational, the runner never reads it

@dataclass
class WorkerInputs:                               # every field defaults to {}; keys ascending
    views: dict[int, View]                        # by source node id
    edits: dict[int, Edit]
    reports: dict[int, str]
    source_types: dict[int, WorkerType]           # source node id -> its worker type

@dataclass
class WorkerOutputs:
    view: View | None = None
    edit: Edit | None = None
    report: str | None = None
    rollout_ids: list[str] = field(default_factory=list)

class Worker(ABC):                                 # graph/workers/base.py
    type: ClassVar[WorkerType]
    async def run(self, ctx: RunContext, node: WorkerNode, inputs: WorkerInputs,
                  props: dict[str, Any]) -> WorkerOutputs
    # base.py also provides:
    def make_agent(self, ctx, node, props, *, tools: list[str], system_prompt: str,
                   user_prompt: str, max_steps: int,
                   constructors: list[str] | None = None) -> Agent
    # builds an in-memory AgentConfig (path f"graph/{node.type.value}/{node.id}",
    # subagent_name f"{type}_{id}", prompt wrappers "{{PROMPT}}", workspace_policy
    # ISOLATED, no subagents, hooks ["tool_call"] iff tools else [], constructors default
    # ["default"] plus "directory" iff tools), a fresh Workspace(ctx.session), passes
    # client=ctx.client_factory(), then sets agent.view_recorder and wraps agent.tools
    # with recording_tools when tools are present. Forbidden tools raise ValueError.

@dataclass
class GraphResult:
    edit: Edit
    diff: str
    outputs: dict[int, WorkerOutputs]
    order: list[int]

class GraphRunner:
    def __init__(self, graph: WorkerGraph,
                 workers: dict[WorkerType, type[Worker]] | None = None) -> None  # validates
    async def run(self, ctx: RunContext) -> GraphResult   # raises GraphFailure
```

Execution: repeatedly take every not-yet-run worker whose sources have all finished (one
level, ascending id) and run them concurrently as tasks until all finish or one fails; on a
failure cancel the rest and raise the failure of the lowest node id. Properties are
resolved with `properties.resolve` right before a worker runs. A worker receives exactly
the payloads carried by its incoming edges (an EDIT edge from `src` carries
`outputs[src].edit`, etc.). As soon as a worker finishes, each of its outgoing edges (END
included) must have a non-`None` payload, else `GraphFailure` names the edge. The END
result is `outputs[end_source].edit`. Non-`GraphFailure` exceptions become
`GraphFailure("node N (type) failed: ...")`; every worker failure gets a `node_id` attribute.
`GraphResult.order` lists worker ids in start order. Worker classes come from
`ace.graph.workers.WORKERS` unless `workers` overrides them.

Worker helpers (in `base.py`): `render_inputs_views(ctx, inputs, budget, full_weight) ->
tuple[str, View]` takes seeds from `ctx.codegraph.seeds_from_views` over the input views,
calls `ctx.codegraph.select`, renders, and returns both the text and the View actually
shown (which becomes the forwarded view); with no input views it returns `("", View())`.
`render_reports(inputs) -> str` labels each report `REPORT from worker {id} ({type})`.
`render_edits(ctx, inputs) -> tuple[str, list[int]]` labels each diff
`CANDIDATE <n> (worker <id>)` in sorted source id order, followed by the git patch in a
```` ```diff ```` fence (longer than any backtick run in it) or `(no changes)`, and returns
the source ids (candidate `n` is `ids[n - 1]`). Also: `run_agent` (wraps non-`GraphFailure`
errors and a `None` answer), `followup_agent`, `build_user_prompt`, `system_prompt_for`
(fills the ACE placeholder), `parse_ace_output`, `first_int`.

Per-worker behaviour:

- **VIEWER**: agent with `tool_set` tools (viewing/search only), `max_steps`; outputs
  recorded view (finalized) + final message as report.
- **LISTVIEWER**: no tools. Seeds from input views (`view_full_weight`). Step 1 user
  message: task prompt + a candidate list built by `ctx.codegraph.candidate_list` over the
  keys of `ctx.codegraph.select(seeds, list_budget)` in rank order (falling back to the
  whole ranked graph when that list is empty). The agent replies with a code block
  containing one `path::element` or `path` per line. Parse code-block lines (all lines if
  there is no block), stripping bullets, numbering, backticks and anything after the first
  token; drop entries that do not resolve in the pristine session; if none resolve raise
  `GraphFailure`. Expand the chosen items with `select` to `context_budget` seeded by
  them, merge with the selection, render, send as a followup; the agent replies with the
  report. Output view = what was rendered; report = final message.
- **EDITOR**: agent with `tool_set` (viewing+search+editing). If views are present they
  are rendered to `context_budget` (FULL weight 2; EDITOR has no `view_full_weight`) and
  appended to the user prompt as context, likewise reports. Outputs: recorded view merged
  with the rendered input view, `Edit.from_workspace`, final message as report. An empty
  edit is not a failure.
- **VIEWEDITOR**: no tools. Prompt = task + rendered views (+ reports). One LLM step;
  parse ACE with `ace_format`; apply → Edit. View = rendered view. An empty rendering
  raises `GraphFailure`.
- **CRITIC**: like VIEWEDITOR plus the single input edit shown as a diff under
  `# Candidate edit under review ...`, labelled `from worker <id> (<type>)`, with the
  author's report (if any) directly beneath it; other reports follow separately. The agent
  must output the complete new edit set in ACE format; nothing of the input edit is carried
  over.
- **JUDGEMERGER**: no tools. Prompt = task + rendered views + all candidate diffs + reports.
  Agent replies with exactly one candidate number (parse the first integer in the
  message; none or out of range → `GraphFailure`). Output edit = that candidate's Edit
  object, unchanged. View = rendered view.
- **PICKMERGER**: like JUDGEMERGER but outputs ACE: a whole new edit set composed from the
  best parts.

VIEWER, LISTVIEWER and EDITOR raise `GraphFailure` on an empty report. Mergers need at
least two input edits and a non-empty rendering. All ACE-producing workers and JUDGEMERGER
run with `max_steps = 1` and no hooks; LISTVIEWER runs two steps and no hooks.

### 9.1 LLM client (`graph/llm.py`)

```python
def make_client_factory(settings: ModelSettings, *, max_retries: int = 5,
                        backoff: float = 1.0, sleep=asyncio.sleep
                        ) -> Callable[[], RetryingClient]
```

Returns a factory producing a fresh `RetryingClient` per call around
`AsyncOpenAI(api_key=os.environ[api_key_env_var], base_url=api_base_url, max_retries=0)`.
`chat.completions.create` retries on connection errors, timeouts, rate limits (429) and
any status >= 500, waiting `min(backoff * 2**attempt, 30)` seconds, for at most
`max_retries + 1` calls; it does not retry on 4xx other than 429. Other attributes fall
through to the wrapped client. The SDK's own retries are disabled so attempts are not
multiplied. `ModelSettings` is defined here and re-exported by `runner.py`. Tests inject a
fake factory.

---

## 10. Evolution (`neat/`)

### 10.1 Hyperparameters (`neat/config.py`)

```python
class SpeciationConfig(BaseModel):
    enabled: bool = True
    target_species: int = 5                 # K
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_properties: float = 0.5
    initial_threshold: float = 3.0
    threshold_step: float = 0.15            # relative adjustment per generation toward K:
                                            # threshold *= (1 - step) if < K species,
                                            # *= (1 + step) if > K; floor 0.05
    stagnation_generations: int = 15
    elitism_per_species: int = 1
    survival_fraction: float = 0.4

class MutationConfig(BaseModel):
    p_property: float = 0.8                 # probability a child gets property mutations
    p_property_per_gene: float = 0.1
    p_add_connection: float = 0.3
    p_add_worker: float = 0.15
    p_insert_merger: float = 0.05
    p_remove_connection: float = 0.05
    p_remove_worker: float = 0.05
    p_toggle_enable: float = 0.02
    max_attempts: int = 20                  # per operator before giving up

class NeatConfig(BaseModel):
    population_size: int = 20
    generations: int = 10
    node_cost: float = 0.02                 # λ: adjusted = raw - λ * num_workers
    crossover_rate: float = 0.75
    initial_min_workers: int = 1
    initial_max_workers: int = 5
    max_workers: int = 12
    seed: int = 0
    speciation: SpeciationConfig = SpeciationConfig()
    mutation: MutationConfig = MutationConfig()
    history_dir: Path = Path(".ace/neat")

    @classmethod
    def from_toml(cls, path) -> NeatConfig    # top-level fields + [speciation]/[mutation]
                                              # tables, or a single [neat] root table
    @classmethod
    def load(cls, path | None) -> NeatConfig  # defaults when None
    def to_toml_dict(self) -> dict            # plain dict, round-trips via model_validate
```

All models forbid extra keys. Probabilities and fractions lie in `[0, 1]`;
`population_size >= 2`; `1 <= target_species <= population_size`;
`1 <= initial_min_workers <= initial_max_workers <= max_workers`; `node_cost >= 0`;
`0 <= threshold_step < 1`; `initial_threshold > 0`; `max_attempts >= 1`;
`stagnation_generations >= 1`. Violations raise `pydantic.ValidationError`.

### 10.2 Mutation (`neat/mutation.py`)

```python
@dataclass
class MutationContext:
    rng: random.Random
    innovations: InnovationDB
    config: MutationConfig
    max_workers: int

def mutate(genome: Genome, ctx: MutationContext) -> Genome      # returns a new, valid genome
def mutate_properties(genome, ctx) -> bool
def add_connection(genome, ctx) -> bool
def add_worker(genome, ctx, *, key_prefix: str | None = None, row: str | None = None) -> bool
def insert_merger(genome, ctx, candidate: int | None = None, *,
                  key_prefix: str | None = None) -> bool
def remove_connection(genome, ctx) -> bool
def remove_worker(genome, ctx) -> bool
def toggle_enable(genome, ctx) -> bool
def random_valid_genome(innovations, rng, config, *, min_workers: int, max_workers: int,
                        generation: int = 0) -> Genome
OPERATORS: dict[str, Callable[..., bool]]
ADD_WORKER_ROWS = ("listviewer", "critic", "viewer", "vieweditor", "editor")
```

Each operator mutates in place, returns True on success, and leaves the genome unchanged
(and valid) on failure. Operators try up to `max_attempts` random proposals, each on a copy
validated with `Genome.is_valid()`; the first valid one is swapped in. A proposal that
would exceed `max_workers` is rejected. `key_prefix` makes every created node use the key
`(key_prefix, len(genome.nodes), type)`.

`mutate` copies the genome with a new id, resets `fitness`, `adjusted_fitness` and
`species_id`, applies `mutate_properties` with probability `p_property` and each
structural operator with its probability, and appends applied operator names to the
copy's existing `operators`; it does not set `generation` or `parents`. If the result is
invalid it returns a fresh copy of the input with `"noop"` appended; an invalid input
raises `InvalidGraph`. `mutate_properties` mutates each node with `p_property_per_gene`
and forces one random node when none was picked. `add_connection` adds (or re-enables) a
legal, acyclic `(src, dst, kind)` edge between workers and never targets END.
`remove_connection` deletes a random enabled gene. `toggle_enable` flips any gene.
`random_valid_genome` starts from `minimal_genome` with random editor properties and
applies `add_worker` (75%) or `insert_merger` (25%) with `key_prefix="init"` until a
worker count drawn from `[min_workers, max_workers]` is reached.

**Typed insertion table** for `add_worker` (choose a row at random, then a random site):

| Site | Inserted worker and wiring |
|------|----------------------------|
| any enabled VIEW edge `s→d` | LISTVIEWER `n`: `s→n` VIEW, `n→d` VIEW; original edge disabled. |
| any enabled EDIT edge `s→d` (d may be END) | CRITIC `n`: `s→n` EDIT, `n→d` EDIT; plus VIEW from `s` if `s` outputs VIEW, else VIEW from a random VIEW-producing ancestor of `s` or a new VIEWER source; original EDIT edge disabled. |
| any node `d` with a VIEW slot | new VIEWER `n` as source: `n→d` VIEW (+ REPORT with probability 0.5 if `d` accepts). |
| any node `d` with ≥1 enabled VIEW input | new VIEWEDITOR `n` fed by the same VIEW sources as `d`; wire `n→` a random existing merger that is not an ancestor of `n` (EDIT), else `insert_merger` with `n` as candidate. |
| any VIEW-producing node `v` with no enabled edge to an EDITOR | new EDITOR `n`: `v→n` VIEW (+ REPORT if `v` outputs it); then `insert_merger` with `n` as candidate. |

A row is picked uniformly per attempt, then a random site; a row without a site rejects the
attempt.

`insert_merger(candidate=None)`: pick an existing second EDIT producer `c` that does not
reach END through an enabled EDIT path, else create one with the EDITOR or VIEWEDITOR row
(random order); take the current END source `e`; create
JUDGEMERGER or PICKMERGER `m` (random); edges `e→m` EDIT, `c→m` EDIT, `m→END` EDIT, and
VIEW edges into `m` from every VIEW-producing member of `{e, c}` (if neither produces VIEW,
from a VIEW-producing ancestor, else create a VIEWER source); disable `e→END`. Optional
REPORT edges from `e`,`c` if they output REPORT, with probability 0.5 each.

`remove_worker` removes a random worker other than the END source (the END source is
never removed) and all its edges. A merger left with exactly one enabled EDIT input is
collapsed: the remaining candidate inherits every enabled outgoing edge of the merger
whose kind it outputs (so a removed candidate of the END-source merger hands `→END` to the
other), and the merger is removed. Any other orphan created by the removal is rejected
(validate fails). `remove_connection` only succeeds if the result validates (so required
slots are never emptied). `toggle_enable` likewise.

### 10.3 Crossover (`neat/crossover.py`)

```python
def crossover(a: Genome, b: Genome, rng: random.Random, *, innovations: InnovationDB,
              max_workers: int | None = None) -> Genome
```

`max_workers` (optional): a child with more workers than this (possible when equally fit
parents both contribute disjoint genes) falls back like an unrepairable child. Population
passes `cfg.max_workers`.

Standard NEAT alignment by innovation. "Fitter" compares `adjusted_fitness` when both
are set, else raw `fitness`, else the parents are equal (and `a` is preferred where one
must be). Matched node genes: each property picked at random from either parent
(pick-one, never merged). Matched connections: enabled flag from a random parent (if
disabled in either parent, 75% chance disabled, as in NEAT). Disjoint and excess genes come
from the fitter parent; when fitness is equal, from both. The child has
`generation = max(parent generations) + 1`, `parents = (a.id, b.id)` and
`operators = ["crossover"]`; no innovations are allocated. The child is validated; if
invalid, repair step by step, stopping at the first valid state: (1) remove connections
whose endpoints are missing; (2) fix END: with no enabled input enable one disabled EDIT
input, with several enabled keep one; prefer the fitter parent's END source, else the
lowest innovation; (3) remove nodes that cannot reach END through enabled edges, to a
fixed point; (4) remove nodes whose required input slots are unmet, re-applying (3), to a
fixed point. If still invalid, or the child has more than `max_workers` workers, return a
copy of the fitter parent with a fresh id, the child's generation and parents,
`operators == ["crossover_fallback"]` (a fresh list) and fitness and species reset.
`assert_types_match` is called first.

### 10.4 Speciation (`neat/speciation.py`)

```python
def distance(a: Genome, b: Genome, cfg: SpeciationConfig) -> float
    # δ = c_excess*E/N + c_disjoint*D/N + c_properties*H over node and connection genes
    # (E, D over the union of node and connection innovations, enabled or not; a gene is
    # excess when above the other genome's largest innovation; N = 1 always: genomes are
    # small by design and the classic switch to N = gene count at 20 genes collapsed
    # species mid-run); H = mean over matched node genes of the fraction of property names
    # whose indices differ (0 if no matched nodes).

@dataclass(eq=False)                       # identity equality, hashable by identity
class Species:
    id: int
    representative: Genome
    members: list[Genome] = []
    created: int = 0
    best_fitness: float = -inf
    last_improved: int = 0

class Speciator:
    def __init__(self, cfg: SpeciationConfig, rng: random.Random | None = None) -> None
                                           # rng picks representatives; default Random(0)
    threshold: float
    species: dict[int, Species]
    def speciate(self, population: list[Genome], generation: int) -> None
        # each species first picks a random previous member as representative and clears
        # its members; a genome joins the first species (ascending id) within threshold,
        # else founds one (created = last_improved = generation); species ids are never
        # reused; empty species are dropped. Does not adjust the threshold.
    def set_threshold_for(self, representatives: list[Genome]) -> None
        # threshold = min pairwise distance among representatives - 1e-6, so each is in
        # its own species; floor at 0.05
    def adjust_threshold(self) -> None     # relative step toward target_species: fewer
                                           # species -> threshold *= (1 - threshold_step),
                                           # more -> *= (1 + threshold_step); floor 0.05
    def share_fitness(self) -> None        # adjusted = (adjusted + shift) / len(members),
                                           # shift = -min adjusted over the population if
                                           # negative, else 0; raw fitness never changed
    def allocate_offspring(self, total: int) -> dict[int, int]
        # proportional to summed adjusted fitness (shifted by -min if negative; member
        # counts if all zero), largest-remainder rounding, ties by ascending species id
    def best_genome(self) -> Genome | None
    def prune_stagnant(self, generation: int) -> None   # never remove the species holding
                                                        # the best genome; no-op if no genome
                                                        # has a raw fitness; last_improved
                                                        # moves only on strict improvement;
                                                        # removes species with generation -
                                                        # last_improved > stagnation_generations
```

When `enabled=False`, `speciate` places everyone in species 0, no sharing, no pruning, no
threshold adjustment, and `allocate_offspring` returns `{0: total}`.

### 10.5 Population (`neat/population.py`)

```python
class Population:
    def __init__(self, cfg: NeatConfig, evaluator: Evaluator, *,
                 innovations: InnovationDB | None = None, history: History | None = None,
                 rng: random.Random | None = None)
        # rng defaults to Random(cfg.seed) and is shared with the speciator; history
        # defaults to History(cfg.history_dir); innovations default to the history's
        # innovations.json when present, else a fresh DB
    genomes: list[Genome]
    generation: int
    stats: list[dict]
    def initialize(self) -> None
        # cfg.population_size genomes from random_valid_genome with
        # initial_min..initial_max workers ("init" keys), re-rolled up to 10 times on an
        # exact duplicate; operators == ["init"]; saves the innovation DB.
    async def evaluate(self) -> None
        # evaluates genomes whose fitness is None, concurrently; adjusted = raw -
        # node_cost * num_workers; logs each genome after scoring
    async def bootstrap_species(self) -> None
        # after gen-0 evaluation: choose K representatives greedily by adjusted fitness
        # (ties: fewer workers), skipping genomes within 0.05 of a chosen one; regenerate
        # the rest as mutants of the representatives (round robin), evaluate them; set the
        # threshold via set_threshold_for(representatives); speciate the representatives
        # (one species each); every other genome joins the species of its nearest
        # representative. K = 1 when speciation is disabled.
    async def step(self) -> None
        # generation += 1; speciate; adjust_threshold; share; prune stagnant; allocate
        # offspring; per species: min(elitism_per_species, quota) elites carried unchanged
        # and not re-evaluated, survivors = top max(1, ceil(survival_fraction * size)),
        # children by crossover of two survivors (crossover_rate, max_workers passed)
        # then mutation, else mutation of one survivor; every child carries >= 1
        # mutation operator; shortfall filled with mutants of best(); evaluate; log.
    async def run(self) -> Genome        # initialize, evaluate, bootstrap if empty,
                                         # then step until generation == cfg.generations
    def best(self) -> Genome             # highest raw fitness; ties: fewer workers
    def generation_stats(self) -> dict
    def save_state(self, path) -> None   # config, generation, genomes, speciator, species,
                                         # innovation DB, RNG state (JSON, atomic)
    def load_state(self, path) -> None   # restores in place
```

Adjusted fitness is recomputed from raw at the start and end of `step`; sharing divides it
only transiently (allocation and crossover parent comparison), so logged adjusted values
are never shared values. Plain `speciate` is not used in the bootstrap because it would
found extra species for mutants beyond the tight threshold, so the bootstrap would not
yield K species.

### 10.6 History (`neat/history.py`)

```python
class History:
    def __init__(self, directory: Path) -> None
    def log_genome(self, genome: Genome) -> None          # genomes.jsonl: full genome JSON,
                                                          # parents, operators, species,
                                                          # fitness, generation, structure_hash
    def log_species(self, generation: int, species: list[Species], threshold: float) -> None
    def log_generation(self, generation: int, stats: dict) -> None
    def save_innovations(self, db: InnovationDB) -> None
    def load_innovations(self) -> InnovationDB | None
    def lineage(self, genome_id: str) -> list[dict]        # walk parents back to gen 0
```

Files under the directory (created on the first write), one JSON object per line:

```
genomes.jsonl      {"id", "generation", "parents", "operators", "species_id", "fitness",
                    "adjusted_fitness", "structure_hash", "num_workers", "genome"}
species.jsonl      {"generation", "threshold", "species": [{"id", "size", "member_ids",
                    "representative_id", "best_fitness", "created", "last_improved"}]}
generations.jsonl  {"generation", **stats}
innovations.json   InnovationDB.save format (a single JSON document)
```

Every genome ever evaluated is logged exactly once (an id already present, including from
an earlier process, is skipped) with its parent ids and the operator list, so the full
history is reconstructible. `species.jsonl` holds the bootstrap species as generation 0 and,
for each step, the species used to breed it. Malformed lines are skipped with a warning.
`lineage` returns the genome's own record first and its logged ancestors breadth-first,
oldest last.

### 10.7 Evaluation (`neat/evaluation.py`)

```python
class Evaluator(Protocol):
    async def evaluate(self, genome: Genome) -> float      # raw fitness, higher is better

class MockEvaluator:
    """Deterministic structural score for tests: rewards specific structures."""
    def __init__(self, *, noise: float = 0.0, scores: dict[str, float] | None = None,
                 delay: float = 0.0) -> None
    def score(self, genome: Genome) -> float

@dataclass
class Task:
    project_dir: Path
    prompt: str
    scorer: Callable[[Edit, Session], Awaitable[float] | float]

class TaskEvaluator:
    def __init__(self, tasks: list[Task], settings: ModelSettings,
                 client_factory: Callable[[], Any] | None = None,
                 concurrency: int = 4, rollout_dir: Path | None = None) -> None
    async def evaluate(self, genome: Genome) -> float
        # mean over tasks; a GraphFailure on a task scores 0 for that task

def nonempty_diff(edit: Edit, session: Session) -> float       # 1.0 iff edit non-empty
def resolve_scorer(spec: str) -> Scorer        # "nonempty_diff" or "package.module:function"
def load_tasks(path: Path | str) -> list[Task] # JSON list of {"project_dir", "prompt",
                                               # "scorer"}; relative dirs resolve against
                                               # the file's directory
```

MockEvaluator score: `1.0 * distinct worker types + 0.5 * [END source is a merger] +
0.25 * CRITIC count - 0.1 * max(0, workers - 6) + noise * (2u - 1)`, with `u` drawn from
`random.Random(int(structure_hash, 16))`. A `scores` entry for the structure hash overrides
the formula. Properties never affect the score.

TaskEvaluator: each task gets a fresh `Session` and `RunContext`; the client factory
defaults to `make_client_factory(settings)`. One semaphore of size `concurrency` bounds
running task graphs across all concurrent `evaluate` calls. **Rollout isolation:** task `i`
of genome `g` uses the absolute rollout dir `<root>/<g.id>/<i>/` for both the Session and
`RunContext.rollout_dir`, where `root` is `rollout_dir` or, when `None`, a private temporary
directory created on first use and kept. Nothing is written into a project folder, and each
Session starts without rollouts to load. A root inside a task's project folder on a
non-hidden path raises `ValueError`, because `Session.refresh_files` would read the rollouts
as project files and change the pristine base. A `GraphFailure` scores 0.0 and is logged;
scorer and Session errors propagate. `last_results` and `results_by_genome` hold per-task
records.

SWE-bench plugs in as a list of `Task`s whose scorer is a `module:function` called as
`scorer(edit, session)` with the END `Edit` and the task's `Session`; it materialises the
edit (`edit.to_workspace(session)` or `edit.git_patch(session)`) and runs the harness.

---

## 11. Permitted changes to existing code

ACE requires no changes to agent2.

- `core/tools/tool.py`: none. Recording wrappers duplicate the `Tool` interface.
- `core/agent.py`: none. Workers construct `AgentConfig` in memory and pass `client=`.
- `code_parser/interface/renderer.py`: none; ACE renders with its own line-based renderer.
- `core/globals.py`: do not register new tools there; recording wrappers are built per
  worker.

## 12. Testing conventions

- Fake LLM clients live in `tests/graph_tests/fakes.py` (the `FakeClient` pattern from
  agent2's `tests/core_tests/test_lifecycle.py`); tests import them from there.
- Fixtures build a small temporary project directory with a few Python files; never
  depend on network or on `ANTHROPIC_API_KEY`.
- `uv run pytest -q` from the project root runs `tests/graph_tests` and `tests/neat_tests`
  and must pass.

---

## 13. Run recording (`neat/history.py`, `neat/evaluation.py`, `neat/population.py`, `graph/runner.py`)

Every evolution run writes a self-describing **run directory** (`cfg.history_dir`, default
`.ace/neat/<run_id>` when the CLI is used; `History(directory)` writes wherever it is pointed).
All files are append-only JSONL except the three JSON snapshots. Timestamps are ISO-8601 UTC
strings; durations are float seconds. Enums are written as their values. Nothing in a run
directory is ever rewritten except `run.json`, `population_state.json` and `best_genome.json`.

| File | One record per | Written by |
|------|----------------|------------|
| `run.json` | run (rewritten on every generation end and at run end) | Population |
| `events.jsonl` | progress event | Population, TaskEvaluator |
| `generations.jsonl` | generation (including generation 0 bootstrap) | Population |
| `genomes.jsonl` | genome, exactly once | Population (after evaluation) |
| `species.jsonl` | speciation snapshot | Population |
| `evaluations.jsonl` | (genome, task) evaluation | TaskEvaluator / MockEvaluator |
| `innovations.json` | snapshot of the InnovationDB | Population |
| `population_state.json` | resumable snapshot | Population / CLI |
| `best_genome.json` | best genome + its graph | Population / CLI |
| `rollouts/<genome_id>/<task_index>/<rollout_id>.json` | agent rollout (agent2 format) | graph runtime |

### 13.1 `run.json`

```json
{"run_id": str, "ace_version": str, "status": "running" | "finished" | "failed",
 "started_at": ts, "finished_at": ts | null, "duration_s": float | null,
 "resumed_from": str | null, "seed": int,
 "config": <NeatConfig.to_toml_dict()>,
 "model": {"model": str, "api_base_url": str | null, "api_key_env_var": str | null},
 "evaluator": {"type": "mock" | "task", "tasks": [{"index": int, "project_dir": str,
               "prompt": str, "scorer": str}]},
 "generations_completed": int, "best_genome_id": str | null, "best_fitness": float | null,
 "error": str | null}
```

The API key is never written. `ace_version` is `importlib.metadata.version("ace")` or "0".

### 13.2 `events.jsonl`

`{"ts": ts, "event": name, ...}` with these names and extra keys:
`run_start` (run_id), `generation_start` (generation), `evaluation_start` (genome_id,
task_index), `evaluation_end` (genome_id, task_index, score, failure, duration_s),
`genome_evaluated` (genome_id, fitness, adjusted_fitness, duration_s), `speciated`
(generation, species_count, threshold), `generation_end` (generation, duration_s,
best_fitness), `run_end` (status, duration_s), `error` (message). The WebUI tails this file
for live progress.

### 13.3 `generations.jsonl`

```json
{"generation": int, "started_at": ts, "finished_at": ts, "duration_s": float,
 "evaluated": int, "elites": int, "species_count": int, "threshold": float,
 "best_fitness": float, "mean_fitness": float, "median_fitness": float,
 "best_adjusted": float, "mean_adjusted": float, "mean_workers": float,
 "best_genome_id": str, "operator_counts": {op: int},
 "population": [{"genome_id": str, "species_id": int | null, "fitness": float,
                 "adjusted_fitness": float, "num_workers": int, "structure_hash": str,
                 "is_elite": bool, "parents": [str], "operators": [str],
                 "worker_types": {type: int}}]}
```

`population` is the full evaluated population of that generation (elites included). The
existing stat keys keep their names; new keys are additive.

### 13.4 `genomes.jsonl` (unchanged keys plus)

`created_at` (ts), `evaluated_at` (ts), `eval_duration_s` (float), `worker_types`
({type: count}), `task_scores` ([float] in task index order, or [] for mock).

### 13.5 `evaluations.jsonl`

One record per (genome, task), written when the task finishes (success or failure):

```json
{"genome_id": str, "generation": int, "task_index": int,
 "task": {"project_dir": str, "prompt": str, "scorer": str},
 "started_at": ts, "finished_at": ts, "duration_s": float,
 "score": float, "failure": str | null, "failure_node_id": int | null,
 "failure_type": str | null,
 "diff": str | null, "touched_paths": [str], "diff_chars": int,
 "order": [int],
 "workers": [{"node_id": int, "type": str, "properties": {name: value},
              "started_at": ts, "finished_at": ts, "duration_s": float,
              "status": "ok" | "failed" | "skipped", "error": str | null,
              "rollout_ids": [str], "rollout_dir": str,
              "llm_calls": int, "tool_calls": int,
              "report": str | null, "view": <View.to_json()> | null,
              "view_summary": {"files": int, "entries": int, "rendered_chars": int},
              "edit_touched_paths": [str], "edit_chars": int}],
 "inputs": {node_id: {"views_from": [int], "edits_from": [int], "reports_from": [int]}}}
```

`MockEvaluator` writes `{"genome_id", "generation", "task_index": 0, "score", "mock": true,
"started_at", "finished_at", "duration_s"}`. `llm_calls` = number of assistant messages in
the worker's rollouts; `tool_calls` = number of tool messages. `workers` includes every node
of the graph; nodes that never ran because a sibling failed have status `skipped`.

To support this, `GraphResult` gains `timings: dict[int, tuple[ts, ts]]` and `GraphRunner`
records per-worker start/end; on failure the runner attaches `partial_outputs`,
`partial_timings` and `failed_node` to the raised `GraphFailure` (attributes, additive).

### 13.6 Public recording API

```python
class History:                      # additive methods
    def log_run(self, record: dict) -> None          # rewrites run.json
    def log_event(self, event: str, **fields) -> None
    def log_evaluation(self, record: dict) -> None
    def iter_events(self) -> Iterator[dict]
    def iter_evaluations(self) -> Iterator[dict]
    def load_run(self) -> dict | None
    def evaluations_for(self, genome_id: str) -> list[dict]
    @staticmethod
    def now() -> str                                 # ISO-8601 UTC

class RunRecorder:                  # neat/recording.py — the only writer of run.json/events
    def __init__(self, history: History, *, run_id: str, config: NeatConfig,
                 model: ModelSettings | None, evaluator_info: dict, seed: int,
                 resumed_from: str | None = None) -> None
    def start(self) -> None
    def generation_started(self, generation: int) -> None
    def generation_finished(self, generation: int, stats: dict) -> None
    def finish(self, status: str = "finished", error: str | None = None) -> None
```

`Population` accepts an optional `recorder: RunRecorder | None`; the CLI always creates
one. `TaskEvaluator` and `MockEvaluator` accept an optional `history: History | None` and
write `evaluations.jsonl` and the evaluation events through it; `Population.evaluate`
passes its history into the evaluator if the evaluator has a `history` attribute that is
None.

---

## 14. Analysis WebUI (`webui/`)

A read-only FastAPI application plus static vanilla JavaScript (no build step, no CDN),
mirroring `agent2/demo`. It never writes to a run directory.

```
src/ace/webui/
    __init__.py
    app.py        create_app(runs_root: Path) -> FastAPI; mounts /api and /static, / serves index.html
    store.py      RunStore: discovers runs, loads/caches/tails JSONL, builds derived tables
    analysis.py   pure functions over records: curves, species timeline, operator stats,
                  lineage tree, genome comparison, graph layout (layers by topological level)
    static/       index.html, style.css, app.js (router + views), api.js, charts.js (SVG
                  line/area/bar charts), graph.js (SVG DAG from the layout JSON), diff.js,
                  chat.js (rollout viewer)
scripts/run_webui.py   --runs-root DIR (default .ace/neat) --host --port --no-browser
```

A **run** is any directory under `runs_root` containing `run.json` or `generations.jsonl`;
`run_id` is the directory name. The store re-reads a JSONL file only when its size changed
(tail from the last offset) so live runs are cheap to poll.

### 14.1 API (all JSON; 404 for unknown ids)

| Endpoint | Returns |
|----------|---------|
| `GET /api/runs` | `[{run_id, status, started_at, finished_at, generations_completed, best_fitness, population_size, evaluator_type}]` |
| `GET /api/runs/{run}` | `run.json` plus `{"counts": {genomes, evaluations, generations, species}}` |
| `GET /api/runs/{run}/progress` | `{"generations": [generations.jsonl records without `population`], "events_tail": last 50 events, "status"}` |
| `GET /api/runs/{run}/generations/{g}` | the full generation record including `population` |
| `GET /api/runs/{run}/species` | `{"timeline": [{generation, threshold, species: [{id, size, best_fitness, representative_id}]}], "species": {id: {created, last_seen, best_fitness, max_size}}}` |
| `GET /api/runs/{run}/genomes?generation=&species=&min_fitness=&sort=&limit=&offset=` | `{"total": n, "items": [genome records without the `genome` body]}` |
| `GET /api/runs/{run}/genomes/{id}` | genome record + `graph` (WorkerGraph JSON) + `layout` (`{nodes: [{id, type, level, x, y, properties_resolved}], edges: [{src, dst, kind}]}`) + `lineage` (list of ancestor records, oldest last) + `children` (ids) + `evaluations` (records without `workers[*].view`) |
| `GET /api/runs/{run}/genomes/{id}/dot` | Graphviz DOT text of the graph (typed edge colours, END node) |
| `GET /api/runs/{run}/evaluations?genome=&task=&failed=&limit=&offset=` | `{"total", "items": [records without workers]}` |
| `GET /api/runs/{run}/evaluations/{genome}/{task_index}` | full evaluation record including workers (views as JSON) |
| `GET /api/runs/{run}/evaluations/{genome}/{task_index}/workers/{node}/view` | `{"view": View JSON, "rendered": str | null}` — rendered through `render_view` when the task's project_dir still exists, else null |
| `GET /api/runs/{run}/rollouts/{genome}/{task_index}/{rollout_id}` | the rollout JSON (`raw_chat` etc.) |
| `GET /api/runs/{run}/operators` | `[{operator, count, mean_child_fitness, mean_parent_fitness, mean_delta, success_rate}]` where delta = child adjusted − best parent adjusted |
| `GET /api/runs/{run}/compare?a=&b=` | both genomes' graphs/layouts, property diffs per matched node, distance (via `speciation.distance` with the run's config) |
| `GET /api/runs/{run}/innovations` | `{"nodes": [{id, key, type}], "connections": [{id, src, dst, kind}]}` |
| `GET /api/runs/{run}/failures` | `[{failure_type, node_type, count, examples: [{genome_id, task_index, message}]}]` |
| `GET /api/health` | `{"ok": true, "runs_root": str}` |

### 14.2 Views (frontend)

Hash router: `#/runs`, `#/run/{id}` (overview: metadata, config, curves of best/mean/median
raw and adjusted fitness, species count + threshold, mean workers, evaluation count and
duration per generation; live polling every 3 s while status is running),
`#/run/{id}/generations/{g}` (population table with species colours, worker-type chips,
operator chips; click → genome), `#/run/{id}/species` (stacked species-size area chart +
per-species best fitness lines), `#/run/{id}/genomes` (filterable table),
`#/run/{id}/genome/{gid}` (SVG DAG with typed edge colours, resolved properties per node,
lineage chain with operators, children, evaluation list), `#/run/{id}/eval/{gid}/{task}`
(score/failure, git diff with syntax colouring, per-worker panels: timing, report, rendered
view or JSON, rollout chat viewer with tool calls collapsed), `#/run/{id}/operators`,
`#/run/{id}/failures`, `#/run/{id}/compare?a=&b=`, `#/run/{id}/innovations`.
Charts are inline SVG drawn by `charts.js`; graphs by `graph.js`; no external assets.
