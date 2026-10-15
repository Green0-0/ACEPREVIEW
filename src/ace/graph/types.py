"""Core types of the worker graph: payload kinds, worker types, slot tables, errors.

See GRAPH_NEAT_DESIGN.md sections 1 and 7.1. This module has no dependencies on the rest
of the graph package so every other module can import it freely.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

END = "END"
"""Reserved id of the end node. Exactly one EDIT edge may enter it."""


class Kind(str, Enum):
    """What an edge carries."""

    VIEW = "view"
    EDIT = "edit"
    REPORT = "report"


class WorkerType(str, Enum):
    VIEWER = "viewer"
    LISTVIEWER = "listviewer"
    EDITOR = "editor"
    VIEWEDITOR = "vieweditor"
    CRITIC = "critic"
    JUDGEMERGER = "judgemerger"
    PICKMERGER = "pickmerger"


@dataclass(frozen=True)
class SlotSpec:
    """Arity of one input kind on a worker type. ``max=None`` means unbounded."""

    kind: Kind
    min: int
    max: int | None

    def accepts(self, count: int) -> bool:
        return count >= self.min and (self.max is None or count <= self.max)


@dataclass(frozen=True)
class WorkerSpec:
    type: WorkerType
    inputs: dict[Kind, SlotSpec]
    outputs: frozenset[Kind]
    uses_tools: bool

    def slot(self, kind: Kind) -> SlotSpec | None:
        return self.inputs.get(kind)

    def required_kinds(self) -> frozenset[Kind]:
        return frozenset(kind for kind, slot in self.inputs.items() if slot.min > 0)


def _slots(*specs: tuple[Kind, int, int | None]) -> dict[Kind, SlotSpec]:
    return {kind: SlotSpec(kind, lo, hi) for kind, lo, hi in specs}


WORKER_SPECS: dict[WorkerType, WorkerSpec] = {
    WorkerType.VIEWER: WorkerSpec(
        WorkerType.VIEWER,
        inputs=_slots(),
        outputs=frozenset({Kind.VIEW, Kind.REPORT}),
        uses_tools=True,
    ),
    WorkerType.LISTVIEWER: WorkerSpec(
        WorkerType.LISTVIEWER,
        inputs=_slots((Kind.VIEW, 1, None)),
        outputs=frozenset({Kind.VIEW, Kind.REPORT}),
        uses_tools=False,
    ),
    WorkerType.EDITOR: WorkerSpec(
        WorkerType.EDITOR,
        inputs=_slots((Kind.VIEW, 0, None), (Kind.REPORT, 0, None)),
        outputs=frozenset({Kind.VIEW, Kind.EDIT, Kind.REPORT}),
        uses_tools=True,
    ),
    WorkerType.VIEWEDITOR: WorkerSpec(
        WorkerType.VIEWEDITOR,
        inputs=_slots((Kind.VIEW, 1, None), (Kind.REPORT, 0, None)),
        outputs=frozenset({Kind.VIEW, Kind.EDIT}),
        uses_tools=False,
    ),
    WorkerType.CRITIC: WorkerSpec(
        WorkerType.CRITIC,
        inputs=_slots((Kind.VIEW, 1, None), (Kind.EDIT, 1, 1), (Kind.REPORT, 0, None)),
        outputs=frozenset({Kind.VIEW, Kind.EDIT}),
        uses_tools=False,
    ),
    WorkerType.JUDGEMERGER: WorkerSpec(
        WorkerType.JUDGEMERGER,
        inputs=_slots((Kind.VIEW, 1, None), (Kind.EDIT, 2, None), (Kind.REPORT, 0, None)),
        outputs=frozenset({Kind.VIEW, Kind.EDIT}),
        uses_tools=False,
    ),
    WorkerType.PICKMERGER: WorkerSpec(
        WorkerType.PICKMERGER,
        inputs=_slots((Kind.VIEW, 1, None), (Kind.EDIT, 2, None), (Kind.REPORT, 0, None)),
        outputs=frozenset({Kind.VIEW, Kind.EDIT}),
        uses_tools=False,
    ),
}

EDIT_PRODUCERS: frozenset[WorkerType] = frozenset(
    t for t, spec in WORKER_SPECS.items() if Kind.EDIT in spec.outputs
)
VIEW_PRODUCERS: frozenset[WorkerType] = frozenset(
    t for t, spec in WORKER_SPECS.items() if Kind.VIEW in spec.outputs
)
REPORT_PRODUCERS: frozenset[WorkerType] = frozenset(
    t for t, spec in WORKER_SPECS.items() if Kind.REPORT in spec.outputs
)
MERGER_TYPES: frozenset[WorkerType] = frozenset({WorkerType.JUDGEMERGER, WorkerType.PICKMERGER})


class GraphFailure(Exception):
    """A worker-graph run failed for one task. The task scores zero."""


class InvalidGraph(ValueError):
    """A graph or genome violates one of the invariants in GRAPH_NEAT_DESIGN.md 1.3."""
