"""Categorical property genes per worker type (GRAPH_NEAT_DESIGN.md section 8).

A genome stores, per worker, a mapping ``property name -> option index``. This module
holds the fixed option tables (:data:`PROPERTIES`) and the helpers to create, check and
resolve those mappings, plus :func:`build_system_prompt`.

Choices made where the contract leaves room:

* Index 0 of every option tuple is the default (:func:`default_properties`), so the
  tuples list a sensible default first rather than in ascending order.
* ``ace_format`` options are the plain ``AceFormat`` string values; ``graph/ace.py`` is
  not imported here. Workers convert with ``AceFormat(value)``.
* Prompt-slot texts live in :mod:`ace.graph.prompts`; this module only indexes them.
* :func:`build_system_prompt` leaves ``{ace_instructions}`` in place (fill it with
  ``str.replace``) and appends ``{{FILES*}}`` for tool-using worker types.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ace.graph import prompts
from ace.graph.types import WORKER_SPECS, WorkerType

PROMPT_SLOTS: tuple[str, ...] = ("role", "strategy", "report_focus", "instruction")

ACE_FORMAT_VALUES: tuple[str, ...] = ("element", "element_file", "element_file_lines", "file_lines")


@dataclass(frozen=True)
class PropertySpec:
    """One categorical property: its name, its concrete JSON-serialisable options and a description."""

    name: str
    options: tuple[Any, ...]
    description: str


_ALL = frozenset(WorkerType)
_W = WorkerType

# name -> (applicable types, description). Order here fixes the per-type property order.
_APPLICABILITY: dict[str, tuple[frozenset[WorkerType], str]] = {
    "role": (_ALL, "System prompt slot: who the agent is."),
    "strategy": (_ALL, "System prompt slot: how to approach the task."),
    "report_focus": (
        frozenset({_W.VIEWER, _W.LISTVIEWER, _W.EDITOR}),
        "System prompt slot: what the final report must contain.",
    ),
    "instruction": (_ALL, "Initial user-message framing placed before the task prompt."),
    "context_budget": (
        frozenset({_W.LISTVIEWER, _W.EDITOR, _W.VIEWEDITOR, _W.CRITIC, _W.JUDGEMERGER, _W.PICKMERGER}),
        "Characters of rendered views offered to the agent.",
    ),
    "max_steps": (frozenset({_W.VIEWER, _W.EDITOR}), "Agent loop steps."),
    "tool_set": (frozenset({_W.VIEWER, _W.EDITOR}), "Name of a tool list in prompts.TOOL_SETS."),
    "ace_format": (
        frozenset({_W.VIEWEDITOR, _W.CRITIC, _W.PICKMERGER}),
        "AceFormat value: which ACE target headers are permitted.",
    ),
    "list_budget": (
        frozenset({_W.LISTVIEWER}),
        "Characters of the candidate list offered before the agent picks.",
    ),
    "view_full_weight": (
        frozenset({_W.LISTVIEWER, _W.VIEWEDITOR, _W.CRITIC, _W.JUDGEMERGER, _W.PICKMERGER}),
        "Relevance weight of a FULL sighting relative to a COLLAPSED one.",
    ),
}

_NUMERIC_OPTIONS: dict[str, tuple[int, ...]] = {
    "context_budget": (12000, 6000, 24000, 48000),
    "list_budget": (8000, 4000, 16000),
    "view_full_weight": (2, 1, 3),
}

_MAX_STEPS: dict[WorkerType, tuple[int, ...]] = {
    _W.VIEWER: (20, 10, 40),
    _W.EDITOR: (30, 15, 60),
}

_TOOL_SET_OPTIONS: dict[WorkerType, tuple[str, ...]] = {
    _W.VIEWER: prompts.VIEWER_TOOL_SETS,
    _W.EDITOR: prompts.EDITOR_TOOL_SETS,
}

_SLOT_TABLES: dict[str, dict[WorkerType, tuple[str, ...]]] = {
    "role": prompts.ROLES,
    "strategy": prompts.STRATEGIES,
    "report_focus": prompts.REPORT_FOCI,
    "instruction": prompts.INSTRUCTIONS,
}


def _options_for(t: WorkerType, name: str) -> tuple[Any, ...]:
    if name in _SLOT_TABLES:
        return tuple(_SLOT_TABLES[name][t])
    if name == "max_steps":
        return _MAX_STEPS[t]
    if name == "tool_set":
        return _TOOL_SET_OPTIONS[t]
    if name == "ace_format":
        return ACE_FORMAT_VALUES
    return _NUMERIC_OPTIONS[name]


def _build_properties() -> dict[WorkerType, dict[str, PropertySpec]]:
    table: dict[WorkerType, dict[str, PropertySpec]] = {}
    for t in WorkerType:
        specs: dict[str, PropertySpec] = {}
        for name, (types, description) in _APPLICABILITY.items():
            if t in types:
                specs[name] = PropertySpec(name, _options_for(t, name), description)
        table[t] = specs
    return table


PROPERTIES: dict[WorkerType, dict[str, PropertySpec]] = _build_properties()


def default_properties(t: WorkerType) -> dict[str, int]:
    """Every applicable property of ``t`` at option index 0."""
    return {name: 0 for name in PROPERTIES[t]}


def random_properties(t: WorkerType, rng: random.Random) -> dict[str, int]:
    """A uniformly random option index for every applicable property of ``t``."""
    return {name: rng.randrange(len(spec.options)) for name, spec in PROPERTIES[t].items()}


def validate_properties(t: WorkerType, indices: dict[str, int]) -> None:
    """Raise ``ValueError`` naming every unknown, missing or out-of-range property."""
    specs = PROPERTIES[t]
    problems: list[str] = []
    unknown = sorted(set(indices) - set(specs))
    missing = [name for name in specs if name not in indices]
    if unknown:
        problems.append(f"unknown properties {unknown}")
    if missing:
        problems.append(f"missing properties {missing}")
    for name, index in indices.items():
        if name not in specs:
            continue
        count = len(specs[name].options)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            problems.append(f"property {name!r} index {index!r} out of range 0..{count - 1}")
    if problems:
        raise ValueError(f"invalid properties for {t.value}: " + "; ".join(problems))


def resolve(t: WorkerType, indices: dict[str, int]) -> dict[str, Any]:
    """Map option indices to concrete values (validates first)."""
    validate_properties(t, indices)
    return {name: spec.options[indices[name]] for name, spec in PROPERTIES[t].items()}


def build_system_prompt(t: WorkerType, resolved: dict[str, Any]) -> str:
    """Concatenate role, strategy, report_focus (if present) and the type's output block.

    ``{ace_instructions}`` is left intact for the worker to fill with ``str.replace``.
    Tool-using types get a trailing ``{{FILES*}}`` line for the ``directory`` constructor.
    """
    parts = [resolved["role"], resolved["strategy"]]
    if resolved.get("report_focus"):
        parts.append(resolved["report_focus"])
    parts.append(prompts.OUTPUT_FORMAT_BLOCKS[t])
    if WORKER_SPECS[t].uses_tools:
        parts.append(prompts.FILES_LINE)
    return "\n\n".join(part.strip() for part in parts)
