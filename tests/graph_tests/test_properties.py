"""Tests for ace.graph.properties and the prompt tables in ace.graph.prompts."""
from __future__ import annotations

import json
import random

import pytest

from agent2.core.globals import TOOLS
from ace.graph import prompts
from ace.graph.properties import (
    PROMPT_SLOTS,
    PROPERTIES,
    PropertySpec,
    build_system_prompt,
    default_properties,
    random_properties,
    resolve,
    validate_properties,
)
from ace.graph.types import WORKER_SPECS, WorkerType

W = WorkerType
ALL = set(W)
APPLICABILITY: dict[str, set[WorkerType]] = {
    "role": ALL,
    "strategy": ALL,
    "report_focus": {W.VIEWER, W.LISTVIEWER, W.EDITOR},
    "instruction": ALL,
    "context_budget": {W.LISTVIEWER, W.EDITOR, W.VIEWEDITOR, W.CRITIC, W.JUDGEMERGER, W.PICKMERGER},
    "max_steps": {W.VIEWER, W.EDITOR},
    "tool_set": {W.VIEWER, W.EDITOR},
    "ace_format": {W.VIEWEDITOR, W.CRITIC, W.PICKMERGER},
    "list_budget": {W.LISTVIEWER},
    "view_full_weight": {W.LISTVIEWER, W.VIEWEDITOR, W.CRITIC, W.JUDGEMERGER, W.PICKMERGER},
}
VALUE_TYPES = {
    "role": str, "strategy": str, "report_focus": str, "instruction": str,
    "context_budget": int, "max_steps": int, "tool_set": str, "ace_format": str,
    "list_budget": int, "view_full_weight": int,
}
FORBIDDEN = {"delegate", "message_subagent", "cancel_subagent", "bash", "tools"}
VIEW_AND_SEARCH = {
    "search", "list_folder", "search_folders", "view", "view_path", "view_files", "view_file",
    "views_at", "view_at", "view_element", "view_multi_lines", "view_lines",
}
EDITING = {
    "replace_file", "replace_lines", "replace_element", "replace_element_at", "open_file",
    "open_lines", "open_element", "open_element_at", "make_empty_files", "move_files",
}


@pytest.mark.parametrize("t", list(W))
def test_every_type_has_exactly_its_applicable_properties(t):
    expected = {name for name, types in APPLICABILITY.items() if t in types}
    assert set(PROPERTIES[t]) == expected
    for name, spec in PROPERTIES[t].items():
        assert isinstance(spec, PropertySpec)
        assert spec.name == name
        assert spec.description


@pytest.mark.parametrize("t", list(W))
def test_option_counts_types_and_json(t):
    for name, spec in PROPERTIES[t].items():
        minimum = 4 if name in PROMPT_SLOTS else 3
        assert len(spec.options) >= minimum, (t, name)
        assert len(set(spec.options)) == len(spec.options), (t, name)
        for option in spec.options:
            assert isinstance(option, VALUE_TYPES[name]) and not isinstance(option, bool)
        json.dumps(list(spec.options))
        if name in PROMPT_SLOTS:
            assert all(len(option) > 40 for option in spec.options)


def test_prompt_slots_constant():
    assert PROMPT_SLOTS == ("role", "strategy", "report_focus", "instruction")


def test_prompt_texts_are_written_per_type():
    roles = [PROPERTIES[t]["role"].options for t in W]
    assert len({tuple(r) for r in roles}) == len(roles)
    strategies = [PROPERTIES[t]["strategy"].options for t in W]
    assert len({tuple(s) for s in strategies}) == len(strategies)


@pytest.mark.parametrize("t", list(W))
def test_defaults_and_random_validate_and_resolve(t):
    defaults = default_properties(t)
    assert set(defaults) == set(PROPERTIES[t]) and set(defaults.values()) == {0}
    validate_properties(t, defaults)
    rng = random.Random(1234)
    for _ in range(25):
        indices = random_properties(t, rng)
        validate_properties(t, indices)
        resolved = resolve(t, indices)
        assert set(resolved) == set(PROPERTIES[t])
        for name, value in resolved.items():
            assert value == PROPERTIES[t][name].options[indices[name]]
            assert isinstance(value, VALUE_TYPES[name])


def test_random_properties_is_deterministic_per_seed():
    a = random_properties(W.EDITOR, random.Random(7))
    b = random_properties(W.EDITOR, random.Random(7))
    assert a == b


def test_validate_properties_errors_name_the_problem():
    good = default_properties(W.EDITOR)
    with pytest.raises(ValueError, match="bogus"):
        validate_properties(W.EDITOR, {**good, "bogus": 0})
    missing = dict(good)
    del missing["tool_set"]
    with pytest.raises(ValueError, match="tool_set"):
        validate_properties(W.EDITOR, missing)
    with pytest.raises(ValueError, match="max_steps"):
        validate_properties(W.EDITOR, {**good, "max_steps": 99})
    with pytest.raises(ValueError, match="role"):
        validate_properties(W.EDITOR, {**good, "role": -1})
    # A property that exists on another type is unknown here.
    with pytest.raises(ValueError, match="ace_format"):
        validate_properties(W.EDITOR, {**good, "ace_format": 0})
    with pytest.raises(ValueError):
        resolve(W.EDITOR, {})


@pytest.mark.parametrize("t", list(W))
def test_build_system_prompt(t):
    resolved = resolve(t, random_properties(t, random.Random(3)))
    text = build_system_prompt(t, resolved)
    assert resolved["role"].strip() in text
    assert resolved["strategy"].strip() in text
    assert prompts.OUTPUT_FORMAT_BLOCKS[t].strip() in text
    if "report_focus" in resolved:
        assert resolved["report_focus"].strip() in text
    assert text.index(resolved["role"].strip()) < text.index(resolved["strategy"].strip())
    if WORKER_SPECS[t].uses_tools:
        assert "{{FILES*}}" in text
    else:
        assert "{{FILES" not in text
    if t in {W.VIEWEDITOR, W.CRITIC, W.PICKMERGER}:
        assert "{ace_instructions}" in text
        filled = text.replace("{ace_instructions}", "ACE RULES")
        assert "ACE RULES" in filled and "{ace_instructions}" not in filled
    else:
        assert "{ace_instructions}" not in text


def test_output_blocks_cover_every_type():
    assert set(prompts.OUTPUT_FORMAT_BLOCKS) == set(W)
    assert "candidate number" in prompts.OUTPUT_FORMAT_BLOCKS[W.JUDGEMERGER]
    assert "path::" in prompts.OUTPUT_FORMAT_BLOCKS[W.LISTVIEWER]


def test_tool_sets_are_valid_and_families_distinct():
    viewer_sets = [n for n in prompts.TOOL_SETS if n.startswith("viewer_")]
    editor_sets = [n for n in prompts.TOOL_SETS if n.startswith("editor_")]
    assert len(viewer_sets) >= 3 and len(editor_sets) >= 3
    assert set(viewer_sets) | set(editor_sets) == set(prompts.TOOL_SETS)
    for name, tools in prompts.TOOL_SETS.items():
        assert tools and len(tools) == len(set(tools)), name
        assert set(tools) <= set(TOOLS), name
        assert not set(tools) & FORBIDDEN, name
        if name.startswith("viewer_"):
            assert set(tools) <= VIEW_AND_SEARCH, name
        else:
            assert set(tools) <= VIEW_AND_SEARCH | EDITING, name
            assert set(tools) & EDITING, name


def test_tool_set_options_reference_right_family():
    viewer = PROPERTIES[W.VIEWER]["tool_set"].options
    editor = PROPERTIES[W.EDITOR]["tool_set"].options
    assert all(n in prompts.TOOL_SETS and n.startswith("viewer_") for n in viewer)
    assert all(n in prompts.TOOL_SETS and n.startswith("editor_") for n in editor)
    assert len(viewer) >= 3 and len(editor) >= 3


def test_ace_format_options():
    expected = {"element", "element_file", "element_file_lines", "file_lines"}
    for t in (W.VIEWEDITOR, W.CRITIC, W.PICKMERGER):
        options = PROPERTIES[t]["ace_format"].options
        assert set(options) == expected and len(options) == 4


def test_assemble_user_prompt_skips_empty_sections():
    text = prompts.assemble_user_prompt(
        "Do the thing.",
        "Fix the bug.",
        [(prompts.VIEWS_HEADER, "code here"), (prompts.REPORTS_HEADER, "  "), ("# Extra", "x")],
    )
    assert text == (
        f"Do the thing.\n\n{prompts.TASK_HEADER}\nFix the bug.\n\n"
        f"{prompts.VIEWS_HEADER}\ncode here\n\n# Extra\nx"
    )
    assert prompts.REPORTS_HEADER not in text
    assert prompts.assemble_user_prompt("", "T", []) == f"{prompts.TASK_HEADER}\nT"


def test_labels_format():
    assert prompts.CANDIDATE_LABEL.format(n=2, id=17) == "CANDIDATE 2 (worker 17)"
    assert prompts.REPORT_LABEL.format(id=3, type="viewer") == "REPORT from worker 3 (viewer)"
    for header in (
        prompts.TASK_HEADER, prompts.VIEWS_HEADER, prompts.REPORTS_HEADER,
        prompts.CANDIDATES_HEADER, prompts.LISTVIEWER_CANDIDATES_HEADER,
        prompts.LISTVIEWER_FOLLOWUP,
    ):
        assert isinstance(header, str) and header and "{" not in header
