"""Tests for ace.graph.codegraph (GRAPH_NEAT_DESIGN.md section 5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ace.graph.codegraph import (
    CONTAINMENT_BACK_WEIGHT,
    CONTAINMENT_WEIGHT,
    IMPORT_WEIGHT,
    SIBLING_WEIGHT,
    CodebaseGraph,
)
from ace.graph.rendering import render_view
from ace.graph.views import Level, View

MODELS = "pkg/models.py"
UTILS = "pkg/utils.py"
SERVICE = "pkg/service.py"


@pytest.fixture
def graph(session) -> CodebaseGraph:
    return CodebaseGraph(session)


# --------------------------------------------------------------------------- nodes/edges


def test_nodes_include_files_and_elements(graph: CodebaseGraph) -> None:
    for key in [
        (MODELS, ""),
        (UTILS, ""),
        ("README.md", ""),
        ("notes.txt", ""),
        ("pkg/__init__.py", ""),
        (MODELS, "Item"),
        (MODELS, "Item.slug"),
        (MODELS, "Cart.total"),
        (UTILS, "slugify"),
        (SERVICE, "checkout"),
    ]:
        assert key in graph.nodes
    # Deduped by CodeNode.path: no llm_path (line-suffixed) keys.
    assert not any(element.split(".")[-1].isdigit() for _, element in graph.nodes if element)
    # Plain files contribute only their file node.
    assert [key for key in graph.nodes if key[0] == "README.md"] == [("README.md", "")]


def test_containment_edges_both_directions(graph: CodebaseGraph) -> None:
    assert graph.neighbors((MODELS, ""))[(MODELS, "Item")] == pytest.approx(CONTAINMENT_WEIGHT)
    assert graph.neighbors((MODELS, "Item"))[(MODELS, "Item.slug")] == pytest.approx(
        CONTAINMENT_WEIGHT
    )
    assert graph.neighbors((MODELS, "Item.slug"))[(MODELS, "Item")] == pytest.approx(
        CONTAINMENT_BACK_WEIGHT
    )
    assert graph.neighbors((MODELS, "Item"))[(MODELS, "")] == pytest.approx(
        CONTAINMENT_BACK_WEIGHT
    )
    # File does not link directly to nested elements.
    assert (MODELS, "Item.slug") not in graph.neighbors((MODELS, ""))


def test_sibling_edges(graph: CodebaseGraph) -> None:
    assert graph.neighbors((MODELS, "Item.__init__"))[(MODELS, "Item.slug")] == pytest.approx(
        SIBLING_WEIGHT
    )
    assert graph.neighbors((MODELS, "Item.slug"))[(MODELS, "Item.__init__")] == pytest.approx(
        SIBLING_WEIGHT
    )
    assert (MODELS, "Cart") in graph.neighbors((MODELS, "Item"))
    assert (MODELS, "Item") in graph.neighbors((MODELS, "Cart"))
    # Not consecutive: __init__ and to_json.
    assert (MODELS, "Item.to_json") not in graph.neighbors((MODELS, "Item.__init__"))


def test_import_edges(graph: CodebaseGraph) -> None:
    assert graph.neighbors((MODELS, ""))[(UTILS, "")] == pytest.approx(IMPORT_WEIGHT)
    service = graph.neighbors((SERVICE, ""))
    assert (MODELS, "") in service and (UTILS, "") in service
    # Stdlib imports (json, re) do not resolve to project files.
    assert all(dst[0] in {MODELS, UTILS, SERVICE} for dst in service)
    assert (MODELS, "") not in graph.neighbors((UTILS, ""))


def test_name_reference_edges(graph: CodebaseGraph) -> None:
    assert (UTILS, "slugify") in graph.neighbors((MODELS, "Item.slug"))
    checkout = graph.neighbors((SERVICE, "checkout"))
    assert (UTILS, "clamp") in checkout
    assert (MODELS, "Cart.total") in checkout
    build = graph.neighbors((SERVICE, "build_cart"))
    assert (MODELS, "Cart") in build and (MODELS, "Item") in build
    # Builtins/keywords and short names are skipped; definition names are not references.
    assert all(dst[1] not in {"sum", "max", "min"} for dst in graph.neighbors((UTILS, "clamp")))
    # Module-level code is not an element body: the file links to slugify by containment only.
    assert graph.neighbors((UTILS, ""))[(UTILS, "slugify")] == pytest.approx(CONTAINMENT_WEIGHT)
    # Element names inside a class body (def names) are not references from the class.
    assert graph.neighbors((MODELS, "Item"))[(MODELS, "Item.slug")] == pytest.approx(
        CONTAINMENT_WEIGHT
    )


def test_relative_and_package_imports(tmp_path: Path) -> None:
    from agent2.core.session import Session

    root = tmp_path / "proj"
    files = {
        "app/__init__.py": "from .core import run\n",
        "app/core.py": "from . import helpers\nfrom .sub import thing\n\ndef run():\n    return 1\n",
        "app/helpers.py": "def helper_fn():\n    return 2\n",
        "app/sub/__init__.py": "thing = 3\n",
        "app/sub/deep.py": "from ..helpers import helper_fn\nimport app.core\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    graph = CodebaseGraph(Session(str(root)))
    assert ("app/core.py", "") in graph.neighbors(("app/__init__.py", ""))
    core = graph.neighbors(("app/core.py", ""))
    assert ("app/helpers.py", "") in core
    assert ("app/sub/__init__.py", "") in core
    deep = graph.neighbors(("app/sub/deep.py", ""))
    assert ("app/helpers.py", "") in deep
    assert ("app/core.py", "") in deep


# --------------------------------------------------------------------------- rank


def test_rank_seed_on_slug_prefers_slugify(graph: CodebaseGraph) -> None:
    ranked = graph.rank({(MODELS, "Item.slug"): 1.0})
    scores = dict(ranked)
    assert scores[(UTILS, "slugify")] > scores[(UTILS, "clamp")]
    assert ranked[0][0] == (MODELS, "Item.slug")
    assert len(ranked) == len(graph.nodes)
    assert sum(scores.values()) == pytest.approx(1.0)
    # Sorted descending, ties broken by key.
    assert ranked == sorted(ranked, key=lambda item: (-item[1], item[0]))


def test_rank_empty_seeds_uses_file_nodes(graph: CodebaseGraph) -> None:
    ranked = graph.rank({})
    assert sum(score for _, score in ranked) == pytest.approx(1.0)
    assert dict(ranked)[("README.md", "")] > 0
    assert graph.rank({("missing.py", ""): 5.0, (MODELS, "Item"): 0.0}) == ranked


# --------------------------------------------------------------------------- select


def test_select_respects_budget_and_includes_seed(graph: CodebaseGraph, session) -> None:
    seeds = {(MODELS, "Item.slug"): 3.0, (SERVICE, "checkout"): 1.0}
    view = graph.select(seeds, 800, session)
    assert view.files[MODELS].levels["Item.slug"] == Level.FULL
    assert view.files[MODELS].levels[""] == Level.COLLAPSED
    assert view.files[SERVICE].levels["checkout"] == Level.FULL
    for path, file_view in view.files.items():
        for element in file_view.levels:
            if element:
                assert file_view.levels[""] == Level.COLLAPSED
                assert file_view.levels[element] == Level.FULL
    assert len(render_view(session, view)) <= 800 * 1.5


def test_select_small_budget_skips_and_continues(graph: CodebaseGraph, session) -> None:
    view = graph.select({(MODELS, "Item.slug"): 1.0}, 300, session)
    # The seed does not fit with its file skeleton, but smaller nodes still do.
    assert MODELS not in view.files
    assert not view.is_empty()
    assert graph.select({(MODELS, "Item.slug"): 1.0}, 0, session).is_empty()


def test_select_rendered_length_close_to_budget(graph: CodebaseGraph, session) -> None:
    budget = 1500
    for seeds in ({(MODELS, "Item.slug"): 1.0}, {}):
        view = graph.select(seeds, budget, session)
        rendered = render_view(session, view)
        assert 0 < len(rendered) <= budget * 1.5
        assert len(rendered) >= budget / 2


def test_select_empty_seeds_collapses_files(graph: CodebaseGraph, session) -> None:
    view = graph.select({}, 100_000, session)
    assert set(view.files) == {key[0] for key in graph.nodes}
    assert all(fv.levels[""] == Level.COLLAPSED for fv in view.files.values())


# --------------------------------------------------------------------------- sizes


def test_size_of(graph: CodebaseGraph, session) -> None:
    full_item = graph.size_of((MODELS, "Item"), Level.FULL)
    slug = graph.size_of((MODELS, "Item.slug"), Level.FULL)
    assert full_item > slug > graph.size_of((MODELS, "Item.slug"), Level.COLLAPSED) > 0
    skeleton = graph.size_of((MODELS, ""), Level.COLLAPSED)
    whole = graph.size_of((MODELS, ""), Level.FULL)
    assert skeleton < whole
    view = View()
    view.add_file(MODELS, Level.FULL)
    assert abs(len(render_view(session, view)) - whole) <= 10
    view = View()
    view.add_file(MODELS, Level.COLLAPSED)
    assert abs(len(render_view(session, view)) - skeleton) <= 0.25 * skeleton
    assert graph.size_of(("notes.txt", ""), Level.COLLAPSED) == graph.size_of(
        ("notes.txt", ""), Level.FULL
    )
    with pytest.raises(KeyError):
        graph.size_of(("nope.py", ""), Level.FULL)


# --------------------------------------------------------------------------- seeds


def test_resolve_lines(graph: CodebaseGraph) -> None:
    # Lines 13-15 are Item.slug inside class Item (lines 6-18).
    assert graph.resolve_lines(MODELS, 14, 14) == [(MODELS, "Item"), (MODELS, "Item.slug")]
    span = graph.resolve_lines(MODELS, 17, 22)
    assert {(MODELS, "Item"), (MODELS, "Item.to_json"), (MODELS, "Cart")} <= set(span)
    assert (MODELS, "") not in span
    assert graph.resolve_lines(MODELS, 1, 3) == [(MODELS, "")]
    assert graph.resolve_lines("notes.txt", 1, 2) == [("notes.txt", "")]
    assert graph.resolve_lines("missing.py", 1, 2) == []
    assert graph.resolve_lines(MODELS, 15, 13) == graph.resolve_lines(MODELS, 13, 15)


def test_seeds_from_views(graph: CodebaseGraph) -> None:
    a = View()
    a.add_file(MODELS, Level.COLLAPSED)
    a.add_element(MODELS, "Item.slug", Level.FULL)
    b = View()
    b.add_element(MODELS, "Item.slug", Level.FULL)
    b.add_file(UTILS, Level.FULL)
    b.add_lines(MODELS, 14, 14)
    b.add_element("nope.py", "x", Level.FULL)

    seeds = graph.seeds_from_views([a, b], full_weight=3)
    assert seeds[(MODELS, "Item.slug")] == 3 + 3 + 1  # two FULL sightings + line range
    assert seeds[(MODELS, "Item")] == 1  # resolved from the line range
    assert seeds[(MODELS, "")] == 1 + 1  # COLLAPSED in a, line range in b
    assert seeds[(UTILS, "")] == 3
    assert ("nope.py", "x") not in seeds

    default = graph.seeds_from_views([a, b])
    assert default[(MODELS, "Item.slug")] == 2 + 2 + 1
    assert graph.seeds_from_views([]) == {}


# --------------------------------------------------------------------------- listing


def test_candidate_list_format_and_truncation(graph: CodebaseGraph, session) -> None:
    keys = [(UTILS, "slugify"), (MODELS, "Item.slug"), (UTILS, ""), ("missing.py", "")]
    text = graph.candidate_list(keys, session, 10_000)
    assert text.splitlines() == [
        "pkg/utils.py::slugify  def slugify(text):",
        "pkg/models.py::Item.slug  def slug(self):",
        "pkg/utils.py",
    ]
    truncated = graph.candidate_list(keys, session, 60)
    assert len(truncated) <= 60
    assert truncated.splitlines() == ["pkg/utils.py::slugify  def slugify(text):"]
    assert graph.candidate_list(keys, session, 5) == ""

    view = graph.select({(MODELS, "Item.slug"): 1.0}, 1500, session)
    listing = graph.candidate_list(view, session, 10_000, seeds={(MODELS, "Item.slug"): 1.0})
    lines = listing.splitlines()
    assert lines[0] == "pkg/models.py::Item.slug  def slug(self):"
    assert len(lines) == sum(len(fv.levels) for fv in view.files.values())


# --------------------------------------------------------------------------- determinism


def test_deterministic_across_constructions(session) -> None:
    first, second = CodebaseGraph(session), CodebaseGraph(session)
    assert first.nodes == second.nodes
    for key in sorted(first.nodes):
        assert list(first.neighbors(key).items()) == list(second.neighbors(key).items())
    seeds = {(MODELS, "Item.slug"): 2.0, (SERVICE, ""): 1.0}
    assert first.rank(seeds) == second.rank(seeds)
    assert first.select(seeds, 1500, session).to_json() == second.select(
        seeds, 1500, session
    ).to_json()
    assert first.candidate_list(
        first.select(seeds, 1500, session), session, 500
    ) == second.candidate_list(second.select(seeds, 1500, session), session, 500)
