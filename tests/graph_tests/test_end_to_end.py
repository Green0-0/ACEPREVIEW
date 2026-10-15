"""End-to-end tests: real worker registry driven through ``GraphRunner.run``.

Every graph below runs the actual worker implementations (``ace.graph.workers``) on
the sample project. A single :class:`~fakes.RoutingClient` serves every agent of a run;
its routes dispatch on the worker type's output-format block in the system prompt (and,
where two workers share a type, on their ``instruction`` text in the user message).
"""
from __future__ import annotations

import asyncio
import json
from typing import Callable, Iterable

import pytest

from ace.graph import prompts
from ace.graph.ace import AceError, AceFormat, ace_instructions
from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode, minimal_graph
from ace.graph.properties import PROPERTIES, default_properties
from ace.graph.runner import GraphResult, GraphRunner, ModelSettings, RunContext
from ace.graph.types import END, GraphFailure, Kind, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers.critic import AUTHOR_REPORT_SUFFIX
from fakes import FakeMessage, RoutingClient, system_prompt_of, text_message, tool_call_message
from sample_project import SAMPLE_FILES

T = WorkerType
V, E, R = Kind.VIEW, Kind.EDIT, Kind.REPORT
PROMPT = "Make clamp handle reversed bounds and make slugs use underscores."
UTILS = "pkg/utils.py"
MODELS = "pkg/models.py"

# ---------------------------------------------------------------------------- code

ORIGINAL_CLAMP = "def clamp(value, low, high):\n    return max(low, min(high, value))"
EDITOR_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        low, high = high, low\n"
    "    return max(low, min(high, value))  # EDITOR"
)
SECOND_EDITOR_CLAMP = (
    "def clamp(value, low, high):\n"
    "    low, high = min(low, high), max(low, high)\n"
    "    return max(low, min(high, value))  # SECOND"
)
CRITIC_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        raise ValueError('CRITIC bounds')\n"
    "    return max(low, min(high, value))"
)
ORIGINAL_SLUGIFY = (
    'def slugify(text):\n'
    '    """Lowercase and dash-join the text."""\n'
    '    return PATTERN.sub("-", text.lower()).strip("-")'
)
UNDERSCORE_SLUGIFY = (
    'def slugify(text):\n'
    '    """Lowercase and underscore-join the text."""\n'
    '    return PATTERN.sub("_", text.lower()).strip("_")'
)


def utils_with(*replacements: tuple[str, str]) -> str:
    """The pristine ``pkg/utils.py`` with each ``(old, new)`` replacement applied."""
    text = SAMPLE_FILES[UTILS]
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    return text


def ace(*blocks: tuple[str, str], prose: str = "Here is the edit.") -> str:
    """An ACE reply: prose followed by one fenced block per ``(header, body)``."""
    parts = [prose]
    for header, body in blocks:
        parts.append(f"```python\n@@ {header}\n{body}\n```")
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------- routing


def marker(t: WorkerType) -> str:
    """Text of ``t``'s output-format block that survives ACE placeholder filling."""
    return prompts.OUTPUT_FORMAT_BLOCKS[t].split(prompts.ACE_PLACEHOLDER)[0].strip()


def test_markers_are_distinct():
    for a in WorkerType:
        for b in WorkerType:
            if a is not b:
                assert marker(a) not in prompts.OUTPUT_FORMAT_BLOCKS[b]


def user_text(kwargs: dict) -> str:
    return "\n\n".join(
        m.get("content") or "" for m in kwargs.get("messages", []) if m.get("role") == "user"
    )


def first_user(kwargs: dict) -> str:
    return next(m["content"] for m in kwargs["messages"] if m.get("role") == "user")


def is_type(t: WorkerType, *, user_contains: str | None = None) -> Callable[[dict], bool]:
    def predicate(kwargs: dict) -> bool:
        if marker(t) not in system_prompt_of(kwargs):
            return False
        return user_contains is None or user_contains in first_user(kwargs)

    return predicate


def requests_of(client: RoutingClient, t: WorkerType, user_contains: str | None = None) -> list[dict]:
    predicate = is_type(t, user_contains=user_contains)
    return [kwargs for kwargs in client.requests if predicate(kwargs)]


def make_ctx(session, routes: list[tuple[Callable[[dict], bool], Iterable[FakeMessage]]]):
    client = RoutingClient(routes)
    ctx = RunContext(
        session=session,
        prompt=PROMPT,
        settings=ModelSettings(model="fake-model"),
        client_factory=lambda: client,
        run_id="e2e",
    )
    return ctx, client


def run_graph(graph: WorkerGraph, ctx: RunContext) -> GraphResult:
    return asyncio.run(GraphRunner(graph).run(ctx))


# --------------------------------------------------------------------------- scripts


def viewer_script(report: str = "Item.slug in pkg/models.py delegates to pkg/utils.py::slugify."):
    return [
        tool_call_message(("view", json.dumps({"paths": f"{MODELS}::Item.slug"})), prefix="v"),
        text_message(report),
    ]


def editor_script(new_clamp: str, report: str, prefix: str = "e"):
    return [
        tool_call_message(("view", json.dumps({"paths": f"{UTILS}::clamp"})), prefix=f"{prefix}a"),
        tool_call_message(
            (
                "replace_element",
                json.dumps(
                    {"file_path": UTILS, "element_path": "clamp", "new_content": new_clamp}
                ),
            ),
            prefix=f"{prefix}b",
        ),
        text_message(report),
    ]


EDITOR_REPORT = "Swapped reversed bounds in pkg/utils.py::clamp (EDITOR)."
LISTVIEWER_REPORT = "pkg/utils.py::slugify joins with dashes; switch it to underscores (LV)."


# --------------------------------------------------------------------------- checks


def node(nid: int, t: WorkerType, **overrides: int) -> WorkerNode:
    indices = default_properties(t)
    indices.update(overrides)
    return WorkerNode(nid, t, indices)


def option(t: WorkerType, name: str, value) -> int:
    return PROPERTIES[t][name].options.index(value)


def assert_topological(graph: WorkerGraph, order: list[int]) -> None:
    assert sorted(order) == sorted(graph.nodes)
    assert len(order) == len(set(order))
    position = {nid: i for i, nid in enumerate(order)}
    for edge in graph.edges:
        if edge.dst != END:
            assert position[edge.src] < position[edge.dst], edge.label()


def assert_outputs_complete(graph: WorkerGraph, result: GraphResult, session=None) -> None:
    """One output per worker, each with exactly one distinct, persisted rollout id."""
    assert set(result.outputs) == set(graph.nodes)
    all_ids: list[str] = []
    for nid, outputs in result.outputs.items():
        assert len(outputs.rollout_ids) == 1 and outputs.rollout_ids[0], nid
        all_ids.extend(outputs.rollout_ids)
        if session is not None:
            (rollout_id,) = outputs.rollout_ids
            assert (session.rollout_path / f"{rollout_id}.json").is_file(), nid
            assert rollout_id in session.datastore
    assert len(all_ids) == len(set(all_ids))  # one distinct rollout per worker


def assert_covers(big: View, small: View) -> None:
    """Every element/file level in ``small`` is present in ``big`` at the same or higher level."""
    assert not small.is_empty()
    for path, file_view in small.files.items():
        assert path in big.files, path
        for element, level in file_view.levels.items():
            assert big.files[path].levels.get(element, 0) >= level, (path, element)


def assert_no_leftover(client: RoutingClient) -> None:
    for _, messages in client.routes:
        assert next(messages, None) is None, "a scripted reply was never requested"


# ---------------------------------------------------------------------------- (a)


def test_minimal_editor_graph(session):
    graph = minimal_graph()
    ctx, client = make_ctx(session, [(is_type(T.EDITOR), editor_script(EDITOR_CLAMP, EDITOR_REPORT))])
    result = run_graph(graph, ctx)

    assert result.order == [0]
    assert_outputs_complete(graph, result, session)
    assert result.edit is result.outputs[0].edit
    assert result.edit.files == {UTILS: utils_with((ORIGINAL_CLAMP, EDITOR_CLAMP)).encode()}
    assert result.diff == result.edit.unified_diff(session)
    assert "+        low, high = high, low" in result.diff
    assert "-    return max(low, min(high, value))" in result.diff
    out = result.outputs[0]
    assert out.report == EDITOR_REPORT
    assert out.view.files[UTILS].levels.get("clamp") == Level.FULL

    (first, *_rest) = client.requests
    assert len(client.requests) == 3
    assert first["model"] == "fake-model"
    assert sorted(t["function"]["name"] for t in first["tools"]) == sorted(
        prompts.TOOL_SETS["editor_standard"]
    )
    assert PROMPT in first_user(first)
    assert_no_leftover(client)


# ---------------------------------------------------------------------------- (b)

SLUG_ACE = ace((f"{UTILS}::slugify", UNDERSCORE_SLUGIFY))
LV_SELECTION = f"I want to read these:\n```\n{UTILS}::slugify\n{MODELS}::Item.slug\n```\n"


def chain_graph() -> WorkerGraph:
    """VIEWER 1 -> LISTVIEWER 2 -> VIEWEDITOR 3 -> END (VIEW chain, plus 2->3 REPORT)."""
    graph = WorkerGraph(
        {1: node(1, T.VIEWER), 2: node(2, T.LISTVIEWER), 3: node(3, T.VIEWEDITOR)},
        [WorkerEdge(1, 2, V), WorkerEdge(2, 3, V), WorkerEdge(2, 3, R), WorkerEdge(3, END, E)],
    )
    graph.validate()
    return graph


def chain_routes(*, vieweditor_reply: str = SLUG_ACE, viewer_report: str | None = None):
    viewer = viewer_script() if viewer_report is None else viewer_script(viewer_report)
    return [
        (is_type(T.VIEWER), viewer),
        (is_type(T.LISTVIEWER), [text_message(LV_SELECTION), text_message(LISTVIEWER_REPORT)]),
        (is_type(T.VIEWEDITOR), [text_message(vieweditor_reply)]),
    ]


def test_view_chain_with_listviewer(session):
    graph = chain_graph()
    ctx, client = make_ctx(session, chain_routes())
    result = run_graph(graph, ctx)

    assert result.order == [1, 2, 3]
    assert_topological(graph, result.order)
    assert_outputs_complete(graph, result, session)

    # Final edit: exactly the vieweditor's ACE edit.
    assert result.edit is result.outputs[3].edit
    assert result.edit.files == {UTILS: utils_with((ORIGINAL_SLUGIFY, UNDERSCORE_SLUGIFY)).encode()}
    assert '+    return PATTERN.sub("_", text.lower()).strip("_")' in result.diff
    assert result.outputs[3].report is None

    # Viewer: recorded what it viewed with tools.
    viewer_view = result.outputs[1].view
    assert viewer_view.files[MODELS].levels.get("Item.slug") == Level.FULL

    # ListViewer: two-step protocol on one agent; candidates seeded by the viewer's view.
    lv = requests_of(client, T.LISTVIEWER)
    assert len(lv) == 2
    assert not lv[0].get("tools")
    step1 = user_text(lv[0])
    assert prompts.LISTVIEWER_CANDIDATES_HEADER in step1
    assert f"{MODELS}::Item.slug" in step1
    step2_users = [m["content"] for m in lv[1]["messages"] if m["role"] == "user"]
    assert len(step2_users) == 2
    assert step2_users[1].startswith(prompts.LISTVIEWER_FOLLOWUP)
    assert "def slugify" in step2_users[1]
    lv_out = result.outputs[2]
    assert lv_out.report == LISTVIEWER_REPORT
    assert lv_out.view.files[UTILS].levels.get("slugify") == Level.FULL
    assert lv_out.view.files[MODELS].levels.get("Item.slug") == Level.FULL

    # ViewEditor: sees the listviewer's view (and report, labelled by source).
    (ve,) = requests_of(client, T.VIEWEDITOR)
    ve_user = first_user(ve)
    assert prompts.VIEWS_HEADER in ve_user and "def slugify" in ve_user
    assert prompts.REPORT_LABEL.format(id=2, type="listviewer") in ve_user
    assert LISTVIEWER_REPORT in ve_user
    assert ace_instructions(AceFormat.ELEMENT) in system_prompt_of(ve)
    assert_covers(result.outputs[3].view, lv_out.view)
    assert_covers(result.outputs[3].view, viewer_view)
    assert_no_leftover(client)


def test_vieweditor_without_ace_block_fails_run(session):
    ctx, _ = make_ctx(session, chain_routes(vieweditor_reply="slugify looks fine to me."))
    with pytest.raises(GraphFailure) as info:
        run_graph(chain_graph(), ctx)
    assert isinstance(info.value, AceError)
    assert info.value.node_id == 3
    assert any("node 3 (vieweditor)" in note for note in info.value.__notes__)


@pytest.mark.parametrize("report", ["", "   \n"])
def test_empty_viewer_report_fails_run(session, report):
    ctx, client = make_ctx(session, chain_routes(viewer_report=report))
    with pytest.raises(GraphFailure, match=r"node 1 \(viewer\)") as info:
        run_graph(chain_graph(), ctx)
    assert info.value.node_id == 1
    assert not requests_of(client, T.LISTVIEWER)  # downstream workers never ran


def test_empty_listviewer_report_fails_run(session):
    routes = chain_routes()
    routes[1] = (is_type(T.LISTVIEWER), [text_message(LV_SELECTION), text_message("  ")])
    ctx, client = make_ctx(session, routes)
    with pytest.raises(GraphFailure, match="empty report") as info:
        run_graph(chain_graph(), ctx)
    assert info.value.node_id == 2
    assert not requests_of(client, T.VIEWEDITOR)


# ---------------------------------------------------------------------------- (c)

VE_CLAMP = (
    "def clamp(value, low, high):\n"
    "    return min(max(value, min(low, high)), max(low, high))  # VIEWEDITOR"
)


def diamond_graph(**merger_overrides: int) -> WorkerGraph:
    """VIEWER 1 -> {EDITOR 2, VIEWEDITOR 3} -> JUDGEMERGER 4 -> END."""
    graph = WorkerGraph(
        {
            1: node(1, T.VIEWER),
            2: node(2, T.EDITOR),
            3: node(3, T.VIEWEDITOR),
            4: node(4, T.JUDGEMERGER, **merger_overrides),
        },
        [
            WorkerEdge(1, 2, V),
            WorkerEdge(1, 2, R),
            WorkerEdge(1, 3, V),
            WorkerEdge(2, 4, E),
            WorkerEdge(2, 4, V),
            WorkerEdge(2, 4, R),
            WorkerEdge(3, 4, E),
            WorkerEdge(3, 4, V),
            WorkerEdge(4, END, E),
        ],
    )
    graph.validate()
    return graph


VIEWER_REPORT = "Item.slug in pkg/models.py delegates to pkg/utils.py::slugify; clamp is in pkg/utils.py."


def diamond_routes(judge_reply: str = "2"):
    return [
        (is_type(T.VIEWER), viewer_script(VIEWER_REPORT)),
        (is_type(T.EDITOR), editor_script(EDITOR_CLAMP, EDITOR_REPORT)),
        (is_type(T.VIEWEDITOR), [text_message(ace((f"{UTILS}::clamp", VE_CLAMP)))]),
        (is_type(T.JUDGEMERGER), [text_message(judge_reply)]),
    ]


def test_diamond_with_judgemerger(session):
    graph = diamond_graph()
    ctx, client = make_ctx(session, diamond_routes("2"))
    result = run_graph(graph, ctx)

    assert_topological(graph, result.order)
    assert result.order[0] == 1 and result.order[-1] == 4
    assert_outputs_complete(graph, result, session)

    # The judge picked candidate 2 = worker 3 (vieweditor), forwarded unchanged.
    assert result.edit is result.outputs[3].edit
    assert result.edit.files == {UTILS: utils_with((ORIGINAL_CLAMP, VE_CLAMP)).encode()}
    assert "# VIEWEDITOR" in result.diff and "# EDITOR" not in result.diff
    assert result.outputs[4].report is None

    # Editor got the viewer's view and report, grouped and labelled by source.
    editor_user = first_user(requests_of(client, T.EDITOR)[0])
    assert prompts.VIEWS_HEADER in editor_user and "def slug" in editor_user
    assert prompts.REPORT_LABEL.format(id=1, type="viewer") + "\n" + VIEWER_REPORT in editor_user
    assert result.outputs[2].edit.files == {UTILS: utils_with((ORIGINAL_CLAMP, EDITOR_CLAMP)).encode()}
    assert result.outputs[2].view.files[MODELS].levels.get("Item.slug") == Level.FULL

    # VIEWEDITOR got only the view (no REPORT edge 1->3).
    ve_user = first_user(requests_of(client, T.VIEWEDITOR)[0])
    assert prompts.REPORTS_HEADER not in ve_user and "def slug" in ve_user

    # Judge: candidates in ascending source order, the editor's report labelled.
    (judge,) = requests_of(client, T.JUDGEMERGER)
    judge_user = first_user(judge)
    c1 = judge_user.index(prompts.CANDIDATE_LABEL.format(n=1, id=2))
    c2 = judge_user.index(prompts.CANDIDATE_LABEL.format(n=2, id=3))
    assert c1 < c2
    assert "# EDITOR" in judge_user[c1:c2] and "# VIEWEDITOR" in judge_user[c2:]
    assert prompts.REPORT_LABEL.format(id=2, type="editor") + "\n" + EDITOR_REPORT in judge_user
    assert "REPORT from worker 1" not in judge_user and "REPORT from worker 3" not in judge_user

    # Forwarded views: the merger's view contains what the viewer saw.
    viewer_view = result.outputs[1].view
    assert_covers(result.outputs[2].view, viewer_view)
    assert_covers(result.outputs[3].view, viewer_view)
    assert_covers(result.outputs[4].view, viewer_view)
    assert_covers(result.outputs[4].view, result.outputs[2].view)
    assert_no_leftover(client)


def test_diamond_judge_picks_first(session):
    ctx, _ = make_ctx(session, diamond_routes("Candidate 1 is correct."))
    result = run_graph(diamond_graph(), ctx)
    assert result.edit is result.outputs[2].edit
    assert "# EDITOR" in result.diff


@pytest.mark.parametrize("reply", ["3", "0", "no idea"])
def test_judgemerger_out_of_range_fails_run(session, reply):
    ctx, _ = make_ctx(session, diamond_routes(reply))
    with pytest.raises(GraphFailure, match=r"node 4 \(judgemerger\)") as info:
        run_graph(diamond_graph(), ctx)
    assert info.value.node_id == 4


class BlockingEditorClient(RoutingClient):
    """Editor requests block until cancelled; everything else is routed normally."""

    def __init__(self, routes) -> None:
        super().__init__(routes)
        self.editor_started = asyncio.Event()
        self.editor_cancelled = False

    async def create(self, **kwargs):
        if is_type(T.EDITOR)(kwargs):
            self.editor_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.editor_cancelled = True
                raise
        elif is_type(T.VIEWEDITOR)(kwargs):
            await self.editor_started.wait()  # fail only once the sibling is mid-run
        return await super().create(**kwargs)


def test_sibling_failure_cancels_running_worker(session):
    routes = diamond_routes()
    routes[2] = (is_type(T.VIEWEDITOR), [text_message("No code blocks here.")])
    client = BlockingEditorClient(routes)
    ctx = RunContext(session, PROMPT, ModelSettings(model="fake-model"), lambda: client, run_id="c")

    async def main() -> None:
        await asyncio.wait_for(GraphRunner(diamond_graph()).run(ctx), timeout=10)

    with pytest.raises(AceError) as info:
        asyncio.run(main())
    assert info.value.node_id == 3
    assert client.editor_cancelled
    assert not requests_of(client, T.JUDGEMERGER)


def test_graph_from_json_runs_identically(session):
    graph = diamond_graph()
    loaded = WorkerGraph.from_json(json.loads(json.dumps(graph.to_json())))
    assert loaded == graph

    ctx1, client1 = make_ctx(session, diamond_routes("2"))
    first = run_graph(graph, ctx1)
    ctx2, client2 = make_ctx(session, diamond_routes("2"))
    second = run_graph(loaded, ctx2)

    assert second.order == first.order
    assert second.edit == first.edit
    assert second.diff == first.diff
    for nid in graph.nodes:
        a, b = first.outputs[nid], second.outputs[nid]
        assert (a.view, a.edit, a.report) == (b.view, b.edit, b.report)

    def transcript(client: RoutingClient) -> list[str]:
        return sorted(json.dumps(r["messages"], sort_keys=True) for r in client.requests)

    assert transcript(client2) == transcript(client1)


# ---------------------------------------------------------------------------- (d)


def critic_graph() -> WorkerGraph:
    """EDITOR 1 -> CRITIC 2 (EDIT, VIEW, REPORT) -> END."""
    graph = WorkerGraph(
        {1: node(1, T.EDITOR), 2: node(2, T.CRITIC)},
        [WorkerEdge(1, 2, E), WorkerEdge(1, 2, V), WorkerEdge(1, 2, R), WorkerEdge(2, END, E)],
    )
    graph.validate()
    return graph


def test_editor_then_critic(session):
    graph = critic_graph()
    routes = [
        (is_type(T.EDITOR), editor_script(EDITOR_CLAMP, EDITOR_REPORT)),
        (is_type(T.CRITIC), [text_message(ace((f"{UTILS}::clamp", CRITIC_CLAMP)))]),
    ]
    ctx, client = make_ctx(session, routes)
    result = run_graph(graph, ctx)

    assert result.order == [1, 2]
    assert_outputs_complete(graph, result, session)
    assert result.edit is result.outputs[2].edit
    assert result.edit.files == {UTILS: utils_with((ORIGINAL_CLAMP, CRITIC_CLAMP)).encode()}
    assert "CRITIC bounds" in result.diff and "# EDITOR" not in result.diff
    assert result.outputs[2].report is None

    (critic,) = requests_of(client, T.CRITIC)
    user = first_user(critic)
    section = user.split(prompts.EDIT_UNDER_REVIEW_HEADER, 1)[1]
    assert section.lstrip("\n").startswith("from worker 1 (editor)\n```diff\n")
    assert "# EDITOR" in section
    author = prompts.REPORT_LABEL.format(id=1, type="editor") + AUTHOR_REPORT_SUFFIX
    assert author + "\n" + EDITOR_REPORT in section
    assert prompts.REPORTS_HEADER not in user  # the only report is the author's
    assert "def clamp" in user.split(prompts.VIEWS_HEADER, 1)[1]
    assert_covers(result.outputs[2].view, result.outputs[1].view)
    assert_no_leftover(client)


# ---------------------------------------------------------------------------- (e)

SECOND_INSTRUCTION = 2  # EDITOR instruction option used to tell the two editors apart


def pick_graph() -> WorkerGraph:
    """EDITOR 1 + EDITOR 2 -> PICKMERGER 3 -> END, each editor also sends its VIEW."""
    graph = WorkerGraph(
        {
            1: node(1, T.EDITOR),
            2: node(2, T.EDITOR, instruction=SECOND_INSTRUCTION),
            3: node(3, T.PICKMERGER),
        },
        [
            WorkerEdge(1, 3, E),
            WorkerEdge(1, 3, V),
            WorkerEdge(2, 3, E),
            WorkerEdge(2, 3, V),
            WorkerEdge(3, END, E),
        ],
    )
    graph.validate()
    return graph


def test_two_editors_then_pickmerger(session):
    graph = pick_graph()
    first_instruction = prompts.INSTRUCTIONS[T.EDITOR][0]
    second_instruction = prompts.INSTRUCTIONS[T.EDITOR][SECOND_INSTRUCTION]
    merged = ace(
        (f"{UTILS}::clamp", EDITOR_CLAMP),
        (f"{UTILS}::slugify", UNDERSCORE_SLUGIFY),
    )
    routes = [
        (
            is_type(T.EDITOR, user_contains=first_instruction),
            editor_script(EDITOR_CLAMP, "First editor: swapped bounds.", prefix="x"),
        ),
        (
            is_type(T.EDITOR, user_contains=second_instruction),
            editor_script(SECOND_EDITOR_CLAMP, "Second editor: normalised bounds.", prefix="y"),
        ),
        (is_type(T.PICKMERGER), [text_message(merged)]),
    ]
    ctx, client = make_ctx(session, routes)
    result = run_graph(graph, ctx)

    assert result.order == [1, 2, 3]
    assert_topological(graph, result.order)
    assert_outputs_complete(graph, result, session)
    assert "# EDITOR" in result.outputs[1].edit.files[UTILS].decode()
    assert "# SECOND" in result.outputs[2].edit.files[UTILS].decode()

    assert result.edit is result.outputs[3].edit
    assert result.edit.files == {
        UTILS: utils_with(
            (ORIGINAL_CLAMP, EDITOR_CLAMP), (ORIGINAL_SLUGIFY, UNDERSCORE_SLUGIFY)
        ).encode()
    }
    assert result.outputs[3].report is None

    (pick,) = requests_of(client, T.PICKMERGER)
    user = first_user(pick)
    c1 = user.index(prompts.CANDIDATE_LABEL.format(n=1, id=1))
    c2 = user.index(prompts.CANDIDATE_LABEL.format(n=2, id=2))
    assert c1 < c2
    assert "# EDITOR" in user[c1:c2] and "# SECOND" not in user[c1:c2]
    assert "# SECOND" in user[c2:]
    assert prompts.REPORTS_HEADER not in user  # no REPORT edges into the merger
    assert ace_instructions(AceFormat.ELEMENT) in system_prompt_of(pick)
    assert_covers(result.outputs[3].view, result.outputs[1].view)
    assert_covers(result.outputs[3].view, result.outputs[2].view)
    assert len(requests_of(client, T.EDITOR, first_instruction)) == 3
    assert len(requests_of(client, T.EDITOR, second_instruction)) == 3
    assert_no_leftover(client)


# ----------------------------------------------------------------------- properties


def test_non_default_properties_change_prompts(session):
    """Index 1 for ``role`` (and other slots) reaches the system prompt / user message."""
    graph = WorkerGraph(
        {
            1: node(1, T.VIEWER, role=1),
            2: node(2, T.LISTVIEWER, role=1, instruction=1),
            3: node(
                3,
                T.VIEWEDITOR,
                role=1,
                strategy=3,
                ace_format=option(T.VIEWEDITOR, "ace_format", "element_file"),
            ),
        },
        chain_graph().edges,
    )
    graph.validate()
    whole_file = ace((UTILS, "VALUE = 1"))
    ctx, client = make_ctx(session, chain_routes(vieweditor_reply=whole_file))
    result = run_graph(graph, ctx)

    viewer_system = system_prompt_of(requests_of(client, T.VIEWER)[0])
    assert prompts.ROLES[T.VIEWER][1] in viewer_system
    assert prompts.ROLES[T.VIEWER][0] not in viewer_system

    lv = requests_of(client, T.LISTVIEWER)[0]
    assert prompts.ROLES[T.LISTVIEWER][1] in system_prompt_of(lv)
    assert first_user(lv).startswith(prompts.INSTRUCTIONS[T.LISTVIEWER][1])

    ve_system = system_prompt_of(requests_of(client, T.VIEWEDITOR)[0])
    assert prompts.ROLES[T.VIEWEDITOR][1] in ve_system
    assert prompts.ROLES[T.VIEWEDITOR][0] not in ve_system
    assert prompts.STRATEGIES[T.VIEWEDITOR][3] in ve_system
    assert ace_instructions(AceFormat.ELEMENT_FILE) in ve_system

    # The element_file format permits the whole-file header the default format rejects.
    assert result.edit.files == {UTILS: b"VALUE = 1\n"}

    ctx, _ = make_ctx(session, chain_routes(vieweditor_reply=whole_file))
    with pytest.raises(AceError):
        run_graph(chain_graph(), ctx)


def test_editor_role_index_one(session):
    graph = minimal_graph(dict(default_properties(T.EDITOR), role=1))
    ctx, client = make_ctx(session, [(is_type(T.EDITOR), editor_script(EDITOR_CLAMP, EDITOR_REPORT))])
    run_graph(graph, ctx)
    system = system_prompt_of(client.requests[0])
    assert prompts.ROLES[T.EDITOR][1] in system
    assert prompts.ROLES[T.EDITOR][0] not in system


def test_editor_new_file_and_remapped_view_reach_critic(session):
    """A created file and a post-edit line view flow through the critic correctly."""
    short_slugify = "def slugify(text):\n    # one\n    # two\n    return text"
    editor = [
        tool_call_message(("make_empty_files", json.dumps({"file_paths": "pkg/extra.py"})), prefix="a"),
        tool_call_message(
            ("replace_file", json.dumps({"file_path": "pkg/extra.py", "new_content": "def extra():\n    return 1\n"})),
            prefix="b",
        ),
        tool_call_message(
            (
                "replace_element",
                json.dumps({"file_path": UTILS, "element_path": "slugify", "new_content": short_slugify}),
            ),
            prefix="c",
        ),
        # Lines 12-14 of the edited file are pristine lines 11-13 (slugify grew by one line).
        tool_call_message(
            ("view_lines", json.dumps({"file_path": UTILS, "line_start": 12, "line_end": 14})), prefix="d"
        ),
        tool_call_message(("view", json.dumps({"paths": "pkg/extra.py"})), prefix="e"),
        text_message("Added pkg/extra.py and reworked slugify."),
    ]
    critic = ace(("pkg/extra.py::NEW", "def extra():\n    return 2"), (f"{UTILS}::clamp", CRITIC_CLAMP))
    ctx, client = make_ctx(
        session, [(is_type(T.EDITOR), editor), (is_type(T.CRITIC), [text_message(critic)])]
    )
    graph = critic_graph()
    result = run_graph(graph, ctx)
    assert_outputs_complete(graph, result, session)

    editor_out = result.outputs[1]
    assert editor_out.edit.files["pkg/extra.py"] == b"def extra():\n    return 1\n"
    assert editor_out.edit.files[UTILS] == utils_with((ORIGINAL_SLUGIFY, short_slugify)).encode()
    # Created files are never part of a view; the line range is remapped onto pristine.
    assert set(editor_out.view.files) == {UTILS}
    assert editor_out.view.files[UTILS].line_ranges == [(11, 13)]

    section = first_user(requests_of(client, T.CRITIC)[0]).split(prompts.EDIT_UNDER_REVIEW_HEADER, 1)[1]
    assert "+++ b/pkg/extra.py" in section and "+    return 1" in section
    assert "+    # two" in section

    assert result.edit.files == {
        "pkg/extra.py": b"def extra():\n    return 2\n",
        UTILS: utils_with((ORIGINAL_CLAMP, CRITIC_CLAMP)).encode(),
    }
    assert "--- /dev/null\n+++ b/pkg/extra.py" in result.diff


def test_bad_property_index_from_json_fails_node(session):
    data = minimal_graph().to_json()
    data["nodes"][0]["properties"]["role"] = 99
    graph = WorkerGraph.from_json(data)
    ctx, client = make_ctx(session, [(is_type(T.EDITOR), [])])
    with pytest.raises(GraphFailure, match=r"node 0 \(editor\)") as info:
        run_graph(graph, ctx)
    assert info.value.node_id == 0
    assert not client.requests
