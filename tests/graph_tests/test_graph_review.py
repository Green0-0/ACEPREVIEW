"""Regression tests for defects found in the adversarial review of ``ace.graph``.

Each test pins one confirmed defect (it failed before the matching fix).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agent2.core.agent import Agent
from agent2.core.objects import AgentConfig, WorkspacePolicy
from agent2.core.session import Session
from agent2.core.workspace import Workspace
from ace.graph.ace import AceError, AceFormat, ace_instructions, apply_ace, parse_ace
from ace.graph.codegraph import CodebaseGraph
from ace.graph.llm import ModelSettings, make_client_factory
from ace.graph.recording import ViewRecorder, recording_tools
from ace.graph.rendering import render_file_view, render_view
from ace.graph.views import FileView, Level, View

ALL = AceFormat.ELEMENT_FILE_LINES


def make_session(tmp_path: Path, files: dict[str, str | bytes]) -> Session:
    root = tmp_path / "project"
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content.encode() if isinstance(content, str) else content)
    return Session(str(root))


def ace(session: Session, text: str, fmt: AceFormat = ALL) -> dict[str, bytes | None]:
    return apply_ace(parse_ace(text, fmt), session, fmt).files


def block(header: str, body: str, fence: str = "```") -> str:
    return f"{fence}python\n@@ {header}\n{body}{fence}\n"


PROPERTY_SRC = '''class C:
    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value

    def other(self):
        return 1
'''


# --------------------------------------------------------------------------- ACE

def test_ace_ambiguous_element_path_raises_instead_of_editing_last(tmp_path):
    # Getter and setter share the path ``C.p``; code_nodes keeps only the setter, so a
    # bare ``C.p`` header used to silently replace the setter with the new getter.
    session = make_session(tmp_path, {"a.py": PROPERTY_SRC})
    getter = "@property\ndef p(self):\n    return self._p * 2\n"
    for header in ("a.py::C.p", "a.py::C.p::BEFORE", "a.py::C.p::AFTER"):
        with pytest.raises(AceError) as info:
            ace(session, block(header, getter))
        message = str(info.value)
        assert "ambiguous" in message.lower()
        assert "C.p.2" in message and "C.p.6" in message


def test_ace_line_suffix_disambiguates_duplicate_element(tmp_path):
    session = make_session(tmp_path, {"a.py": PROPERTY_SRC})
    getter = "@property\ndef p(self):\n    return self._p * 2\n"
    out = ace(session, block("a.py::C.p.2", getter))["a.py"].decode()
    assert "return self._p * 2" in out
    assert "@p.setter" in out and "self._p = value" in out  # setter untouched
    ast.parse(out)


CRLF_SRC = b"class C:\r\n    def m(self):\r\n        return 1\r\n\r\n\r\ndef f():\r\n    return 2\r\n"


@pytest.mark.parametrize(
    "header, body",
    [
        ("a.py::C.m", "def m(self):\n    x = 1\n    return x\n"),
        ("a.py::C.m::AFTER", "def n(self):\n    return 3\n"),
        ("a.py::C.m::BEFORE", "def n(self):\n    return 3\n"),
        ("a.py::f::AFTER", "def g():\n    return 3\n"),
        ("a.py::3", "        return 42\n"),
    ],
)
def test_ace_keeps_crlf_line_endings(tmp_path, header, body):
    session = make_session(tmp_path, {"a.py": CRLF_SRC})
    out = ace(session, block(header, body))["a.py"]
    assert b"\n" not in out.replace(b"\r\n", b""), out  # no bare LF introduced
    assert b"\r\r" not in out
    ast.parse(out.decode())


def test_ace_windows_style_path_in_header(tmp_path):
    session = make_session(tmp_path, {"pkg/a.py": "def f():\n    return 1\n"})
    out = ace(session, block(".\\pkg\\a.py::f", "def f():\n    return 2\n"))
    assert out == {"pkg/a.py": b"def f():\n    return 2\n"}


def test_ace_preindented_body_with_column0_string_stays_valid(tmp_path):
    # A method copied at its original indentation that contains a triple-quoted
    # string whose continuation lines sit at column 0: textwrap.dedent finds no common
    # prefix, so the body used to be indented twice (SyntaxError) and the literal changed.
    src = 'class C:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n'
    session = make_session(tmp_path, {"a.py": src})
    body = '    def m(self):\n        x = """\nfoo\n"""\n        return x\n'
    out = ace(session, block("a.py::C.m", body))["a.py"].decode()
    tree = ast.parse(out)
    assert '    def m(self):\n        x = """\nfoo\n"""\n        return x\n\n    def n' in out
    assert tree is not None


def test_ace_instructions_mention_longer_fences_and_line_suffix():
    text = ace_instructions(AceFormat.ELEMENT)
    assert "````" in text  # how to fence a body that itself contains ```
    assert "line number" in text.lower() and "Class.method.12" in text
    # and a four-backtick fence around a body containing ``` really parses whole
    body = 'def f():\n    """Example:\n\n    ```\n    f()\n    ```\n    """\n'
    (parsed,) = parse_ace(block("a.py::f", body, fence="````"), AceFormat.ELEMENT)
    assert parsed.body == body


# --------------------------------------------------------------------------- rendering

DUP_SRC = '''import os

if os.name == "nt":
    def helper():
        """Windows."""
        windows_only = 1
        return windows_only
else:
    def helper():
        """Posix."""
        posix_only = 2
        posix_only += 1
        posix_only += 1
        posix_only += 1
        posix_only += 1
        return posix_only
'''


def test_skeleton_hides_bodies_of_duplicate_top_level_elements(tmp_path):
    session = make_session(tmp_path, {"a.py": DUP_SRC})
    text = render_file_view(session.files["a.py"], FileView("a.py", {"": Level.COLLAPSED}))
    assert "windows_only" not in text
    assert "posix_only" not in text  # the second definition used to be shown in full
    assert '"""Posix."""' in text  # its head is still shown


def test_codegraph_skeleton_size_matches_rendering_with_duplicates(tmp_path):
    session = make_session(tmp_path, {"a.py": DUP_SRC})
    graph = CodebaseGraph(session)
    view = View()
    view.add_file("a.py", Level.COLLAPSED)
    rendered = len(render_view(session, view))
    assert abs(graph.size_of(("a.py", ""), Level.COLLAPSED) - rendered) <= 40


def test_full_element_with_shared_path_renders_and_costs_every_definition(tmp_path):
    session = make_session(tmp_path, {"a.py": PROPERTY_SRC})
    view = View()
    view.add_element("a.py", "C.p", Level.FULL)
    text = render_view(session, view)
    assert "return self._p" in text and "self._p = value" in text
    graph = CodebaseGraph(session)
    # size_of must describe what rendering shows: both definitions (lines 2-4 and 6-8)
    shown = [line for line in text.splitlines() if line and not line.startswith("--- ")]
    assert len(shown) == 6
    assert graph.size_of(("a.py", "C.p"), Level.FULL) == sum(len(line) + 1 for line in shown)


def test_collapsed_multiline_signature_with_inline_body_shows_signature(tmp_path):
    src = "class C:\n    def multi(self,\n              a): return a\n\n    def x(self):\n        return 1\n"
    session = make_session(tmp_path, {"a.py": src})
    text = render_file_view(session.files["a.py"], FileView("a.py", {"C": Level.COLLAPSED}))
    assert "a): return a" in text, text


def test_signature_line_skips_multiline_decorator(tmp_path):
    session = make_session(tmp_path, {"a.py": "@dec(\n    a=1,\n)\ndef f():\n    return 1\n"})
    graph = CodebaseGraph(session)
    assert graph.signature_line(("a.py", "f")) == "def f():"


# --------------------------------------------------------------------------- llm

def test_client_factory_disables_sdk_internal_retries(monkeypatch):
    # The retry policy lives in RetryingClient; the inner AsyncOpenAI client must not
    # retry on its own (it would also retry 408/409, which the contract forbids, and
    # multiply the attempt count).
    monkeypatch.setenv("REVIEW_TEST_KEY", "sk-test")
    client = make_client_factory(ModelSettings(model="m", api_key_env_var="REVIEW_TEST_KEY"))()
    assert client.inner.max_retries == 0


# --------------------------------------------------------------------------- recording

def _tool_agent(session: Session, tools: list[str]) -> tuple[Agent, ViewRecorder]:
    config = AgentConfig(
        path="graph/test/0", subagent_name="t", subagent_description="t", model="m",
        system_prompt="s", prompt_wrap_bottom="{{PROMPT}}", followup_prompt_wrapper="{{PROMPT}}",
        max_steps=3, constructors=["default"], hooks=["tool_call"], primary_tools=tools,
        secondary_tools=[], workspace_policy=WorkspacePolicy.ISOLATED, subagent_names=[],
    )
    agent = Agent(session, Workspace(session), config, "p", client=object())
    recorder = ViewRecorder(session)
    agent.tools = recording_tools(agent.tools, recorder)
    return agent, recorder


def test_reversed_line_range_is_not_recorded_as_seen(tmp_path):
    session = make_session(tmp_path, {"n.txt": "".join(f"line {i}\n" for i in range(1, 11))})
    agent, recorder = _tool_agent(session, ["view_lines", "view_multi_lines"])
    shown = agent.tools["view_lines"](session=session, agent=agent, file_path="n.txt", line_start=8, line_end=3)
    assert "line 3" not in shown  # the tool showed nothing...
    agent.tools["view_multi_lines"](session=session, agent=agent, file_line_paths="n.txt::9::4")
    assert recorder.finalize(agent.workspace).is_empty()  # ...so nothing was seen


# --------------------------------------------------------------------------- runner

def test_missing_payload_failure_carries_producer_node_id(session):
    import asyncio

    from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode
    from ace.graph.properties import default_properties
    from ace.graph.runner import GraphRunner, RunContext, WorkerOutputs
    from ace.graph.types import END, GraphFailure, Kind, WorkerType
    from ace.graph.workers.base import Worker

    class NoEdit(Worker):
        type = WorkerType.EDITOR

        async def run(self, ctx, node, inputs, props):  # noqa: ANN001
            return WorkerOutputs(view=View(), edit=None, report="r")

    graph = WorkerGraph(
        {7: WorkerNode(7, WorkerType.EDITOR, default_properties(WorkerType.EDITOR))},
        [WorkerEdge(7, END, Kind.EDIT)],
    )
    ctx = RunContext(session, "task", ModelSettings(model="m"), lambda: None)
    runner = GraphRunner(graph, workers={WorkerType.EDITOR: NoEdit})
    with pytest.raises(GraphFailure) as info:
        asyncio.run(runner.run(ctx))
    assert getattr(info.value, "node_id", None) == 7  # evaluators log the failing node


def test_ace_fences_indented_inside_markdown_list(tmp_path):
    # CommonMark strips the fence's indentation from its content lines; the parser kept
    # it, so headers inside list items were not recognised (message -> "No ACE blocks").
    session = make_session(tmp_path, {"a.py": "def f():\n    return 1\n", "n.txt": "a\nb\n"})
    message = (
        "1. Fix f:\n   ```python\n   @@ a.py::f\n   def f():\n       return 2\n   ```\n"
        "2. Fix notes:\n   ```\n   @@ n.txt::2\n   c\n   ```\n"
    )
    assert ace(session, message) == {"a.py": b"def f():\n    return 2\n", "n.txt": b"a\nc\n"}


def test_ancestor_headers_resolved_by_containment_for_shared_paths(tmp_path):
    src = (
        "import sys\n\n"
        "if sys.platform == 'win32':\n"
        "    class Outer:\n"
        "        x = 1\n\n"
        "        class Inner:  # first\n"
        "            y = 1\n\n"
        "            def m(self):\n"
        "                return 'first'\n"
        "else:\n"
        "    class Outer:\n"
        "        x = 2\n\n"
        "        class Inner:  # second\n"
        "            y = 2\n\n"
        "            def m(self):\n"
        "                return 'second'\n"
    )
    session = make_session(tmp_path, {"a.py": src})
    view = FileView("a.py", {"": Level.COLLAPSED, "Outer.Inner.m": Level.FULL})
    text = render_file_view(session.files["a.py"], view)
    assert "return 'first'" in text and "return 'second'" in text
    assert "class Inner:  # first" in text  # header of the right enclosing class
    assert "class Inner:  # second" in text


def test_ace_quoted_diff_hunk_is_not_an_ace_header(tmp_path):
    # CRITIC/PICKMERGER are shown unified diffs; a reply that quotes a hunk in a code
    # block ("@@ -1,2 +1,2 @@") used to abort the whole message as a malformed header.
    session = make_session(tmp_path, {"a.py": "def f():\n    return 1\n"})
    message = (
        "The candidate changed this hunk:\n```diff\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n"
        "+    return 3\n```\nIt should return 2 instead:\n"
        + block("a.py::f", "def f():\n    return 2\n")
    )
    for fmt in AceFormat:
        if fmt is AceFormat.FILE_LINES:
            continue
        assert ace(session, message, fmt) == {"a.py": b"def f():\n    return 2\n"}
