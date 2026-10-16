"""Worker base class and the helpers concrete workers build on (GRAPH_NEAT_DESIGN.md §9).

A :class:`Worker` turns its grouped :class:`~ace.graph.runner.WorkerInputs` into
:class:`~ace.graph.runner.WorkerOutputs`, usually by running one agent2 ``Agent``
built with :meth:`Worker.make_agent` on a fresh isolated ``Workspace``.

Choices made where the contract is open:

* ``make_agent``'s ``constructors`` argument is optional: ``None`` means ``["default"]``
  plus ``["directory"]`` when tools are present. The config is in memory only:
  ``path=f"graph/{type}/{id}"``, ``subagent_name=f"{type}_{id}"``, prompt wrappers
  ``"{{PROMPT}}"``, hooks ``["tool_call"]`` iff tools, ``workspace_policy=ISOLATED``.
  When tools are present the agent gets ``agent.view_recorder`` (a ``ViewRecorder``) and
  ``agent.tools = recording_tools(agent.tools, recorder)`` (wrapping the agent's
  own tool objects so per-agent schemas survive); otherwise ``agent.view_recorder`` is
  ``None``. ``agent.graph_node`` is the ``WorkerNode`` (used to name failures).
  Forbidden tools (``prompts.FORBIDDEN_TOOLS``) raise ``ValueError``.
* :func:`run_agent` re-raises a ``GraphFailure`` from the agent unchanged and wraps any
  other exception (or a ``None`` final message) in ``GraphFailure`` naming the node.
* :func:`render_inputs_views` takes the seeds from ``ctx.codegraph.seeds_from_views``
  (a ``CodebaseGraph`` method) over the input views in ascending source id order.
* :func:`render_reports` labels with ``prompts.REPORT_LABEL``; a source whose type is not
  in ``inputs.source_types`` is labelled ``unknown``. Reports are separated by blank lines.
* :func:`render_edits` returns ``(text, ordered_source_ids)`` (the contract's "mapping
  used": candidate ``n`` is ``ids[n - 1]``). Each ``prompts.CANDIDATE_LABEL`` line is
  followed by the diff in a ```` ```diff ```` fence (made longer than any backtick run in
  the diff), or by ``(no changes)`` for an empty edit.
* :func:`system_prompt_for` fills ``prompts.ACE_PLACEHOLDER`` from ``ace_fmt`` or, when
  that is ``None``, from ``props["ace_format"]`` if present.
* :func:`first_int` returns the first run of decimal digits (no sign) or ``None``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from agent2.core.agent import Agent
from agent2.core.objects import AgentConfig, WorkspacePolicy
from agent2.core.workspace import Workspace
from ace.graph import prompts
from ace.graph.ace import AceFormat, ace_instructions, apply_ace, parse_ace
from ace.graph.edits import Edit
from ace.graph.graph import WorkerNode
from ace.graph.properties import build_system_prompt
from ace.graph.rendering import render_view
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import View

if TYPE_CHECKING:
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = [
    "Worker",
    "build_user_prompt",
    "first_int",
    "followup_agent",
    "parse_ace_output",
    "render_edits",
    "render_inputs_views",
    "render_reports",
    "run_agent",
    "system_prompt_for",
]

PROMPT_WRAPPER = "{{PROMPT}}"
"""Wrapper used for both the initial prompt and followups: the text is passed verbatim."""


class Worker(ABC):
    """One worker type's behaviour. Subclasses set ``type`` and implement :meth:`run`."""

    type: ClassVar[WorkerType]

    @abstractmethod
    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Produce this worker's outputs; raise ``GraphFailure`` on any task-level failure."""

    def make_agent(
        self,
        ctx: RunContext,
        node: WorkerNode,
        props: dict[str, Any],
        *,
        tools: list[str],
        system_prompt: str,
        user_prompt: str,
        max_steps: int,
        constructors: list[str] | None = None,
    ) -> Agent:
        """Build an ``Agent`` for ``node`` on a fresh isolated workspace (see module doc).

        Args:
            ctx: The run context (session, model settings, client factory).
            node: The worker node; names the config path and subagent name.
            props: Resolved properties (unused here; kept for subclasses/overrides).
            tools: Tool names from ``agent2.core.globals.TOOLS``; empty for tool-less workers.
            system_prompt: Full system prompt (placeholders other than ``{{FILES*}}`` filled).
            user_prompt: First user message.
            max_steps: Agent loop step limit.
            constructors: Constructor names; default ``["default"]`` (+ ``"directory"`` with tools).
        """
        tools = list(tools)
        forbidden = sorted(set(tools) & prompts.FORBIDDEN_TOOLS)
        if forbidden:
            raise ValueError(f"{node.label()}: forbidden tools {forbidden}")
        if constructors is None:
            constructors = ["default", "directory"] if tools else ["default"]
        type_name = node.type.value
        config = AgentConfig(
            path=f"graph/{type_name}/{node.id}",
            subagent_name=f"{type_name}_{node.id}",
            subagent_description=f"Worker-graph {type_name} worker (node {node.id}).",
            model=ctx.settings.model,
            api_base_url=ctx.settings.api_base_url,
            api_key_env_var=ctx.settings.api_key_env_var,
            system_prompt=system_prompt,
            prompt_wrap_bottom=PROMPT_WRAPPER,
            followup_prompt_wrapper=PROMPT_WRAPPER,
            max_steps=max_steps,
            constructors=list(constructors),
            hooks=["tool_call"] if tools else [],
            primary_tools=tools,
            secondary_tools=[],
            workspace_policy=WorkspacePolicy.ISOLATED,
            subagent_names=[],
        )
        agent = Agent(
            ctx.session,
            Workspace(ctx.session),
            config,
            user_prompt,
            client=ctx.client_factory(),
        )
        agent.graph_node = node  # type: ignore[attr-defined]
        agent.view_recorder = None  # type: ignore[attr-defined]
        if tools:
            from ace.graph.recording import ViewRecorder, recording_tools

            recorder = ViewRecorder(ctx.session)
            agent.view_recorder = recorder  # type: ignore[attr-defined]
            agent.tools = recording_tools(agent.tools, recorder)
        return agent


def _agent_label(agent: Agent) -> str:
    node = getattr(agent, "graph_node", None)
    if isinstance(node, WorkerNode):
        return node.label()
    return f"agent {agent.config.subagent_name}"


async def run_agent(agent: Agent) -> str:
    """Run ``agent`` to its final message; ``GraphFailure`` (naming the node) on error or ``None``."""
    label = _agent_label(agent)
    try:
        result = await agent.run()
    except GraphFailure:
        raise
    except Exception as exc:
        raise GraphFailure(f"{label}: agent run failed: {type(exc).__name__}: {exc}") from exc
    finally:
        release = getattr(agent.client, "release", None)  # pooled clients: free the slot
        if callable(release):
            release()
    if result is None:
        raise GraphFailure(f"{label}: agent produced no final message")
    return result if isinstance(result, str) else str(result)


async def followup_agent(agent: Agent, prompt: str) -> str:
    """Append ``prompt`` as a followup user message and run the agent again."""
    agent.followup(prompt)
    return await run_agent(agent)


def render_inputs_views(
    ctx: RunContext, inputs: WorkerInputs, budget: int, full_weight: int
) -> tuple[str, View]:
    """Expand-and-contract the input views to ``budget`` characters and render them.

    Returns ``(text, view)`` where ``view`` is exactly what was rendered (the worker's
    forwarded view). With no input views, returns ``("", View())``.
    """
    views = [inputs.views[sid] for sid in sorted(inputs.views)]
    if not views:
        return "", View()
    graph = ctx.codegraph
    seeds = graph.seeds_from_views(views, full_weight)
    view = graph.select(seeds, budget, ctx.session)
    return render_view(ctx.session, view), view


def render_reports(inputs: WorkerInputs) -> str:
    """Every input report under ``REPORT_LABEL`` (source id and type), ascending source id."""
    parts: list[str] = []
    for sid in sorted(inputs.reports):
        source_type = inputs.source_types.get(sid)
        type_name = source_type.value if source_type is not None else "unknown"
        label = prompts.REPORT_LABEL.format(id=sid, type=type_name)
        body = inputs.reports[sid].strip() or "(empty report)"
        parts.append(f"{label}\n{body}")
    return "\n\n".join(parts)


def _fence(text: str, lang: str = "") -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


def render_edits(ctx: RunContext, inputs: WorkerInputs) -> tuple[str, list[int]]:
    """Every input edit as ``CANDIDATE n (worker id)`` + unified diff, ascending source id.

    Returns ``(text, ids)``; candidate ``n`` (1-based) is source ``ids[n - 1]``.
    """
    ids = sorted(inputs.edits)
    parts: list[str] = []
    for n, sid in enumerate(ids, start=1):
        diff = inputs.edits[sid].unified_diff(ctx.session)
        body = _fence(diff, "diff") if diff.strip() else "(no changes)"
        parts.append(f"{prompts.CANDIDATE_LABEL.format(n=n, id=sid)}\n{body}")
    return "\n\n".join(parts), ids


def build_user_prompt(
    props: dict[str, Any], ctx: RunContext, sections: list[tuple[str, str]]
) -> str:
    """``prompts.assemble_user_prompt`` with the ``instruction`` property and the task prompt."""
    return prompts.assemble_user_prompt(props.get("instruction", "") or "", ctx.prompt, sections)


def system_prompt_for(
    node_type: WorkerType, props: dict[str, Any], *, ace_fmt: AceFormat | str | None = None
) -> str:
    """The type's system prompt with the ACE placeholder filled (see module doc)."""
    text = build_system_prompt(node_type, props)
    if ace_fmt is None:
        ace_fmt = props.get("ace_format")
    if ace_fmt is not None:
        text = text.replace(prompts.ACE_PLACEHOLDER, ace_instructions(AceFormat(ace_fmt)))
    return text


def parse_ace_output(text: str | None, fmt: AceFormat | str, ctx: RunContext) -> Edit:
    """Parse and apply an ACE reply against the pristine session; ``AceError`` propagates."""
    fmt = AceFormat(fmt)
    return apply_ace(parse_ace(text or "", fmt), ctx.session, fmt)


_INT_RE = re.compile(r"\d+")


def first_int(text: str | None) -> int | None:
    """The first run of decimal digits in ``text`` as an int, or ``None``."""
    match = _INT_RE.search(text or "")
    return int(match.group()) if match else None
