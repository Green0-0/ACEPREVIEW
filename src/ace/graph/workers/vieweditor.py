"""VIEWEDITOR worker: one tool-less LLM step that edits the shown code in ACE format.

See GRAPH_NEAT_DESIGN.md sections 1.2, 4.1 and 9. The input views are expanded and
contracted to ``context_budget`` (``render_inputs_views``, weighted by
``view_full_weight``) and rendered under ``VIEWS_HEADER``; input reports (if any) follow
under ``REPORTS_HEADER``. The system prompt carries the ACE instructions for the
``ace_format`` property. The single reply is parsed and applied against the pristine
session with ``parse_ace_output``; any ``AceError`` (no block, header not permitted by
the format, unknown target, overlap) propagates as a ``GraphFailure``.

Choices where the contract is open: if the rendered input views are empty the worker
raises ``GraphFailure`` (a VIEWEDITOR must be shown code). Outputs: ``view`` = the view
actually rendered, ``edit`` = the applied ACE edit, ``report`` = ``None`` (ACE workers
emit no report), ``rollout_ids`` = the agent's rollout id.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.ace import AceFormat
from ace.graph.graph import WorkerNode
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    parse_ace_output,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)

if TYPE_CHECKING:
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["ViewEditorWorker"]

MAX_STEPS = 1
"""ACE-producing workers run a single LLM step."""


@register
class ViewEditorWorker(Worker):
    """Tool-less editor: rendered views (+ reports) in, ACE edit out."""

    type = WorkerType.VIEWEDITOR

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Render the inputs, run one LLM step and apply the ACE reply (see module docstring)."""
        from ace.graph.runner import WorkerOutputs

        fmt = AceFormat(props["ace_format"])
        system_prompt = system_prompt_for(self.type, props, ace_fmt=fmt)
        text, shown = render_inputs_views(
            ctx, inputs, int(props["context_budget"]), props["view_full_weight"]
        )
        if not text.strip():
            raise GraphFailure(f"{node.label()}: no input code to show")
        sections = [(prompts.VIEWS_HEADER, text)]
        if inputs.reports:
            sections.append((prompts.REPORTS_HEADER, render_reports(inputs)))
        agent = self.make_agent(
            ctx,
            node,
            props,
            tools=[],
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(props, ctx, sections),
            max_steps=MAX_STEPS,
        )
        reply = await run_agent(agent)
        edit = parse_ace_output(reply, fmt, ctx)
        return WorkerOutputs(view=shown, edit=edit, report=None, rollout_ids=[agent.rollout.id])
