"""EDITOR worker: a tool-using engineer editing a private workspace (GRAPH_NEAT_DESIGN.md §9).

The worker runs one agent2 ``Agent`` with the viewing/search/editing tools named by its
``tool_set`` property (``prompts.TOOL_SETS``), for at most ``max_steps`` steps. Optional
inputs are added to the user prompt as context:

* input VIEWs are expanded-and-contracted to ``context_budget`` characters with
  :func:`~ace.graph.workers.base.render_inputs_views` and placed under
  ``prompts.VIEWS_HEADER``;
* input REPORTs are rendered with :func:`~ace.graph.workers.base.render_reports`
  under ``prompts.REPORTS_HEADER``.

Outputs:

* ``view``: the tool-recorded view (finalized onto pristine) merged with the view that
  was rendered into the prompt, so the forwarded view covers everything it was shown;
* ``edit``: ``Edit.from_workspace(agent.workspace)``;
* ``report``: the agent's final message;
* ``rollout_ids``: the agent's rollout id.

Choices made where the contract is open:

* EDITOR has no ``view_full_weight`` property; FULL sightings are weighted 2
  (:data:`FULL_WEIGHT`) when seeding the input-view selection.
* An empty edit is not a failure here (the runner / evaluation decide what it is worth).
  An empty report (``None`` or whitespace only) raises ``GraphFailure``.
* Agent failures are handled by ``run_agent``; nothing is caught here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.edits import Edit
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import View
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)

if TYPE_CHECKING:
    from ace.graph.graph import WorkerNode
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["EditorWorker", "FULL_WEIGHT"]

FULL_WEIGHT = 2
"""Relevance weight of a FULL sighting vs COLLAPSED when rendering input views."""


@register
class EditorWorker(Worker):
    """Edits a private workspace with tools; outputs VIEW, EDIT and REPORT."""

    type = WorkerType.EDITOR

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Run the editing agent with any input views/reports as context."""
        from ace.graph.runner import WorkerOutputs

        sections: list[tuple[str, str]] = []
        shown = View()
        if inputs.views:
            text, shown = render_inputs_views(
                ctx, inputs, int(props["context_budget"]), full_weight=FULL_WEIGHT
            )
            sections.append((prompts.VIEWS_HEADER, text))
        if inputs.reports:
            sections.append((prompts.REPORTS_HEADER, render_reports(inputs)))

        agent = self.make_agent(
            ctx,
            node,
            props,
            tools=list(prompts.TOOL_SETS[props["tool_set"]]),
            system_prompt=system_prompt_for(self.type, props),
            user_prompt=build_user_prompt(props, ctx, sections),
            max_steps=int(props["max_steps"]),
        )
        report = await run_agent(agent)
        if not report or not report.strip():
            raise GraphFailure(f"{node.label()}: empty report")
        recorded = agent.view_recorder.finalize(agent.workspace)  # type: ignore[attr-defined]
        return WorkerOutputs(
            view=recorded.merge(shown),
            edit=Edit.from_workspace(agent.workspace),
            report=report,
            rollout_ids=[agent.rollout.id],
        )
