"""PICKMERGER worker: compose one new ACE edit set from several candidate edits (§9).

Tool-less, one LLM step. The user prompt is the JUDGEMERGER one: after the task, the input
views rendered to ``context_budget`` (``prompts.VIEWS_HEADER``), every candidate edit from
:func:`~ace.graph.workers.base.render_edits` (``prompts.CANDIDATES_HEADER``) and every
input report (``prompts.REPORTS_HEADER``). The system prompt carries the ACE instructions
for the ``ace_format`` property.

The reply is parsed as ACE and applied to the pristine session; the result is a fresh
``Edit`` (candidates are never applied, only the reply's blocks). Outputs: view = what was
rendered, edit = the parsed edit, no report.

Choices made where the contract is open: fewer than two input edits or missing views
(empty rendering) raise ``GraphFailure`` naming the node; ``AceError`` propagates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.ace import AceFormat
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    parse_ace_output,
    render_edits,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)

if TYPE_CHECKING:
    from ace.graph.graph import WorkerNode
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["PickMergerWorker"]


@register
class PickMergerWorker(Worker):
    """Merges the best parts of two or more candidate edits into one ACE edit set."""

    type = WorkerType.PICKMERGER

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Render views, candidates and reports; run one step; parse the ACE reply."""
        from ace.graph.runner import WorkerOutputs

        if len(inputs.edits) < 2:
            raise GraphFailure(
                f"{node.label()}: pickmerger needs at least two input edits, "
                f"got {len(inputs.edits)}"
            )
        fmt = AceFormat(props["ace_format"])
        system_prompt = system_prompt_for(self.type, props, ace_fmt=fmt)
        views_text, shown = render_inputs_views(
            ctx, inputs, props["context_budget"], props["view_full_weight"]
        )
        if not views_text.strip():
            raise GraphFailure(
                f"{node.label()}: pickmerger needs at least one non-empty input view"
            )
        candidates_text, _ids = render_edits(ctx, inputs)
        user_prompt = build_user_prompt(
            props,
            ctx,
            [
                (prompts.VIEWS_HEADER, views_text),
                (prompts.CANDIDATES_HEADER, candidates_text),
                (prompts.REPORTS_HEADER, render_reports(inputs)),
            ],
        )
        agent = self.make_agent(
            ctx,
            node,
            props,
            tools=[],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_steps=1,
        )
        reply = await run_agent(agent)
        edit = parse_ace_output(reply, fmt, ctx)
        return WorkerOutputs(view=shown, edit=edit, report=None, rollout_ids=[agent.rollout.id])
