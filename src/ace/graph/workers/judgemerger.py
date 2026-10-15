"""JUDGEMERGER worker: pick one candidate edit and forward it unchanged (§9).

Tool-less, one LLM step. The user prompt holds, after the task: the input views rendered
to ``context_budget`` (``prompts.VIEWS_HEADER``), every candidate edit from
:func:`~ace.graph.workers.base.render_edits` (``prompts.CANDIDATES_HEADER``; candidate
``n`` is source ``ids[n - 1]``, ascending source id) and every input report
(``prompts.REPORTS_HEADER``).

The first integer in the reply (:func:`~ace.graph.workers.base.first_int`) is the chosen
candidate number. Outputs: view = what was rendered, edit = the chosen input ``Edit``
object itself (identity preserved), no report.

Choices made where the contract is open:

* ``WorkerOutputs`` has no slot for the chosen source, so the worker records it on
  ``self.last_choice`` (source node id of the forwarded edit; ``None`` before any run).
  A worker instance run concurrently on several nodes would overwrite it; it is meant
  for tests and debugging only.
* Fewer than two input edits, missing views (empty rendering), or a reply with no integer
  or an out-of-range number raise ``GraphFailure`` naming the node (and quoting the reply).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    first_int,
    render_edits,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)

if TYPE_CHECKING:
    from ace.graph.graph import WorkerNode
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["JudgeMergerWorker"]

_REPLY_EXCERPT = 200


@register
class JudgeMergerWorker(Worker):
    """Chooses the best of two or more candidate edits by number."""

    type = WorkerType.JUDGEMERGER

    def __init__(self) -> None:
        self.last_choice: int | None = None
        """Source node id of the edit forwarded by the most recent run (debug/tests)."""

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Render views, candidates and reports; run one step; forward the chosen edit."""
        from ace.graph.runner import WorkerOutputs

        if len(inputs.edits) < 2:
            raise GraphFailure(
                f"{node.label()}: judgemerger needs at least two input edits, "
                f"got {len(inputs.edits)}"
            )
        system_prompt = system_prompt_for(self.type, props)
        views_text, shown = render_inputs_views(
            ctx, inputs, props["context_budget"], props["view_full_weight"]
        )
        if not views_text.strip():
            raise GraphFailure(
                f"{node.label()}: judgemerger needs at least one non-empty input view"
            )
        candidates_text, ids = render_edits(ctx, inputs)
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
        n = first_int(reply)
        if n is None or not 1 <= n <= len(ids):
            excerpt = reply.strip()[:_REPLY_EXCERPT]
            raise GraphFailure(
                f"{node.label()}: reply does not name a candidate 1..{len(ids)}: {excerpt!r}"
            )
        chosen = ids[n - 1]
        self.last_choice = chosen
        return WorkerOutputs(
            view=shown, edit=inputs.edits[chosen], report=None, rollout_ids=[agent.rollout.id]
        )
