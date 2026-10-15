"""VIEWER worker: a read-only, tool-using investigator (GRAPH_NEAT_DESIGN.md sections 1.2, 9).

The worker runs one agent2 ``Agent`` with the viewing/search tools named by its
``tool_set`` property (``prompts.TOOL_SETS``), for at most ``max_steps`` steps. It has no
inputs. Its outputs are:

* ``view``: everything it looked at with tools, remapped onto the pristine repository
  (``agent.view_recorder.finalize(agent.workspace)``);
* ``report``: the agent's final message;
* ``rollout_ids``: the agent's rollout id.

Choices made where the contract is open:

* An empty report (``None`` or whitespace only) raises ``GraphFailure``: a REPORT edge
  would otherwise carry nothing useful. ``None`` is already rejected by
  :func:`~ace.graph.workers.base.run_agent`.
* Agent failures are handled by ``run_agent`` (wrapped into ``GraphFailure`` naming the
  node); nothing is caught here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.workers import register
from ace.graph.workers.base import Worker, build_user_prompt, run_agent, system_prompt_for

if TYPE_CHECKING:
    from ace.graph.graph import WorkerNode
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["ViewerWorker"]


@register
class ViewerWorker(Worker):
    """Explores the repository with viewing/search tools; outputs VIEW and REPORT."""

    type = WorkerType.VIEWER

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Run the viewing agent and return its recorded view and final report."""
        from ace.graph.runner import WorkerOutputs

        agent = self.make_agent(
            ctx,
            node,
            props,
            tools=list(prompts.TOOL_SETS[props["tool_set"]]),
            system_prompt=system_prompt_for(self.type, props),
            user_prompt=build_user_prompt(props, ctx, []),
            max_steps=int(props["max_steps"]),
        )
        report = await run_agent(agent)
        if not report or not report.strip():
            raise GraphFailure(f"{node.label()}: empty report")
        view = agent.view_recorder.finalize(agent.workspace)  # type: ignore[attr-defined]
        return WorkerOutputs(view=view, report=report, rollout_ids=[agent.rollout.id])
