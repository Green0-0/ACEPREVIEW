"""CRITIC worker: rewrite a single candidate edit as a complete ACE edit set (§9).

Tool-less, one LLM step. The user prompt holds, after the task:

* the input views rendered to ``context_budget`` (``prompts.VIEWS_HEADER``);
* the single input edit under ``prompts.EDIT_UNDER_REVIEW_HEADER``, labelled
  ``from worker {id} ({type})`` and followed by its unified diff in a ```` ```diff ````
  fence (or ``(no changes)`` for an empty edit). If the edit's author also sent a report,
  it is shown immediately under the diff, labelled with ``prompts.REPORT_LABEL`` plus
  ``(author of the edit under review)``;
* every other input report under ``prompts.REPORTS_HEADER``.

The reply is parsed as ACE with the ``ace_format`` property and applied to the pristine
session. The critic outputs the COMPLETE new edit set, so its edit replaces the input edit
entirely (nothing from the input edit is carried over). Outputs: view = what was rendered,
edit = the parsed edit, no report.

Choices made where the contract is open: a source with no entry in
``inputs.source_types`` is labelled ``unknown``; missing views (empty rendering) or an
edit count other than one raise ``GraphFailure`` naming the node; ``AceError`` from the
reply propagates unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.ace import AceFormat
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    _fence,
    build_user_prompt,
    parse_ace_output,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)

if TYPE_CHECKING:
    from ace.graph.graph import WorkerNode
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["CriticWorker", "AUTHOR_REPORT_SUFFIX"]

AUTHOR_REPORT_SUFFIX = " (author of the edit under review)"
"""Appended to the author's ``REPORT_LABEL`` line under the diff."""


def _type_name(inputs: WorkerInputs, sid: int) -> str:
    source_type = inputs.source_types.get(sid)
    return source_type.value if source_type is not None else "unknown"


@register
class CriticWorker(Worker):
    """Reviews one candidate edit and outputs the complete corrected edit set in ACE."""

    type = WorkerType.CRITIC

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Render views + the edit under review (+ reports), run one step, parse ACE."""
        from ace.graph.runner import WorkerOutputs

        if len(inputs.edits) != 1:
            raise GraphFailure(
                f"{node.label()}: critic needs exactly one input edit, got {len(inputs.edits)}"
            )
        fmt = AceFormat(props["ace_format"])
        system_prompt = system_prompt_for(self.type, props, ace_fmt=fmt)
        views_text, shown = render_inputs_views(
            ctx, inputs, props["context_budget"], props["view_full_weight"]
        )
        if not views_text.strip():
            raise GraphFailure(f"{node.label()}: critic needs at least one non-empty input view")

        (author,) = inputs.edits
        edit = inputs.edits[author]
        diff = edit.unified_diff(ctx.session)
        body = _fence(diff, "diff") if diff.strip() else "(no changes)"
        parts = [f"from worker {author} ({_type_name(inputs, author)})\n{body}"]
        if author in inputs.reports:
            label = prompts.REPORT_LABEL.format(id=author, type=_type_name(inputs, author))
            body = inputs.reports[author].strip() or "(empty report)"
            parts.append(f"{label}{AUTHOR_REPORT_SUFFIX}\n{body}")
        edit_text = "\n\n".join(parts)

        others = {sid: text for sid, text in inputs.reports.items() if sid != author}
        reports_text = render_reports(replace(inputs, reports=others))

        user_prompt = build_user_prompt(
            props,
            ctx,
            [
                (prompts.VIEWS_HEADER, views_text),
                (prompts.EDIT_UNDER_REVIEW_HEADER, edit_text),
                (prompts.REPORTS_HEADER, reports_text),
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
        new_edit = parse_ace_output(reply, fmt, ctx)
        return WorkerOutputs(
            view=shown, edit=new_edit, report=None, rollout_ids=[agent.rollout.id]
        )
