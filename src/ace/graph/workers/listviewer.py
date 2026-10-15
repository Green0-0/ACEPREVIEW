"""LISTVIEWER worker: pick locations from a ranked candidate list, then report on them.

See GRAPH_NEAT_DESIGN.md sections 1.2 and 9. The worker uses no tools and runs two LLM
turns on one agent:

1. The user prompt is the task plus a candidate list (``path::element  <signature>`` or
   ``path`` lines) built from the input views: seeds come from
   ``codegraph.seeds_from_views`` (weighted by ``view_full_weight``), the candidates are
   the keys of ``codegraph.select(seeds, list_budget)`` listed in rank order by
   ``codegraph.candidate_list`` (at most ``list_budget`` characters). If that listing is
   empty, the ranked keys of the whole graph (uniform seeds) are listed instead.
2. The agent's selection is parsed (rules below) into a View (elements FULL, bare files
   COLLAPSED), expanded with ``codegraph.select`` seeded by that View to
   ``context_budget`` and merged with it (so every chosen item is always shown), rendered,
   and sent as ``LISTVIEWER_FOLLOWUP + "\\n\\n" + rendered``. The reply is the report.

Selection parsing rules (choices where the contract is open):

* If the reply contains complete fenced code blocks (``extract_all_codeblocks``), every
  line of every block is a candidate entry; otherwise every line of the whole reply is.
* Per line: surrounding whitespace is stripped, then leading bullets (``-``, ``*``,
  ``+``, ``•``) and numbering (``1.``, ``1)``, ``(1)``), each followed by whitespace, are
  removed, then all backticks are removed. Only the first whitespace-separated token is
  kept, so a candidate line copied together with its signature still parses. Surrounding
  ``*``/quotes and trailing ``,``/``;`` are stripped and a leading ``./`` is dropped.
  Empty lines are skipped.
* ``path::element`` selects an element; ``path`` (or ``path::``) selects the file. The
  path must be a file in the pristine session; the element is resolved with
  ``normalize_element_path`` (so ``llm_path`` forms such as ``Class.method.42`` work).
  Entries that do not resolve (unknown file, unknown element, element on an unparsed
  file, line-span forms) are dropped silently; duplicates are ignored.
* If nothing resolves, ``GraphFailure`` is raised. An empty (whitespace-only) report also
  raises ``GraphFailure``.

Outputs: ``view`` = the merged View that was rendered, ``report`` = the second reply,
``rollout_ids`` = the agent's rollout id.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ace.graph import prompts
from ace.graph.graph import WorkerNode
from ace.graph.rendering import render_view
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View, normalize_element_path
from ace.graph.workers import register
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    followup_agent,
    run_agent,
    system_prompt_for,
)
from agent2.utils.codeblocks import extract_all_codeblocks

if TYPE_CHECKING:
    from agent2.core.session import Session
    from ace.graph.runner import RunContext, WorkerInputs, WorkerOutputs

__all__ = ["ListViewerWorker", "parse_selection", "selection_entries"]

MAX_STEPS = 2
"""Two LLM turns: the selection and the report."""

_BULLET_RE = re.compile(r"^(?:[-*+\u2022]\s+|\(?\d+[.)]\s+)+")


def _clean_entry(line: str) -> str:
    """One selection line reduced to its ``path`` / ``path::element`` token (may be ``""``)."""
    text = _BULLET_RE.sub("", line.strip())
    text = text.replace("`", "").strip()
    if not text:
        return ""
    token = text.split()[0].strip("*\"'").rstrip(",;")
    if token.startswith("./"):
        token = token[2:]
    return token


def selection_entries(reply: str) -> list[str]:
    """Candidate entries from the agent's reply (code-block lines, else all lines)."""
    blocks = extract_all_codeblocks(reply or "")
    lines = [line for block in blocks for line in block.splitlines()] if blocks else (reply or "").splitlines()
    entries: list[str] = []
    for line in lines:
        entry = _clean_entry(line)
        if entry:
            entries.append(entry)
    return entries


def parse_selection(reply: str, session: Session) -> View:
    """Resolve the reply's entries against the pristine session into a View.

    Elements are added at FULL, bare files at COLLAPSED; unresolvable entries are dropped.
    The returned View may be empty (the caller decides whether that is a failure).
    """
    view = View()
    for entry in selection_entries(reply):
        path, _, element = entry.partition("::")
        path = path.strip()
        file = session.files.get(path)
        if file is None:
            continue
        canonical = normalize_element_path(file, element)
        if canonical is None:
            continue
        if canonical:
            view.add_element(path, canonical, Level.FULL)
        else:
            view.add_file(path, Level.COLLAPSED)
    return view


@register
class ListViewerWorker(Worker):
    """Tool-less two-step viewer: choose from a candidate list, then report on the code."""

    type = WorkerType.LISTVIEWER

    def _candidates(self, ctx: RunContext, inputs: WorkerInputs, props: dict[str, Any]) -> str:
        graph = ctx.codegraph
        budget = int(props["list_budget"])
        views = [inputs.views[sid] for sid in sorted(inputs.views)]
        seeds = graph.seeds_from_views(views, props["view_full_weight"])
        selected = graph.select(seeds, budget, ctx.session)
        text = graph.candidate_list(selected, ctx.session, budget, seeds=seeds)
        if not text.strip():
            ranked = [key for key, _ in graph.rank({})]
            text = graph.candidate_list(ranked, ctx.session, budget)
        return text

    async def run(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        props: dict[str, Any],
    ) -> WorkerOutputs:
        """Run the selection turn and the report turn (see module docstring)."""
        from ace.graph.runner import WorkerOutputs

        candidates = self._candidates(ctx, inputs, props)
        user_prompt = build_user_prompt(
            props, ctx, [(prompts.LISTVIEWER_CANDIDATES_HEADER, candidates)]
        )
        agent = self.make_agent(
            ctx,
            node,
            props,
            tools=[],
            system_prompt=system_prompt_for(self.type, props),
            user_prompt=user_prompt,
            max_steps=MAX_STEPS,
        )
        selection = await run_agent(agent)
        chosen = parse_selection(selection, ctx.session)
        if chosen.is_empty():
            raise GraphFailure(f"{node.label()}: no selected location resolves in the repository")

        graph = ctx.codegraph
        chosen_seeds = graph.seeds_from_views([chosen], props["view_full_weight"])
        final_view = graph.select(chosen_seeds, int(props["context_budget"]), ctx.session).merge(chosen)
        rendered = render_view(ctx.session, final_view)

        report = await followup_agent(agent, prompts.LISTVIEWER_FOLLOWUP + "\n\n" + rendered)
        if not report.strip():
            raise GraphFailure(f"{node.label()}: empty report")
        return WorkerOutputs(view=final_view, report=report, rollout_ids=[agent.rollout.id])
