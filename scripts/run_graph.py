#!/usr/bin/env python3
"""Run one worker graph on a project folder and print the resulting diff.

Usage::

    uv run python scripts/run_graph.py <project_folder> <prompt> [--graph graph.json]
        [--model MODEL] [--api-base URL] [--api-key-env VAR] [--rollout-dir DIR]
        [--print-views]

The graph JSON is ``WorkerGraph.to_json()`` output (default: ``minimal_graph()``, one
EDITOR feeding END). Nothing is written to the project files: the graph runtime writes
only worker rollouts, into the Session's rollout dir. ``--rollout-dir`` chooses it
(resolved against the current directory); the default is the project's ``.agent2/task``
(the Session default). Exits with status 1 on ``GraphFailure`` or an invalid graph.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agent2.core.session import Session
from ace.graph.graph import WorkerGraph, minimal_graph
from ace.graph.llm import ModelSettings, make_client_factory
from ace.graph.rendering import measure, render_view
from ace.graph.runner import GraphResult, GraphRunner, RunContext
from ace.graph.types import GraphFailure, InvalidGraph

DEFAULT_MODEL = "claude-haiku-5-5"  # same defaults as config/agents/basic_agent.toml
DEFAULT_API_BASE = "https://api.anthropic.com/v1/"
REPORT_EXCERPT = 300


def load_env_key(var: str) -> None:
    """Load ``var`` from a git-ignored ``.env`` when it is not already set.

    Looks in the ACE project root first, then in the attached ``agent2/`` library folder.
    """
    if var in os.environ:
        return
    root = Path(__file__).resolve().parents[1]
    for env_path in (root / ".env", root / "agent2" / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.removeprefix("export ").strip() == var:
                os.environ[var] = value.strip().strip("'\"")
                return


def load_graph(path: Path | None) -> WorkerGraph:
    if path is None:
        return minimal_graph()
    return WorkerGraph.from_json(json.loads(path.read_text(encoding="utf-8")))


def excerpt(text: str | None, limit: int = REPORT_EXCERPT) -> str:
    if text is None:
        return "(none)"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def print_summary(
    session: Session, graph: WorkerGraph, result: GraphResult, *, print_views: bool
) -> None:
    print("\nWorkers (execution order):\n")
    for nid in result.order:
        node = graph.nodes[nid]
        outputs = result.outputs[nid]
        print(f"[{nid}] {node.type.value}")
        print(f"    rollouts: {', '.join(outputs.rollout_ids) or '(none)'}")
        if outputs.view is None:
            print("    view:     (none)")
        else:
            rendered = render_view(session, outputs.view)
            elements = sum(len(fv.levels) for fv in outputs.view.files.values())
            print(
                f"    view:     {len(outputs.view.files)} files, {elements} entries, "
                f"{measure(rendered)} chars rendered"
            )
        if outputs.edit is not None:
            print(f"    edit:     {len(outputs.edit.files)} file(s) {sorted(outputs.edit.touched_paths())}")
        print(f"    report:   {excerpt(outputs.report)}")
        if print_views and outputs.view is not None:
            print(f"\n----- view of worker {nid} -----")
            print(render_view(session, outputs.view) or "(empty view)")
            print(f"----- end of view of worker {nid} -----\n")


DEFAULT_ROLLOUT_DIR = ".agent2/task"
"""Session default, relative to the project folder."""


async def run(args: argparse.Namespace) -> int:
    rollout_dir = str(args.rollout_dir.resolve()) if args.rollout_dir else DEFAULT_ROLLOUT_DIR
    session = Session(str(args.folder), rollout_dir=rollout_dir)
    try:
        graph = load_graph(args.graph)
        runner = GraphRunner(graph)
    except (InvalidGraph, ValueError, KeyError) as exc:
        print(f"Invalid graph: {exc}", file=sys.stderr)
        return 1

    settings = ModelSettings(
        model=args.model, api_base_url=args.api_base, api_key_env_var=args.api_key_env
    )
    ctx = RunContext(
        session=session,
        prompt=args.prompt,
        settings=settings,
        client_factory=make_client_factory(settings),
        rollout_dir=session.rollout_path,
    )
    try:
        result = await runner.run(ctx)
    except GraphFailure as exc:
        node_id = getattr(exc, "node_id", None)
        where = f" (node {node_id})" if node_id is not None else ""
        print(f"Graph run failed{where}: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1

    print_summary(session, graph, result, print_views=args.print_views)
    print("\nFinal diff:\n")
    print(result.diff or "No file changes.")
    print(f"\nRollouts saved to {session.rollout_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a worker graph on a project and print the resulting diff."
    )
    parser.add_argument("folder", type=Path, help="Project folder the workers operate on")
    parser.add_argument("prompt", help="Task prompt given to every worker")
    parser.add_argument(
        "--graph", type=Path, default=None, help="WorkerGraph JSON (default: one EDITOR -> END)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE, help=f"OpenAI-compatible API base URL (default {DEFAULT_API_BASE})"
    )
    parser.add_argument(
        "--api-key-env",
        default="ANTHROPIC_API_KEY",
        help="Environment variable holding the API key (default ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help=f"Directory for worker rollouts (default: <project_folder>/{DEFAULT_ROLLOUT_DIR})",
    )
    parser.add_argument(
        "--print-views", action="store_true", help="Print every worker's rendered view"
    )
    args = parser.parse_args()
    if args.api_key_env:
        load_env_key(args.api_key_env)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
