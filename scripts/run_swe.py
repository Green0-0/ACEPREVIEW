#!/usr/bin/env python3
"""Run the baseline agent2 agent or an ACE worker graph on SWE instances; write predictions.

Examples::

    # baseline: agent2's standard single-agent pipeline with all code tools
    python scripts/run_swe.py --mode baseline --instances data/verified_tasks.jsonl \\
        --out runs/verified/baseline/preds.jsonl
    # an evolved graph (best_genome.json, or any WorkerGraph JSON)
    python scripts/run_swe.py --mode graph --graph runs/neat/best_genome.json \\
        --instances data/verified_tasks.jsonl --out runs/verified/neat/preds.jsonl

Output: one JSON line per instance, in the SWE-bench predictions format
(``instance_id``, ``model_name_or_path``, ``model_patch``) plus ``usage`` (LLM calls and
tokens from ``ace.serving``), ``duration_s``, ``failure`` and, when the instance carries a
gold ``patch`` (training/validation pools), the execution-free ``proxy`` scores. The file
is appended to and resumable: instances already present are skipped. ``--shard k/n`` runs
every n-th instance starting at k, so several processes can split one instance file.

Both modes use the same LLM settings (``ace.serving`` pooled clients over the vLLM
endpoints, max_tokens cap, thinking disabled), the same task prompt
(``ace.swebench.data.make_prompt``), the same graceful step limit (``Agent.graceful_step_limit``)
and the same directory-listing budget (``DirectoryConstructor.max_chars``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from agent2.core.agent import Agent
from agent2.core.constructors import DirectoryConstructor
from agent2.core.objects import AgentConfig, WorkspacePolicy
from agent2.core.pipeline import SingleAgentPipeline
from agent2.core.workspace import Workspace

from ace.graph.edits import Edit
from ace.graph.graph import WorkerGraph
from ace.graph.llm import ModelSettings
from ace.graph.runner import GraphRunner
from ace.graph.types import GraphFailure
from ace.neat.minibatch import CachedRunContext
from ace.serving import make_pool_client_factory, metered
from ace.sessions import SessionCache

logger = logging.getLogger("run_swe")

LISTING_CHARS = 8000

# agent2's basic_agent system prompt (config/agents/basic_agent.toml), verbatim.
BASELINE_SYSTEM_PROMPT = """
You are a coding agent operating on an isolated in-memory workspace. Inspect the
project with search and view tools, then make only the edits needed to complete
the user's request. Your edits will be shown as a diff and will not be committed
to disk.

{{FILES*}}
"""

# Every implemented agent2 code tool. Excluded: delegate / message_subagent /
# cancel_subagent (sub-agent orchestration, i.e. a different pipeline; ACE forbids them too)
# and bash / tools (declared but unimplemented in agent2).
BASELINE_TOOLS = [
    "search", "list_folder", "search_folders",
    "view", "view_path", "view_files", "view_file", "views_at", "view_at",
    "view_element", "view_multi_lines", "view_lines",
    "open_file", "open_lines", "open_element", "open_element_at",
    "replace_file", "replace_lines", "replace_element", "replace_element_at",
    "make_empty_files", "move_files",
]


def baseline_config(model: str, max_steps: int) -> AgentConfig:
    return AgentConfig(
        path="config/agents/swe_baseline.toml",
        subagent_name="swe_baseline",
        subagent_description="agent2 basic agent with every code tool",
        model=model,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        prompt_wrap_bottom="{{PROMPT}}",
        followup_prompt_wrapper="{{PROMPT}}",
        max_steps=max_steps,
        constructors=["default", "directory"],
        hooks=["tool_call"],
        primary_tools=list(BASELINE_TOOLS),
        secondary_tools=[],
        workspace_policy=WorkspacePolicy.ISOLATED,
        subagent_names=[],
    )


class BaselinePipeline(SingleAgentPipeline):
    """agent2's SingleAgentPipeline with an in-memory config and an injected client."""

    def __init__(self, config: AgentConfig, prompt: str, client) -> None:
        super().__init__(config.path, prompt)
        self.config = config
        self.client = client

    def _create_agent(self, session):
        return Agent(session=session, workspace=Workspace(session), config=self.config,
                     prompt=self.prompt, client=self.client)


def load_graph(path: Path) -> WorkerGraph:
    data = json.loads(path.read_text())
    if "graph" in data:
        data = data["graph"]
    graph = WorkerGraph.from_json(data)
    graph.validate()
    return graph


def done_ids(out: Path) -> set[str]:
    ids: set[str] = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                ids.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


async def run_one(args, instance, sessions: SessionCache, factory, graph, write) -> None:
    from ace.swebench.data import make_prompt

    iid = instance["instance_id"]
    rollout_dir = Path(args.rollout_root) / iid
    started = time.monotonic()
    record = {"instance_id": iid, "model_name_or_path": args.name, "mode": args.mode,
              "model_patch": "", "failure": None}
    session = await asyncio.to_thread(sessions.session, instance["project_dir"], rollout_dir)
    prompt = make_prompt(instance["problem_statement"])
    with metered() as meter:
        try:
            if args.mode == "baseline":
                client = factory()
                pipeline = BaselinePipeline(baseline_config(args.model, args.max_steps), prompt, client)
                try:
                    result = await pipeline.run(session)
                finally:
                    client.release()
                edit = Edit.from_workspace(result.workspace)
                record["final_answer"] = (result.answer or "")[:2000] if isinstance(result.answer, str) else None
            else:
                ctx = CachedRunContext(session, prompt, ModelSettings(model=args.model, api_key_env_var=None),
                                       factory, rollout_dir=rollout_dir)
                result = await GraphRunner(graph).run(ctx)
                edit = result.edit
            record["model_patch"] = edit.git_patch(session)
        except GraphFailure as exc:
            record["failure"] = f"GraphFailure: {exc}"[:2000]
        except Exception as exc:  # noqa: BLE001 - one instance must not stop the run
            record["failure"] = f"{type(exc).__name__}: {exc}"[:2000]
            logger.debug("instance %s failed:\n%s", iid, traceback.format_exc())
    record["usage"] = meter.to_json()
    record["duration_s"] = round(time.monotonic() - started, 2)
    if instance.get("patch"):
        from ace.swebench.proxy import localization_scores, patch_similarity

        record["proxy_score"] = patch_similarity(record["model_patch"], instance["patch"])
        record["proxy"] = localization_scores(record["model_patch"], instance["patch"])
    write(record)
    logger.info("%s done in %.0fs: patch %d chars, %d calls, %d tokens%s", iid, record["duration_s"],
                len(record["model_patch"]), meter.calls, meter.prompt_tokens + meter.completion_tokens,
                f", FAILED {record['failure'][:120]}" if record["failure"] else "")


async def main_async(args) -> None:
    Agent.graceful_step_limit = True
    DirectoryConstructor.max_chars = LISTING_CHARS
    instances = [json.loads(l) for l in Path(args.instances).read_text().splitlines() if l.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        instances = [i for i in instances if i["instance_id"] in wanted]
    k, n = (int(x) for x in args.shard.split("/"))
    instances = instances[k::n]
    if args.limit:
        instances = instances[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip: set[str] = set()
    for other in sorted(out.parent.glob("preds*.jsonl")):  # every shard's results
        skip |= done_ids(other)
    todo = [i for i in instances if i["instance_id"] not in skip]
    missing = [i for i in todo if not Path(i["project_dir"]).is_dir()]
    if missing:
        print(f"skipping {len(missing)} instances without a materialized project_dir", flush=True)
        todo = [i for i in todo if Path(i["project_dir"]).is_dir()]
    print(f"{len(todo)} to run ({len(instances) - len(todo)} already in {out})", flush=True)
    graph = load_graph(Path(args.graph)) if args.mode == "graph" else None
    factory = make_pool_client_factory(args.model, endpoints_dir=args.endpoints_dir,
                                       static_urls=args.url or None, max_tokens=args.max_tokens,
                                       per_endpoint_cap=args.per_endpoint_cap,
                                       cap_key=args.cap_key)
    sessions = SessionCache(max_entries=4)  # Verified snapshots are never reused
    fh = out.open("a")

    def write(record) -> None:
        fh.write(json.dumps(record) + "\n")
        fh.flush()

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(instance):
        async with sem:
            await run_one(args, instance, sessions, factory, graph, write)

    await asyncio.gather(*(guarded(i) for i in todo))
    fh.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=["baseline", "graph"], required=True)
    p.add_argument("--graph", help="best_genome.json or WorkerGraph JSON (graph mode)")
    p.add_argument("--instances", required=True, help="JSONL of instances (instance_id, project_dir, problem_statement)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None, help="model_name_or_path (default: mode)")
    p.add_argument("--model", default="gemma4-26b")
    p.add_argument("--endpoints-dir", default="/usr/project/xtmp/kg349/ace/endpoints")
    p.add_argument("--url", action="append", help="static endpoint URL(s) instead of discovery")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--per-endpoint-cap", type=int, default=None,
                   help="max active conversations per vLLM endpoint from this process")
    p.add_argument("--cap-key", default="swe",
                   help="key in <endpoints_dir>/../admission.json overriding --per-endpoint-cap live")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="", help="comma-separated instance ids to run")
    p.add_argument("--rollout-root", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.mode == "graph" and not args.graph:
        p.error("--graph is required in graph mode")
    args.name = args.name or args.mode
    args.rollout_root = args.rollout_root or str(Path(args.out).parent / "rollouts")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
