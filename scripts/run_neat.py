#!/usr/bin/env python3
"""Evolve worker graphs with NEAT.

Examples::

    uv run python scripts/run_neat.py --mock --generations 5 --population 12
    uv run python scripts/run_neat.py --tasks tasks.json --model claude-haiku-5-5 \\
        --history-dir runs/neat1
    uv run python scripts/run_neat.py --tasks tasks.json --resume runs/neat1/population_state.json

``--mock`` scores genomes structurally with ``MockEvaluator`` (no network). Otherwise
``--tasks`` names a JSON list of ``{"project_dir": ..., "prompt": ..., "scorer": ...}``
objects. ``scorer`` is ``"nonempty_diff"`` (builtin: 1.0 if the graph's edit is non-empty,
else 0.0) or ``"package.module:function"``, a callable ``(edit, session) -> float`` (sync or
async). SWE-bench style evaluation plugs in this way: the scorer applies the Edit (e.g.
``edit.to_workspace(session)``) and runs the task's test harness. Relative project dirs
resolve against the tasks file's directory.

Run directory: every run records into its own directory (``run.json``, ``events.jsonl``,
``generations.jsonl``, ``genomes.jsonl``, ``species.jsonl``, ``evaluations.jsonl``, ...; see
the recording spec). Without ``--history-dir`` it is ``<config history_dir>/<run id>``
(default ``.ace/neat/<run id>``, run id ``YYYYmmdd-HHMMSS-<6 hex>``); with an explicit
``--history-dir DIR`` the run records directly into ``DIR`` (run id = ``DIR``'s name).
``--resume STATE`` without ``--history-dir`` continues in the directory holding ``STATE``.

Worker rollouts of task runs go to ``<run dir>/rollouts/<genome id>/<task index>/``
(never into the task's project folder; see ``TaskEvaluator``).

The population state is saved to ``<run dir>/population_state.json`` after every
generation (pass it to ``--resume``) and the best genome to ``<run dir>/best_genome.json``
(genome JSON plus its worker-graph JSON under ``"graph"``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ace.graph.llm import ModelSettings
from ace.neat.config import NeatConfig
from ace.neat.evaluation import Evaluator, MockEvaluator, TaskEvaluator, load_tasks
from ace.neat.history import History
from ace.neat.population import Population
from ace.neat.recording import RunRecorder

STATE_FILE = "population_state.json"
BEST_FILE = "best_genome.json"


def load_env(var: str) -> None:
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


def build_config(args: argparse.Namespace) -> NeatConfig:
    """Config file (or defaults) with command-line overrides applied."""
    data = NeatConfig.load(args.config).to_toml_dict()
    if args.generations is not None:
        data["generations"] = args.generations
    if args.population is not None:
        data["population_size"] = args.population
        data["speciation"]["target_species"] = min(
            data["speciation"]["target_species"], args.population
        )
    if args.seed is not None:
        data["seed"] = args.seed
    if args.history_dir is not None:
        data["history_dir"] = str(args.history_dir)
    return NeatConfig.model_validate(data)


def new_run_id() -> str:
    """``YYYYmmdd-HHMMSS-<6 hex>`` (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def resolve_run_dir(args: argparse.Namespace, cfg: NeatConfig) -> tuple[Path, str]:
    """``(run directory, run id)`` following the rules in the module docstring."""
    if args.history_dir is not None:
        run_dir = Path(args.history_dir)
    elif args.resume is not None:
        run_dir = Path(args.resume).resolve().parent
    else:
        run_dir = Path(cfg.history_dir) / new_run_id()
    return run_dir, run_dir.resolve().name


def model_settings(args: argparse.Namespace) -> ModelSettings | None:
    if args.mock:
        return None
    return ModelSettings(
        model=args.model, api_base_url=args.api_base, api_key_env_var=args.api_key_env
    )


def build_evaluator(args: argparse.Namespace, history_dir: Path) -> Evaluator:
    if args.mock:
        return MockEvaluator(noise=args.mock_noise)
    if args.tasks is None:
        raise SystemExit("either --mock or --tasks TASKS.json is required")
    load_env(args.api_key_env)
    tasks = load_tasks(args.tasks)
    settings = ModelSettings(
        model=args.model, api_base_url=args.api_base, api_key_env_var=args.api_key_env
    )
    return TaskEvaluator(
        tasks,
        settings,
        concurrency=args.concurrency,
        rollout_dir=history_dir / "rollouts",
    )


def summary_line(stats: dict) -> str:
    def fmt(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    return (
        f"gen {stats['generation']:>3} | best {fmt(stats['best_fitness'])} "
        f"mean {fmt(stats['mean_fitness'])} median {fmt(stats['median_fitness'])} | "
        f"adj best {fmt(stats['best_adjusted'])} | species {stats['num_species']} "
        f"(thr {fmt(stats['threshold'])}) | workers {fmt(stats['mean_workers'])} | "
        f"elites {stats['num_elites']} | evaluated {stats['evaluated']}"
    )


async def run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    history_dir, run_id = resolve_run_dir(args, cfg)
    cfg = cfg.model_copy(update={"history_dir": history_dir})
    history = History(history_dir)
    evaluator = build_evaluator(args, history_dir)
    describe = getattr(evaluator, "describe", None)
    recorder = RunRecorder(
        history,
        run_id=run_id,
        config=cfg,
        model=model_settings(args),
        evaluator_info=describe() if callable(describe) else {"type": "task", "tasks": []},
        seed=cfg.seed,
        resumed_from=str(args.resume) if args.resume is not None else None,
    )
    population = Population(cfg, evaluator, history=history, recorder=recorder)
    print(f"Run {run_id}: recording to {history_dir}")
    recorder.start()
    try:
        await evolve(args, cfg, population, history_dir)
    except BaseException as exc:
        recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    recorder.finish("finished")
    print(f"Run directory: {history_dir}")


async def evolve(
    args: argparse.Namespace, cfg: NeatConfig, population: Population, history_dir: Path
) -> None:
    state_path = history_dir / STATE_FILE

    if args.resume is not None:
        population.load_state(args.resume)
        print(f"Resumed from {args.resume} at generation {population.generation}")
    else:
        population.initialize()
        await population.evaluate()
        await population.bootstrap_species()
        population.save_state(state_path)
        print(summary_line(population.stats[-1]))
    while population.generation < cfg.generations:
        await population.step()
        population.save_state(state_path)
        print(summary_line(population.stats[-1]))

    best = population.best()
    best_path = history_dir / BEST_FILE
    best_path.write_text(
        json.dumps({**best.to_json(), "graph": best.to_graph().to_json()}, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nBest genome {best.id}: fitness {best.fitness:.3f}, "
        f"{best.num_workers()} workers, operators {best.operators}"
    )
    print(f"Best genome graph JSON: {best_path}")
    print(f"History: {history_dir} (state: {state_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve worker graphs with NEAT.")
    parser.add_argument("--config", type=Path, default=None, help="NEAT config TOML file")
    parser.add_argument("--mock", action="store_true", help="use the structural MockEvaluator")
    parser.add_argument("--mock-noise", type=float, default=0.1,
                        help="per-structure noise amplitude for --mock (default 0.1)")
    parser.add_argument("--tasks", type=Path, default=None, help="tasks JSON file")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--api-base", default="https://api.anthropic.com/v1/",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                        help="environment variable holding the API key")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="maximum concurrently running task graphs")
    parser.add_argument(
        "--history-dir", type=Path, default=None,
        help="run directory. Default: a fresh <config history_dir>/<run id> "
             "(.ace/neat/<run id>), or the directory of --resume STATE. When given "
             "explicitly, the run records directly into this directory.",
    )
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="population state JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
