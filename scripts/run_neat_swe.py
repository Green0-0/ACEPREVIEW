#!/usr/bin/env python3
"""Evolve worker graphs with NEAT on minibatches of SWE training instances.

Example::

    python scripts/run_neat_swe.py --config configs/neat_swe_a.toml \\
        --pool /usr/project/xtmp/kg349/ace/data/train_pool.jsonl \\
        --history-dir /usr/project/xtmp/kg349/ace/runs/neat_a --batch-size 6

Like ``scripts/run_neat.py`` but the evaluator is ``ace.neat.minibatch.MinibatchTaskEvaluator``:
every generation draws ``--batch-size`` fresh instances from ``--pool`` and scores each
graph by the execution-free patch-similarity proxy against the gold patch; elites are
re-scored on each new batch (running mean). LLM calls go through ``ace.serving`` pooled
clients over the vLLM endpoints directory. ``--resume <run dir>/population_state.json``
continues a run (the batch sequence is a function of the seed, so it is reproduced).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from agent2.core.agent import Agent
from agent2.core.constructors import DirectoryConstructor

from ace.graph.llm import ModelSettings
from ace.neat.config import NeatConfig
from ace.neat.history import History
from ace.neat.minibatch import MinibatchTaskEvaluator
from ace.neat.population import Population
from ace.neat.recording import RunRecorder
from ace.serving import make_pool_client_factory

STATE_FILE = "population_state.json"
BEST_FILE = "best_genome.json"
LISTING_CHARS = 8000


def summary_line(stats: dict) -> str:
    def fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    return (f"gen {stats['generation']:>3} | best {fmt(stats['best_fitness'])} "
            f"mean {fmt(stats['mean_fitness'])} median {fmt(stats['median_fitness'])} | "
            f"species {stats['num_species']} | workers {fmt(stats['mean_workers'])} | "
            f"elites {stats['num_elites']} | evaluated {stats['evaluated']}")


async def run(args: argparse.Namespace) -> None:
    Agent.graceful_step_limit = True
    DirectoryConstructor.max_chars = LISTING_CHARS
    data = NeatConfig.load(args.config).to_toml_dict()
    if args.generations is not None:
        data["generations"] = args.generations
    if args.seed is not None:
        data["seed"] = args.seed
    data["history_dir"] = str(args.history_dir)
    cfg = NeatConfig.model_validate(data)
    run_dir = Path(args.history_dir)
    history = History(run_dir)

    from ace.swebench.data import load_pool

    pool = load_pool(args.pool)
    settings = ModelSettings(model=args.model, api_base_url=None, api_key_env_var=None)
    factory = make_pool_client_factory(args.model, endpoints_dir=args.endpoints_dir,
                                       max_tokens=args.max_tokens,
                                       per_endpoint_cap=args.per_endpoint_cap,
                                       cap_key=args.cap_key)
    evaluator = MinibatchTaskEvaluator(
        pool, settings, batch_size=args.batch_size, seed=cfg.seed, client_factory=factory,
        concurrency=args.concurrency, rollout_dir=run_dir / "rollouts", history=history,
    )
    recorder = RunRecorder(history, run_id=run_dir.resolve().name, config=cfg, model=settings,
                           evaluator_info=evaluator.describe(), seed=cfg.seed,
                           resumed_from=str(args.resume) if args.resume else None)
    population = Population(cfg, evaluator, history=history, recorder=recorder)
    (run_dir / "cli_args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"recording to {run_dir}; pool {len(pool)}; batch {args.batch_size}", flush=True)
    recorder.start()
    state_path = run_dir / STATE_FILE
    try:
        if args.resume:
            population.load_state(args.resume)
            evaluator.new_batch(population.generation)
            print(f"resumed at generation {population.generation}", flush=True)
        else:
            population.initialize()
            await population.evaluate()
            await population.bootstrap_species()
            population.save_state(state_path)
            print(summary_line(population.stats[-1]), flush=True)
        while population.generation < cfg.generations:
            await population.step()
            population.save_state(state_path)
            print(summary_line(population.stats[-1]), flush=True)
    except BaseException as exc:
        recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    recorder.finish("finished")
    best = population.best()
    (run_dir / BEST_FILE).write_text(
        json.dumps({**best.to_json(), "graph": best.to_graph().to_json()}, indent=2))
    print(f"best {best.id}: fitness {best.fitness:.3f}, {best.num_workers()} workers", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="NEAT over worker graphs on SWE minibatches")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--pool", required=True)
    p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default="gemma4-26b")
    p.add_argument("--endpoints-dir", default="/usr/project/xtmp/kg349/ace/endpoints")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=32, help="concurrent graph runs")
    p.add_argument("--per-endpoint-cap", type=int, default=None,
                   help="max active conversations per vLLM endpoint from this process")
    p.add_argument("--cap-key", default="neat",
                   help="key in <endpoints_dir>/../admission.json overriding --per-endpoint-cap live")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
