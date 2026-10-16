"""Minibatch SWE task evaluation for NEAT runs on real repositories.

:class:`MinibatchTaskEvaluator` is a :class:`~ace.neat.evaluation.TaskEvaluator` whose
task list is a minibatch drawn from a pool of SWE instances. The population calls
:meth:`new_batch` at the start of every generation (``Population.step``), so every
generation is scored on fresh issues (cheaper evaluations, less overfitting to a fixed
set). Generation 0 (initial population and bootstrap mutants) uses batch 0. Batches are a
deterministic function of ``(seed, generation)``, so a resumed run sees the same batches.

Because fitness values from different batches are not comparable, ``reevaluate_elites``
is True: ``Population.step`` re-scores carried-over elites on the new batch and keeps the
running mean over every batch they have seen (see ``Population.step``).

Per task the fitness is an execution-free proxy (``ace.swebench.proxy.patch_similarity``
of the graph's git patch against the gold patch). Each ``evaluations.jsonl`` record gains
``instance_id``, ``batch``, ``usage`` (tokens of the run, from ``ace.serving.metered``)
and ``proxy`` (``localization_scores``). Sessions and codebase graphs are cached per
repository snapshot (``ace.sessions.SessionCache``), so the genomes of one generation
parse each repository once.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import statistics
from pathlib import Path
from typing import Any, Callable

from ace.graph.clock import utc_now
from ace.graph.llm import ModelSettings
from ace.graph.runner import RunContext
from ace.graph.types import GraphFailure
from ace.neat.evaluation import Task, TaskEvaluator, build_evaluation_record
from ace.serving import metered
from ace.sessions import FrozenSession, SessionCache

logger = logging.getLogger(__name__)


class CachedRunContext(RunContext):
    """RunContext whose codegraph comes from the frozen session's shared cache."""

    @property
    def codegraph(self):  # type: ignore[override]
        if self._codegraph is None:
            session = self.session
            if isinstance(session, FrozenSession):
                self._codegraph = session.codegraph()
            else:
                from ace.graph.codegraph import CodebaseGraph

                self._codegraph = CodebaseGraph(session)
        return self._codegraph

    @codegraph.setter
    def codegraph(self, value) -> None:
        self._codegraph = value


def _proxy_scorer(gold_patch: str) -> Callable[[Any, Any], float]:
    from ace.swebench.proxy import patch_similarity

    def score(edit, session) -> float:
        return float(patch_similarity(edit.git_patch(session), gold_patch))

    return score


def instance_task(instance: dict[str, Any]) -> Task:
    """A Task for one SWE instance dict (``project_dir``, ``problem_statement``, ``patch``)."""
    from ace.swebench.data import make_prompt

    task = Task(Path(instance["project_dir"]), make_prompt(instance["problem_statement"]),
                _proxy_scorer(instance.get("patch", "")))
    task.instance = instance  # type: ignore[attr-defined]
    return task


class MinibatchTaskEvaluator(TaskEvaluator):
    """TaskEvaluator over a fresh minibatch of SWE instances per generation."""

    reevaluate_elites = True

    def __init__(
        self,
        pool: list[dict[str, Any]],
        settings: ModelSettings,
        *,
        batch_size: int,
        seed: int = 0,
        client_factory: Callable[[], Any] | None = None,
        concurrency: int = 32,
        rollout_dir: Path | None = None,
        history=None,
        session_cache: SessionCache | None = None,
        fixed: bool = False,
    ) -> None:
        super().__init__([], settings, client_factory=client_factory, concurrency=concurrency,
                         rollout_dir=rollout_dir, history=history)
        self.pool = list(pool)
        self.batch_size = batch_size
        self.seed = seed
        self.fixed = fixed  # True: evaluate on the whole pool (validation / test sets)
        self.sessions = session_cache or SessionCache()
        self._load_locks: dict[str, asyncio.Lock] = {}
        self.batch = -1
        if fixed:
            self.tasks = [instance_task(i) for i in self.pool]
            self.batch = 0
        else:
            self.new_batch(0)

    # ---------------------------------------------------------------- batches

    def new_batch(self, generation: int) -> None:
        if self.fixed or generation == self.batch:
            return
        from ace.swebench.data import sample_minibatch

        rng = random.Random(f"{self.seed}:{generation}")
        batch = sample_minibatch(self.pool, self.batch_size, rng)
        self.tasks = [instance_task(i) for i in batch]
        self.batch = generation
        logger.info("generation %d minibatch: %s", generation,
                    [t.instance["instance_id"] for t in self.tasks])  # type: ignore[attr-defined]
        if self.history is not None:
            self.history.log_event(
                "minibatch", generation=generation,
                instance_ids=[t.instance["instance_id"] for t in self.tasks],  # type: ignore[attr-defined]
            )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "swe_minibatch",
            "pool_size": len(self.pool),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "fixed": self.fixed,
            "tasks": [],
        }

    # ---------------------------------------------------------------- sessions

    async def _session(self, task: Task, rollout_dir: Path) -> FrozenSession:
        key = str(Path(task.project_dir).resolve())
        lock = self._load_locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Parsing a repository is CPU bound; do it off the event loop once per repo.
            await asyncio.to_thread(self.sessions.base, key, self.rollout_root / "_base")
        return self.sessions.session(key, rollout_dir)

    # ---------------------------------------------------------------- one task

    def task_rollout_dir(self, genome, index: int, task: Task) -> Path:
        iid = getattr(task, "instance", {}).get("instance_id", str(index))
        return self.rollout_root / genome.id / f"b{self.batch}_{iid}"

    async def _run_task(self, genome, index: int, task: Task) -> dict[str, Any]:
        from ace.graph.runner import GraphRunner

        instance = getattr(task, "instance", {})
        record: dict[str, Any] = {
            "task": index,
            "instance_id": instance.get("instance_id"),
            "batch": self.batch,
            "project_dir": str(task.project_dir),
            "score": 0.0,
            "failure": None,
            "node_id": None,
            "diff_chars": 0,
        }
        rollout_dir = self.task_rollout_dir(genome, index, task)
        record["rollout_dir"] = str(rollout_dir)
        async with self._get_semaphore():
            started = utc_now()
            if self.history is not None:
                self.history.log_event("evaluation_start", genome_id=genome.id, task_index=index)
            session = await self._session(task, rollout_dir)
            ctx = CachedRunContext(session, task.prompt, self.settings, self.client_factory,
                                   rollout_dir=rollout_dir)
            graph = genome.to_graph()
            result = None
            failure: GraphFailure | None = None
            with metered() as meter:
                try:
                    result = await GraphRunner(graph).run(ctx)
                except GraphFailure as exc:
                    failure = exc
                    record["failure"] = f"{type(exc).__name__}: {exc}"
                    record["node_id"] = getattr(exc, "node_id", None)
                    logger.warning("genome %s failed on %s: %s", genome.id,
                                   record["instance_id"], record["failure"][:300])
            record["usage"] = meter.to_json()
            if result is not None:
                score = task.scorer(result.edit, session)
                if inspect.isawaitable(score):
                    score = await score
                record["score"] = float(score)
                record["diff_chars"] = len(result.diff)
                if instance.get("patch"):
                    from ace.swebench.proxy import localization_scores

                    try:
                        record["proxy"] = localization_scores(result.diff, instance["patch"])
                    except Exception as exc:  # noqa: BLE001 - analysis only
                        record["proxy"] = {"error": str(exc)}
            if self.history is not None:
                full = build_evaluation_record(
                    genome=genome, graph=graph, task_index=index, task=task, session=session,
                    rollout_dir=rollout_dir, started_at=started, finished_at=utc_now(),
                    score=record["score"], result=result, failure=failure,
                )
                full.update({k: record.get(k) for k in ("instance_id", "batch", "usage", "proxy")})
                self.history.log_evaluation(full)
                self.history.log_event(
                    "evaluation_end", genome_id=genome.id, task_index=index,
                    score=record["score"], failure=record["failure"],
                    duration_s=full["duration_s"],
                )
        return record

    async def evaluate(self, genome) -> float:
        records = list(
            await asyncio.gather(*(self._run_task(genome, i, t) for i, t in enumerate(self.tasks)))
        )
        self.last_results = records
        self.results_by_genome.setdefault(genome.id, [])
        self.results_by_genome[genome.id] = records
        if not records:
            return 0.0
        return statistics.fmean(r["score"] for r in records)


__all__ = ["CachedRunContext", "MinibatchTaskEvaluator", "instance_task"]
