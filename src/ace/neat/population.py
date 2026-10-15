"""The NEAT population: initialisation, evaluation, speciation and reproduction.

GRAPH_NEAT_DESIGN.md section 10.5. Contract readings / additions (documented per the
contract's ambiguity rule):

* ``Population(cfg, evaluator, *, innovations=None, history=None, rng=None)``. ``rng``
  defaults to ``random.Random(cfg.seed)`` and drives every random choice (the speciator
  shares it). ``history`` defaults to ``History(cfg.history_dir)``. ``innovations``
  defaults to the history's saved ``innovations.json`` when present, else a fresh DB.
* **Adjusted fitness** is always recomputed from raw as ``raw - node_cost * num_workers``
  at the start and end of :meth:`step`; fitness sharing divides it only transiently, during
  offspring allocation and crossover parent comparison. So logged / reported adjusted values
  are never shared values and elites do not compound sharing across generations.
* **initialize** draws genomes with :func:`~ace.neat.mutation.random_valid_genome`
  (``initial_min_workers..initial_max_workers``), re-rolling up to 10 times when a genome
  duplicates an earlier one exactly (same structure hash, node innovations, types and
  properties). Initial genomes get ``operators == ["init"]``.
* **evaluate** evaluates only genomes whose ``fitness`` is None (concurrently with
  ``asyncio.gather``) and then logs each of them with ``History.log_genome`` (the history
  logs a genome once, so logging happens after evaluation). Initial genomes are therefore
  logged with ``species_id = None``; children are logged with the id of the species they
  were bred in (speciation later re-assigns them; ``species.jsonl`` has the memberships).
* **bootstrap_species** representatives are taken greedily from the population sorted by
  adjusted fitness (desc; ties: fewer workers first), skipping a genome closer than the
  speciator's threshold floor (0.05) to an already chosen representative, because two such
  genomes can never be in separate species; this equals "the top K" unless near-duplicates
  sit in the top K. After ``set_threshold_for(representatives)`` the representatives are
  speciated first (founding exactly one species each), and **every other genome joins the
  species of its nearest representative** instead of going through ``speciate``'s
  first-match-within-threshold rule. Deviation: plain ``speciate(genomes, 0)`` would found
  extra species for mutants farther than the (deliberately tight) threshold from every
  representative, so the bootstrap would not yield K species. With speciation disabled,
  K = 1 and ``speciate`` is used directly.
* **step**: ``generation += 1``; speciate; ``adjust_threshold``; ``share_fitness``;
  ``prune_stagnant``; ``allocate_offspring(population_size)``. Per species (ascending id)
  with a positive allocation ``n``: members sorted by shared adjusted fitness (desc; ties:
  higher raw, fewer workers), ``min(elitism_per_species, n)`` elites are carried unchanged
  (same object, same id, fitness kept: not re-evaluated), survivors are the top
  ``max(1, ceil(survival_fraction * size))`` members, and the rest of ``n`` are children:
  with probability ``crossover_rate`` (and >= 2 survivors) ``crossover`` of two distinct
  random survivors followed by ``mutate``, else ``mutate`` of a random survivor. Every child
  carries at least one mutation operator: ``mutate`` is retried up to 10 times when it
  applies nothing, then ``mutate_properties`` is forced. ``crossover`` is called with
  ``max_workers=cfg.max_workers`` (an oversize child falls back to the fitter parent); as a
  safety net a crossover result with more than ``max_workers`` workers is still discarded
  in favour of a mutation-only child of its first parent. Children get ``generation = self.generation``, ``parents`` = the survivors used,
  and mutation-only children have ``operators`` = just the mutation names. A shortfall
  (dead species, rounding) is topped up with mutated copies of :meth:`best`.
* **Species log**: ``species.jsonl`` gets the bootstrap species as generation 0 and, for
  each step ``g``, the species used to breed generation ``g`` (members are the previous
  generation). ``generations.jsonl`` gets stats for generation 0 (after the bootstrap) and
  each step. The innovation DB is saved to the history after initialisation, the bootstrap
  and every step.
* **run** initialises, evaluates and bootstraps only when the population is empty, then
  calls :meth:`step` until ``generation == cfg.generations`` (``cfg.generations`` steps from
  a fresh start; a population restored with :meth:`load_state` continues where it stopped).
* **best** is the genome with the highest raw fitness in the current population (ties:
  fewer workers, then earlier position); ``ValueError`` if nothing is evaluated yet.
* **save_state / load_state** write / read one JSON file holding the config, generation,
  genomes, speciator threshold and species (representatives and members as full genome
  JSON, so the next ``speciate`` sees the same candidates), the speciator's next species id,
  the innovation DB and the RNG state. Loading replaces this population's state in place.

Run recording (recording spec 11.2-11.4, 11.6):

* ``recorder`` (optional :class:`~ace.neat.recording.RunRecorder`): ``generation_started``
  is called by :meth:`initialize` (generation 0) and :meth:`step`; ``generation_finished``
  with the full generation record after the bootstrap and every step. :meth:`run` calls
  ``recorder.start()`` / ``finish()`` (``finish("failed", error=...)`` on an exception)
  unless the recorder was already started by the caller (the CLI drives the lifecycle).
* Before evaluating, a ``history`` attribute of the evaluator that is ``None`` is set to
  this population's history, so evaluators write ``evaluations.jsonl``.
* Every genome gets ``created_at`` when it is created (initialisation, bootstrap mutants,
  children; recording fields inherited from a parent are cleared) and ``evaluated_at``,
  ``eval_duration_s`` (wall time of its ``evaluate`` call) and ``task_scores`` (scores of
  ``evaluator.results_by_genome[id]`` when present, else ``[]``) after evaluation; then a
  ``genome_evaluated`` event is written. A ``speciated`` event (generation, species_count,
  threshold) is written with every species snapshot.
* Generation records (``generations.jsonl`` and :attr:`stats`) have every key of spec
  11.3. ``elites`` is the number of carried-over elites (spec 11.3; the ids are in
  ``elite_ids`` and in ``population[*].is_elite``), ``population`` is the full evaluated
  population list (its size is ``population_size``), ``species_count`` equals
  ``num_species``, ``best_genome_id`` equals ``best_id``. ``operator_counts`` counts the
  operators of every non-elite genome of the generation (for generation 0: the initial
  representatives' ``init`` and the bootstrap mutants' operators). ``started_at`` of
  generation 0 is the :meth:`initialize` time (the bootstrap start when the population
  was filled by other means).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ace.graph.clock import seconds_between, utc_now

from ace.neat.config import NeatConfig
from ace.neat.crossover import crossover
from ace.neat.evaluation import Evaluator
from ace.neat.genome import Genome
from ace.neat.history import History, worker_type_counts
from ace.neat.innovation import InnovationDB
from ace.neat.mutation import MutationContext, mutate, mutate_properties, random_valid_genome
from ace.neat.speciation import THRESHOLD_FLOOR, Species, Speciator, distance

if TYPE_CHECKING:
    from ace.neat.recording import RunRecorder

STATE_VERSION = 1
INIT_REROLLS = 10
MUTATE_RETRIES = 10


def _dedupe_key(genome: Genome) -> tuple:
    nodes = tuple(
        (inn, g.type.value, tuple(sorted(g.properties.items())))
        for inn, g in sorted(genome.nodes.items())
    )
    return (genome.structure_hash(), nodes)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


class Population:
    """A NEAT population of worker-graph genomes."""

    def __init__(
        self,
        cfg: NeatConfig,
        evaluator: Evaluator,
        *,
        innovations: InnovationDB | None = None,
        history: History | None = None,
        rng: random.Random | None = None,
        recorder: RunRecorder | None = None,
    ) -> None:
        self.cfg = cfg
        self.recorder = recorder
        self.evaluator = evaluator
        self.rng = rng if rng is not None else random.Random(cfg.seed)
        self.history = history if history is not None else History(cfg.history_dir)
        if innovations is None:
            innovations = self.history.load_innovations() or InnovationDB()
        self.innovations = innovations
        self.speciator = Speciator(cfg.speciation, rng=self.rng)
        self.genomes: list[Genome] = []
        self.generation = 0
        self.stats: list[dict[str, Any]] = []
        self.last_elites: list[str] = []
        self._generation_started_at: str | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _mark_created(genome: Genome) -> None:
        """Stamp a newly created genome; clear evaluation fields copied from a parent."""
        genome.created_at = utc_now()
        genome.evaluated_at = None
        genome.eval_duration_s = None
        genome.task_scores = []

    def _generation_begins(self) -> None:
        self._generation_started_at = utc_now()
        if self.recorder is not None:
            self.recorder.generation_started(self.generation)

    def _log_species(self) -> None:
        species = self._species_list()
        self.history.log_species(self.generation, species, self.speciator.threshold)
        self.history.log_event(
            "speciated",
            generation=self.generation,
            species_count=len(species),
            threshold=self.speciator.threshold,
        )

    @property
    def mutation_context(self) -> MutationContext:
        """Context for mutation operators (always uses the current innovation DB)."""
        return MutationContext(
            rng=self.rng,
            innovations=self.innovations,
            config=self.cfg.mutation,
            max_workers=self.cfg.max_workers,
        )

    def _base_adjusted(self, genome: Genome) -> float:
        assert genome.fitness is not None
        return genome.fitness - self.cfg.node_cost * genome.num_workers()

    def _reset_adjusted(self) -> None:
        for genome in self.genomes:
            if genome.fitness is not None:
                genome.adjusted_fitness = self._base_adjusted(genome)

    def _mutated_child(self, parent: Genome) -> tuple[Genome, list[str]]:
        """``mutate(parent)`` with at least one applied operator; returns (child, names)."""
        ctx = self.mutation_context
        base = len(parent.operators)
        child = mutate(parent, ctx)
        for _ in range(MUTATE_RETRIES):
            applied = [op for op in child.operators[base:] if op != "noop"]
            if applied:
                return child, applied
            child = mutate(parent, ctx)
        applied = [op for op in child.operators[base:] if op != "noop"]
        if not applied:
            child.operators = child.operators[:base]
            if mutate_properties(child, ctx):
                child.operators.append("mutate_properties")
            applied = child.operators[base:]
        return child, applied

    def _mutation_only(self, parent: Genome) -> Genome:
        child, applied = self._mutated_child(parent)
        child.operators = list(applied)
        child.generation = self.generation
        child.parents = (parent.id,)
        self._mark_created(child)
        return child

    def _crossover_child(self, a: Genome, b: Genome) -> Genome:
        crossed = crossover(
            a, b, self.rng, innovations=self.innovations, max_workers=self.cfg.max_workers
        )
        if crossed.num_workers() > self.cfg.max_workers:  # safety net
            return self._mutation_only(a)
        child, _ = self._mutated_child(crossed)
        child.generation = self.generation
        child.parents = (a.id, b.id)
        self._mark_created(child)
        return child

    def _save_innovations(self) -> None:
        self.history.save_innovations(self.innovations)

    # --------------------------------------------------------------- lifecycle

    def initialize(self) -> None:
        """Create ``population_size`` random valid genomes for generation 0."""
        cfg = self.cfg
        self.genomes = []
        self.generation = 0
        self._generation_begins()
        seen: set[tuple] = set()
        for _ in range(cfg.population_size):
            genome = None
            for _ in range(INIT_REROLLS + 1):
                genome = random_valid_genome(
                    self.innovations,
                    self.rng,
                    cfg.mutation,
                    min_workers=cfg.initial_min_workers,
                    max_workers=cfg.initial_max_workers,
                    generation=0,
                )
                if _dedupe_key(genome) not in seen:
                    break
            assert genome is not None
            seen.add(_dedupe_key(genome))
            genome.operators = ["init"]
            genome.parents = ()
            self._mark_created(genome)
            self.genomes.append(genome)
        self._save_innovations()

    def _attach_history(self) -> None:
        evaluator = self.evaluator
        if hasattr(evaluator, "history") and getattr(evaluator, "history", None) is None:
            try:
                evaluator.history = self.history  # type: ignore[attr-defined]
            except AttributeError:  # read-only attribute: recording is best effort
                pass

    async def _timed_evaluate(self, genome: Genome) -> float:
        started = utc_now()
        score = await self.evaluator.evaluate(genome)
        finished = utc_now()
        genome.evaluated_at = finished
        genome.eval_duration_s = seconds_between(started, finished)
        results = getattr(self.evaluator, "results_by_genome", None)
        records = results.get(genome.id) if isinstance(results, dict) else None
        genome.task_scores = (
            [float(r.get("score", 0.0)) for r in records] if records else []
        )
        return score

    async def _evaluate_genomes(self, genomes: list[Genome]) -> None:
        if not genomes:
            return
        self._attach_history()
        for genome in genomes:
            if genome.created_at is None:
                genome.created_at = utc_now()
        scores = await asyncio.gather(*(self._timed_evaluate(g) for g in genomes))
        for genome, score in zip(genomes, scores):
            genome.fitness = float(score)
            genome.adjusted_fitness = self._base_adjusted(genome)
        for genome in genomes:
            self.history.log_genome(genome)
            self.history.log_event(
                "genome_evaluated",
                genome_id=genome.id,
                fitness=genome.fitness,
                adjusted_fitness=genome.adjusted_fitness,
                duration_s=genome.eval_duration_s,
            )

    async def evaluate(self) -> None:
        """Evaluate every genome without fitness; set raw and adjusted fitness; log them."""
        await self._evaluate_genomes([g for g in self.genomes if g.fitness is None])

    def _ranked(self, genomes: list[Genome]) -> list[Genome]:
        return sorted(
            genomes,
            key=lambda g: (-(g.adjusted_fitness if g.adjusted_fitness is not None else -math.inf),
                           g.num_workers()),
        )

    def _choose_representatives(self, k: int) -> list[Genome]:
        ranked = self._ranked(self.genomes)
        reps: list[Genome] = []
        for genome in ranked:
            if len(reps) == k:
                break
            if all(distance(genome, r, self.cfg.speciation) > THRESHOLD_FLOOR for r in reps):
                reps.append(genome)
        for genome in ranked:  # not enough distinct genomes: fill with the next best
            if len(reps) == k:
                break
            if not any(genome is r for r in reps):
                reps.append(genome)
        return reps

    async def bootstrap_species(self) -> None:
        """Keep the top K genomes as species representatives and regrow the population."""
        if self._generation_started_at is None:
            self._generation_begins()
        await self.evaluate()
        cfg = self.cfg
        k = cfg.speciation.target_species if cfg.speciation.enabled else 1
        k = min(k, len(self.genomes))
        reps = self._choose_representatives(k)
        children: list[Genome] = []
        for i in range(cfg.population_size - len(reps)):
            rep = reps[i % len(reps)]
            child, applied = self._mutated_child(rep)
            child.generation = 0
            child.parents = (rep.id,)
            child.operators = list(applied)
            self._mark_created(child)
            children.append(child)
        self.genomes = list(reps) + children
        await self._evaluate_genomes(children)
        if cfg.speciation.enabled:
            self.speciator.set_threshold_for(reps)
            self.speciator.speciate(reps, 0)
            founded = [self.speciator.species[sid] for sid in sorted(self.speciator.species)]
            for child in children:
                home = min(
                    founded,
                    key=lambda sp: distance(child, sp.representative, cfg.speciation),
                )
                home.members.append(child)
                child.species_id = home.id
        else:
            self.speciator.speciate(self.genomes, 0)
        self.speciator.prune_stagnant(0)  # records best_fitness / last_improved only
        self._save_innovations()
        self._log_species()
        self.last_elites = []
        self._log_stats(evaluated=len(children))

    def _species_list(self) -> list[Species]:
        return [self.speciator.species[sid] for sid in sorted(self.speciator.species)]

    async def step(self) -> None:
        """Breed and evaluate the next generation."""
        if any(g.fitness is None for g in self.genomes):
            await self.evaluate()
        cfg = self.cfg
        rng = self.rng
        self.generation += 1
        self._generation_begins()
        self._reset_adjusted()
        speciator = self.speciator
        speciator.speciate(self.genomes, self.generation)
        speciator.adjust_threshold()
        speciator.share_fitness()
        speciator.prune_stagnant(self.generation)
        allocation = speciator.allocate_offspring(cfg.population_size)
        self._log_species()

        next_genomes: list[Genome] = []
        elites: list[Genome] = []
        for sid in sorted(allocation):
            quota = allocation[sid]
            species = speciator.species.get(sid)
            if quota <= 0 or species is None or not species.members:
                continue
            members = sorted(
                species.members,
                key=lambda g: (-(g.adjusted_fitness or 0.0), -(g.fitness or 0.0), g.num_workers()),
            )
            keep = members[: min(cfg.speciation.elitism_per_species, quota, len(members))]
            elites.extend(keep)
            next_genomes.extend(keep)
            survivors = members[: max(1, math.ceil(cfg.speciation.survival_fraction * len(members)))]
            for _ in range(quota - len(keep)):
                if len(survivors) >= 2 and rng.random() < cfg.crossover_rate:
                    a, b = rng.sample(survivors, 2)
                    child = self._crossover_child(a, b)
                else:
                    child = self._mutation_only(rng.choice(survivors))
                child.species_id = sid
                next_genomes.append(child)

        if len(next_genomes) < cfg.population_size:
            best = self.best()
            while len(next_genomes) < cfg.population_size:
                child = self._mutation_only(best)
                child.species_id = best.species_id
                next_genomes.append(child)
        self.genomes = next_genomes[: cfg.population_size]
        self._reset_adjusted()
        children = [g for g in self.genomes if g.fitness is None]
        await self._evaluate_genomes(children)
        self._reset_adjusted()
        self._save_innovations()
        self.last_elites = [g.id for g in elites]
        self._log_stats(evaluated=len(children))

    async def run(self) -> Genome:
        """Initialise (if empty), evaluate, bootstrap, then step up to ``cfg.generations``."""
        recorder = self.recorder
        owns = recorder is not None and recorder.started_at is None
        if owns:
            recorder.start()
        try:
            if not self.genomes:
                self.initialize()
                await self.evaluate()
                await self.bootstrap_species()
            while self.generation < self.cfg.generations:
                await self.step()
            best = self.best()
        except BaseException as exc:
            if owns:
                recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        if owns:
            recorder.finish("finished")
        return best

    def best(self) -> Genome:
        """Highest raw fitness among evaluated genomes (ties: fewer workers)."""
        scored = [g for g in self.genomes if g.fitness is not None]
        if not scored:
            raise ValueError("no evaluated genomes in the population")
        return min(scored, key=lambda g: (-g.fitness, g.num_workers()))  # type: ignore[operator]

    # -------------------------------------------------------------------- stats

    def generation_stats(self) -> dict[str, Any]:
        """Summary statistics of the current population."""
        scored = [g for g in self.genomes if g.fitness is not None]
        raw = [g.fitness for g in scored]
        adjusted = [self._base_adjusted(g) for g in scored]
        workers = [g.num_workers() for g in self.genomes]
        best = self.best() if scored else None
        return {
            "population": len(self.genomes),
            "best_fitness": max(raw) if raw else None,
            "mean_fitness": statistics.fmean(raw) if raw else None,
            "median_fitness": statistics.median(raw) if raw else None,
            "best_adjusted": max(adjusted) if adjusted else None,
            "mean_adjusted": statistics.fmean(adjusted) if adjusted else None,
            "median_adjusted": statistics.median(adjusted) if adjusted else None,
            "num_species": len(self.speciator.species),
            "species_sizes": {
                str(sid): len(sp.members) for sid, sp in sorted(self.speciator.species.items())
            },
            "threshold": self.speciator.threshold,
            "mean_workers": statistics.fmean(workers) if workers else None,
            "best_id": best.id if best else None,
            "best_workers": best.num_workers() if best else None,
            "num_elites": len(self.last_elites),
            "elites": len(self.last_elites),
            "elite_ids": list(self.last_elites),
        }

    def population_records(self) -> list[dict[str, Any]]:
        """The ``population`` list of a generation record (spec 11.3)."""
        elites = set(self.last_elites)
        records = []
        for g in self.genomes:
            graph = g.to_graph()
            records.append(
                {
                    "genome_id": g.id,
                    "species_id": g.species_id,
                    "fitness": g.fitness,
                    "adjusted_fitness": (
                        self._base_adjusted(g) if g.fitness is not None else None
                    ),
                    "num_workers": g.num_workers(),
                    "structure_hash": graph.structure_hash(),
                    "is_elite": g.id in elites,
                    "parents": list(g.parents),
                    "operators": list(g.operators),
                    "worker_types": worker_type_counts(graph.nodes),
                }
            )
        return records

    def _log_stats(self, *, evaluated: int) -> None:
        finished = utc_now()
        started = self._generation_started_at or finished
        elites = set(self.last_elites)
        operator_counts = Counter(
            op for g in self.genomes if g.id not in elites for op in g.operators
        )
        stats = self.generation_stats()
        stats["population_size"] = stats.pop("population")
        stats.update(
            {
                "started_at": started,
                "finished_at": finished,
                "duration_s": seconds_between(started, finished),
                "evaluated": evaluated,
                "species_count": stats["num_species"],
                "best_genome_id": stats["best_id"],
                "operator_counts": dict(sorted(operator_counts.items())),
                "population": self.population_records(),
            }
        )
        self._generation_started_at = None
        self.stats.append({"generation": self.generation, **stats})
        self.history.log_generation(self.generation, stats)
        if self.recorder is not None:
            self.recorder.generation_finished(self.generation, {"generation": self.generation, **stats})

    # -------------------------------------------------------------------- state

    def save_state(self, path: Path | str) -> None:
        """Write everything needed to resume this population to ``path`` (JSON, atomic)."""
        path = Path(path)
        species = [
            {
                "id": sp.id,
                "representative": sp.representative.to_json(),
                "members": [m.to_json() for m in sp.members],
                "created": sp.created,
                "best_fitness": _finite_or_none(sp.best_fitness),
                "last_improved": sp.last_improved,
            }
            for sp in self._species_list()
        ]
        version, internal, gauss = self.rng.getstate()
        state = {
            "version": STATE_VERSION,
            "config": self.cfg.to_toml_dict(),
            "generation": self.generation,
            "genomes": [g.to_json() for g in self.genomes],
            "speciator": {
                "threshold": self.speciator.threshold,
                "next_species_id": getattr(self.speciator, "_next_id", None),
                "species": species,
            },
            "innovations": self.innovations.to_json(),
            "rng_state": [version, list(internal), gauss],
            "last_elites": list(self.last_elites),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)

    def load_state(self, path: Path | str) -> None:
        """Replace this population's state with one written by :meth:`save_state`."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported population state version {state.get('version')!r}")
        self.generation = int(state["generation"])
        self.genomes = [Genome.from_json(d) for d in state["genomes"]]
        by_id = {g.id: g for g in self.genomes}

        def restore(data: dict) -> Genome:
            return by_id.get(data["id"]) or Genome.from_json(data)

        sp_state = state["speciator"]
        self.speciator.threshold = float(sp_state["threshold"])
        self.speciator.species = {}
        for raw in sp_state["species"]:
            best = raw.get("best_fitness")
            sp = Species(
                id=int(raw["id"]),
                representative=restore(raw["representative"]),
                members=[restore(m) for m in raw["members"]],
                created=int(raw["created"]),
                best_fitness=-math.inf if best is None else float(best),
                last_improved=int(raw["last_improved"]),
            )
            self.speciator.species[sp.id] = sp
        next_id = sp_state.get("next_species_id")
        if next_id is None:
            next_id = max(self.speciator.species, default=-1) + 1
        self.speciator._next_id = int(next_id)  # noqa: SLF001 - restoring private counter
        self.innovations = InnovationDB.from_json(state["innovations"])
        version, internal, gauss = state["rng_state"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.last_elites = list(state.get("last_elites", []))
        self._save_innovations()


__all__ = ["Population"]
