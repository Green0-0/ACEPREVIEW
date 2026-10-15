"""Speciation, fitness sharing, offspring allocation and stagnation (GRAPH_NEAT_DESIGN.md 10.4).

Compatibility distance::

    delta = c_excess * E / N + c_disjoint * D / N + c_properties * H

* Genes are the union of node and connection innovations of a genome (enabled and
  disabled connections alike). A gene is *matching* if its innovation is in both genomes.
  A non-matching gene of genome X is *excess* if its innovation is greater than the
  largest innovation of the other genome (all of X's genes are excess if the other genome
  is empty), otherwise *disjoint*. E and D count both genomes' non-matching genes.
* ``N = 1`` always. Worker graphs are small by design (``max_workers`` nodes plus a few
  dozen connections), so excess and disjoint genes are counted raw. The original NEAT
  switch to ``N = max(gene counts)`` at 20 genes was removed because it shrank every
  distance 20x mid-run and collapsed the population into one species.
* ``H`` is the mean, over matched node genes, of the fraction of property names (union of
  both genes' names) whose indices differ; a name missing on one side counts as differing;
  a matched node with no property names contributes 0. ``H = 0`` with no matched nodes.

Contract readings / additions:

* ``Speciator(cfg, rng=None)``: ``rng`` (a ``random.Random``) chooses representatives;
  ``None`` means ``random.Random(0)`` so runs are reproducible.
* ``speciate`` joins a genome to the first species (ascending id) whose representative is
  at distance ``<= threshold``. At its start every existing species picks a random member
  of its previous members as representative (keeping the old one if it had no members),
  then its member list is cleared. Species ids are allocated from a counter starting at 0
  and are never reused. New species record ``created = last_improved = generation`` and
  ``best_fitness = -inf``. Species left without members are dropped.
* ``adjust_threshold`` is not called by ``speciate``; the population calls it. The step
  is *relative*: with fewer species than ``target_species`` the threshold is multiplied by
  ``1 - threshold_step`` (lower threshold, more species), with more it is multiplied by
  ``1 + threshold_step``; floor 0.05. (An absolute step barely moved thresholds that are
  typically 3 to 15 after the bootstrap.)
* ``share_fitness`` raises ``ValueError`` if a member's ``adjusted_fitness`` is None. If
  the lowest ``adjusted_fitness`` over all species is negative, every member is first
  shifted by ``-min`` (so the minimum becomes 0) and then divided by its species size;
  non-negative values are not shifted. Raw ``fitness`` is never changed.
* ``allocate_offspring`` reads ``adjusted_fitness`` (raises ``ValueError`` if None). If the
  minimum is negative every member is shifted by ``-min``. Species share = sum of shifted
  member values. If all shares are 0 the allocation is proportional to member counts.
  Largest-remainder rounding, ties broken by ascending species id. Returns ``{}`` when
  there are no species.
* ``prune_stagnant`` updates ``best_fitness`` / ``last_improved`` from members' raw
  ``fitness`` (None ignored; ``last_improved`` moves only on a strict improvement), then
  removes species with ``generation - last_improved > stagnation_generations``, except the
  species holding the genome with the highest raw fitness overall. If no member of any
  species has a raw fitness, nothing is pruned.
* ``Species`` is declared with ``eq=False`` (identity equality, hashable); fields after
  ``representative`` have defaults (``members=[]``, ``created=0``,
  ``best_fitness=-inf``, ``last_improved=0``).
* ``best_genome`` returns the member with the highest raw fitness across all species
  (None if no member has a fitness).
* With ``cfg.enabled == False``: ``speciate`` puts everyone in species 0 (keeping that
  species' ``created``/``best_fitness``/``last_improved`` across calls); ``share_fitness``,
  ``prune_stagnant`` and ``adjust_threshold`` are no-ops; ``allocate_offspring`` returns
  ``{0: total}``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations

from ace.neat.config import SpeciationConfig
from ace.neat.genome import Genome

THRESHOLD_FLOOR = 0.05
"""Lowest compatibility threshold ever used."""



def _genes(genome: Genome) -> set[int]:
    return set(genome.nodes) | set(genome.connections)


def property_difference(a: Genome, b: Genome) -> float:
    """``H``: mean fraction of differing property indices over matched node genes."""
    matched = sorted(a.nodes.keys() & b.nodes.keys())
    if not matched:
        return 0.0
    total = 0.0
    for inn in matched:
        pa, pb = a.nodes[inn].properties, b.nodes[inn].properties
        names = pa.keys() | pb.keys()
        if not names:
            continue
        differing = sum(1 for n in names if pa.get(n) != pb.get(n))
        total += differing / len(names)
    return total / len(matched)


def excess_disjoint(a: Genome, b: Genome) -> tuple[int, int]:
    """``(E, D)`` over the union of node and connection innovations."""
    ga, gb = _genes(a), _genes(b)
    max_a = max(ga, default=-1)
    max_b = max(gb, default=-1)
    excess = disjoint = 0
    for inn in ga - gb:
        if inn > max_b:
            excess += 1
        else:
            disjoint += 1
    for inn in gb - ga:
        if inn > max_a:
            excess += 1
        else:
            disjoint += 1
    return excess, disjoint


def distance(a: Genome, b: Genome, cfg: SpeciationConfig) -> float:
    """NEAT compatibility distance (formula in the module docstring)."""
    excess, disjoint = excess_disjoint(a, b)
    return (
        cfg.c_excess * excess
        + cfg.c_disjoint * disjoint
        + cfg.c_properties * property_difference(a, b)
    )


@dataclass(eq=False)
class Species:
    """A group of compatible genomes."""

    id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    created: int = 0
    best_fitness: float = -math.inf
    last_improved: int = 0


class Speciator:
    """Assigns genomes to species and runs sharing, allocation and stagnation pruning."""

    def __init__(self, cfg: SpeciationConfig, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.rng = rng if rng is not None else random.Random(0)
        self.threshold: float = cfg.initial_threshold
        self.species: dict[int, Species] = {}
        self._next_id = 0

    # ------------------------------------------------------------- speciation

    def _new_species(self, genome: Genome, generation: int) -> Species:
        sp = Species(
            id=self._next_id,
            representative=genome,
            members=[],
            created=generation,
            best_fitness=-math.inf,
            last_improved=generation,
        )
        self._next_id += 1
        self.species[sp.id] = sp
        return sp

    def speciate(self, population: list[Genome], generation: int) -> None:
        """Assign every genome to a species and set ``genome.species_id``."""
        if not self.cfg.enabled:
            self._speciate_disabled(population, generation)
            return
        ordered = [self.species[sid] for sid in sorted(self.species)]
        for sp in ordered:
            if sp.members:
                sp.representative = self.rng.choice(sp.members)
            sp.members = []
        for genome in population:
            home: Species | None = None
            for sp in ordered:
                if distance(genome, sp.representative, self.cfg) <= self.threshold:
                    home = sp
                    break
            if home is None:
                home = self._new_species(genome, generation)
                ordered.append(home)
            home.members.append(genome)
            genome.species_id = home.id
        self.species = {sp.id: sp for sp in ordered if sp.members}

    def _speciate_disabled(self, population: list[Genome], generation: int) -> None:
        if not population:
            self.species = {}
            return
        sp = self.species.get(0)
        if sp is None:
            sp = Species(id=0, representative=population[0], created=generation,
                         last_improved=generation)
        sp.representative = population[0]
        sp.members = list(population)
        for genome in population:
            genome.species_id = 0
        self.species = {0: sp}
        self._next_id = max(self._next_id, 1)

    # -------------------------------------------------------------- threshold

    def set_threshold_for(self, representatives: list[Genome]) -> None:
        """Threshold just below the smallest pairwise representative distance (floor 0.05).

        With fewer than two representatives the threshold is unchanged.
        """
        if len(representatives) < 2:
            return
        smallest = min(distance(a, b, self.cfg) for a, b in combinations(representatives, 2))
        self.threshold = max(THRESHOLD_FLOOR, smallest - 1e-6)

    def adjust_threshold(self) -> None:
        """Scale the threshold by a relative ``threshold_step`` toward ``target_species``.

        Too few species means the threshold is too high (everyone groups together), so it
        is multiplied by ``1 - threshold_step``; too many species multiplies it by
        ``1 + threshold_step``. Floor 0.05.
        """
        if not self.cfg.enabled:
            return
        count = len(self.species)
        if count < self.cfg.target_species:
            self.threshold *= 1.0 - self.cfg.threshold_step
        elif count > self.cfg.target_species:
            self.threshold *= 1.0 + self.cfg.threshold_step
        self.threshold = max(THRESHOLD_FLOOR, self.threshold)

    # ---------------------------------------------------------------- fitness

    def _all_members(self) -> list[Genome]:
        return [g for sid in sorted(self.species) for g in self.species[sid].members]

    def share_fitness(self) -> None:
        """Explicit fitness sharing: ``adjusted_fitness = (adjusted + shift) / species size``.

        ``shift = -min(adjusted_fitness)`` over every member of every species when that
        minimum is negative, else 0, so no shared value is negative (dividing a negative
        value by the species size would otherwise favour large species). Raw ``fitness`` is
        never changed. Raises ``ValueError`` if a member has no ``adjusted_fitness``.
        """
        if not self.cfg.enabled:
            return
        members = self._all_members()
        for genome in members:
            if genome.adjusted_fitness is None:
                raise ValueError(f"genome {genome.id} has no adjusted_fitness to share")
        low = min((g.adjusted_fitness for g in members), default=0.0)  # type: ignore[type-var]
        shift = -low if low < 0 else 0.0  # type: ignore[operator]
        for sp in self.species.values():
            size = len(sp.members)
            for genome in sp.members:
                genome.adjusted_fitness = (genome.adjusted_fitness + shift) / size  # type: ignore[operator]

    def allocate_offspring(self, total: int) -> dict[int, int]:
        """Offspring per species, proportional to summed adjusted fitness; sums to ``total``."""
        if not self.cfg.enabled:
            return {0: total}
        if not self.species:
            return {}
        values: dict[int, list[float]] = {}
        for sid in sorted(self.species):
            vals = []
            for genome in self.species[sid].members:
                if genome.adjusted_fitness is None:
                    raise ValueError(f"genome {genome.id} has no adjusted_fitness")
                vals.append(genome.adjusted_fitness)
            values[sid] = vals
        low = min((v for vals in values.values() for v in vals), default=0.0)
        shift = -low if low < 0 else 0.0
        shares = {sid: sum(v + shift for v in vals) for sid, vals in values.items()}
        if sum(shares.values()) <= 0:
            shares = {sid: float(len(vals)) for sid, vals in values.items()}
        weight = sum(shares.values())
        if weight <= 0:  # only possible with all species empty
            shares = {sid: 1.0 for sid in values}
            weight = float(len(shares))
        quotas = {sid: total * share / weight for sid, share in shares.items()}
        alloc = {sid: int(math.floor(q)) for sid, q in quotas.items()}
        remaining = total - sum(alloc.values())
        by_remainder = sorted(quotas, key=lambda sid: (-(quotas[sid] - alloc[sid]), sid))
        for sid in by_remainder[:remaining]:
            alloc[sid] += 1
        return alloc

    # ------------------------------------------------------------- stagnation

    def best_genome(self) -> Genome | None:
        """Member with the highest raw fitness across all species (None if none scored)."""
        best: Genome | None = None
        for genome in self._all_members():
            if genome.fitness is None:
                continue
            if best is None or genome.fitness > best.fitness:  # type: ignore[operator]
                best = genome
        return best

    def prune_stagnant(self, generation: int) -> None:
        """Update improvement records and drop stagnant species (never the best one)."""
        if not self.cfg.enabled:
            return
        best = self.best_genome()
        if best is None:  # nothing scored yet: no evidence of stagnation
            return
        for sp in self.species.values():
            scored = [g.fitness for g in sp.members if g.fitness is not None]
            if scored and max(scored) > sp.best_fitness:
                sp.best_fitness = max(scored)
                sp.last_improved = generation
        protected = best.species_id
        for sid, sp in self.species.items():
            if any(g is best for g in sp.members):
                protected = sid
                break
        self.species = {
            sid: sp
            for sid, sp in self.species.items()
            if sid == protected
            or generation - sp.last_improved <= self.cfg.stagnation_generations
        }


__all__ = [
    "Species",
    "Speciator",
    "distance",
    "excess_disjoint",
    "property_difference",
]
