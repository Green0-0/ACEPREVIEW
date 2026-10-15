"""Tests for ace.neat.speciation."""
from __future__ import annotations

import random

import pytest

from ace.graph.types import Kind, WorkerType
from ace.neat.config import SpeciationConfig
from ace.neat.genome import Genome, minimal_genome
from ace.neat.innovation import InnovationDB
from ace.neat.speciation import Species, Speciator, distance, excess_disjoint

PROPS = {"a": 0, "b": 0}


def with_viewers(db: InnovationDB, count: int, *, start: int = 0) -> Genome:
    """Minimal editor genome plus ``count`` VIEWER sources (2 genes each) feeding it."""
    g = minimal_genome(db, PROPS)
    for i in range(start, start + count):
        v = g.add_node(db, ("source", 0, Kind.VIEW, i, WorkerType.VIEWER), {})
        g.add_connection(db, v.innovation, 0, Kind.VIEW)
    g.validate()
    return g


def scored(g: Genome, fitness: float, adjusted: float | None = None) -> Genome:
    g.fitness = fitness
    g.adjusted_fitness = fitness if adjusted is None else adjusted
    return g


# ------------------------------------------------------------------- distance


def test_distance_zero_for_copy() -> None:
    db = InnovationDB()
    g = with_viewers(db, 3)
    assert distance(g, g.copy(), SpeciationConfig()) == 0.0


def test_distance_grows_with_excess_genes() -> None:
    db = InnovationDB()
    base = with_viewers(db, 0)
    one = with_viewers(db, 1)
    two = with_viewers(db, 2)
    cfg = SpeciationConfig()
    assert excess_disjoint(base, one) == (2, 0)
    assert distance(base, one, cfg) == pytest.approx(2.0)
    assert distance(base, two, cfg) == pytest.approx(4.0)
    assert distance(one, base, cfg) == distance(base, one, cfg)


def test_disjoint_and_excess_split() -> None:
    db = InnovationDB()
    a = with_viewers(db, 1, start=0)   # genes 0,1,2,3
    b = with_viewers(db, 1, start=1)   # genes 0,1,4,5
    assert excess_disjoint(a, b) == (2, 2)
    cfg = SpeciationConfig(c_excess=1.0, c_disjoint=2.0)
    assert distance(a, b, cfg) == pytest.approx(2 * 1.0 + 2 * 2.0)


def test_property_term() -> None:
    db = InnovationDB()
    a = with_viewers(db, 1)
    b = a.copy()
    b.nodes[0].properties["a"] = 1  # editor: 1 of 2 names differ; viewer has no props
    cfg = SpeciationConfig(c_properties=0.5)
    # H = mean over matched nodes (editor 0.5, viewer 0.0) = 0.25
    assert distance(a, b, cfg) == pytest.approx(0.5 * 0.25)
    b.nodes[0].properties["b"] = 2
    assert distance(a, b, cfg) == pytest.approx(0.5 * 0.5)


def test_distance_is_never_normalised_by_gene_count() -> None:
    db = InnovationDB()
    base = with_viewers(db, 0)          # 2 genes
    eighteen = with_viewers(db, 8)      # 18 genes
    twenty = with_viewers(db, 9)        # 20 genes: no scale change at this size
    cfg = SpeciationConfig()
    assert distance(base, eighteen, cfg) == pytest.approx(16.0)
    assert distance(base, twenty, cfg) == pytest.approx(18.0)


# ----------------------------------------------------------------- speciation


def test_speciate_groups_copies_and_separates_distant() -> None:
    db = InnovationDB()
    near = with_viewers(db, 0)
    far = with_viewers(db, 5)  # distance 10 > threshold 3
    population = [near, far, near.copy(), far.copy(), near.copy()]
    sp = Speciator(SpeciationConfig(), rng=random.Random(1))
    sp.speciate(population, 0)
    assert len(sp.species) == 2
    groups = {sid: {g.id for g in s.members} for sid, s in sp.species.items()}
    assert {population[0].id, population[2].id, population[4].id} in groups.values()
    assert {population[1].id, population[3].id} in groups.values()
    for g in population:
        assert g in sp.species[g.species_id].members
    # Next generation: species ids persist, representatives come from previous members.
    old_members = {sid: list(s.members) for sid, s in sp.species.items()}
    sp.speciate([g.copy() for g in population], 1)
    assert set(sp.species) == set(old_members)
    for sid, s in sp.species.items():
        assert any(s.representative is m for m in old_members[sid])


def test_speciate_drops_empty_species() -> None:
    db = InnovationDB()
    near, far = with_viewers(db, 0), with_viewers(db, 5)
    sp = Speciator(SpeciationConfig(), rng=random.Random(0))
    sp.speciate([near, far], 0)
    far_id = far.species_id
    sp.speciate([near.copy()], 1)
    assert far_id not in sp.species
    assert len(sp.species) == 1


def test_set_threshold_for_separates_representatives() -> None:
    db = InnovationDB()
    reps = [with_viewers(db, 0), with_viewers(db, 1), with_viewers(db, 2)]
    sp = Speciator(SpeciationConfig(initial_threshold=3.0), rng=random.Random(0))
    sp.set_threshold_for(reps)
    assert sp.threshold == pytest.approx(2.0 - 1e-6)
    sp.speciate(reps, 0)
    assert len(sp.species) == 3
    assert len({g.species_id for g in reps}) == 3
    # One representative keeps the threshold; identical ones hit the floor.
    sp.set_threshold_for(reps[:1])
    assert sp.threshold == pytest.approx(2.0 - 1e-6)
    sp.set_threshold_for([reps[0], reps[0].copy()])
    assert sp.threshold == 0.05


def test_adjust_threshold_moves_toward_target() -> None:
    db = InnovationDB()
    cfg = SpeciationConfig(target_species=2, threshold_step=0.5, initial_threshold=3.0)
    sp = Speciator(cfg, rng=random.Random(0))
    sp.speciate([with_viewers(db, 0)], 0)  # 1 species < 2: lower (relative step)
    sp.adjust_threshold()
    assert sp.threshold == pytest.approx(3.0 * 0.5)
    sp.speciate([with_viewers(db, 0), with_viewers(db, 3), with_viewers(db, 6)], 1)
    assert len(sp.species) == 3  # > 2: raise
    sp.adjust_threshold()
    assert sp.threshold == pytest.approx(1.5 * 1.5)
    sp.speciate([with_viewers(db, 0), with_viewers(db, 3)], 2)
    assert len(sp.species) == 2  # == 2: unchanged
    sp.adjust_threshold()
    assert sp.threshold == pytest.approx(2.25)
    sp.threshold = 0.08
    sp.speciate([with_viewers(db, 0)], 3)
    sp.adjust_threshold()  # 0.04 -> floor
    assert sp.threshold == 0.05


def test_adjust_threshold_step_is_relative() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig(target_species=2, threshold_step=0.15, initial_threshold=10.0))
    sp.speciate([with_viewers(db, 0)], 0)
    sp.adjust_threshold()
    assert sp.threshold == pytest.approx(8.5)  # an absolute step would give 9.85


def test_threshold_reaches_target_species_quickly() -> None:
    # Five structures pairwise >= 4 apart; a threshold of 15 groups them into 2 species.
    # The default relative step must reach K = 5 within a dozen generations (an absolute
    # 0.1 step would need over 100).
    db = InnovationDB()
    structures = [with_viewers(db, n) for n in (0, 2, 4, 6, 8)]
    cfg = SpeciationConfig(target_species=5, initial_threshold=15.0)
    assert cfg.threshold_step == 0.15
    sp = Speciator(cfg, rng=random.Random(0))
    counts = []
    for generation in range(12):
        sp.speciate([g.copy() for g in structures for _ in range(3)], generation)
        counts.append(len(sp.species))
        sp.adjust_threshold()
    assert counts[0] < 5
    reached = counts.index(5)
    assert all(c == 5 for c in counts[reached:])  # stays at K once reached


# -------------------------------------------------------------------- fitness


def _manual(sp: Speciator, groups: dict[int, list[Genome]]) -> None:
    sp.species = {
        sid: Species(id=sid, representative=members[0], members=members, created=0,
                     best_fitness=float("-inf"), last_improved=0)
        for sid, members in groups.items()
    }
    for sid, members in groups.items():
        for g in members:
            g.species_id = sid


def test_share_fitness_divides_by_species_size() -> None:
    db = InnovationDB()
    a, b, c = (scored(with_viewers(db, 0), f) for f in (4.0, 6.0, 5.0))
    sp = Speciator(SpeciationConfig())
    _manual(sp, {0: [a, b], 1: [c]})
    sp.share_fitness()
    assert (a.adjusted_fitness, b.adjusted_fitness, c.adjusted_fitness) == (2.0, 3.0, 5.0)
    c.adjusted_fitness = None
    with pytest.raises(ValueError):
        sp.share_fitness()


def test_share_fitness_shifts_negative_values() -> None:
    db = InnovationDB()
    a = scored(with_viewers(db, 0), 3.0, adjusted=-2.0)
    b = scored(with_viewers(db, 0), 4.0, adjusted=0.0)
    c = scored(with_viewers(db, 0), 5.0, adjusted=-1.0)
    sp = Speciator(SpeciationConfig())
    _manual(sp, {0: [a, b], 1: [c]})
    sp.share_fitness()
    # population minimum -2 -> shift +2, then divide by species size
    assert (a.adjusted_fitness, b.adjusted_fitness, c.adjusted_fitness) == (0.0, 1.0, 1.0)
    assert (a.fitness, b.fitness, c.fitness) == (3.0, 4.0, 5.0)  # raw never changes
    assert min(g.adjusted_fitness for g in (a, b, c)) == 0.0
    alloc = sp.allocate_offspring(7)
    assert sum(alloc.values()) == 7
    assert alloc == {0: 4, 1: 3}  # equal shares 1 and 1; tie on remainder -> lower id


def test_share_fitness_all_equal_negative_allocates_by_member_count() -> None:
    db = InnovationDB()
    gs = [scored(with_viewers(db, 0), 1.0, adjusted=-1.0) for _ in range(3)]
    sp = Speciator(SpeciationConfig())
    _manual(sp, {0: gs[:2], 1: gs[2:]})
    sp.share_fitness()
    assert [g.adjusted_fitness for g in gs] == [0.0, 0.0, 0.0]
    assert sp.allocate_offspring(6) == {0: 4, 1: 2}


def test_allocate_offspring_sums_to_total() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig())
    gs = [scored(with_viewers(db, 0), f) for f in (1.0, 1.0, 2.0, 0.5, 3.0)]
    _manual(sp, {0: gs[:2], 1: gs[2:3], 2: gs[3:]})
    for total in (0, 1, 5, 7, 20):
        alloc = sp.allocate_offspring(total)
        assert set(alloc) == {0, 1, 2}
        assert sum(alloc.values()) == total
        assert all(v >= 0 for v in alloc.values())
    # shares 2, 2, 3.5 of 7.5 -> 10 offspring: 2.67, 2.67, 4.67 -> 3, 2, 5 (ties by id)
    assert sp.allocate_offspring(10) == {0: 3, 1: 2, 2: 5}


def test_allocate_offspring_handles_negative_fitness() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig())
    a, b = scored(with_viewers(db, 0), -1.0), scored(with_viewers(db, 0), -1.0)
    c = scored(with_viewers(db, 0), 1.0)
    _manual(sp, {0: [a, b], 1: [c]})
    # shift by +1: species 0 -> 0, species 1 -> 2
    assert sp.allocate_offspring(6) == {0: 0, 1: 6}
    # all equal negative -> zero shares -> proportional to member counts
    c.adjusted_fitness = -1.0
    alloc = sp.allocate_offspring(6)
    assert alloc == {0: 4, 1: 2}


def test_prune_stagnant_keeps_best_species() -> None:
    db = InnovationDB()
    cfg = SpeciationConfig(stagnation_generations=2)
    sp = Speciator(cfg)
    weak = scored(with_viewers(db, 0), 1.0)
    strong = scored(with_viewers(db, 0), 5.0)
    _manual(sp, {0: [weak], 1: [strong]})
    sp.prune_stagnant(0)
    assert sp.species[0].best_fitness == 1.0 and sp.species[1].best_fitness == 5.0
    assert set(sp.species) == {0, 1}
    sp.prune_stagnant(2)  # 2 - 0 == 2: not yet stagnant
    assert set(sp.species) == {0, 1}
    sp.prune_stagnant(3)  # both stagnant; species 1 holds the best genome
    assert set(sp.species) == {1}
    assert sp.best_genome() is strong


def test_prune_stagnant_noop_without_any_fitness() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig(stagnation_generations=2))
    a, b = with_viewers(db, 0), with_viewers(db, 0)
    _manual(sp, {0: [a], 1: [b]})
    sp.prune_stagnant(100)  # nothing scored: no evidence of stagnation
    assert set(sp.species) == {0, 1}
    assert sp.species[0].best_fitness == float("-inf")
    assert sp.species[0].last_improved == 0
    scored(a, 1.0)
    sp.prune_stagnant(100)  # species 0 improves now; species 1 is still unscored
    assert sp.species[0].last_improved == 100
    assert set(sp.species) == {0}


def test_prune_stagnant_requires_strict_improvement() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig(stagnation_generations=50))
    g = scored(with_viewers(db, 0), 1.0)
    other = scored(with_viewers(db, 0), 9.0)
    _manual(sp, {0: [g], 1: [other]})
    sp.prune_stagnant(0)
    assert (sp.species[0].best_fitness, sp.species[0].last_improved) == (1.0, 0)
    sp.prune_stagnant(3)  # equal best: no reset
    assert (sp.species[0].best_fitness, sp.species[0].last_improved) == (1.0, 0)
    g.fitness = 0.5
    sp.prune_stagnant(4)  # worse: no reset
    assert (sp.species[0].best_fitness, sp.species[0].last_improved) == (1.0, 0)
    g.fitness = 1.5
    sp.prune_stagnant(6)
    assert (sp.species[0].best_fitness, sp.species[0].last_improved) == (1.5, 6)


def test_speciate_sets_created_at_creation() -> None:
    db = InnovationDB()
    near, far = with_viewers(db, 0), with_viewers(db, 5)
    sp = Speciator(SpeciationConfig(), rng=random.Random(0))
    sp.speciate([near], 7)
    (first,) = sp.species.values()
    assert (first.created, first.last_improved, first.best_fitness) == (7, 7, float("-inf"))
    sp.speciate([near.copy(), far], 9)
    assert sp.species[first.id].created == 7  # existing species keep their creation time
    new = sp.species[far.species_id]
    assert new is not first and (new.created, new.last_improved) == (9, 9)


def test_prune_stagnant_resets_on_improvement() -> None:
    db = InnovationDB()
    sp = Speciator(SpeciationConfig(stagnation_generations=1))
    g = scored(with_viewers(db, 0), 1.0)
    best = scored(with_viewers(db, 0), 9.0)
    _manual(sp, {0: [g], 1: [best]})
    sp.prune_stagnant(0)
    g.fitness = 2.0
    sp.prune_stagnant(5)
    assert sp.species[0].last_improved == 5
    assert set(sp.species) == {0, 1}


# ------------------------------------------------------------------- disabled


def test_disabled_speciation_single_species() -> None:
    db = InnovationDB()
    cfg = SpeciationConfig(enabled=False, stagnation_generations=1)
    sp = Speciator(cfg)
    population = [scored(with_viewers(db, n), float(n)) for n in (0, 5, 9)]
    sp.speciate(population, 0)
    assert list(sp.species) == [0]
    assert all(g.species_id == 0 for g in population)
    assert len(sp.species[0].members) == 3
    before = [g.adjusted_fitness for g in population]
    sp.share_fitness()
    assert [g.adjusted_fitness for g in population] == before
    assert sp.allocate_offspring(7) == {0: 7}
    threshold = sp.threshold
    sp.adjust_threshold()
    assert sp.threshold == threshold
    sp.prune_stagnant(100)
    assert list(sp.species) == [0]
    assert sp.best_genome() is population[2]
