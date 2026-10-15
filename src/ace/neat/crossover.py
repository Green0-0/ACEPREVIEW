"""NEAT crossover of two genomes (GRAPH_NEAT_DESIGN.md section 10.3).

:func:`crossover` aligns the parents' node genes and connection genes separately by
innovation number (:func:`align`), builds a child from the aligned genes, validates it and,
if it is invalid, repairs it (:func:`repair`). If the repair does not yield a valid genome,
a copy of the fitter parent is returned instead (operator ``"crossover_fallback"``).

Contract readings (documented per the contract's ambiguity rule):

* **Fitness comparison** (:func:`compare_fitness`): ``adjusted_fitness`` is compared when it
  is set on both parents, else raw ``fitness`` when it is set on both, else the parents are
  treated as equally fit. When they are equally fit, disjoint and excess genes come from
  both parents, and ``a`` plays the role of "the fitter parent" wherever one parent must
  be preferred (END source choice in the repair, the fallback copy).
* **Matched node genes** get a new :class:`NodeGene` with the shared innovation and type;
  each property name is drawn independently (``rng.choice``) from one of the two parents,
  and a name present in only one parent is taken from that parent. Property names are
  visited in sorted order so a seeded ``rng`` gives reproducible children.
* **Matched connection genes**: if the gene is disabled in either parent, the child's gene
  is disabled with probability 0.75 (enabled with 0.25); otherwise the flag is taken from
  a random parent (which is then necessarily ``True``). The ``rng`` is consumed in
  innovation order.
* **Repair** (:func:`repair`) runs the four steps below in order, re-validating after each
  and stopping as soon as the genome is valid:

  1. drop connection genes (enabled or disabled) whose ``src`` or ``dst`` is missing;
  2. fix END: with no enabled input, enable the single disabled EDIT input, or among
     several the one coming from the fitter parent's END source if present, else the
     lowest innovation; with several enabled inputs, keep the one from the fitter
     parent's END source (else the lowest innovation) and disable the rest;
  3. remove nodes that cannot reach END through enabled edges (with all their connection
     genes), iterated to a fixed point;
  4. remove nodes whose required input slots (``SlotSpec.min``) are not met by enabled
     edges, iterated to a fixed point. Because removing such a node can strand its
     upstream nodes, each round of step 4 also re-applies step 3; the fixed point is over
     both removals.

  Repairs never touch maximum arities or cycles; such children fall back.
* **Fallback** returns ``fitter.copy()`` with a fresh id, ``generation`` and ``parents``
  set as for a normal child, ``operators == ["crossover_fallback"]`` (a fresh list, not
  appended to the fitter's history, so the list names exactly the operators that produced
  this genome) and fitness / adjusted fitness / species reset to ``None``.
* **max_workers** (optional keyword, default None = unbounded): when equally fit parents
  contribute disjoint genes from both sides the child can hold more workers than either
  parent. A child whose repaired worker count exceeds ``max_workers`` is treated like an
  unrepairable child and falls back. Callers enforcing ``NeatConfig.max_workers`` must
  pass it.
* ``innovations`` is accepted for signature compatibility with the rest of the NEAT
  operators; crossover never allocates new innovations because every child gene already
  exists in a parent.
"""
from __future__ import annotations

import random
import uuid
from collections import defaultdict

from ace.graph.types import END, WORKER_SPECS, Kind
from ace.neat.genome import ConnectionGene, Genome, NodeGene
from ace.neat.innovation import InnovationDB

P_DISABLE_IF_EITHER_DISABLED = 0.75
"""Probability that a matched connection disabled in either parent is disabled in the child."""

Gene = NodeGene | ConnectionGene

__all__ = [
    "P_DISABLE_IF_EITHER_DISABLED",
    "align",
    "compare_fitness",
    "crossover",
    "repair",
]


# ---------------------------------------------------------------------- helpers


def compare_fitness(a: Genome, b: Genome) -> int:
    """Return 1 if ``a`` is fitter, -1 if ``b`` is fitter, 0 if equal or not comparable.

    Uses ``adjusted_fitness`` when both are set, else ``fitness`` when both are set.
    """
    if a.adjusted_fitness is not None and b.adjusted_fitness is not None:
        fa, fb = a.adjusted_fitness, b.adjusted_fitness
    elif a.fitness is not None and b.fitness is not None:
        fa, fb = a.fitness, b.fitness
    else:
        return 0
    if fa > fb:
        return 1
    if fb > fa:
        return -1
    return 0


def align(
    a: Genome, b: Genome
) -> tuple[
    list[tuple[NodeGene, NodeGene]],
    list[tuple[ConnectionGene, ConnectionGene]],
    list[Gene],
    list[Gene],
]:
    """Align two genomes by innovation number.

    Returns ``(matched_nodes, matched_conns, disjoint_excess_a, disjoint_excess_b)``:
    ``matched_nodes`` / ``matched_conns`` are ``(gene_in_a, gene_in_b)`` pairs for
    innovations present in both genomes; ``disjoint_excess_a`` holds the node genes then the
    connection genes present only in ``a`` (likewise for ``b``). Every list is sorted by
    innovation (node genes before connection genes in the disjoint lists).
    """
    matched_nodes = [(a.nodes[i], b.nodes[i]) for i in sorted(a.nodes.keys() & b.nodes.keys())]
    matched_conns = [
        (a.connections[i], b.connections[i])
        for i in sorted(a.connections.keys() & b.connections.keys())
    ]
    only_a: list[Gene] = [a.nodes[i] for i in sorted(a.nodes.keys() - b.nodes.keys())]
    only_a += [a.connections[i] for i in sorted(a.connections.keys() - b.connections.keys())]
    only_b: list[Gene] = [b.nodes[i] for i in sorted(b.nodes.keys() - a.nodes.keys())]
    only_b += [b.connections[i] for i in sorted(b.connections.keys() - a.connections.keys())]
    return matched_nodes, matched_conns, only_a, only_b


def _cross_node(ga: NodeGene, gb: NodeGene, rng: random.Random) -> NodeGene:
    """A new node gene whose properties are picked name by name from one parent."""
    props: dict[str, int] = {}
    for name in sorted(ga.properties.keys() | gb.properties.keys()):
        if name in ga.properties and name in gb.properties:
            props[name] = rng.choice((ga.properties[name], gb.properties[name]))
        elif name in ga.properties:
            props[name] = ga.properties[name]
        else:
            props[name] = gb.properties[name]
    return NodeGene(innovation=ga.innovation, type=ga.type, properties=props)


def _cross_connection(ca: ConnectionGene, cb: ConnectionGene, rng: random.Random) -> ConnectionGene:
    """A new connection gene with the NEAT enabled-flag rule."""
    if not ca.enabled or not cb.enabled:
        enabled = rng.random() >= P_DISABLE_IF_EITHER_DISABLED
    else:
        enabled = rng.choice((ca, cb)).enabled
    return ConnectionGene(ca.innovation, ca.src, ca.dst, ca.kind, enabled)


def _copy_gene(gene: Gene) -> Gene:
    if isinstance(gene, NodeGene):
        return NodeGene(gene.innovation, gene.type, dict(gene.properties))
    return ConnectionGene(gene.innovation, gene.src, gene.dst, gene.kind, gene.enabled)


def _end_source(genome: Genome) -> int | None:
    """The source of ``genome``'s single enabled END connection, or None."""
    into_end = genome.connections_into(END, enabled_only=True)
    return into_end[0].src if len(into_end) == 1 else None


# ----------------------------------------------------------------------- repair


def _remove_dangling_connections(genome: Genome) -> bool:
    """Step 1: drop connection genes with a missing endpoint. Returns whether any changed."""
    dangling = [
        inn
        for inn, c in genome.connections.items()
        if c.src not in genome.nodes or (c.dst != END and c.dst not in genome.nodes)
    ]
    for inn in dangling:
        del genome.connections[inn]
    return bool(dangling)


def _fix_end(genome: Genome, fitter: Genome) -> bool:
    """Step 2: make END have exactly one enabled EDIT input. Returns whether any changed."""
    preferred = _end_source(fitter)
    enabled = [c for c in genome.connections_into(END, enabled_only=True) if c.kind is Kind.EDIT]

    def pick(candidates: list[ConnectionGene]) -> ConnectionGene:
        for c in candidates:
            if c.src == preferred:
                return c
        return candidates[0]  # sorted by innovation

    if not enabled:
        disabled = [
            c for c in genome.connections_into(END) if not c.enabled and c.kind is Kind.EDIT
        ]
        if not disabled:
            return False
        pick(disabled).enabled = True
        return True
    if len(enabled) > 1:
        keep = pick(enabled)
        for c in enabled:
            if c is not keep:
                c.enabled = False
        return True
    return False


def _remove_nodes(genome: Genome, doomed: set[int]) -> None:
    for inn in doomed:
        del genome.nodes[inn]
    for inn in [i for i, c in genome.connections.items() if c.src in doomed or c.dst in doomed]:
        del genome.connections[inn]


def _remove_unreachable(genome: Genome) -> bool:
    """Step 3: remove nodes with no enabled path to END (fixed point). Returns whether any changed."""
    changed = False
    while True:
        parents: dict[int | str, list[int]] = defaultdict(list)
        for c in genome.connections.values():
            if c.enabled:
                parents[c.dst].append(c.src)
        seen: set[int | str] = {END}
        stack: list[int | str] = [END]
        while stack:
            for p in parents.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        doomed = {inn for inn in genome.nodes if inn not in seen}
        if not doomed:
            return changed
        _remove_nodes(genome, doomed)
        changed = True


def _remove_unsatisfied(genome: Genome) -> bool:
    """Step 4: remove nodes missing required inputs (fixed point, with step 3 re-applied)."""
    changed = False
    while True:
        counts: dict[int, dict[Kind, int]] = defaultdict(lambda: defaultdict(int))
        for c in genome.connections.values():
            if c.enabled and c.dst != END:
                counts[c.dst][c.kind] += 1  # type: ignore[index]
        doomed = {
            inn
            for inn, node in genome.nodes.items()
            if any(
                counts[inn][kind] < slot.min
                for kind, slot in WORKER_SPECS[node.type].inputs.items()
            )
        }
        if not doomed:
            return changed
        _remove_nodes(genome, doomed)
        _remove_unreachable(genome)
        changed = True


def repair(genome: Genome, fitter: Genome) -> bool:
    """Repair ``genome`` in place (see module docstring); return whether it is now valid.

    ``fitter`` is the parent whose END source is preferred when END's inputs are fixed.
    Validity is re-checked after every step and the repair stops at the first valid state.
    """
    if genome.is_valid():
        return True
    for step in (
        lambda: _remove_dangling_connections(genome),
        lambda: _fix_end(genome, fitter),
        lambda: _remove_unreachable(genome),
        lambda: _remove_unsatisfied(genome),
    ):
        if step() and genome.is_valid():
            return True
    return genome.is_valid()


# -------------------------------------------------------------------- crossover


def crossover(
    a: Genome,
    b: Genome,
    rng: random.Random,
    *,
    innovations: InnovationDB,
    max_workers: int | None = None,
) -> Genome:
    """Return a valid child of ``a`` and ``b`` by NEAT crossover (section 10.3).

    Raises ``InvalidGraph`` if the parents disagree on the type of a shared innovation.
    The parents are not modified. ``innovations`` is unused (see module docstring).
    """
    del innovations  # crossover never allocates innovations
    a.assert_types_match(b)

    cmp = compare_fitness(a, b)
    fitter = b if cmp < 0 else a
    matched_nodes, matched_conns, only_a, only_b = align(a, b)

    child = Genome(
        id=uuid.uuid4().hex,
        nodes={},
        connections={},
        generation=max(a.generation, b.generation) + 1,
        parents=(a.id, b.id),
        operators=["crossover"],
    )
    genes: list[Gene] = []
    for ga, gb in matched_nodes:
        genes.append(_cross_node(ga, gb, rng))
    for ca, cb in matched_conns:
        genes.append(_cross_connection(ca, cb, rng))
    if cmp >= 0:
        genes.extend(_copy_gene(g) for g in only_a)
    if cmp <= 0:
        genes.extend(_copy_gene(g) for g in only_b)
    for gene in sorted(genes, key=lambda g: g.innovation):
        if isinstance(gene, NodeGene):
            child.nodes[gene.innovation] = gene
        else:
            child.connections[gene.innovation] = gene

    if repair(child, fitter) and (max_workers is None or child.num_workers() <= max_workers):
        return child

    fallback = fitter.copy()
    fallback.generation = child.generation
    fallback.parents = (a.id, b.id)
    fallback.operators = ["crossover_fallback"]
    fallback.fitness = None
    fallback.adjusted_fitness = None
    fallback.species_id = None
    return fallback
