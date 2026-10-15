"""Mutation operators and the typed insertion table (GRAPH_NEAT_DESIGN.md section 10.2).

Every operator takes ``(genome, ctx)``, mutates ``genome`` **in place**, returns ``True`` on
success and leaves the genome unchanged (and valid) on failure. Internally each operator
runs up to ``ctx.config.max_attempts`` proposals; a proposal works on a deep copy of the
genome and, if it validates (``Genome.is_valid()``), its genes are swapped into the
original. :func:`mutate` copies the genome first, resets fitness bookkeeping, applies the
operators with their configured probabilities and never returns an invalid genome.

Innovation keys (section 7.4): new nodes are allocated from ``ctx.innovations`` with

* LISTVIEWER split of a VIEW edge: ``("split", conn_innovation, LISTVIEWER)``
* CRITIC split of an EDIT edge: ``("critic", conn_innovation, CRITIC)``
* VIEWER source feeding ``d``: ``("source", d, Kind.VIEW, VIEWER)``; the same key is used
  for the fallback VIEWER created to give a new CRITIC or merger a VIEW input (``d`` is then
  the critic / merger innovation).
* merger: ``("merge", end_source_innovation, candidate_innovation, MERGER_TYPE)``

Contract readings / additions (documented per the contract's ambiguity rule):

* **Keys for the VIEWEDITOR and EDITOR rows.** Section 7.4 has no key for these two
  add_worker rows, so two extra families are used: ``("mirror", d, VIEWEDITOR)`` for the
  VIEWEDITOR fed by the same VIEW sources as ``d``, and ``("branch", v, EDITOR)`` for the
  EDITOR fed by ``v``. Both include the worker type, so types are never mixed.
* **Key collisions.** If a key yields an innovation already present in the genome (the
  same structural event happened before, e.g. a second VIEWER source for the same node),
  that proposal is rejected and another one is tried.
* **Init keys.** :func:`add_worker` and :func:`insert_merger` accept ``key_prefix``. When
  given, every node they create is keyed ``(key_prefix, len(genome.nodes), type)``
  instead (``len(nodes)`` at creation time is the running index of the seed-build
  sequence, because initialisation only ever adds nodes). :func:`random_valid_genome`
  passes ``"init"``.
* **Row selection.** ``add_worker`` picks a row uniformly per attempt, then a random site;
  a row with no site rejects that attempt. From the minimal genome only the CRITIC, VIEWER
  and EDITOR rows have sites (there are no VIEW edges yet).
* **VIEWER source REPORT edge** is added with probability 0.5 when ``d`` has a REPORT slot.
* **LISTVIEWER split** wires VIEW only (no REPORT), exactly as in the table.
* **VIEWEDITOR row** wires ``n`` into a random existing merger that is not an ancestor of
  ``n`` (all merger EDIT slots are unbounded) if one exists, otherwise calls the merger
  insertion with ``n`` as the candidate.
* **EDITOR row** sites are VIEW-producing nodes with no enabled outgoing edge to an EDITOR.
  The same row (and the VIEWEDITOR row) creates the candidate in ``insert_merger`` when no
  existing candidate is found; existing candidates are always preferred.
* **add_connection** never targets END (END accepts exactly one edge). A disabled gene for
  the chosen ``(src, dst, kind)`` is re-enabled instead of adding a new gene.
* **remove_connection** deletes a random *enabled* connection gene (disabled genes are NEAT
  history; ``toggle_enable`` and ``remove_worker`` handle those).
* **toggle_enable** flips a random connection gene, enabled or disabled.
* **remove_worker** never removes the END source (rejected outright; removing a candidate
  of an END-source merger collapses that merger, which covers the contract's exception).
  When a merger is left with one enabled EDIT input it is collapsed: the remaining
  candidate inherits every enabled outgoing edge of the merger whose kind it outputs (EDIT,
  and also VIEW/REPORT, which keeps more collapses valid), and the merger is removed.
* **mutate** appends the names of applied operators to the copy's existing ``operators``
  list (so ``["crossover"]`` from a crossover child is kept), resets ``fitness``,
  ``adjusted_fitness`` and ``species_id`` to None, gives the child a new id and does not
  touch ``generation`` or ``parents`` (the caller sets those). If the result is somehow
  invalid it returns a fresh copy of the input with ``"noop"`` appended; if the input itself
  is invalid it raises ``InvalidGraph``. Property validity is not re-checked (operators only
  ever write in-range indices).
* **random_valid_genome** accepts either a ``MutationConfig`` or a ``NeatConfig`` (its
  ``.mutation`` is used). It draws a target worker count uniformly from
  ``[min_workers, max_workers]`` and applies ``add_worker`` (75%) / ``insert_merger`` (25%)
  with ``max_workers=target`` until the target is reached.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ace.graph.properties import PROPERTIES, random_properties
from ace.graph.types import END, MERGER_TYPES, WORKER_SPECS, InvalidGraph, Kind, WorkerType
from ace.neat.config import MutationConfig
from ace.neat.genome import Genome, NodeGene, minimal_genome
from ace.neat.innovation import InnovationDB

INIT_KEY_PREFIX = "init"
"""Key prefix used for nodes created during population initialisation."""

ADD_WORKER_ROWS: tuple[str, ...] = ("listviewer", "critic", "viewer", "vieweditor", "editor")
"""Names of the typed insertion table rows, in table order."""

_MERGER_ORDER: tuple[WorkerType, ...] = tuple(sorted(MERGER_TYPES, key=lambda t: t.value))
_REPORT_PROBABILITY = 0.5


@dataclass
class MutationContext:
    """Everything an operator needs: randomness, the global innovation DB, settings."""

    rng: random.Random
    innovations: InnovationDB
    config: MutationConfig
    max_workers: int


class _Reject(Exception):
    """A proposal is not applicable (no site, key collision, size limit, ...)."""


# --------------------------------------------------------------------------- helpers


def _outputs(genome: Genome, nid: int) -> frozenset[Kind]:
    return WORKER_SPECS[genome.nodes[nid].type].outputs


def _has_slot(genome: Genome, nid: int, kind: Kind) -> bool:
    return WORKER_SPECS[genome.nodes[nid].type].slot(kind) is not None


def _ancestors(genome: Genome, nid: int | str) -> set[int]:
    """Nodes with a path of enabled edges (any kind) to ``nid`` (``nid`` excluded)."""
    parents: dict[int | str, list[int]] = defaultdict(list)
    for c in genome.enabled_connections():
        parents[c.dst].append(c.src)
    seen: set[int] = set()
    stack = list(parents.get(nid, ()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parents.get(cur, ()))
    seen.discard(nid)  # type: ignore[arg-type]
    return seen


def _edit_reaches_end(genome: Genome, nid: int) -> bool:
    """Whether ``nid``'s EDIT output reaches END through enabled EDIT edges."""
    children: dict[int | str, list[int | str]] = defaultdict(list)
    for c in genome.enabled_connections():
        if c.kind is Kind.EDIT:
            children[c.src].append(c.dst)
    seen: set[int | str] = {nid}
    stack: list[int | str] = [nid]
    while stack:
        cur = stack.pop()
        for nxt in children.get(cur, ()):
            if nxt == END:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _end_source(genome: Genome) -> int:
    srcs = [c.src for c in genome.connections_into(END, enabled_only=True)]
    if len(srcs) != 1:
        raise _Reject
    return srcs[0]


def _view_sources(genome: Genome, nid: int) -> list[int]:
    return [c.src for c in genome.connections_into(nid, enabled_only=True) if c.kind is Kind.VIEW]


def _new_node(
    genome: Genome, ctx: MutationContext, key: tuple, key_prefix: str | None
) -> int:
    """Create a node gene of type ``key[-1]`` with random properties; return its innovation."""
    node_type = WorkerType(key[-1])
    if genome.num_workers() + 1 > ctx.max_workers:
        raise _Reject
    if key_prefix is not None:
        key = (key_prefix, len(genome.nodes), node_type)
    inn = ctx.innovations.node(key)
    if inn in genome.nodes or inn in genome.connections:
        raise _Reject
    genome.nodes[inn] = NodeGene(inn, node_type, random_properties(node_type, ctx.rng))
    return inn


def _connect(genome: Genome, ctx: MutationContext, src: int, dst: int | str, kind: Kind) -> None:
    genome.add_connection(ctx.innovations, src, dst, kind, enabled=True)


def _delete_node(genome: Genome, nid: int) -> None:
    del genome.nodes[nid]
    for inn in [i for i, c in genome.connections.items() if c.src == nid or c.dst == nid]:
        del genome.connections[inn]


def _feed_view(
    genome: Genome,
    ctx: MutationContext,
    target: int,
    members: list[int],
    ancestors: set[int],
    key_prefix: str | None,
) -> None:
    """Give ``target`` VIEW input: from each VIEW-producing member, else from a random
    VIEW-producing ancestor, else from a brand-new VIEWER source keyed on ``target``."""
    producers = [m for m in members if Kind.VIEW in _outputs(genome, m)]
    if producers:
        for m in producers:
            _connect(genome, ctx, m, target, Kind.VIEW)
        return
    pool = sorted(a for a in ancestors if a in genome.nodes and Kind.VIEW in _outputs(genome, a))
    if pool:
        _connect(genome, ctx, ctx.rng.choice(pool), target, Kind.VIEW)
        return
    v = _new_node(genome, ctx, ("source", target, Kind.VIEW, WorkerType.VIEWER), key_prefix)
    _connect(genome, ctx, v, target, Kind.VIEW)


def _attempt(genome: Genome, ctx: MutationContext, propose: Callable[[Genome], None]) -> bool:
    """Run up to ``max_attempts`` proposals on copies; swap the first valid result in."""
    for _ in range(ctx.config.max_attempts):
        work = genome.copy(new_id=False)
        try:
            propose(work)
        except _Reject:
            continue
        if work.is_valid():
            genome.nodes = work.nodes
            genome.connections = work.connections
            return True
    return False


# ------------------------------------------------------------ insertion table rows


def _row_listviewer(g: Genome, ctx: MutationContext, kp: str | None) -> None:
    edges = [c for c in g.enabled_connections() if c.kind is Kind.VIEW]
    if not edges:
        raise _Reject
    conn = ctx.rng.choice(edges)
    n = _new_node(g, ctx, ("split", conn.innovation, WorkerType.LISTVIEWER), kp)
    conn.enabled = False
    _connect(g, ctx, conn.src, n, Kind.VIEW)
    _connect(g, ctx, n, conn.dst, Kind.VIEW)


def _row_critic(g: Genome, ctx: MutationContext, kp: str | None) -> None:
    edges = [c for c in g.enabled_connections() if c.kind is Kind.EDIT]
    if not edges:
        raise _Reject
    conn = ctx.rng.choice(edges)
    s, d = conn.src, conn.dst
    ancestors = _ancestors(g, s)
    n = _new_node(g, ctx, ("critic", conn.innovation, WorkerType.CRITIC), kp)
    conn.enabled = False
    _connect(g, ctx, s, n, Kind.EDIT)
    _connect(g, ctx, n, d, Kind.EDIT)
    _feed_view(g, ctx, n, [s], ancestors, kp)


def _row_viewer(g: Genome, ctx: MutationContext, kp: str | None) -> None:
    sites = sorted(nid for nid in g.nodes if _has_slot(g, nid, Kind.VIEW))
    if not sites:
        raise _Reject
    d = ctx.rng.choice(sites)
    n = _new_node(g, ctx, ("source", d, Kind.VIEW, WorkerType.VIEWER), kp)
    _connect(g, ctx, n, d, Kind.VIEW)
    if _has_slot(g, d, Kind.REPORT) and ctx.rng.random() < _REPORT_PROBABILITY:
        _connect(g, ctx, n, d, Kind.REPORT)


def _make_vieweditor(g: Genome, ctx: MutationContext, kp: str | None) -> int:
    """New VIEWEDITOR fed by the same VIEW sources as a random node with >=1 VIEW input."""
    sites = sorted(nid for nid in g.nodes if _view_sources(g, nid))
    if not sites:
        raise _Reject
    d = ctx.rng.choice(sites)
    sources = _view_sources(g, d)
    n = _new_node(g, ctx, ("mirror", d, WorkerType.VIEWEDITOR), kp)
    for u in sources:
        _connect(g, ctx, u, n, Kind.VIEW)
    return n


def _make_editor(g: Genome, ctx: MutationContext, kp: str | None) -> int:
    """New EDITOR fed by a random VIEW-producing node that does not feed an editor."""
    sites = sorted(
        v
        for v in g.nodes
        if Kind.VIEW in _outputs(g, v)
        and not any(
            c.dst != END and g.nodes[c.dst].type is WorkerType.EDITOR  # type: ignore[index]
            for c in g.connections_from(v, enabled_only=True)
        )
    )
    if not sites:
        raise _Reject
    v = ctx.rng.choice(sites)
    n = _new_node(g, ctx, ("branch", v, WorkerType.EDITOR), kp)
    _connect(g, ctx, v, n, Kind.VIEW)
    if Kind.REPORT in _outputs(g, v):
        _connect(g, ctx, v, n, Kind.REPORT)
    return n


def _row_vieweditor(g: Genome, ctx: MutationContext, kp: str | None) -> None:
    n = _make_vieweditor(g, ctx, kp)
    upstream = _ancestors(g, n)
    mergers = sorted(
        m for m, node in g.nodes.items() if node.type in MERGER_TYPES and m not in upstream
    )
    if mergers:
        _connect(g, ctx, n, ctx.rng.choice(mergers), Kind.EDIT)
    else:
        _merge(g, ctx, n, kp)


def _row_editor(g: Genome, ctx: MutationContext, kp: str | None) -> None:
    n = _make_editor(g, ctx, kp)
    _merge(g, ctx, n, kp)


_ROWS: dict[str, Callable[[Genome, MutationContext, str | None], None]] = {
    "listviewer": _row_listviewer,
    "critic": _row_critic,
    "viewer": _row_viewer,
    "vieweditor": _row_vieweditor,
    "editor": _row_editor,
}


def _merge(g: Genome, ctx: MutationContext, c: int, kp: str | None) -> None:
    """Insert a merger joining the END source and candidate ``c`` in front of END."""
    e = _end_source(g)
    if c not in g.nodes or c == e or Kind.EDIT not in _outputs(g, c) or _edit_reaches_end(g, c):
        raise _Reject
    ancestors = _ancestors(g, e) | _ancestors(g, c)
    mtype = ctx.rng.choice(_MERGER_ORDER)
    m = _new_node(g, ctx, ("merge", e, c, mtype), kp)
    end_conn = g.find_connection(e, END, Kind.EDIT, enabled_only=True)
    assert end_conn is not None
    end_conn.enabled = False
    _connect(g, ctx, e, m, Kind.EDIT)
    _connect(g, ctx, c, m, Kind.EDIT)
    _connect(g, ctx, m, END, Kind.EDIT)
    _feed_view(g, ctx, m, [e, c], ancestors, kp)
    for x in (e, c):
        if Kind.REPORT in _outputs(g, x) and ctx.rng.random() < _REPORT_PROBABILITY:
            _connect(g, ctx, x, m, Kind.REPORT)


# ------------------------------------------------------------------------ operators


def _mutate_node_properties(node: NodeGene, rng: random.Random) -> bool:
    specs = PROPERTIES.get(node.type, {})
    names = [name for name, spec in specs.items() if len(spec.options) > 1]
    if not names:
        return False
    name = rng.choice(names)
    current = node.properties.get(name, 0)
    node.properties[name] = rng.choice(
        [i for i in range(len(specs[name].options)) if i != current]
    )
    return True


def mutate_properties(genome: Genome, ctx: MutationContext) -> bool:
    """Per node gene with ``p_property_per_gene``: set one random property to a different
    option index. If no gene was picked, one random gene is mutated so the call always
    changes something (when any property has more than one option)."""
    rng = ctx.rng
    ids = sorted(genome.nodes)
    changed = False
    for nid in ids:
        if rng.random() < ctx.config.p_property_per_gene:
            changed |= _mutate_node_properties(genome.nodes[nid], rng)
    if not changed and ids:
        changed = _mutate_node_properties(genome.nodes[rng.choice(ids)], rng)
    return changed


def add_connection(genome: Genome, ctx: MutationContext) -> bool:
    """Add (or re-enable) a random legal ``(src, dst, kind)`` edge between workers."""

    def propose(g: Genome) -> None:
        enabled = {(c.src, c.dst, c.kind) for c in g.enabled_connections()}
        ids = sorted(g.nodes)
        candidates = [
            (s, d, k)
            for s in ids
            for k in Kind
            if k in _outputs(g, s)
            for d in ids
            if d != s and _has_slot(g, d, k) and (s, d, k) not in enabled
        ]
        if not candidates:
            raise _Reject
        s, d, k = ctx.rng.choice(candidates)
        if d in _ancestors(g, s):
            raise _Reject  # would close a cycle
        _connect(g, ctx, s, d, k)

    return _attempt(genome, ctx, propose)


def add_worker(
    genome: Genome,
    ctx: MutationContext,
    *,
    key_prefix: str | None = None,
    row: str | None = None,
) -> bool:
    """Insert one worker via the typed insertion table (a random row per attempt, or the
    named ``row`` from :data:`ADD_WORKER_ROWS`)."""
    if row is not None and row not in _ROWS:
        raise ValueError(f"unknown add_worker row {row!r}; expected one of {ADD_WORKER_ROWS}")

    def propose(g: Genome) -> None:
        name = row if row is not None else ctx.rng.choice(ADD_WORKER_ROWS)
        _ROWS[name](g, ctx, key_prefix)

    return _attempt(genome, ctx, propose)


def insert_merger(
    genome: Genome,
    ctx: MutationContext,
    candidate: int | None = None,
    *,
    key_prefix: str | None = None,
) -> bool:
    """Put a JUDGEMERGER/PICKMERGER in front of END joining the END source and a second
    EDIT producer whose edit does not yet reach END (picked, given, or newly created)."""

    def propose(g: Genome) -> None:
        c = candidate
        if c is None:
            e = _end_source(g)
            pool = sorted(
                x
                for x in g.nodes
                if x != e and Kind.EDIT in _outputs(g, x) and not _edit_reaches_end(g, x)
            )
            if pool:
                c = ctx.rng.choice(pool)
            else:
                makers = [_make_editor, _make_vieweditor]
                ctx.rng.shuffle(makers)
                for make in makers:
                    try:
                        c = make(g, ctx, key_prefix)
                        break
                    except _Reject:
                        continue
                else:
                    raise _Reject
        _merge(g, ctx, c, key_prefix)

    return _attempt(genome, ctx, propose)


def remove_connection(genome: Genome, ctx: MutationContext) -> bool:
    """Delete a random enabled connection gene if the genome stays valid."""

    def propose(g: Genome) -> None:
        enabled = g.enabled_connections()
        if not enabled:
            raise _Reject
        del g.connections[ctx.rng.choice(enabled).innovation]

    return _attempt(genome, ctx, propose)


def remove_worker(genome: Genome, ctx: MutationContext) -> bool:
    """Remove a random non-END-source worker and its edges, collapsing any merger left with
    a single EDIT input (the remaining candidate inherits the merger's outgoing edges)."""

    def propose(g: Genome) -> None:
        e = _end_source(g)
        pool = sorted(n for n in g.nodes if n != e)
        if not pool:
            raise _Reject
        x = ctx.rng.choice(pool)
        mergers = sorted(
            {
                c.dst  # type: ignore[misc]
                for c in g.connections_from(x, enabled_only=True)
                if c.kind is Kind.EDIT and c.dst != END and g.nodes[c.dst].type in MERGER_TYPES  # type: ignore[index]
            }
        )
        _delete_node(g, x)
        for m in mergers:
            edit_in = [c for c in g.connections_into(m, enabled_only=True) if c.kind is Kind.EDIT]
            if len(edit_in) != 1:
                continue
            r = edit_in[0].src
            for out in g.connections_from(m, enabled_only=True):
                if out.kind in _outputs(g, r) and out.dst != r:
                    _connect(g, ctx, r, out.dst, out.kind)
            _delete_node(g, m)

    return _attempt(genome, ctx, propose)


def toggle_enable(genome: Genome, ctx: MutationContext) -> bool:
    """Flip the ``enabled`` flag of a random connection gene if the genome stays valid."""

    def propose(g: Genome) -> None:
        if not g.connections:
            raise _Reject
        conn = g.connections[ctx.rng.choice(sorted(g.connections))]
        conn.enabled = not conn.enabled

    return _attempt(genome, ctx, propose)


OPERATORS: dict[str, Callable[..., bool]] = {
    "mutate_properties": mutate_properties,
    "add_connection": add_connection,
    "add_worker": add_worker,
    "insert_merger": insert_merger,
    "remove_connection": remove_connection,
    "remove_worker": remove_worker,
    "toggle_enable": toggle_enable,
}
"""Every operator by name; each is called as ``op(genome, ctx) -> bool``."""


def _reset(genome: Genome) -> Genome:
    genome.fitness = None
    genome.adjusted_fitness = None
    genome.species_id = None
    return genome


def mutate(genome: Genome, ctx: MutationContext) -> Genome:
    """Return a mutated, valid copy of ``genome`` (see module docstring for bookkeeping)."""
    cfg = ctx.config
    rng = ctx.rng
    child = _reset(genome.copy())
    applied: list[str] = []
    if rng.random() < cfg.p_property and mutate_properties(child, ctx):
        applied.append("mutate_properties")
    for name, p in (
        ("add_connection", cfg.p_add_connection),
        ("add_worker", cfg.p_add_worker),
        ("insert_merger", cfg.p_insert_merger),
        ("remove_connection", cfg.p_remove_connection),
        ("remove_worker", cfg.p_remove_worker),
        ("toggle_enable", cfg.p_toggle_enable),
    ):
        if rng.random() < p and OPERATORS[name](child, ctx):
            applied.append(name)
    if child.is_valid():
        child.operators.extend(applied)
        return child
    fallback = _reset(genome.copy())
    fallback.validate()  # raises InvalidGraph if the input itself was invalid
    fallback.operators.append("noop")
    return fallback


def random_valid_genome(
    innovations: InnovationDB,
    rng: random.Random,
    config: MutationConfig | Any,
    *,
    min_workers: int,
    max_workers: int,
    generation: int = 0,
) -> Genome:
    """A random valid genome with a worker count drawn from ``[min_workers, max_workers]``.

    Starts from :func:`minimal_genome` (editor with random properties) and applies
    ``add_worker`` / ``insert_merger`` with ``("init", index, type)`` node keys.
    """
    mcfg: MutationConfig = getattr(config, "mutation", config)
    if not 1 <= min_workers <= max_workers:
        raise ValueError(f"need 1 <= min_workers <= max_workers, got {min_workers}, {max_workers}")
    genome = minimal_genome(
        innovations, random_properties(WorkerType.EDITOR, rng), generation=generation
    )
    target = rng.randint(min_workers, max_workers)
    ctx = MutationContext(rng=rng, innovations=innovations, config=mcfg, max_workers=target)
    failures = 0
    while genome.num_workers() < target and failures < 200:
        if rng.random() < 0.25:
            ok = insert_merger(genome, ctx, key_prefix=INIT_KEY_PREFIX)
        else:
            ok = add_worker(genome, ctx, key_prefix=INIT_KEY_PREFIX)
        failures += 0 if ok else 1
    if genome.num_workers() < min_workers:  # pragma: no cover - VIEWER row always applies
        raise InvalidGraph(f"could not grow a genome to {min_workers} workers")
    genome.validate()
    return genome


__all__ = [
    "ADD_WORKER_ROWS",
    "INIT_KEY_PREFIX",
    "OPERATORS",
    "MutationContext",
    "add_connection",
    "add_worker",
    "insert_merger",
    "mutate",
    "mutate_properties",
    "random_valid_genome",
    "remove_connection",
    "remove_worker",
    "toggle_enable",
]
