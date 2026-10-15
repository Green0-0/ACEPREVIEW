"""Pure functions over run records (no I/O): the derived tables behind the WebUI API.

Contract readings:

* **Operator stats** use *adjusted* fitness throughout: for each child genome that names an
  operator (an operator listed twice for one child counts once), ``mean_child_fitness`` is
  the mean child adjusted fitness, ``mean_parent_fitness`` the mean of the *best* logged
  parent's adjusted fitness, ``mean_delta`` their difference (= mean of per-child deltas)
  and ``success_rate`` the fraction of children with adjusted fitness >= best parent.
  ``count`` counts every child naming the operator; the means only use children that have
  an adjusted fitness and at least one logged parent with one (``null`` when none do).
* **Lineage** is breadth-first over logged parents, de-duplicated, own record excluded,
  oldest ancestors last (stable sort by descending generation), genome bodies stripped.
* **Layout** levels: a worker with no incoming worker edge is level 0, otherwise
  ``1 + max(level of its sources)``; END is one level below the deepest worker. Within a
  level nodes are ordered by the mean x of their sources (then id). Coordinates are
  normalised: ``x = (i + 1) / (n + 1)`` for the i-th of n nodes in the level,
  ``y = level / end_level`` (0 at the top, 1 at END).
"""
from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Iterable
from enum import Enum
from typing import Any

from ace.graph.graph import WorkerGraph
from ace.graph.types import END, WorkerType

EDGE_COLOURS = {"view": "#1f77b4", "edit": "#d62728", "report": "#2ca02c"}
"""DOT edge colour per kind (view blue, edit red, report green)."""

EVENTS_TAIL = 50


# --------------------------------------------------------------------------- helpers


def jsonable(value: Any) -> Any:
    """Turn enums/tuples/sets/other objects into JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def strip_genome_body(record: dict[str, Any]) -> dict[str, Any]:
    """A genome record without its ``genome`` body."""
    return {k: v for k, v in record.items() if k != "genome"}


def strip_workers(record: dict[str, Any]) -> dict[str, Any]:
    """An evaluation record without ``workers``."""
    return {k: v for k, v in record.items() if k != "workers"}


def strip_worker_views(record: dict[str, Any]) -> dict[str, Any]:
    """An evaluation record whose workers have no ``view`` key."""
    out = dict(record)
    if isinstance(record.get("workers"), list):
        out["workers"] = [
            {k: v for k, v in w.items() if k != "view"} if isinstance(w, dict) else w
            for w in record["workers"]
        ]
    return out


# --------------------------------------------------------------------------- runs


def run_summary(run_id: str, run: dict[str, Any] | None, generations: list[dict[str, Any]], status: str) -> dict[str, Any]:
    """One row of ``GET /api/runs``."""
    run = run or {}
    config = run.get("config") if isinstance(run.get("config"), dict) else {}
    population_size = config.get("population_size")
    if population_size is None and generations:
        last = generations[-1]
        pop = last.get("population")
        population_size = len(pop) if isinstance(pop, list) else last.get("evaluated")
    evaluator = run.get("evaluator") if isinstance(run.get("evaluator"), dict) else {}
    best = run.get("best_fitness")
    if best is None and generations:
        bests = [_num(g.get("best_fitness")) for g in generations]
        bests = [b for b in bests if b is not None]
        best = max(bests) if bests else None
    completed = run.get("generations_completed")
    if completed is None:
        completed = len(generations)
    started = run.get("started_at")
    if started is None and generations:
        started = generations[0].get("started_at")
    return {
        "run_id": run_id,
        "status": status,
        "started_at": started,
        "finished_at": run.get("finished_at"),
        "generations_completed": completed,
        "best_fitness": best,
        "population_size": population_size,
        "evaluator_type": evaluator.get("type"),
    }


def progress(generations: list[dict[str, Any]], events: list[dict[str, Any]], status: str) -> dict[str, Any]:
    """``GET /progress``: generation records without population, last 50 events, status."""
    return {
        "generations": [{k: v for k, v in g.items() if k != "population"} for g in generations],
        "events_tail": events[-EVENTS_TAIL:],
        "status": status,
    }


# --------------------------------------------------------------------------- species


def species_timeline(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-generation species sizes plus a per-species summary (keys are id strings)."""
    timeline = []
    summary: dict[str, dict[str, Any]] = {}
    for snap in snapshots:
        gen = snap.get("generation")
        rows = []
        for sp in snap.get("species") or []:
            sid = sp.get("id")
            size = sp.get("size")
            if size is None and isinstance(sp.get("member_ids"), list):
                size = len(sp["member_ids"])
            best = _num(sp.get("best_fitness"))
            rows.append(
                {"id": sid, "size": size, "best_fitness": best, "representative_id": sp.get("representative_id")}
            )
            entry = summary.setdefault(
                str(sid),
                {"created": sp.get("created", gen), "last_seen": gen, "best_fitness": None, "max_size": 0},
            )
            if entry["created"] is None:
                entry["created"] = gen
            entry["last_seen"] = gen
            if best is not None and (entry["best_fitness"] is None or best > entry["best_fitness"]):
                entry["best_fitness"] = best
            if isinstance(size, int) and size > entry["max_size"]:
                entry["max_size"] = size
        timeline.append({"generation": gen, "threshold": snap.get("threshold"), "species": rows})
    return {"timeline": timeline, "species": summary}


# --------------------------------------------------------------------------- genomes


def filter_genomes(
    records: list[dict[str, Any]],
    *,
    generation: int | None = None,
    species: int | None = None,
    min_fitness: float | None = None,
    sort: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Filter, sort (``field`` ascending, ``-field`` descending; nulls last), paginate."""
    items = [
        r
        for r in records
        if (generation is None or r.get("generation") == generation)
        and (species is None or r.get("species_id") == species)
        and (min_fitness is None or ((_num(r.get("fitness")) is not None) and r["fitness"] >= min_fitness))
    ]
    if sort:
        descending = sort.startswith("-")
        key = sort.lstrip("-+")
        present = [r for r in items if r.get(key) is not None]
        missing = [r for r in items if r.get(key) is None]
        try:
            present.sort(key=lambda r: r[key], reverse=descending)
        except TypeError:
            present.sort(key=lambda r: str(r[key]), reverse=descending)
        items = present + missing
    total = len(items)
    page = items[max(0, offset): max(0, offset) + max(0, limit)]
    return {"total": total, "items": [strip_genome_body(r) for r in page]}


def lineage(records: dict[str, dict[str, Any]], genome_id: str) -> list[dict[str, Any]]:
    """Ancestor records of ``genome_id`` (bodies stripped), breadth-first, oldest last."""
    if genome_id not in records:
        return []
    chain: list[dict[str, Any]] = []
    seen = {genome_id}
    queue: deque[str] = deque([genome_id])
    while queue:
        current = queue.popleft()
        for parent in records[current].get("parents") or []:
            if parent in seen or parent not in records:
                continue
            seen.add(parent)
            queue.append(parent)
            chain.append(records[parent])
    if all(isinstance(r.get("generation"), int) for r in chain):
        chain.sort(key=lambda r: -r["generation"])
    return [strip_genome_body(r) for r in chain]


def operator_stats(records: Iterable[dict[str, Any]], by_id: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per-operator child-vs-best-parent adjusted fitness (see module docstring)."""
    records = list(records)
    if by_id is None:
        by_id = {}
        for r in records:
            by_id.setdefault(r.get("id"), r)
    acc: dict[str, dict[str, Any]] = {}
    for child in records:
        ops = list(dict.fromkeys(str(o) for o in child.get("operators") or []))
        if not ops:
            continue
        child_adj = _num(child.get("adjusted_fitness"))
        parent_adjs = [
            _num(by_id[p].get("adjusted_fitness"))
            for p in child.get("parents") or []
            if p in by_id
        ]
        parent_adjs = [p for p in parent_adjs if p is not None]
        best_parent = max(parent_adjs) if parent_adjs else None
        for op in ops:
            entry = acc.setdefault(op, {"count": 0, "child": [], "parent": [], "success": 0})
            entry["count"] += 1
            if child_adj is not None and best_parent is not None:
                entry["child"].append(child_adj)
                entry["parent"].append(best_parent)
                if child_adj >= best_parent:
                    entry["success"] += 1
    out = []
    for op, entry in acc.items():
        n = len(entry["child"])
        mean_child = _mean(entry["child"])
        mean_parent = _mean(entry["parent"])
        out.append(
            {
                "operator": op,
                "count": entry["count"],
                "mean_child_fitness": mean_child,
                "mean_parent_fitness": mean_parent,
                "mean_delta": None if n == 0 else mean_child - mean_parent,  # type: ignore[operator]
                "success_rate": None if n == 0 else entry["success"] / n,
            }
        )
    out.sort(key=lambda r: (-r["count"], r["operator"]))
    return out


# --------------------------------------------------------------------------- graphs


def resolved_properties(node_type: str, properties: dict[str, Any]) -> dict[str, Any]:
    """``properties.resolve`` as JSON; falls back to the raw indices when invalid."""
    try:
        from ace.graph.properties import resolve

        return jsonable(resolve(WorkerType(node_type), dict(properties)))
    except Exception:  # noqa: BLE001 - unknown type / out-of-range index: show raw values
        return jsonable(dict(properties))


def graph_levels(graph: WorkerGraph) -> dict[Any, int]:
    """Topological level of every worker plus END (one below the deepest worker)."""
    try:
        order = graph.topological_order()
    except Exception:  # noqa: BLE001 - cyclic/invalid: fall back to id order
        order = sorted(graph.nodes)
    levels: dict[Any, int] = {}
    for nid in order:
        sources = [e.src for e in graph.incoming(nid) if e.src in levels]
        levels[nid] = 1 + max(levels[s] for s in sources) if sources else 0
    levels[END] = (max(levels.values()) + 1) if levels else 0
    return levels


def layout(graph: WorkerGraph) -> dict[str, Any]:
    """``{nodes: [{id, type, level, x, y, properties_resolved}], edges: [{src, dst, kind}]}``."""
    levels = graph_levels(graph)
    end_level = levels[END]
    by_level: dict[int, list[Any]] = {}
    for nid, lvl in levels.items():
        by_level.setdefault(lvl, []).append(nid)
    xs: dict[Any, float] = {}
    for lvl in sorted(by_level):
        members = by_level[lvl]

        def bary(nid: Any) -> tuple[float, float]:
            srcs = [xs[e.src] for e in graph.incoming(nid) if e.src in xs]
            return (sum(srcs) / len(srcs) if srcs else 0.5, -1 if nid == END else float(nid))

        members.sort(key=bary)
        n = len(members)
        for i, nid in enumerate(members):
            xs[nid] = (i + 1) / (n + 1)
    nodes = []
    for nid in [*sorted(graph.nodes), END]:
        lvl = levels[nid]
        if nid == END:
            node_type, props = "end", {}
        else:
            node = graph.nodes[nid]
            node_type = node.type.value
            props = resolved_properties(node_type, node.properties)
        nodes.append(
            {
                "id": nid,
                "type": node_type,
                "level": lvl,
                "x": xs[nid],
                "y": lvl / end_level if end_level else 0.0,
                "properties_resolved": props,
            }
        )
    edges = [{"src": e.src, "dst": e.dst, "kind": jsonable(e.kind)} for e in graph.edges]
    return {"nodes": nodes, "edges": edges}


def to_dot(graph: WorkerGraph, name: str = "genome") -> str:
    """Graphviz DOT with one box per worker, an END node and kind-coloured edges."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "genome"
    lines = [f"digraph g_{safe} {{", "  rankdir=TB;", '  node [shape=box, style="rounded,filled", fillcolor="#f5f5f5"];']
    for nid, node in sorted(graph.nodes.items()):
        lines.append(f'  n{nid} [label="{nid}: {node.type.value}"];')
    lines.append('  END [label="END", shape=doublecircle, fillcolor="#333333", fontcolor="white"];')
    for e in graph.edges:
        kind = jsonable(e.kind)
        dst = "END" if e.dst == END else f"n{e.dst}"
        colour = EDGE_COLOURS.get(kind, "#000000")
        lines.append(f'  n{e.src} -> {dst} [label="{kind}", color="{colour}", fontcolor="{colour}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def genome_graph(record: dict[str, Any]) -> WorkerGraph | None:
    """The WorkerGraph of a genome record's body, or None if there is no usable body."""
    body = record.get("genome")
    if not isinstance(body, dict):
        return None
    from ace.neat.genome import Genome

    try:
        return Genome.from_json(body).to_graph()
    except Exception:  # noqa: BLE001 - malformed body
        return None


# --------------------------------------------------------------------------- compare


def compare_genomes(record_a: dict[str, Any], record_b: dict[str, Any], speciation_cfg: Any) -> dict[str, Any]:
    """Graphs/layouts of both genomes, per-matched-node property diffs and NEAT distance."""
    from ace.neat.genome import Genome
    from ace.neat.speciation import distance, excess_disjoint, property_difference

    ga = Genome.from_json(record_a["genome"])
    gb = Genome.from_json(record_b["genome"])
    graph_a, graph_b = ga.to_graph(), gb.to_graph()
    matched = sorted(ga.nodes.keys() & gb.nodes.keys())
    diffs = []
    for inn in matched:
        na, nb = ga.nodes[inn], gb.nodes[inn]
        ra = resolved_properties(na.type.value, na.properties)
        rb = resolved_properties(nb.type.value, nb.properties)
        changes = {}
        for prop in sorted(na.properties.keys() | nb.properties.keys()):
            ia, ib = na.properties.get(prop), nb.properties.get(prop)
            if ia != ib:
                changes[prop] = {"a": ia, "b": ib, "a_resolved": ra.get(prop), "b_resolved": rb.get(prop)}
        diffs.append({"node": inn, "type": na.type.value, "changed": len(changes), "diffs": changes})
    excess, disjoint = excess_disjoint(ga, gb)
    conns_a, conns_b = set(ga.connections), set(gb.connections)

    def side(record: dict[str, Any], graph: WorkerGraph) -> dict[str, Any]:
        return {"genome": strip_genome_body(record), "graph": graph.to_json(), "layout": layout(graph)}

    return {
        "a": side(record_a, graph_a),
        "b": side(record_b, graph_b),
        "matched_nodes": matched,
        "only_a_nodes": sorted(ga.nodes.keys() - gb.nodes.keys()),
        "only_b_nodes": sorted(gb.nodes.keys() - ga.nodes.keys()),
        "matched_connections": sorted(conns_a & conns_b),
        "only_a_connections": sorted(conns_a - conns_b),
        "only_b_connections": sorted(conns_b - conns_a),
        "property_diffs": diffs,
        "excess": excess,
        "disjoint": disjoint,
        "property_difference": property_difference(ga, gb),
        "distance": distance(ga, gb, speciation_cfg),
    }


def speciation_config(run: dict[str, Any] | None) -> Any:
    """``SpeciationConfig`` from ``run.json`` config (defaults when absent or invalid)."""
    from ace.neat.config import SpeciationConfig

    config = (run or {}).get("config")
    raw = config.get("speciation") if isinstance(config, dict) else None
    if isinstance(raw, dict):
        try:
            return SpeciationConfig.model_validate(raw)
        except Exception:  # noqa: BLE001 - foreign/old config: keep the known fields
            known = {k: v for k, v in raw.items() if k in SpeciationConfig.model_fields}
            try:
                return SpeciationConfig.model_validate(known)
            except Exception:  # noqa: BLE001
                pass
    return SpeciationConfig()


# --------------------------------------------------------------------------- innovations


def innovations_table(data: dict[str, Any] | None) -> dict[str, Any]:
    """``{"nodes": [{id, key, type}], "connections": [{id, src, dst, kind}]}`` (by id)."""
    if not data:
        return {"nodes": [], "connections": []}
    from ace.neat.innovation import decode_key

    nodes = []
    for entry in data.get("nodes") or []:
        try:
            key = list(decode_key(entry["key"]))
        except Exception:  # noqa: BLE001 - unknown encoding: keep it raw
            key = entry.get("key") or []
        last = key[-1] if key else None
        nodes.append({"id": entry.get("id"), "key": jsonable(key), "type": jsonable(last)})
    connections = [
        {"id": e.get("id"), "src": e.get("src"), "dst": e.get("dst"), "kind": e.get("kind")}
        for e in data.get("connections") or []
    ]
    nodes.sort(key=lambda r: (r["id"] is None, r["id"] or 0))
    connections.sort(key=lambda r: (r["id"] is None, r["id"] or 0))
    return {"nodes": nodes, "connections": connections}


# --------------------------------------------------------------------------- evaluations


def filter_evaluations(
    records: list[dict[str, Any]],
    *,
    genome: str | None = None,
    task: int | None = None,
    failed: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    items = [
        r
        for r in records
        if (genome is None or r.get("genome_id") == genome)
        and (task is None or r.get("task_index", 0) == task)
        and (failed is None or (r.get("failure") is not None) == failed)
    ]
    page = items[max(0, offset): max(0, offset) + max(0, limit)]
    return {"total": len(items), "items": [strip_workers(r) for r in page]}


def failure_node_type(record: dict[str, Any]) -> str | None:
    """Type of the failing worker: ``failure_node_id``'s worker, else the first failed one."""
    workers = [w for w in record.get("workers") or [] if isinstance(w, dict)]
    node_id = record.get("failure_node_id")
    if node_id is not None:
        for w in workers:
            if w.get("node_id") == node_id:
                return w.get("type")
    for w in workers:
        if w.get("status") == "failed":
            return w.get("type")
    return None


def failures(records: list[dict[str, Any]], examples: int = 5) -> list[dict[str, Any]]:
    """Failed evaluations grouped by ``(failure_type, failing node type)``, most frequent first."""
    groups: dict[tuple[Any, Any], dict[str, Any]] = {}
    counts: Counter[tuple[Any, Any]] = Counter()
    for r in records:
        if r.get("failure") is None:
            continue
        key = (r.get("failure_type"), failure_node_type(r))
        counts[key] += 1
        group = groups.setdefault(key, {"failure_type": key[0], "node_type": key[1], "count": 0, "examples": []})
        if len(group["examples"]) < examples:
            group["examples"].append(
                {"genome_id": r.get("genome_id"), "task_index": r.get("task_index", 0), "message": r.get("failure")}
            )
    for key, group in groups.items():
        group["count"] = counts[key]
    return sorted(groups.values(), key=lambda g: (-g["count"], str(g["failure_type"]), str(g["node_type"])))


__all__ = [
    "EDGE_COLOURS",
    "compare_genomes",
    "failures",
    "filter_evaluations",
    "filter_genomes",
    "genome_graph",
    "graph_levels",
    "innovations_table",
    "jsonable",
    "layout",
    "lineage",
    "operator_stats",
    "progress",
    "resolved_properties",
    "run_summary",
    "species_timeline",
    "speciation_config",
    "strip_genome_body",
    "strip_worker_views",
    "strip_workers",
    "to_dot",
]
