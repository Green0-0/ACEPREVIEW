#!/usr/bin/env python3
"""Pick candidate genomes from NEAT run directories for held-out validation.

    python scripts/select_genomes.py runs/neat_a runs/neat_b --top 3 --out runs/selection

Minibatch fitness is noisy, so a genome's training score is the mean over every
minibatch it was scored on (``evaluations.jsonl`` records, grouped by ``batch``), and the
ranking key is a shrunk mean ``(n * mean + k * prior) / (n + k)`` over its task scores
(``k`` pseudo-tasks at the population's overall mean task score). This prefers genomes
that did well on many tasks over ones that got lucky on a single batch. Graphs with an
identical WorkerGraph JSON (same structure and properties) are pooled. Writes
``<out>/<run>_<rank>_<genome id>.json`` (best_genome.json format) and ``<out>/selection.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--top", type=int, default=3, help="candidates per run")
    p.add_argument("--k", type=float, default=6.0, help="shrinkage pseudo-tasks")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection = []
    for run in args.runs:
        genomes = {g["id"]: g for g in load_jsonl(run / "genomes.jsonl")}
        state = run / "population_state.json"
        if state.is_file():  # latest copies (fitness_history of elites)
            for g in json.loads(state.read_text()).get("genomes", []):
                genomes.setdefault(g["id"], {"id": g["id"], "genome": g})
        scores: dict[str, list[float]] = defaultdict(list)
        for rec in load_jsonl(run / "evaluations.jsonl"):
            if rec.get("mock"):
                continue
            scores[rec["genome_id"]].append(float(rec.get("score", 0.0)))
        all_scores = [s for v in scores.values() for s in v]
        if not all_scores:
            print(f"{run}: no evaluations yet")
            continue
        prior = statistics.fmean(all_scores)
        # pool identical graphs
        from ace.neat.genome import Genome

        pooled: dict[str, dict] = {}
        for gid, sc in scores.items():
            rec = genomes.get(gid)
            if rec is None:
                continue
            genome = Genome.from_json(rec.get("genome") or rec)
            key = json.dumps(genome.to_graph().to_json(), sort_keys=True)
            entry = pooled.setdefault(key, {"ids": [], "scores": [], "genome": genome})
            entry["ids"].append(gid)
            entry["scores"].extend(sc)
        ranked = []
        for entry in pooled.values():
            n = len(entry["scores"])
            mean = statistics.fmean(entry["scores"])
            shrunk = (n * mean + args.k * prior) / (n + args.k)
            ranked.append((shrunk, mean, n, entry))
        ranked.sort(key=lambda t: (-t[0], t[3]["genome"].num_workers()))
        for rank, (shrunk, mean, n, entry) in enumerate(ranked[: args.top]):
            genome = entry["genome"]
            name = f"{run.name}_{rank}_{genome.id[:8]}"
            path = args.out / f"{name}.json"
            path.write_text(json.dumps({**genome.to_json(), "graph": genome.to_graph().to_json()}, indent=2))
            row = {"name": name, "path": str(path), "run": str(run), "genome_ids": entry["ids"],
                   "train_tasks": n, "train_mean": round(mean, 4), "shrunk": round(shrunk, 4),
                   "workers": genome.num_workers(),
                   "types": sorted(n_.type.value for n_ in genome.nodes.values())}
            selection.append(row)
            print(json.dumps(row))
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
