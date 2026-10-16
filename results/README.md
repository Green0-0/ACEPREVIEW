# Results

The outputs of the SWE-bench Verified experiment described in the top-level `README.md`. Per-instance predictions, rollouts, evaluation logs and NEAT histories are too large for git, so they stay under `/usr/project/xtmp/kg349/ace/runs/`.

| Path | Contents |
|---|---|
| `analysis_v2/` | Main comparison on all 500 instances: the no-step-limit baseline (reference), two NEAT runs and two 30-step baseline runs. Includes resolve rates with Wilson CIs, McNemar tests, a paired bootstrap, GEE, token tests, proxy validity and NEAT curves (`scripts/analyze.py`). |
| `analysis_v2_gold/` | The same comparison on the 464 instances whose gold patch resolves in the sandbox. |
| `analysis_v1/`, `analysis_v1_gold/` | The first comparison: NEAT vs the 30-step baseline only. |
| `graph_analysis/` | Secondary analysis of evolved graphs: size, worker types and settings, topology, cost, failure modes and lineage (`scripts/graph_analysis.py`). Has three figures; `graphs.csv` and `features.csv` hold the data behind them. |
| `selection/` | The candidate genomes picked from the NEAT runs for validation, with `selection.json`. |
| `graphs/neat_best.json` | The winning worker graph (`neat_b_0_37b607a6`): two editors joined by a pick-merger. |
| `graphs/nested_tree.json` | The hand-built nested tree: 4 chained editors, 2 pick-mergers and a judge-merger. |
| `gold_validation/` | Gold-patch sandbox validation summary, and the list of gold-resolvable instance ids. |
| `INTERVENTIONS.md` | A timestamped log of operational decisions made during the runs (admission caps, restarts, pauses). |
