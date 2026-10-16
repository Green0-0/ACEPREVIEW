# ACE: Discovering Agentic Pipelines with NEAT

ACE builds coding pipelines out of LLM workers and evolves their structure with NEAT. A pipeline is a typed directed acyclic graph of workers (viewers, list viewers, editors, view-editors, critics, judge-mergers and pick-mergers) whose edges carry views of the code, edits and reports, and which ends in exactly one proposed edit of a repository. Every worker reads the same pristine repository and edits a private workspace, so a run never changes the project on disk: its result is a git patch.

## Layout

```
src/ace/graph/          worker-graph runtime (types, views, edits, prompts, LLM client, runner)
src/ace/graph/workers/  one module per worker type
src/ace/neat/           NEAT: genome, mutation, crossover, speciation, population, evaluation
src/ace/swebench/       SWE-bench data loading, task prompt, patch-similarity proxy, sandbox eval
src/ace/serving.py      pooled clients over a directory of vLLM endpoints
src/ace/webui/          read-only web UI over NEAT run directories
scripts/                command-line entry points (see below)
configs/                NEAT hyperparameter files
slurm/                  Slurm job files for vLLM servers, drivers, image pulls and evaluation
tests/                  pytest suite (fake LLM clients, no network)
results/                reports, figures, the evolved graphs and the run log (results/README.md)
agent2/                 the agent runtime ACE is built on
```

## Setup

```
uv sync                    # runtime and tests
uv sync --group analysis   # adds numpy, scipy, statsmodels, pandas, matplotlib, datasets for the analysis scripts
```

Python 3.12 or newer. Runs against vLLM servers need no API key. Runs against a hosted model read the key from the environment or from `.env` at the project root (`cp .env.example .env`). `.env` is git-ignored; never commit a key.

## Tests

```
uv run pytest -q
```

## Running a single graph

```
uv run python scripts/run_graph.py path/to/project "Fix the off-by-one error in pagination"
uv run python scripts/run_graph.py path/to/project "..." --graph results/graphs/neat_best.json --print-views
```

Without `--graph` the graph is a single editor. `--graph` takes a graph JSON or a genome file such as `best_genome.json`. The script prints each worker's view size, edited paths and report, then the final patch. `--model`, `--api-base` and `--api-key-env` select the endpoint.

## Running an evolution

A dry run with a mock evaluator, no network:

```
uv run python scripts/run_neat.py --mock --generations 5 --population 12
```

Evolution on SWE-style tasks with the patch-similarity fitness, against vLLM endpoints registered under `--endpoints-dir`:

```
uv run python scripts/run_neat_swe.py --config configs/neat_swe_b.toml --pool data/train_pool.jsonl --history-dir runs/neat_b --batch-size 8
uv run python scripts/run_neat_swe.py --config configs/neat_swe_b.toml --pool data/train_pool.jsonl --history-dir runs/neat_b --resume runs/neat_b/population_state.json
```

The history directory gets `genomes.jsonl`, `species.jsonl`, `generations.jsonl`, `evaluations.jsonl`, `innovations.json`, `population_state.json` (saved every generation), `best_genome.json` and the worker rollouts. Generic tasks with a custom scorer go through `scripts/run_neat.py --tasks tasks.json` instead.

## SWE-bench pipeline

Data preparation, in order: `scripts/swebench_select_candidates.py` (filter SWE-Fixer-Train), `scripts/swebench_materialize.py` (check out base commits), `scripts/swebench_finalize.py` (write the training pool, validation and test task files). `slurm/pull_images.sbatch` pulls the sandbox images and `slurm/vllm_server.sbatch` starts a model server.

```
uv run python scripts/select_genomes.py runs/neat_a runs/neat_b --top 2 --out runs/selection
uv run python scripts/run_swe.py --mode graph --graph runs/selection/<candidate>.json --instances data/train_val40.jsonl --out runs/val/<candidate>/preds.jsonl
uv run python scripts/run_swe.py --mode baseline --instances data/verified_tasks.jsonl --max-steps 30 --out runs/verified/baseline/preds.jsonl --shard 0/12
uv run python scripts/run_swe.py --mode graph --graph results/graphs/neat_best.json --instances data/verified_tasks.jsonl --out runs/verified/neat_best/preds.jsonl --shard 0/12
uv run python scripts/swebench_eval.py --predictions runs/verified/neat_best/preds.jsonl --out runs/verified/neat_best/eval.jsonl
```

`run_swe.py --mode baseline` is the naive single agent; `--mode graph` runs a worker graph. `swebench_eval.py` runs patches in the Apptainer sandbox (`slurm/swebench_eval.sbatch` wraps it as an array job).

## Analysis

```
uv run --group analysis python scripts/analyze.py --system baseline=runs/verified/baseline --system neat=runs/verified/neat_best --neat-runs runs/neat_a runs/neat_b --out results/analysis
uv run --group analysis python scripts/report_stats.py
uv run --group analysis python scripts/graph_analysis.py --runs runs/neat_a runs/neat_b --out results/graph_analysis
```

`analyze.py` needs the per-instance prediction and evaluation files; `report_stats.py` works from the committed JSON in `results/` alone.

## Web UI

```
uv run python scripts/run_webui.py --runs-root runs --port 8766
```

Opens a browser view of every NEAT run directory under `--runs-root`: generations, species, lineages, per-task scores and rollouts. Live runs are picked up while they are written.

## Data not in the repository

Per-instance predictions, rollouts, evaluation logs, sandbox images, repository snapshots and NEAT histories are too large for git and stay on the cluster under `/usr/project/xtmp/kg349/ace/`. Everything derived from them that the report uses is under `results/`.
