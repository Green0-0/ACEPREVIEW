# ACE

ACE builds coding pipelines out of LLM workers and evolves them. A worker graph is a typed DAG of workers (viewers, list viewers, editors, view-editors, critics, judge-mergers and pick-mergers) that pass views of the code, edits and reports to each other and end in exactly one proposed edit of a repository. Every worker reads the same pristine repository and edits a private workspace, so a graph run never changes the project on disk: its result is an `Edit` and a git patch.

The evolution side (`ace.neat`) applies NEAT to these graphs. Genomes encode workers and typed connections with global innovation numbers; mutation uses a typed insertion table that only produces valid graphs, crossover repairs or falls back, and speciation protects new structures. Worker settings (role, strategy, instructions, context budgets, tool sets, edit format) are categorical genes. Fitness comes from running each graph on a set of tasks with a scorer, or from a deterministic structural mock for dry runs.

## Layout

```
src/ace/graph/        worker-graph runtime: types, graph, views, rendering, edits, ACE format,
                      codebase graph, view recording, properties, prompts, LLM client, runner
src/ace/graph/workers/  one module per worker type, plus the Worker base class
src/ace/neat/         config, genome, innovation DB, mutation, crossover, speciation,
                      population, history, evaluation
scripts/run_graph.py  run one worker graph on a project and print the diff
scripts/run_neat.py   evolve worker graphs
tests/graph_tests/    graph runtime tests (fake LLM clients, no network)
tests/neat_tests/     evolution tests
agent2/               the attached agent2 library (its own git repository)
ACE_SCHEMA.md         how ACE works
GRAPH_NEAT_DESIGN.md  the interface contract for ace.graph and ace.neat
```

## Setup

```
uv sync
cp .env.example .env    # then set ANTHROPIC_API_KEY in .env
```

The scripts read the key from the environment, then from `.env` at the project root, then from `agent2/.env`, so a key already kept in `agent2/.env` also works. Both `.env` files are git-ignored. Never commit a key.

## Tests

```
uv run pytest -q
```

The suite runs with fake LLM clients and needs no network or API key.

## Running a Single Graph

```
uv run python scripts/run_graph.py path/to/project "Fix the off-by-one error in pagination"
uv run python scripts/run_graph.py path/to/project "..." --graph graph.json --print-views
```

Without `--graph` the graph is one EDITOR feeding END. A graph file is `WorkerGraph.to_json()` output, for example the `"graph"` field of a `best_genome.json`. The script prints each worker's rollouts, view size, edited paths and report excerpt, then the final git patch. Project files are never modified; worker rollouts go to `<project>/.agent2/task` unless `--rollout-dir` is given. `--model`, `--api-base` and `--api-key-env` select the endpoint.

## Running an Evolution

A dry run with the structural mock evaluator, no network:

```
uv run python scripts/run_neat.py --mock --generations 5 --population 12
```

A real run needs a tasks file, a JSON list of task objects:

```json
[
  {"project_dir": "projects/shop", "prompt": "Make the cart total include tax.", "scorer": "nonempty_diff"},
  {"project_dir": "/abs/path/to/repo", "prompt": "...", "scorer": "mypkg.scoring:run_tests"}
]
```

```
uv run python scripts/run_neat.py --tasks tasks.json --model claude-haiku-5-5 --history-dir runs/neat1
uv run python scripts/run_neat.py --tasks tasks.json --history-dir runs/neat1 --resume runs/neat1/population_state.json
```

Relative project dirs resolve against the tasks file. `scorer` is `nonempty_diff` or a `module:function` called as `scorer(edit, session)`; a SWE-bench scorer applies the edit and runs the instance's tests this way. `--config` takes a NEAT TOML file; `--generations`, `--population`, `--seed` and `--concurrency` override it.

## Where Output Goes

The history directory defaults to `.ace/neat` (relative to the working directory) and is git-ignored. It holds `genomes.jsonl`, `species.jsonl`, `generations.jsonl`, `innovations.json`, `population_state.json` (saved every generation), `best_genome.json`, and `rollouts/<genome id>/<task index>/` with the worker rollouts of every task run. Nothing is written into task project folders.

## agent2

`agent2/` is an attached library with its own git repository, installed as an editable path dependency. ACE uses its Session, Workspace, Agent, tools and code parser without modifying them. Commit changes to agent2 in that repository, not in this one. Its architecture is described in `agent2/CURRENT_SCHEMA.md`.

## Documentation

- `ACE_SCHEMA.md`: how ACE works, including known limitations.
- `GRAPH_NEAT_DESIGN.md`: the binding interface contract for `ace.graph` and `ace.neat`.
