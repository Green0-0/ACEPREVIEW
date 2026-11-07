# ACE: Discovering Agentic Pipelines with NEAT

ACE builds coding pipelines out of LLM workers and evolves their structure with NEAT. A pipeline is a typed directed acyclic graph of workers (viewers, list viewers, editors, view-editors, critics, judge-mergers and pick-mergers) whose edges carry views of the code, edits and reports, and which ends in exactly one proposed edit of a repository. Every worker reads the same pristine repository and edits a private workspace, so a run never changes the project on disk: its result is a git patch.

## The problem

Coding agents are harnesses: a program that gives an LLM tools and prompts and loops until the task is done. The good ones are designed by hand, and nothing says that a hand-built harness is close to the best one for a given model and task. Prior automation either tunes prompts, searches over workflows written as code, or optimises the wiring between a fixed set of agents. ACE asks a narrower question: if the pipeline is a graph of LLM workers, can an evolutionary algorithm find a better graph than a person would write? NEAT (NeuroEvolution of Augmenting Topologies) was built to grow network topologies and their parameters together, so it is the natural fit.

The task is resolving GitHub issues: the input is a user's informal description of a bug, the output is a patch. Everything in the repository is organised around that experiment, but the graph runtime and NEAT are generic: `scripts/run_neat.py --tasks` evolves graphs on any task with a scorer.

## How the system works

### Worker graphs

A pipeline is a directed acyclic graph of workers with one distinguished `END` node. Each edge carries one of three typed payloads:

- a **view**: what code a worker looked at, recorded per file and per code element (function, class, method) at either collapsed (signature only) or full detail, always expressed against the pristine repository;
- an **edit**: a proposed change, held as a private workspace overlay and exportable as a git patch;
- a **report**: the worker's free-text summary of what it found or did.

There are seven worker types in three families:

| Family | Type | Does | Inputs | Outputs |
|---|---|---|---|---|
| Readers | viewer | explores the repository with search and view tools | none | view, report |
| | list viewer | picks locations from a ranked candidate list and reads them, no tools | view | view, report |
| Writers | editor | a tool-using engineer with a private workspace | views, reports (optional) | view, edit, report |
| | view-editor | writes an edit directly from the code it is shown, in one tool-less step | views, reports | view, edit |
| | critic | rewrites one edit it is given, in one tool-less step | views, one edit, reports | view, edit |
| Mergers | judge-merger | reads several candidate edits and forwards one of them unchanged | views, two or more edits, reports | view, edit |
| | pick-merger | reads several candidate edits and writes a new combined edit | views, two or more edits, reports | view, edit |

Exactly one edit edge reaches `END`, and that edit is the answer. Tool-less workers (view-editor, critic, pick-merger) write their edits in ACE, a fenced-code-block format where the first line names the target: a whole file, a line range, or a code element by path. Addressing code by element rather than by line lets a model that saw a function once rewrite it without re-reading the file, and lets the runtime validate the target before applying anything.

Every worker gets its own copy of the original repository and never sees another worker's workspace. Views, edits and reports are the only way information moves. Views are forwarded as "what the parent saw": the receiving worker gets the same files and elements at the same detail, rendered within its own context budget. This keeps the prompt of every worker reproducible from the graph alone and makes the graph the whole description of the pipeline.

Each worker also carries a handful of categorical properties: which role and strategy text goes into its system prompt, how the task is framed, how many characters of views it is offered, its step limit, which tool set it has, and which ACE target forms it may use. Every property is an index into a fixed option table in `src/ace/graph/properties.py` and `src/ace/graph/prompts.py`.

### Running a graph

`GraphRunner` topologically sorts the graph, runs each worker as an agent2 agent on an isolated workspace, and passes the typed outputs along the edges. Nothing is written into the project: the only disk output is the rollout JSON every agent saves after each model step, which is what the web UI and the demo page read. A worker that raises (a parse failure, an empty report, a step limit) ends the run with a `GraphFailure` naming the node.

### Evolving graphs with NEAT

A genome is the graph plus its properties, with every node and connection tagged by an innovation number from a global database, so the same structural event gets the same number in every genome that performs it. That is what lets crossover align two parents gene by gene and lets speciation measure how far apart two genomes are. The loop in `src/ace/neat/population.py` is the classic one: evaluate, speciate, share fitness within species, allocate offspring per species, keep elites, breed by crossover and mutation, prune species that stop improving.

Three things differ from textbook NEAT:

- **Typed mutation.** The "add node" operator is replaced by a typed insertion table. A mutation splits a view edge with a list viewer, splits an edit edge with a critic, adds a viewer feeding an existing worker, branches a new editor off an existing view, or adds a new writer whose edit is combined with the current output by a newly inserted merger. Every proposal is validated against the slot rules in the table above and discarded if it fails, so every genome in the population is runnable. Property mutation picks another option from the table; crossover takes each gene from one parent.
- **Minibatch fitness.** Running a graph on a real repository is expensive, so each generation is scored on a fresh minibatch of training issues, and elites are re-scored every generation and keep a running mean. Fitness within a generation is comparable, fitness across generations is not, which is why fitness curves in the web UI track the batch difficulty rather than trend upward.
- **A proxy reward.** Instead of running tests, a patch is scored by similarity to the human-written gold patch: half for touching the right files, half for the text overlap of the changed lines. It is execution-free and takes milliseconds. Across the final experiment it predicted actual resolution with an AUC of 0.89.

Candidates are picked from each run by the training mean shrunk toward the run mean, re-scored on a held-out validation set, and the best one goes to the test set.

## Design choices and why

- **The model is fixed and not a gene.** The question is which pipeline structure works for a given model, so the model, its thinking level and its sampling settings are held constant across every worker and every genome. Letting the search also pick models would confound structure with model quality and make results depend on whichever model happened to be strongest.
- **No LLM-driven mutation.** Mutations only change graph structure and pick from fixed option tables; no model rewrites prompts or proposes graphs. This keeps the search space enumerable, keeps every genome reproducible from its JSON, and keeps the method about structure rather than about prompt optimisation, which prior work already covers.
- **Typed edges and slot rules.** Because every mutation is validated against the slot table, the search never wastes evaluations on graphs that cannot run. The types also carry meaning: a critic can only sit on an edit edge, a merger must have two or more edits, which is how the search discovers patterns like "two editors feeding a merger" rather than arbitrary wiring.
- **Private workspaces and forwarded views.** Workers never share a workspace, so a graph has no hidden state: everything a worker knows arrived over an edge. This makes rollouts reproducible, makes the diff the only side effect, and means a graph can be read as a description of information flow.
- **Cost is node count only.** Fitness is penalised by `node_cost` per worker and nothing else. Token and wall-clock costs were left out of fitness because they vary with the batch and the endpoint load, which would make them noise rather than signal. The penalty is there to stop graphs growing without benefit; it is not a claim that small graphs are better, and the winner was found by comparing candidates on the validation set, not by size.
- **No fallbacks inside the graph.** A failed worker makes the task score zero and the genome is removed only by selection. Transient API errors are retried in the LLM client, which is infrastructure, but a graph that fails for structural reasons is supposed to look bad. Adding retries or fallbacks at the graph level would hide exactly the failures the search should see.
- **Speciation on, with raw gene counts.** Speciation protects new structures long enough to be evaluated, and the target species count is a hyperparameter with the threshold adjusted toward it each generation. The classic normalisation by genome size was removed: worker graphs have at most a few dozen genes, and the switch at twenty genes shrank every distance and collapsed the population into one species.
- **A proxy reward rather than test execution.** Sandboxed test runs cost minutes per instance and need an image per repository. The similarity proxy makes evolution affordable, and its validity is checked separately: the final evaluation runs the real tests in a sandbox, so any overfitting to the proxy would show up as a weaker test result, not a stronger one.
- **A naive agent as the baseline.** The comparison is against a single agent with the same model, tools and prompt, at a 30 step limit and with no limit. That isolates the effect of the pipeline structure from the model and the tooling.

## What the experiment found

The experiment evolved pipelines for Gemma4 26B on SWE-style training issues and tested the selected graph on SWE-bench Verified. The full numbers, figures and statistical tests are in `results/` and in the report; the headline is that the evolved pipeline resolved 124 of 500 instances against 43 for the naive agent with no step limit, with a one-sided McNemar p of 9e-15, while using fewer tokens per instance. The selected graph (`results/graphs/neat_best.json`) is two editors, the first passing its view and report to the second, and a pick-merger that reads both edits and writes the final one. The pattern echoes self-consistency and mixture-of-agents results: parallel attempts aggregated by a merger beat one long attempt. Both evolution runs are in `runs/` and can be browsed in the web UI.

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
runs/                   the two NEAT run histories, the selected candidates and their validation runs
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

Every genome page has a **Run demo** button that opens the genome's demo page: type a prompt and the graph runs it against `agent2/smoke_repos/schedule` (a small Python library) with `claude-haiku-5-5`, thinking off. The run is followed live: the graph colours each worker by status, clicking a worker shows its transcript (refreshed after every model step), the diff it produced, its details and resolved properties, and the view it forwarded; the bottom panel shows the output diff, the reports and a log. The project is never modified; workers edit private workspaces and only their rollouts are written, into a temporary folder. Previous demos of a genome are listed on its demo page for as long as the server runs. This needs `ANTHROPIC_API_KEY` in the environment or in `.env`. `--demo-project` picks another folder, `--demo-model` another model, `--no-demo` turns it off. The command above works from a clone: `runs/neat_a` and `runs/neat_b` are the two evolution runs from the report with their complete histories (`genomes.jsonl`, `generations.jsonl`, `species.jsonl`, `evaluations.jsonl`, `events.jsonl`, `innovations.json`, `population_state.json`, `best_genome.json`). Worker rollouts are kept only for the four genomes that went to validation (`runs/selection/`), including the winner `neat_b_0_37b607a6`, so the trajectory viewer opens for those and reports "not found" for the rest. `runs/val/` holds the validation predictions of each candidate, the naive baseline and the hand-written graph.

## Data not in the repository

The SWE-bench Verified predictions and evaluation logs, the remaining worker rollouts, sandbox images and repository snapshots are too large for git and stay on the cluster under `/usr/project/xtmp/kg349/ace/`. Everything derived from them that the report uses is under `results/`.
