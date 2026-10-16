# SWE-bench experiment: evolved worker graph vs. agent2 baseline

**Question.** Which agentic pipeline gives the best performance for its token cost? **Hypothesis.** The NEAT-optimized ACE graph resolves more SWE-bench Verified instances than the naive agent2 single-agent pipeline (H1: rate(NEAT) − rate(baseline) > 0).

**Model.** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` served by vLLM 0.26 on A5000s through Slurm. Thinking is disabled and the default samplers are used. Both systems run with `max_tokens=4096` per call, the same task prompt (`ace.swebench.data.make_prompt`), the same graceful step limit and the same 8000-character directory listing.

Reports, figures, the winning graphs and the run log are in `results/` (see `results/README.md`). Bulky state (predictions, rollouts, images, NEAT histories) stays under `/usr/project/xtmp/kg349/ace/` (`$ACE` below). `/home` has an inode quota.

## Pipeline

| Step | What | Command |
|---|---|---|
| 1 | Data: the training pool (894 SWE-Fixer-Train instances, 224 repos, .py-only gold), a validation set from 27 disjoint repos (`train_val40.jsonl`), Verified tasks and repo snapshots | `scripts/swebench_materialize.py`, `ace.swebench.data` |
| 2 | Sandbox images: 500 Verified SIFs | `slurm/pull_images.sbatch` |
| 3 | vLLM fleet: each server runs PP=2 on 2×A5000 (about 508K KV tokens), registers itself in `$ACE/endpoints/*.json` and exits after 20 idle minutes | `sbatch slurm/vllm_server.sbatch` (one per server) |
| 4 | NEAT evolution on train minibatches, with fitness = patch-similarity proxy against gold, elites re-scored every generation (running mean), and two runs (A: pop 16 / batch 6, B: pop 12 / batch 8) | `sbatch --job-name=ace-neat-b slurm/driver.sbatch scripts/run_neat_swe.py --config configs/neat_swe_b.toml --pool $ACE/data/train_pool.jsonl --history-dir $ACE/runs/neat_b --batch-size 8` |
| 5 | Selection: the top genomes by shrunk-mean training proxy are scored on the held-out val40; the pre-registered rule takes the highest val mean | `scripts/select_genomes.py`, then `scripts/run_swe.py --mode graph --instances $ACE/data/train_val40.jsonl ...` |
| 6 | Test: baseline and winner on all 500 Verified instances, sharded | `scripts/run_swe.py --mode {baseline,graph} --instances $ACE/data/verified_tasks.jsonl --shard k/12 ...` |
| 7 | Execution eval in the sandbox | `sbatch --array=0-3 slurm/swebench_eval.sbatch <all_preds.jsonl> <eval.jsonl>` |
| 8 | Statistics | `scripts/analyze.py --system baseline=$ACE/runs/verified/baseline_r1 --system neat=$ACE/runs/verified/neat_best --neat-runs $ACE/runs/neat_a $ACE/runs/neat_b --out results/analysis_v2 [--ids-file results/gold_validation/gold_resolvable_ids.txt]` |

The driver jobs are CPU-only. They run on `--partition=compsci-gpu --exclude=compsci-cluster-fitz-01` because the CPU partition is contended.

## Isolation

Model patches are evaluated in Apptainer with `--containall --cleanenv --no-home --fakeroot --network none`, a per-run overlay and a read-only input bind. The harness never executes agent code on the host. Agents themselves only edit in-memory workspaces; there are no shell tools. A malicious-patch check (`$ACE/runs/isolation_check/`) confirmed that a patch cannot reach the host filesystem or the network.

Gold patches resolve on 464 of 500 instances in this sandbox. The 36 deterministic failures need the network or a host feature (matplotlib ghostscript, requests/httpbin, sphinx linkcheck, a few env issues). Results are therefore reported on all 500 and on the 464-instance gold-resolvable subset.

## Throughput

- **Prefix caching.** Admission control (`ace.serving`, live caps in `$ACE/admission.json`) limits the active conversations per endpoint. Without it, hit rates dropped to about 0% from thrashing; with it they are about 85%.
- **Driver sharding.** Python drivers are GIL-bound on large repositories, so every run is split into 8–12 shard processes with 4 CPUs each.
- **Session reuse.** Training-repo sessions are frozen and forked per rollout (`ace.sessions`).

Interventions made during the runs are logged in `results/INTERVENTIONS.md`.

## Statistics (`scripts/analyze.py`)

- Wilson 95% CIs on resolve rates
- Exact McNemar test on paired outcomes (one-sided for H1)
- Paired bootstrap CI of the rate difference
- GEE logistic regression `resolved ~ system + difficulty`, clustered by instance
- Wilcoxon signed-rank test on per-instance tokens, plus resolved per million tokens
- Proxy validity (AUC and point-biserial correlation of the training proxy against execution resolution)
- NEAT learning curves with a Spearman trend test
- A second independent run of each system (`baseline_r2`, `neat_best_r2`) to estimate run-to-run variance

## Results (updated 2026-10-16)

| System | Resolved /500 | Rate (Wilson 95%) | Gold-resolvable /464 | Median tokens | Resolved per Mtok |
|---|---|---|---|---|---|
| **baseline, no step limit** (primary reference) | 43 | 8.6% (6.4–11.4) | 43 (9.3%) | 701K | 0.03 |
| baseline r1 (30 steps) | 37 | 7.4% (5.4–10.0) | 37 (8.0%) | 362K | 0.19 |
| baseline r2 (30 steps) | 36 | 7.2% (5.2–9.8) | 36 (7.8%) | 368K | 0.18 |
| NEAT best r1 | 124 | 24.8% (21.2–28.8) | 124 (26.7%) | 440K | 0.48 |
| NEAT best r2 | 117 | 23.4% (19.9–27.3) | 117 (25.2%) | 349K | 0.49 |

- **NEAT vs the no-step-limit baseline (primary).** The 30-step limit cut off 255/500 baseline runs, so the baseline was re-run with `max_steps=500`, which leaves the 64K context as the only bound. NEAT r1 scores +16.2 pts (bootstrap 95% +12.2..+20.4). McNemar on 19 vs 100 discordant pairs: p < 1e-4. GEE odds ratio 3.65 (2.60–5.13, p = 8e-14). NEAT r2 scores +14.8 pts (+10.8..+18.6), with odds ratio 3.40 (2.41–4.80). NEAT uses 0.61× the tokens per instance (median ratio; Wilcoxon p < 1e-20).
- **The step limit was not what held the baseline back.** The no-limit baseline resolves 43 vs 37/36 (+1.2 to +1.4 pts, McNemar p ≈ 0.4, n.s.). It uses about 2.4× the tokens (median 701K; 1.48B in total vs 0.19B), and 315/500 runs still end with an empty patch. Extra steps mostly go into exploration that never produces an edit.
- **NEAT r1 vs baseline r1.** +17.4 pts (paired bootstrap 95% +13.4..+21.4). Exact McNemar on 16 vs 103 discordant pairs: p < 1e-4. GEE odds ratio adjusted for difficulty: 4.3 (95% CI 3.0–6.2, p = 1.6e-15).
- **NEAT r2 vs baseline r1.** +16.0 pts (+12.2..+20.0), on 15 vs 95 discordant pairs.
- **Baseline r2 vs baseline r1.** −0.2 pts (−2.4..+2.0), McNemar p = 1.0. Run-to-run noise is small next to the effect.
- **Tokens.** The NEAT graph uses about 1.4–1.5× the baseline's tokens per instance (Wilcoxon p < 1e-5), but resolves about 2.5× more per million tokens.
- **Proxy validity.** The training proxy (patch similarity to gold) predicts execution resolution with AUC 0.88.

The winning graph is `results/graphs/neat_best.json`. It runs two independent editors (one of them after a viewer report) and a pick-merger chooses between their edits.

Full reports: `results/analysis_v2/report.md` (all 500, all five systems) and `results/analysis_v2_gold/report.md` (gold-resolvable subset). The older two-system reports (30-step baseline only) are in `results/analysis_v1/` and `results/analysis_v1_gold/`. The secondary analysis of evolved graphs is `results/graph_analysis/report.md` (`scripts/graph_analysis.py`).

Serving note: the no-limit baseline's 35–64K contexts thrashed the prefix cache (hit rate dropped from about 85% to 0.2%) until there were about 9 conversations per server, which is roughly what fits in the 508K-token KV cache. See `results/INTERVENTIONS.md`.

### Hand-built nested tree (val40)

NEAT's merger insertion only builds ladders (`_merge` always goes in front of END), so a balanced tree is unreachable by mutation. A hand-built tree was tested on val40: 4 chained editors, a PICKMERGER over editors 1–2 and another over 3–4, and a JUDGEMERGER over the two (`results/graphs/nested_tree.json`). It scored a proxy of 0.326 (11/40 failures, median 918K tokens) against the winner's 0.316 (10/40 failures, 403K tokens). The paired Wilcoxon p is 0.46 (12 wins, 20 ties, 8 losses). The tree costs about 2.3× the tokens for no measurable gain.
