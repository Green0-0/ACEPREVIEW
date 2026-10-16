"""SWE-style task data: pool loading, per-generation minibatch sampling, prompt wrapper.

Pool files (jsonl) live under /usr/project/xtmp/kg349/ace/data/:
  train_pool.jsonl     minibatch pool (SWE-Fixer-Train-110K subset, gold patch included)
  train_val.jsonl      held-out proxy validation set (same schema; never minibatched)
  verified_tasks.jsonl SWE-bench Verified tasks as agents see them (no gold/test info)

Each line: instance_id, repo, base_commit, problem_statement, project_dir
(+ patch, gold_files for training files).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

__all__ = ["DATA_DIR", "PROMPT_TEMPLATE", "load_pool", "sample_minibatch", "make_prompt"]

DATA_DIR = Path("/usr/project/xtmp/kg349/ace/data")

PROMPT_TEMPLATE = (
    "Resolve the following GitHub issue in this repository by editing the source code. "
    "Do not modify or add tests.\n\n<issue>\n{issue}\n</issue>"
)


def load_pool(path: str | Path) -> list[dict]:
    """Read a jsonl pool file into a list of instance dicts (blank lines skipped)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_minibatch(pool: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Sample ``size`` distinct instances, avoiding repeated repos when possible.

    Repos are visited in random order, one instance each per round, so a
    minibatch holds at most one instance per repo unless ``size`` exceeds the
    number of repos.  Deterministic given ``rng`` state and pool order.
    """
    if size <= 0:
        return []
    if size >= len(pool):
        out = list(pool)
        rng.shuffle(out)
        return out
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for inst in pool:
        by_repo[inst.get("repo", inst["instance_id"])].append(inst)
    queues = {}
    for repo in sorted(by_repo):
        items = list(by_repo[repo])
        rng.shuffle(items)
        queues[repo] = items
    order = sorted(queues)
    out: list[dict] = []
    while len(out) < size:
        rng.shuffle(order)
        for repo in order:
            if queues[repo]:
                out.append(queues[repo].pop())
                if len(out) == size:
                    break
        order = [r for r in order if queues[r]]
    return out


def make_prompt(problem_statement: str) -> str:
    """Task prompt given to every agent graph (baseline and evolved alike)."""
    return PROMPT_TEMPLATE.format(issue=problem_statement.strip())
