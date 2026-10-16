"""Combine candidates + materialization results into pool/val/verified task files."""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ace.swebench.proxy import patch_similarity  # noqa: E402

POOL_KEYS = ("instance_id", "repo", "base_commit", "problem_statement", "patch", "gold_files", "project_dir")
VER_KEYS = ("instance_id", "repo", "base_commit", "problem_statement", "project_dir")
TRIVIAL = {"setup.py", "version.py", "_version.py", "__version__.py"}


def write_atomic(path, rows, keys):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for c in rows:
            f.write(json.dumps({k: c[k] for k in keys}) + "\n")
    tmp.replace(path)


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/usr/project/xtmp/kg349/ace/data")
    ap.add_argument("--per-repo", type=int, default=5)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.data)
    stats = collections.Counter()

    # --- training pool ---
    cands = {c["instance_id"]: c for c in read(d / "train_candidates.jsonl")}
    excl_path = d / "train_excluded_repos.json"  # docs/translation/non-Python repos (manual)
    excluded = set(json.load(open(excl_path))) if excl_path.exists() else set()
    by_repo = collections.defaultdict(list)
    for r in read(d / "train_materialize_results.jsonl"):
        if r["status"] != "ok":
            continue
        c = dict(cands[r["instance_id"]], project_dir=r["project_dir"])
        if c["repo"] in excluded:
            stats["dropped_excluded_repo"] += 1
            continue
        if not all(f.endswith(".py") for f in c["gold_files"]):
            stats["dropped_non_py_gold"] += 1
            continue
        if all(f.split("/")[-1] in TRIVIAL for f in c["gold_files"]):
            stats["dropped_trivial_version_only"] += 1
            continue
        by_repo[c["repo"]].append(c)
    rng = random.Random(a.seed)
    insts = []
    for repo in sorted(by_repo):
        kept = []
        for c in sorted(by_repo[repo], key=lambda c: c["instance_id"]):
            if len(kept) >= a.per_repo:
                break
            if any(" ".join(k["problem_statement"].split())[:200] == " ".join(c["problem_statement"].split())[:200]
                   or patch_similarity(c["patch"], k["patch"]) > 0.8 for k in kept):
                stats["dropped_near_duplicate"] += 1
                continue
            kept.append(c)
        insts.extend(kept)
    # repo-disjoint validation split
    repos = sorted({c["repo"] for c in insts})
    rng.shuffle(repos)
    val_repos, n = set(), 0
    for r in repos:
        if n >= a.val_size:
            break
        val_repos.add(r)
        n += sum(c["repo"] == r for c in insts)
    pool = [c for c in insts if c["repo"] not in val_repos]
    val = [c for c in insts if c["repo"] in val_repos]
    for name, rows in (("train_pool.jsonl", pool), ("train_val.jsonl", val)):
        write_atomic(d / name, rows, POOL_KEYS)
        stats[name] = len(rows)
        stats[name + "_repos"] = len({c["repo"] for c in rows})

    # --- verified tasks ---
    ok = {r["instance_id"]: r for p in sorted(d.glob("verified_materialize_results*.jsonl"))
          for r in read(p) if r["status"] == "ok"}
    vrows = []
    for v in read(d / "verified_full.jsonl"):
        if v["instance_id"] in ok:
            vrows.append(dict(v, project_dir=ok[v["instance_id"]]["project_dir"]))
        else:
            stats["verified_missing"] += 1
    write_atomic(d / "verified_tasks.jsonl", vrows, VER_KEYS)
    stats["verified_tasks"] = len(vrows)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
