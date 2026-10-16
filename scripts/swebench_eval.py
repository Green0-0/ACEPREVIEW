#!/usr/bin/env python
"""Evaluate SWE-bench Verified predictions in the Apptainer sandbox.

    python scripts/swebench_eval.py --predictions preds.jsonl --out results.jsonl \
        [--workers 10] [--instances ID ...] [--shard 0/4] [--timeout 1800]

preds.jsonl lines: {"instance_id", "model_patch", optional "run", "model_name_or_path"}.
Special value --predictions gold (or empty) evaluates the dataset's gold patch
(or an empty patch) for every selected instance.

Resumable: (instance_id, run) pairs already in --out are skipped, except rows
with status "error" (harness failure, e.g. missing image), which are retried;
the summary uses the latest row per pair. Each result
line is the harness result plus "run"/"model_name_or_path"; a summary is
printed and written to <out>.summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ace.swebench.harness import load_verified, run_instance, sif_path  # noqa: E402


def key(iid, run):
    return f"{iid}\t{run if run is not None else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="JSONL path, or 'gold' / 'empty'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) // 3 or 1)
    ap.add_argument("--instances", nargs="*", help="restrict to these instance ids")
    ap.add_argument("--instances-file", help="file with one instance id per line")
    ap.add_argument("--shard", help="i/n: evaluate only the i-th of n interleaved shards")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--log-dir", help="default: <out dir>/logs")
    ap.add_argument("--skip-empty", action="store_true",
                    help="mark empty patches unresolved without running tests")
    ap.add_argument("--network-repos", default="",
                    help="comma-separated repos (e.g. psf/requests) whose containers get host network; "
                         "default: none (all runs offline)")
    ap.add_argument("--summary-only", action="store_true",
                    help="only (re)write <out>.summary.json from existing results")
    args = ap.parse_args()

    dataset = {r["instance_id"]: r for r in load_verified()}
    if args.predictions in ("gold", "empty"):
        preds = [{"instance_id": i, "model_patch": r["patch"] if args.predictions == "gold" else "",
                  "model_name_or_path": args.predictions} for i, r in dataset.items()]
    else:
        with open(args.predictions) as f:
            preds = [json.loads(line) for line in f if line.strip()]

    wanted = set(args.instances or [])
    if args.instances_file:
        wanted |= {l.strip() for l in open(args.instances_file) if l.strip()}
    if wanted:
        preds = [p for p in preds if p["instance_id"] in wanted]
    unknown = [p["instance_id"] for p in preds if p["instance_id"] not in dataset]
    if unknown:
        print(f"WARNING: {len(unknown)} predictions for unknown instances skipped, e.g. {unknown[:3]}")
        preds = [p for p in preds if p["instance_id"] in dataset]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        preds = preds[i::n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out.parent / "logs"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
                if r.get("status") != "error":  # harness errors are retried
                    done.add(key(r["instance_id"], r.get("run")))
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [p for p in preds if key(p["instance_id"], p.get("run")) not in done]
    if args.summary_only:
        todo = []
    print(f"{len(preds)} predictions, {len(preds) - len(todo)} already done, {len(todo)} to run, "
          f"workers={args.workers}", flush=True)

    net_repos = {r for r in args.network_repos.split(",") if r}
    lock = threading.Lock()
    counts = Counter()

    def work(p):
        iid, run = p["instance_id"], p.get("run")
        patch = p.get("model_patch") or ""
        if args.skip_empty and not patch.strip():
            r = {"instance_id": iid, "resolved": False, "status": "unresolved", "tests_status": {},
                 "duration_s": 0.0, "log_path": None, "note": "empty patch, not run"}
        else:
            tag = f"run{run}" if run is not None else ""
            r = run_instance(dataset[iid], patch, timeout=args.timeout,
                             network=dataset[iid]["repo"] in net_repos,
                             log_dir=log_dir / str(p.get("model_name_or_path", "model")).replace("/", "_"), tag=tag)
        r["run"] = run
        r["model_name_or_path"] = p.get("model_name_or_path")
        return r

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, open(out, "a") as fout:
        futs = [ex.submit(work, p) for p in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                counts[r["status"]] += 1
            print(f"[{n}/{len(todo)}] {r['instance_id']} {r['status']} {r['duration_s']}s"
                  + (f" ({r['error']})" if r.get("error") else ""), flush=True)

    # Summary over everything in --out (including earlier resumed runs).
    last = {}
    for l in open(out):
        if l.strip():
            r = json.loads(l)
            last[key(r["instance_id"], r.get("run"))] = r  # latest attempt wins
    rows = list(last.values())
    status = Counter(r["status"] for r in rows)
    summary = {
        "n": len(rows),
        "resolved": sum(r["resolved"] for r in rows),
        "resolve_rate": round(sum(r["resolved"] for r in rows) / max(1, len(rows)), 4),
        "status": dict(status),
        "mean_duration_s": round(sum(r["duration_s"] for r in rows) / max(1, len(rows)), 1),
        "missing_images": sorted({r["instance_id"] for r in rows if "missing image" in (r.get("error") or "")})[:50],
        "not_resolved_ids": sorted(r["instance_id"] for r in rows if not r["resolved"]),
    }
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "not_resolved_ids"}, indent=1))


if __name__ == "__main__":
    main()
