"""Materialize instances as plain-file snapshots (no .git) at base_commit.

Input: jsonl with instance_id, repo, base_commit, patch.  For each repo (in
parallel across repos), a bare repo under --git-root holds fetched commits;
each instance is checked out with a private index into <out-root>/<instance_id>/,
then the gold patch is checked with `git apply --check`.

Modes:
  --shallow   (training) per-commit `fetch --depth 1`; a tree-only probe of the
              first commit enforces --max-files before downloading blobs, and
              ls-tree sizes enforce --max-bytes before checkout.  Stops a repo
              after --max-ok successes.
  default     full bare clone (used for the 12 SWE-bench repos).

Writes one JSON result line per attempted instance to --results.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

BIN_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".tif", ".tiff", ".webp", ".pdf", ".zip", ".gz",
    ".tgz", ".bz2", ".xz", ".7z", ".tar", ".jar", ".whl", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".mp3", ".mp4", ".wav",
    ".avi", ".mov", ".ttf", ".otf", ".woff", ".woff2", ".eot", ".db", ".sqlite", ".parquet", ".pb",
    ".mat", ".fits", ".nc", ".dat", ".model", ".ckpt", ".safetensors", ".psd", ".svgz", ".class", ".o",
}
VENDOR_PARTS = {"vendor", "vendored", "_vendor", "third_party", "thirdparty", "node_modules", "site-packages", "external", "extern"}

ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="true", LC_ALL="C")


def git(args, cwd=None, timeout=900, env=None, input=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=timeout,
                          env=env or ENV, input=input)


def hidden(path: str) -> bool:
    return any(p.startswith(".") for p in path.split("/"))


def tree_stats(gdir, sha, with_sizes):
    args = ["--git-dir", gdir, "ls-tree", "-r", "-z"] + (["-l"] if with_sizes else ["--name-only"]) + [sha]
    r = git(args, timeout=300)
    if r.returncode:
        return None
    n = tot = binb = vend = 0
    for ent in r.stdout.split(b"\0"):
        if not ent:
            continue
        if with_sizes:
            meta, path = ent.split(b"\t", 1)
            parts = meta.split()
            size = int(parts[3]) if parts[3] != b"-" else 0
        else:
            path, size = ent, 0
        path = path.decode("utf-8", "surrogateescape")
        if hidden(path):
            continue
        n += 1
        tot += size
        if os.path.splitext(path)[1].lower() in BIN_EXT:
            binb += size
        if any(p.lower() in VENDOR_PARTS for p in path.split("/")[:-1]):
            vend += 1
    return {"files": n, "bytes": tot, "bin_bytes": binb, "vendored_files": vend}


def repo_ok(st, a):
    if st["files"] > a.max_files:
        return "too_many_files"
    if st["vendored_files"] > 0.3 * max(1, st["files"]):
        return "vendored"
    if a.max_bytes and st["bytes"] - st["bin_bytes"] > a.max_bytes:
        return "too_big"
    if a.max_bytes and st["bin_bytes"] > 0.5 * max(1, st["bytes"]) and st["bin_bytes"] > 5e6:
        return "binary_heavy"
    return None


def process_repo(job):
    repo, items, a = job
    url = f"https://github.com/{repo}.git"
    gdir = os.path.join(a.git_root, repo.replace("/", "__") + ".git")
    out_root = Path(a.out_root).resolve()
    results = []

    def rec(it, status, **kw):
        results.append({"instance_id": it["instance_id"], "repo": repo, "status": status, **kw})

    try:
        if not os.path.isdir(gdir):
            os.makedirs(a.git_root, exist_ok=True)
            if a.shallow:
                git(["init", "-q", "--bare", gdir])
            else:
                r = git(["clone", "-q", "--bare", url, gdir], timeout=3600)
                if r.returncode:
                    for it in items:
                        rec(it, "clone_failed", err=r.stderr.decode()[-300:])
                    return results
        if a.shallow:
            # tree-only probe of first commit to bound file count before downloading blobs
            probe = tempfile.mkdtemp(dir=a.git_root, prefix="probe_")
            try:
                git(["init", "-q", "--bare", probe])
                r = git(["-C", probe, "fetch", "-q", "--depth", "1", "--filter=blob:none", url, items[0]["base_commit"]], timeout=600)
                if r.returncode == 0:
                    st = tree_stats(probe, items[0]["base_commit"], False)
                    if st and st["files"] > a.max_files:
                        for it in items:
                            rec(it, "repo_too_many_files", files=st["files"])
                        return results
                    if st and st["vendored_files"] > 0.3 * max(1, st["files"]):
                        for it in items:
                            rec(it, "repo_vendored", files=st["files"])
                        return results
            finally:
                shutil.rmtree(probe, ignore_errors=True)
        n_ok = 0
        size_fail = 0
        for it in items:
            if a.max_ok and n_ok >= a.max_ok:
                break
            if size_fail >= 2:
                rec(it, "repo_size_skip")
                continue
            sha = it["base_commit"]
            dest = out_root / it["instance_id"]
            if git(["--git-dir", gdir, "cat-file", "-e", sha + "^{commit}"]).returncode:
                r = git(["--git-dir", gdir, "fetch", "-q", "--depth", "1", url, sha], timeout=1200)
                if r.returncode:
                    rec(it, "fetch_failed", err=r.stderr.decode()[-200:])
                    continue
            st = tree_stats(gdir, sha, True)
            if st is None:
                rec(it, "ls_tree_failed")
                continue
            bad = repo_ok(st, a) if a.shallow else None
            if bad:
                size_fail += 1
                rec(it, bad, **st)
                continue
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            idx = os.path.join(a.git_root, f"index_{os.getpid()}_{it['instance_id']}")
            env = dict(ENV, GIT_INDEX_FILE=idx)
            r = git(["--git-dir", gdir, "--work-tree", str(dest), "-c", "core.autocrlf=false",
                     "-c", "core.symlinks=true", "checkout", "-q", "-f", sha, "--", "."], env=env, timeout=1200)
            try:
                os.remove(idx)
            except OSError:
                pass
            if r.returncode:
                shutil.rmtree(dest, ignore_errors=True)
                rec(it, "checkout_failed", err=r.stderr.decode()[-200:])
                continue
            # verify gold patch applies (outside any git repo discovery)
            aenv = dict(ENV, GIT_CEILING_DIRECTORIES=str(out_root.parent))
            r = subprocess.run(["git", "apply", "--check", "-"], cwd=dest, input=it["patch"].encode("utf-8", "surrogateescape"),
                               capture_output=True, env=aenv, timeout=120)
            if r.returncode:
                shutil.rmtree(dest, ignore_errors=True)
                rec(it, "patch_does_not_apply", err=r.stderr.decode()[-200:])
                continue
            n_ok += 1
            rec(it, "ok", project_dir=str(dest), **st)
        return results
    except Exception as e:  # noqa: BLE001
        done = {r["instance_id"] for r in results}
        for it in items:
            if it["instance_id"] not in done:
                rec(it, "error", err=repr(e)[-300:])
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--git-root", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--shallow", action="store_true")
    ap.add_argument("--max-files", type=int, default=2500)
    ap.add_argument("--max-bytes", type=float, default=60e6)
    ap.add_argument("--max-ok", type=int, default=0)
    ap.add_argument("--max-repos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-repo", type=int, default=1,
                    help="split each repo's instances into N parallel jobs (needs an existing clone)")
    ap.add_argument("--only-repo", default=None)
    a = ap.parse_args()
    a.git_root = os.path.abspath(a.git_root)
    by_repo = collections.OrderedDict()
    for line in open(a.input):
        it = json.loads(line)
        by_repo.setdefault(it["repo"], []).append(it)
    repos = list(by_repo)
    random.Random(a.seed).shuffle(repos)
    if a.max_repos:
        repos = repos[: a.max_repos]
    # big repos first for full-clone mode so they don't straggle
    repos.sort(key=lambda r: -len(by_repo[r]))
    if a.only_repo:
        repos = [r for r in repos if r == a.only_repo]
    k = max(1, a.split_repo)
    jobs = [(r, by_repo[r][i::k], a) for r in repos for i in range(k) if by_repo[r][i::k]]
    t0 = time.time()
    counts = collections.Counter()
    with open(a.results, "w") as f, Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_repo, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
                counts[r["status"]] += 1
            f.flush()
            print(f"[{time.time()-t0:7.0f}s] {i+1}/{len(jobs)} repos {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
