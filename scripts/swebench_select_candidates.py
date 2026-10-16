"""Select training candidates from SWE-Fixer-Train-110K and write Verified task metadata.

Outputs (under --out, default /usr/project/xtmp/kg349/ace/data):
  train_candidates.jsonl   up to --per-repo-candidates per repo, filtered (pre-materialization)
  verified_full.jsonl      Verified instances incl. gold patch (internal; for apply checks)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ace.swebench.proxy import is_test_path, parse_patch  # noqa: E402

DOC_EXT = (".rst", ".md", ".txt", ".adoc")
DOC_NAMES = re.compile(r"(^|/)(changes|changelog|history|news|authors|contributors|release[-_]?notes)[^/]*$", re.I)


def split_file_chunks(patch: str) -> list[tuple[str, str]]:
    """Split a git diff into (path, chunk_text) per 'diff --git' block."""
    chunks, cur, path = [], [], None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur:
                chunks.append((path, "".join(cur)))
            cur = [line]
            m = re.match(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?\s*$', line)
            path = m.group(2) if m else None
        else:
            cur.append(line)
    if cur:
        chunks.append((path, "".join(cur)))
    return chunks


def clean_patch(patch: str):
    """Return (code_patch, py_files, n_changed) or None if unusable."""
    if not patch or "diff --git" not in patch:
        return None
    keep, files = [], []
    for path, chunk in split_file_chunks(patch):
        if path is None:
            return None
        if is_test_path(path):
            continue
        if path.lower().endswith(DOC_EXT) or DOC_NAMES.search(path) or path.startswith(("docs/", "doc/")):
            continue
        if not path.endswith(".py"):
            return None  # non-python code/config change: drop instance
        if "GIT binary patch" in chunk or "Binary files" in chunk:
            return None
        keep.append(chunk)
        files.append(path)
    if not keep:
        return None
    code = "".join(keep)
    if not code.endswith("\n"):
        code += "\n"
    parsed = parse_patch(code)
    if set(parsed) != set(files):
        return None
    n = sum(f.n_changed for f in parsed.values())
    return code, files, n


_STOP = {"the", "is", "to", "and", "a", "of", "in", "it", "this", "that", "when", "with", "not", "for",
         "be", "an", "on", "i", "if", "should", "but", "are", "as", "error", "from", "can", "have"}


def looks_english(s: str) -> bool:
    words = re.findall(r"[A-Za-z\u00C0-\uFFFF]+", s)
    if len(words) < 20:
        return False
    nonascii = sum(1 for ch in s if ord(ch) > 127)
    if nonascii > 0.05 * len(s):
        return False
    return sum(w.lower() in _STOP for w in words) >= 0.06 * len(words)


def norm_ps(s: str) -> str:
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/usr/project/xtmp/kg349/ace/data")
    ap.add_argument("--per-repo-candidates", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = load_dataset("princeton-nlp/SWE-bench", split="test")
    ver = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    bench_names = {r.split("/")[1].lower() for r in set(full["repo"]) | set(ver["repo"])}
    print("SWE-bench repo names excluded:", sorted(bench_names))
    ver_ps = {norm_ps(p) for p in ver["problem_statement"]}
    ver_ps_prefix = {p[:300] for p in ver_ps}

    with open(out / "verified_full.jsonl", "w") as f:
        for r in ver:
            f.write(json.dumps({k: r[k] for k in (
                "instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch",
                "FAIL_TO_PASS", "PASS_TO_PASS", "version", "environment_setup_commit")}) + "\n")

    ds = load_dataset("internlm/SWE-Fixer-Train-110K", split="train")
    stats = collections.Counter()
    by_repo = collections.defaultdict(list)
    seen_ids = set()
    for r in ds:
        stats["total"] += 1
        repo = r["repo"]
        if "/" not in repo:
            stats["bad_repo"] += 1
            continue
        if repo.split("/")[1].lower() in bench_names:
            stats["swebench_repo"] += 1
            continue
        ps = r["problem_statement"] or ""
        if not (200 <= len(ps) <= 6000):
            stats["ps_len"] += 1
            continue
        if not looks_english(ps):
            stats["non_english"] += 1
            continue
        nps = norm_ps(ps)
        if nps in ver_ps or nps[:300] in ver_ps_prefix:
            stats["ps_dup_verified"] += 1
            continue
        res = clean_patch(r["patch"])
        if res is None:
            stats["patch_unusable"] += 1
            continue
        code, files, n = res
        if not (1 <= len(files) <= 3) or n > 60:
            stats["patch_size"] += 1
            continue
        if r["instance_id"] in seen_ids:
            stats["dup_id"] += 1
            continue
        seen_ids.add(r["instance_id"])
        stats["eligible"] += 1
        by_repo[repo].append({
            "instance_id": r["instance_id"], "repo": repo, "base_commit": r["base_commit"],
            "problem_statement": ps, "patch": code, "gold_files": files, "n_changed": n,
            "created_at": str(r["created_at"]),
        })
    rng = random.Random(args.seed)
    n_out = 0
    with open(out / "train_candidates.jsonl", "w") as f:
        for repo in sorted(by_repo):
            items = by_repo[repo]
            rng.shuffle(items)
            for it in items[: args.per_repo_candidates]:
                f.write(json.dumps(it) + "\n")
                n_out += 1
    stats["eligible_repos"] = len(by_repo)
    stats["candidates_written"] = n_out
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
