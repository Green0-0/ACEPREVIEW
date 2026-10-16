"""Execution-free patch-similarity proxy fitness (in the spirit of SWE-RL).

``patch_similarity(pred, gold)`` compares a predicted unified diff with the gold
diff without running anything:

    score = 0.5 * F1(touched files of pred, touched files of gold)
          + 0.5 * mean_{f in pred ∪ gold} SequenceMatcher(norm(pred_f), norm(gold_f)).ratio()

where ``norm`` keeps only the ``+``/``-`` lines of a file's hunks with
whitespace collapsed, and a file present on only one side contributes 0.
An empty or unparseable prediction scores 0.0.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

__all__ = [
    "FileDiff",
    "parse_patch",
    "is_test_path",
    "patch_similarity",
    "localization_scores",
]

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_TEST_DIR_PARTS = {"test", "tests", "testing", "unittests", "unit_tests"}


@dataclass
class FileDiff:
    path: str
    lines: list[str] = field(default_factory=list)  # raw "+"/"-" lines (with sign)
    old_changed: list[int] = field(default_factory=list)  # original line numbers removed/anchoring additions
    binary: bool = False
    new_file: bool = False
    deleted_file: bool = False

    @property
    def n_changed(self) -> int:
        return len(self.lines)


def is_test_path(path: str) -> bool:
    """Heuristic: is ``path`` a test file (tests/ dir, test_*.py, *_test.py, conftest.py)?"""
    parts = path.lower().split("/")
    name = parts[-1]
    if any(p in _TEST_DIR_PARTS for p in parts[:-1]):
        return True
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_tests.py")
        or name in {"conftest.py", "tests.py", "test.py"}
    )


def _strip_prefix(p: str) -> str:
    p = p.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0]
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_patch(patch: str) -> dict[str, FileDiff]:
    """Parse a (git-style or plain) unified diff into ``{path: FileDiff}``.

    Robust to missing ``diff --git`` headers; files with no hunks and no
    binary marker are dropped unless they are pure renames/new/deleted files.
    """
    files: dict[str, FileDiff] = {}
    if not patch or not patch.strip():
        return files
    cur: FileDiff | None = None
    old_path: str | None = None
    old_ln = new_ln = 0
    in_hunk = False
    rem_old = rem_new = 0

    def start(path: str) -> FileDiff:
        fd = files.get(path)
        if fd is None:
            fd = FileDiff(path)
            files[path] = fd
        return fd

    for raw in patch.splitlines():
        if in_hunk and (rem_old > 0 or rem_new > 0) and cur is not None:
            if raw.startswith("+") :
                cur.lines.append(raw)
                cur.old_changed.append(old_ln)
                rem_new -= 1
                continue
            if raw.startswith("-"):
                cur.lines.append(raw)
                cur.old_changed.append(old_ln)
                old_ln += 1
                rem_old -= 1
                continue
            if raw.startswith(" ") or raw == "":
                old_ln += 1
                rem_old -= 1
                rem_new -= 1
                continue
            if raw.startswith("\\"):
                continue
            in_hunk = False  # malformed; fall through to header parsing
        if raw.startswith("diff --git "):
            in_hunk = False
            m = re.match(r'^diff --git ("?a/.+?"?) ("?b/.+"?)$', raw)
            cur = start(_strip_prefix(m.group(2))) if m else None
            old_path = _strip_prefix(m.group(1)) if m else None
        elif raw.startswith("--- "):
            in_hunk = False
            old_path = _strip_prefix(raw[4:])
        elif raw.startswith("+++ "):
            in_hunk = False
            new = _strip_prefix(raw[4:])
            if new == "/dev/null" or new == "dev/null":
                new = old_path or new
                fd = start(new)
                fd.deleted_file = True
            else:
                fd = start(new)
            if cur is not None and cur.path != fd.path and not cur.lines and cur.path in files:
                # diff --git header named a different (renamed) path: keep the +++ one
                if cur.path != new:
                    files.pop(cur.path, None)
            cur = fd
        elif raw.startswith("new file mode") and cur is not None:
            cur.new_file = True
        elif raw.startswith("deleted file mode") and cur is not None:
            cur.deleted_file = True
        elif (raw.startswith("Binary files") or raw.startswith("GIT binary patch")) and cur is not None:
            cur.binary = True
        else:
            m = _HUNK_RE.match(raw)
            if m and cur is not None:
                old_ln = int(m.group(1))
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
                in_hunk = True
    return {
        p: f for p, f in files.items()
        if f.lines or f.binary or f.new_file or f.deleted_file
    }


def _norm_body(fd: FileDiff) -> list[str]:
    return [l[0] + " ".join(l[1:].split()) for l in fd.lines if l[1:].strip()]


def _file_f1(pred: set[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(gold)
    return 2 * p * r / (p + r)


def _body_ratio(a: FileDiff | None, b: FileDiff | None) -> float:
    if a is None or b is None:
        return 0.0
    la, lb = _norm_body(a), _norm_body(b)
    if not la and not lb:
        return 1.0
    return difflib.SequenceMatcher(None, la, lb, autojunk=False).ratio()


def patch_similarity(pred_patch: str, gold_patch: str) -> float:
    """Similarity in [0, 1] between a predicted and a gold unified diff."""
    try:
        pred = parse_patch(pred_patch)
        gold = parse_patch(gold_patch)
    except Exception:
        return 0.0
    if not pred or not gold:
        return 0.0
    f1 = _file_f1(set(pred), set(gold))
    union = set(pred) | set(gold)
    body = sum(_body_ratio(pred.get(f), gold.get(f)) for f in union) / len(union)
    return max(0.0, min(1.0, 0.5 * f1 + 0.5 * body))


def localization_scores(pred_patch: str, gold_patch: str, window: int = 3) -> dict:
    """Analysis metrics: file_f1, hunk_overlap, similarity, plus file counts.

    ``hunk_overlap`` is the fraction of gold-changed original line numbers that
    lie within ``window`` lines of some pred-changed original line in the same file.
    """
    try:
        pred = parse_patch(pred_patch)
        gold = parse_patch(gold_patch)
    except Exception:
        pred, gold = {}, {}
    f1 = _file_f1(set(pred), set(gold))
    total = hit = 0
    for path, g in gold.items():
        glines = sorted(set(g.old_changed))
        total += len(glines)
        p = pred.get(path)
        if p is None or not glines:
            continue
        plines = sorted(set(p.old_changed))
        if not plines:
            continue
        import bisect
        for ln in glines:
            i = bisect.bisect_left(plines, ln - window)
            if i < len(plines) and plines[i] <= ln + window:
                hit += 1
    return {
        "file_f1": f1,
        "hunk_overlap": (hit / total) if total else 0.0,
        "similarity": patch_similarity(pred_patch, gold_patch),
        "pred_files": sorted(pred),
        "gold_files": sorted(gold),
    }
