import json
import random

from ace.swebench.data import load_pool, make_prompt, sample_minibatch


def _pool():
    return [{"instance_id": f"r{r}__{i}", "repo": f"o/r{r}"} for r in range(5) for i in range(r + 1)]


def test_minibatch_no_repo_repeat():
    pool = _pool()
    for seed in range(20):
        mb = sample_minibatch(pool, 5, random.Random(seed))
        assert len(mb) == 5 and len({m["repo"] for m in mb}) == 5


def test_minibatch_overflow_and_distinct():
    pool = _pool()
    mb = sample_minibatch(pool, 9, random.Random(0))
    assert len({m["instance_id"] for m in mb}) == 9
    assert len(sample_minibatch(pool, 100, random.Random(0))) == len(pool)
    assert sample_minibatch(pool, 4, random.Random(3)) == sample_minibatch(pool, 4, random.Random(3))


def test_load_pool_and_prompt(tmp_path):
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in _pool()) + "\n\n")
    assert load_pool(p) == _pool()
    s = make_prompt("  Bug here\n")
    assert s.endswith("<issue>\nBug here\n</issue>") and "Do not modify or add tests." in s
