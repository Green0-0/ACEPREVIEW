"""Tests for ace.neat.evaluation: MockEvaluator, Task, TaskEvaluator and task loading."""
from __future__ import annotations

import asyncio
import json
import random
import tempfile
from pathlib import Path

import pytest

from ace.graph.llm import ModelSettings
from ace.graph.types import END, Kind, WorkerType
from ace.neat.config import MutationConfig
from ace.neat.evaluation import (
    Evaluator,
    MockEvaluator,
    Task,
    TaskEvaluator,
    load_tasks,
    nonempty_diff,
    resolve_scorer,
)
from ace.neat.genome import Genome, minimal_genome
from ace.neat.innovation import InnovationDB
from ace.neat.mutation import random_valid_genome
from fakes import FakeClient, text_message, tool_call_message

T = WorkerType
SETTINGS = ModelSettings(model="fake-model", api_base_url="http://x", api_key_env_var="FAKE_KEY")

NEW_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        low, high = high, low\n"
    "    return max(low, min(high, value))\n"
)


def editor_script() -> list:
    return [
        tool_call_message(
            (
                "replace_element",
                json.dumps(
                    {"file_path": "pkg/utils.py", "element_path": "clamp", "new_content": NEW_CLAMP}
                ),
            )
        ),
        text_message("Swapped reversed bounds in clamp."),
    ]


def merger_genome(db: InnovationDB) -> Genome:
    """EDITOR -> CRITIC -> JUDGEMERGER <- EDITOR, with views: a hand-built valid genome."""
    g = Genome(id="g", nodes={}, connections={})
    e0 = g.add_node(db, ("init", 0, T.EDITOR)).innovation
    e1 = g.add_node(db, ("init", 1, T.EDITOR)).innovation
    c = g.add_node(db, ("init", 2, T.CRITIC)).innovation
    m = g.add_node(db, ("init", 3, T.JUDGEMERGER)).innovation
    g.add_connection(db, e0, c, Kind.EDIT)
    g.add_connection(db, e0, c, Kind.VIEW)
    g.add_connection(db, c, m, Kind.EDIT)
    g.add_connection(db, e1, m, Kind.EDIT)
    g.add_connection(db, e1, m, Kind.VIEW)
    g.add_connection(db, m, END, Kind.EDIT)
    g.validate()
    return g


# ------------------------------------------------------------------ MockEvaluator


def test_mock_is_an_evaluator():
    assert isinstance(MockEvaluator(), Evaluator)


def test_mock_formula_minimal_and_merger():
    db = InnovationDB()
    ev = MockEvaluator()
    assert ev.score(minimal_genome(db)) == pytest.approx(1.0)
    # types {EDITOR, CRITIC, JUDGEMERGER} = 3, merger at END +0.5, one critic +0.25
    assert ev.score(merger_genome(db)) == pytest.approx(3.75)


def test_mock_size_penalty():
    db = InnovationDB()
    g = minimal_genome(db)
    editor = next(iter(g.nodes))
    for i in range(7):  # 7 VIEWER sources -> 8 workers
        v = g.add_node(db, ("test", i, T.VIEWER)).innovation
        g.add_connection(db, v, editor, Kind.VIEW)
    g.validate()
    assert g.num_workers() == 8
    assert MockEvaluator().score(g) == pytest.approx(2.0 - 0.1 * 2)


def test_mock_deterministic_with_noise():
    rng = random.Random(3)
    db = InnovationDB()
    genomes = [
        random_valid_genome(db, rng, MutationConfig(), min_workers=1, max_workers=6)
        for _ in range(10)
    ]
    a, b = MockEvaluator(noise=0.3), MockEvaluator(noise=0.3)
    first = [asyncio.run(a.evaluate(g)) for g in genomes]
    second = [asyncio.run(b.evaluate(g)) for g in genomes]
    assert first == second
    base = MockEvaluator()
    for g, value in zip(genomes, first):
        assert abs(value - base.score(g)) <= 0.3
        # same structure, different ids -> same score
        assert a.score(g.copy()) == value
    assert a.calls == len(genomes)
    assert a.evaluated_ids == [g.id for g in genomes]


def test_mock_scores_override():
    db = InnovationDB()
    g = minimal_genome(db)
    other = merger_genome(db)
    ev = MockEvaluator(scores={g.structure_hash(): 42.0})
    assert asyncio.run(ev.evaluate(g)) == 42.0
    assert asyncio.run(ev.evaluate(other)) == pytest.approx(3.75)


# ------------------------------------------------------------------ TaskEvaluator


@pytest.fixture(autouse=True)
def _private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default rollout roots (``tempfile.mkdtemp``) land in this test's tmp dir."""
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))


def editor_factory(clients: list[FakeClient]):
    def factory() -> FakeClient:
        client = FakeClient(editor_script())
        clients.append(client)
        return client

    return factory


def test_task_evaluator_scores_editor_edit(project_dir: Path):
    clients: list[FakeClient] = []
    task = Task(project_dir, "Make clamp handle reversed bounds.", nonempty_diff)
    ev = TaskEvaluator([task], SETTINGS, client_factory=editor_factory(clients))
    genome = minimal_genome(InnovationDB())
    assert asyncio.run(ev.evaluate(genome)) == 1.0
    assert len(clients) == 1
    (record,) = ev.last_results
    assert record["score"] == 1.0 and record["failure"] is None and record["diff_chars"] > 0
    assert ev.results_by_genome[genome.id] == ev.last_results
    # pristine project untouched
    assert "low, high = high, low" not in (project_dir / "pkg/utils.py").read_text()


def test_task_evaluator_async_scorer_and_mean(project_dir: Path):
    seen = []

    async def half(edit, session):
        seen.append(sorted(edit.files))
        return 0.5

    tasks = [
        Task(project_dir, "fix clamp", nonempty_diff),
        Task(project_dir, "fix clamp", half),
    ]
    ev = TaskEvaluator(tasks, SETTINGS, client_factory=editor_factory([]))
    assert asyncio.run(ev.evaluate(minimal_genome(InnovationDB()))) == pytest.approx(0.75)
    assert seen == [["pkg/utils.py"]]
    assert [r["task"] for r in ev.last_results] == [0, 1]


def test_task_evaluator_failure_scores_zero(project_dir: Path):
    class Boom:
        def __init__(self) -> None:
            async def create(**kwargs):
                raise RuntimeError("model exploded")

            from types import SimpleNamespace

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    task = Task(project_dir, "fix clamp", nonempty_diff)
    ev = TaskEvaluator([task], SETTINGS, client_factory=Boom)
    genome = minimal_genome(InnovationDB())
    assert asyncio.run(ev.evaluate(genome)) == 0.0
    (record,) = ev.last_results
    assert record["score"] == 0.0
    assert record["failure"] is not None and "GraphFailure" in record["failure"]
    assert record["node_id"] == next(iter(genome.nodes))


def test_task_evaluator_isolates_rollouts(project_dir: Path, tmp_path: Path):
    """Rollouts go to <rollout_dir>/<genome id>/<task index>; the project gets no .agent2."""
    rollouts = tmp_path / "history" / "rollouts"
    seen_paths: list[Path] = []

    def scorer(edit, session):
        seen_paths.append(session.rollout_path)
        return 1.0

    tasks = [Task(project_dir, "fix clamp", scorer), Task(project_dir, "fix clamp", scorer)]
    ev = TaskEvaluator(tasks, SETTINGS, client_factory=editor_factory([]), rollout_dir=rollouts)
    db = InnovationDB()
    genomes = [minimal_genome(db), minimal_genome(db)]
    for genome in genomes:
        assert asyncio.run(ev.evaluate(genome)) == 1.0

    assert not (project_dir / ".agent2").exists()
    expected = {rollouts / g.id / str(i) for g in genomes for i in range(2)}
    assert set(seen_paths) == expected
    for path in expected:
        files = list(path.glob("*.json"))
        assert len(files) == 1  # one EDITOR rollout per task run, nothing inherited
        assert json.loads(files[0].read_text())["raw_chat"]
    records = ev.results_by_genome[genomes[0].id]
    assert [r["rollout_dir"] for r in records] == [
        str(rollouts / genomes[0].id / "0"), str(rollouts / genomes[0].id / "1")
    ]


def test_task_evaluator_run_context_rollout_dir(project_dir: Path, tmp_path: Path, monkeypatch):
    """RunContext.rollout_dir is the same absolute path the Session writes to."""
    from ace.graph import runner

    contexts = []
    original = runner.GraphRunner.run

    async def spy(self, ctx):
        contexts.append(ctx)
        return await original(self, ctx)

    monkeypatch.setattr(runner.GraphRunner, "run", spy)
    rollouts = tmp_path / "r"
    ev = TaskEvaluator(
        [Task(project_dir, "fix clamp", nonempty_diff)], SETTINGS,
        client_factory=editor_factory([]), rollout_dir=rollouts,
    )
    genome = minimal_genome(InnovationDB())
    asyncio.run(ev.evaluate(genome))
    (ctx,) = contexts
    assert ctx.rollout_dir == rollouts.resolve() / genome.id / "0"
    assert ctx.rollout_dir.is_absolute()
    assert ctx.session.rollout_path == ctx.rollout_dir


def test_task_evaluator_default_rollout_root_is_temp(project_dir: Path):
    ev = TaskEvaluator(
        [Task(project_dir, "fix clamp", nonempty_diff)], SETTINGS, client_factory=editor_factory([])
    )
    genome = minimal_genome(InnovationDB())
    assert asyncio.run(ev.evaluate(genome)) == 1.0
    root = ev.rollout_root
    assert root.is_absolute() and root.name.startswith("agent2-rollouts-")
    assert Path(tempfile.gettempdir()).resolve() in root.parents
    assert list((root / genome.id / "0").glob("*.json"))
    assert not (project_dir / ".agent2").exists()


def test_task_evaluator_rejects_rollout_dir_scanned_as_project_files(project_dir: Path):
    task = Task(project_dir, "fix clamp", nonempty_diff)
    genome = minimal_genome(InnovationDB())
    inside = TaskEvaluator([task], SETTINGS, client_factory=editor_factory([]),
                           rollout_dir=project_dir / "runs")
    with pytest.raises(ValueError, match="inside project"):
        asyncio.run(inside.evaluate(genome))
    hidden = TaskEvaluator([task], SETTINGS, client_factory=editor_factory([]),
                           rollout_dir=project_dir / ".rollouts")
    assert asyncio.run(hidden.evaluate(genome)) == 1.0


def test_task_evaluator_no_tasks():
    ev = TaskEvaluator([], SETTINGS, client_factory=lambda: None)
    assert asyncio.run(ev.evaluate(minimal_genome(InnovationDB()))) == 0.0
    assert ev.last_results == []


def test_task_evaluator_concurrency_limit(project_dir: Path):
    state = {"active": 0, "peak": 0}

    class SlowClient(FakeClient):
        async def create(self, **kwargs):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            try:
                await asyncio.sleep(0.02)
                return await super().create(**kwargs)
            finally:
                state["active"] -= 1

    def factory() -> SlowClient:
        return SlowClient(editor_script())

    tasks = [Task(project_dir, "fix clamp", nonempty_diff) for _ in range(5)]
    ev = TaskEvaluator(tasks, SETTINGS, client_factory=factory, concurrency=2)
    db = InnovationDB()
    genomes = [minimal_genome(db) for _ in range(2)]

    async def both() -> list[float]:
        return await asyncio.gather(*(ev.evaluate(g) for g in genomes))

    assert asyncio.run(both()) == [1.0, 1.0]
    assert state["peak"] == 2
    # a second event loop gets a fresh semaphore
    assert asyncio.run(ev.evaluate(genomes[0])) == 1.0


def test_task_evaluator_default_factory():
    ev = TaskEvaluator([], SETTINGS)
    assert callable(ev.client_factory)
    with pytest.raises(ValueError):
        TaskEvaluator([], SETTINGS, concurrency=0)


# ----------------------------------------------------------- run recording (11.5)


EVAL_KEYS = {
    "genome_id", "generation", "task_index", "task", "started_at", "finished_at",
    "duration_s", "score", "failure", "failure_node_id", "failure_type", "diff",
    "touched_paths", "diff_chars", "order", "workers", "inputs",
}
WORKER_KEYS = {
    "node_id", "type", "properties", "started_at", "finished_at", "duration_s", "status",
    "error", "rollout_ids", "rollout_dir", "llm_calls", "tool_calls", "report", "view",
    "view_summary", "edit_touched_paths", "edit_chars",
}


def test_task_evaluator_writes_full_evaluation_record(project_dir: Path, tmp_path: Path):
    from ace.graph.views import View
    from ace.neat.history import History

    history = History(tmp_path / "run")
    task = Task(project_dir, "Make clamp handle reversed bounds.", nonempty_diff)

    def viewing_editor() -> FakeClient:
        view_call = tool_call_message(
            ("view", json.dumps({"paths": "pkg/utils.py::clamp"})), prefix="view"
        )
        return FakeClient([view_call, *editor_script()])

    ev = TaskEvaluator([task], SETTINGS, client_factory=viewing_editor,
                       rollout_dir=tmp_path / "run" / "rollouts", history=history)
    genome = minimal_genome(InnovationDB())
    genome.generation = 4
    assert asyncio.run(ev.evaluate(genome)) == 1.0
    assert ev.last_results[0]["score"] == 1.0  # legacy records still kept

    (record,) = list(history.iter_evaluations())
    assert set(record) == EVAL_KEYS
    assert record["genome_id"] == genome.id and record["generation"] == 4
    assert record["task_index"] == 0
    assert record["task"] == {"project_dir": str(project_dir), "prompt": task.prompt,
                              "scorer": "nonempty_diff"}
    assert record["score"] == 1.0
    assert record["failure"] is None and record["failure_node_id"] is None
    assert record["failure_type"] is None
    assert "low, high = high, low" in record["diff"]
    assert record["diff_chars"] == len(record["diff"])
    assert record["touched_paths"] == ["pkg/utils.py"]
    assert record["duration_s"] >= 0 and record["started_at"] <= record["finished_at"]
    editor = next(iter(genome.nodes))
    assert record["order"] == [editor]
    assert record["inputs"] == {str(editor): {"views_from": [], "edits_from": [], "reports_from": []}}

    (worker,) = record["workers"]
    assert set(worker) == WORKER_KEYS
    assert worker["node_id"] == editor and worker["type"] == "editor"
    assert worker["status"] == "ok" and worker["error"] is None
    assert isinstance(worker["properties"]["max_steps"], int)
    assert worker["llm_calls"] >= 1 and worker["tool_calls"] >= 1
    assert worker["llm_calls"] == 3 and worker["tool_calls"] == 2
    assert len(worker["rollout_ids"]) == 1
    rollout_dir = Path(worker["rollout_dir"])
    assert rollout_dir == (tmp_path / "run" / "rollouts").resolve() / genome.id / "0"
    assert (rollout_dir / f"{worker['rollout_ids'][0]}.json").exists()
    assert worker["report"] == "Swapped reversed bounds in clamp."
    assert "pkg/utils.py" in View.from_json(worker["view"]).files
    summary = worker["view_summary"]
    assert summary["files"] >= 1 and summary["entries"] >= 1 and summary["rendered_chars"] > 0
    assert worker["edit_touched_paths"] == ["pkg/utils.py"] and worker["edit_chars"] > 0
    assert worker["duration_s"] >= 0

    events = [(e["event"], e.get("task_index")) for e in history.iter_events()]
    assert events == [("evaluation_start", 0), ("evaluation_end", 0)]
    end = list(history.iter_events())[1]
    assert end["genome_id"] == genome.id and end["score"] == 1.0 and end["failure"] is None
    assert end["duration_s"] == record["duration_s"]


def test_task_evaluator_failure_record_marks_failed_and_skipped(project_dir: Path, tmp_path: Path):
    from types import SimpleNamespace

    from ace.neat.history import History

    class Boom:
        def __init__(self) -> None:
            async def create(**kwargs):
                raise RuntimeError("model exploded")

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    history = History(tmp_path / "run")
    db = InnovationDB()
    genome = merger_genome(db)
    ev = TaskEvaluator([Task(project_dir, "fix clamp", nonempty_diff)], SETTINGS,
                       client_factory=Boom, rollout_dir=tmp_path / "run" / "rollouts",
                       history=history)
    assert asyncio.run(ev.evaluate(genome)) == 0.0
    (record,) = list(history.iter_evaluations())
    assert set(record) == EVAL_KEYS
    graph = genome.to_graph()
    editors = sorted(n for n, node in graph.nodes.items() if node.type is T.EDITOR)
    downstream = sorted(n for n, node in graph.nodes.items() if node.type is not T.EDITOR)
    assert record["score"] == 0.0
    assert record["failure"] is not None and "model exploded" in record["failure"]
    assert record["failure_node_id"] == editors[0]
    assert isinstance(record["failure_type"], str) and record["failure_type"]
    assert record["diff"] is None and record["touched_paths"] == [] and record["diff_chars"] == 0
    workers = {w["node_id"]: w for w in record["workers"]}
    assert sorted(workers) == sorted(graph.nodes)  # every node is listed
    assert workers[editors[0]]["status"] == "failed"
    assert "model exploded" in workers[editors[0]]["error"]
    assert workers[editors[0]]["started_at"] is not None
    for nid in downstream:
        assert workers[nid]["status"] == "skipped"
        assert workers[nid]["started_at"] is None and workers[nid]["llm_calls"] == 0
        assert workers[nid]["view"] is None
    assert set(record["order"]) <= set(editors)
    critic = next(n for n in downstream if graph.nodes[n].type is T.CRITIC)
    merger = next(n for n in downstream if graph.nodes[n].type is T.JUDGEMERGER)
    assert record["inputs"][str(critic)] == {
        "views_from": [editors[0]], "edits_from": [editors[0]], "reports_from": []
    }
    assert record["inputs"][str(merger)]["edits_from"] == sorted([critic, editors[1]])
    end = [e for e in history.iter_events() if e["event"] == "evaluation_end"]
    assert end[0]["failure"] == record["failure"]


def test_mock_evaluator_writes_mock_records(tmp_path: Path):
    from ace.neat.history import History

    history = History(tmp_path)
    ev = MockEvaluator(history=history)
    genome = minimal_genome(InnovationDB())
    genome.generation = 2
    score = asyncio.run(ev.evaluate(genome))
    (record,) = list(history.iter_evaluations())
    assert set(record) == {"genome_id", "generation", "task_index", "score", "mock",
                           "started_at", "finished_at", "duration_s"}
    assert record["genome_id"] == genome.id and record["generation"] == 2
    assert record["task_index"] == 0 and record["mock"] is True and record["score"] == score
    assert record["duration_s"] >= 0
    assert MockEvaluator().describe() == {"type": "mock", "tasks": []}


def test_task_evaluator_describe(project_dir: Path):
    ev = TaskEvaluator([Task(project_dir, "fix clamp", nonempty_diff)], SETTINGS,
                       client_factory=lambda: None)
    assert ev.history is None
    assert ev.describe() == {"type": "task", "tasks": [
        {"index": 0, "project_dir": str(project_dir), "prompt": "fix clamp",
         "scorer": "nonempty_diff"}
    ]}


# --------------------------------------------------------------------- task files


def test_resolve_scorer():
    assert resolve_scorer("nonempty_diff") is nonempty_diff
    assert resolve_scorer("ace.neat.evaluation:nonempty_diff") is nonempty_diff
    with pytest.raises(ValueError):
        resolve_scorer("no_such_scorer")


def test_load_tasks(tmp_path: Path, project_dir: Path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                {"project_dir": str(project_dir), "prompt": "a", "scorer": "nonempty_diff"},
                {"project_dir": "project", "prompt": "b"},
            ]
        )
    )
    tasks = load_tasks(path)
    assert [t.prompt for t in tasks] == ["a", "b"]
    assert tasks[1].project_dir == project_dir.resolve()
    assert all(t.scorer is nonempty_diff for t in tasks)
    path.write_text(json.dumps([{"prompt": "x"}]))
    with pytest.raises(ValueError):
        load_tasks(path)
