"""Demo runs: run one genome's worker graph on a sample project from the web UI.

A demo is started from a genome's demo page with a prompt typed by the user. The graph runs
exactly as in an evaluation (``GraphRunner`` over a ``Session`` of the demo project), but
the project is never modified: workers edit private workspaces and the only thing written
to disk is the worker rollouts, into a temporary directory that belongs to the demo.

While the demo runs, the web UI polls its state: per-worker status, LLM call counts, the
diff each worker produced, a log, and the live transcripts, which are read from the rollout
files the agents save after every step. When it finishes, the evaluation-style record
(``build_evaluation_record``) completes the picture (views, reports, tool counts).

The model defaults to ``claude-haiku-5-5`` through the Anthropic OpenAI-compatible endpoint
with thinking turned off (the lowest level that endpoint exposes). ``DemoSettings.client_factory``
replaces the network client, which is how the tests drive a demo offline.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from ace.graph.clock import seconds_between, utc_now
from ace.graph.graph import WorkerGraph
from ace.graph.llm import ModelSettings, make_client_factory
from ace.graph.types import END, GraphFailure, Kind

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_API_BASE = "https://api.anthropic.com/v1/"
DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MAX_TOKENS = 8192
# The OpenAI-compatible endpoint ignores ``output_config.effort``; ``thinking`` is honoured.
DEFAULT_EXTRA_BODY: dict[str, Any] = {"thinking": {"type": "disabled"}}
MAX_PROMPT_CHARS = 4000
MAX_LOG_LINES = 500

_CURRENT_NODE: ContextVar[int | None] = ContextVar("ace_demo_node", default=None)
"""Set from the runner's observer inside each worker's task, so the client the worker
builds through ``client_factory`` knows which node it belongs to."""


class DemoUnavailable(Exception):
    """The demo cannot start: no project, no API key, or another demo is running (HTTP 409)."""


class DemoRejected(Exception):
    """The request itself is invalid, e.g. an empty prompt (HTTP 422)."""


@dataclass
class DemoSettings:
    """Where and with which model demo runs happen."""

    project_dir: Path | None
    model: str = DEFAULT_MODEL
    api_base_url: str | None = DEFAULT_API_BASE
    api_key_env_var: str | None = DEFAULT_KEY_ENV
    max_tokens: int = DEFAULT_MAX_TOKENS
    extra_body: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EXTRA_BODY))
    client_factory: Callable[[], Any] | None = None

    def model_settings(self) -> ModelSettings:
        return ModelSettings(model=self.model, api_base_url=self.api_base_url,
                             api_key_env_var=self.api_key_env_var)

    def key_set(self) -> bool:
        if self.client_factory is not None or not self.api_key_env_var:
            return True
        return bool(os.environ.get(self.api_key_env_var))

    def project_ok(self) -> bool:
        return self.project_dir is not None and Path(self.project_dir).is_dir()

    def unavailable_reason(self) -> str | None:
        if not self.project_ok():
            return "no demo project: start the web UI with --demo-project <folder>"
        if not self.key_set():
            return (f"{self.api_key_env_var} is not set: put it in .env at the project root "
                    "and restart the web UI")
        return None


class _DemoClient:
    """Proxy over the model client that applies the demo's request defaults and counts calls."""

    def __init__(self, inner: Any, settings: DemoSettings, demo: Demo, node_id: int | None) -> None:
        self.inner = inner
        self.settings = settings
        self.demo = demo
        self.node_id = node_id
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        kwargs.setdefault("max_tokens", self.settings.max_tokens)
        if self.settings.extra_body:
            body = dict(self.settings.extra_body)
            body.update(kwargs.get("extra_body") or {})
            kwargs["extra_body"] = body
        self.demo.llm_calls += 1
        worker = self.demo.workers.get(self.node_id) if self.node_id is not None else None
        if worker is not None:
            worker.llm_calls += 1
            worker.last_activity = utc_now()
        return await self.inner.chat.completions.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class WorkerState:
    """Live state of one worker of a demo (``pending → running → ok | failed | skipped``)."""

    node_id: int
    type: str
    properties: dict[str, Any]
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    last_activity: str | None = None
    llm_calls: int = 0
    error: str | None = None
    report: str | None = None
    diff: str | None = None
    touched_paths: list[str] = field(default_factory=list)
    rollout_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        end = self.finished_at or (utc_now() if self.started_at else None)
        return {
            "node_id": self.node_id,
            "type": self.type,
            "properties": self.properties,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": seconds_between(self.started_at, end) if self.started_at and end else None,
            "last_activity": self.last_activity,
            "llm_calls": self.llm_calls,
            "error": self.error,
            "report": self.report,
            "diff": self.diff,
            "diff_chars": len(self.diff) if self.diff else 0,
            "edit_touched_paths": list(self.touched_paths),
            "rollout_ids": list(self.rollout_ids),
        }


def _count_roles(rollout: dict[str, Any]) -> tuple[int, int]:
    assistant = tool = 0
    for message in rollout.get("raw_chat") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            assistant += 1
            tool += len(message.get("tool_calls") or [])
    return assistant, tool


@dataclass
class Demo:
    """One demo run: live worker states, the log and, once finished, the evaluation record."""

    id: str
    run_id: str
    genome_id: str
    prompt: str
    graph: WorkerGraph
    project_dir: Path
    rollout_dir: Path
    model: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"  # running | done | failed | cancelled
    llm_calls: int = 0
    error: str | None = None
    workers: dict[int, WorkerState] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    record: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None

    # ----------------------------------------------------------------- helpers
    def log_line(self, text: str, level: str = "info") -> None:
        self.log.append({"ts": utc_now(), "level": level, "text": text})
        if len(self.log) > MAX_LOG_LINES:
            del self.log[: len(self.log) - MAX_LOG_LINES]

    def final_node(self) -> int | None:
        edges = self.graph.incoming(END)
        return edges[0].src if edges else None

    def inputs(self) -> dict[str, dict[str, list[int]]]:
        out: dict[str, dict[str, list[int]]] = {}
        key = {Kind.VIEW: "views_from", Kind.EDIT: "edits_from", Kind.REPORT: "reports_from"}
        for nid in sorted(self.graph.nodes):
            entry: dict[str, list[int]] = {"views_from": [], "edits_from": [], "reports_from": []}
            for edge in self.graph.incoming(nid):
                entry[key[edge.kind]].append(edge.src)
            out[str(nid)] = {k: sorted(v) for k, v in entry.items()}
        return out

    def rollouts(self) -> dict[str, dict[str, Any]]:
        """Every rollout the agents have saved so far, by id (read fresh from disk)."""
        found: dict[str, dict[str, Any]] = {}
        try:
            paths = sorted(p for p in self.rollout_dir.iterdir() if p.suffix == ".json")
        except OSError:
            return found
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                found[str(data.get("id") or path.stem)] = data
        return found

    def worker_rollouts(self, node_id: int) -> list[dict[str, Any]]:
        """The rollouts of one worker, oldest first (``config_path`` is ``graph/<type>/<id>``)."""
        from ace.webui.store import NotFound

        worker = self.workers.get(node_id)
        if worker is None:
            raise NotFound(f"worker {node_id} not found in demo {self.id!r}")
        wanted = f"graph/{worker.type}/{node_id}"
        rows = [r for r in self.rollouts().values()
                if str(r.get("config_path")) == wanted or r.get("id") in worker.rollout_ids]
        rows.sort(key=lambda r: (str(r.get("date_deployed") or ""), str(r.get("id") or "")))
        return rows

    def rollouts_written(self) -> int:
        try:
            return sum(1 for p in self.rollout_dir.iterdir() if p.suffix == ".json")
        except OSError:
            return 0

    # ---------------------------------------------------------------- payloads
    def summary(self) -> dict[str, Any]:
        end = self.finished_at or utc_now()
        counts = {"ok": 0, "failed": 0, "skipped": 0, "running": 0, "pending": 0}
        for w in self.workers.values():
            counts[w.status] = counts.get(w.status, 0) + 1
        return {
            "id": self.id,
            "run_id": self.run_id,
            "genome_id": self.genome_id,
            "prompt": self.prompt,
            "status": self.status,
            "model": self.model,
            "project_dir": str(self.project_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": seconds_between(self.started_at, end),
            "llm_calls": self.llm_calls,
            "rollouts": self.rollouts_written(),
            "workers_total": len(self.workers),
            "workers_done": counts["ok"] + counts["failed"] + counts["skipped"],
            "worker_counts": counts,
            "error": self.error,
        }

    def to_json(self) -> dict[str, Any]:
        from ace.webui.analysis import strip_worker_views

        data = self.summary()
        rollouts = self.rollouts()
        by_config: dict[str, list[str]] = {}
        for rid, r in sorted(rollouts.items(), key=lambda kv: (str(kv[1].get("date_deployed")), kv[0])):
            by_config.setdefault(str(r.get("config_path")), []).append(rid)
        recorded: dict[int, dict[str, Any]] = {}
        if self.record is not None:
            for w in strip_worker_views(self.record).get("workers") or []:
                if isinstance(w, dict):
                    recorded[int(w["node_id"])] = w
        workers: list[dict[str, Any]] = []
        for nid in sorted(self.workers):
            live = self.workers[nid].to_json()
            rids = list(live["rollout_ids"])
            for rid in by_config.get(f"graph/{live['type']}/{nid}", []):
                if rid not in rids:
                    rids.append(rid)
            steps = tool_calls = 0
            for rid in rids:
                if rid in rollouts:
                    a, t = _count_roles(rollouts[rid])
                    steps += a
                    tool_calls += t
            merged = dict(recorded.get(nid, {}))
            merged.update(live)
            merged["rollout_ids"] = rids
            merged["steps"] = steps
            merged["tool_calls"] = tool_calls
            merged["rollout_dir"] = str(self.rollout_dir)
            workers.append(merged)
        final = self.final_node()
        data.update({
            "order": list(self.order),
            "inputs": self.inputs(),
            "final_node": final,
            "workers": workers,
            "log": list(self.log),
            "failure": None,
            "failure_node_id": None,
            "failure_type": None,
            "diff": None,
            "touched_paths": [],
            "diff_chars": 0,
        })
        if self.record is not None:
            for key in ("failure", "failure_node_id", "failure_type", "diff", "touched_paths", "diff_chars"):
                data[key] = self.record.get(key)
        return data

    def rollout_path(self, rollout_id: str) -> Path:
        from ace.webui.store import NotFound, check_name

        check_name(rollout_id, "rollout id")
        name = rollout_id if rollout_id.endswith(".json") else f"{rollout_id}.json"
        root = self.rollout_dir.resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise NotFound(f"rollout {rollout_id!r} not found in demo {self.id!r}")
        return path

    def recorded_worker(self, node_id: int) -> dict[str, Any]:
        from ace.webui.store import NotFound

        if node_id not in self.workers:
            raise NotFound(f"worker {node_id} not found in demo {self.id!r}")
        for w in (self.record or {}).get("workers") or []:
            if isinstance(w, dict) and w.get("node_id") == node_id:
                return w
        return {"node_id": node_id, "view": None}


class DemoRunner:
    """Starts demos as asyncio tasks and keeps them for the lifetime of the server."""

    def __init__(self, settings: DemoSettings) -> None:
        self.settings = settings
        self.demos: dict[str, Demo] = {}

    # ------------------------------------------------------------------ queries
    def info(self) -> dict[str, Any]:
        running = self.running()
        return {
            "enabled": self.settings.project_ok(),
            "project_dir": str(self.settings.project_dir) if self.settings.project_dir else None,
            "model": self.settings.model,
            "api_base_url": self.settings.api_base_url,
            "key_env": self.settings.api_key_env_var,
            "key_set": self.settings.key_set(),
            "thinking": "off",
            "max_tokens": self.settings.max_tokens,
            "reason": self.settings.unavailable_reason(),
            "running": running.id if running else None,
        }

    def get(self, demo_id: str) -> Demo:
        from ace.webui.store import NotFound, check_name

        check_name(demo_id, "demo id")
        demo = self.demos.get(demo_id)
        if demo is None:
            raise NotFound(f"demo {demo_id!r} not found")
        return demo

    def running(self) -> Demo | None:
        return next((d for d in self.demos.values() if d.status == "running"), None)

    def list(self, run_id: str | None = None, genome_id: str | None = None) -> list[dict[str, Any]]:
        rows = [d for d in self.demos.values()
                if (run_id is None or d.run_id == run_id) and (genome_id is None or d.genome_id == genome_id)]
        return [d.summary() for d in sorted(rows, key=lambda d: d.started_at, reverse=True)]

    # ------------------------------------------------------------------ control
    def start(self, run_id: str, genome_id: str, graph: WorkerGraph, prompt: str) -> Demo:
        from ace.graph.properties import resolve

        reason = self.settings.unavailable_reason()
        if reason:
            raise DemoUnavailable(reason)
        prompt = (prompt or "").strip()
        if not prompt:
            raise DemoRejected("prompt is empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise DemoRejected(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
        busy = self.running()
        if busy is not None:
            raise DemoUnavailable(f"demo {busy.id} is still running; wait for it or cancel it")
        rollout_dir = Path(tempfile.mkdtemp(prefix="ace-demo-")) / "rollouts"
        rollout_dir.mkdir()
        demo = Demo(
            id=uuid4().hex, run_id=run_id, genome_id=genome_id, prompt=prompt, graph=graph,
            project_dir=Path(self.settings.project_dir).resolve(), rollout_dir=rollout_dir,
            model=self.settings.model, started_at=utc_now(),
        )
        for nid in sorted(graph.nodes):
            node = graph.nodes[nid]
            try:
                properties = resolve(node.type, node.properties)
            except Exception:  # noqa: BLE001 - show the raw gene when it does not resolve
                properties = dict(node.properties)
            demo.workers[nid] = WorkerState(node_id=nid, type=node.type.value, properties=properties)
        demo.log_line(f"demo {demo.id} started: {len(graph.nodes)} workers, model {demo.model}, project {demo.project_dir}")
        self.demos[demo.id] = demo
        demo.task = asyncio.create_task(self._execute(demo))
        return demo

    def cancel(self, demo_id: str) -> Demo:
        demo = self.get(demo_id)
        if demo.status == "running" and demo.task is not None:
            demo.log_line("cancel requested", "warn")
            demo.task.cancel()
        return demo

    # ---------------------------------------------------------------- execution
    def _client_factory(self, demo: Demo) -> Callable[[], Any]:
        base = self.settings.client_factory or make_client_factory(self.settings.model_settings())
        return lambda: _DemoClient(base(), self.settings, demo, _CURRENT_NODE.get())

    @staticmethod
    def _observer(demo: Demo, session: Any) -> Callable[[str, int, Any], None]:
        def observe(event: str, node_id: int, payload: Any) -> None:
            worker = demo.workers.get(node_id)
            if worker is None:
                return
            label = f"worker #{node_id} ({worker.type})"
            if event == "start":
                _CURRENT_NODE.set(node_id)
                worker.status = "running"
                worker.started_at = worker.last_activity = utc_now()
                demo.order.append(node_id)
                demo.log_line(f"{label} started")
            elif event == "done":
                worker.status = "ok"
                worker.finished_at = worker.last_activity = utc_now()
                worker.rollout_ids = list(getattr(payload, "rollout_ids", []) or [])
                worker.report = getattr(payload, "report", None)
                edit = getattr(payload, "edit", None)
                if edit is not None:
                    try:
                        worker.diff = edit.unified_diff(session)
                        worker.touched_paths = sorted(edit.touched_paths())
                    except Exception as exc:  # noqa: BLE001
                        demo.log_line(f"{label}: could not render its diff: {exc}", "warn")
                what = []
                if getattr(payload, "view", None) is not None:
                    what.append("a view")
                if edit is not None:
                    what.append(f"an edit of {len(worker.touched_paths)} file(s)")
                if worker.report:
                    what.append("a report")
                demo.log_line(f"{label} finished after {worker.llm_calls} LLM call(s): "
                              + (", ".join(what) if what else "no outputs"))
            elif event == "failed":
                worker.finished_at = worker.last_activity = utc_now()
                if isinstance(payload, asyncio.CancelledError):
                    worker.status = "skipped"
                    worker.error = "cancelled"
                    demo.log_line(f"{label} cancelled", "warn")
                else:
                    worker.status = "failed"
                    worker.error = f"{type(payload).__name__}: {payload}"
                    demo.log_line(f"{label} failed: {worker.error}", "error")
        return observe

    async def _execute(self, demo: Demo) -> None:
        from agent2.core.session import Session

        from ace.graph.runner import GraphRunner, RunContext

        session: Session | None = None
        result: Any = None
        failure: GraphFailure | None = None
        try:
            session = await asyncio.to_thread(Session, str(demo.project_dir), rollout_dir=str(demo.rollout_dir))
            demo.log_line(f"session ready: {len(getattr(session, 'files', {}) or {})} files indexed")
            runner = GraphRunner(demo.graph)
            ctx = RunContext(
                session=session, prompt=demo.prompt, settings=self.settings.model_settings(),
                client_factory=self._client_factory(demo), rollout_dir=demo.rollout_dir,
                observer=self._observer(demo, session),
            )
            result = await runner.run(ctx)
            demo.status = "done"
            demo.log_line(f"graph finished: edit of node #{demo.final_node()} reached END, "
                          f"{len(result.diff or '')} diff chars, {demo.llm_calls} LLM calls in total")
        except asyncio.CancelledError:
            demo.status = "cancelled"
            demo.error = "cancelled"
            demo.log_line("demo cancelled", "warn")
        except GraphFailure as exc:
            failure = exc
            demo.status = "failed"
            demo.error = f"{type(exc).__name__}: {exc}"
            demo.log_line(f"graph failed: {demo.error}", "error")
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI instead of lost in a task
            demo.status = "failed"
            demo.error = f"{type(exc).__name__}: {exc}"
            demo.log_line(f"demo failed: {demo.error}", "error")
        demo.finished_at = utc_now()
        for worker in demo.workers.values():
            if worker.status == "pending":
                worker.status = "skipped"
            elif worker.status == "running":
                worker.status = "failed" if demo.status == "failed" else "skipped"
                worker.error = worker.error or demo.error
                worker.finished_at = demo.finished_at
        if session is None:
            return
        try:
            demo.record = self._record(demo, session, result, failure)
        except Exception as exc:  # noqa: BLE001
            demo.log_line(f"could not build the demo record: {type(exc).__name__}: {exc}", "warn")

    def _record(self, demo: Demo, session: Any, result: Any, failure: GraphFailure | None) -> dict[str, Any]:
        from ace.neat.evaluation import Task, build_evaluation_record, nonempty_diff

        score = nonempty_diff(result.edit, session) if result is not None else 0.0
        return build_evaluation_record(
            genome=SimpleNamespace(id=demo.genome_id, generation=None),  # type: ignore[arg-type]
            graph=demo.graph,
            task_index=0,
            task=Task(project_dir=demo.project_dir, prompt=demo.prompt, scorer=nonempty_diff),
            session=session,
            rollout_dir=demo.rollout_dir,
            started_at=demo.started_at,
            finished_at=demo.finished_at or utc_now(),
            score=score,
            result=result,
            failure=failure,
        )


def default_project_dir() -> Path | None:
    """The agent2 smoke repository shipped with ACE, if present."""
    candidate = Path(__file__).resolve().parents[3] / "agent2" / "smoke_repos" / "schedule"
    return candidate if candidate.is_dir() else None


__all__ = [
    "DEFAULT_MODEL",
    "Demo",
    "DemoRejected",
    "DemoRunner",
    "DemoSettings",
    "DemoUnavailable",
    "WorkerState",
    "default_project_dir",
]
