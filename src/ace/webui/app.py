"""FastAPI application of the read-only analysis WebUI (spec section 12.1).

``create_app(runs_root)`` exposes the JSON API under ``/api``, serves ``/`` (``index.html``
or a placeholder while the frontend does not exist) and ``/static/*`` from
``ace/webui/static``. Unknown ids answer 404; ids that are not plain file/directory names
(``..``, anything with a path separator or a leading dot) answer 400.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from ace.webui import analysis
from ace.webui.store import InvalidName, NotFound, Run, RunStore, check_name

STATIC_DIR = Path(__file__).resolve().parent / "static"
PLACEHOLDER = (
    '<!doctype html><html><head><meta charset="utf-8"><title>ACE runs</title></head>'
    "<body>ACE WebUI backend is running; the frontend is not installed yet (see /api/runs).</body></html>"
)


def render_worker_view(project_dir: str | None, view_json: dict[str, Any] | None) -> str | None:
    """Render ``view_json`` against a fresh Session of ``project_dir`` (None if impossible).

    The Session's rollout directory is an absolute path inside a temporary directory, so
    nothing is written into the task project.
    """
    if not view_json or not project_dir or not Path(project_dir).is_dir():
        return None
    from agent2.core.session import Session

    from ace.graph.rendering import render_view
    from ace.graph.views import View

    with tempfile.TemporaryDirectory(prefix="ace-webui-") as tmp:
        try:
            session = Session(str(project_dir), rollout_dir=str(Path(tmp).resolve() / "rollouts"))
            return render_view(session, View.from_json(view_json))
        except Exception:  # noqa: BLE001 - a project that no longer parses: no rendering
            return None


def create_app(runs_root: Path) -> FastAPI:
    """Build the WebUI app over the run directories under ``runs_root``."""
    runs_root = Path(runs_root)
    store = RunStore(runs_root)
    app = FastAPI(title="ACE runs", version="0.1.0")
    app.state.store = store

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidName)
    async def _invalid(_: Request, exc: InvalidName) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def get_run(run_id: str) -> Run:
        return store.get(run_id)

    def graph_or_404(run: Run, record: dict[str, Any]):  # type: ignore[no-untyped-def]
        graph = analysis.genome_graph(record)
        if graph is None:
            raise NotFound(f"genome {record.get('id')!r} of run {run.run_id!r} has no genome body")
        return graph

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "runs_root": str(runs_root)}

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        rows = [
            analysis.run_summary(run.run_id, run.run_json(), run.generation_records(), run.status())
            for run in store.runs()
        ]
        # newest first, runs without a start time last (by run id)
        dated = sorted((r for r in rows if r["started_at"]), key=lambda r: r["started_at"], reverse=True)
        undated = [r for r in rows if not r["started_at"]]
        return dated + undated

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        data = dict(run.run_json() or {})
        index = run.index()
        if not data:
            data = {"run_id": run.run_id, "status": run.status()}
        species_ids = {sp.get("id") for snap in index.species for sp in snap.get("species") or []}
        data["counts"] = {
            "genomes": len(index.genomes),
            "evaluations": len(run.evaluation_records()),
            "generations": len(index.generations),
            "species": len(species_ids),
        }
        return data

    @app.get("/api/runs/{run_id}/progress")
    async def run_progress(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        return analysis.progress(run.generation_records(), run.events(), run.status())

    @app.get("/api/runs/{run_id}/generations/{generation}")
    async def generation_detail(run_id: str, generation: int) -> dict[str, Any]:
        return get_run(run_id).generation(generation)

    @app.get("/api/runs/{run_id}/species")
    async def species(run_id: str) -> dict[str, Any]:
        return analysis.species_timeline(get_run(run_id).species_snapshots())

    # --------------------------------------------------------------- genomes
    @app.get("/api/runs/{run_id}/genomes")
    async def list_genomes(
        run_id: str,
        generation: int | None = None,
        species: int | None = None,
        min_fitness: float | None = None,
        sort: str | None = None,
        limit: int = Query(100, ge=0, le=10000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        return analysis.filter_genomes(
            get_run(run_id).genome_records(),
            generation=generation,
            species=species,
            min_fitness=min_fitness,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{run_id}/genomes/{genome_id}")
    async def genome_detail(run_id: str, genome_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        check_name(genome_id, "genome id")
        record = run.genome(genome_id)
        index = run.index()
        graph = analysis.genome_graph(record)
        return {
            **record,
            "graph": graph.to_json() if graph is not None else None,
            "layout": analysis.layout(graph) if graph is not None else None,
            "lineage": analysis.lineage(index.genomes, genome_id),
            "children": list(index.children.get(genome_id, [])),
            "evaluations": [analysis.strip_worker_views(r) for r in index.evaluations_by_genome.get(genome_id, [])],
        }

    @app.get("/api/runs/{run_id}/genomes/{genome_id}/dot", response_class=PlainTextResponse)
    async def genome_dot(run_id: str, genome_id: str) -> PlainTextResponse:
        run = get_run(run_id)
        check_name(genome_id, "genome id")
        graph = graph_or_404(run, run.genome(genome_id))
        return PlainTextResponse(analysis.to_dot(graph, genome_id), media_type="text/vnd.graphviz")

    # ----------------------------------------------------------- evaluations
    @app.get("/api/runs/{run_id}/evaluations")
    async def list_evaluations(
        run_id: str,
        genome: str | None = None,
        task: int | None = None,
        failed: bool | None = None,
        limit: int = Query(100, ge=0, le=10000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        return analysis.filter_evaluations(
            get_run(run_id).evaluation_records(),
            genome=genome,
            task=task,
            failed=failed,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{run_id}/evaluations/{genome_id}/{task_index}")
    async def evaluation_detail(run_id: str, genome_id: str, task_index: int) -> dict[str, Any]:
        check_name(genome_id, "genome id")
        return get_run(run_id).evaluation(genome_id, task_index)

    @app.get("/api/runs/{run_id}/evaluations/{genome_id}/{task_index}/workers/{node_id}/view")
    def worker_view(run_id: str, genome_id: str, task_index: int, node_id: int) -> dict[str, Any]:
        # Sync on purpose: rendering parses the project, so FastAPI runs it in a thread.
        check_name(genome_id, "genome id")
        record = get_run(run_id).evaluation(genome_id, task_index)
        worker = next(
            (w for w in record.get("workers") or [] if isinstance(w, dict) and w.get("node_id") == node_id),
            None,
        )
        if worker is None:
            raise NotFound(f"worker {node_id} not found in evaluation ({genome_id!r}, task {task_index})")
        view = worker.get("view")
        task = record.get("task") if isinstance(record.get("task"), dict) else {}
        return {"view": view, "rendered": render_worker_view(task.get("project_dir"), view)}

    @app.get("/api/runs/{run_id}/rollouts/{genome_id}/{task_index}/{rollout_id}")
    async def rollout(run_id: str, genome_id: str, task_index: int, rollout_id: str) -> Any:
        path = get_run(run_id).rollout_path(genome_id, task_index, rollout_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, detail=f"rollout {rollout_id!r} is not readable JSON: {exc}") from exc

    # -------------------------------------------------------------- analysis
    @app.get("/api/runs/{run_id}/operators")
    async def operators(run_id: str) -> list[dict[str, Any]]:
        run = get_run(run_id)
        index = run.index()
        return analysis.operator_stats(run.genome_records(), index.genomes)

    @app.get("/api/runs/{run_id}/compare")
    async def compare(run_id: str, a: str, b: str) -> dict[str, Any]:
        run = get_run(run_id)
        check_name(a, "genome id")
        check_name(b, "genome id")
        record_a, record_b = run.genome(a), run.genome(b)
        graph_or_404(run, record_a)
        graph_or_404(run, record_b)
        return analysis.compare_genomes(record_a, record_b, analysis.speciation_config(run.run_json()))

    @app.get("/api/runs/{run_id}/innovations")
    async def innovations(run_id: str) -> dict[str, Any]:
        return analysis.innovations_table(get_run(run_id).innovations())

    @app.get("/api/runs/{run_id}/failures")
    async def failures(run_id: str) -> list[dict[str, Any]]:
        return analysis.failures(get_run(run_id).evaluation_records())

    # ---------------------------------------------------------------- static
    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse(PLACEHOLDER)

    @app.get("/static/{asset_path:path}", include_in_schema=False)
    async def static_asset(asset_path: str) -> FileResponse:
        """Serve ``webui/static`` files; resolved per request so files may appear later."""
        root = STATIC_DIR.resolve()
        candidate = (root / asset_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(404, detail=f"{asset_path!r} not found")
        return FileResponse(candidate)

    return app


__all__ = ["create_app", "render_worker_view"]
