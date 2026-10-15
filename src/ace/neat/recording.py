"""Run-level recording: ``run.json`` and lifecycle events (recording spec 11.1, 11.2, 11.6).

:class:`RunRecorder` owns ``run.json`` (rewritten atomically through
:meth:`History.log_run`) and the lifecycle events ``run_start``, ``generation_start``,
``generation_end``, ``run_end`` and ``error``. Per-evaluation events (``evaluation_start``,
``evaluation_end``) come from the evaluators, ``genome_evaluated`` and ``speciated`` from
the population; all go through the same :class:`History`.

Contract readings:

* ``run.json`` is written by :meth:`start`, after every :meth:`generation_finished` and by
  :meth:`finish`.
* ``generations_completed`` is the number of the last finished generation (generation 0 is
  the bootstrap, so a fresh run of ``cfg.generations = N`` ends with ``N``, matching
  ``config.generations``); it is 0 before any generation finished. On resume it starts at
  the restored generation once the first new generation finishes.
* ``best_genome_id`` / ``best_fitness`` track the highest ``best_fitness`` seen in any
  finished generation's stats (the population keeps elites, so this is also the final best).
* ``model`` holds exactly ``model``, ``api_base_url`` and ``api_key_env_var`` (the *name* of
  the environment variable); the API key itself is never read or written. ``None`` model
  settings (mock runs) give ``{"model": None, "api_base_url": None, "api_key_env_var": None}``.
* ``ace_version`` is ``importlib.metadata.version("ace")`` or ``"0"`` when unavailable.
"""
from __future__ import annotations

import importlib.metadata
import math
from typing import TYPE_CHECKING, Any

from ace.graph.clock import seconds_between, utc_now
from ace.neat.history import History

if TYPE_CHECKING:
    from ace.graph.llm import ModelSettings
    from ace.neat.config import NeatConfig

__all__ = ["RunRecorder", "ace_version", "model_info"]


def ace_version() -> str:
    """The installed ``ace`` distribution version, or ``"0"``."""
    try:
        return importlib.metadata.version("ace")
    except Exception:  # noqa: BLE001 - metadata may be missing in odd installs
        return "0"


def model_info(model: ModelSettings | None) -> dict[str, Any]:
    """The model dict written to ``run.json`` (never contains an API key)."""
    return {
        "model": getattr(model, "model", None),
        "api_base_url": getattr(model, "api_base_url", None),
        "api_key_env_var": getattr(model, "api_key_env_var", None),
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


class RunRecorder:
    """Writes ``run.json`` and the run lifecycle events of one evolution run."""

    def __init__(
        self,
        history: History,
        *,
        run_id: str,
        config: NeatConfig,
        model: ModelSettings | None,
        evaluator_info: dict,
        seed: int,
        resumed_from: str | None = None,
    ) -> None:
        self.history = history
        self.run_id = run_id
        self.config = config
        self.model = model
        self.evaluator_info = dict(evaluator_info)
        self.seed = seed
        self.resumed_from = None if resumed_from is None else str(resumed_from)
        self.status = "running"
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.generations_completed = 0
        self.best_genome_id: str | None = None
        self.best_fitness: float | None = None
        self.error: str | None = None
        self._generation_started: dict[int, str] = {}

    # ---------------------------------------------------------------- record

    def record(self) -> dict[str, Any]:
        """The current ``run.json`` content."""
        duration = (
            seconds_between(self.started_at, self.finished_at)
            if self.started_at and self.finished_at
            else None
        )
        return {
            "run_id": self.run_id,
            "ace_version": ace_version(),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": duration,
            "resumed_from": self.resumed_from,
            "seed": self.seed,
            "config": self.config.to_toml_dict(),
            "model": model_info(self.model),
            "evaluator": self.evaluator_info,
            "generations_completed": self.generations_completed,
            "best_genome_id": self.best_genome_id,
            "best_fitness": self.best_fitness,
            "error": self.error,
        }

    def _write(self) -> None:
        self.history.log_run(self.record())

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Mark the run as running, write ``run.json`` and the ``run_start`` event."""
        self.status = "running"
        self.started_at = utc_now()
        self.finished_at = None
        self.error = None
        self._write()
        self.history.log_event("run_start", run_id=self.run_id)

    def generation_started(self, generation: int) -> None:
        """Log ``generation_start``."""
        self._generation_started[generation] = utc_now()
        self.history.log_event("generation_start", generation=generation)

    def generation_finished(self, generation: int, stats: dict) -> None:
        """Log ``generation_end`` and rewrite ``run.json`` with the new progress."""
        duration = _finite(stats.get("duration_s"))
        if duration is None and generation in self._generation_started:
            duration = seconds_between(self._generation_started[generation], utc_now())
        self._generation_started.pop(generation, None)
        best = _finite(stats.get("best_fitness"))
        best_id = stats.get("best_genome_id", stats.get("best_id"))
        if best is not None and (self.best_fitness is None or best > self.best_fitness):
            self.best_fitness = best
            self.best_genome_id = best_id
        self.generations_completed = max(self.generations_completed, int(generation))
        self.history.log_event(
            "generation_end", generation=generation, duration_s=duration, best_fitness=best
        )
        self._write()

    def finish(self, status: str = "finished", error: str | None = None) -> None:
        """Log ``error`` (if any) and ``run_end``; write the final ``run.json``."""
        if self.started_at is None:
            self.started_at = utc_now()
        self.status = status
        self.error = error
        self.finished_at = utc_now()
        if error is not None:
            self.history.log_event("error", message=error)
        self.history.log_event(
            "run_end",
            status=status,
            duration_s=seconds_between(self.started_at, self.finished_at),
        )
        self._write()
