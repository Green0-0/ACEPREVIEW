"""Frozen, shared agent2 sessions for evaluation runs.

agent2's ``Session`` re-scans and re-reads the whole project folder on every tool call
(``refresh_files``), and every task run builds a new Session that parses every file again.
On SWE-bench-sized repositories on NFS that dominates the CPU time of a run. Task projects
are never written during evaluation (every worker edits a private ``Workspace``), so:

* :class:`FrozenSession` loads the project once and makes ``refresh_files`` a no-op after
  the initial load. The pristine base can therefore never drift mid-run.
* :class:`SessionCache` keeps one loaded base session per project directory (LRU) and hands
  out cheap forks: a fork shares the base's ``files`` and ``configs`` but has its own
  rollout directory and rollout ``datastore``, so concurrent runs on the same project do
  not see each other's rollouts.

Sharing ``files`` is safe because nothing in the evaluation path mutates session files:
workspaces clone a file before editing it and ``Workspace.commit`` is never called.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from agent2.core.session import Session


class FrozenSession(Session):
    """A Session whose files are loaded once and never refreshed from disk."""

    _frozen = False

    def __init__(self, project_dir: str, config_dir: str = "config/agents",
                 rollout_dir: str = ".agent2/task") -> None:
        super().__init__(project_dir, config_dir=config_dir, rollout_dir=rollout_dir)
        self._frozen = True

    def refresh_files(self) -> None:
        if self._frozen:
            return
        super().refresh_files()

    def fork(self, rollout_dir: str | Path) -> FrozenSession:
        """A session sharing this one's files and configs, with its own rollout store."""
        child = FrozenSession.__new__(FrozenSession)
        child.configs = self.configs
        child.datastore = OrderedDict()
        child.files = self.files
        child.project_path = self.project_path
        child.agent2_path = self.agent2_path
        child.config_path = self.config_path
        child.rollout_path = (self.project_path / rollout_dir).resolve()
        child.rollout_path.mkdir(parents=True, exist_ok=True)
        child._frozen = True
        child._base = self
        return child

    def codegraph(self):
        """The (cached) CodebaseGraph of the shared files; built on first use."""
        base = getattr(self, "_base", None) or self
        graph = getattr(base, "_codegraph_cache", None)
        if graph is None:
            from ace.graph.codegraph import CodebaseGraph

            graph = CodebaseGraph(base)
            base._codegraph_cache = graph
        return graph


class SessionCache:
    """LRU cache of frozen base sessions keyed by resolved project directory."""

    def __init__(self, max_entries: int = 24) -> None:
        self.max_entries = max_entries
        self._bases: OrderedDict[str, FrozenSession] = OrderedDict()
        self._lock = threading.Lock()
        self.loads = 0
        self.hits = 0

    def base(self, project_dir: str | Path, scratch_rollouts: str | Path) -> FrozenSession:
        key = str(Path(project_dir).resolve())
        with self._lock:
            if key in self._bases:
                self._bases.move_to_end(key)
                self.hits += 1
                return self._bases[key]
        base = FrozenSession(key, rollout_dir=str(Path(scratch_rollouts).resolve()))
        with self._lock:
            self.loads += 1
            self._bases[key] = base
            self._bases.move_to_end(key)
            while len(self._bases) > self.max_entries:
                self._bases.popitem(last=False)
        return base

    def session(self, project_dir: str | Path, rollout_dir: str | Path) -> FrozenSession:
        """A fork of the cached base for ``project_dir`` with ``rollout_dir`` (absolute)."""
        rollout_dir = Path(rollout_dir).resolve()
        base = self.base(project_dir, rollout_dir.parent / "_base")
        return base.fork(rollout_dir)


__all__ = ["FrozenSession", "SessionCache"]
