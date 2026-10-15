"""Worker implementations and their registry. See GRAPH_NEAT_DESIGN.md section 9.

``WORKERS`` maps each :class:`~ace.graph.types.WorkerType` to its
:class:`~ace.graph.workers.base.Worker` subclass. Concrete modules register their
class with the :func:`register` decorator::

    from ace.graph.workers import register
    from ace.graph.workers.base import Worker

    @register
    class ViewerWorker(Worker):
        type = WorkerType.VIEWER
        ...

Loading is lazy so that ``ace.graph.workers.base`` (and the runner) can be imported
even while concrete worker modules are missing or broken: the concrete modules
(``viewer``, ``listviewer``, ``editor``, ``vieweditor``, ``critic``, ``judgemerger``,
``pickmerger``) are imported by :func:`load_workers`, which runs on the first access of
the ``WORKERS`` attribute (``from ace.graph.workers import WORKERS`` triggers it).
Each import is guarded by ``try/except ImportError``: a module that does not exist yet,
or fails to import, is skipped and its error is kept in :data:`IMPORT_ERRORS` (module
name -> error) so the runner can report why a worker type has no implementation.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from ace.graph.types import WorkerType
    from ace.graph.workers.base import Worker

WORKER_MODULES: tuple[str, ...] = (
    "viewer",
    "listviewer",
    "editor",
    "vieweditor",
    "critic",
    "judgemerger",
    "pickmerger",
)
"""Concrete worker modules imported by :func:`load_workers`, in order."""

_REGISTRY: dict[WorkerType, type[Worker]] = {}

if TYPE_CHECKING:
    WORKERS: dict[WorkerType, type[Worker]]
    """The registry (served lazily by the module ``__getattr__``)."""

IMPORT_ERRORS: dict[str, ImportError] = {}
"""Module name -> the ImportError raised while loading it (missing or broken modules)."""

_loaded = False

W = TypeVar("W", bound="type[Worker]")


def register(cls: W) -> W:
    """Class decorator: insert ``cls`` into the registry under ``cls.type``."""
    _REGISTRY[cls.type] = cls
    return cls


def load_workers() -> dict[WorkerType, type[Worker]]:
    """Import every concrete worker module once (skipping import failures); return the registry."""
    global _loaded
    if not _loaded:
        _loaded = True  # set first: concrete modules may import this package re-entrantly
        for name in WORKER_MODULES:
            try:
                importlib.import_module(f"{__name__}.{name}")
            except ImportError as exc:
                IMPORT_ERRORS[name] = exc
            else:
                IMPORT_ERRORS.pop(name, None)
    return _REGISTRY


def __getattr__(name: str) -> Any:
    """Module attribute hook: ``WORKERS`` loads the concrete modules on first access."""
    if name == "WORKERS":
        return load_workers()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
