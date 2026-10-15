"""Read-only analysis WebUI over NEAT run directories (GRAPH_NEAT_DESIGN.md / spec section 12).

``create_app(runs_root)`` (in :mod:`ace.webui.app`) builds the FastAPI application;
:mod:`ace.webui.store` loads and tails run directories; :mod:`ace.webui.analysis` holds the
pure functions behind the derived endpoints. Nothing here ever writes to a run directory.
"""
from __future__ import annotations

__all__ = ["create_app"]


def create_app(runs_root):  # type: ignore[no-untyped-def]
    """Lazy wrapper around :func:`ace.webui.app.create_app` (keeps FastAPI import optional)."""
    from ace.webui.app import create_app as _create_app

    return _create_app(runs_root)
