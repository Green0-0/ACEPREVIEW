"""Shared wall-clock helper for run recording (ISO-8601 UTC timestamps)."""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["utc_now", "parse_ts", "seconds_between"]


def utc_now() -> str:
    """The current time as an ISO-8601 UTC string, e.g. ``2026-10-15T12:00:00.123456+00:00``."""
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: str) -> datetime:
    """Inverse of :func:`utc_now`."""
    return datetime.fromisoformat(value)


def seconds_between(start: str, end: str) -> float:
    """Float seconds from ``start`` to ``end`` (both :func:`utc_now` strings)."""
    return (parse_ts(end) - parse_ts(start)).total_seconds()
