"""Aware UTC clocks for operational metadata, separate from event-time policies."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return an aware UTC datetime; never infer a source publication timezone."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return a second-precision UTC ISO timestamp for generated audit artifacts."""
    return utc_now().isoformat(timespec="seconds")
