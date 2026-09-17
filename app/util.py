"""Shared helper."""

import time


def elapsed_ms(start: float) -> int:
    """Milliseconds since a time.perf_counter() mark."""
    return round((time.perf_counter() - start) * 1000)
