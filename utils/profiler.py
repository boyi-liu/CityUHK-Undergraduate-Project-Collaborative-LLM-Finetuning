"""
WindowProfiler: tracks trainable window durations per client.

Records completed windows (open→close) and provides estimates for
budget planning in adaptive training algorithms.
"""

from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple


class WindowProfiler:
    """
    Tracks per-client windows and estimates remaining window time.

    Usage
    -----
    - ``record(client_id, open_time, close_time)`` — append a single window.
    - ``seed(client_id, windows)``                 — pre-populate with a client's windows.
    - ``estimate(client_id, current_time)``        — returns mean remaining window duration
                                                     in minutes for windows that contain
                                                     current_time's time-of-day, excluding
                                                     windows on the same calendar day as
                                                     current_time, or None.
    """

    def __init__(self) -> None:
        # client_id -> list of (open_time, close_time) pairs
        self._windows: Dict[str, List[Tuple[datetime, datetime]]] = {}

    def record(self, client_id, open_time: datetime, close_time: datetime) -> None:
        """Record a completed window (open → close)."""
        if (close_time - open_time).total_seconds() <= 0:
            return
        key = str(client_id)
        self._windows.setdefault(key, []).append((open_time, close_time))

    def seed(self, client_id, windows: List[dict]) -> None:
        """
        Pre-populate with historical windows.

        Parameters
        ----------
        windows : list of dict
            Each element must have ``"start"`` and ``"end"`` keys (datetime).
        """
        for w in windows:
            self.record(client_id, w['start'], w['end'])

    @staticmethod
    def _tod_seconds(dt: datetime) -> float:
        """Time-of-day in seconds (ignores date)."""
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6

    def estimate(self, client_id, current_time: datetime) -> Optional[float]:
        """
        Return the mean remaining window duration (in minutes) across all
        windows whose time-of-day range contains current_time's time-of-day,
        drawn from both past and future days but excluding any window on the
        same calendar day as current_time.  Returns None if no matching
        windows are found.
        """
        key = str(client_id)
        recorded = self._windows.get(key)
        if not recorded:
            return None

        cur_tod  = self._tod_seconds(current_time)
        cur_date = current_time.date()

        remainings = []
        for open_t, close_t in recorded:
            if open_t.date() == cur_date:   # exclude the current day's window
                continue
            open_tod  = self._tod_seconds(open_t)
            close_tod = self._tod_seconds(close_t)
            if open_tod <= cur_tod <= close_tod:
                remaining_min = (close_tod - cur_tod) / 60.0
                remainings.append(remaining_min)

        return mean(remainings) if remainings else None
