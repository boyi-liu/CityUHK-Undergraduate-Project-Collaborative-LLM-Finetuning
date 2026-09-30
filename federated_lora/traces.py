"""Reconstruct availability using the original repository's OR/charging/Wi-Fi rule."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import groupby
from pathlib import Path


STATE_EVENTS = {
    "screen_off": ("off", True), "screen_on": ("off", False),
    "screen_lock": ("locked", True), "screen_unlock": ("locked", False),
    "battery_charged_on": ("charging", True),
    "battery_charged_off": ("charging", False), "wifi": ("wifi", True),
    "Unknown": ("wifi", False), "2G": ("wifi", False),
    "3G": ("wifi", False), "4G": ("wifi", False), "5G": ("wifi", False),
}


def available(state: dict) -> bool:
    return ((state["off"] is True or state["locked"] is True)
            and state["charging"] is True and state["wifi"] is True)


def parse_device(record: dict, min_window_seconds: float = 60) -> dict:
    """Stable-sort events; resolve a timestamp's state changes before opening a window.

    As in the inherited parser, a state persists until its next recorded change.
    No interval is inferred beyond the first/last observation. Battery percentages
    are metadata, not an additional eligibility threshold. Short-window filtering
    is offline and uses future trace information, as does the original code.
    """
    events, seen = [], set()
    stats = {"lines": 0, "malformed": 0, "duplicates": 0,
             "out_of_order": 0, "ignored_events": 0, "short_windows": 0}
    previous = None
    messages = record["messages"]
    if isinstance(messages, list):
        messages = "".join(messages)
    for line in messages.splitlines():
        if not line.strip():
            continue
        stats["lines"] += 1
        try:
            stamp, event = line.strip().split("\t")
            stamp = datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M:%S")
            event = event.strip()
        except ValueError:
            stats["malformed"] += 1
            continue
        if previous is not None and stamp < previous:
            stats["out_of_order"] += 1
        previous = stamp
        if (stamp, event) in seen:
            stats["duplicates"] += 1
            continue
        seen.add((stamp, event))
        events.append((stamp, event))
    if not events:
        raise ValueError("Device trace has no valid timestamped events")
    events.sort(key=lambda x: x[0])
    state = dict.fromkeys(("off", "locked", "charging", "wifi"))
    windows, start = [], None

    def close(end):
        if (end - start).total_seconds() >= min_window_seconds:
            windows.append({"id": len(windows), "start": start.isoformat(), "end": end.isoformat()})
        else:
            stats["short_windows"] += 1

    for stamp, same_time in groupby(events, key=lambda x: x[0]):
        was_available = available(state)
        for _, event in same_time:
            if event in STATE_EVENTS:
                field, value = STATE_EVENTS[event]
                state[field] = value
            else:
                stats["ignored_events"] += 1
        now_available = available(state)
        if now_available and not was_available:
            start = stamp
        elif was_available and not now_available:
            close(stamp)
            start = None
    if start is not None:
        close(events[-1][0])
    return {"model": record.get("model"), "coverage_start": events[0][0].isoformat(),
            "coverage_end": events[-1][0].isoformat(), "windows": windows, "stats": stats}


def load_traces(path: Path, min_window_seconds: float = 60) -> dict:
    if min_window_seconds < 0:
        raise ValueError("Minimum window duration cannot be negative")
    raw = path.read_bytes()
    records = json.loads(raw)
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_repository_commit": "9f5a4eb6ec7ec1e31b6a61d6c6926f39489298fb",
        "provenance": "Inherited trace file; original collection source/timezone unverified",
        "availability_rule": "(screen_off OR screen_locked) AND charging AND wifi",
        "min_window_seconds": min_window_seconds,
        "state_gap_policy": "Last observed state persists within recorded coverage, as in original parser",
        "devices": {str(key): parse_device(value, min_window_seconds) for key, value in records.items()},
    }
