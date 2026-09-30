"""Prepare a reproducible chronological sequence of sessions and client traces."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from .traces import load_traces


def prepare_sessions(config: dict, root: Path) -> dict:
    settings = config["sessions"]
    traces = load_traces(root / settings["trace_path"], settings.get("min_window_seconds", 60))
    ids = [str(x) for x in settings["trace_ids"]]
    if len(ids) != config["federation"]["client_count"] or len(set(ids)) != len(ids):
        raise ValueError("Each client needs one distinct trace ID")
    if any(key not in traces["devices"] for key in ids):
        raise ValueError("A mapped device trace is missing")
    day = date.fromisoformat(settings["start_date"])
    start_clock, end_clock = time.fromisoformat(settings["start_time"]), time.fromisoformat(settings["end_time"])
    if start_clock.tzinfo or end_clock.tzinfo or settings["time_basis"] != "recorded_local":
        raise ValueError("Use recorded_local timestamps; the inherited timezone is unknown")
    if settings["count"] < 1:
        raise ValueError("At least one session is required")
    sessions = []
    for index in range(settings["count"]):
        start = datetime.combine(day + timedelta(days=index), start_clock)
        end = datetime.combine(start.date(), end_clock)
        if end <= start:
            end += timedelta(days=1)
        sessions.append({"index": index, "start": start.isoformat(), "deadline": end.isoformat()})
    epoch = datetime.fromisoformat(sessions[0]["start"])

    def ticks(stamp):
        return round((datetime.fromisoformat(stamp) - epoch).total_seconds() * 1_000_000)

    clients = []
    for client_id, key in enumerate(ids):
        device = traces["devices"][key]
        if device["coverage_start"] > sessions[0]["start"] or device["coverage_end"] < sessions[-1]["deadline"]:
            raise ValueError(f"Trace {key} does not cover all requested sessions; choose covered dates/IDs")
        windows = [{**w, "start_us": ticks(w["start"]), "end_us": ticks(w["end"])}
                   for w in device["windows"]]
        clients.append({"client_id": client_id, "trace_id": key, **device, "windows": windows})
    for session in sessions:
        session.update(start_us=ticks(session["start"]), deadline_us=ticks(session["deadline"]))
        session["client_available_seconds"] = [
            sum(max(0, min(w["end_us"], session["deadline_us"]) - max(w["start_us"], session["start_us"]))
                for w in client["windows"]) / 1_000_000 for client in clients]
    return {"format_version": 1, **{k: v for k, v in traces.items() if k != "devices"},
            "time_basis": "recorded_local (timezone unverified)", "epoch": epoch.isoformat(),
            "continuity": "global_adapter", "sessions": sessions, "clients": clients}


def write_schedule(manifest: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    row_height, left, width = 25, 175, 780
    panel_height = 80 + row_height * len(manifest["clients"])
    height = 35 + panel_height * len(manifest["sessions"])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#142c46">',
             '<text x="20" y="22">Recorded availability: (screen off OR locked) AND charging AND Wi-Fi</text>']
    for session in manifest["sessions"]:
        top = 35 + session["index"] * panel_height
        span = session["deadline_us"] - session["start_us"]
        parts.append(f'<text x="20" y="{top + 18}">{escape(session["start"])} to {escape(session["deadline"])}</text>')
        for client in manifest["clients"]:
            y = top + 40 + client["client_id"] * row_height
            label = f'Client {client["client_id"]} / trace {client["trace_id"]}'
            parts.append(f'<text x="20" y="{y + 13}">{escape(label)}</text><rect x="{left}" y="{y}" width="{width}" height="17" fill="#edf1f5"/>')
            for window in client["windows"]:
                start = max(window["start_us"], session["start_us"])
                end = min(window["end_us"], session["deadline_us"])
                if end > start:
                    x = left + width * (start - session["start_us"]) / span
                    size = width * (end - start) / span
                    parts.append(f'<rect x="{x:.3f}" y="{y}" width="{size:.3f}" height="17" fill="#198377"><title>{escape(window["start"])} to {escape(window["end"])}</title></rect>')
    parts.append('</g></svg>')
    (directory / "availability.svg").write_text("\n".join(parts), encoding="utf-8")
