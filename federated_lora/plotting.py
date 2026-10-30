"""Availability and MMLU plots from matching, measured experiment outputs.

Missing GPU results produce explicitly empty panels, never invented scores.
Scheduling dry runs are not used as accuracy data.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from .comparison import CONDITIONS, MODES, comparison_configs
from .sessions import prepare_sessions, run_directory, write_schedule


COLORS = {"sync": "#2764a5", "async": "#d36b22"}


def _pyplot():
    # Keep plotting caches inside the experiment output, including on servers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    return plt


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, directory, name):
    fig.canvas.draw()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")


def _availability_plot(plt, manifest, directory):
    sessions, clients = manifest["sessions"], manifest["clients"]
    fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.2 + 0.29 * len(clients) * len(sessions)),
                             squeeze=False, layout="constrained")
    for ax, session in zip(axes[:, 0], sessions):
        hours = (session["deadline_us"] - session["start_us"]) / 3_600_000_000
        for client in clients:
            y = client["client_id"]
            ax.broken_barh([(0, hours)], (y - 0.34, 0.68), facecolors="#edf1f5")
            intervals = []
            for window in client["windows"]:
                start = max(window["start_us"], session["start_us"])
                end = min(window["end_us"], session["deadline_us"])
                if end > start:
                    intervals.append(((start - session["start_us"]) / 3_600_000_000,
                                      (end - start) / 3_600_000_000))
            ax.broken_barh(intervals, (y - 0.34, 0.68), facecolors="#198377")
        ax.set(yticks=list(range(len(clients))),
               yticklabels=[f"Client {c['client_id']} / trace {c['trace_id']}" for c in clients],
               xlim=(0, hours), ylim=(len(clients) - 0.5, -0.6),
               xlabel="Hours since session start (recorded local time)",
               title=f"Session {session['index'] + 1}: {session['start']} to {session['deadline']}")
        ax.set_xticks(list(range(math.floor(hours) + 1)))
        ax.grid(axis="x", alpha=0.2)
        ax.axvline(hours, color="#9a3340", linestyle="--", linewidth=1.5, clip_on=False)
    fig.suptitle("Device availability from recorded usage traces\n"
                 "Green: (screen off OR locked) AND charging AND Wi-Fi; grey: unavailable", fontsize=14)
    _save_figure(fig, directory, "device-availability")
    plt.close(fig)


def _load_results(config, root, manifests):
    results, missing, shared = {}, [], None
    for candidate in comparison_configs(config):
        condition, mode = candidate["sessions"]["availability"], candidate["federation"]["mode"]
        directory = run_directory(candidate, root)
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            missing.append(f"{condition}/{mode}")
            continue
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        if (report.get("simulation_only") or not report.get("completed")
                or report.get("availability") != condition or report.get("mode") != mode
                or report.get("participation") != "repeated"
                or identity.get("scheduler_format_version") != 2
                or identity["config"] != candidate
                or identity["sessions"] != manifests[condition]):
            raise ValueError(f"Unmatched or incomplete experiment: {directory}")
        # Never join scores from different adapters, data, or evaluation protocols.
        comparable = {key: identity[key] for key in
                      ("data", "client_content_sha256", "initial_adapter_sha256", "evaluation", "adapter_bytes")}
        if shared is None:
            shared = comparable
        elif shared != comparable:
            raise ValueError("Comparison runs differ in initial adapter, client data, or MMLU protocol")
        points = [{"session": -1, "kind": "initial", "time_us": 0, "version": 0,
                   "elapsed_session_seconds": 0, "evaluation": report["initial_evaluation"]}]
        points += report["evaluations"]
        expected = []
        elapsed_us = 0
        interval = candidate["evaluation"]["interval_seconds"]
        for session in manifests[condition]["sessions"]:
            offsets = ([] if interval is None else
                       list(range(round(interval * 1_000_000),
                                  session["deadline_us"] - session["start_us"], round(interval * 1_000_000))))
            offsets.append(session["deadline_us"] - session["start_us"])
            for offset in offsets:
                expected.append((session["index"], session["start_us"] + offset,
                                 (elapsed_us + offset) / 1_000_000,
                                 "deadline" if offset == offsets[-1] else "periodic"))
            elapsed_us += offsets[-1]
        actual = [(p["session"], p["time_us"], p["elapsed_session_seconds"], p["kind"])
                  for p in report["evaluations"]]
        if actual != expected or len(report["sessions"]) != len(manifests[condition]["sessions"]):
            raise ValueError(f"Missing, duplicate, or mistimed evaluation points: {directory}")
        for point in points:
            metrics = point["evaluation"]
            score = metrics["mmlu_accuracy"]
            if (not math.isfinite(score) or not 0 <= score <= 1
                    or metrics["scope"] != identity["evaluation"]["scope"]
                    or metrics["count"] != identity["evaluation"]["evaluated_questions"]):
                raise ValueError(f"Invalid MMLU score or scope in {directory}")
        results[(condition, mode)] = {"report": report, "points": points}
    return results, missing


def plot_comparison(config: dict, root: Path) -> dict:
    """Write three plots, measured point CSVs and deadline comparisons.

    The x axis accumulates session hours only; gaps between nights are excluded.
    Both accuracy plots use the same y limits and mark every session deadline.
    """
    candidates = list(comparison_configs(config))
    manifests = {c: prepare_sessions(next(x for x in candidates if x["sessions"]["availability"] == c), root)
                 for c in CONDITIONS}
    results, missing = _load_results(config, root, manifests)
    directory = root / config["output_dir"] / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    plt = _pyplot()
    write_schedule(manifests["trace"], directory)
    _availability_plot(plt, manifests["trace"], directory)
    deadlines, elapsed = [], 0
    for session in manifests["trace"]["sessions"]:
        elapsed += (session["deadline_us"] - session["start_us"]) / 3_600_000_000
        deadlines.append(elapsed)
    scores = [100 * p["evaluation"]["mmlu_accuracy"] for r in results.values() for p in r["points"]]
    low, high = (max(0, min(scores) - 5), min(100, max(scores) + 5)) if scores else (0, 100)
    rows, deadline_rows = [], []
    for condition in CONDITIONS:
        fig, ax = plt.subplots(figsize=(11, 5.6))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.17)
        for mode in MODES:
            result = results.get((condition, mode))
            if result is None:
                continue
            points = result["points"]
            x = [p["elapsed_session_seconds"] / 3600 for p in points]
            y = [100 * p["evaluation"]["mmlu_accuracy"] for p in points]
            scope = result["report"]["scope"]
            ax.plot(x, y, marker="s" if mode == "sync" else "o", markersize=4,
                    linestyle="--" if mode == "sync" else "-",
                    markerfacecolor="white" if mode == "sync" else COLORS[mode],
                    zorder=3 if mode == "sync" else 2,
                    label=f"{mode.capitalize()} ({scope.replace('_', ' ')})",
                    color=COLORS[mode], linewidth=1.7)
            for point in points:
                rows.append({"availability": condition, "mode": mode, "kind": point["kind"],
                             "session": point["session"] + 1,
                             "elapsed_session_hours": point["elapsed_session_seconds"] / 3600,
                             "recorded_elapsed_seconds": point["time_us"] / 1_000_000,
                             "version": point["version"],
                             "mmlu_accuracy": point["evaluation"]["mmlu_accuracy"],
                             "questions": point["evaluation"]["count"],
                             "scope": point["evaluation"]["scope"]})
                if point["kind"] == "deadline":
                    ax.scatter([point["elapsed_session_seconds"] / 3600],
                               [100 * point["evaluation"]["mmlu_accuracy"]],
                               marker="D", color=COLORS[mode], s=45, zorder=4)
            for session in result["report"]["sessions"]:
                row = {"availability": condition, "mode": mode, "session": session["index"] + 1,
                       "deadline_hours": deadlines[session["index"]],
                       "mmlu_accuracy": session["evaluation"]["mmlu_accuracy"],
                       "change_from_initial_pp": 100 * (session["evaluation"]["mmlu_accuracy"]
                                                       - points[0]["evaluation"]["mmlu_accuracy"]),
                       **{k: session[k] for k in ("server_updates", "trained_jobs", "completed_uploads",
                                                 "canceled_jobs", "incomplete_round_uploads")}}
                control = results.get(("always", mode))
                row["gap_to_always_pp"] = (100 * (control["report"]["sessions"][session["index"]]
                                                  ["evaluation"]["mmlu_accuracy"] - row["mmlu_accuracy"])
                                                 if control else "")
                deadline_rows.append(row)
        for i, hour in enumerate(deadlines):
            ax.axvline(hour, color="#9a3340", linestyle="--", linewidth=1, alpha=0.65)
            ax.text(hour, 0.97, f"Deadline {i + 1}", transform=ax.get_xaxis_transform(),
                    ha="right", va="top", color="#9a3340", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
        absent = [m for m in MODES if (condition, m) not in results]
        if len(absent) == len(MODES):
            ax.text(0.5, 0.5, "No measured MMLU results yet\nCUDA training and evaluation required",
                    transform=ax.transAxes, ha="center", va="center", fontsize=15,
                    color="#566270", bbox={"facecolor": "white", "edgecolor": "#cbd3dc", "pad": 16})
        else:
            ax.legend(loc="best")
            if absent:
                ax.text(0.02, 0.03, "Missing measured run: " + ", ".join(absent), transform=ax.transAxes)
        title = "All devices always available" if condition == "always" else "Availability from usage traces"
        train = config["training"]
        batch_size = train["micro_batch_size"] * train["grad_accum"]
        ax.set(title=f"MMLU accuracy: {title}\n{train['local_epochs']} local epoch(s); batch size {batch_size}; repeated training",
               xlabel="Simulated elapsed session time (hours; daytime gaps excluded)",
               ylabel="MMLU accuracy (%)", xlim=(0, deadlines[-1] * 1.02), ylim=(low, high))
        ax.grid(alpha=0.18)
        footer = ("Points are measured evaluations. Lines connect points. GPU evaluation time is excluded."
                  if any((condition, m) in results for m in MODES) else
                  "No accuracy data. This panel will fill after GPU training and evaluation.")
        fig.text(0.5, 0.025, footer,
                 ha="center", fontsize=9, color="#566270")
        _save_figure(fig, directory, f"accuracy-{condition}")
        plt.close(fig)
    _write_csv(directory / "accuracy.csv", rows,
               ["availability", "mode", "kind", "session", "elapsed_session_hours", "recorded_elapsed_seconds",
                "version", "mmlu_accuracy", "questions", "scope"])
    _write_csv(directory / "deadline-comparison.csv", deadline_rows,
               ["availability", "mode", "session", "deadline_hours", "mmlu_accuracy", "change_from_initial_pp",
                "server_updates", "trained_jobs", "completed_uploads", "canceled_jobs", "incomplete_round_uploads",
                "gap_to_always_pp"])
    report = {"complete": not missing, "missing_measured_runs": missing,
              "note": "Empty accuracy panels are placeholders, not experiment results" if missing else "Measured MMLU results",
              "time_axis": "Cumulative simulated session hours; daytime gaps and evaluation runtime excluded",
              "directory": str(directory),
              "plots": ["device-availability", "accuracy-always", "accuracy-trace"]}
    (directory / "plot_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
