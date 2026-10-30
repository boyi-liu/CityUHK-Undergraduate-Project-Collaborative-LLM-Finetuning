import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from federated_lora.scheduler import SessionScheduler
from federated_lora.timing import job_durations
from federated_lora.traces import parse_device
from federated_lora.sessions import prepare_sessions


def config(mode="async"):
    return {"seed": 42, "model": {"name": "Qwen/Qwen2.5-0.5B"},
            # Historical rules remain testable; comparison configs use repeated.
            "sessions": {"participation": "once_per_window"},
            "training": {"local_epochs": 1, "micro_batch_size": 1, "grad_accum": 32},
            "timing": {"agx_client_ids": [0], "upload_mbps": 1.0, "speed_scale": 1.0},
            "federation": {"mode": mode, "client_count": 2, "clients_per_round": 2, "max_active": 2}}


def manifest(windows, sessions=((0, 100),)):
    return {"clients": [{"client_id": i, "windows": [
        {"id": n, "start_us": start, "end_us": end} for n, (start, end) in enumerate(group)]}
        for i, group in enumerate(windows)],
        "sessions": [{"index": i, "start_us": start, "deadline_us": end}
                     for i, (start, end) in enumerate(sessions)]}


def train(client, adapter, job):
    return {"adapter": adapter + 1, "example_count": 1, "train_loss": 1.0}


def aggregate(adapter, updates, mode, stale):
    if mode == "sync":
        return sum(u["adapter"] for u in updates) / len(updates)
    return adapter * 0.5 + updates[0]["adapter"] * 0.5


def costs(train_us=10, upload_us=5):
    return {"train_us": train_us, "upload_us": upload_us, "adapter_bytes": 100,
            "profile": "test", "step_seconds": 0.000005}


class AvailabilityTests(unittest.TestCase):
    def test_or_rule_battery_ignored_and_short_intervals_filtered(self):
        lines = [
            "2020-01-01 00:00:00\tscreen_off", "2020-01-01 00:00:00\tscreen_unlock",
            "2020-01-01 00:00:00\tbattery_charged_on", "2020-01-01 00:00:00\twifi",
            "2020-01-01 00:00:30\t1.0%",  # no battery threshold
            "2020-01-01 00:01:00\tscreen_on", "2020-01-01 00:01:00\tscreen_lock",
            "2020-01-01 00:02:00\tscreen_unlock",  # on and unlocked ends eligibility
            "2020-01-01 00:03:00\tscreen_off", "2020-01-01 00:03:30\tbattery_charged_off",
        ]
        result = parse_device({"messages": "\n".join(lines)})
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["end"], "2020-01-01T00:02:00")
        self.assertEqual(result["stats"]["short_windows"], 1)

    def test_unknown_state_is_not_available_and_never_extends_coverage(self):
        result = parse_device({"messages": "2020-01-01 00:00:00\tscreen_off\n2020-01-01 01:00:00\twifi"})
        self.assertEqual(result["windows"], [])
        result = parse_device({"messages": "\n".join([
            "2020-01-01 00:00:00\tscreen_lock", "2020-01-01 00:00:00\twifi",
            "2020-01-01 00:00:00\tbattery_charged_on", "2020-01-01 00:02:00\t100.0%",
        ])})
        self.assertEqual(result["windows"][0]["end"], result["coverage_end"])

    def test_sorting_duplicates_and_malformed_lines_are_reported(self):
        result = parse_device({"messages": "\n".join([
            "2020-01-01 00:02:00\tbattery_charged_off",
            "2020-01-01 00:00:00\tscreen_lock", "2020-01-01 00:00:00\twifi",
            "2020-01-01 00:00:00\twifi", "broken",
            "2020-01-01 00:00:00\tbattery_charged_on",
        ])})
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["stats"]["duplicates"], 1)
        self.assertEqual(result["stats"]["malformed"], 1)
        self.assertGreater(result["stats"]["out_of_order"], 0)

    def test_timing_is_reproducible_and_upload_units_are_correct(self):
        first = job_durations(config(), 0, 0, 125000, 64)
        self.assertEqual(first, job_durations(config(), 0, 0, 125000, 64))
        self.assertEqual(first["upload_us"], 1000000)
        self.assertGreater(first["train_us"], 0)
        larger = job_durations(config(), 0, 0, 125000, 129)
        self.assertEqual(first["optimizer_steps"], 2)
        self.assertEqual(larger["optimizer_steps"], 5)
        self.assertAlmostEqual(larger["train_us"] / first["train_us"], 2.5, places=6)

    def test_scheduler_uses_each_clients_epoch_size(self):
        settings = config()
        engine = SessionScheduler(settings, manifest([[(0, 100_000_000)], [(0, 100_000_000)]]),
                                  100, 0.0, train, aggregate, client_sizes=[32, 65])
        state = engine.run()
        jobs = {row["client"]: row for row in state["history"] if row["event"] == "job_started"}
        self.assertEqual(jobs[0]["optimizer_steps"], 1)
        self.assertEqual(jobs[1]["optimizer_steps"], 3)
        self.assertEqual(jobs[1]["training_examples"], 65)


class SchedulerTests(unittest.TestCase):
    def engine(self, windows, mode="async", sessions=((0, 100),), **kwargs):
        return SessionScheduler(config(mode), manifest(windows, sessions), 100, 0.0,
                                train, aggregate, **kwargs)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_async_one_upload_each_window_and_stale_arrival(self, _):
        state = self.engine([[(0, 100)], [(0, 100)]]).run()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["summaries"][0]["staleness"], [0, 1])
        self.assertEqual(state["next_job_id"], 2)

    @patch("federated_lora.scheduler.job_durations", return_value=costs(10, 20))
    def test_upload_interrupted_and_canceled_event_never_aggregates(self, _):
        state = self.engine([[(0, 20)], []]).run()
        self.assertEqual(state["version"], 0)
        self.assertEqual(state["summaries"][0]["canceled_jobs"], 1)
        self.assertTrue(state["finished"])

    @patch("federated_lora.scheduler.job_durations", return_value=costs(10, 20))
    def test_cancellation_frees_capacity_for_waiting_client(self, _):
        engine = self.engine([[(0, 20)], [(0, 100)]])
        engine.config["federation"]["max_active"] = 1
        state = engine.run()
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["summaries"][0]["participants"], [1])

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_known_window_fit_skips_training(self, _):
        state = self.engine([[(0, 5)], []]).run()
        self.assertEqual(state["summaries"][0]["trained_jobs"], 0)
        self.assertEqual(state["summaries"][0]["skipped_jobs"], 1)

    @patch("federated_lora.scheduler.job_durations", return_value=costs(10, 10))
    def test_upload_at_window_end_and_deadline_is_included(self, _):
        state = self.engine([[(0, 20)], [(0, 20)]], sessions=((0, 20),)).run()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["summaries"][0]["canceled_jobs"], 0)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_upload_after_deadline_is_discarded(self, _):
        state = self.engine([[(0, 100)], []], sessions=((0, 12),)).run()
        self.assertEqual(state["version"], 0)
        self.assertEqual(state["summaries"][0]["canceled_jobs"], 1)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_sync_waits_for_selected_unavailable_client(self, _):
        state = self.engine([[(0, 25)], [(50, 90)]], mode="sync").run()
        aggregations = [x for x in state["history"] if x["event"] == "aggregation"]
        self.assertEqual(aggregations[0]["time_us"], 65)
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["summaries"][0]["incomplete_round_uploads"], 1)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_sync_reuses_already_available_clients_next_round(self, _):
        state = self.engine([[(0, 100)], [(0, 100)]], mode="sync").run()
        self.assertGreater(state["version"], 1)
        starts = [x for x in state["history"] if x["event"] == "job_started"]
        self.assertEqual([x["base_version"] for x in starts[:4]], [0, 0, 1, 1])

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_quiet_sessions_evaluate_and_keep_global_weights(self, _):
        observed = []
        state = self.engine([[], []], sessions=((0, 100), (200, 300), (400, 500)),
                            evaluate=lambda adapter, i: observed.append((i, adapter))).run()
        self.assertEqual(observed, [(0, 0.0), (1, 0.0), (2, 0.0)])
        self.assertEqual(len(state["summaries"]), 3)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_global_continuity_and_resume_with_pending_jobs(self, _):
        windows = [[(0, 100), (200, 300)], [(0, 100), (200, 300)]]
        sessions = ((0, 100), (200, 300))
        full = self.engine(windows, sessions=sessions).run()
        partial = self.engine(windows, sessions=sessions).run(max_events=4)
        self.assertTrue(partial["jobs"])
        restored = self.engine(windows, sessions=sessions, state=copy.deepcopy(partial)).run()
        self.assertEqual(full, restored)
        self.assertEqual(full["summaries"][1]["start_version"], full["summaries"][0]["version"])
        self.assertGreater(full["adapter"], 1.0)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_sync_resume_preserves_buffered_upload(self, _):
        windows = [[(0, 25)], [(50, 90)]]
        full = self.engine(windows, mode="sync").run()
        partial = self.engine(windows, mode="sync").run(max_events=4)
        self.assertEqual(len(partial["round_updates"]), 1)
        restored = self.engine(windows, mode="sync", state=copy.deepcopy(partial)).run()
        self.assertEqual(full, restored)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_one_upload_limit_survives_same_window_across_sessions(self, _):
        state = self.engine([[(0, 300)], [(0, 300)]], sessions=((0, 100), (200, 300))).run()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["summaries"][1]["server_updates"], 0)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_repeated_jobs_start_from_latest_global_after_upload(self, _):
        settings = config()
        settings["sessions"]["participation"] = "repeated"
        state = SessionScheduler(settings, manifest([[(0, 60)], []], ((0, 60),)),
                                 100, 0.0, train, aggregate).run()
        starts = [e for e in state["history"] if e["event"] == "job_started"]
        self.assertEqual([e["time_us"] for e in starts], [0, 15, 30, 45])
        self.assertEqual([e["base_version"] for e in starts], [0, 1, 2, 3])
        self.assertEqual(state["version"], 4)  # Upload at the deadline counts.
        self.assertEqual(state["adapter"], 2.0)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_repeated_training_gives_waiting_clients_capacity(self, _):
        settings = config()
        settings["sessions"]["participation"] = "repeated"
        settings["federation"]["max_active"] = 1
        state = SessionScheduler(settings, manifest([[(0, 60)], [(0, 60)]], ((0, 60),)),
                                 100, 0.0, train, aggregate).run()
        starts = [e for e in state["history"] if e["event"] == "job_started"]
        self.assertEqual([e["client"] for e in starts], [0, 1, 0, 1])

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_periodic_evaluation_observes_arrivals_and_does_not_advance_clock(self, _):
        settings = config()
        settings["sessions"]["participation"] = "repeated"
        settings["evaluation"] = {"interval_seconds": 0.000015}
        observed = []

        def evaluate(adapter, index, point):
            observed.append((point["time_us"], adapter))
            return {"mmlu_accuracy": 0.5}

        engine = SessionScheduler(settings, manifest([[(0, 60)], []], ((0, 60),)),
                                  100, 0.0, train, aggregate, evaluate_periodic=evaluate,
                                  evaluate=lambda adapter, index: {"mmlu_accuracy": 0.5})
        full = engine.run()
        self.assertEqual(observed, [(15, 0.5), (30, 1.0), (45, 1.5)])
        self.assertEqual([p["time_us"] for p in full["evaluations"]], [15, 30, 45, 60])
        settings["evaluation"]["interval_seconds"] = None
        without = SessionScheduler(settings, manifest([[(0, 60)], []], ((0, 60),)),
                                   100, 0.0, train, aggregate).run()
        for field in ("adapter", "version", "attempts"):
            self.assertEqual(full[field], without[field])

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_periodic_resume_keeps_pending_jobs_and_evaluation_points(self, _):
        settings = config()
        settings["sessions"]["participation"] = "repeated"
        settings["evaluation"] = {"interval_seconds": 0.000020}
        schedule = manifest([[(0, 300)], [(0, 300)]], ((0, 60), (200, 260)))

        def engine(state=None):
            return SessionScheduler(settings, schedule, 100, 0.0, train, aggregate, state=state,
                                    evaluate=lambda a, i: {"mmlu_accuracy": a},
                                    evaluate_periodic=lambda a, i, p: {"mmlu_accuracy": a})

        full = engine().run()
        partial = engine().run(max_events=8)
        self.assertTrue(partial["jobs"])
        resumed = engine(copy.deepcopy(partial)).run()
        self.assertEqual(full, resumed)
        self.assertEqual(len(full["evaluations"]), 6)
        self.assertEqual(full["evaluations"][-1]["elapsed_session_seconds"], 0.000120)
        self.assertGreater(full["summaries"][1]["server_updates"], 0)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_quiet_periodic_points_and_nondivisible_deadline(self, _):
        settings = config()
        settings["evaluation"] = {"interval_seconds": 0.000030}
        state = SessionScheduler(settings, manifest([[], []]), 100, 0.0, train, aggregate).run()
        self.assertEqual([p["time_us"] for p in state["evaluations"]], [30, 60, 90, 100])
        self.assertEqual([p["version"] for p in state["evaluations"]], [0] * 4)
        self.assertEqual(state["evaluations"][-1]["kind"], "deadline")

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_repeated_training_stops_when_availability_ends_during_upload(self, _):
        settings = config()
        settings["sessions"]["participation"] = "repeated"
        state = SessionScheduler(settings, manifest([[(0, 29)], []]), 100, 0.0, train, aggregate).run()
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["summaries"][0]["trained_jobs"], 2)
        self.assertEqual(state["summaries"][0]["canceled_jobs"], 1)
        self.assertEqual(state["adapter"], 0.5)

    @patch("federated_lora.scheduler.job_durations", return_value=costs())
    def test_periodic_sync_evaluation_uses_global_not_buffered_client_adapter(self, _):
        settings = config("sync")
        settings["sessions"]["participation"] = "repeated"
        settings["evaluation"] = {"interval_seconds": 0.000030}
        observed = []
        SessionScheduler(settings, manifest([[(0, 25)], [(50, 90)]]), 100, 0.0, train, aggregate,
                         evaluate_periodic=lambda a, i, p: observed.append((p["time_us"], a))).run()
        self.assertEqual(observed, [(30, 0.0), (60, 0.0), (90, 1.0)])


class SessionPreparationTests(unittest.TestCase):
    def settings(self):
        return {"federation": {"client_count": 1}, "sessions": {
            "trace_path": "unused.json", "trace_ids": [0], "start_date": "2020-01-01",
            "start_time": "22:00", "end_time": "07:00", "count": 2,
            "time_basis": "recorded_local"}}

    def traces(self):
        return {"devices": {"0": {"coverage_start": "2020-01-01T00:00:00",
            "coverage_end": "2020-01-04T00:00:00", "windows": [
                {"id": 0, "start": "2020-01-01T21:59:00", "end": "2020-01-01T22:00:30"}]}}}

    def test_cross_midnight_and_clip_after_minimum_window_filter(self):
        with patch("federated_lora.sessions.load_traces", return_value=self.traces()):
            result = prepare_sessions(self.settings(), Path("."))
        self.assertEqual(result["sessions"][0]["deadline"], "2020-01-02T07:00:00")
        self.assertEqual(result["sessions"][1]["start"], "2020-01-02T22:00:00")
        self.assertEqual(result["sessions"][0]["client_available_seconds"], [30.0])

    def test_missing_coverage_is_rejected(self):
        traces = self.traces()
        traces["devices"]["0"]["coverage_end"] = "2020-01-02T00:00:00"
        with patch("federated_lora.sessions.load_traces", return_value=traces):
            with self.assertRaisesRegex(ValueError, "does not cover"):
                prepare_sessions(self.settings(), Path("."))

    def test_always_available_has_full_session_windows_with_same_trace_mapping(self):
        settings = self.settings()
        with patch("federated_lora.sessions.load_traces", return_value=self.traces()):
            trace = prepare_sessions(settings, Path("."))
            settings["sessions"]["availability"] = "always"
            always = prepare_sessions(settings, Path("."))
        self.assertEqual(always["clients"][0]["trace_id"], trace["clients"][0]["trace_id"])
        self.assertEqual([s["client_available_seconds"] for s in always["sessions"]], [[32400], [32400]])
        self.assertEqual(always["sessions"][0]["start"], trace["sessions"][0]["start"])
        self.assertEqual(always["clients"][0]["windows"][-1]["end"], always["sessions"][-1]["deadline"])


if __name__ == "__main__":
    unittest.main()
