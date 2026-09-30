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
            "training": {"local_steps": 2},
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
        first = job_durations(config(), 0, 0, 125000)
        self.assertEqual(first, job_durations(config(), 0, 0, 125000))
        self.assertEqual(first["upload_us"], 1000000)
        self.assertGreater(first["train_us"], 0)


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


if __name__ == "__main__":
    unittest.main()
