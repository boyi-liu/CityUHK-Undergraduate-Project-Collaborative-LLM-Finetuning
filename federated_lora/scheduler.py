"""Trace event engine shared by synchronous and asynchronous LoRA runs.

The engine is independent of torch. Training, aggregation and deadline evaluation
are callbacks; the complete state can be checkpointed after each atomic event.
Original policies: 60s offline windows, known-window job fit, sync all-client
sampling, async one upload per window, interruption discard, upload-only transfer.
"""

from __future__ import annotations

import copy
import heapq
import random

from .timing import job_durations


PRIORITY = {"session_start": -1, "training_done": 0, "upload_done": 1,
            "window_close": 2, "window_open": 3, "session_end": 4}


class SessionScheduler:
    def __init__(self, config, manifest, adapter_bytes, initial_adapter, train, aggregate,
                 evaluate=None, save=None, state=None):
        self.config, self.manifest, self.adapter_bytes = config, manifest, adapter_bytes
        self.train, self.aggregate, self.evaluate, self.save = train, aggregate, evaluate, save
        self.mode = config["federation"]["mode"]
        self.rng = random.Random(config["seed"])
        if state is None:
            self.state = {
                "format_version": 1, "adapter": copy.deepcopy(initial_adapter), "version": 0,
                "session_index": 0, "next_job_id": 0, "next_event_id": 0,
                "attempts": [0] * config["federation"]["client_count"],
                "used_windows": set(), "history": [], "summaries": [],
                "rng_state": self.rng.getstate(), "finished": False,
            }
            self._begin_session(0)
        else:
            if state.get("format_version") != 1:
                raise ValueError("Unsupported session checkpoint format")
            self.state = state
            self.rng.setstate(state["rng_state"])

    def _push(self, when, kind, payload=None):
        state = self.state
        heapq.heappush(state["queue"], (when, PRIORITY[kind], state["next_event_id"], kind, payload))
        state["next_event_id"] += 1

    def _record(self, kind, **values):
        self.state["history"].append({"session": self.state["session_index"],
                                      "time_us": self.state["now_us"], "event": kind, **values})

    def _begin_session(self, index):
        state = self.state
        session = self.manifest["sessions"][index]
        state.update(session_index=index, now_us=session["start_us"], queue=[], jobs={},
                     clients=[{"window": None, "busy": None} for _ in self.manifest["clients"]],
                     round_selected=[], round_updates={}, round_index=0, skipped_windows=set(),
                     session_start_version=state["version"],
                     counters={"trained_jobs": 0, "completed_uploads": 0, "canceled_jobs": 0,
                               "skipped_jobs": 0, "uploaded_bytes": 0,
                               "real_train_seconds": 0.0, "participants": set(), "staleness": []})
        self._push(session["start_us"], "session_start")
        for client in self.manifest["clients"]:
            for window in client["windows"]:
                if window["end_us"] <= session["start_us"] or window["start_us"] >= session["deadline_us"]:
                    continue
                payload = {"client_id": client["client_id"], "window": window}
                self._push(max(window["start_us"], session["start_us"]), "window_open", payload)
                if window["end_us"] <= session["deadline_us"]:
                    self._push(window["end_us"], "window_close", payload)
        # Always present, even if no client has an eligible interval.
        self._push(session["deadline_us"], "session_end")

    def _new_round(self):
        state, fed = self.state, self.config["federation"]
        state["round_index"] += 1
        state["round_updates"] = {}
        # Preserve original selection from ALL clients, including unavailable ones.
        state["round_selected"] = self.rng.sample(range(fed["client_count"]), fed["clients_per_round"])
        self._record("round_start", round=state["round_index"], clients=state["round_selected"][:],
                     base_version=state["version"])

    def _fill(self):
        state = self.state
        if state["now_us"] >= self.manifest["sessions"][state["session_index"]]["deadline_us"]:
            return
        for client_id, client in enumerate(state["clients"]):
            window = client["window"]
            if window is None or client["busy"] is not None or state["now_us"] >= window["end_us"]:
                continue
            key = (client_id, window["id"])
            if key in state["skipped_windows"]:
                continue
            if self.mode == "sync":
                if client_id not in state["round_selected"] or client_id in state["round_updates"]:
                    continue
            else:
                if key in state["used_windows"] or len(state["jobs"]) >= self.config["federation"]["max_active"]:
                    continue
            self._launch(client_id, window)

    def _launch(self, client_id, window):
        state = self.state
        attempt = state["attempts"][client_id]
        state["attempts"][client_id] += 1
        costs = job_durations(self.config, client_id, attempt, self.adapter_bytes)
        finish = state["now_us"] + costs["train_us"]
        job_id = state["next_job_id"]
        state["next_job_id"] += 1
        # Inherited oracle fit check; do not interpret this as an availability predictor.
        if finish > window["end_us"]:
            state["skipped_windows"].add((client_id, window["id"]))
            state["counters"]["skipped_jobs"] += 1
            # Original async code retains its assigned slot until this window closes.
            state["jobs"][job_id] = {"job_id": job_id, "client_id": client_id, "phase": "skipped"}
            state["clients"][client_id]["busy"] = job_id
            self._record("job_skipped", client=client_id, window=window["id"],
                         reason="training_exceeds_known_window", **costs)
            return
        result = self.train(client_id, state["adapter"], job_id)
        job = {"job_id": job_id, "client_id": client_id, "window_id": window["id"],
               "base_version": state["version"], "phase": "training", **costs, **result}
        state["jobs"][job_id] = job
        state["clients"][client_id]["busy"] = job_id
        state["counters"]["trained_jobs"] += 1
        state["counters"]["real_train_seconds"] += result.get("real_train_seconds", 0.0)
        self._record("job_started", job_id=job_id, client=client_id, window=window["id"],
                     base_version=job["base_version"], training_finish_us=finish,
                     train_loss=result.get("train_loss"), **costs)
        self._push(finish, "training_done", job_id)

    def _cancel(self, job_id, reason):
        job = self.state["jobs"].pop(job_id, None)
        if job is None:
            return
        self.state["clients"][job["client_id"]]["busy"] = None
        if job["phase"] == "skipped":
            self._record("skipped_slot_released", job_id=job_id, client=job["client_id"], reason=reason)
            return
        self.state["counters"]["canceled_jobs"] += 1
        self._record("job_canceled", job_id=job_id, client=job["client_id"],
                     phase=job["phase"], reason=reason)

    def _upload(self, job_id):
        state = self.state
        job = state["jobs"].pop(job_id, None)
        if job is None:  # Canceled completion events cannot update the global model.
            return
        state["clients"][job["client_id"]]["busy"] = None
        counters = state["counters"]
        counters["completed_uploads"] += 1
        counters["uploaded_bytes"] += job["adapter_bytes"]
        counters["participants"].add(job["client_id"])
        self._record("upload_done", job_id=job_id, client=job["client_id"], base_version=job["base_version"])
        if self.mode == "async":
            staleness = state["version"] - job["base_version"]
            state["adapter"] = self.aggregate(state["adapter"], [job], "async", staleness)
            state["version"] += 1
            state["used_windows"].add((job["client_id"], job["window_id"]))
            counters["staleness"].append(staleness)
            self._record("aggregation", version=state["version"], jobs=[job_id], staleness=staleness)
        else:
            state["round_updates"][job["client_id"]] = job
            if set(state["round_updates"]) == set(state["round_selected"]):
                updates = [state["round_updates"][i] for i in state["round_selected"]]
                state["adapter"] = self.aggregate(state["adapter"], updates, "sync", 0)
                state["version"] += 1
                self._record("aggregation", version=state["version"],
                             jobs=[u["job_id"] for u in updates], round=state["round_index"])
                state["round_updates"] = {}
                if state["now_us"] < self.manifest["sessions"][state["session_index"]]["deadline_us"]:
                    self._new_round()
        self._fill()

    def _end_session(self):
        state = self.state
        for job_id in list(state["jobs"]):
            self._cancel(job_id, "session_deadline")
        buffered = len(state["round_updates"])
        if buffered:
            self._record("incomplete_round", buffered_uploads=buffered, round=state["round_index"])
        state["round_updates"] = {}
        counters = state["counters"]
        summary = {**self.manifest["sessions"][state["session_index"]], "mode": self.mode,
                   "start_version": state["session_start_version"], "version": state["version"],
                   "server_updates": state["version"] - state["session_start_version"],
                   "incomplete_round_uploads": buffered,
                   **{k: v for k, v in counters.items() if k != "participants"},
                   "participants": sorted(counters["participants"])}
        self._record("session_deadline", version=state["version"])
        if self.evaluate is not None:
            summary["evaluation"] = self.evaluate(state["adapter"], state["session_index"])
        state["summaries"].append(summary)
        if state["session_index"] + 1 == len(self.manifest["sessions"]):
            state["finished"] = True
            state["queue"] = []
        else:
            self._begin_session(state["session_index"] + 1)

    def run(self, max_events=None):
        processed = 0
        while not self.state["finished"] and (max_events is None or processed < max_events):
            when, _, _, kind, payload = heapq.heappop(self.state["queue"])
            self.state["now_us"] = when
            if kind == "session_start":
                self._record("session_start", version=self.state["version"])
                if self.mode == "sync":
                    self._new_round()
            elif kind == "window_open":
                self.state["clients"][payload["client_id"]]["window"] = payload["window"]
                self._record("window_open", client=payload["client_id"], window=payload["window"]["id"])
                self._fill()
            elif kind == "window_close":
                client = self.state["clients"][payload["client_id"]]
                if client["window"] and client["window"]["id"] == payload["window"]["id"]:
                    if client["busy"] is not None:
                        self._cancel(client["busy"], "availability_ended")
                    client["window"] = None
                    self._record("window_close", client=payload["client_id"])
                    self._fill()
            elif kind == "training_done":
                job = self.state["jobs"].get(payload)
                if job is not None:
                    job["phase"] = "uploading"
                    self._record("training_done", job_id=payload, client=job["client_id"])
                    self._push(when + job["upload_us"], "upload_done", payload)
            elif kind == "upload_done":
                self._upload(payload)
            else:
                self._end_session()
            self.state["rng_state"] = self.rng.getstate()
            if self.save is not None:
                self.save(self.state)
            processed += 1
        return self.state
