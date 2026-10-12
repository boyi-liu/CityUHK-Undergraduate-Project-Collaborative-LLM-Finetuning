"""Resumable Alpaca task grouping with concurrent Codex CLI calls.

Each completed 25-row batch is saved before another batch starts. This keeps
the result recoverable if the machine, network, or Codex usage window stops.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import ctypes
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import time
from collections import deque
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "alpaca_raw" / "alpaca_data.json"
OUTPUT = ROOT / "dataset" / "alpaca_classification_checkpoints"
FINAL = ROOT / "dataset" / "alpaca_task_groups.jsonl"
MANIFEST = ROOT / "dataset" / "alpaca_task_groups.manifest.json"
OVERRIDES = ROOT / "dataset" / "alpaca_task_group_overrides.jsonl"
MODEL = "gpt-5.6-sol"
EFFORT = "high"
BATCH_SIZE = 25
MIN_FREE_RAM_BYTES = 4 * 1024**3
LABELS = (
    "create_content", "transform_content", "explain_describe", "find_fact",
    "suggest_design", "classify", "summarize", "calculate",
)
PROMPT_HEADER = """Classify each Alpaca instruction by the primary task the USER REQUESTS.
Use the instruction wording and its meaning, not just the first verb. Return
exactly one group per source_index, in input order. Do not use tools.

Groups:
- create_content: produce new prose, code, lists or other content from scratch.
- transform_content: edit, rewrite, translate or change the form of supplied content.
- explain_describe: explain a concept, process, difference or properties.
- find_fact: retrieve or name a specific factual answer.
- suggest_design: propose a solution, recommendation, idea, plan, design or procedure.
- classify: assign a category, label, or true/false judgment.
- summarize: condense longer source material into a shorter account.
- calculate: perform a numerical computation or estimate a numerical value.

Resolve boundaries by the requested OUTPUT: code that calculates something is
create_content; an arithmetic answer is calculate; a summary is summarize even
if the instruction starts with 'write'. Give a concise reason (at most 12 words).
Choose the closest of the eight groups for every row. Map action execution,
scheduling and model-training requests to procedural assistance in suggest_design.
Map qualitative predictions to explain_describe and numerical forecasts to calculate.
Use input context to resolve vague instructions. Do not use reference answers.

Instructions:
"""


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
        ("extended", ctypes.c_ulonglong),
    ]


def available_ram_bytes() -> int:
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Could not read available RAM")
    return status.available


def expected_ids(batch_id: int, row_count: int) -> list[int]:
    return list(range(batch_id * BATCH_SIZE, min((batch_id + 1) * BATCH_SIZE, row_count)))


@lru_cache(maxsize=1)
def reviewed_overrides() -> dict:
    """Keep the reviewed assignments when rebuilding from original batch caches."""
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    reviews = metadata.get("reviews", [])
    if not OVERRIDES.exists():
        if reviews:
            raise ValueError("Reviewed task-group overrides are missing")
        return {}
    raw = OVERRIDES.read_bytes()
    if (not reviews or reviews[-1]["overrides_sha256"] != hashlib.sha256(raw).hexdigest()
            or metadata["source_sha256"] != hashlib.sha256(RAW.read_bytes()).hexdigest()):
        raise ValueError("Reviewed overrides differ from their manifest or Alpaca source")
    result = {}
    for line in raw.splitlines():
        row = json.loads(line)
        index = row["source_index"]
        if type(index) is not int or index < 0 or index in result or row["task_group"] not in LABELS:
            raise ValueError("Duplicate index or invalid reviewed task group")
        result[index] = row
    if len(result) != reviews[-1]["reviewed_rows"]:
        raise ValueError("Reviewed overrides are incomplete")
    return result


def batch_results(path: Path) -> list[dict]:
    overrides = reviewed_overrides()
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    for row in results:
        if row["source_index"] in overrides:
            reviewed = overrides[row["source_index"]]
            row.update(group=reviewed["task_group"], reason=reviewed["group_reason"])
    return results


def valid_result(path: Path, ids: list[int]) -> bool:
    try:
        results = batch_results(path)
        return (
            isinstance(results, list)
            and len(results) == len(ids)
            and sorted(r["source_index"] for r in results) == ids
            and all(r["group"] in LABELS and isinstance(r["reason"], str) for r in results)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def classify_batch(batch_id: int, alpaca: list[dict], schema: Path) -> dict:
    ids = expected_ids(batch_id, len(alpaca))
    finished = OUTPUT / "batches" / f"{batch_id:05d}.json"
    if valid_result(finished, ids):
        return {"batch": batch_id, "rows": len(ids), "seconds": 0.0, "cached": True}

    rows = [{"source_index": i, "instruction": alpaca[i]["instruction"],
             "input": alpaca[i]["input"]} for i in ids]
    prompt = PROMPT_HEADER + json.dumps(rows, ensure_ascii=False)
    pending = OUTPUT / "batches" / f"{batch_id:05d}.pending.json"
    log = OUTPUT / "logs" / f"{batch_id:05d}.txt"
    command = [
        "codex", "exec", "-m", MODEL, "--ephemeral",
        "-c", f"model_reasoning_effort={EFFORT}", "-s", "read-only",
        "-C", str(ROOT), "--output-schema", str(schema),
        "--output-last-message", str(pending), "-",
    ]
    started = time.perf_counter()
    failure = "unknown error"
    for attempt in range(1, 4):
        try:
            completed = subprocess.run(
                command, input=prompt, text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=180,
            )
            log.write_text(completed.stderr[-8000:], encoding="utf-8")
            if completed.returncode == 0 and valid_result(pending, ids):
                pending.replace(finished)
                return {
                    "batch": batch_id, "rows": len(ids),
                    "seconds": round(time.perf_counter() - started, 2),
                    "cached": False, "attempts": attempt,
                }
            failure = f"exit={completed.returncode}; invalid or missing output; {completed.stderr[-400:]}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            failure = str(exc)
        if attempt < 3:
            time.sleep(5 * attempt)
    return {"batch": batch_id, "rows": 0, "error": failure}


def write_final(alpaca: list[dict], source_hash: str) -> None:
    reviewed_overrides()  # Fail before replacing any file if review integrity changed.
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    all_results = {}
    for batch_id in range((len(alpaca) + BATCH_SIZE - 1) // BATCH_SIZE):
        path = OUTPUT / "batches" / f"{batch_id:05d}.json"
        ids = expected_ids(batch_id, len(alpaca))
        if not valid_result(path, ids):
            raise ValueError(f"Missing or invalid batch {batch_id}")
        for item in batch_results(path):
            idx = item["source_index"]
            if idx in all_results:
                raise ValueError(f"Duplicate source index {idx}")
            all_results[idx] = {
                "source_index": idx, "task_group": item["group"],
                "group_reason": item["reason"],
            }
    if sorted(all_results) != list(range(len(alpaca))):
        raise ValueError("Final result does not cover every Alpaca source index")
    temp = FINAL.with_suffix(".pending.jsonl")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for idx in range(len(alpaca)):
            stream.write(json.dumps(all_results[idx], ensure_ascii=False) + "\n")
    temp.replace(FINAL)
    counts = {label: 0 for label in LABELS}
    for item in all_results.values():
        counts[item["task_group"]] += 1
    MANIFEST.write_text(json.dumps({
        "source_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_hash, "source_rows": len(alpaca),
        "model": MODEL, "reasoning_effort": EFFORT,
        "batch_size": BATCH_SIZE, "labels": list(LABELS), "counts": counts,
        "classification_input": "instruction and input for new batches and reviewed rows; cached original batches used instruction only",
        "reviews": previous.get("reviews", []),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"FINAL: {len(all_results)} labels written to {FINAL}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-new-batches", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 400:
        raise ValueError("--workers must be between 1 and 400")
    raw_bytes = RAW.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    alpaca = json.loads(raw_bytes)
    overrides = reviewed_overrides()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "batches").mkdir(exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    identity = OUTPUT / "source.json"
    source_meta = {
        "source_sha256": source_hash, "source_rows": len(alpaca),
        "model": MODEL, "effort": EFFORT, "batch_size": BATCH_SIZE,
        "labels": list(LABELS),
    }
    if identity.exists():
        previous = json.loads(identity.read_text(encoding="utf-8"))
        if previous != source_meta:
            # Only the retired category may differ; source and model settings must match.
            migrated = {**previous, "labels": list(LABELS)}
            if (not overrides or previous.get("labels") != list(LABELS) + ["other_unclear"]
                    or migrated != source_meta):
                raise ValueError("Source or settings changed since this run started")
    identity.write_text(json.dumps(source_meta, indent=2), encoding="utf-8")
    schema = OUTPUT / "schema.json"
    schema.write_text(json.dumps({
        "type": "object", "properties": {"results": {
            "type": "array", "items": {"type": "object", "properties": {
                "source_index": {"type": "integer"},
                "group": {"type": "string", "enum": list(LABELS)},
                "reason": {"type": "string"},
            }, "required": ["source_index", "group", "reason"],
                "additionalProperties": False},
        }}, "required": ["results"], "additionalProperties": False,
    }, indent=2), encoding="utf-8")
    total_batches = (len(alpaca) + BATCH_SIZE - 1) // BATCH_SIZE
    missing = [
        i for i in range(total_batches)
        if not valid_result(OUTPUT / "batches" / f"{i:05d}.json", expected_ids(i, len(alpaca)))
    ]
    if args.max_new_batches is not None:
        missing = missing[:args.max_new_batches]
    print(f"SOURCE: {len(alpaca)} rows; {total_batches} batches; {len(missing)} to run; workers={args.workers}", flush=True)
    started = time.perf_counter()
    completed_count = 0
    failures = []
    peak_active = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending_batches = deque(missing)
        active = {}
        def fill_workers() -> None:
            nonlocal peak_active
            while pending_batches and len(active) < args.workers and len(failures) < 3:
                free = available_ram_bytes()
                if free < MIN_FREE_RAM_BYTES and active:
                    break
                batch_id = pending_batches.popleft()
                active[pool.submit(classify_batch, batch_id, alpaca, schema)] = batch_id
                peak_active = max(peak_active, len(active))
                time.sleep(0.08)

        fill_workers()
        print(f"ACTIVE: {len(active)} initial calls; free RAM {available_ram_bytes()/1024**3:.1f} GB", flush=True)
        while active or (pending_batches and len(failures) < 3):
            if not active:
                time.sleep(5)
                fill_workers()
                continue
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                batch_id = active.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = {"batch": batch_id, "rows": 0, "error": str(exc)}
                completed_count += 1
                if "error" in status:
                    failures.append(status)
                    print(f"FAILED batch {batch_id}: {status['error'][:250]}", flush=True)
                if completed_count % 20 == 0 or completed_count == len(missing):
                    print(f"PROGRESS {completed_count}/{len(missing)} new batches, {len(failures)} failed, {time.perf_counter()-started:.1f}s, active={len(active)}, peak={peak_active}", flush=True)
            fill_workers()
    if failures:
        (OUTPUT / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"STOPPED: {len(failures)} failed batches; rerun to retry them", flush=True)
        return
    remaining = sum(
        not valid_result(OUTPUT / "batches" / f"{i:05d}.json", expected_ids(i, len(alpaca)))
        for i in range(total_batches)
    )
    print(f"RUN COMPLETE: {completed_count} new batches in {time.perf_counter()-started:.1f}s; {remaining} remaining", flush=True)
    if remaining == 0:
        write_final(alpaca, source_hash)


if __name__ == "__main__":
    main()
