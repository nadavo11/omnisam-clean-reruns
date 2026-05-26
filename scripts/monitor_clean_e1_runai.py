#!/usr/bin/env python3
"""Background monitor for the clean E1 RunAI matrix.

Polls frequently right after submission, then backs off to hourly checks.
Writes a machine-readable state file plus a human-readable log, and stores
tail logs for failed jobs so the failure mode is preserved automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_JOBS = [
    "monuseg-e1-a0-s0-clean",
    "monuseg-e1-a0-s1-clean",
    "monuseg-e1-a0-s2-clean",
    "monuseg-e1-a2-s0-clean",
    "monuseg-e1-a2-s1-clean",
    "monuseg-e1-a2-s2-clean",
    "monuseg-e1-a3-s0-clean",
    "monuseg-e1-a3-s1-clean",
    "monuseg-e1-a3-s2-clean",
]

TERMINAL = {"Succeeded", "Failed", "Deleted", "Completed", "Error"}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def describe_status(job: str, project: str) -> tuple[str, str]:
    proc = run(["runai", "describe", "job", job, "-p", project])
    if proc.returncode != 0:
        return "MISSING", proc.stderr.strip() or proc.stdout.strip()
    status = "UNKNOWN"
    for line in proc.stdout.splitlines():
        if line.startswith("Status: "):
            status = line.split(": ", 1)[1].strip()
            break
    return status, proc.stdout


def tail_logs(job: str, project: str, lines: int) -> str:
    proc = run(["runai", "logs", job, "-p", project, "--tail", str(lines)])
    if proc.returncode != 0:
        return proc.stderr.strip() or proc.stdout.strip()
    return proc.stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="avidan")
    parser.add_argument("--state-path", default="reports/runai_clean_monitor_state.json")
    parser.add_argument("--log-path", default="reports/runai_clean_monitor.log")
    parser.add_argument("--failure-dir", default="reports/runai_failures")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--warmup-minutes", type=int, default=15)
    parser.add_argument("--hourly-seconds", type=int, default=3600)
    parser.add_argument("--tail-lines", type=int, default=200)
    parser.add_argument("jobs", nargs="*", default=DEFAULT_JOBS)
    args = parser.parse_args()

    state_path = Path(args.state_path)
    log_path = Path(args.log_path)
    failure_dir = Path(args.failure_dir)
    tracked = {
        job: {
            "status": None,
            "history": [],
            "failure_log_path": None,
        }
        for job in args.jobs
    }
    start = time.time()

    while True:
        now = datetime.now(timezone.utc).isoformat()
        all_terminal = True
        log_lines: list[str] = []
        for job in args.jobs:
            status, raw = describe_status(job, args.project)
            rec = tracked[job]
            if status not in TERMINAL:
                all_terminal = False
            if rec["status"] != status:
                rec["history"].append({"ts": now, "status": status})
                rec["status"] = status
                log_lines.append(f"[{now}] {job} -> {status}")
                if status in {"Failed", "Error"}:
                    failure_log = tail_logs(job, args.project, args.tail_lines)
                    failure_path = failure_dir / f"{job}.log"
                    write_text(failure_path, failure_log)
                    rec["failure_log_path"] = str(failure_path)
                elif status == "Succeeded":
                    success_tail = tail_logs(job, args.project, min(args.tail_lines, 80))
                    success_path = failure_dir / f"{job}.success.log"
                    write_text(success_path, success_tail)
            rec["last_checked"] = now
            rec["last_raw_excerpt"] = raw[:2000]

        state = {
            "updated_at": now,
            "jobs": tracked,
        }
        write_text(state_path, json.dumps(state, indent=2, sort_keys=True))
        if log_lines:
            with log_path.open("a", encoding="utf-8") as fh:
                for line in log_lines:
                    fh.write(line + "\n")

        if all_terminal:
            return 0

        elapsed = time.time() - start
        sleep_s = args.poll_seconds if elapsed < args.warmup_minutes * 60 else args.hourly_seconds
        time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
