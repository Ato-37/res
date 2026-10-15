#!/usr/bin/env python3
"""Pair Windows/WSL memory counters around one synthetic Qwen run and idle tail.

This is a bounded diagnostic, not a faithfulness evaluation. It stores only
aggregate counters and UTC timestamps. The Windows PowerShell sampler must be
visible to this WSL process (per-command elevated execution may be required).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


GUEST_FIELDS = (
    "utc", "mem_total_kib", "mem_available_kib", "cgroup_current_bytes",
    "cgroup_file_bytes", "cgroup_anon_bytes", "events_high", "events_max",
    "events_oom", "events_oom_kill",
)
HOST_COMMAND = r"""
$ErrorActionPreference='Stop'
$os=Get-CimInstance Win32_OperatingSystem
$commit=Get-Counter '\Memory\Committed Bytes'
$proc=Get-CimInstance Win32_Process -Filter "Name='vmmemWSL'"
$ws=[int64]0; if ($proc) { $ws=[int64]$proc.WorkingSetSize }
[pscustomobject]@{
  utc=[DateTime]::UtcNow.ToString('o')
  host_total_bytes=([int64]$os.TotalVisibleMemorySize * 1024)
  host_free_bytes=([int64]$os.FreePhysicalMemory * 1024)
  host_commit_bytes=[int64]$commit.CounterSamples[0].CookedValue
  wsl_vm_working_set_bytes=$ws
} | ConvertTo-Json -Compress
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_guest_row() -> dict[str, int | str]:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].rstrip(":") in {"MemTotal", "MemAvailable"}:
            if fields[2] != "kB" or fields[0].rstrip(":") in meminfo:
                raise ValueError("invalid guest meminfo")
            meminfo[fields[0].rstrip(":")] = int(fields[1])
    if set(meminfo) != {"MemTotal", "MemAvailable"}:
        raise ValueError("missing guest meminfo")
    root = Path("/sys/fs/cgroup")
    stat = {}
    for line in (root / "memory.stat").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            stat[fields[0]] = int(fields[1])
    events = {}
    for line in (root / "memory.events").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            events[fields[0]] = int(fields[1])
    current = int((root / "memory.current").read_text().strip())
    required_events = {"high", "max", "oom", "oom_kill"}
    if not required_events.issubset(events) or "file" not in stat or "anon" not in stat:
        raise ValueError("incomplete cgroup v2 counters")
    return {
        "utc": utc_now(), "mem_total_kib": meminfo["MemTotal"],
        "mem_available_kib": meminfo["MemAvailable"],
        "cgroup_current_bytes": current, "cgroup_file_bytes": stat["file"],
        "cgroup_anon_bytes": stat["anon"],
        "events_high": events["high"], "events_max": events["max"],
        "events_oom": events["oom"], "events_oom_kill": events["oom_kill"],
    }


class Samplers:
    def __init__(self, host_path: Path, guest_path: Path, interval: float):
        self.host_path = host_path
        self.guest_path = guest_path
        self.interval = interval
        self.stop = threading.Event()
        self.started = threading.Event()
        self.errors = {"host": 0, "guest": 0}
        self.counts = {"host": 0, "guest": 0}

    def _host_loop(self):
        while not self.stop.is_set():
            try:
                proc = subprocess.run(
                    ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                     "-Command", HOST_COMMAND],
                    check=True, text=True, capture_output=True, timeout=12,
                )
                row = json.loads(proc.stdout.strip())
                required = {"utc", "host_total_bytes", "host_free_bytes",
                            "host_commit_bytes", "wsl_vm_working_set_bytes"}
                if set(row) != required or any(type(row[k]) is not int for k in required - {"utc"}):
                    raise ValueError("invalid host counter schema")
                with self.host_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                self.counts["host"] += 1
                self.started.set()
            except Exception:
                self.errors["host"] += 1
            self.stop.wait(self.interval)

    def _guest_loop(self):
        while not self.stop.is_set():
            try:
                row = read_guest_row()
                with self.guest_path.open("a", encoding="utf-8", newline="\n") as stream:
                    writer = csv.DictWriter(stream, fieldnames=GUEST_FIELDS, lineterminator="\n")
                    if self.counts["guest"] == 0:
                        writer.writeheader()
                    writer.writerow(row)
                self.counts["guest"] += 1
                self.started.set()
            except Exception:
                self.errors["guest"] += 1
            self.stop.wait(1.0)

    def start(self):
        self.host_thread = threading.Thread(target=self._host_loop, daemon=True)
        self.guest_thread = threading.Thread(target=self._guest_loop, daemon=True)
        self.host_thread.start()
        self.guest_thread.start()

    def finish(self):
        self.stop.set()
        self.host_thread.join(timeout=15)
        self.guest_thread.join(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True,
                        help="unique YYYYMMDD_HHMMSS label for output files")
    parser.add_argument("--baseline-seconds", type=int, default=15)
    parser.add_argument("--tail-seconds", type=int, default=180)
    parser.add_argument("--host-interval-seconds", type=float, default=2.0)
    parser.add_argument("--smoke-samples", action="store_true",
                        help="collect three paired samples only; do not start GPU runner")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (not re.fullmatch(r"\d{8}_\d{6}", args.run_id)
            or args.baseline_seconds < 5 or args.tail_seconds < 30
            or not 1 <= args.host_interval_seconds <= 10):
        raise SystemExit("invalid run id or duration")
    repo = Path(__file__).resolve().parents[1]
    results = repo / "results"
    output_directory = Path("/tmp") if args.smoke_samples else results
    host_path = output_directory / f"wsl_windows_memory_tail_{args.run_id}_host.jsonl"
    guest_path = output_directory / f"wsl_windows_memory_tail_{args.run_id}_guest.csv"
    runner_basename = f"qwen_temporal_position_allocator_memory_32f_{args.run_id}"
    runner_prefix = f"results/{runner_basename}"
    runner_path = results / f"{runner_basename}.json"
    for path in (host_path, guest_path, runner_path,
                 results / f"{runner_basename}_run_manifest.json"):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path.name}")
    if not shutil_which_powershell():
        raise SystemExit("powershell.exe is unavailable in this WSL process")
    host_path.touch(exist_ok=False)
    guest_path.touch(exist_ok=False)
    samplers = Samplers(host_path, guest_path, args.host_interval_seconds)
    samplers.start()
    try:
        baseline_deadline = time.monotonic() + (3 if args.smoke_samples else args.baseline_seconds)
        while time.monotonic() < baseline_deadline:
            if samplers.errors["host"] > 3 or samplers.errors["guest"] > 3:
                raise SystemExit("sampler smoke failed; GPU runner not started")
            time.sleep(0.2)
        if samplers.counts["host"] < 1 or samplers.counts["guest"] < 2:
            raise SystemExit("paired baseline rows missing; GPU runner not started")
        if args.smoke_samples:
            time.sleep(2)
            return 0
        command = [sys.executable,
                   str(repo / "scripts/run_qwen_temporal_position_gpu_check_and_publish.py"),
                   "--output-prefix", runner_prefix,
                   "--synthetic-frame-count", "32"]
        completed = subprocess.run(command, cwd=repo, check=False)
        if completed.returncode != 0:
            return completed.returncode
        time.sleep(args.tail_seconds)
        if not runner_path.is_file():
            raise SystemExit("runner summary missing after publisher returned")
        return 0
    finally:
        samplers.finish()
        print(json.dumps({"host_rows": samplers.counts["host"],
                          "guest_rows": samplers.counts["guest"],
                          "host_errors": samplers.errors["host"],
                          "guest_errors": samplers.errors["guest"],
                          "host_trace": host_path.name,
                          "guest_trace": guest_path.name,
                          "runner_summary": runner_path.name},
                         sort_keys=True))


def shutil_which_powershell():
    from shutil import which
    return which("powershell.exe")


if __name__ == "__main__":
    raise SystemExit(main())
