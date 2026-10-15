#!/usr/bin/env python3
"""Sample aggregate Windows-host and WSL guest memory counters.

This read-only utility records no process names, PIDs, paths, or project data.
It is intended for short baseline comparisons, not causal attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


HOST_COMMAND = r"""
$ErrorActionPreference='Stop'
$os=Get-CimInstance Win32_OperatingSystem
$commit=Get-Counter '\Memory\Committed Bytes'
$proc=Get-CimInstance Win32_Process -Filter "Name='vmmemWSL'"
if (-not $proc) { throw 'vmmemWSL process is not present' }
$ws=[int64]$proc.WorkingSetSize
[pscustomobject]@{
  utc=[DateTime]::UtcNow.ToString('o')
  host_total_bytes=([int64]$os.TotalVisibleMemorySize * 1024)
  host_free_bytes=([int64]$os.FreePhysicalMemory * 1024)
  host_commit_bytes=[int64]$commit.CounterSamples[0].CookedValue
  vmmem_wss_bytes=$ws
} | ConvertTo-Json -Compress
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_guest() -> dict[str, int | str]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "Cached", "Active(file)", "Inactive(file)"}:
            fields = rest.split()
            if len(fields) < 2 or fields[1] != "kB":
                raise ValueError("invalid /proc/meminfo field")
            meminfo[key] = int(fields[0]) * 1024
    if not {"MemTotal", "MemAvailable"}.issubset(meminfo):
        raise ValueError("missing guest memory fields")
    root = Path("/sys/fs/cgroup")
    stat = {k: int(v) for k, v in
            (line.split() for line in (root / "memory.stat").read_text().splitlines())}
    events = {k: int(v) for k, v in
              (line.split() for line in (root / "memory.events").read_text().splitlines())}
    required = {"file", "active_file", "inactive_file", "anon"}
    if not required.issubset(stat):
        raise ValueError("missing cgroup v2 memory.stat fields")
    return {
        "utc": utc_now(), "kind": "guest", "meminfo_bytes": meminfo,
        "cgroup_current_bytes": int((root / "memory.current").read_text()),
        "cgroup_swap_current_bytes": int((root / "memory.swap.current").read_text()),
        "cgroup_memory_stat_bytes": {key: stat[key] for key in sorted(required)},
        "cgroup_memory_events": events,
    }


def read_host() -> dict[str, int | str]:
    proc = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
         "-Command", HOST_COMMAND],
        check=True, text=True, capture_output=True, timeout=15,
    )
    row = json.loads(proc.stdout.strip())
    expected = {"utc", "host_total_bytes", "host_free_bytes", "host_commit_bytes", "vmmem_wss_bytes"}
    if set(row) != expected or any(type(row[k]) is not int for k in expected - {"utc"}):
        raise ValueError("invalid Windows counter response")
    return {"kind": "host", **row}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, required=True, help="sampling duration (5–3600 seconds)")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path; existing files are never overwritten")
    parser.add_argument("--host-interval", type=float, default=5.0, help="seconds to wait after each host query, in addition to query time (1–30)")
    parser.add_argument("--guest-interval", type=float, default=1.0, help="seconds between guest queries (0.2–10)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (5 <= args.seconds <= 3600 and 1 <= args.host_interval <= 30
            and 0.2 <= args.guest_interval <= 10):
        raise SystemExit("invalid duration or sampling interval")
    if args.output.is_absolute() or ".." in args.output.parts:
        raise SystemExit("output must be a relative path under the repository")
    repo = Path(__file__).resolve().parents[1]
    output = repo / args.output
    if not output.resolve().is_relative_to(repo.resolve()):
        raise SystemExit("output must remain under the repository")
    manifest_path = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite existing output or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=False)

    errors = {"host": 0, "guest": 0}
    counts = {"host": 0, "guest": 0}
    lock = threading.Lock()
    stop = threading.Event()
    started = utc_now()
    def sampler(kind: str, interval: float, read):
        while not stop.is_set():
            try:
                row = read()
                with lock, output.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    counts[kind] += 1
            except Exception:
                with lock:
                    errors[kind] += 1
            stop.wait(interval)

    workers = [
        threading.Thread(target=sampler, args=("guest", args.guest_interval, read_guest)),
        threading.Thread(target=sampler, args=("host", args.host_interval, read_host)),
    ]
    started_workers = []
    try:
        for worker in workers:
            worker.start()
            started_workers.append(worker)
        time.sleep(args.seconds)
    finally:
        stop.set()
        for worker in started_workers:
            worker.join(timeout=20)
        if any(worker.is_alive() for worker in started_workers):
            raise RuntimeError("sampler thread failed to stop")
    finished = utc_now()
    manifest = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": finished,
        "requested_seconds": args.seconds,
        "host_post_query_wait_seconds": args.host_interval,
        "guest_interval_seconds": args.guest_interval,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": counts,
        "errors": errors,
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0 if all(value == 0 for value in errors.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
