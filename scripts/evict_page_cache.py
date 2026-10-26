#!/usr/bin/env python3
"""Evict clean page-cache pages of large local files (videos, model weights) with posix_fadvise(DONTNEED); no root needed.

WSL2 keeps file cache as VM memory that the Windows host counts as used, so long GPU jobs that stream videos and re-read weights push host memory
use above 90%. Safe: only drops clean cache (data are re-read from disk when needed). Usage: evict_page_cache.py [--loop SECONDS] [paths...]
"""
import argparse
import os
import sys
import time
from pathlib import Path

DEFAULTS = ("data/private_nextgqa", str(Path.home() / ".cache/huggingface/hub"))


def evict(paths) -> int:
    n = 0
    for root in paths:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    fd = os.open(p, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return n


def cached_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1]) // 1024
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = once)")
    ap.add_argument("paths", nargs="*", default=list(DEFAULTS))
    a = ap.parse_args()
    while True:
        before = cached_mb()
        n = evict(a.paths)
        print(f"{time.strftime('%FT%TZ', time.gmtime())} evicted {n} files; cached {before} -> {cached_mb()} MB", flush=True)
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
