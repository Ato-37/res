#!/usr/bin/env bash
# Run a command while holding the machine-wide GPU lock, so two agents never run GPU/RAM-heavy jobs at once.
# Usage: scripts/with_gpu_lock.sh [-w SECONDS] COMMAND [ARGS...]   (default: wait up to 6 hours for the lock)
set -euo pipefail
WAIT=21600
if [ "${1:-}" = "-w" ]; then WAIT=$2; shift 2; fi
LOCK=${GPU_LOCK_FILE:-$HOME/.gpu_job.lock}
exec 9>"$LOCK"
if ! flock -w "$WAIT" 9; then echo "gpu_lock_timeout: another GPU job holds $LOCK" >&2; exit 75; fi
echo "gpu_lock_acquired $(date -u +%FT%TZ) pid=$$ cmd=$*" >> "$LOCK.log"
"$@"
