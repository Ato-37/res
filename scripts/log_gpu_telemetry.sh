#!/usr/bin/env bash
# Append a timestamped GPU telemetry line every INTERVAL seconds (default 2) so the state just before a crash survives a reboot.
# Usage: nohup bash scripts/log_gpu_telemetry.sh [interval_seconds] &    Output: data/gpu_telemetry/gpu_YYYYMMDD.csv
set -euo pipefail
cd "$(dirname "$0")/.."
INTERVAL=${1:-2}
mkdir -p data/gpu_telemetry
FIELDS=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks.max.sm,utilization.gpu,utilization.memory,memory.used,pstate,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active
while true; do
  OUT=data/gpu_telemetry/gpu_$(date -u +%Y%m%d).csv
  [ -s "$OUT" ] || echo "utc,$FIELDS" > "$OUT"
  LINE=$(nvidia-smi --query-gpu=$FIELDS --format=csv,noheader,nounits 2>&1 | tr -d '\r' | head -1) || LINE="nvidia-smi-error"
  echo "$(date -u +%FT%TZ),$LINE" >> "$OUT"
  sleep "$INTERVAL"
done
