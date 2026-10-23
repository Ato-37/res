#!/usr/bin/env bash
# WP0 (experiments/2026-10-23_pathway_knockout_replication_plan.md): waits for any running key-bias job, then smoke (n=3) -> full (n=80).
# Crash-tolerant: the runner appends one record per video; rerun this script with the same TAG and RESUME=1 to continue the full run.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_40}
RESUME=${RESUME:-0}
ROOT=$D/evidence_focusing_test_confirm_candidates_20261023_02/NExTVideo
EXCL=$D/pathway_placeholder_exclusion_20261023_01.json
PY=.venv/bin/python
LOG=$D/pathway_knockout_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f '^\.venv/bin/python scripts/run_(nextgqa_annotation_key_bias|pathway_knockout)' > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
COMMON="--dataset-split test --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
SMOKE=$D/pathway_knockout_smoke_$TAG.json
FULL=$D/pathway_knockout_full_$TAG.json
if [ "$RESUME" != "1" ]; then
  $PY scripts/run_pathway_knockout.py --mode smoke --n 3 $COMMON --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
  echo "smoke done $(date -u +%FT%TZ)"
fi
RES=""; [ "$RESUME" = "1" ] && RES="--resume"
$PY scripts/run_pathway_knockout.py --mode full --n 80 $COMMON $RES --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "full done $(date -u +%FT%TZ)"
