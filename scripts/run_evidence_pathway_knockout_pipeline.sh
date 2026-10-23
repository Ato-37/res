#!/usr/bin/env bash
# WP1 (experiments/2026-10-23_evidence_pathway_knockout_plan.md): waits for WP0 and the evidence-focusing Stage 2, screens sham geometry
# on test candidates 301-500, then smoke (n=3) -> full -> analysis. Crash-tolerant via partial JSONL; rerun with RESUME=1 and the same TAG.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_60}
RESUME=${RESUME:-0}
ROOT=$D/evidence_pathway_test_candidates_20261023_03/NExTVideo
EXCL0=$D/known_media_sha_union_test_wp1_20261023_01.json
EXCL=$D/known_media_sha_union_test_wp1_shamscreen_$TAG.json
PY=.venv/bin/python
LOG=$D/evidence_pathway_knockout_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f 'run_pathway_knockout_pipeline.sh|run_evidence_focusing_stage2_pipeline.sh' > /dev/null || pgrep -f '^\.venv/bin/python scripts/run_(nextgqa_annotation_key_bias|pathway_knockout|evidence_pathway_knockout)' > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
if [ "$RESUME" != "1" ]; then
  POOL=200
  for attempt in 1 2 3; do
    if OUT=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --dataset-split test --pool-n $POOL --output-manifest $EXCL 2>&1); then break; fi
    FOUND=$(echo "$OUT" | grep -o 'but_found_[0-9]*' | tail -1 | grep -o '[0-9]*$' || true)
    echo "screen attempt $attempt failed (pool $POOL); found=${FOUND:-none}"
    [ -n "${FOUND:-}" ] || { echo "$OUT" | tail -3; exit 1; }
    POOL=$FOUND
  done
  SCREEN=$(echo "$OUT" | tail -1)
  echo "screen $SCREEN"
  echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])' > $D/evidence_pathway_knockout_N_$TAG.txt
fi
E=$(cat $D/evidence_pathway_knockout_N_$TAG.txt)
COMMON="--dataset-split test --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
SMOKE=$D/evidence_pathway_knockout_smoke_$TAG.json
FULL=$D/evidence_pathway_knockout_full_$TAG.json
if [ "$RESUME" != "1" ]; then
  $PY scripts/run_evidence_pathway_knockout.py --mode smoke --n 3 $COMMON --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
  echo "smoke done $(date -u +%FT%TZ)"
fi
RES=""; [ "$RESUME" = "1" ] && RES="--resume"
$PY scripts/run_evidence_pathway_knockout.py --mode full --n $E $COMMON $RES --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "full done $(date -u +%FT%TZ)"
cd scripts
../.venv/bin/python analyze_evidence_pathway_knockout.py --receipt ../$FULL --output ../results/evidence_pathway_knockout_summary_$TAG.json
echo "all done $(date -u +%FT%TZ)"
