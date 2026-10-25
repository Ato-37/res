#!/usr/bin/env bash
# SGC (experiments/2026-10-23_self_guided_cropping_plan.md): waits for WP0/Stage 2/WP1/Stage 2b, then
# primary cohort (val, all outcome-blind candidates): screen -> smoke (n=3) -> full -> analysis;
# replication cohort (test candidates ranked 501-990, the Stage 2b cohort): full -> analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_80}
PY=.venv/bin/python
LOG=$D/self_guided_cropping_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f 'run_pathway_knockout_pipeline.sh|run_evidence_focusing_stage2_pipeline.sh|run_evidence_pathway_knockout_pipeline.sh|run_evidence_focusing_stage2b_pipeline.sh' > /dev/null || pgrep -f '^\.venv/bin/python scripts/run_(nextgqa_annotation_key_bias|pathway_knockout|evidence_pathway_knockout|self_guided_cropping)' > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
# ---- primary: val
ROOT=$D/annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo
EXCL0=$D/pathway_placeholder_exclusion_20261023_01.json
EXCL=$D/known_media_sha_union_val_sgc_shamscreen_$TAG.json
POOL=525
for attempt in 1 2 3; do
  if OUT=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --dataset-split val --pool-n $POOL --output-manifest $EXCL 2>&1); then break; fi
  FOUND=$(echo "$OUT" | grep -o 'but_found_[0-9]*' | tail -1 | grep -o '[0-9]*$' || true)
  echo "screen attempt $attempt failed (pool $POOL); found=${FOUND:-none}"
  [ -n "${FOUND:-}" ] || { echo "$OUT" | tail -3; exit 1; }
  POOL=$FOUND
done
SCREEN=$(echo "$OUT" | tail -1); echo "val screen $SCREEN"
EV=$(echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])')
COMMON="--dataset-split val --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
$PY scripts/run_self_guided_cropping.py --mode smoke --n 3 $COMMON --output $D/sgc_val_smoke_$TAG.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "val smoke done $(date -u +%FT%TZ)"
sleep 90  # let the smoke process release VRAM/host memory before the resource guard of the next process
for t in 1 2 3; do $PY scripts/run_self_guided_cropping.py --mode full --n $EV $COMMON --output $D/sgc_val_full_$TAG.json --resume 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3; [ -f $D/sgc_val_full_$TAG.json ] && break; sleep 120; done
echo "val full done $(date -u +%FT%TZ)"
(cd scripts && ../.venv/bin/python analyze_self_guided_cropping.py --receipt ../$D/sgc_val_full_$TAG.json --output ../results/sgc_val_summary_$TAG.json)
# ---- replication: test (Stage 2b cohort)
S2BLOG=$D/evidence_focusing_stage2b_pipeline_20261023_70.log
ET=$(grep -o '"eligible": [0-9]*' $S2BLOG | tail -1 | grep -o '[0-9]*$')
ROOT=$D/evidence_focusing_test_all_candidates_20261023_04/NExTVideo
EXCLT=$D/known_media_sha_union_test_stage2b_shamscreen_20261023_70.json
COMMONT="--dataset-split test --frames 32 --exclude-media-sha256 $EXCLT --video-root $ROOT --user-authorized-local-only"
sleep 90
for t in 1 2 3; do $PY scripts/run_self_guided_cropping.py --mode full --n $ET $COMMONT --output $D/sgc_test_full_$TAG.json --resume 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3; [ -f $D/sgc_test_full_$TAG.json ] && break; sleep 120; done
echo "test full done $(date -u +%FT%TZ)"
(cd scripts && ../.venv/bin/python analyze_self_guided_cropping.py --receipt ../$D/sgc_test_full_$TAG.json --output ../results/sgc_test_summary_$TAG.json)
echo "all done $(date -u +%FT%TZ)"
