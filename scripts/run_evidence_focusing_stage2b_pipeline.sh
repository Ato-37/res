#!/usr/bin/env bash
# Stage 2b (second confirmation cohort, NExT-GQA test candidates ranked 501-990) of experiments/2026-10-23_evidence_focusing_steering_plan.md.
# Waits for WP0, Stage 2 and WP1, then screen -> preflight -> smoke -> full (-ln256, keep mode) -> cohort analysis -> pooled Stage 2 + 2b analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_70}
S2RECEIPT=${S2RECEIPT:-$D/evidence_focusing_testE80_keep_neg_ln256_full_20261023_50.json}
ROOT=$D/evidence_focusing_test_all_candidates_20261023_04/NExTVideo
EXCL0=$D/known_media_sha_union_test_stage2b_20261023_01.json
EXCL=$D/known_media_sha_union_test_stage2b_shamscreen_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/evidence_focusing_stage2b_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f 'run_pathway_knockout_pipeline.sh|run_evidence_focusing_stage2_pipeline.sh|run_evidence_pathway_knockout_pipeline.sh' > /dev/null || pgrep -f '^\.venv/bin/python scripts/run_(nextgqa_annotation_key_bias|pathway_knockout|evidence_pathway_knockout)' > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
POOL=490
for attempt in 1 2 3; do
  if OUT=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --dataset-split test --pool-n $POOL --output-manifest $EXCL 2>&1); then break; fi
  FOUND=$(echo "$OUT" | grep -o 'but_found_[0-9]*' | tail -1 | grep -o '[0-9]*$' || true)
  echo "screen attempt $attempt failed (pool $POOL); found=${FOUND:-none}"
  [ -n "${FOUND:-}" ] || { echo "$OUT" | tail -3; exit 1; }
  POOL=$FOUND
done
SCREEN=$(echo "$OUT" | tail -1)
echo "screen $SCREEN"
E=$(echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])')
COMMON="--dataset-split test --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
LOCK=$D/evidence_focusing_test2bE${E}_lock_$TAG.json
PRE=$D/evidence_focusing_test2bE${E}_preflight_$TAG.json
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ) E=$E"
OPTS="--layers all --query-scope post_video --arm-mode keep --bias-nats -5.545177444479562"
SMOKE=$D/evidence_focusing_test2bE${E}_keep_neg_ln256_smoke_$TAG.json
FULL=$D/evidence_focusing_test2bE${E}_keep_neg_ln256_full_$TAG.json
$PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "full done $(date -u +%FT%TZ)"
cd scripts
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$FULL --expected-n $E --delta-min 0.3 --output ../results/evidence_focusing_stage2b_testE${E}_summary_$TAG.json >/dev/null
../.venv/bin/python analyze_evidence_focusing_pooled.py --receipt ../$S2RECEIPT ../$FULL --output ../results/evidence_focusing_pooled_stage2_2b_summary_$TAG.json
echo "all done $(date -u +%FT%TZ)"
