#!/usr/bin/env bash
# Step 2 of experiments/2026-10-23_annotation_key_bias_multiquery_control_plan.md (control passed): +ln4, +ln16, layers 1-28,
# all post-video query rows, same cohort/lock/preflight as the multi-query control.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
E=149
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL=$D/known_media_sha_union_val_shamscreen_20261022_01.json
LOCK=$D/annotation_key_bias_valE${E}_multiquery_lock_20261023_02.json
PRE=$D/annotation_key_bias_valE${E}_multiquery_preflight_20261023_02.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
TAG=20261023_03
LOG=$D/annotation_key_bias_multiquery_positive_pipeline_$TAG.log
exec >> "$LOG" 2>&1
echo "start $(date -u +%FT%TZ)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
for DOSE in ln4:1.3862943611198906 ln16:2.772588722239781; do
  NAME=${DOSE%%:*}; VAL=${DOSE##*:}
  SMOKE=$D/annotation_key_bias_valE${E}_multiquery_${NAME}_smoke_$TAG.json
  FULL=$D/annotation_key_bias_valE${E}_multiquery_${NAME}_full_$TAG.json
  OPTS="--layers all --query-scope post_video --bias-nats $VAL"
  $PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME smoke done $(date -u +%FT%TZ)"
  $PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME full done $(date -u +%FT%TZ)"
done
echo "all done $(date -u +%FT%TZ)"
