#!/usr/bin/env bash
# Prespecified dose-response (plan: experiments/2026-10-22_annotation_key_bias_dose_response_plan.md).
# Reuses the locked N=149 cohort + preflight; one smoke+full per dose; stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=20261022_01
E=149
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL=$D/known_media_sha_union_val_shamscreen_$TAG.json
LOCK=$D/annotation_key_bias_valE${E}_lock_$TAG.json
PRE=$D/annotation_key_bias_valE${E}_preflight_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_dose_response_pipeline_$TAG.log
exec >> "$LOG" 2>&1
echo "start $(date -u +%FT%TZ)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
for DOSE in ln4:1.3862943611198906 ln8:2.0794415416798357 ln16:2.772588722239781; do
  NAME=${DOSE%%:*}; VAL=${DOSE##*:}
  SMOKE=$D/annotation_key_bias_valE${E}_${NAME}_smoke_$TAG.json
  FULL=$D/annotation_key_bias_valE${E}_${NAME}_full_$TAG.json
  $PY $RUN --mode smoke $COMMON --bias-nats $VAL --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME smoke done $(date -u +%FT%TZ)"
  $PY $RUN --mode full $COMMON --bias-nats $VAL --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME full done $(date -u +%FT%TZ)"
done
echo "all doses done $(date -u +%FT%TZ)"
