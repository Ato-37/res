#!/usr/bin/env bash
# Prespecified suppression positive control (plan: experiments/2026-10-22_annotation_key_bias_positive_control_plan.md).
# New preflight (base runner hash changed) on the same N=149 cohort, then smoke+full for -ln16 and -ln256.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=20261022_02
E=149
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL=$D/known_media_sha_union_val_shamscreen_20261022_01.json
LOCK=$D/annotation_key_bias_valE${E}_poscontrol_lock_$TAG.json
PRE=$D/annotation_key_bias_valE${E}_poscontrol_preflight_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_positive_control_pipeline_$TAG.log
exec >> "$LOG" 2>&1
echo "start $(date -u +%FT%TZ)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ) lock_sha=$(sha256sum $LOCK | cut -c1-16)"
for DOSE in neg_ln16:-2.772588722239781 neg_ln256:-5.545177444479562; do
  NAME=${DOSE%%:*}; VAL=${DOSE##*:}
  SMOKE=$D/annotation_key_bias_valE${E}_${NAME}_smoke_$TAG.json
  FULL=$D/annotation_key_bias_valE${E}_${NAME}_full_$TAG.json
  $PY $RUN --mode smoke $COMMON --bias-nats $VAL --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME smoke done $(date -u +%FT%TZ)"
  $PY $RUN --mode full $COMMON --bias-nats $VAL --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME full done $(date -u +%FT%TZ)"
done
echo "all done $(date -u +%FT%TZ)"
