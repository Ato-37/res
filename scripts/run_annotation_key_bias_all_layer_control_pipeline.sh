#!/usr/bin/env bash
# Step 1 of experiments/2026-10-23_annotation_key_bias_all_layer_control_plan.md: -ln256, layers 1-28, same cohort/lock/preflight.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
E=149
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL=$D/known_media_sha_union_val_shamscreen_20261022_01.json
LOCK=$D/annotation_key_bias_valE${E}_poscontrol_lock_20261022_02.json
PRE=$D/annotation_key_bias_valE${E}_poscontrol_preflight_20261022_02.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
TAG=20261023_01
LOG=$D/annotation_key_bias_all_layer_control_pipeline_$TAG.log
exec >> "$LOG" 2>&1
echo "start $(date -u +%FT%TZ)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
SMOKE=$D/annotation_key_bias_valE${E}_alllayers_neg_ln256_smoke_$TAG.json
FULL=$D/annotation_key_bias_valE${E}_alllayers_neg_ln256_full_$TAG.json
$PY $RUN --mode smoke $COMMON --layers all --bias-nats -5.545177444479562 --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON --layers all --bias-nats -5.545177444479562 --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "full done $(date -u +%FT%TZ)"
