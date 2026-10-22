#!/usr/bin/env bash
# Step 1 of experiments/2026-10-23_annotation_key_bias_multiquery_control_plan.md: -ln256, layers 1-28, all post-video query rows.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
E=149
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL=$D/known_media_sha_union_val_shamscreen_20261022_01.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
TAG=20261023_02
LOCK=$D/annotation_key_bias_valE${E}_multiquery_lock_$TAG.json
PRE=$D/annotation_key_bias_valE${E}_multiquery_preflight_$TAG.json
LOG=$D/annotation_key_bias_multiquery_control_pipeline_$TAG.log
exec >> "$LOG" 2>&1
echo "start $(date -u +%FT%TZ)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ) lock_sha=$(sha256sum $LOCK | cut -c1-16)"
SMOKE=$D/annotation_key_bias_valE${E}_multiquery_neg_ln256_smoke_$TAG.json
FULL=$D/annotation_key_bias_valE${E}_multiquery_neg_ln256_full_$TAG.json
OPTS="--layers all --query-scope post_video --bias-nats -5.545177444479562"
$PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "full done $(date -u +%FT%TZ)"
