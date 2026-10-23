#!/usr/bin/env bash
# Resume of Stage 1 (experiments/2026-10-23_evidence_focusing_steering_plan.md) after a PC bugcheck interrupted the -ln256 full run.
# Reuses the E=123 lock/preflight and the completed -ln4 receipt; reruns smoke+full for -ln256 under a new tag, then analyses both doses.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
OLD=20261023_20
TAG=${1:-20261023_30}
E=123
ROOT=$D/annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo
EXCL=$D/known_media_sha_union_val_dev_shamscreen_$OLD.json
LOCK=$D/annotation_key_bias_valDevE${E}_lock_$OLD.json
PRE=$D/annotation_key_bias_valDevE${E}_preflight_$OLD.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_evidence_focusing_dev_resume_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f '^\.venv/bin/python scripts/run_nextgqa_annotation_key_bias' > /dev/null; do sleep 20; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
NAME=keep_neg_ln256; VAL=-5.545177444479562
SMOKE=$D/annotation_key_bias_valDevE${E}_${NAME}_smoke_$TAG.json
FULL=$D/annotation_key_bias_valDevE${E}_${NAME}_full_$TAG.json
OPTS="--layers all --query-scope post_video --arm-mode keep --bias-nats $VAL"
$PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "$NAME smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "$NAME full done $(date -u +%FT%TZ)"
cd scripts
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$D/annotation_key_bias_valDevE${E}_keep_neg_ln4_full_$OLD.json ../$FULL --expected-n $E --output ../results/evidence_focusing_dev_valE${E}_summary_$TAG.json
echo "all done $(date -u +%FT%TZ)"
