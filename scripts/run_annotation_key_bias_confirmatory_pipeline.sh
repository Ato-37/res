#!/usr/bin/env bash
# Chained, prespecified confirmatory pipeline (plan: experiments/2026-10-22_annotation_key_bias_confirmatory_val160_plan.md).
# screen (CPU) -> preflight(E) -> smoke -> full -> analysis. Stops at the first failure; never overwrites outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261022_01}
ROOT=$D/annotation_key_bias_val160_candidates_20261022_01/NExTVideo
EXCL0=$D/known_media_sha_union_val160_20261022_01.json
EXCL=$D/known_media_sha_union_val_shamscreen_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_confirmatory_pipeline_$TAG.log
exec > >(grep -v -i -E "warn|Loading weights" >> "$LOG") 2>&1
echo "start $(date -u +%FT%TZ)"
SCREEN=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --pool-n 155 --output-manifest $EXCL | tail -1)
echo "screen $SCREEN"
E=$(echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])')
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
LOCK=$D/annotation_key_bias_valE${E}_lock_$TAG.json
PRE=$D/annotation_key_bias_valE${E}_preflight_$TAG.json
SMOKE=$D/annotation_key_bias_valE${E}_smoke_$TAG.json
FULL=$D/annotation_key_bias_valE${E}_full_$TAG.json
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE
echo "preflight done $(date -u +%FT%TZ)"
$PY $RUN --mode smoke $COMMON --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE
echo "smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL
echo "full done $(date -u +%FT%TZ)"
$PY scripts/analyze_annotation_key_bias_confirmatory.py --receipt $FULL --expected-n $E --delta-min 0.025 --output results/annotation_key_bias_confirmatory_valE${E}_summary_$TAG.json
echo "analysis done $(date -u +%FT%TZ)"
