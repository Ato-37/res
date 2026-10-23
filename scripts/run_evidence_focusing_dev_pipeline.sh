#!/usr/bin/env bash
# Stage 1 (development, exploratory) of experiments/2026-10-23_evidence_focusing_steering_plan.md.
# screen (CPU, pool size found by the freezer's own eligibility) -> preflight -> keep-mode runs at -ln4 and -ln256 -> analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_20}
ROOT=$D/annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo
EXCL0=$D/known_media_sha_union_val_dev_20261023_01.json
EXCL=$D/known_media_sha_union_val_dev_shamscreen_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_evidence_focusing_dev_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f '^\.venv/bin/python scripts/run_nextgqa_annotation_key_bias' > /dev/null; do sleep 20; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
POOL=125
for attempt in 1 2; do
  if OUT=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --pool-n $POOL --output-manifest $EXCL 2>&1); then break; fi
  FOUND=$(echo "$OUT" | grep -o 'but_found_[0-9]*' | tail -1 | grep -o '[0-9]*$' || true)
  echo "screen attempt $attempt failed (pool $POOL); found=${FOUND:-none}"
  [ -n "${FOUND:-}" ] || { echo "$OUT" | tail -3; exit 1; }
  POOL=$FOUND
done
SCREEN=$(echo "$OUT" | tail -1)
echo "screen $SCREEN"
E=$(echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])')
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
LOCK=$D/annotation_key_bias_valDevE${E}_lock_$TAG.json
PRE=$D/annotation_key_bias_valDevE${E}_preflight_$TAG.json
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ) E=$E"
for ARM in keep_neg_ln4:-1.3862943611198906 keep_neg_ln256:-5.545177444479562; do
  NAME=${ARM%%:*}; VAL=${ARM##*:}
  SMOKE=$D/annotation_key_bias_valDevE${E}_${NAME}_smoke_$TAG.json
  FULL=$D/annotation_key_bias_valDevE${E}_${NAME}_full_$TAG.json
  OPTS="--layers all --query-scope post_video --arm-mode keep --bias-nats $VAL"
  $PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME smoke done $(date -u +%FT%TZ)"
  $PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME full done $(date -u +%FT%TZ)"
done
cd scripts
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$D/annotation_key_bias_valDevE${E}_keep_neg_ln4_full_$TAG.json ../$D/annotation_key_bias_valDevE${E}_keep_neg_ln256_full_$TAG.json --expected-n $E --output ../results/evidence_focusing_dev_valE${E}_summary_$TAG.json
echo "all done $(date -u +%FT%TZ)"
