#!/usr/bin/env bash
# Prespecified fresh-cohort replication (plan: experiments/2026-10-23_annotation_key_bias_fresh_cohort_replication_plan.md).
# Waits for any running experiment, then: screen (CPU) -> preflight(E) -> A (-ln256), B (+ln4), C (+ln16): smoke+full each -> analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_10}
ROOT=$D/annotation_key_bias_val_replication2_candidates_20261023_01/NExTVideo
EXCL0=$D/known_media_sha_union_val_rep2_20261023_01.json
EXCL=$D/known_media_sha_union_val_rep2_shamscreen_$TAG.json
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/annotation_key_bias_fresh_replication_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f '^\.venv/bin/python scripts/run_nextgqa_annotation_key_bias' > /dev/null; do sleep 20; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
SCREEN=$($PY scripts/screen_annotation_key_bias_sham_eligibility.py --exclude-media-sha256 $EXCL0 --video-root $ROOT --pool-n 198 --output-manifest $EXCL | tail -1)
echo "screen $SCREEN"
E=$(echo "$SCREEN" | $PY -c 'import json,sys; print(json.load(sys.stdin)["eligible"])')
COMMON="--dataset-split val --n $E --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
LOCK=$D/annotation_key_bias_valRep2E${E}_lock_$TAG.json
PRE=$D/annotation_key_bias_valRep2E${E}_preflight_$TAG.json
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ)"
for ARM in A_neg_ln256:-5.545177444479562 B_ln4:1.3862943611198906 C_ln16:2.772588722239781; do
  NAME=${ARM%%:*}; VAL=${ARM##*:}
  SMOKE=$D/annotation_key_bias_valRep2E${E}_${NAME}_smoke_$TAG.json
  FULL=$D/annotation_key_bias_valRep2E${E}_${NAME}_full_$TAG.json
  OPTS="--layers all --query-scope post_video --bias-nats $VAL"
  $PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME smoke done $(date -u +%FT%TZ)"
  $PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
  echo "$NAME full done $(date -u +%FT%TZ)"
done
cd scripts
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$D/annotation_key_bias_valRep2E${E}_A_neg_ln256_full_$TAG.json --expected-n $E --output ../results/annotation_key_bias_rep2_control_valE${E}_summary_$TAG.json
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$D/annotation_key_bias_valRep2E${E}_B_ln4_full_$TAG.json ../$D/annotation_key_bias_valRep2E${E}_C_ln16_full_$TAG.json --expected-n $E --output ../results/annotation_key_bias_rep2_positive_valE${E}_summary_$TAG.json
echo "all done $(date -u +%FT%TZ)"
