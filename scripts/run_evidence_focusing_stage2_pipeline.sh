#!/usr/bin/env bash
# Stage 2 (confirmation, NExT-GQA test split) of experiments/2026-10-23_evidence_focusing_steering_plan.md at -ln256.
# Waits for the pathway-knockout replication (WP0) and any key-bias job, excludes WP0's media, screens sham geometry, then
# preflight -> smoke -> full -> analysis -> prespecified verdict.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/private_nextgqa
TAG=${1:-20261023_50}
WP0TAG=${WP0TAG:-20261023_40}
ROOT=$D/evidence_focusing_test_confirm_candidates_20261023_02/NExTVideo
PY=.venv/bin/python
RUN=scripts/run_nextgqa_annotation_key_bias_fp32_readout.py
LOG=$D/evidence_focusing_stage2_pipeline_$TAG.log
exec >> "$LOG" 2>&1
while pgrep -f 'run_pathway_knockout_pipeline.sh' > /dev/null || pgrep -f '^\.venv/bin/python scripts/run_(nextgqa_annotation_key_bias|pathway_knockout)' > /dev/null; do sleep 30; done
echo "start $(date -u +%FT%TZ) memtotal_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
EXCL0=$D/known_media_sha_union_test_stage2_$TAG.json
$PY - "$D" "$WP0TAG" "$EXCL0" <<'PYEOF'
import hashlib, json, os, sys
d, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
hashes = set()
full = f"{d}/pathway_knockout_full_{tag}.json"
partial = full + ".partial.jsonl"
if os.path.exists(full):
    hashes |= {r["media_sha256"] for r in json.load(open(full))["records"]}
elif os.path.exists(partial):
    hashes |= {json.loads(l)["media_sha256"] for l in open(partial) if l.strip()}
hashes.add(hashlib.sha256(b"placeholder: no test-split exclusions for the pathway-knockout plan").hexdigest())
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, json.dumps({"media_sha256": sorted(hashes), "schema_version": 1}).encode()); os.close(fd)
print("excluded media (WP0 + placeholder):", len(hashes))
PYEOF
EXCL=$D/known_media_sha_union_test_stage2_shamscreen_$TAG.json
POOL=300
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
LOCK=$D/evidence_focusing_testE${E}_lock_$TAG.json
PRE=$D/evidence_focusing_testE${E}_preflight_$TAG.json
$PY $RUN --mode preflight $COMMON --cohort-lock-output $LOCK --output $PRE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "preflight done $(date -u +%FT%TZ) E=$E"
VAL=-5.545177444479562
OPTS="--layers all --query-scope post_video --arm-mode keep --bias-nats $VAL"
SMOKE=$D/evidence_focusing_testE${E}_keep_neg_ln256_smoke_$TAG.json
FULL=$D/evidence_focusing_testE${E}_keep_neg_ln256_full_$TAG.json
$PY $RUN --mode smoke $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --output $SMOKE 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "smoke done $(date -u +%FT%TZ)"
$PY $RUN --mode full $COMMON $OPTS --cohort-lock $LOCK --preflight-receipt $PRE --smoke-receipt $SMOKE --output $FULL 2>&1 | grep -v -i -E "warn|Loading weights" | tail -2
echo "full done $(date -u +%FT%TZ)"
cd scripts
SUM=../results/evidence_focusing_stage2_testE${E}_summary_$TAG.json
../.venv/bin/python analyze_annotation_key_bias_dose_response.py --receipt ../$FULL --expected-n $E --delta-min 0.3 --output $SUM
../.venv/bin/python verdict_evidence_focusing_confirmation.py $SUM
echo "all done $(date -u +%FT%TZ)"
