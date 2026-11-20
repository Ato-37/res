#!/usr/bin/env bash
# Server queue (2026-11-20): dev retake on one lab GPU server, official-compatible env (venv310).
# Runs jobs one at a time; waits until no other compute process is on the GPU before each job; never overwrites outputs.
# Usage (on the server): setsid nohup bash scripts/mtf_v2/queue_dev40_server.sh TAG > $W/logs/queue_TAG.log 2>&1 < /dev/null &
set -u
TAG=${1:?tag}
source /nas.dbms/okamoto/mtf/env.sh
source "$W/venv310/bin/activate"
cd "$W/res"
SPLIT=scripts/mtf_v2/tvbench_split_v1.json
COMMON="--split $SPLIT --part dev --n_per_task 40 --ks 1,3,5,7,9,13 --null --fp32_readout"
echo "queue $TAG start $(date -Is) host=$(hostname) res=$(git log -1 --format=%h)"

wait_gpu_free() {
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    echo "$(date -Is) GPU busy, waiting"; sleep 120
  done
}

run_job() {  # name tasks
  local name=$1 tasks=$2 out="$W/outputs/${TAG}_$1.jsonl"
  if [ -e "$out" ]; then echo "skip $name: $out exists"; return; fi
  wait_gpu_free
  echo "$(date -Is) start $name tasks=[$tasks]"
  python scripts/mtf_v2/flow_v2.py --out "$out" --tasks "$tasks" $COMMON && echo "$(date -Is) done $name" || echo "$(date -Is) FAIL $name"
}

# Job 1: the four tasks whose videos come from the TVBench HF release
run_job dev40_4tasks "Action Sequence,Object Count,Scene Transition,Moving Direction"
# Job 2: Action Antonym only if all 320 NTU-derived videos are in place (not on HF; transferred separately)
AA=$W/data/TVBench/video/action_antonym
if [ "$(ls "$AA" 2>/dev/null | wc -l)" -ge 320 ]; then run_job dev40_aa "Action Antonym"; else echo "skip Action Antonym: videos missing in $AA"; fi
echo "queue $TAG end $(date -Is)"
