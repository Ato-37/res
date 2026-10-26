set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=data/private_nextgqa
PY=.venv/bin/python
exec >> $D/post_queue_20261026.log 2>&1
while pgrep -f "scratchpad/master_queue.sh" >/dev/null; do sleep 120; done
echo "post start $(date -u +%FT%TZ)"
stage() {
  local name=$1 runner=$2 analyzer=$3 n=$4 split=$5 root=$6 excl=$7; shift 7
  local AARGS=${ANALYZER_ARGS:-}
  local T=20261026d_$name
  local COMMON="--dataset-split $split --frames 32 --exclude-media-sha256 $excl --video-root $root --user-authorized-local-only $*"
  echo "== stage $name start $(date -u +%FT%TZ)"
  sleep 90
  $PY scripts/$runner --mode smoke --n 3 $COMMON --output $D/${name}_smoke_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
  if [ ! -f $D/${name}_smoke_$T.json ]; then echo "stage $name: smoke failed, skipped"; return 0; fi
  sleep 90
  $PY scripts/$runner --mode full --n $n $COMMON --output $D/${name}_full_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
  if [ ! -f $D/${name}_full_$T.json ]; then echo "stage $name: full failed"; return 0; fi
  (cd scripts && ../$PY $analyzer --receipt ../$D/${name}_full_$T.json --output ../results/${name}_summary_$T.json $AARGS) || true
  echo "== stage $name done $(date -u +%FT%TZ)"
}

WP0ROOT=$D/evidence_focusing_test_confirm_candidates_20261023_02/NExTVideo
WP0EXCL=$D/pathway_placeholder_exclusion_20261023_01.json
stage prog run_progressive_knockout.py analyze_progressive_knockout.py 80 test $WP0ROOT $WP0EXCL --edges video_to_text,text_to_last
echo "post done $(date -u +%FT%TZ)"
