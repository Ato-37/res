set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=data/private_nextgqa
PY=.venv/bin/python
LOG=$D/master_queue_20261026e.log
exec >> $LOG 2>&1
echo "master start $(date -u +%FT%TZ)"
VALROOT=$D/annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo
VALEXCL=$D/known_media_sha_union_val_sgc_shamscreen_20261025_83.json
WPROOT=$D/evidence_pathway_test_candidates_20261023_03/NExTVideo
WPEXCL=$D/known_media_sha_union_test_wp1_shamscreen_20261023_60.json
WPN=$(cat $D/evidence_pathway_knockout_N_20261023_60.txt)
stage() {
  local name=$1 runner=$2 analyzer=$3 n=$4 split=$5 root=$6 excl=$7; shift 7
  local AARGS=${ANALYZER_ARGS:-}
  local T=20261026c_$name
  local COMMON="--dataset-split $split --frames 32 --exclude-media-sha256 $excl --video-root $root --user-authorized-local-only $*"
  echo "== stage $name start $(date -u +%FT%TZ)"
  sleep 90
  local RES=""
  if [ -f $D/${name}_full_$T.json.partial.jsonl ]; then
    RES="--resume"; echo "stage $name: resuming from partial"
  else
    $PY scripts/$runner --mode smoke --n 3 $COMMON --output $D/${name}_smoke_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
    if [ ! -f $D/${name}_smoke_$T.json ]; then echo "stage $name: smoke failed, skipped"; return 0; fi
    sleep 90
  fi
  $PY scripts/$runner --mode full --n $n $COMMON $RES --output $D/${name}_full_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
  if [ ! -f $D/${name}_full_$T.json ]; then echo "stage $name: full failed"; return 0; fi
  (cd scripts && ../$PY $analyzer --receipt ../$D/${name}_full_$T.json --output ../results/${name}_summary_$T.json $AARGS) || true
  echo "== stage $name done $(date -u +%FT%TZ)"
}
# test cohort = SGC test / Stage 2b cohort
TESTROOT=$D/evidence_focusing_test_all_candidates_20261023_04/NExTVideo
TESTEXCL=$D/known_media_sha_union_test_stage2b_shamscreen_20261023_70.json
stage cenval run_crop_ensemble.py analyze_crop_ensemble.py 300 val $VALROOT $VALEXCL
stage centest run_crop_ensemble.py analyze_crop_ensemble.py 278 test $TESTROOT $TESTEXCL
stage wp1dval run_wp1_overlap_dose.py analyze_wp1_overlap_dose.py 300 val $VALROOT $VALEXCL
stage wp2 run_wp2_cited_evidence.py analyze_wp2_cited_evidence.py 300 val $VALROOT $VALEXCL
stage wp1s run_wp1_overlap_dose.py analyze_wp1_overlap_dose.py $WPN test $WPROOT $WPEXCL --design sliding
stage wp1d run_wp1_overlap_dose.py analyze_wp1_overlap_dose.py $WPN test $WPROOT $WPEXCL
(cd scripts && ../$PY analyze_selective_prediction.py --val ../$D/sgc_val_full_20261025_84.json --test ../$D/sgc_test_full_20261025_85.json --output ../results/selective_prediction_20261026.json)
echo "master done $(date -u +%FT%TZ)"
