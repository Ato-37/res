set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=data/private_nextgqa; T=20261025_85
ROOT=$D/evidence_focusing_test_all_candidates_20261023_04/NExTVideo
EXCLT=$D/known_media_sha_union_test_stage2b_shamscreen_20261023_70.json
exec >> $D/sgc_test_$T.log 2>&1
.venv/bin/python scripts/run_self_guided_cropping.py --mode full --n 278 --dataset-split test --frames 32 --exclude-media-sha256 $EXCLT --video-root $ROOT --user-authorized-local-only --output $D/sgc_test_full_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "full done $(date -u +%FT%TZ)"
(cd scripts && ../.venv/bin/python analyze_self_guided_cropping.py --receipt ../$D/sgc_test_full_$T.json --output ../results/sgc_test_summary_$T.json)
echo "analysis done $(date -u +%FT%TZ)"
