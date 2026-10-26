set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=data/private_nextgqa; T=20261025_84; EXCL=data/private_nextgqa/known_media_sha_union_val_sgc_shamscreen_20261025_83.json
ROOT=$D/annotation_key_bias_val_dev_candidates_20261023_01/NExTVideo
COMMON="--dataset-split val --frames 32 --exclude-media-sha256 $EXCL --video-root $ROOT --user-authorized-local-only"
exec >> $D/sgc_direct_$T.log 2>&1
.venv/bin/python scripts/run_self_guided_cropping.py --mode smoke --n 3 $COMMON --output $D/sgc_val_smoke_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "smoke done $(date -u +%FT%TZ)"
sleep 90
.venv/bin/python scripts/run_self_guided_cropping.py --mode full --n 508 $COMMON --output $D/sgc_val_full_$T.json 2>&1 | grep -v -i -E "warn|Loading weights" | tail -3
echo "full done $(date -u +%FT%TZ)"
(cd scripts && ../.venv/bin/python analyze_self_guided_cropping.py --receipt ../$D/sgc_val_full_$T.json --output ../results/sgc_val_summary_$T.json)
echo "analysis done $(date -u +%FT%TZ)"
