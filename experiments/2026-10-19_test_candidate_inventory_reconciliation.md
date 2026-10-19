# Local test-video candidate inventory reconciliation (2026-10-19)

## Purpose

Reconcile apparently conflicting historical counts for available local
NExT-GQA test videos before deciding whether a new real-video GPU run has a
defensible cohort. This is an outcome-blind, CPU-only availability and exact
byte-exposure audit. It does not score model outputs or establish direct
faithfulness.

## Method and reproduced observations

Used `scripts/audit_nextgqa_local_media_byte_diversity.py` against the current
`NExT-GQA/datasets/nextgqa/test.csv` and official
`map_vid_vidorID.json`, with local media root
`data/private_nextgqa/zip_gold_candidates_20261017_01/NExTVideo`. The script
resolves each CSV video ID through the official map, hashes media bytes in
place, and emits aggregate counts only. I ran it separately with the supplied
partial 2026-10-16 exposure index and the expanded known-history 2026-10-18
index; both runs exited successfully.

SHA-256 provenance: current test CSV
`0443ede53f521de4cbfb5e4fb02253926b90f81f62041b18bd17207a3511d114`, official
video map `1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685`,
audit script
`48dade10a60b71d6422d8e6697d59f5254db57dc86a1989a427ab470287b7ec8`, partial
index `dc198161c4e38e2f909eda6db7cd33ae0223a0a47bc8a8babfaf545a1670a59a`, and
expanded index
`1bf45806cb76911885f238944bb174d92a97cc83e103eaa629eb153497e61e5d`.

Reproduction command (run twice, changing only `--prior-index` to the other
listed index):

```bash
PYTHONPATH=scripts ./.venv/bin/python scripts/audit_nextgqa_local_media_byte_diversity.py \
  --test-csv NExT-GQA/datasets/nextgqa/test.csv \
  --video-map-json NExT-GQA/datasets/nextgqa/map_vid_vidorID.json \
  --video-root data/private_nextgqa/zip_gold_candidates_20261017_01/NExTVideo \
  --prior-index data/private_nextgqa/prior_exposure_expanded_known_history_20261018_01.json \
  --hmac-key data/private_nextgqa/prior_exposure_hmac_key_20261016_01.bin
```

| Aggregate | Result |
| --- | ---: |
| Current test CSV rows / distinct video IDs | 5,553 / 990 |
| Official-map-resolved local MP4s / missing | 490 / 500 |
| Distinct byte hashes / exact duplicate groups | 490 / 0 |
| Bytes streamed | 1,990,584,365 |
| Matches to supplied partial index | 0 / 490 |
| Matches to expanded known-history index | 21 / 490 |
| Candidates unmatched by both supplied indexes | 469 / 490 |

The two prior indexes are not authenticated as complete histories. Their
non-matches therefore mean only “not found in these supplied indexes”; they do
not establish that any candidate is fresh, unexposed, or source-event
independent. Rights status remains unverified; all material stayed local.

## Reconciliation and interpretation

The 490 count is reproducible for the ZIP-candidate media root and current
working-tree CSV snapshot. Earlier figures of 533 media files and 79 eligible
candidates refer to a different media root and different filtering stage.
Earlier 500-file and 85-match counts likewise refer to another inventory
snapshot. They are not contradictory counts of the same cohort. A historical
“11 matches” figure cannot currently be reproduced from the two available
indexes; the exact older ledger or source snapshot is unresolved. Previous
logs report that the current checkout's CSV was edited after some of those
counts, so the old denominators cannot be carried over to this snapshot.

This inventory is entirely in the test split. The research plan reserves test
for confirmation and requires a development/validation cohort with history
exclusions; the 490 candidates cannot be substituted as validation. Further,
the formal direct-faithfulness gates remain incomplete: 0/3 generated claims
reviewed, 0/15 player-clock/decoder-frame correspondences verified, 0/3
prediction-link judgments, and 0/3 direct scores. No direct effect estimate,
confidence interval, or test exists. The result does not justify a GPU score
run, and it does not change any prior branch decision.

## Next action

Continue the low-friction human review handoff using the local nickname-to-
question HTML artifact. For GPU work, first establish a usable validation
media cohort under the current CSV snapshot and frozen exposure rules, then
complete the review and scoring authorization gates. Do not describe the 469
non-matches as fresh and do not use these test candidates for development
scoring.

## Validation-media follow-up

To check for a development split without substituting test, I ran the same
aggregate hashing function against the current `val.csv` and the designated
local `NExTVideo` root. The current file contains 3,358 rows / 567 distinct
video IDs; 33 mapped local videos were present, with 33 unique byte hashes.
All 33 matched the supplied partial exposure index. The separate expanded
known-history index matched 0/33, which shows the two supplied index snapshots
are not interchangeable; the conservative union still excludes all 33. The
full-history completeness flag is false, so these counts do not prove what was
seen outside the supplied indices. The current validation CSV SHA-256 is
`ac5a1eb1d6dee9cd84ab7f9c94ab345b2404646f24b79ac182cd2aa9e2d424c1`.

Reproduction uses the function already identified above, with `test_csv` set
to `NExT-GQA/datasets/nextgqa/val.csv` and `video_root` set to the local
`nextgqa_sample/NExTVideo` directory. It returns only aggregate counts; no
video was decoded or displayed. An additional metadata-only scan found 0/567
validation files in the ZIP candidate root and 0/3,870 train files across both
roots. Therefore there is no currently available validation cohort after the
supplied exposure exclusions. The 490 test candidates remain unsuitable for
development scoring under the frozen split policy.

**Updated next action:** obtain the missing validation media through the
already permitted local workflow, or use the current low-friction human-review
artifact to finish the human-dependent gates. Do not run model scoring on the
test-only inventory to work around the missing validation cohort. Direct
faithfulness remains 0/3 claims scored; this availability audit adds no effect
estimate.
