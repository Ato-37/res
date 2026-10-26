# Local exposure census and development-cohort decision (2026-10-22)

## Purpose

Determine whether current local NExT-GQA video files contain an outcome-unseen
validation cohort for a distinct, defensible video faithfulness experiment.
This was an aggregate-only CPU inventory. It did not open model inputs for
inference, select by outcomes, or run a GPU experiment.

## Exposure inventory

Scanned all 1,170 JSON files under `data/private_nextgqa` through the existing
safe exposure-manifest builder. Of these, 390 were recognized as claim
generation packets or scoring receipts. The new owner-only manifest
`data/private_nextgqa/claim_exposure_union_manifest_20261022_01.json` contains
204 distinct source-video SHA-256 values; file mode is `0600`, and its SHA-256
is `6cb6c488f181ae4962d24d1b01e04e1c4fd76f01c2a19b97d1148ce0b6844080`.
This heuristic unions every currently present local JSON receipt/packet, but
cannot authenticate completeness of historical model exposure or account for
unrecorded manual viewing.

Independently streamed and SHA-256 hashed the MP4 files in all 12 local
`NExTVideo` roots under `data/private_nextgqa`: 646 files, 539 distinct byte
hashes, 2.53 GiB total. Compared with the local exposure union, 89 distinct
hashes were recorded as exposed and 450 were absent from the recognized local
claim/scoring history. Within the validation-pool roots, all 43 available
files were present in the local exposure union. The 490-file ZIP-derived
candidate root is test split; its 450 locally unmatched files are not eligible
for validation/development because test is reserved for confirmation. A
non-match is not proof of freshness: manifest historical completeness is
explicitly unauthenticated.

The builder and media-diversity audit focused tests passed 12/12. The manifest
is private and was not published. Dataset/model rights remain unverified.

## External annotation-source lead (schema audit only)

An official-source search found a plausible future alternative to manual
claim/frame review: the Google DeepMind MINERVA dataset reports roughly 1.5K
five-choice video QA items with manually annotated reasoning traces, including
timestamps where needed and key actions/objects. Its repository says the JSON
contains YouTube IDs plus annotations, not video bytes, and provides materials
under CC-BY 4.0. The official JSON download redirected to an account login in
the available browser, so local availability and per-item timestamp coverage
remain unverified. This is not ready for GPU scoring and its traces may require
item-level parsing/selection before use.

Official sources: [MINERVA repository](https://github.com/google-deepmind/neptune)
and [ActivityNet Captions task page](https://activity-net.org/challenges/2021/tasks/anet_captioning.html).
ActivityNet Captions separately reports 20K videos and 100K human descriptions
with unique start/end times, but those descriptions are not QA explanations.
The previously downloaded Perception Test sample is also unsuitable as a
substitute: its local aggregate preflight has only 4 videos and 18 MC questions
per split, with 5 overlapping question IDs across train/valid, and no model
accuracy measurement. These alternatives motivate data-fit work only; none
changes the current experiment or direct-faithfulness count.

## Research decision

No currently available, non-exposed validation cohort is established for a new
GPU proxy experiment. The current test candidates cannot be repurposed to tune
or develop the method; the validation candidates are already in the known
exposure union. Existing claim, QA-interval, temporal, frame-mean, and NOAA
proxies are already run on their frozen cohorts and have mixed/null results.
Running another outcome on those rows would not discriminate a new hypothesis.
Therefore no GPU inference was justified by this census.

Direct generated-claim faithfulness remains 0/3 on the current artifact and
0/25 formally reviewed. The existing scorer smoke remains engineering-only.
The next direct action is to consume a response bound to the current artifact;
for further development without those judgments, first obtain or identify a
validation source with sufficiently complete exposure history, then freeze a
distinct estimand/cohort before model execution. Do not call the 450 test-split
nonmatches fresh or use them to develop the intervention.
