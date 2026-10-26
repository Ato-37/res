# NExT-GQA test-associated QTA exploratory preflight v2

## Purpose

Repair the v1 exploratory video-disjoint plan's selected-qid annotation issue
without changing its frozen calibration/evaluation video membership. V1 had one
selected evaluation row with a negative gsub interval start, so that plan is not
eligible for a model run. V2 deterministically selects a qid only among that
video's structurally valid gsub annotations. This is a CPU metadata/provenance
preflight, not a model experiment.

## Setup

- Dataset association: NExT-GQA official `test.csv`, `gsub_test.json`,
  `frame2time_test.json`, and `map_vid_vidorID.json`; all use local protected
  data and remain unverified for redistribution rights.
- Video membership: unchanged from private frozen plan
  `nextgqa_tlvs_exploratory_split_20261018_01.json` (343 calibration, 147
  evaluation). No qid or video is dropped; selection is among a video's qids
  whose gsub spans satisfy finite `0 <= start < end <= duration`.
- Selection: minimum SHA-256 of the fixed v2 protocol tag, qid marker, video ID,
  and qid. V1 plan, source code, and test.csv hashes are checked. V2 pins hashes
  of test.csv, gsub, frame2time, mapping, v1 plan, v1 splitter, and v2 builder.
- Each selected record binds the canonical digest of its full CSV row, gsub
  duration and spans, exact frame2time list, map target, and exact local media
  SHA-256. The frame2time list is checked for 32 finite numeric entries but is
  not treated as decoder/player calibration or required to be monotonic.
- Private plan and receipt are created exclusively under the mode-0700 private
  root with mode 0600. Stdout contains only status, counts, and hashes. No GPU,
  model, video decode, or model forward was run.

## Verified results

- Frozen candidate pool: 490 videos, each with at least one valid test qid;
  2,384 structurally valid qid/video annotation pairs.
- V2 allocation: 343/343 calibration and 147/147 evaluation videos received a
  valid selected qid. V1 membership was preserved; no evaluation row was
  silently removed.
- Receipt: 147/147 evaluation rows bound; 147 valid intervals, zero invalid
  intervals, zero missing joins. Receipt status is
  `preflight_receipt_only_no_model_scoring`.
- Private receipt SHA-256:
  `81a3b9f6a6c08fd98250c4f32db497e3e8be4bc9b7ca3e2f327587ee511d7299`.
- Focused CPU tests passed: 12/12, including source/input pinning, changed
  annotation content, missing joins, split overlap, valid-qid selection,
  all-or-none failure, private permissions, no overwrite, and content redaction
  from public receipt/stdout shapes. Python compilation passed.

## Follow-up: full-cohort CPU geometry gate

After implementing the separately named runner, the full requested evaluation
partition was checked through media decoding and processor token-geometry
preflight with no model load or CUDA call. This gate stopped all-or-none after
49 records completed; the next attempted record could not form the frozen
`single_overlap_anchor` evidence/control pair (`annotation_anchor_control_unavailable`).
The private failure receipt has effective denominator zero and contains no
score. The partition is therefore not ready for a GPU run under this frozen
policy. No rows were dropped, substituted, or rescored, and the policy must not
be changed against these observed outcomes without a separately frozen protocol.

The runner now supports a full-cohort `--preflight-only --n 147` receipt and
requires a successful, source-bound exact receipt before its full scoring mode.
This implementation gate has not passed for the current policy. It establishes
no annotation dependence effect or explanation faithfulness. The private
failure artifact and any future run receipt remain local-only.

## Interpretation and limitations

The preflight establishes integrity and interval-structure feasibility for the
chosen local rows only. It is not a clean or fresh cohort: both partitions are
inside the official NExT-GQA test-associated pool. Exact media hashes do not
exclude re-encodes or near-duplicate scenes. Structural interval checks do not
verify annotation semantics or align frame2time with the decoder clock. This
result is neither direct faithfulness evidence nor an efficacy result. Rights
status remains unverified and all private artifacts must stay local.

## Faithfulness progress

`未実施`: 0/147 direct explanation-faithfulness outcomes. The run measured no
Qwen scores, interventions, or faithfulness metric. Engineering progress is
limited to a validated cohort/provenance preflight. Human claim/frame/prediction
review gates remain open, so this receipt does not authorize the dependent
direct claim-linked scoring branch.

## Next action

Keep the frozen v1 artifact unchanged and label its selected-qid issue as
superseded by v2. Continue independent human-review setup and no-human-required
protocol checks. Do not start model scoring from this receipt alone; the
independent review/release and timing gates remain prerequisites. Any later
exploratory QTA proxy must remain labeled test-associated and non-confirmatory.
