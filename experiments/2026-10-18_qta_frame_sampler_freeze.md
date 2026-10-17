# QTA pre-review frame-sampler contract (2026-10-18)

## Purpose

Remove the remaining frame-sampling choice that could otherwise change which
human-reviewed intervals qualify as sampled evidence. Freeze this input rule
before stage-1 video review or any QTA outcome access.

## Contract

The QTA draft now specifies 32 endpoint-inclusive decoded RGB frames. For a
video with `N` decoded frames, use
`[round(i * (N - 1) / 31) for i in range(32)]` with Python 3.12 built-in
round-to-nearest, ties-to-even. Reject duplicate indices, missing or
non-increasing PTS, changed source bytes or full PTS across decode passes,
changed RGB pixels for any selected frame, and any selected temporal grid
exceeding **1/20 second (50 ms)** maximum deviation.
Do not replace rows or tune the frame count/tolerance based on human interval
overlap, baseline predictions, or treatment outcomes.

The 50 ms value is carried from prior three-video timing-feasibility work, before
human interval review or model-outcome inspection. It is outcome-blind with
respect to review/model outcomes, though informed by sample timing properties.
It is an engineering acceptance parameter, not an uncertainty interval,
calibration error bound, or claim that the sampled clock is accurate to 50 ms.
The decoder validates a uniform sampled PTS grid, checks input bytes/full PTS
and selected RGB pixels across two passes, and derives an exact effective
sample rate. The Qwen path passes that scalar rate as video metadata with
processor frame-resampling disabled. Because the source PTS can deviate from an
ideal uniform grid by up to the 50 ms limit, preserve actual PTS and maximum
deviation for every row, map review intervals using actual PTS, and require
the exact group sets to match those obtained from the processor's implied
uniform timestamps. The batch mapper now rejects any mismatch before scoring.
Then freeze one exact full-video processor batch per row for all nine forwards.

The runbook now requires the exact QTA draft, decoder/helper sources, runtime,
and processor metadata contract to be bound before the first review opens.
Independent static audit confirmed the index formula and 50 ms provenance,
and found/corrected the overstatement about all-frame pixel checks and the
non-dispositive boundary mismatch language. The prior Qwen smoke supports only
this pinned path, not a general processor guarantee.
The draft remains an engineering protocol, not a preregistration or execution
authorization.

## Real-video feasibility run and verification

Ran `scripts/run_qta_real_video_sampler_preflight.py` on the three locally
staged QTA candidate videos bound by the current candidate inventory
(`73/73` slots). Requested and effective rows were both 3; all
three videos decoded twice and produced 32 distinct sampled RGB frames. Full
decoded-frame counts were 510, 1,140, and 1,258. Effective sample rates were
1.825287, 0.815690, and 0.739134 FPS. Maximum sampled-grid deviations were
16.145, 16.145, and 16.355 ms, each below the fixed 50 ms limit. Source and
PTS integrity plus all 32 selected RGB hashes were checked; each staged media
hash matched its ordered inventory role and current source hashes matched the
sampler/protocol inventory. Exact command used:

```bash
PYTHONPATH=scripts ./.venv/bin/python scripts/run_qta_real_video_sampler_preflight.py \
  --candidate-inventory data/private_nextgqa/pre_review_inventory_qta_20261018_0840_01/pre_review_candidate_byte_inventory.json \
  --output-json data/private_nextgqa/qta_decode_preflight_20261018_0841.json
```

The detailed hash/PTS manifest remains local with mode 0600; no filenames, frame pixels,
questions, answers, or reviewer annotations were published.

This confirms technical decoder/sampler feasibility for those three staged
files only. The videos were not viewed for semantic evidence, no player/decoder
clock calibration was applied, and no model was loaded or faithfulness score
computed. The environment reports `torch.cuda.is_available() == False` in the
ordinary sandbox, so this CPU-only preflight did not use the GPU.

- QTA protocol draft SHA-256: `533cfb31e37be0dcf3a498e273741305981f802d09dfd62a8642f04edaaa1721`
- Human-review runbook SHA-256: `93f3fb3e9c37e390307d1786fdc576b5b8adfa1b885225c55e34290966f9e55e`
- `scripts/claim_mask_video_decoder.py`: `df0baf0b31afcac9715e35da2027e3e510f63cfc2bdf1aa1cd568e63045e38dd`
- `scripts/necessity_choice_margin_utils.py`: `5e61fa36a60abc03795c5c2206a0f80c0c6a6684716743336c156dd9790f9164`
- `scripts/presentation_time_utils.py`: `515b26ecca9a7e917b1c050ba0cede8dce3d9c837955dec28def31bbc48669a5`
- Existing capacity-smoke configuration source:
  `scripts/run_nextgqa_review_cohort_gpu_capacity_smoke.py`, SHA-256
  `ceb9a997e3fd9111b619dc7962efa1ac9922ac991da5b0350dcfc67f0a5ebdc7`.
- Decoder, uniform-index, and presentation-time suites: **54/54 passed** after the wording correction.
- Independent audit verified the actual two-pass guarantee is full source/PTS integrity plus selected-frame RGB equality (not all-frame pixel equality). It also verified that Qwen receives scalar FPS while source PTS may differ from the implied uniform grid by up to 50 ms; the runbook now preserves row-level deviation and rejects boundary assignment disagreements before scoring.
- Fresh private inventory: `data/private_nextgqa/pre_review_inventory_qta_20261018_0840_01`; 73/73 slots, 0 missing, current QTA draft/runbook and interval-mapper/runner hashes bound, mode 0700 directory/0600 file. It remains `candidate_only`, `joint_freeze=false`, `score_ready=false`, `execution_authorization=not_granted`, and `rights_status=unverified_local_only`.
- Latest focused mapper/runner/inventory tests: **50/50 passed**. Full `scripts` suite after the final runner/inventory changes: **1,602/1,602 passed**; expected negative-path fixture output and a Transformers deprecation warning were emitted.
- The reproducible media runner now requires and verifies the current candidate
inventory, its three ordered staged-media hashes, and current sampler/protocol
source hashes; it rejects non-default media directories. New production-path
gate compares actual PTS group assignment with endpoint-derived scalar-FPS uniform timestamps and fails closed on any difference; the processor-batch wrapper enables it and validates the supplied FPS against the PTS endpoints.
- Independent audit found no arithmetic blocker. A sub-50 ms jitter case crossing a group boundary is rejected, and a `temporal_patch_size=2` matching-grid case passes. The lower-level mapper documents that its check is opt-in; production Qwen callers go through the processor-batch wrapper, which requires it.
- `git diff --check`: passed.

- `scripts/qwen_review_interval_token_groups.py` SHA-256: `e4e6d4ff415a439f5bcfb99be1e7a05f19a87c68f86f4a6bf0d5dd1801f69eaf`
- `scripts/qwen_processor_batch_review_mapping.py` SHA-256: `1c8c9023d649b584302e53c151c52769c280930cc5c983b6f96a2cbbab008af0`
- `scripts/run_qta_real_video_sampler_preflight.py` SHA-256: `3035b1ea1c963fabe6f0721582885c6c0da299529ffb2cf90de38357b9bed723`

## Limits and faithfulness progress

This establishes video decoder/sampler feasibility only, not player-to-decoder
correspondence or evidence validity. The 50 ms criterion only describes
sampled-grid regularity; the separate 15-pair browser/decoder review and clock
calibration remain required. No model was loaded and no GPU inference ran.

`直接評価済み`: still 0/3 human-reviewed claims, 0/3 stage-2 prediction-link
judgments, 0/15 verified player-clock/decoder-frame correspondences, and 0/3
scores. No `DeltaF`, interval, test, or efficacy estimate exists.

`補助診断のみ`: 3/3 actual staged videos passed the fixed sampling gate, plus
50/50 focused tests and 1,602/1,602 full-suite tests. These results establish
media/implementation feasibility only and do not test explanation-to-evidence
faithfulness. Rights remain unverified; protected material stays local.

## Next action

Next implement the CPU-only, all-three-row pre-baseline readiness report while
keeping it non-authorizing. It should compose existing original-byte audits and
preserve the fixed denominator; do not run QTA scoring until rostered human
review, verified timing correspondences, and signed phase gates are complete.
