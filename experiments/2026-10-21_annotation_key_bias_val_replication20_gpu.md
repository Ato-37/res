# Second validation cohort for QA-annotation key bias (exploratory replication)

## Protocol status

This is a post-result, same-protocol exploratory replication of
[`2026-10-21_annotation_key_bias_val_20_gpu.md`](2026-10-21_annotation_key_bias_val_20_gpu.md).
The plan was written before second-cohort selection in
[`2026-10-21_annotation_key_bias_val_replication_plan.md`](2026-10-21_annotation_key_bias_val_replication_plan.md).
The research question, primary score, bias dose/layers, fixed baseline foil,
sham algorithm, frame count, model and analysis were carried over unchanged.
The two cohorts are reported separately; no pooled estimate is used.

## Outcome-blind candidate construction

The complete NExT-GQA validation archive was scanned with the same prior
exposure index and 14 locally available sample manifests. The deterministic
candidate order was SHA-256 rank of
`[nextgqa-zip-candidate-selection-v1, val, media_sha256]`. The second builder
extracted the first 40 ranked, eligible unique media; direct byte hashing
verified that exactly 20 were the first cohort and exactly 20 were new. The
second-cohort lock's 20 media hashes equal that new set exactly, with zero
overlap against cohort one and zero matches to the previously known raw-media
SHA set. A mode-0600 exclusion manifest combined the 1,034-item prior raw-SHA
union with the 20 first-cohort media hashes. Outcome scores were not used for
selection.

Preflight passed **20 exact distinct validation rows**, strict QA interval
checks 20/20, decode failures 0, and sham-geometry failures 0. There were no
row replacements. Timing calibration and human review were not performed.

## Execution and audit

The same runner, dataset files, Qwen2.5-VL-7B-Instruct revision, five model
shards, 4-bit weights, BF16 compute, eager attention, layers 5–16,
`+ln(2)` requested bias, fixed gold-minus-baseline-best-foil score, and
equal-count disjoint hash sham were used. The smoke passed 3/3 forwards. The
full run completed **20/20 rows and 60/60 forwards**, with no failure or
replacement. Runtime was 236.72 seconds; peak allocated VRAM was 11.92 GiB.
Sampled RTX 5070 Ti utilization during the forward phase was 88–100%.

Independent receipt audit confirmed: second-cohort data, question and row
hashes are unique and disjoint from cohort one; every row binds to its second
preflight batch; all score vectors are finite; all fixed foils, margins,
correctness values and condition contrasts recompute; all masks change only
the intended cells; each biased condition calls all 12 selected layers once;
the five-weight hash, config, source and dataset bindings match; all private
artifacts are mode 0600. No raw questions or media are printed by the analyzer.

## Results, kept separate from cohort one

The primary annotation-bias-minus-sham margin contrast was mean **+0.025000
nat**, median approximately **0.000000** nat, and range **[−0.375000,
+0.375000]**. With `1e-7` treated as numerical zero, 8 rows were positive, 5
approximately zero, and 7 negative. Its unadjusted descriptive one-sample t
interval was **[−0.04752,+0.09752] nat**. This interval includes zero and is
not population-valid for this availability-filtered, deterministic
hash-ranked cohort. No hypothesis test or population inference was made.

Annotation bias minus baseline had mean **+0.031250 nat**, median near zero,
range [−0.125001,+0.250000], and descriptive t interval
[−0.01852,+0.08102]. Sham minus baseline had mean **+0.006250 nat**, median
near zero, range [−0.125000,+0.249999], and descriptive t interval
[−0.04205,+0.05455]. Five-choice accuracy was **11/20 in each condition**;
neither intervention changed a predicted choice relative to baseline.

## Interpretation and thesis value

This cohort also shows no QA accuracy improvement. The positive average
annotation-minus-sham score contrast coexists with an effectively zero median,
seven negative rows and a wide range, so the mean is not evidence of a robust
per-video effect. Cohort one's primary mean was +0.01250 nat with the same
near-zero median; its accuracy was 15/20 baseline, 15/20 sham and 14/20 with
annotation bias. Both independent groups therefore show small average
score shifts with no stable row-level separation and no accuracy gain, but
remain separate descriptive cohorts.

This strengthens the thesis topic **“動画VLMの内部介入によるスコア変化は
回答・時間区間・根拠依存の改善に結び付くか”** as a matched-control
measurement/negative-result study. The contribution is not an intervention
that demonstrably improves faithfulness; it is evidence that a real-video
internal key intervention can be audited and can shift an internal margin
slightly without improving QA accuracy in these two validation cohorts.
NExT-GQA QA-time intervals are not reviewed explanation support spans, so
direct explanation faithfulness remains unmeasured (formal support review
0/25; this second cohort's human review 0/20). Rights status remains
unverified and all content/results stay local.

## Provenance and reproduction

- Builder summary:
  `data/private_nextgqa/annotation_key_bias_val_replication_candidates_20261021_01/summary.json`
  SHA-256 `f0f4e87ab0bc43aa12a70a69a71f5b45b05c265ce881d33f69811ff5e621a22a`.
- Combined media exclusion: `data/private_nextgqa/known_media_sha_union_plus_cohort1_20261021_01.json`
  (owner-only; includes the known prior set and first cohort media hashes).
- Cohort lock: `data/private_nextgqa/annotation_key_bias_val_replication_lock_20261021_01.json`
  SHA-256 `f9d84acee104c275f146d14a81ce97f568e9c4e1cd64e8bf86770cec9a41e1dc`.
- Preflight: `data/private_nextgqa/annotation_key_bias_val_replication_preflight_20261021_01.json`
  SHA-256 `f111f774d5756549e7d3ad7523fad17cb13a0c73d3c3e9e94eab371cba097da0`.
- Smoke: `data/private_nextgqa/annotation_key_bias_val_replication_smoke_20261021_01.json`
  SHA-256 `ae36d69aed2c79fc55b290c1268186b1099f5b270e5fc70b461cf13f67f6f42c`.
- Full result: `data/private_nextgqa/annotation_key_bias_val_replication_full_20261021_01.json`
  SHA-256 `06298303c3387fa7efa62d03ed7e458bb10c8f55b2e44f480b05c1711aa97577`.
- Reproduce protected-data-free aggregates with:

  ```bash
  PYTHONPATH=scripts .venv/bin/python scripts/analyze_nextgqa_annotation_key_bias_result.py \
    --receipt data/private_nextgqa/annotation_key_bias_val_replication_full_20261021_01.json
  ```

The aggregate analyzer source SHA-256 is
`801564f0b40661297f6416c67a5f56d77b072a08c40b98e16db41c317588903b`.

All lock and receipt files are owner-only mode 0600. No experiment result was
published.
