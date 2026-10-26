# Validation-split QA-annotation temporal-key bias (exploratory)

## Research question and scope

This experiment asks whether a fixed internal bias on visual keys corresponding
to NExT-GQA's QA-time annotation changes the correct answer's five-choice
log-probability margin more than an equal-size, disjoint hash-selected sham.
For each video, the foil is fixed to the baseline's highest-scoring incorrect
choice. The primary quantity is

`(gold − fixed baseline foil margin under annotation-key bias) − (same margin under hash sham)`.

This is a QA-annotation intervention-sensitivity proxy. The interval is not a
reviewed explanation-support span, clock-to-video correspondence was not
independently calibrated, and human review was not performed. The outcome is
not explanation faithfulness or grounding accuracy.

## Frozen cohort and implementation

The full local NExT-GQA validation archive contained 3,358 CSV rows. The
outcome-blind candidate scan mapped all 3,358 rows; after the local prior-media
index, 3,081 question/interval rows remained eligible on 525 unique videos,
with 274 rows excluded by media hash and three invalid intervals. The 20
selected media were the lowest SHA-256-ranked eligible media under the fixed
`[nextgqa-zip-candidate-selection-v1, val, media_sha256]` rule. All eligible
question pairs on those 20 videos were retained (104 pairs); extracted media
totaled 99,178,011 bytes. Selected videos had zero matches to the separately
applied 1,034-item private raw-SHA union. Freshness is only relative to these
available local exclusion histories; their completeness is unverified.

Outcome-blind preflight froze 20 distinct video/question rows, 32 decoded
frames per video, and 16 Qwen temporal groups per video. The strict annotation
window check passed on 20/20; decode failures were 0; all 20 rows had an
equal-key, disjoint sham. Annotation groups covered 1–7 of the 16 temporal
groups. No row was replaced. Inputs and generated receipts remain local and
owner-only.

The scorer used Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute, eager
attention, layers 5–16, and requested `+ln(2)` attention-logit bias on the
selected visual keys for the final prompt query predicting the first A–E
choice token. The realized BF16 mask dose was `0.69140625`. All masks passed a
full-tensor delta audit: only the intended cells changed, with one hook call
per selected layer.

## Verified GPU results

The one-row smoke passed 3/3 forwards. Its QA gold margin was 0.000 nat at
baseline, −0.125 nat under annotation bias, and 0.000 nat under sham. It passed
cohort, fixed-foil, finite-score, mask, layer-hook, and checkpoint-hash checks;
it was not added as an extra independent row.

The full frozen cohort completed **20/20 rows and 60/60 forwards**, with no
failed row, failed condition, or replacement. The runner-reported runtime was
235.01 seconds; peak allocated VRAM was 11.09 GiB, from 14.66 GiB initially
free. The RTX 5070 Ti reached 91–100% utilization in sampled mid-run checks.
The baseline, annotation-bias, and sham conditions had 15/20, 14/20, and
15/20 correct choices. Annotation bias changed the predicted choice on one
row, from correct to incorrect; sham changed no predicted choice.

| Contrast (gold-minus-fixed-foil margin) | Mean nat | Median nat | Range nat | Direction after `1e-7` numerical-zero tolerance |
|---|---:|---:|---:|---|
| Annotation bias − sham (primary) | +0.012500 | +0.00000003 | [−0.125000, +0.250000] | 8 positive, 5 approximately zero, 7 negative |
| Annotation bias − baseline | −0.006250 | ≈0 | [−0.125000, +0.125000] | 5 positive, 8 approximately zero, 7 negative |
| Sham − baseline | −0.018750 | ≈0 | [−0.250000, +0.125000] | 5 positive, 7 approximately zero, 8 negative |

For reference only, a descriptive one-sample t interval for the primary
contrast is [−0.04085, +0.06585] nat (n=20); it includes zero. No hypothesis
test or population claim is made. Rows were selected by a deterministic hash
from a prefiltered availability pool, not sampled from all videos, and the
primary effect is tiny relative to its spread: its median is effectively zero,
with 12/20 rows within the numerical-zero band or below. The one-answer
accuracy decrease is not proof of harm, but it rules out a claim that this run
improved QA correctness.

## Provenance and checks

- Source commit recorded in receipt: `da704d3135b0a3f72c90cff5d54bf5bab07d775f`.
- Runner SHA-256: `51cd4aa3f3bb816b45e3d4227a7fb50d29e0431f57218ce8828fea4b3daf6820`.
- Aggregate analyzer SHA-256: `801564f0b40661297f6416c67a5f56d77b072a08c40b98e16db41c317588903b`.
- Focused runner tests: **17/17 passed**; `py_compile`, CLI `--help`, and scoped
  `git diff --check` passed.
- Pinned model `config.json` SHA-256:
  `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`.
- Five weight-shard aggregate SHA-256:
  `48e0d4da86179caa0ddf827e76495641aacc52e16a57b1cf789fa90d2f3882f0`.
- Dataset hashes: `val.csv`
  `ac5a1eb1d6dee9cd84ab7f9c94ab345b2404646f24b79ac182cd2aa9e2d424c1`,
  `gsub_val.json`
  `4feb23e53102cb2d4779e857ff8f9fd9b552fc38435621406b2ed9b0bf76204b`, and
  `map_vid_vidorID.json`
  `1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685`.
- Cohort lock: `data/private_nextgqa/annotation_key_bias_val_lock_20261021_04.json`
  (SHA-256 `3d49b789f671947cfd24135030e23fc6f6f1545a077e96f4ddd571a3cfa7ea65`).
- Outcome-blind preflight: `data/private_nextgqa/annotation_key_bias_val_preflight_20261021_04.json`
  (SHA-256 `887635b0be67e126e7981c173449b66a5266edc02e3c0555c959aae7b1e19e7e`).
- Smoke: `data/private_nextgqa/annotation_key_bias_val_smoke_20261021_01.json`
  (SHA-256 `093bfe3a5a89aecd4c4f69d929881e3ebcdff6f47e48e28d9276d83816c70011`).
- Full receipt: `data/private_nextgqa/annotation_key_bias_val_full_20261021_01.json`
  (SHA-256 `fc9ccb72ddac58d7ae76401fdf5ff44ff82cfc8601620568d740c0ea94ea335e`,
  mode `0600`).
- The independent primary audit rechecked all 20 row bindings, 60 finite
  score vectors, fixed foils and margins, correctness labels, 12-layer hooks,
  exact mask deltas, private artifact modes, split/source/data hashes, and all
  five weight-shard hashes. No arithmetic or receipt-binding mismatch was
  found.
- Reproduce the protected-data-free aggregate output locally with:

  ```bash
  PYTHONPATH=scripts .venv/bin/python scripts/analyze_nextgqa_annotation_key_bias_result.py \
    --receipt data/private_nextgqa/annotation_key_bias_val_full_20261021_01.json
  ```

The analysis command emits aggregate values only. It has been run and returned
the counts and descriptive contrasts above. No protected artifacts were
published. Rights status remains unverified; handling was local-only under the
user's instruction.

## Interpretation and next action

This result does not support the proposed intervention as an answer-margin or
accuracy improvement: mean annotation-minus-baseline is slightly negative,
accuracy drops from 15/20 to 14/20, and annotation-minus-sham is small with
mixed row directions. It does show a working, auditable internal intervention
pipeline on 20 real validation videos. It is a bounded negative/inconclusive
mechanism result, not a positive faithfulness result.

The graduation thesis can center on whether internal visual steering improves
claim-specific video evidence dependence, and use this and the existing
experiments as evidence that changes in internal likelihood or attention do
not by themselves guarantee better answers or better grounded explanations.
The concrete empirical contribution is the matched-sham, fixed-foil
measurement and the observed score/output dissociation across real videos.
Direct explanation-support faithfulness remains unmeasured (formal review
0/25; this run's human review 0/20), so a claim-specific success conclusion
still requires reviewed support spans. Do not tune bias dose, layer set,
metric, or subgroup on these outcomes; move to an independently reviewed
claim-support cohort if human evidence becomes available.
