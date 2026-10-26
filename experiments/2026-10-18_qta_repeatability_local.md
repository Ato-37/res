# QTA fixed-resolution repeatability diagnostic (local result)

## Purpose and scope

Measure whether the frozen untreated QTA temporal-group selector returns the
same selected support on two separate forwards of the identical materialized
processor batch. This distinguishes same-batch run-to-run variation from the
16/32-frame resolution change measured in the companion diagnostic. It is a
fixed-cohort selector-reliability check only: it is not an explanation
faithfulness measure, an intervention effect, or a population estimate.

The run and derived result remain local because the inputs use protected
NExT-GQA/video material. Rights status is unverified. Output contains only
fixed-ordinal metrics and aggregate hashes; no IDs, questions, answers, logits,
selected group indices, or media rows are saved.

## Frozen setup and execution

- Cohort: the same deterministic first three eligible NExT-GQA test candidates
  selected before decoding; no replacement. Each candidate uses the existing
  qid-sorted question selection.
- Input: one 32-frame sampled clip per candidate, processor output materialized
  once and reused as the same in-memory batch for both forwards.
- Selector: untreated question-token attention averaged over heads/layers
  5–16; select `ceil(0.10*T)` complete temporal groups. The model stays in
  `eval()` and the random seed is initialized once per process, not between
  forwards.
- Model/runtime: pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
  PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5070 Ti. The GPU was available with
  14.53 GiB free before execution. Each run hashes all regular files in the
  local model snapshot immediately before loading and after forwards and
  requires the content hashes to match.
- Smoke: one pair, 2/2 completed forwards; elapsed 56.06 s, including 24.09 s
  of model snapshot hashing. Passed with Jaccard 1.0 and center-W1 0.0.
- Main run: 3/3 fixed pairs, 6/6 completed forwards, no failures; elapsed
  84.06 s, including 26.35 s of model snapshot hashing.
- Main result: `data/private_nextgqa/qta_repeatability_20261018_2154/full.json`,
  mode `0600` under a mode `0700` directory, SHA-256
  `c3b2604345de4815cf0f2526679ea4f631804fe7727601ecad1e4855de1bca41`.
  It binds source, annotation, processor, candidate-exclusion, batch and input
  hashes. The runner SHA-256 recorded in the artifact is
  `9c3cec80169ceec1a6029f9e8a0a1b92cf864fa4cd82a71c80fe61c7341bf6da`.
- Independent-process replication: reran the same fixed three candidates in a
  second process with a fresh model load. It again completed 3/3 pairs and 6/6
  forwards; all per-pair and aggregate metrics, source hashes, and model
  snapshot hashes were identical. Elapsed time was 90.18 s (25.75 s hashing).
  Its local artifact is
  `data/private_nextgqa/qta_repeatability_20261018_2210/full_process_replication.json`,
  SHA-256 `3482fb07106b9e18d0dc11a0f41492df22c2ad8f4c0f0b0b71ea18e1487fa9a1`.

## Verified results

| Fixed pair ordinal | Interval Jaccard | Selected-center W1 |
|---:|---:|---:|
| 1 | 1.0000 | 0.0000 |
| 2 | 1.0000 | 0.0000 |
| 3 | 1.0000 | 0.0000 |
| Mean | **1.0000** | **0.0000** |

An independent local readback recomputed the two means from the three stored
pair values, verified requested/attempted/completed counts, private file modes,
and before/after model snapshot hash equality. A separate auditor also
recomputed the first run's metrics and confirmed runner provenance and the
identifier-free summary schema. The second process' fixed-pair metrics,
aggregate metrics, source hashes, and model snapshot hash exactly matched the
first. Each run records all three batch identities as verified and all six
attention capture receipts as valid.

## Interpretation, limitations, and next action

For these three fixed inputs, QTA selected exactly the same temporal support on
the repeated forward. This suggests the partial overlap in the companion
16/32-frame diagnostic arose from changing input resolution/tokenization rather
than same-batch forward noise under this exact setup. It does not prove that
the selected supports are correct, explanation-grounded, predictive, or
faithful. The cohort is reused and tiny; there is no inferential CI or test.
The model snapshot hash checks local file-content stability, not which exact
byte ranges the loader consumed.

Direct faithfulness remains **0/3** reviewed claims, **0/15** independently
verified browser/player-to-decoder frame correspondences, **0/3** prediction-
link judgments, and **0/3** direct scores. There is no direct `DeltaF`, CI,
or hypothesis test. Human review remains blocked on authentic independent
role/key assignments, an out-of-band root pin, signed review-only custody,
witnessed unopened handoff, and a passing pre-open audit. Do not infer or
fabricate any of those human judgments from this computation.

The two process-level replications make same-batch random forward noise an
unlikely explanation for the preceding 16/32-frame discrepancy on these three
fixed examples, but they do not establish that frame count alone caused it:
resolution changes tokenization and may change the sampled temporal support.
Keep the 32-frame primary configuration frozen; do not tune from these
outcomes. Any further selector-sensitivity experiment needs a prespecified
sampling perturbation and must stay secondary to the human-reviewed claim
faithfulness study. Do not publish these protected-derived JSON files or this
log.
