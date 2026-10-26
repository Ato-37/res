# Test-split internal visual-key bias × frame-mean interaction (exploratory)

## Question and frozen estimand

Does the fixed visual-key attention-logit bias increase the dependence of a
saved explanation claim on spatial video structure, as measured by its
conditional likelihood on original video versus a per-frame RGB-mean control?
For each of the three already frozen test-split rows, compute

`[(LL(original,bias) - LL(frame-mean,bias)) - (LL(original,base) - LL(frame-mean,base))]`.

The intervention is fixed at layers 5–16, requested `+ln(2)` on all
processor-expanded visual keys, with claim-token prediction rows as queries.
The baseline conditions are the exact previously completed test-split
frame-mean receipts. No rows, settings, or outcomes are selected or changed.

This is a post-result exploratory split extension: validation results motivated
the question, model exposure history is incomplete, and the claims are
model-generated. It is not a confirmatory or population estimate. A positive
interaction would only show an increased likelihood contrast against a severe
spatially-uniform control; it would not show claim truth, semantic support,
frame-localized evidence use, or improved faithfulness.

## Expected execution

Three rows, two new teacher-forced forwards each (6 total). First smoke one row
(2 forwards); if prompt, target tokens, frame/timestamp geometry, processor
pixel hashes, nonvisual inputs, baseline receipt joins, intervention hooks, and
finite scores all pass, run the two remaining locked rows (4 forwards). Use
the pinned Qwen2.5-VL-7B-Instruct checkpoint, 4-bit weights/BF16 compute,
eager attention, CUDA resource guards, offline mode, unique owner-only receipts,
and local-only handling. Preserve failures and never replace rows.

## Outcome and limits

Record every row, all forward counts/failures, finite-cohort mean/median/range,
and source/input hashes. Do not calculate a p-value or population interval for
this selected n=3 set. Direct semantic faithfulness and actual human review
remain unmeasured; rights remain unverified. These receipts and all protected
inputs stay local.

## Executed result (2026-10-21)

The smoke passed (2/2 forwards), then both remaining locked rows passed
(4/4). Total: **3/3 rows and 6/6 new forwards**, no failed row or replacement.
Execution used the RTX 5070 Ti and pinned Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
eager attention, and layers 5–16 `+ln(2)` visual-key bias. Per-row peak
allocated VRAM was 7.29, 7.39, and 7.27 GiB; measured runner times were
24.2, 21.8, and 24.6 seconds.

| Frozen row ordinal | Baseline original−frame-mean | Biased original−frame-mean | Difference-in-differences |
|---:|---:|---:|---:|
| 0 | +0.480102 | +0.542069 | +0.061967 |
| 1 | +2.214195 | +2.248450 | +0.034255 |
| 2 | +0.466227 | +0.487827 | +0.021600 |
| **Finite-cohort mean** |  |  | **+0.039274 nat/token** |

Median interaction is **+0.034255**, range **[+0.021600,+0.061967]**;
3/3 rows are positive. These are finite-cohort descriptive results only: no
CI, p-value, or population inference was specified or computed. Baseline
original-minus-mean scores are reused from the prior fixed test-split run,
and the cohort was outcome-exposed to those earlier content-sensitivity
measurements. This is not an independent or confirmatory replication.

The result is a small, consistent increase in saved-claim likelihood
sensitivity to intact spatial video structure under one global key-bias
intervention. It is a useful **internal-intervention mechanism signal** for a
thesis, but it does not establish improved explanation faithfulness: the
frame-mean control is out of distribution, claims are model-generated, the
bias is not claim-specific, and no human support judgments were performed.
The separate validation global-bias × gray-video interaction was near zero;
the different control and cohort mean these outcomes must not be pooled. Direct
claim-to-frame review and direct faithfulness scores remain **0/25 formal**
and **0/3 on this cohort**. Rights remain unverified and all receipts are
owner-only local files.

The result analyzer independently verified each receipt's mode, baseline
receipt digest, 12/12 layer hooks per biased branch, score semantics, forward
counts and interaction arithmetic. Unit tests passed **5/5**; `py_compile`,
CLI help, lock/packet/baseline joins, and scoped `git diff --check` passed.
Audit note: the frozen lock has a null `candidate_pair_filter_sha256` field.
The runner binds the populated selected-filter hash, while the post-run analyzer
and independent audit verified each packet's candidate-pair digest is actually
in that frozen filter. This did not create a row mismatch; future runners should
also enforce membership before loading the model.
Reproduce only the derived summary (without printing raw row content) with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/analyze_test_split_internal_bias_frame_mean_interaction.py
```

Receipt SHA-256 values in ordinal order:

- `test_split_internal_bias_frame_mean_interaction_smoke_20261021_01.json`: `2706a6c3229818921d9b0c03b5dc15a985e9d5bd0ef5c8c6d23e70c867f9af1b`
- `test_split_internal_bias_frame_mean_interaction_row0_20261021_01.json`: `c0e5a4a3103006f13b3eeec9be70af92f1a7f02104f6ea570ba68373593144d9`
- `test_split_internal_bias_frame_mean_interaction_row1_20261021_01.json`: `e7975aaf54c758256529d78165dab8230c2722c56319afd01bcd6b40a34b4d78`

**Next decision:** retain this as a thesis-relevant proxy result, not the
faithfulness endpoint. Do not tune the same cohort. The decisive next study is
a result-unseen cohort with claim-specific support intervals independently
reviewed, then compare baseline, support-key steering and equal-budget sham
on a fixed-prediction support-ablation margin. Until then the thesis can
honestly report a positive internal sensitivity signal alongside prior null
and sham-equivalent QA outcomes, with direct semantic faithfulness unresolved.

## Informal review handoff (2026-10-21)

Created a Japanese, three-item, no-file-picker review page for exactly these
frozen test-split question/claim/video pairs. The flow starts with a nickname,
then asks the three existing informal questions (clarity, answer fit, and
visible-evidence sufficiency), permits consecutive reviewers with one combined
JSON export, and keeps the original English text behind a disclosure. The
artifact does not show gold answers or infer formal review completion.

- Artifact: `data/private_nextgqa/quick_review_test_split_internal_bias_20261021_01.html`
- Embedded artifact fingerprint: `0a4dab8461d47185e84771cf96b3d9c8b5e87d1d3d996536146fed676ffa6bc5`
- Artifact file SHA-256: `9f92b8c93bd877932a3f74efdb0990bd047f3c196ad68a62ecf0bc2cc48df12a`
- Mode: artifact `0600`, reviewer package directory `0700`, package files `0600`
- Audit: 3/3 rows matched packet question/claim/item binding; all 3 embedded MP4
  byte hashes matched package media. Fingerprint recomputation and nickname →
  question → next-reviewer → combined-export controls passed static audit.
- Existing builder/inspector tests passed 33/33. The separate Windows Chrome E2E test
  did **not** pass: its exact-profile cleanup assertion failed; therefore no
  runtime-browser pass is claimed for this newly generated artifact. No
  Linux Chromium/Playwright installation is available in this environment.

No new human responses exist for this artifact. Its labels remain informal
impressions, not claim-to-frame support annotations or direct faithfulness
scores. Open locally from the repository root with:

```bash
xdg-open ./data/private_nextgqa/quick_review_test_split_internal_bias_20261021_01.html
```

The two downloaded October 19 JSON files do not bind to this artifact and are
not accepted as responses to it. All page, video, translation, package and
future response data remain local; rights are unverified.
