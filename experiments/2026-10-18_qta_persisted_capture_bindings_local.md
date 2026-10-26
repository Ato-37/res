# QTA persisted capture-binding audit diagnostic (local, 2026-10-18)

## Purpose

Make the independent expected bindings used by the QTA attention-call receipt
validator available for post-run verification. The preceding corrected run
validated against capture-time values held in memory, but did not persist that
separate comparator. This is an instrumentation and reproducibility check,
not a faithfulness experiment.

## Change and frozen run

`scripts/smoke_qta_repeatability.py` now writes the separately constructed
`expected_capture_call_bindings` into its private summary. Each row contains an
ordinal-derived record token, the runtime descriptor digest, input length,
question-token positions, and the captured attention-payload digest. The
payload itself and all video/question/answer identifiers remain excluded.
The exact serialized call-receipt index is then independently revalidated
against this list.

- Dataset/cohort: same first three deterministic eligible NExT-GQA test
  candidates, frozen before decoding; no replacement.
- Input/model: 32 decoded frames; pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit NF4, BF16 compute and
  activations; RTX 5070 Ti; PyTorch `2.11.0+cu128`, CUDA `12.8`, Transformers
  `5.18.0`; seed `20261018`.
- GPU cost: 3 videos, 6 forwards; latest run completed in 92.64 seconds.
  Free VRAM was 14.66 GiB before model load and 8.68 GiB after load.
- Output: `data/private_nextgqa/qta_repeatability_receipt_20261018_2235/full_r3_persisted_bindings.json`,
  mode `0600` in a mode `0700` directory, SHA-256
  `0171e85b76de313ba99e45524a4fe7dc41fcbec24531c0665e0cd481d0dff42a`.
- Runner SHA-256: `a440ac0ad317433efcd2163b73af24301590c7efa72256c2fe70ff9965cbd584`.
- Model snapshot SHA-256 before load and after forwards:
  `1ef510079cd42b8d5d3273b9da96ce3881fdd1cf2220cc315f9fa39a9f41aa93`.
- Runtime descriptor SHA-256:
  `316e5a32cbd7bfea80b7e6d7df020e58f6ff00a113edaf883ca3c32e7a8cda6c`.

## Verified observations

| Measure | Result |
|---|---:|
| Requested/effective videos | 3/3 |
| Requested/attempted/completed forwards | 6/6/6 |
| Failed/not-attempted forwards | 0/0 |
| Persisted expected bindings | 6 |
| Call receipts independently revalidated | 6/6 |
| Mean/min interval Jaccard | 1.0000/1.0000 |
| Mean/max selected-center Wasserstein-1 | 0.0000/0.0000 |

The standalone readback reconstructed canonical index bytes from the saved
artifact and passed the saved expected bindings to the receipt validator. Its
index digest was `b55cd5c6c458509e49c6c160920debe82b5fdf865e7cca28e275e28e2145b2e9`.
Metrics were recomputed from the three fixed-pair rows. The result remains
explicitly non-authorizing: source/capture authentication and scoring are
false, and execution authorization is `not_granted`. A one-video, two-forward
smoke of this persisted-output change also passed and was read back in the
same manner (2/2 receipts; output SHA-256
`e753a2da5925ec017f3a82253f6f8dd329a5990480d702cc4952820d2bc196cc`).

An independent source/result audit verified the artifact, all 10 repository
source hashes, all five installed Transformers/Qwen utility source hashes,
checkpoint/runtime hashes, six exact persisted-to-receipt bindings, counts,
metrics, privacy fields, and private file mode. It reran receipt validation
against the persisted expected list and reproduced the reported index digest.
The expected rows and receipts still come from the same unsigned process, so
this is not cryptographic execution attestation. Focused CPU suites after the
code change passed 55/55, and full repository discovery passed 2,061/2,061 in
88.51 seconds (expected synthetic failure diagnostics were printed by tests).
`py_compile` and `git diff --check` passed.

## Interpretation and limits

On these three fixed inputs, the same in-memory processor batch produced the
same selected support across two untreated forwards. The persisted expected
bindings now let a later reader validate the receipt join without deriving
expectations from the receipt objects. This establishes engineering
repeatability and a more auditable serialization path only. It does not show
that attention follows the explanation, that selected frames support a
claim, that an intervention changes prediction dependence, or that the
method generalizes. The fixed sample is too small and selected for any
population inference; no confidence interval or hypothesis test is relevant
to this diagnostic.

## Faithfulness progress and next action

**Direct evaluation: not performed.** Current direct status remains **0/3
claim-support reviews, 0/15 browser-clock-to-decoder frame correspondences,
0/3 prediction-link judgments, and 0/3 direct faithfulness scores**. No direct
`DeltaF`, confidence interval, or hypothesis test exists. This GPU run does
not increase those denominators.

Human review still needs authentic person-to-key assignments for seven
separate roles, an independently pinned trust-root fingerprint, signed
review-only custody, a witnessed unopened handoff, and passing pre-open
audits. Continue independent CPU/protocol and scorer-contract checks while
those external human inputs are unavailable. Once the human gates pass, use
the frozen reviewed claim/support rows for the prespecified prediction-link
and internal-intervention contrast; do not treat this selector diagnostic as
support for changing the estimand or tuning the selector.

Rights status remains unverified. The result and this derived log stay local
and must not be published. No commit/push is made because the worktree
contains unrelated pre-existing user changes and this result is protected.
