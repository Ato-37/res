# QTA two-phase gate ordering correction (2026-10-19)

## Purpose

Correct the pre-load structural diagnostic so it does not treat prediction-link
review as a prerequisite to baseline scoring. The protocol requires the frozen
baseline predictions and `p/q` manifest to exist before Stage-2 reviewers can
judge whether the explanation-linked prediction is supported.

## Implementation and checks

- Changed `scripts/qta_direct_faithfulness_preload_gate.py` to schema version 2
  and explicitly label its phase `pre_baseline_candidate`.
- Replaced the misleading `frozen_baseline_prediction_link: blocked` field with
  `baseline_prediction_manifest: not_applicable_pre_baseline` and
  `prediction_link_review: not_yet_applicable_baseline_required`. This diagnostic
  still has no model, forward, scoring, or authorization path; it cannot grant
  the three baseline forwards.
- Added focused assertions for the phase-aware statuses and retained all
  denial-state checks.
- CPU command from `scripts/`:

  ```bash
  ../.venv/bin/python -m unittest -v \
    test_qta_direct_faithfulness_preload_gate \
    test_qta_actual_candidate_input_bridge
  ```

  Result: **27/27 tests passed**. Python compilation and scoped `git diff
  --check` also passed. All fixtures are synthetic; these tests do not validate
  human identity, semantic review, clock-to-frame correspondence, GPU scoring,
  or execution authorization.

## Runtime and research status

The normal sandbox cannot access NVML/CUDA. An individually elevated,
read-only runtime check confirmed `.venv` PyTorch 2.11.0+cu128 sees one
NVIDIA GeForce RTX 5070 Ti. No model or dataset was loaded and no GPU workload
was started: the direct-scoring gates remain unmet, so a real-video score run is
not justified yet.

Direct faithfulness remains **0/3 reviewed claims**, **0/15 verified
player-clock/decoder-frame anchors**, **0/3 prediction-link judgments**, and
**0/3 direct scores**. No direct `ΔF`, confidence interval, or test exists.
The head confirmation remains closed at 79/79, −0.0756 nat, 95% t CI
`[−0.1578, +0.0065]`, below the preregistered +0.25 nat threshold. Existing
real-video QTA and generated-text runs are closed exploratory proxies, not
direct evidence.

## Interpretation and next action

The ordering bug in the status report is fixed; the production path is still
incomplete. The next real experiment requires independently trusted Stage-1
claim/release materials, all 15 reviewed clock/frame correspondences, and a
verified cohort/runtime/batch freeze. If those pass, the first GPU transaction
is exactly three untreated baseline forwards. Only after freezing their `p/q`
manifest can Stage-2 prediction-link review happen; only a separately validated
post-baseline authorization can permit the 24 remaining branch forwards.
Current direct score status remains `no_go` until those artifacts exist.
Rights status is unverified; protected data and derived artifacts remain local.
