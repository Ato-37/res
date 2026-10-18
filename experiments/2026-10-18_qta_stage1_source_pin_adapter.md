# QTA stage-1 source pin adapter and review inventory (2026-10-18)

## Purpose

Close the structural handoff gap between the existing stage-1 review-bundle audit,
actual NExT-GQA QA-source binding, and the baseline p/q candidate contract. This is
protocol plumbing for a fixed three-record human-reviewed cohort. It does not
constitute a human review, model run, or faithfulness efficacy result.

## Implementation

Added `scripts/qta_stage1_source_pin_adapter.py` and its synthetic regression tests.
The adapter reruns the stage-1 audit and QA-source binder on original byte inputs,
requires all three rows and all mandatory audit components, verifies ordered token,
question, five-option, claim, video, support-decision, and evidence-interval bindings,
then derives an exact per-token source map accepted by
`qta_baseline_pq_candidate.build_baseline_pq_candidate`. The result is explicitly
classified `private_local_only_structural_source_pins`; all model/scoring/execution
authorization flags remain denied. It does not authenticate identities, prove that
any reviewer watched the video, establish semantic correctness, or calibrate
browser timestamps to decoder frames.

## Checks and observed results

- Focused adapter suite: **18/18 passed** after the final combined baseline-source-map
  addition.
- Repository suite: **1,682/1,682 passed**. Existing negative-path tests emitted
  their expected failure diagnostics; the suite completed with `OK`.
- A local, private pre-review byte inventory for the current package/media/HTML
  inputs reported **77 roles, 0 missing**, and `candidate_only`. The inventory and
  its hashes stay under `data/private_nextgqa/` and are not published.
- GPU availability was checked through the authorized per-command WSL path:
  RTX 5070 Ti, CUDA-enabled `.venv` (`torch 2.11.0+cu128`, CUDA 12.8), approximately
  15.1 GiB VRAM free at inspection. No model job ran. This branch has no authorized
  or scientifically valid score input until the human-review and trust gates pass.
- GitHub state was not re-fetched in this cycle because the normal sandbox could not
  resolve GitHub; the previous cycle had already verified commit
  `b95a9a267bd740280b8598a1929b8f279c66f08d` on
  `origin/internal-ablation-wiring-20261018`. This entry and adapter are not yet
  committed.

## Faithfulness progress and interpretation

Direct human/video faithfulness progress remains **0/3 reviewed claims**, **0/15
browser/player-to-decoder frame correspondences**, **0/3 prediction-link judgments**,
and **0/3 claim-linked QTA scores**. There is no empirical `ΔF`, confidence interval,
or inferential test. The inventory and adapter establish only candidate byte
completeness and structural source binding; they provide no evidence that model
predictions depend on the explanation-indicated video evidence.

The GPU is technically available, but a baseline-only run now would lack the trusted
review identity/root and actual review evidence needed to bind scores to valid
video-supported claims. Running it would create ungrounded scores, so the next useful
step is to complete the external review/trust ceremony and then repeat the exact
release-order audit. Rights status remains unverified; all protected inputs and
derived private bindings remain local.

## Limitations and next action

An independent auditor confirmed the adapter's non-authorizing behavior, exact
interval digest derivation, and the just-added baseline input source-map delta. The
current local inventory is not the production freeze receipt; candidate-only status
does not authorize opening/scoring. A valid human review still requires a trusted
root/identity mapping, distinct role assignments and signatures, freeze-owner receipt,
release witness, three semantic claim/frame reviews, and later prediction-link
judgments. Continue preparing/rechecking the local handoff; do not run faithfulness
inference until those gates pass. After they pass, run the bounded three-row baseline
full pass, freeze its output, conduct the score-free human review, and only then
consider treatment comparisons under the predeclared QTA-Bias `ΔF`.

## Provenance hashes

- Adapter source SHA-256: `740fb9fb77bc5ea4d48bc1f02cdeced84ec03ce83e15d4a1b845b7a723e055ad`
- Adapter test SHA-256: `bfe53fa32e7b3a70e10a34656f732996321c24ea54e1df512e356682c94d89aa`
- Scoped commit: `e45434892076ac7184360d38ead4c77bd3fe70b7`
