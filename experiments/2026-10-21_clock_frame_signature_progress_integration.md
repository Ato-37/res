# Clock/frame audit progress integration — 2026-10-21

## Purpose

Connect the existing signed 15-anchor screenshot/decoder audit to the direct-faithfulness pre-load diagnostic, so a later real review package can contribute an untrusted structural count without treating signatures as verified identity or semantic truth.

## Implementation and checks

`evaluate_qta_direct_faithfulness_preload` now optionally accepts the exact raw inputs of `audit_nextgqa_clock_frame_evidence.audit_evidence` and reruns that audit. It reports a count only when the v3 audit reports all 15 anchors, all 15 match decisions, no ambiguity/mismatch, byte checks, reviewer/capture signatures, and the rostered frame-review key binding. The count is explicitly named `signature_bound_frame_pair_decisions`; the separate `validated_clock_frame_anchors` stays zero, `clock_frame_correspondence` stays blocked, and external root origin, real-world identity, and authorization remain false. A caller-provided summary is rejected because it is not an accepted audit argument set.

The pre-load gate and its newly connected local dependencies were added to the pre-review inventory's code closure. The focused gate and inventory tests passed **35/35**. The broader v3 builder/auditor, pre-load gate, and inventory suite passed **86/86**. One new integration test passes the real synthetic v3 evidence bundle through the actual clock/frame audit and into the preload diagnostic; only the independent source/review bridge is stubbed with a structurally valid, explicitly non-authorizing candidate. Separate negative cases verify that legacy, incomplete, ambiguous, mismatched, or caller-forged audit summaries contribute zero decisions. A fresh local candidate inventory contains **100 roles, zero missing slots**, and remains candidate-only. After the additional negative-case tests changed the registered source bytes, the inventory and CPU sampler were refreshed again; the current 100-role candidate and sampler receipt are the ones described here. The CPU sampler ran on the fixed actual-video cohort: **3/3 videos**, 32 sampled frames each, decoded frame counts 510/1,140/1,258 and maximum sampling-grid deviations 16.145/16.145/16.355 ms. The actual-QA hash join again returned **3/3**, with all scoring, model-load, and authorization fields denied.

## Interpretation and limits

This is handoff/readiness engineering, not completed human review. The signed-decision count remains zero until a valid v3 evidence package is supplied; even a later 15/15 signature-bound count would not authenticate the trust root's out-of-band origin, prove reviewer identity, or independently compare imagery. The current result still has **0/3 Stage-1 claim reviews, 0/15 signed frame-pair decisions, 0/3 Stage-2 prediction-link reviews, and 0/3 direct scores**. No effect estimate, interval, hypothesis test, GPU inference, or faithfulness result was produced. Dataset/model rights remain unverified and protected artifacts remain local.

## Next action

Obtain the authentic assigned reviewer/release keys and independently pinned root fingerprint, then complete Stage-1 review and capture/frame artifacts. Rerun this integrated diagnostic from the exact raw files. If the trust and semantic gates pass, finish the production atomic baseline-only runner before any model loading. Continue to report structural review counts separately from direct model-faithfulness scores.
