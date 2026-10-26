# PDCA checkpoint: attention hook hardening and review inventory preparation (2026-10-18)

## Purpose

Close two implementation robustness gaps found in the attention-bias helper, then prepare the current offline human-review package for later independent review without treating packet integrity as review completion.

## Changes and checks

- `scripts/qwen_visual_attention_boost.py` now copies selector positions and layer indices before registering closures. Mutating a caller-owned list after hook registration cannot change the treatment.
- The attention-mask helper rejects NaN and positive infinity anywhere in the additive mask while preserving valid causal `-inf` entries. Selected biased values must still be finite.
- Focused helper tests passed **5/5**. The hook regression test mutates the original list after registration; mask tests cover out-of-selection NaN/+inf and valid causal -inf.
- Full repository unit suite passed **1,457/1,457** in 29.45 s. `py_compile` and `git diff --check` passed.
- Independent audit of the two helper changes passed; it independently confirmed both fixes and found no residual issue within scope.
- Rebuilt a local-only pre-review byte inventory against the repaired review package and its byte-matching staged media. It binds **65 artifacts**, has **0 missing required slots**, and is stored under mode `0700` with the inventory file at `0600`. A prior attempt against a nonmatching staging set remains untouched and is not used.
- The inventory itself states `candidate_only`, `review_completed=false`, `score_ready=false`, and `execution_authorization=not_granted`. It is a byte-integrity receipt only; it does not establish reviewer identity, human review, browser-to-decoder timing, or authorization.

## Faithfulness progress

**Direct evaluation: not performed.** No generated explanation was reviewed or scored in this checkpoint; no faithfulness estimand, effect, confidence interval, or hypothesis test was computed. The current project status remains **0/3 independently reviewed, 0/3 scored, 0/15 verified browser-to-decoder correspondences**, pending the actual human and interactive capture steps. The earlier three-video real-Qwen attention-bias smoke is a wiring diagnostic only and does not establish explanation-evidence dependence or treatment benefit.

Engineering support is limited to safer hook semantics, passing code tests, and a complete candidate byte inventory. Rights status remains unverified; all review material and derived inventory remain local.

## Provenance

- Attention helper SHA-256: `f162b6e30af7f77b552db5c8deb3f1d8c7db8049f6f346d077cf55c804fe7b41`
- Focused test SHA-256: `909f0da72eb107dbd1ea2927fb8a1f22d6664b9facefe539aa376dc4e687ad10`
- Inventory builder SHA-256: `f004fe6ac0afaa6f6eadbc7fbf90cd3cf8b0783a3cdc940f4d9391093a0c1544`
- Private candidate inventory: `data/private_nextgqa/pre_review_inventory_candidate_20261018_1936/pre_review_candidate_byte_inventory.json` (protected-derived artifact; do not publish).

## Interpretation and next action

The helper now fails closed on malformed global masks and freezes its selector at hook registration. This improves treatment reproducibility but does not resolve the treatment estimand. The fresh candidate inventory removes a stale-package integrity issue, but remains non-authorizing. Continue by completing an independent audit of fixed baseline prediction/foil token positions and the proposed internal intervention branches. Do not run real faithfulness scoring until independent claim review, browser-to-decoder correspondence, frozen treatment/scorer, and exact cohort bindings are all verified. No GPU run was warranted for these two changes: CPU tests exercise both altered behaviors, and another generic GPU wiring run would not add faithfulness evidence.

## Follow-up inventory audit and QTA annotation-proxy result (2026-10-18)

An independent metadata-only audit clarified that the 65-artifact path listed
above, `pre_review_inventory_candidate_20261018_1936`, has no source-provenance
sidecar and does not bind review source paths; it is not a release-ready
candidate despite its earlier “fresh candidate” wording. A separate candidate
with complete source-path provenance is
`data/private_nextgqa/pre_review_inventory_candidate_20261018_1118/`:
77 artifacts, zero missing, directory mode `0700`, inventory and provenance
files mode `0600`, inventory SHA-256
`1487935e5633802ea32a3d7ab0ad0196d55675aa607e0ea13fc2c39338505963`, and
provenance SHA-256
`181d91a62911ceef395b07b777a6d68e4a04d298b86e5947d0f4c3312616c4d2`.
It binds package `reviewer_package_v6_20261018_0243`, standalone HTML
`claim_review_v6_20261018_0243.html`, and staged media
`review_media_v6_20261017_0635` under the private-data root. Independent
checks confirmed the inventory self-hash, these provenance bindings, and
current hashes for the runbook, builder, integrity verifier, and release-order
auditor. This remains `candidate_only`; it has no freeze, receipt, human
review, or execution authorization. Rebuild a unique inventory again directly
before any future freeze.

The independent audit found the actual human-dependent blockers unchanged:
no real-person-to-key assignments, no independently pinned trust-root
fingerprint, no freeze-owner-signed review-only receipt, and no independent
witnessed unopened handoff. No protected review content was opened in this
audit. The independent QTA annotation-proxy full run separately completed
30/30 rows and 180/180 forwards; its finite-cohort mean ΔF was `+0.00833` nat
(SD `0.14284`, range `−0.25000` to `+0.25000`, 17 positive and 13 negative).
This reused-cohort annotation-dependence proxy is not direct claim-faithfulness
evidence, has no population inference, and does not change the primary status:
0/3 claim reviews, 0/3 prediction-link judgments, 0/15 verified browser-clock
to decoder-frame correspondences, and 0/3 direct scores.

Next: retain the `1118` inventory only as a candidate, wait for the actual
participant/key assignments and out-of-band trust-root pin, and continue
independent scorer-contract work that does not require opening the review
package. Do not use the 65-slot `1936` inventory for a freeze or release.
