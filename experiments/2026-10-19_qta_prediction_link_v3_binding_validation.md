# QTA prediction-link baseline v3 score-binding validation (2026-10-19)

## Purpose

Close a fail-open path found in the stage-2 prediction-link validator. It accepted legacy v1/v2 baseline manifests and v3 rows without score-binding fields, allowing the displayed P/Q values to be checked without requiring the score provenance record that the current phase-2 contract requires.

## Change and verification

- Stage-2 review validation now accepts only the canonical v3 baseline manifest and calls the same strict score-binding validator used by phase-2 planning.
- Each baseline row must include the complete score-binding object and digest. The shared validator checks exact keys, canonical bytes, digest, source-field hashes, answer-token binding shape, score position, option log probabilities, P/Q recomputation, and selector-position binding.
- Added a public `validate_baseline_pq_manifest` wrapper in `qta_phase2_plan_manifest.py` so review validation reuses the contract without reaching into a private helper.
- Replaced old/minimal test manifests with valid synthetic v3 fixtures. Added rejection cases for legacy schemas, missing binding, stale binding digest, malformed binding run digest, and inconsistent P/Q values.
- Focused renderer/validator/attestation suites: **26/26 passed**. Phase-freeze, stage-1 source-pin, composition, and baseline candidate suites: **114/114 passed**. Python compilation and scoped `git diff --check` passed.
- Tests use synthetic questions, rows, digests, and keys only. No protected media, model load, GPU forward, or human review was used.

## Interpretation and limits

The earlier validator accepted a structurally weaker baseline than the current v3 phase-2 contract. The change prevents that schema downgrade and enforces internal consistency between visible P/Q fields and the accompanying score-binding object. A v3 binding can still contain well-formed but untrusted run digests if a caller supplies a matching expected manifest hash; this function does not authenticate the caller's digest origin or independently prove the model forward produced the scores. Those claims require the signed freeze receipts and independently delivered root/cohort pins. A key-signed receipt remains evidence that a key signed bytes, not proof of reviewer identity, actual review, or semantic correctness.

The current tracked `docs/nextgqa_human_review_workspace_ja.html` is the exploratory quick-check launcher; it is not the stage-1/stage-2 integrated review application. This code-level repair does not complete the user's requested in-artifact audit flow. The exact production HTML integration and its browser E2E remain unverified.

## Faithfulness progress and next action

- NExT-GQA direct evidence remains **0/3 claims scored**; formal claim review **0/3**, clock/frame correspondence **0/15**, prediction-link judgment **0/3**. No direct `ΔF`, interval, or test was added.
- This change improves the integrity contract for future human review but is engineering evidence, not a faithfulness result.
- Next, bind the stage-1 receipt and stage-2 v3 baseline bytes into the actual integrated artifact and test success, tampering, stale-state clearing, and non-authorization in a browser. Independently authenticated reviewer assignments, root fingerprint/cohort pin, witnessed handoff, and actual human judgments remain external prerequisites before direct-score GPU work.

## Renderer contract alignment follow-up

An independent source audit then found that the renderer still accepted v1/v2
baseline manifests even though the validator above accepts only v3, and also
accepted minimal rows under a v3 header. The renderer now calls the shared
`validate_baseline_pq_manifest` before rendering and requires the complete
score-bound row schema. Its tests use three synthetic score-bound v3 rows,
reject legacy v1/v2, and reject missing or stale score-binding digests. Primary
focused verification passed **9/9 tests** plus `git diff --check`; independent
audit confirmed the same contract and that the positive fixture passes the
canonical validator. No model, protected media, or GPU was used.

This removes one avoidable schema mismatch but does not create the integrated
browser audit artifact. Direct faithfulness counts and external human/trust
gates remain unchanged.

The broader related suite initially exposed one additional test-fixture
incompatibility: the baseline-candidate test used 7-character placeholder
review tokens, while the canonical v3 validator requires 24-character tokens.
The fixture now uses format-valid opaque tokens; the candidate builder and
production contract were not relaxed. The combined renderer, prediction-link
validator, attestation, baseline-candidate, and phase-freeze suites then passed
**80/80 tests**. Independent audit confirmed the renderer/candidate fixture
contract. This is still software-integrity evidence only.
