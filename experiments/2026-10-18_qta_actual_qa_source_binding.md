# Actual QTA QA-source binding smoke (2026-10-18)

## Purpose

Check whether the fixed three-row local NExT-GQA claim candidate can be joined
to its exact test-split question/choice/answer rows without exposing those
values, opening video content, loading a model, or authorizing scoring. This is
source plumbing only; it is not a faithfulness experiment.

## Method and provenance

The first real-data invocation used an early, uncommitted version of
`scripts/run_qta_actual_qa_source_binding_smoke.py` against the local
`NExT-GQA` test CSV and the private three-row claim, worksheet, lineage, and
diagnostic-target artifacts. Its protocol and treatment-selection bytes
matched the tracked Git blobs at commit
`08b8a1ef2418fadf8aa18166962b0cd439c3e8af`. The treatment-selection input is
a prose selection record, not an executable model configuration; the binder
checks its bytes but does not interpret it.

The runner read bytes in memory, checked private JSON ownership/mode and input
size caps, constructed the fixed ordered selection from the candidate
artifacts, and discarded the binder's private bundle without writing it. The
CSV had its existing mode `0755` and the tracked protocol draft had mode
`0600`; the runner permits those exact modes for these two non-private source
roles and still checks owner, regular-file identity, symlink/path containment,
and exact pinned Git bytes where applicable. No file permissions or source
data were changed. The first CLI attempt failed closed on the protocol's
`0600` mode (0/3); after adding that exact source mode to the allowlist, the
same command succeeded. This initial failure is recorded rather than hidden.

## Observed result

- Requested denominator: **3**; effective source bindings: **3/3**.
- Status: `source_binding_passed_unvalidated`; no row-level failure.
- Protocol and treatment-selection-document Git-blob matches: **true/true**.
- `model_load_permitted=false`, `execution_authorized=false`,
  `scoring_authorized=false`, forwards **0**, score calls **0**.
- `faithfulness_scored=false`; rights status remains
  `unverified_local_only`.
- The private bundle, question text, choices, answers, gold indices, row IDs,
  and private artifact hashes were not written or printed.

The result establishes that the supplied CSV bytes and supplied candidate
claim/selection/lineage/target bytes agree under the existing binder for the
fixed three rows. The cohort artifacts themselves are not independently pinned
to an authenticated trust root, the supplied lineage does not prove its
generation history, and the logged treatment-selection prose is not a runtime
configuration. The result does not establish claim truth, human review,
reviewer identity, rights, media-byte correspondence to the lineage hashes,
player-clock/decoder-PTS correspondence, baseline `p/q`, model behavior,
internal-intervention effects, or readiness to execute.

The runner itself was uncommitted and did not yet check a candidate inventory
at the time of this first actual-data smoke. Therefore the 3/3 result is only a
provisional observation about the supplied bytes, not an independently frozen
cohort or source-authenticated reproducibility claim. The current wrapper has
since been hardened to check its own code and dependencies against current
HEAD, enforce a candidate-inventory pin against each input's size and digest,
and reject a coherent cohort replacement against the unchanged inventory. It
was then rerun on the same local inputs at `2026-10-18 00:36:17 UTC` from code
commit `94fe91be34235c7add5e6bcb2ed336db13c96f67`. The aggregate-only report
returned `source_binding_passed_unvalidated`, **3/3** rows,
`inventory_content_match=true`, `protocol_blob_match=true`, and
`config_blob_match=true`; model/scoring authorization remained false and
forwards/scores remained zero. The run used a 77-artifact candidate inventory
with zero missing slots.

The runner verified all ten required inventory role byte sizes/digests against
the supplied inputs and the three pinned code files. Its expected inventory
digest and Git HEAD commit were caller-supplied local values; the report labels
both pins unverified and explicitly states `inventory_is_trust_root=false`.
Thus this rerun is reproducible relative to the supplied local candidate
inventory, but the inventory and its expected digest are not independently
authenticated. Rebuilding a different internally consistent candidate and
supplying its new digest could bind a replacement cohort. This does not close
the independent freeze/authority gate.

## Related hardening and workflow correction

The source contract now rejects oversized byte inputs before hashing them. Its
regression test patches the hash function to fail if called, then verifies that
an over-limit input is rejected at 0/3 with no private bundle and no
authorization. Focused source-contract tests pass **10/10**. The hardened CLI
synthetic tests pass **10/10**, including per-role source-byte tampering and
cohort substitution; the inventory-builder tests pass **20/20**. The combined
focused source/diagnostic/inventory/CLI suites pass **47/47**.

Independent code review confirmed the human-review runbook's earlier word
“optional” contradicted `qta_stage1_bundle_audit.py`: the current aggregate
requires a valid detached reviewer-key attestation to pass stage 1. The runbook
now distinguishes saving a local draft (attestation not required) from passing
the stage-1 aggregate (attestation required), while retaining the limits on
identity, actual viewing, semantic validity, and authorization. The independent
review also confirmed the size-cap-before-hash repair and its test.

A fresh private pre-review candidate inventory for the repaired page and
matching package contains **77 artifacts, 0 missing required slots** and
registers the source-binding CLI. It remains candidate-only, with review
incomplete, no joint freeze, score readiness false, and execution authorization
not granted. The expected inventory digest remains a caller-supplied local
pin; the inventory is explicitly not a trust root.

## Faithfulness progress and next action

- Direct QTA faithfulness evidence: **0/3 reviewed; 0/3 baseline prediction-link
  judgments; 0/15 independently verified player/decoder anchors; 0/3 scored**.
- No `DeltaF`, effect estimate, interval, hypothesis test, or multiplicity
  adjustment was produced. No GPU or model forward was run.
- The hardened aggregate-only CLI did run once on actual local source bytes
  after code commit, with 3/3 bindings and matching candidate-inventory roles.
  This improves reproducibility relative to the caller-supplied local
  inventory; it cannot authenticate that inventory or substitute for human
  review and clock evidence.
- This cycle produced engineering/source-plumbing diagnostics only. The next
  evidence-bearing gate remains a real independent stage-1 human review with
  complete occurrence intervals, followed by all 15 clock/frame
  correspondences and calibration. Only when the full pre-load gate passes
  can the protocol's three baseline-only GPU forwards be considered; the
  post-baseline prediction-link gate still precedes treatment forwards.
- Overall project status remains bounded by the separately reported prior
  head-confirmation result; this QTA candidate adds no direct faithfulness
  evidence and does not reopen closed head/layer/alpha search.

Rights status remains unverified, and all protected material stays local.
