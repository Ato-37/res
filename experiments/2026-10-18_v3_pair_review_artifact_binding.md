# Schema-v3 pair-review artifact binding (2026-10-18)

## Purpose

Close the reproducibility gap found in the independent audit of the frame-review
handoff. Schema v2 signed the frame decisions but omitted the exact pair package,
custodian identity map, unsigned review export, and rendered HTML bytes. A later
auditor therefore could not reproduce the signed decision's join to the exact
images shown in the viewer.

## Changes

- Added `audit_export_bytes()` as the shared strict validator for the existing
  private export/package/viewer checks. It validates strict JSON schemas,
  complete 15-pair decisions, the package nonce and export binding, the HTML
  template binding, and embedded PNG hashes. HTML pair JSON now rejects duplicate
  object keys.
- The evidence builder emits schema v3 under a new signature domain. It signs
  exact SHA-256 hashes for the pair package, identity map, review export, and
  rendered reviewer HTML; it also signs the package nonce and each randomized
  pair ID. It validates the exact viewer and export bytes before signing.
- The downstream auditor requires those four exact artifacts for v3, verifies
  their signed hashes, validates one-to-one identity-map coverage, and rejoins
  each signed pair ID through the package, review decision, viewer PNG bytes,
  screenshot crop RGB digest, and decoder frame RGB digest.
- v1 and v2 are retained only as explicitly labeled legacy candidates. They no
  longer receive the current successful candidate status or CLI success code.
- Updated the human-review runbook with v3 behavior and its limits.

## Verification

All verification so far used synthetic fixtures. The three focused suites passed:

- blinded review helper: 11/11;
- clock evidence auditor: 18/18;
- evidence builder and end-to-end integration: 50/50.

The integration test passed the builder's signed v3 output through the real
downstream auditor and verified that appending bytes to each of the four signed
artifacts is rejected. A separately re-signed but malformed identity map is
also rejected by schema validation. An independent auditor reviewed the code and found no
authorization bypass; it confirmed that missing v3 artifacts fail closed and
that all authorization fields remain denied.

The repository-wide suite passed **2063/2063** tests in **86.937 seconds**.
`py_compile` passed for the modified Python files. The human-review inventory's
source closure was also updated to include the new blinded-pair validator; its
fresh hash-only inventory is a separate local-only checkpoint. No protected
review export, screenshot, crop, video, gold answer, or seed interval was
opened. No GPU inference or human review occurred in this work item.

The pre-review inventory source closure now registers the v3 pair-review
validator. Its fresh local-only hash inventory contains 82 roles and 0 missing
slots, with all opening/freeze/scoring flags false and permissions 0700/0600.
The private inventory bytes and digest remain local and are not part of this
experiment log.

Key source hashes at this checkpoint:

| File | SHA-256 |
| --- | --- |
| `scripts/build_blinded_frame_pair_review.py` | `0a7a5699e0ea37079bc72278d87952fc79f5dfb801d88ca2c35466484266efce` |
| `scripts/test_build_blinded_frame_pair_review.py` | `251a36d176ca84b0da68a0d892b2e291e235d68d9ee82634dd7cf9b5acc9e66f` |
| `scripts/build_nextgqa_clock_evidence_manifest.py` | `8e2102402af92eb7cb62893304f3efaefd05a74f33db6a18d719d949f28b78ff` |
| `scripts/test_build_nextgqa_clock_evidence_manifest.py` | `35d8dcf9c0fcab55bfc2b9e39dc34919b84a2548ba48b4daf020f3db153f4bbf` |
| `scripts/audit_nextgqa_clock_frame_evidence.py` | `68f1399ef9559ae298839b38f0e3911fc6c646c176dcd70d276a6d3290669f25` |
| `docs/nextgqa_v6_human_review_runbook.md` | `3cdf98b6a175a721888981a3595fe521984613c96ba0cc3d375e01e05f62b16b` |
| `scripts/build_nextgqa_pre_review_byte_inventory.py` | `c1fd99123651a8ab28a1f3123a2c04441fbfc9160ec4ff8c336d2de19ab65911` |
| `scripts/test_build_nextgqa_pre_review_byte_inventory.py` | `6e5606303b22a9b8fd581f7009eb55c772651e0e38c5b4f01ff46f58f5d7da8e` |

## Interpretation and limits

This change makes the handoff bytes auditable and detects mismatched or altered
pair artifacts. A valid signature only authenticates adoption of the signed
bytes by the selected key. It does not establish the reviewer's identity, prove
that a person actually inspected the images, establish crop purity, prove that
the displayed image is semantically the intended frame, authenticate the
caller-supplied root pin externally, or calibrate the browser clock. The viewer
export itself remains unsigned and self-reported; the frame-review signature
attests the joined candidate only. All execution, model-load, and scoring
authorization fields remain `not_granted`.

## Faithfulness status

This is review-provenance infrastructure, not a faithfulness experiment. Direct
faithfulness remains 0/3 claim reviews, 0/15 browser-clock/decoder
correspondences, 0/3 prediction-link judgments, and 0/3 direct scores. There is
no direct `ΔF`, confidence interval, or statistical test to report. The earlier
head search remains closed at −0.0756 nat (95% t CI [−0.1578, +0.0065], 79/79)
and must not be reopened.

## Follow-up verification and schema-boundary repair (2026-10-18)

The repository-wide suite passed **2063/2063** in **86.937 seconds** before the
additional repair below. An independent audit then reproduced a builder /
downstream schema mismatch: the builder signed identity maps with unknown
top-level or row keys and did not explicitly reject JSON `true` for schema
version 1, while the downstream auditor rejected that map. The builder now
requires one of the two allowed exact top-level key sets, an exact integer
schema version, a valid optional input-manifest digest, and exactly five
identity row fields before signing. New regressions cover all three malformed
forms. A valid map still builds and passes the actual downstream auditor; the
independent re-audit found the mismatch closed.

The helper, builder, and frame-evidence auditor suites passed **80/80**;
`py_compile` and `git diff --check` passed. After the repair, the repository-
wide suite passed **2064/2064** in **97.134 seconds**. The inventory builder was
rerun after its registered evidence-builder source changed: it recorded 82
roles, 0 missing, candidate-only status, denied review/freeze/score/execution
flags, and rights unverified local-only. The new private inventory is mode
0700/0600; its bytes and digest remain local and are not published.

No review package was opened and no human judgment, GPU score, or faithfulness
result was produced. Direct status remains 0/3 claim reviews, 0/15 verified
frame correspondences, 0/3 prediction-link judgments, and 0/3 direct scores.
The independent trust-root/key assignments, review-only receipt, witness
handoff, and human reviews remain outstanding.

## Next action

Do not run another GPU outcome measurement on the already used 30-row
validation pilot or 490-video test-associated pool. The independent lineage
audit found no local source-identity crosswalk. The [VidOR dataset
documentation](https://xdshang.github.io/docs/vidor.html) provides `video_id`
and `video_hash` fields in its train/validation annotation format, but says
test videos are available after participating in the grand challenges; the
[local NExT-GQA map documentation](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)
describes its map as video-ID-to-path only. The [Multimedia Commons YFCC
page](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/), checked
2026-10-18, says Webscope is not currently offering the metadata dataset even
though original media are listed in its AWS bucket. Thus the next justified
human-free source-lineage operation is metadata-only and requires an
independently sourced, provenance-pinned crosswalk; a GPU similarity rerun
would not resolve source identity.

For the human track, wait for actual independently trusted root/key
assignments, signed role rosters, a review-only freeze receipt, and witnessed
unopened handoff before generating a review bundle or opening/scoring material.
The current inventory remains preparation only. Direct faithfulness is still
0/3 claim reviews, 0/15 verified frame correspondences, 0/3 prediction-link
judgments, and 0/3 direct scores; no direct `ΔF`, interval, or test exists.
