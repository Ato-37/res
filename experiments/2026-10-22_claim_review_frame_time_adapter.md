# Claim-review response to frame-time adapter (2026-10-22)

## Purpose

Make a future response to the current low-burden claim/frame review artifact
usable as a private exploratory input without treating self-reported answers as
authenticated or as a faithfulness result. Keep all requested items in the
denominator and prevent reviewer disagreement from becoming selected support.

## Implementation

Added `scripts/adapt_claim_support_frame_review_response.py`. It first runs the
existing exact-fingerprint/schema inspector, then joins each answer to the
same item and its 16 artifact frame timestamps. It emits per-item opaque hashes,
source-lineage digests, decision counts, selected zero-based indices and their
seconds. It excludes a row from exploratory support scoring unless every
submitted reviewer chose `supports`; all items remain present, including
disagreement and non-support. It rejects non-finite, negative, duplicate or
non-monotone frame times. The CLI limits input sizes, rejects non-regular input
files, writes only a new owner-only (`0600`) output under an owner-only (`0700`)
directory, and prints aggregate counts only. Output hard-codes
`model_load_permitted=false` and `execution_authorization=not_granted`.

Five `unittest` cases passed for: exact frame-index-to-time mapping, reviewer
disagreement, non-support retention, wrong artifact fingerprint, and invalid
frame-time order. The new module and test passed `py_compile`; scoped
`git diff --check` passed. The environment has no `pytest` module in either the
repository `.venv` or system Python, so the existing pytest-targeted suite could
not be rerun here; no dependencies were installed or environments changed.
An in-memory synthetic response was structurally joined against the actual
current artifact: 3 requested rows, 0 eligible rows, with human review and
model authorization both false. No response file was created.

## Evidence status and limitations

This is measurement-readiness engineering only. There is still no response
bound to the current artifact, so effective human-reviewed rows remain 0/3.
The adapter's agreement rule is intentionally conservative but does not
authenticate distinct reviewers, validate semantic judgments, prove video
viewing, or recover evidence omitted from the 16 thumbnails. It produces no
model score and cannot authorize model loading.

Independent source audit confirmed there is no current production runner that
consumes this lightweight response format and computes a claim-frame versus
matched-control intervention score. Existing QTA scorers use other frozen
cohorts/estimands and explicitly do not measure direct faithfulness. Therefore
the response-to-time step is now executable, but another adapter into the
formal signed review schema would be a different protocol and should not be
silently implied.

## Next action

After an artifact-bound response exists, run this adapter locally and inspect
only the aggregate/private output. Then implement and test a dedicated scorer
for the exact frozen claim token IDs: full-video baseline, equal-budget
reviewed-support key ablation, and matched non-support temporal control
ablation, with every requested row/failure retained. Before using real response
labels, freeze that score, matching rule, and denominator. Do not relabel this
adapter output as direct faithfulness evidence or reuse a legacy response.

No GPU job ran and no output was published. Human review has not been
performed; rights status remains unverified.

## Cross-reviewer frame-consensus correction (2026-10-22)

An independent implementation audit found the initial adapter pooled selected
frames by union even when multiple reviewers all marked the claim as supported.
That could score a unilateral frame selection as group-supported evidence.
The adapter now maps only frame indices selected by every reviewer
(intersection). If support labels agree but no frame overlaps, the item remains
requested and is marked ineligible; it is not replaced. Two new tests cover
partial intersection and empty intersection. The dedicated scorer protocol now
states this rule. No current response exists, so this change adds zero reviewed
items and no faithfulness score.
