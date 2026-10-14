# Prospective single-interval claim-grounding metric contract

Status: synthetic CPU contract only. This document fixes a candidate metric
calculation, not the final study protocol or authorization. Item-level rights,
reviewed claim mappings, and study-design gates remain separate.

## Unit and metric

The planned output is one claim and one predicted temporal interval per frozen
video/QA unit. Human reviewers must first decide whether the claim is
answer-relevant and which annotated gold intervals support that claim. The
metric receives only the reviewer-approved gold interval set; it must never
choose a gold interval based on the predicted timestamp.

For a valid predicted interval P and approved gold intervals G, the score is
max over g in G of tIoU(P,g), where
tIoU(P,g) = duration(P intersect g) / duration(P union g). The intervals in
G are alternative acceptable evidence locations for this single-interval
prediction, not a union. This follows the official NExT-GQA eval_ground.py
convention: it computes IoU against every gold span and retains the maximum.
See [the official evaluator](https://github.com/doc-doc/NExT-GQA/blob/main/code/TempGQA/eval_ground.py),
especially its per-span loop and mean accumulation. That convention is
appropriate only when the approved spans are alternatives; if multiple moments
are jointly required to support a claim, the data do not fit this single-span
metric and the protocol must be revised before scoring.

## Denominator and failures

The denominator is the entire frozen eligible video/QA set. The official
evaluator skips missing prediction keys, but this project must not inherit that
behavior: absent, unparsable, empty, non-finite, reversed, zero-length, or
out-of-video intervals stay in the denominator with tIoU 0 and a generic
failure code. Do not clamp, reorder, or silently drop intervals. Report each
failure rate alongside the unconditional primary mean.

Claim alignment is reviewed independently of condition labels, predicted
timestamps, and model scores. A reviewer status of mismatch or unverifiable
receives tIoU 0 and a separate failure code while remaining in the
denominator. An unresolved review is a pre-run gate failure and blocks
scoring. For matched, require at least one positive-duration approved gold
interval. Gold intervals and predictions must satisfy
0 <= start < end <= video_duration. Point annotations are outside this
contract; encountering one blocks protocol use instead of silently inheriting a
metric convention.

This separates end-to-end system grounding from complete-case localization.
Excluding malformed predictions or mismatched claims after generation would
condition the primary outcome on model output and could change the estimand.

## Executable contract

scripts/temporal_claim_grounding_utils.py implements the arithmetic and
generic failure codes without loading data or model dependencies.
scripts/test_temporal_claim_grounding_utils.py uses synthetic intervals only.
The helper does not parse timestamps, inspect videos, authenticate reviewers,
or establish that a claim maps semantically to an interval. A CPU pass verifies
only this calculation contract.
