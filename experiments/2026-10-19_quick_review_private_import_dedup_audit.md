# Exploratory review import deduplication audit — 2026-10-19

## Purpose

Prevent repeated imports from increasing the apparent number of informal
reviewers, and stop synthetic importer tests from writing into the real private
response directory.

## Evidence and implementation

The Windows Downloads response was validated against the current v10 artifact
and contained one respondent with three complete broad judgments: two
`roughly_matches`, one `partly_differs`. It exactly matched two existing
owner-only copies in the private response directory. A second distinct
payload also matched the current v10 item IDs and carried the same nickname
field, ordered item IDs, judgments, and optional notes; only its creation time
differed. Because nickname equality cannot authenticate a person, these
payloads are conservatively counted as **one informal respondent**, not two.
No response text, nickname value, or item content is recorded here.

The response directory initially contained **100 files**. There were **50**
exact copies of a known synthetic test fixture, including files created while
the CLI test patched `PRIVATE_ROOT`: `import_response()` had captured the
original module constant as a default argument, so the patch did not redirect
the write. The importer now resolves the default root when called. The exact
50 owner-only fixture files were verified byte-for-byte against the fixture
constructed in the test and removed; all unmatched files were retained. The
directory now contains **49 files**, **8 unique payloads**, and **2** unique
payloads that structurally map to current v10 item IDs. The two valid payloads
represent one conservative informal respondent. This local directory recount
does not authenticate reviewers or validate their judgments.

The importer now searches only matching content-hash filename candidates,
opens them without following symlinks, checks owner/type/mode/size and stable
read metadata, then compares the full SHA-256. An exact match reuses the
existing owner-only file; different bytes still go through exclusive
no-overwrite publication. Synthetic tests cover exact reuse, no extra file,
CLI disclosure, and collision behavior.

## Verification

- Importer, response-inspector, and artifact-builder tests: **32/32 passed**.
- Python compilation and scoped `git diff --check`: passed.
- The saved Windows download was retained. One newly-created redundant copy
  from this audit was removed only after verifying two pre-existing exact
  copies remained. Rights status is unverified; all responses remain local.

## Interpretation and faithfulness progress

The user's qualitative feedback had already motivated the v5 rubric, which
separates question clarity, answer fit, and perceived video-evidence
sufficiency. The old JSON records only a broad impression, so it cannot
retroactively answer those three dimensions. This audit confirms one informal
respondent; it does **not** add a formal claim review, evidence interval,
prediction-link judgment, faithfulness score, or human-ground-truth label.

Direct faithfulness remains **0/3** formal Stage-1 claim reviews, **0/15**
verified clock/frame correspondences, **0/3** Stage-2 prediction-link reviews,
and **0/3** direct scores. There is no empirical direct `ΔF`, interval, or
statistical test. An independent current GPU-eligibility audit found no open
real-video GPU operation that advances this direct estimand before the
outstanding trust and review inputs arrive; the completed proxy cohorts remain
closed.

## Next action

Continue to use the v10 sequential artifact for additional informal calibration,
with one nickname per person and one final combined export. Separately obtain
the actual formal root/role/release materials and complete the Stage-1 and
clock/frame inputs. Once those pass, proceed to the prespecified three
untreated baseline forwards, freeze `p/q`, and then request independent
Stage-2 judgments before treatment inference. Keep response JSON private and
do not count nickname or file count as reviewer identity.
