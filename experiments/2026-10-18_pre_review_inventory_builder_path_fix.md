# Explicitly bind the current human-review package in candidate inventories

## Purpose

The 2026-10-18 disclosure repair produced a current offline review page and
package, while the pre-review inventory builder still selected the 2026-10-17
package and standalone HTML through module-level constants. That mismatch meant
a fresh candidate inventory could not bind the current reviewer-facing bytes.
This repair makes the three package inputs explicit and keeps the inventory
non-authorizing.

## Implementation and checks

- Added all-or-none CLI/API overrides for package directory, staged-media
  directory, and standalone review HTML. Overrides must remain repository-local
  beneath `data/private_nextgqa`; custom package/media directories must be
  owner-controlled mode `0700`, and private files remain mode `0600`.
- Added tests for binding a distinct candidate page/package, path containment,
  and rejection of partial overrides. Focused builder suite passed **19/19**;
  `py_compile` and `git diff --check` passed.
- Built a new local-only candidate inventory from the current
  `reviewer_package_v6_20261018_0243` and
  `claim_review_v6_20261018_0243.html`, using the staged media directory
  `review_media_v6_20261017_0635`. It contained **65 roles, zero missing
  required slots**. A separate hash pass independently verified that the
  standalone HTML and package `index.html` match, and each of the three package
  media files matches its staged counterpart and both inventory role hashes.
- The inventory still reports `candidate_only`, `review_completed=false`,
  `score_ready=false`, and `execution_authorization=not_granted`. It is a local
  artifact and is not included in this repository change.

## Interpretation and limits

This fixes which local review package a future candidate inventory can bind. It
does not establish a review-only freeze, a trusted root/key roster, a witnessed
unopened handoff, human identity, actual viewing, semantic claim validity, or
browser-to-decoder correspondence. The package remains unopened and all
protected material remains local; rights status is unverified.

Faithfulness progress: **direct evaluation not performed**—independent claim
review **0/3**, scored **0/3**, no `ΔF`, interval, or test; browser-to-decoder
correspondence **0/15**. The recent question-token temporal selector result is
engineering-only synthetic evidence and does not change these counts.

## Next action

Use an independently trusted human role/key process to establish the review-only
freeze and witnessed handoff before opening the offline package. Then complete
the three claim reviews and the separate 15-anchor interactive screenshot and
blinded frame-pair review. Do not start efficacy scoring until those inputs and
the distinct internal-treatment protocol/scorer are frozen and independently
audited.
