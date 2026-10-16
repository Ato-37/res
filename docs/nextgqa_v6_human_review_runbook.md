# NExT-GQA v6 local human-review runbook

Status: **preparation only**. Do not open the reviewer package until the
pre-outcome joint-freeze gate below is complete. The local byte inventory is a
candidate receipt, not a freeze, review, or authorization.

## Current local package

The private offline package is
`data/private_nextgqa/reviewer_package_v6_20261017_0635/index.html`. It contains
one HTML page and three opaque-token-named videos. Keep it and all exports on
the local machine; do not upload, commit, email, or otherwise share the package
or derived material. The latest hash-only package inventory is
`data/private_nextgqa/pre_review_inventory_v6_20261017_0830_02/pre_review_candidate_byte_inventory.json`.
It recorded 50 present roles and no missing slots, but has
`joint_freeze=false`, `review_completed=false`, `score_ready=false`, and
`execution_authorization=not_granted`.

## Before anyone views a video

An outcome-blind custodian and the independent pre-outcome freeze owner must
first freeze the exact v6 claim/cohort inputs, source and media bindings, the
v2 protocol, 32-frame sampler, relevant code, capture policy/helper, thresholds,
and the designated reviewers' public-key hashes. This freeze must be made
before a reviewer opens the page. The current protocol is still a draft, and
there is no production joint-freeze receipt; do not treat the candidate
inventory or any self-reported review export as a substitute.

Assign people to these roles before opening the package. The freeze owner may
also serve as custodian, but must not serve as either reviewer.

1. **Pre-outcome freeze owner:** records the exact artifact hashes, protocol,
   sampler, capture policy/helper, thresholds, and reviewer-key hashes before
   any human viewing or outcome access. Keeps the signed freeze receipt local.
2. **Custodian:** verifies source lineage and media bytes, holds the private
   token crosswalk, and validates returned files. Must not generate claims,
   score outcomes, or serve as the claim reviewer.
3. **Claim reviewer:** independent of claim generation, scoring, and custody.
   Receives only the frozen offline page and token-named videos. Must not see
   dataset identifiers/indexes, the crosswalk, gold answers, seed intervals,
   outcome scores, or external answer sources.
4. **Frame-pair reviewer:** independent of capture, custody, pair preparation,
   and scoring. After capture artifacts are prepared, reviews all 15 blinded
   screenshot-crop/decoder-frame pairs and records match, mismatch, or
   ambiguous decisions.

If no independent people can fill these roles, leave the gate blocked; do not
substitute an agent-generated label or an operator self-attestation.

## Claim review

After the freeze owner releases the package, the claim reviewer reads and
acknowledges the full export disclosure before viewing. The reviewer may stop
without viewing if they do not accept its terms. Otherwise, they open the local
`index.html` offline and review all three complete videos at normal speed with
audio muted and text/caption tracks disabled. Visible pixels are the only
permitted evidence; burned-in text cannot support a claim. The reviewer records
whether every material factual component is visibly established and marks
**every distinct sufficient occurrence** as its own positive-duration interval. Do not merge
disjoint occurrences, invent an interval, select only a convenient occurrence,
or replace a failed item. Keep all three items in the denominator.

The page's review JSON, integrity index, and optional clock JSON/index are
self-reported evidence. Save them in the private local custodian area with
files mode 0600 in directories mode 0700; do not publish them. The custodian checks exact token coverage,
package/media hashes, and source mapping with the existing handoff audit. A
structural pass does not authenticate reviewer identity or semantic support.

## Browser-clock and frame correspondence

Using the exact staged media and the frozen plan, perform all 15 planned
beginning/interior/end anchor observations in the actual review browser. The
OS helper needs a human to position the player and trigger each capture; it
captures the foreground window. Produce a video-only crop for every capture
and retain the original screenshot privately. The crop coordinates and hash
alone do not establish crop purity. The separate frame-pair reviewer must
inspect every blinded crop/decoder pair. Missing, mismatched, or ambiguous
pairs block the whole cohort and calibration. Then build and run the signed
evidence audit and the predeclared fit/held-out calibration audit; record
empirical residuals without describing them as uncertainty bounds.

## Route back into the experiment

Once the review exports and clock evidence exist, retain the original bytes,
recompute hashes, and run the existing structural handoff and signed clock
evidence audits. Resolve every audit failure before constructing masks. A real
score still requires a production atomic preflight that binds all three rows,
the exact 32-frame sampler, actual source target and prompt/token IDs, all nine
processor batches, and the pre-outcome joint freeze before model loading. For
every row, the calibrated 32-frame sample must overlap at least one reviewed
evidence interval and admit an equal-count set of distinct non-evidence control
ranks under the frozen distance/tie rule. Zero evidence overlap or too few
controls blocks the full cohort before model loading. Any other missing/failed
row likewise blocks all nine forwards; never score a surviving subset.

The planned result is a finite three-item descriptive contrast only. Even a
complete score does not establish that the model used the evidence or that the
generated explanation is true. Keep per-item scores and all QA/video-derived
material local; publish no protected derivatives.
