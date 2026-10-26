# Exploratory video-review response analysis (2026-10-21)

## Purpose

Inspect the locally supplied three-person exploratory review and make its
item-level pattern reproducible without printing nicknames or individual rows.
This is a usability/construct check for the review artifact, not a faithfulness
experiment.

## Setup and provenance

- Inputs: two locally available legacy exploratory JSON exports dated
  2026-10-19. The later v5 export contains three respondents with three ratings
  each. The earlier v2 export contains one respondent with three overall
  judgments. Their nickname strings do not overlap, but nicknames cannot prove
  distinct human identities; these exports are kept as separate cohorts and are
  not pooled.
- The later v5 response passed the local structural validator and its opaque item IDs
  match the three IDs in the older
  `quick_review_ja_simple_20261019_itemid_v14.html`. A separate comparison
  found that those IDs **do not** match the current GPU-generated-claim review
  artifact `quick_review_val_generated_claims_local_20261019_04.html`. Because
  schema v5 has no artifact fingerprint, the exact HTML byte version for the
  older artifact is **not verified**.
- The response and HTML contain local protected material and remain untracked
  and local. Nicknames, question/claim text, and raw answers are not recorded
  here or in Git.
- Added an opt-in `--show-item-summary` mode that aggregates the three rating
  axes by item without disclosing reviewer rows. Existing default output remains
  aggregate-only.

## Verified observations

There are 9 exploratory ratings (3 people × 3 items):

| Item | Question unclear | Answer does not fit | Evidence insufficient |
|---|---:|---:|---:|
| 1 | 2/3 | 1/3 (1/3 cannot tell) | 2/3 (1/3 cannot tell) |
| 2 | 2/3 | 2/3 | 1/3 (1/3 sufficient, 1/3 cannot tell) |
| 3 | 0/3 | 0/3 | 2/3 |

Across all items, question clarity was 4 clear, 4 unclear, and 1 cannot tell;
answer fit was 5 fits, 3 does not fit, and 1 cannot tell; evidence sufficiency
was 2 sufficient, 5 insufficient, and 2 cannot tell.

The separate earlier v2 export also matches the older artifact's three item
IDs. Its single respondent marked Items 1 and 3 as roughly matching and Item 2
as partly differing. This aligns descriptively with the later v5 export's
2/3 Item-2 answer-fit disagreements, but the ratings use different schemas and
the exports cannot establish independent human identities, so no combined
denominator or agreement statistic is reported.

The Downloads copies were checked separately: both map only to the older v14
item IDs, and neither has the current generated-claim fingerprint. The v2
Downloads file differs byte-for-byte from the same-named workspace copy, so
the Downloads copy was analyzed directly; nicknames/notes were never printed.
The current private importer regression suite passed **14/14**, including exact
fingerprint success, fingerprint mismatch rejection before copy, owner-only
storage, symlink rejection, and legacy opt-in behavior.

The response-inspector tests passed **18/18**. The workspace/attestation checks
from the same session passed **8/8** formal-workspace tests and **64/64**
prediction-link, integrity-receipt, completed-review, and capture-operator
tests. These are software checks, not evidence that formal human gates have
been completed.

## Interpretation and limits

With only three informal respondents and an unbound legacy export, these counts
are descriptive feedback. They do not estimate model faithfulness, establish
claim truth, or authorize scoring. The feedback flags the older artifact's Item
2 for a question/claim review and Item 3 for stronger evidence-bearing wording
or an explanation that states the relevant action sequence. Item 1 has a clarity
problem and needs a careful question review before being reused. These responses
say nothing about the three claims in the current GPU-generated-claim artifact.
Any feedback on those claims must use that artifact's matching IDs and
fingerprint.

## Current GPU-generated-claim artifact check

The current launcher target still matches its recorded SHA-256, is owner-only
mode `0600`, has a valid embedded fingerprint, and contains three unique item
IDs. Its embedded review records have exactly the fields for opaque item ID,
source question, Japanese question, generated claim, Japanese claim, and video
bytes. No gold-answer, baseline-output, score, or correct-choice fields appear
in those records. A scan of 51 local candidate response JSON files found **no**
response bound to the current artifact fingerprint and **no** legacy response
whose item IDs match it. The user's v5 response remains attached only to the
older artifact family.

Direct faithfulness remains **0/3** formally reviewed claims, **0/15** verified
clock/frame correspondences, **0/3** Stage-2 prediction-link reviews, and
**0/3** direct scores. No effect estimate, interval, or statistical test exists.
No GPU inference was run: exploratory response analysis does not satisfy the
pre-load gates. Dataset/model rights remain unverified.

## Next action

Treat the older artifact's results as exploratory feedback only. For the current
GPU-generated claims, collect a fresh response using the matching fingerprinted
artifact. Separately, continue the formal review/release handoff and production
pre-load transaction work; do not promote this legacy feedback into formal gate
evidence.

## Follow-up response-directory scan

After the current-artifact importer default was corrected, I rechecked the
private importer response directory using one parsed artifact snapshot and
bounded, no-follow response reads. The directory contained **50 JSON files**;
**0** matched the current v6 fingerprint and **0** legacy responses matched the
current item IDs. The scan reports only whether a file joins to the current
artifact; nonmatching files are not classified here as invalid because they may
belong to older artifacts. The existing Downloads candidates are the two
previously inspected old exports (one copied in both locations), with no newer
filename present. This confirms that the current human review has not arrived.

Direct counts remain 0/3, 0/15, 0/3, 0/3; no model or GPU forward was run.

Current runtime read-only check (2026-10-21 JST): RTX 5070 Ti, 14,552 MiB
free VRAM, 1% GPU utilization, CUDA available in the existing PyTorch
2.11.0+cu128 environment, BF16 supported, and 14,059,560 KiB host memory
available. Compute capacity is available; it does not satisfy the missing
review, trust-root, or pre-load authorization evidence.

## Legacy response-to-claim continuity audit

The earlier response was revalidated with the repository inspector against the
local v10 HTML using explicit legacy mode. It again yielded three self-reported
respondent slots and nine categorical ratings. The response JSON has no artifact
fingerprint, so the exact viewed bytes cannot be proven cryptographically. Its
creation time falls after the preserved v10 artifact timestamp and before the
preserved v11 artifact timestamp; this is chronological support for a v10
review, not proof that the browser displayed that file.

The embedded item manifests for v10 through v14 were compared field by field
without printing claim, question, or media content. Items 1 and 3 are identical
across all five versions. Item 2 retains the same item ID, question, and claim,
but its embedded video bytes change at v11 and remain changed through v14.
Therefore the v10 ratings remain exploratory impressions of the exact v10
item set; they cannot be transferred to the later item-2 video. This explains
why stable item IDs alone were insufficient to bind the old answers to the
later artifact.

All three v10 question/claim pairs also join exactly, by source-text hashes, to
one row each in the preserved sentence-exploration claim manifest and source
lineage. The seven corresponding `diagnostic_target_*` receipts are explicitly
`diagnostic_candidate_unvalidated` with 0 model forwards and 0 score calls.
Thus the old human impressions are grounded in an actual generated-claim
candidate set, but no corresponding model intervention outcome exists to
compare against them. They do not add an intervention effect or a formal direct
faithfulness score.

The formal current-cohort status is unchanged: current-claim responses 0/3,
clock/frame correspondences 0/15, Stage-2 links 0/3, and direct scores 0/3.
This audit recovers limited, version-scoped human feedback for the older v10
candidate set only. It does not authenticate reviewer identity or independence,
prove video viewing, establish claim truth, or qualify as claim-to-frame
annotation.

## Downloads recheck after the user waived further review collection

The two JSON files currently in the user's Downloads folder were checked again
against the exact current generated-claim HTML and the old v14 artifact. The
schema-v2 one-person export and schema-v5 three-person export both map to the
old v14 item IDs, both have no artifact fingerprint, and each has **0/3** item
IDs matching the current three generated claims. The current-artifact
inspector rejects both for item mapping/judgment mismatch. With explicit
legacy inspection against v14, the v2 export validates as 1 respondent / 3
coarse judgments; the v5 export validates as 3 respondents / 9 ratings, with
aggregate answer-fit counts 5 match, 3 mismatch, 1 cannot tell, and evidence
sufficiency 2 sufficient, 5 insufficient, 2 cannot tell. No names, individual
rows, or free text were emitted. This confirms that informal human feedback
exists for the older v14 check, but it cannot be carried over to the current
GPU-generated claims. The user's waiver allows exploratory work to proceed
without collecting another response; it does not change the artifact binding
or make the current claim review count nonzero.

## Scorer compatibility audit for a possible v10 follow-up

A source audit of `scripts/run_frozen_frame_mean_attention_bias_interaction.py`
found that its packet validator accepts only `generation_prompt_version="v2_short"`
and its output hard-codes `human_review_performed=false`. The old reviewed v10
candidate set is tied to the earlier generation semantics recorded in its
sidecar; it therefore cannot be scored through this runner without changing
prompt binding, frozen packet provenance, and review-status reporting. No GPU
run was launched. Reusing the runner as-is would misstate the prompt and human
review history. Any future follow-up must reconstruct a version-specific locked
packet and state explicitly that the broad human impressions are not formal
claim-to-frame labels; it must not use them as support intervals or direct
faithfulness outcomes.
