# Claim review export contract v6

Policy source: `nextgqa_claim_review_policy_v6.json`, SHA-256
`1e04d9c001b1249bd2d04d9f43730902c45341543d0a7f1aa6badf4b40dbff91`.
This version keeps v5 unchanged and resolves its mismatch with the renderer's
actual export schema before any v6 review begins.

The review JSON records a private reviewer pseudonym and role, declared
independence, browser environment, export-disclosure acknowledgment, tokenized
review decisions, intervals and rationale text, playback/review timestamps,
media duration and display declaration. A separate integrity index binds the
review JSON bytes. Optional clock calibration exports a browser observation
JSON and a separate index; the exact exported field names are enumerated in
the policy's `export_contract` object and disclosed on the page.

The reviewer page asks for a pseudonym rather than a legal name. Rationale text
may contain video-derived information. Store the page outputs, video-derived
notes, custody crosswalk and staged media only in private local storage; use
mode 0600 for files and mode 0700 for directories. Do not publish or externally
share them. The user authorized local-only protected-data operations; rights
status remains unverified.

All reviewer statements, user-agent values, playback times, durations, and
clock observations are self-reported. Hashes establish byte identity, not
reviewer identity, actual viewing, claim truth, or browser-to-decoder frame
identity. The review result remains unvalidated until a separate custodian
audits the review package and source/media crosswalk. The page does not grant
scoring authorization. The v2 protocol and production atomic launch gate
remain unfrozen/unimplemented.
