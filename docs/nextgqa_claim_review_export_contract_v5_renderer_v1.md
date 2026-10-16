# NExT-GQA review v5 renderer export disclosure (v1)

Status: implementation disclosure for newly rendered pages. The frozen
`nextgqa_claim_review_policy_v5.json` is unchanged. Its `review_order` sentence
that says the reviewer exports only tokens and judgments/intervals does not
describe the renderer's complete JSON payload. This document and the page notice
make that discrepancy explicit; they do not retroactively alter already-rendered
pages or authorize review/scoring.

The renderer's review JSON has the following keys (nested keys are shown with
their parent):

- top level: `schema_version`, `purpose`, `status`, `review_completed`,
  `execution_authorization`, `claim_review_policy_id`,
  `claim_review_policy_sha256`, `human_reviewer_role`,
  `human_reviewer_private_id`, `review_environment`, `independence_attestation`,
  `export_disclosure_acknowledged`, `records`;
- `review_environment`: `browser_user_agent`, `offline_static_page`,
  `network_fetch_available`;
- each `records[]` entry: `review_token`, `review`;
- each `records[].review`: `full_source_video_reviewed`,
  `question_and_claim_only_context_shown`,
  `gold_and_seed_hidden_until_mapping_frozen`, `outcome_scores_not_accessed`,
  `media_bytes_verified`, `occurrence_policy_applied`, `support_decision`,
  `evidence_intervals_seconds`, `interval_rationales`, `media_duration_seconds`,
  `reviewer_role`, `first_played_at_utc`, `reviewed_at_utc`, `review_status`,
  `display_player`, `claim_review_policy_sha256`.

The renderer does not add a separate top-level creation timestamp to this
payload. The page includes `browser_user_agent` and `display_player` as declared
review-environment metadata; `interval_rationales` is free text and may contain
video-derived content.

The separately downloaded review integrity index has exactly these keys:
`schema_version`, `purpose`, `review_payload_sha256`,
`claim_review_policy_sha256`, `created_at_utc`, `review_status`, and
`execution_authorization`.

The optional HTML5 clock-observation payload has exactly these top-level keys:
`schema_version`, `purpose`, `status`, `player_clock`,
`presented_frame_metadata`, `decoder_frame_identity`, `created_at_utc`,
`browser_user_agent`, `anchor_id_validation`, `predeclared_plan_binding`,
`fit_heldout_split`, `decoder_pts_pairing`, `execution_authorization`,
`records`. Each record has `review_token` and `events`. Each event has
`anchor_id`, `sequence`, `html5_current_time_seconds`,
`request_video_frame_callback_metadata`, `frame_match_status`,
`performance_now_ms`, `media_duration_seconds`, `playback_rate`, `muted`,
`volume`, `text_tracks_disabled`, `paused`, `seeking`, `ready_state`,
`current_src_is_staged_media`, `visibility_state`, and `captured_at_utc`.
The separate clock integrity index has exactly `schema_version`, `purpose`,
`payload_sha256`, `created_at_utc`, `status`, and `execution_authorization`.

None of these browser-generated fields independently proves who reviewed the
material, what was watched, or when it was watched. In particular, interval
rationale text can contain video-derived content and the reviewer identifier
and user-agent are identifying metadata; keep the JSON, indexes, and clock
observations in the private custody location and do not publish them. A newly
rendered page displays all these fields and requires an explicit disclosure
acknowledgment before enabling review export; that acknowledgment is recorded
as `export_disclosure_acknowledged` in the review JSON.

Any handoff must show this disclosure before review begins. Existing frozen v5
pages remain historical bytes and must not be silently replaced or treated as
having shown this notice. A new page should be rendered and audited as a new
artifact; its policy digest still identifies the frozen v5 policy and does not
hash this implementation disclosure.

The current reviewer page presents the review JSON and integrity index as a
two-step required handoff. After the JSON download is initiated, it explains
that the page must remain open until the index download is initiated; the
page directs both files to the private custodian and offers re-download buttons
for the same frozen payload/index if a browser download fails. The page
explicitly says that initiating a browser download does
not prove that the file reached disk. It avoids persistent browser storage
because the review payload includes private reviewer metadata and free-text
video-derived rationales. When clock observations are collected, their payload
and integrity index must also be kept and returned privately. This is a
handoff usability aid only and does not authenticate human review or authorize
scoring.
