# Review response and spatiotemporal preflight checkpoint (2026-10-22)

## Claim-support review JSON

Inspected the local Downloads file
`claim_support_frame_review_2026-10-21T19-03-05-100Z.json` without copying it
into the repository. Response SHA-256:
`3d9faa75edf9ce69fe9730b01a85094ded411a49bdfdb85761809c00f38cd267`.

The canonical artifact fingerprint binds exactly to the retired
`data/private_nextgqa/claim_support_frame_review_test_20261021_06.html`
(fingerprint `abaa3b8dee8f4fc9cbd2031d1772d2912502a9c1bafb1f2b03dac90365d8a1d4`,
HTML SHA-256 `309739d3aacb39ea46745c41893fbc2736661e421886e228922f1690faa52715`).
The response inspector reports a structurally valid response with 3 reviewer
entries, all 3 items complete per entry (9 total answers), and no attrition.
The response does **not** bind to the current `_08` artifact: its ordered item
IDs differ, and the fingerprint differs. These answers cannot be counted toward
the current `_08` review cohort or formal review gate.

Self-reported decisions by item order:

| Item | Contradicts | Not visible in sampled frames | Cannot judge | Supports |
|---|---:|---:|---:|---:|
| 1 | 2 | 1 | 0 | 0 |
| 2 | 0 | 1 | 2 | 0 |
| 3 | 3 | 0 | 0 | 0 |
| Total | 5 | 2 | 2 | 0 |

The inspector verified response structure, exact `_06` binding, frame-index
syntax, and item ordering. An independent audit reproduced those checks and the
counts. It also confirmed current packet, video, baseline, interaction, and row
token digests match the three embedded item provenance records. The embedded
generator hash does not match the current untracked generator source, so the
exact historical builder code was not recovered. Reviewer identity,
independence, viewing behavior, and semantic support remain unverified; the
JSON itself explicitly records these limits. Treat these as self-reported
exploratory judgments only. In particular, the two “not visible” judgments do
not imply absence from the full video, and there is no positive support vote.

## Image-to-video bbox preflight

Added exact-anchor frame insertion and connected the bbox mapper to the
existing CPU frozen-cohort preflight. The sample count is now the largest even
count up to 64 that fits the video's duration at nominal 2 fps. The exact
annotation frame is inserted by replacing the nearest sample slot; resulting
actual PTS gaps are recorded. The preflight applies the installed Qwen video
processor to the exact two-frame pair containing the annotation frame, then
maps the official bbox into its merged visual grid. This is CPU preprocessing,
not model inference.

Verification: helper and preflight suites passed 45/45 together;
`compileall` passed. The initial fixed-64 run requested 3 rows and passed 0:
the available video was shorter than 31.5 seconds, while two videos were
missing. After the predeclared duration-based adjustment, receipt
`data/private_minerva/minerva_ego_bbox_patch_preflight_20261022_03.json`
(SHA-256 `81782b9491cd2b546bdbd7e548c7a4d6a072a5184877662bd5407d950fefaae5`)
requested 3 rows, passed 1, and retained 2 `video_file_missing` failures.
The passing row used 56 frames over a 27.5-second target window; the exact
annotated frame was sample rank 20 and paired with the next sample at 10.5 s.
The actual Qwen CPU processor returned `[1,22,22]`; the bbox mapped to 6 of 121
merged visual cells, with no clipping. A source audit caught and removed a
stale fixed-64 control-group calculation before this rerun. The 56-frame clip
has 28 temporal groups; evidence group 10, track-overlap groups 10 and 11,
and matched control group 24 are in range. An independent audit reproduced the
cohort/manifest/source bindings, decoded the same 56-frame schedule, reprocessed
the pair with the CPU Qwen video processor, and independently recalculated the
six bbox cells and control group. Receipt/source hashes match. Rights status
remains unverified and all source media/receipt content is local-only.

This preflight establishes one actual-video spatial geometry mapping, not a
causal intervention result. The runner still needs to be generalized from its
old hard-coded row and wire in the new spatial mapper before a GPU run. It still
requires 64 frames/32 groups, while the only available candidate supports 56
frames/28 groups; these counts must be generalized or eligible media retrieved.
The CPU preflight does not yet bind the bbox cells to actual visual-token input
positions or the intervention hook. The temporal patch still fuses two frames; reported
effects must be scoped to the sampled pair. The missing 2/3 candidate videos
block the frozen-cohort GPU run until retrieved and checksum-verified, but do
not block further CPU-only runner repair or a separately labeled one-video
plumbing smoke that is excluded from confirmatory follow-up.

Direct faithfulness progress: this response adds no verified support label or
direct score to the current `_08` cohort. The older `_06` response is useful
exploratory contrary/uncertain feedback, but formal-review and direct
faithfulness counts remain unchanged.
