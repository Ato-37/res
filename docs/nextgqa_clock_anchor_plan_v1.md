# Candidate HTML5 clock anchor plan v1

Status: **candidate outcome-blind operational design; not jointly frozen with
the v2 protocol, not a clock calibration, and not scoring authorization**.
The downstream converter's `predeclared_frozen` field means only that this
anchor schedule is fixed before observations; it does not mean the cohort,
claim inputs, target, sampler, or scoring protocol has been jointly frozen.
Use it only after reviewing its definition and jointly freezing those inputs.

For each exact staged video, decode display-order frames with the pinned
PyAV-based decoder and use the first decoded presentation timestamp as time
zero. Let `T` be the relative PTS of the last decoded frame. At target times
`T × (1/20, 1/4, 1/2, 3/4, 19/20)`, choose the decoded frame minimizing the
absolute time difference; ties select the lower display-order index. The
5%, 50%, and 95% anchors are fit anchors. The 25% and 75% anchors are held-out
anchors. The five selected frame indices must be distinct or the record is
blocked. These positions are independent of claim intervals and model scores.

The builder binds the three opaque review tokens and exact staged-media SHA-256
values to the private worksheet and custody audit. It writes the existing
`predeclared_html5_clock_pts_anchor_plan` schema expected by the capture-input
converter. Because that schema has no fields for the builder version, PTS
sequence digest, or freeze receipt, preserve the builder source hash, exact
input hashes, plan hash, PyAV/FFmpeg versions, stream information, and review
freeze receipt in a separate private manifest before capture. All such values
are private and must not be published.

For integrity, the builder hashes and decodes the same in-memory media bytes.
Its command output deliberately omits private digests and anchor values.

`T` is the last frame's timestamp, not the container duration or the end of
that frame's display interval. A selected frame time is only a requested
anchor target; it does not prove which compositor frame a browser displayed.
Actual capture, exact screenshot-to-decoder-frame correspondence, independent
evidence review, and clock residual checks remain separate gates. This design
does not provide an uncertainty interval or future clock-error guarantee.
