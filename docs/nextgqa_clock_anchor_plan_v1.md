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
converter, plus a required private sidecar manifest. The sidecar records exact
worksheet/lineage/custody input hashes, builder and plan hashes, PTS-sequence
digests, decoder stream/pixel metadata, and PyAV/FFmpeg library versions. Its
joint-freeze receipt is null until a separate joint freeze occurs. All such
values are private and must not be published. The source hashes cover the
builder, decoder, custody auditor, and private-output policy code.

For integrity, the builder hashes and decodes the same in-memory media bytes.
Its command output deliberately omits private digests and anchor values. Both
outputs must be direct children of `data/private_nextgqa/`; the pair is
installed with exclusive hard links and ordinary write failures roll back any
partial installation. Both files use mode 0600.

The HTML5 observation converter requires this sidecar and checks its exact
digest, candidate status, authorization, plan digest, and row/media bindings
before pairing raw observations with the plan. Its Python API requires the
same arguments, and the calibration binder rejects bundles without the
sidecar digest. It carries that digest through to the calibration record.
These checks establish byte continuity only; they do not authenticate the
source declarations or prove screenshot/frame correspondence.

`T` is the last frame's timestamp, not the container duration or the end of
that frame's display interval. A selected frame time is only a requested
anchor target; it does not prove which compositor frame a browser displayed.
Actual capture, exact screenshot-to-decoder-frame correspondence, independent
evidence review, and clock residual checks remain separate gates. This design
does not provide an uncertainty interval or future clock-error guarantee.
