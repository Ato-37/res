# Review interval to Qwen temporal-grid positions (2026-10-18)

## Purpose

Make the mapping from a reviewed source-time interval to complete Qwen visual
key groups testable before scoring. The intended estimand ablates internal
access to groups associated with reviewed evidence and compares it with an
equal-key-budget control fixed without inspecting model outcomes.

## Implementation and verification

`scripts/qwen_review_interval_token_groups.py` validates input token-block
contiguity and the supplied processor-grid dimensions, merge size, temporal
patch size, sampled PTS, source duration, and review interval. It constructs
midpoint (Voronoi) frame-support cells, clipped to `[0, video_duration]`;
each temporal grid group spans the cells for its temporal-patch frames.
Evidence groups are those with positive-duration overlap with the review
interval. The control is a contiguous, disjoint window with the same number of
groups, minimizing duration difference, then center-time distance, then start
index. It reports the largest adjacent PTS gap, first and last clipped edge-cell
durations, and largest temporal-group support duration so sparse sampling is
visible to callers. These are diagnostics only; no gap threshold has been
preregistered. The rule is deterministic and the helper accepts no outcome or
score.

The focused CPU contract tests pass **5/5**, including sparse-PTS coverage and
diagnostic values, single assignment at an exact cell boundary,
duration-over-center control
priority, tie-breaking, and invalid geometry/time/control cases. An independent
audit reviewed the diagnostic formulas and found the implementation correct;
it caught a mistaken test expectation (the final clipped cell is 1.5 seconds),
which was corrected. The final focused interval and internal-ablation tests pass
**10/10**. `py_compile` and `git diff --check` also pass. No GPU or dataset was
used for this mapping test.

## Interpretation and limitations

This is processor-layout geometry, not authenticated evidence mapping. The
midpoint cells are an operational frame-support convention, not proof that a
Qwen temporal patch semantically represents every time inside the cell. It
assumes review timestamps, decoded PTS, and duration share an authenticated
zero-based seconds clock. Sparse samples can create wide edge cells. The
diagnostics expose sparse support but do not establish an acceptable threshold.
Supplied grid and token positions
are not bound to one immutable processor batch; matching total token count does
not authenticate the grid shape, frame order, padding state, processor revision,
or video source. The control can overlap another salient event and is not a
semantically neutral negative control. The hook only blocks direct answer-query
access to selected keys, not evidence propagated through other keys/layers.

Direct faithfulness evidence remains **0/3 independently reviewed, 0/3 scored,
and no real-data ΔF estimate, interval, or test**. This mapping work does not
measure treatment efficacy.

## Next action

Before real scoring, bind PTS, duration, grid, token positions, and frame
sampling metadata to a single immutable processor batch, define a gap/edge
coverage threshold before outcomes, and establish source/browser clock correspondence and
independent claim review. Then run a small internal-ablation wiring smoke. Treat
its output as engineering feasibility only; proceed to efficacy scoring only
after the review and provenance gates pass.
