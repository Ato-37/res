# Prespecified plan: multi-query (all post-video rows) suppression control (written 2026-10-23, before any run)

## Why
Suppressing the annotated frames' keys at the final query row failed to move the answer margin in layers 5-16 and in all 28 layers
(both controls failed). Hypothesis: the final token reads video information through the residual streams of the other post-video
text rows (question, options, prompt tokens), which a last-row edit leaves untouched. The matching lever is to edit every
post-video query row (positions after the last video token through the final position), as in multi-query steering methods.

## Design (fixed now)
- New option `--query-scope post_video` on the base runner (default `last` keeps all earlier behavior). Query rows =
  {last_video_token_index + 1, ..., seq_len - 1}. Bias is added to (rows x annotation key cells) in the attention mask; the
  audit verifies that exactly those cells changed in every layer, per layer, per condition (target and sham use the same rows).
- Same locked N=149 cohort and lock; new preflight (the base runner file changes); model, fp32 readout, sham rule, seed, scoring
  unchanged. Layers 1-28 (maximum leverage for the control).
- Step 1 (control, one run): bias -ln256. PASS if annotation-minus-sham mean is negative, two-sided t-test p < 0.05 and
  |mean| >= 0.025 nat (single test).
- Step 2 (only if Step 1 passes): positive bias +ln4 and +ln16, same scope/layers, Holm over 2 doses, TOST at +/-0.025.
- If Step 1 fails, the attention-key steering family tested here (last row, all post-video rows; layers 5-16 and all) has no
  demonstrated leverage on this endpoint and the search stops there; no further scope/layer/dose variants without a new rationale.
- Not pooled with earlier runs; same cohort reused; delta_min=0.025 is my choice. Scope: answer-margin dependence on the QA time
  annotation, not explanation faithfulness.
