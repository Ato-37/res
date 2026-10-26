# Cached-generation temporal-key bias adapter (2026-10-21)

## Purpose

The thesis target now emphasized by the user is whether an internal video
intervention improves the temporal grounding accuracy underlying an
explanation. Existing visual-key bias hooks only edit fixed teacher-forced
query rows; they cannot alter autoregressive timestamp generation. Existing
multiquery intervention code also requires a square, uncached mask. This
engineering change adds an absolute-position-aware hook for generation prefill
and cached decode, so a separately frozen question-conditioned selector can
reuse the same temporal visual-key positions as each generated interval token
is predicted.

This adapter does not itself select evidence, generate intervals, or measure
grounding. It is infrastructure for a new exploratory QA temporal-localization
endpoint. NExT-GQA gold intervals are QA time labels and cannot be called
claim-specific support intervals without a valid mapping. No head/layer/alpha
search is reopened.

## Implementation and checks

Added `bias_video_keys_during_generation` and
`install_generation_video_key_bias_hooks` to
`scripts/qwen_visual_attention_boost.py`. For a mask with query length Q and
key length K, the adapter uses offset K−Q, targets query rows whose absolute
position is at or after the prompt's final token (the row predicting generated
token 1), and biases only the supplied absolute visual-key positions. It
supports square prefill and cached Q=1 decode, checks strict causal precedence
and visible zero-valued keys, and leaves prompt-only rows unchanged. It also
supports uncached square generation. An independent code audit identified
that K−Q alone cannot distinguish DynamicCache from static-capacity or rebased
caches. The hook installer was therefore hardened to require a square Q=K
prefill, then Q=1 decode masks with key length increasing by exactly one at
every step across selected layers in ascending order. Static, multi-token,
skipped-length, and cropped/rebased cache patterns fail closed. The lower-level
mask helper still assumes absolute key-axis origin zero and must only be called
through the verified hook for generation experiments.

Added CPU tests for prefill, cached decode, prompt-row preservation, causal
key validation, hook reuse across generation steps, hook cleanup, and failure
on static/non-incremental cache shapes. After the cache-origin audit and
hardening, focused CPU validation passed **24/24** tests across the generation
adapter and existing machine-localized claim-mask runner regressions;
`py_compile` passed. An initial unittest invocation omitted the `scripts`
import path and failed at test discovery before any test ran; rerunning with
`PYTHONPATH=scripts` produced the verified passing count.

## Next action and limitations

Next implement a one-row GPU smoke on real video that derives a frozen temporal
key set from the untouched question-conditioned pass, then compares baseline
and biased interval generation under identical prompt, decoder settings, and
parse rules. Preserve invalid generations in the denominator as zero tIoU.
Use `NExT-GQA` QA interval tIoU only as an exploratory QA-localization proxy;
do not label it generated-claim support or explanation validity. The internal
selector/layers/dose and outcome must be frozen before the smoke, and a matched
control-bias arm is required before any larger pilot. No GPU smoke for this new
adapter has run yet, so generation integration is not yet verified.
