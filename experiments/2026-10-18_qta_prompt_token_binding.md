# QTA baseline prompt/token binding repair (2026-10-18)

## Problem and scope

The non-authorizing baseline `p/q` candidate builder accepted one caller-
supplied A–E token-ID list shared across records and checked only its equality
with row declarations. It validated `prompt_sha256` syntax but did not
recompute that digest from rendered prompt bytes or derive the label IDs from
the tokenizer. A caller could therefore provide a self-consistent but
incorrect prompt/token mapping. This did not authorize model execution, but it
left the candidate manifest's prompt/continuation binding weaker than the QTA
protocol requirement to resolve all five choices using the pinned tokenizer.

## Change

- Added `scripts/qta_choice_token_binding.py` to derive A–E as single-token
continuations of each exact rendered prompt, require prefix-token preservation
and five distinct IDs, bind the prompt and tokenizer digests, and check that a
processor-supplied label-extended sequence preserves the processor prefix and
appends the derived label ID.
- Updated `scripts/qta_baseline_pq_candidate.py` to take a per-record rendered
prompt and tokenizer, derive bindings internally, recompute each prompt hash,
check that row and scored-option token IDs match the derived A–E IDs, and bind
per-record token-ID and prompt-token binding hashes into its commitment. The
global caller-provided `expected_answer_token_ids` list was removed.
- The candidate remains structural and non-authorizing. It cannot prove the
caller used the frozen prompt renderer, authenticate tokenizer bytes or
  processor execution, reconstruct video-token expansion or prove that the
  row's multimodal `input_ids` came from the same text prompt. Its status is
  explicitly `...text_token_binding_valid`, and
  `processor_prompt_association_verified=false`; the separate processor
  suffix function is only a caller-side check primitive. It also cannot
  authenticate model outputs, establish reviewer identity, or grant a forward.
  Real-data integration remains unimplemented and gated.
- Added synthetic adversarial coverage for self-consistent forged IDs and
prompt hashes, retokenized prefixes, multi-token suffixes, duplicate IDs,
one-label tokenizer changes, wrong processor prefixes, and wrong processor
suffix IDs. Existing selector-runner continuation tests also remain active.

## Verification

- Focused candidate and selector-runner tests: **32/32 passed**.
- Full `scripts` unittest discovery: **1,721/1,721 passed** in 51.1 s.
- `py_compile` passed for the helper, builder, and tests.
- `git diff --check` passed.
- No model, GPU, review HTML/video/QA content, candidate inventory, or
protected output was accessed or changed in this subcycle. No real prompt,
tokenizer output, baseline candidate, or faithfulness score was produced.

The new helper is only exercised with synthetic tokenizers. Its hashes bind
bytes and derived values but do not authenticate their source. The passing
fixture intentionally has text-tokenizer prefix IDs different from its
multimodal processor `input_ids`, demonstrating why the candidate explicitly
leaves that association unverified. A later real runner must recreate the
exact prompt with the frozen renderer and pinned tokenizer, bind the actual
processor output and suffix checks to the same row before any candidate is
accepted, and still pass the human, timing, source-bundle, and independent
score gates.

Source hashes at this checkpoint:

- `scripts/qta_choice_token_binding.py`:
  `45f7cfce20fd0fcb8fc7709d60f2638fe6e098d38f29b581f6a4c082611552ef`
- `scripts/qta_baseline_pq_candidate.py`:
  `e5afeb00bb3bf501b2ef0e90c2bddf4e7a4c364a08713c7a0897b897f4d2f3dd`
- `scripts/smoke_qwen_question_selector_intervention.py`:
  `1b267e9a2f2d1d35d12431a6bc656c8d993231b9e40a0ef3512158d999f24549`
- `scripts/test_qta_baseline_pq_candidate.py`:
  `0dbe622136b8d4ce75e8cd07dca363b718dc21003481685db1b0d3740918a871`

## Research status and next action

This closes one local input-binding gap only. Direct claim-faithfulness remains
0/3 reviewed, 0/3 prediction-link judgments, 0/15 verified browser-clock to
decoder-frame correspondences, and 0/3 scored. The 30-row annotation-proxy
result remains a separate exploratory finite-cohort result with near-zero
mean ΔF and mixed signs; it is not explanation-faithfulness evidence.

The current 77-slot artifact is only a pre-review candidate inventory; it is
not a freeze or a score-stage source closure. Rebuild it immediately before
any review-only freeze as the runbook requires, and bind this helper plus the
complete production scorer at the later score-freeze. Continue with an
independent audit of the candidate builder and source-bundle chain, then
integrate this helper into a real pre-baseline runner only after the separately
assigned reviewers, trust root, witnessed unopened handoff, and timing data
exist. Do not run GPU scoring or treat this synthetic contract as human review.
