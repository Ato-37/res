# Question-conditioned temporal selector feasibility (2026-10-18)

## Purpose

The user's selected target is dependence of a prediction on video evidence
indicated by its explanation, with internal intervention as the candidate way
to improve that dependence. The treatment is not frozen. An independent design
audit proposed selecting temporal groups from an untreated baseline's
question-token attention, while keeping the reviewed claim interval evaluation
only. This entry tests whether the prompt-token-to-processor mapping and the
group-attention aggregation can be implemented deterministically; it does not
freeze that proposal or evaluate an outcome.

## Implementation and verification

`scripts/qwen_question_temporal_attention_selector.py` provides three helpers:

- Map the unique question text's tokenizer offsets in a rendered video-first
  prompt into processed input IDs. It verifies that the complete processed ID
  vector is exactly the rendered IDs with one video-pad placeholder expanded
  into the contiguous visual-token run; any other processor rewrite fails.
- Average attention mass over a declared set of baseline layers, heads, and
  question-token query positions for each complete, equal-sized, contiguous
  temporal group. It rejects malformed/overlapping groups, invalid weights,
  and noncausal positions.
- Select a requested number of groups by descending score, with earliest group
  index breaking ties.

CPU contract tests pass **4/4** for placeholder expansion, ambiguous text and
rewrites, attention aggregation, deterministic ties, overlap, unequal group
sizes, and noncausal positions. `py_compile` and `git diff --check` pass.

After the first full repository test run exposed a missing engineering-loader
classification for this synthetic-only runner, the classification was added and
the complete suite passed **1,442/1,442** (`unittest discover`, 29.8 s). This is
test coverage, not additional faithfulness evidence.

The mapping helper was then exercised against the pinned local Qwen2.5-VL
processor (`cc594898137f460bfe9f0759e9844b3ce807cfb5`) on one synthetic
32-frame clip using the repository's actual video-first prompt builder and
processor path. The rendered input sequence was 68 tokens; the processed
sequence was 4,163 tokens. The exact question occupied processed positions
4,112–4,120 (9 tokens), after the 4,096 video-token run at positions 15–4,110.
The grid had 16 temporal groups of 256 visual keys each. The reconstructed
expanded ID sequence matched the processor output exactly. This was CPU-only,
synthetic input, and did not load model weights or access real data.

The pinned Qwen eager forward was then instrumented at layers 5–16. Eager
attention transiently materializes each layer's full attention matrix; the hook
immediately reduces its selected question-query rows to group scores and
retains only 16 scalar group scores per layer, not the full matrices.
The first attempt completed the forward/capture path but failed during cleanup
because the runner omitted its hook-removal import; it wrote no result artifact
and is not counted as a successful run. Repeat 0710 completed **1/1** with
finite logits/group scores. That artifact is superseded because the runner's
source-hash inventory was then expanded. Repeat 0715 also completed 1/1 but is
superseded because processor/runtime metadata were then added to the runner.
The final source-matched GPU smoke completed **1/1** baseline forwards with
finite logits and a finite score for every group. It used 9 mapped question
tokens, 16 temporal groups, and 256 visual keys per group. The attention mass
summed to 0.203 across visual groups; the smoke-only top-3 were `[0,5,4]`. This
is an uninterpreted synthetic selector output; `K=3` is not a treatment choice.
Peak allocated/reserved VRAM was 10.72/12.44 GiB; runtime 13.94 s on the RTX
5070 Ti, with bf16 / 4-bit bf16 compute and eager attention. Python 3.12.14,
NumPy 2.5.2, Pillow 12.3.0, Transformers 5.18.0, and qwen-vl-utils 0.0.14 are
recorded. The processor fingerprint is
`a8aa40930855b070f83b7d559409013d4d0f2f8bcf8c08300c92d826821015dc`. The
artifact records 19 local/installed source hashes; independent verification
matched all 19 to the files used. Result:
`results/synthetic_qwen_question_temporal_selector_20261018_0720.json`,
SHA-256 `b5b8ecf2cda6d896012fe9428142cd1ff002949d357b03f31e7037aea40344fc`.
Runner SHA-256:
`37fc0fd7a05887f12ccf256758e6b102901f4f9137911bba9e78532275860be2`.
The locally cached checkpoint weights themselves were not byte-hashed.

## Interpretation and limitations

The successful mapping establishes a concrete route to the question-token
query span for this prompt template, pinned processor revision, one video, and
unbatched sequence. It does not establish correctness for other prompt forms,
padding/batches, future processor implementations, or real-data sampling.
Attention maps are model-internal allocation signals; aggregating them does not
establish semantic evidence relevance. This selector uses the question rather
than the explanation's reviewed claim, so claim evidence remains an
evaluation-only interval and positive results could still reflect generic
question-conditioned salience or attention bias. The attention hook and
aggregation were integrated into one synthetic model forward, but no treatment
or sham has been run. For question-token mapping, a token whose character
offset overlaps the exact question span is included; this allows normal
tokenizer boundary merges but can include adjacent characters in such a merged
token. The tested synthetic English prompt had no visible boundary ambiguity.

Direct faithfulness evidence remains **0/3 independently reviewed, 0/3 scored**,
with no real-data `ΔF`, confidence interval, or test. Rights status remains
unverified; all protected material stays local.

## Next action

Before any real-data scoring, freeze the treatment, dose, layers, group count,
sham, fixed-prediction margin, and analysis; complete independent claim/time
review and browser-to-decoder correspondence. The synthetic attention output
must not be used to choose these parameters. Run no duplicate GPU smoke unless
an integration change presents a distinct failure mode to test.
