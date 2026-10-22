# Synthetic event-video readability smoke (2026-10-19)

## Purpose and scope

This is a pre-outcome model/video-path smoke for whether the pinned
Qwen2.5-VL-7B checkpoint can answer two deliberately simple, generated-video
questions. It is separate from the NExT-GQA direct-faithfulness study and does
not satisfy any of its human, trust, clock/frame, or scoring gates. It is not
an explanation-faithfulness result: this smoke neither generates an
explanation claim nor applies an internal intervention. A two-item result is
too small to validate temporal-event comprehension or establish a readability
rate.

The narrow question is whether the pinned model/processor can handle a paired
input in which a transient red-to-blue event occurs on either the circle or
triangle. The clips contain two unlabelled geometric shapes on a plain
background, 32 frames at 1 fps, with the target turning blue over frames
`[16,20)` and returning to red; both shapes are red in the final frame. The
question explicitly asks which shape briefly turns blue and returns to red.
Both clips use the identical question and identical five answer choices in
identical A–E order; the correct labels are B and D. No text is rendered into
frames. This exact design is frozen in generator v2 before inference.

## Frozen stop rule

Run exactly the two clips once with the pinned checkpoint and BF16/4-bit
inference. Report requested/effective videos, top-choice accuracy, and each
fixed-answer-vs-best-foil log-probability margin. Do not tune prompts, images,
decoding, frame count, or thresholds on these outcomes. A 2/2 result is only a
two-item prompt/processor/model-path smoke; it is not a gate for temporal-event
comprehension or a separate measurement-validation study. A miss is a useful
integration/debugging signal, not evidence that the model cannot understand
temporal events. Any repair requires a new stimulus/prompt version and a new
outcome-blind freeze. Neither outcome says anything about QTA effects,
explanations, or natural video.

## Expected workload and safeguards

- Requested/effective workload: **2 real model forwards**, one per synthetic
  clip, after one processor/model load.
- Input: **32 generated RGB frames per clip**, processed through the pinned
  Qwen video path; no protected or downloaded dataset is read.
- A reproduced CPU processor preflight yielded **4,443 input tokens per
  clip**, `video_grid_thw=[16,26,42]`, and an 82,188,288-byte processed video
  tensor per clip. Both clips produced identical prompt and processor geometry.
- A separate same-stack GPU audit found an earlier 3×32-frame eager run
  reserved 12.44 GiB. Because the current sequence is longer, require at least
  **13.5 GiB free before model load** and recheck for **3 GiB free before each
  forward**. Expected peak reservation is approximately 12.5–13 GiB; run time
  is expected to be under 3 minutes, excluding a
  one-time failure to initialize the local model stack.
- Output: a new local-only mode-0600 aggregate JSON under a unique path. It
  must not contain generated frames, prompts, answer text, or absolute paths.
- The runner records model/checkpoint/source hashes, package versions, random
  seed, exact count, failures, and peak VRAM. The aggregate explicitly marks
  `faithfulness_result=false` and the scope as user-authorized, local-only,
  synthetic baseline QA (not QTA scoring).

## Interpretation and next action

A 2/2 result would only show that this pinned checkpoint emitted the expected
answers for these two fixed synthetic inputs under this processor
configuration. It would not justify a temporal-comprehension claim or validate
a faithfulness measure; a later held-out study would need multiple scene
families, counterbalanced targets, and independent development/confirmation
sets. This smoke does not count toward the NExT-GQA human-review cohort or
establish explanation faithfulness. The NExT-GQA direct status remains claim review **0/3**,
clock/frame correspondence **0/15**, prediction-link review **0/3**, direct
scores **0/3**, with no direct ΔF, interval, or hypothesis test. The separate
direct study remains gated on its authentic human/release inputs.

## Independent pre-outcome design audit and repair (2026-10-19)

Before any model outcome, an independent audit found two confounds in fixture
v1: the second clip changed answer-choice text/label mappings, and the target
color persisted through the final frame, allowing a still-frame shortcut.
Fixture v2 addresses these by holding the full choice list fixed across the
pair and making the target event transient (`[16,20)`), with both shapes red at
the end. The question now explicitly asks about the brief switch and return.
This is still only a two-item integration smoke; even 2/2 does not establish
temporal-event comprehension. No model outcome had been collected as of this
protocol revision.

Independent review of fixture v2 found no blocker for this narrow integration
smoke. The reviewer confirmed that the pair shares prompt/options, the target
event is transient, and both clips end in the same red/red state. The reviewer
also confirmed the key limitation: fixed B/D answer labels and two items cannot
support a readability rate, event-comprehension claim, or temporal-faithfulness
claim. The runner result must retain generator/runner/protocol hashes and
requested/effective forward counts because the source files are not committed.

The actual local Qwen processor preflight was independently repeated before any
model load: **2/2** clips processed, both with 4,443 tokens and identical
`video_grid_thw=[16,26,42]`; **0 forwards**, **0 model weights loaded**. The
current .venv reports Torch `2.11.0+cu128`, CUDA build `12.8`, and GPU
availability is hidden in the normal sandbox. A prior authorized measurement
reported a 15.89 GiB GPU and 12.44 GiB peak reserved VRAM for a closely related
eager-attention workload, which is why the admission floor was raised from 11
to 13.5 GiB free. The final GPU run must check this live and safely write a
failure receipt without loading weights if resources are below the floor.

Human-review handoff was independently rechecked at this checkpoint: 12 local
quick-response JSONs exist, all 12 are invalid for the current v4 artifact,
and the 12 have one unique byte payload. Formal Stage-1 claim review remains
0/3, clock/frame correspondence 0/15, Stage-2 prediction links 0/3, and direct
scores 0/3. These files do not contain a fresh human response; the remaining
quick-check action is for a person to complete the embedded-v4 questions and
save a new response. That response would remain informal only.

## First bounded GPU attempt and recovery (2026-10-19)

The frozen v2 GPU command loaded the pinned 4-bit/BF16 model and completed the
first forward, but safely stopped before the second forward. The private
receipt records **1 attempted / 1 completed** forward, `14.657 GiB` free before
model load, `8.683 GiB` before the first forward, and `1.053 GiB` before the
second guard; host availability remained `11.487 GiB`. Thus no second model
call occurred, and the runner emitted no aggregate answer score or margin.
The GPU returned to idle after the process exited (`0%` utilization,
`15,133 MiB` free). The direct cause of the stop was the measured 1.053-GiB
free value crossing the 3-GiB guard. Source inspection found that the first
call's output and device input references remained live and the unused CUDA
cache was not cleared; this is the likely avoidable contributor. The runner
now deletes per-clip GPU references, synchronizes, and empties the unused cache
before rechecking the unchanged floor. CPU tests verify the
guard/forward/synchronize/cache-clear/guard order. Thresholds and frozen
stimuli remain unchanged. A recovery run of the same pair will therefore
repeat the first call; cumulative GPU forwards across the failed attempt and
recovery will be reported transparently. No score from the failed attempt is
available.

The recovery run completed the exact two-clip workload under the unchanged
guards. Private receipt `data/private_nextgqa/qta_synthetic_event_readability_gpu_20261019_v2_02.json`
is mode `0600` (3,068 bytes). It binds the v2 fixture, runner, protocol,
choice-token binding, QA/video batch builder, processor/tokenizer/runtime, and
the 16 local checkpoint files (16,595,981,281 bytes; checkpoint SHA-256
`b6af8b758e4ff5b03503f3e3354660b883d81727fb04cff572de9510b7f74a93`). The
pinned revision is `cc594898137f460bfe9f0759e9844b3ce807cfb5`; runtime was
PyTorch `2.11.0+cu128`, Transformers `5.18.0`, CUDA `12.8`, RTX 5070 Ti, 4-bit
weights with BF16 compute. Requested/effective videos were **2/2**, attempted
and completed forwards **2/2**, with **0** failures in 34.122 s (model load
24.642 s). Peak allocated/reserved VRAM was **11.377/13.354 GiB**. Both fixed
answers were selected (**2/2**); target-versus-best-foil margins were **3.25**
and **4.50 nats** (mean **3.875**, `n=2`). The earlier failed attempt had one
completed forward whose score was unavailable, so cumulative GPU calls across
both attempts were **3**, of which **2** have scores in the successful receipt.

These observations pass only the two-item synthetic processor/model-path
smoke. They do not establish temporal understanding, a readability rate,
faithfulness, or QTA efficacy; no explanation was generated and no internal
intervention or direct `DeltaF` was measured. The next justified step is the
already-frozen real-video direct branch only after its independent human,
trust/release, and 15/15 clock/frame gates are supplied. Until then, do not
enlarge this synthetic pair or use its two outcomes for tuning. Formal direct
faithfulness remains 0/3 claim reviews, 0/15 clock/frame correspondences,
0/3 prediction-link reviews, and 0/3 scores.
