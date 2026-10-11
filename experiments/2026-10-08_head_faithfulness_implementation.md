# Gradient×activation head faithfulness implementation

## Purpose

Implement the approved primary test: test whether Qwen2.5-VL's top-four gradient×activation heads causally support its own answer margin more than five layer-count- and activation-norm-matched control groups. This replaces neither the research question nor the already documented input-mask sensitivity work.

## Setup and definitions

- Dataset: one multiple-choice QA per unique NExT-GQA video; five A–E choices.
- Score: `log p(baseline predicted option) - log p(strongest foil)`. The predicted option is fixed from the full-input baseline; the strongest foil is recomputed for each intervention.
- Attribution: `abs(sum(z_h * dmargin/dz_h))` at the final prompt token, with `z_h` captured at the input to each layer's `o_proj`.
- Target: top four heads over the requested layers (default all layers), attenuated to `alpha=0.5` at the final prompt position.
- Controls: five disjoint head groups; each has exactly the target group's layer histogram and uses same-layer heads selected by nearest log-ratio of activation L2 norm.
- Per-video primary effect: `target_margin_drop - mean(control_margin_drops)`; positive values favor attribution-selected heads.
- Primary boundary: `theta=0.25` nat. The registered success criterion is the two-sided 95% Student-t interval lower bound above 0.25.

## Implementation

- Added `scripts/faithfulness_utils.py` for deterministic top-K selection, the specified gradient×activation score, exact layer-count/norm-matched disjoint control construction, and primary/robustness summaries.
- Added `scripts/head_faithfulness_test.py` for Qwen2.5-VL attention-output hooks, gradient collection, frozen-answer margin scoring, final-prompt-token attenuation, anonymous per-sample result records, and failure counts.
- Added `scripts/run_head_faithfulness_and_publish.py`. It checks Git identity and tracked-worktree cleanliness before inference, refuses overwrite, and publishes only a privacy-scanned derived JSON and a manifest hash/configuration. It does not stage the raw sample manifest.
- Added SciPy to `scripts/requirements-spike.txt` for the preregistered Student-t interval and test.
- Results include per-sample elapsed time and the explicit primary decision flag (`95% t-CI lower bound > 0.25 nat`), plus baseline accuracy as descriptive-only secondary information.
- Updated the research overview and plan with implementation status and publication privacy rules.

## Verification

- CPU-only helper tests: 6 passed (selection tie-breaking, score definition, frozen prediction/recomputed foil, matched layer histogram, disjointness, infeasible controls, Student-t interval, and sign-flip bounds).
- Publisher privacy tests: 2 passed (anonymous derived output allowed; raw question, QID, and path fields rejected).
- Syntax compilation: passed for all five new scripts/tests.
- The local macOS environment has SciPy but does not have PyTorch, Transformers, or OpenCV. Therefore no model/hook test or GPU inference was run here.
- At the initial implementation checkpoint, no smoke, pilot, or confirmatory result existed. The subsequently published smoke is recorded below; the code change itself is not evidence for faithfulness.

## Limitations and remaining work

- The temporal-attention-to-frame mapping and model-weight-randomization sanity check are not yet implemented. Mapping is required before temporal-validity analysis; weight randomization is required before independent confirmation. The primary-only development pilot is permitted after a fresh validation engineering smoke because it estimates runtime and variance only, and cannot support or tune a confirmatory claim.
- Hook behavior, 4-bit gradient support, actual GPU memory use, and runtime remain unverified until the Ubuntu 5070 Ti smoke run.
- The control matching uses same-layer output L2 norm, not semantic equivalence; this cannot eliminate all distributional differences caused by internal intervention.
- A–E choice-letter scores are specific to this fixed prompt and do not equal free-generation accuracy.

## Next action

On Ubuntu, run:

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/build_samples.py \
  --test-csv NExT-GQA/datasets/nextgqa/test.csv \
  --gsub-json NExT-GQA/datasets/nextgqa/gsub_test.json \
  --map-json NExT-GQA/datasets/nextgqa/map_vid_vidorID.json \
  --video-root ~/nextgqa_sample/NExTVideo \
  --zip-path ~/NExTVideo.zip \
  --n 3 --seed 20261008 --unique-videos \
  --exclude data/nextgqa_samples*.json \
  --out ~/nextgqa_sample/faithfulness_smoke_samples_20261008.json &&
python scripts/run_head_faithfulness_and_publish.py \
  --samples-file ~/nextgqa_sample/faithfulness_smoke_samples_20261008.json \
  --n-samples 3 --layers all --quant 4bit --seed 20261008 \
  --output results/faithfulness_head_smoke_20261008.json
```

The sample manifest remains outside Git; the wrapper should publish only the anonymous result and configuration/hash manifest. The smoke is an engineering check only. If it passes, analyze the published artifact and add the still-missing temporal/sanity-check implementation before the validation pilot. If hook gradients fail or memory/runtime is impractical, stop before scaling up. This command has not been run by Codex.

## 2026-10-08: OOM failure and mitigation pass

### Observed run

The user ran the 3-sample `faithfulness_smoke_samples_20261008.json` smoke with all layers, 4-bit quantization, 16 frames, and Qwen2.5-VL-7B-Instruct on the remote GPU. All 3 samples failed at `gradient_attribution` with `OutOfMemoryError`; the script then exited because no samples completed. The publisher subprocess therefore did not publish an artifact. No margin, attribution ranking, intervention effect, confidence interval, or faithfulness conclusion exists from this run. This is a compute failure, not a negative or positive experimental result.

### Changes prepared

- Pass `cap_pixels_per_frame=True` into the video processor while retaining the registered 16-frame sampling. The warning in the user log says the uncapped path can use substantially more tokens; official Qwen2.5-VL docs also recommend restricting per-frame resolution or total video pixels for available GPU memory ([Transformers Qwen2.5-VL docs](https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/qwen2_5_vl), [Qwen video processing source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)). Record the setting in both result and run manifests.
- Freeze model weights because the requested derivative is with respect to captured activations, not model parameters.
- Enable non-reentrant activation checkpointing for the gradient pass. This trades additional recomputation time for lower saved-activation memory; it is supported by the Qwen2.5-VL architecture, but this repository's hook interaction still requires the GPU smoke to verify correctness ([Transformers gradient checkpointing docs](https://huggingface.co/docs/transformers/main/grad_checkpointing), [Qwen2.5-VL model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)).
- Use `attn_implementation='sdpa'`, not `eager`: this test hooks the concatenated head activations at `o_proj` and does not consume attention matrices. Hugging Face documents that eager materializes the full attention matrix while optimized backends such as SDPA reduce memory traffic ([Attention backends](https://huggingface.co/docs/transformers/attention_interface)). This backend is now a fixed run condition and is recorded in outputs. Numerical parity and hook compatibility remain to be checked on the target machine.
- Stop after the first OOM, classify remaining requested samples as not run, and write an anonymous failure-only artifact. The publication wrapper accepts a fully accounted result with zero effective samples so future resource failures can be tracked without pretending the experiment succeeded.

### Literature/Hugging Face scan (2026-10-08)

- ICLR 2025's modular attribution/intervention work identifies attention modules/heads associated with hallucination in image LVLMs, but its task is hallucination mitigation on image-centric models; it does not validate this project's video-QA answer-margin estimand or its matched-head controls ([ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html)).
- An AAAI 2026 causal-tracing paper studies cross-modal object representations across components/layers, a nearby mechanistic-intervention direction, but not the same selected-head-vs-matched-control test ([AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/40431)).
- A 2025 preprint on faithfulness of visual thinking intervenes on visual versus textual reasoning traces, but studies generated multimodal CoT rather than head attribution ([arXiv:2510.23482](https://arxiv.org/abs/2510.23482)).
- These works confirm that causal intervention and multimodal faithfulness are active topics, not that the present protocol is novel or its effect established. The project-specific contribution still needs to be stated against a fuller related-work review.

### Verification and next action

CPU helper tests: 6 passed; publisher/privacy/completeness tests: 4 passed; Python syntax compilation: 5 files passed; `git diff --check`: passed. This macOS environment has no PyTorch/Transformers/OpenCV, so these checks cannot validate SDPA/checkpointed activation hooks or estimate GPU memory. Rerun the existing three-video smoke with a new output filename (do not regenerate or change the manifest):

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_head_faithfulness_and_publish.py \
  --samples-file ~/nextgqa_sample/faithfulness_smoke_samples_20261008.json \
  --n-samples 3 --layers all --quant 4bit --seed 20261008 \
  --output results/faithfulness_head_smoke_ckpt_20261008.json
```

This remains an engineering smoke only. If OOM persists or checkpoint hooks fail, do not start the 30-video pilot; revise memory/input settings or the gradient method first. The cap is an implementation constraint now fixed for this run family, and any later change to frame count or pixel cap must be logged as a protocol deviation rather than silently compared with uncapped results.

## Published smoke result: 2026-10-08

The result and run manifest are present in commit `3d5784e` (`Publish head faithfulness experiment`). The later command that named the same output was correctly refused by the no-overwrite guard; it did not launch a second GPU run. The result artifact passed local integrity checks: requested/effective/failure counts reconcile, record keys are unique, and the recorded SDPA, checkpointing, and pixel-cap settings match the expected retry configuration.

### Observed results

- Configuration: Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit, bfloat16, SDPA, non-reentrant gradient checkpointing, frozen weights, `cap_pixels_per_frame=True`, 16 frames, all 28 layers, top K=4, alpha=0.5, five matched-control groups, seed 20261008. Hardware: RTX 5070 Ti; Torch 2.11.0+cu128; Transformers 5.17.0.
- Coverage: `n_requested=3`, `n_effective=3`, failures=0. The experiment body took 30.58 seconds (model loading occurs before its run timer, so this is not end-to-end wall time). The run manifest records sample-manifest SHA-256 `5904e21e89a2bc64dcb92f75ed16353384dc25f4876c28d0bdec37adb0f776b1`.
- Per-sample selective margin drops `d_i = target drop - mean matched-control drop` were `+0.150`, `+0.275`, and `-0.050` nat. Their mean is `+0.125` nat (SD `0.164`); the target and average control drops were both `+0.125` and `0.000` nat, respectively. Two of three baseline answers were correct; the primary includes all three, as registered.
- The artifact reports a two-sided 95% t interval `[-0.282, +0.532]` nat, test against `theta=0.25` of `p=0.317`, pointwise bootstrap interval `[-0.050, +0.275]`, and sign-flip `p=0.499`. The interval does not clear the registered +0.25-nat boundary; `primary_threshold_supported=false`.

### Interpretation and limits

This is a successful engineering smoke: all three samples completed attribution and intervention without OOM under the new memory settings. Its estimated effect is imprecise and does not satisfy the preregistered practical threshold. With only three smoke samples, none of these inferential quantities is confirmatory, and they must not be used as evidence for or against faithfulness. The sampled top-four sets vary across the three examples, which likewise does not establish a stable head identity. No multiplicity adjustment is relevant to this single smoke contrast, but the small, non-confirmatory sample is the overriding limitation.

Do not rerun to the same output path: the artifact already exists and is published. The initial OOM-only attempt is superseded as an execution-status report, but remains documented as the motivation for the memory changes.

## Validation pilot runner and protocol clarification (2026-10-08)

### Rationale and scope

The prior smoke used three videos from the earlier test-side manifest. It established that the optimized GPU path can complete, but did not measure 30-video runtime or provide a useful variance estimate. The repository has no validation manifest checked in; however, the official NExT-GQA distribution provides `val.csv` and its matching `gsub_val.json` ([dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)). The new runner builds manifests from those inputs on Ubuntu and never substitutes test data for missing validation files.

`scripts/run_faithfulness_validation_pilot.py` now performs, in order:

1. Preflight required validation files, Git identity, clean tracked/index state, and output collisions.
2. Build three readable validation videos, excluding all previously tracked sample manifests; run the established attribution/intervention path and publish its anonymous result.
3. Stop unless all three inference samples complete with zero failures.
4. Build 30 additional validation videos, excluding both the historical manifests and the just-used smoke videos; run the same fixed all-layer/top-4/alpha-0.5/five-control analysis and publish its anonymous result.
5. Stop rather than silently replace any sample or result after an inference failure.

Only derived result JSON and a run manifest are published. Both raw validation sample manifests remain under `~/nextgqa_sample/`, outside Git. The established publisher performs scoped commit/push separately for smoke and pilot, so a later pilot failure does not erase the successful smoke record.

### Protocol clarification

The 30-video validation pilot is now explicitly limited to engineering, runtime, and paired-difference variance calibration. It is not a hypothesis test and cannot be cited as evidence for faithfulness or used to retune K, alpha, score, or the control construction. Requiring a model-weight-randomization check before this non-confirmatory calibration run would delay an otherwise useful estimate without changing the locked method; that check remains mandatory before the independent confirmation. Temporal frame-token mapping remains mandatory before any temporal-grounding analysis, but is not part of this primary-only pilot. This scope clarification is documented in the research plan and overview.

### Verification and expected work at implementation time

CPU verification: the new runner and tests compile; 13 tests pass across the pilot completion guards, attribution/matched-control helpers, and publication privacy/completeness checks; `git diff --check` passes. No validation GPU job has been run by Codex. The previous observed GPU timing was 30.58 seconds of experiment body for three samples, excluding model loading. A 30-video body might therefore take roughly 5.1 minutes at that rate, plus model loading and video variability; treat this as a rough planning estimate, not a runtime guarantee. The two-stage command runs the GPU model twice, so total time should be budgeted as roughly 6–10 minutes if throughput remains similar. The pilot evaluates 30 new videos, each with a gradient attribution pass plus one target and five matched-control forward interventions.

### Ubuntu command used

The user ran the following after the runner was pushed. It built and published the three-video validation smoke, then proceeded to the disjoint 30-video validation pilot because all three smoke samples completed. The runner refuses to overwrite its dated outputs.

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_faithfulness_validation_pilot.py \
  --val-csv NExT-GQA/datasets/nextgqa/val.csv \
  --gsub-val-json NExT-GQA/datasets/nextgqa/gsub_val.json \
  --map-json NExT-GQA/datasets/nextgqa/map_vid_vidorID.json \
  --video-root ~/nextgqa_sample/NExTVideo \
  --zip-path ~/NExTVideo.zip \
  --smoke-n 3 --pilot-n 30 --seed 20261008
```

The official dataset README documents the validation CSV and matching grounding annotations. The published run manifests confirm 3/3 smoke and 30/30 pilot with zero inference failures. The detailed result analysis follows.

## Published validation smoke and development pilot (2026-10-08)

### Provenance and configuration

- Live GitHub `main` was checked with `git ls-remote` and pointed to `d1f7f515b35bbd08c65a5fc9410541195eefb73a`. The smoke artifact was published in `83a588d`; the pilot artifact followed in `d1f7f51`. Both run manifests and result JSONs were fast-forwarded locally and validated.
- The smoke requested/effective/failures were 3/3/0; the pilot was 30/30/0. Each artifact has unique ordinal sample keys; the pilot has 30 unique records. The runner selects from official `val.csv` plus `gsub_val.json`, excludes historical manifests, and excludes smoke videos from the pilot manifest. Raw manifests were not published; their SHA-256 hashes are in the run manifests.
- Both runs used Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb`, RTX 5070 Ti, CUDA device, Torch 2.11.0+cu128, Transformers 5.17.0, OpenCV 5.0.0, 4-bit/bfloat16, SDPA, non-reentrant checkpointing, 16 frames, all 28 layers, top-4, alpha=0.5, and five matched controls. Seeds were 20261008 (smoke) and 20261009 (pilot).
- Model-body elapsed time was 27.43 s for smoke and 249.76 s (4 min 10 s) for the 30-video pilot; model loading is outside the script's sample-loop timer.

### Observed results

| Run | n | Mean target drop | Mean control drop | Mean selective `d_i` | SD | 95% t CI | two-sided p vs 0.25 | threshold flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| validation smoke | 3 | +0.1250 | +0.0167 | +0.1083 | 0.1507 | [−0.2660, +0.4827] | 0.2450 | false |
| validation pilot | 30 | −0.0083 | +0.0142 | −0.0225 | 0.3774 | [−0.1634, +0.1184] | 0.000452 | false |

For the pilot, the one-sample t test is against the preregistered boundary `theta=+0.25 nat`; the sign-flip robustness p-value against the same boundary was 0.000100. These p-values are descriptive for a development pilot and are not confirmatory. The t interval is wholly below +0.25 nat, so the predeclared success rule (lower CI bound > +0.25) is plainly not met. The pilot mean is near zero and slightly negative. Although 21/30 individual contrasts were positive and the median was +0.05 nat, the largest negative contrast was −1.275 nat; leave-one-out means ranged only from −0.0474 to +0.0207 nat. Removing that extreme observation does not turn the average into a practically large positive effect. Baseline choice accuracy was 26/30 (descriptive only).

An independent CPU analyzer at `scripts/analyze_faithfulness_pilot.py` re-computes `target_drop - control_drop` record by record, checks published mean/SD and completeness, and reports the leave-one-out range. Its tests cover accounting errors, contrast mismatch, and duplicate keys; 17 related CPU tests passed, Python compilation and `git diff --check` passed.

### Interpretation, limits, and decision

The measured contrast does not support proceeding directly to the planned independent confirmation under the current method. The interval is not merely too wide to establish the +0.25-nat target: its upper bound is also below that target in this validation pilot. Because the 30 videos were designated for development/calibration, this is not a confirmatory rejection and does not prove the effect is zero outside this cohort. It is nevertheless a clear reason not to spend the planned multi-hour confirmation budget without first checking whether the gradient×activation explainer depends on learned weights and auditing the measurement pipeline. Do not search K, alpha, subgroups, or alternative scores on this same pilot to manufacture a more favorable result.

The model-weight-randomization sanity check remains the next methodological gate before an independent confirmation. If it shows that head rankings barely change after appropriate weight randomization, stop this explanation method rather than scaling it. Temporal frame-token mapping remains a separate prerequisite only for temporal-validity analysis; this pilot contains no temporal-attention result. The internal attenuation can also move activations off-distribution, matched controls do not eliminate that concern, and A–E choice margins are not free-generation accuracy.

## Limited A–E output-readout randomization diagnostic (2026-10-08; implementation)

### Why this diagnostic, and what it changes

The 30-video validation pilot did not support scaling directly to independent confirmation under the current attribution/intervention protocol. The next action is a small, non-confirmatory check of whether the gradient×activation head ranking depends on learned output-readout parameters. In each replicate, the script replaces only the five final `lm_head` rows for choice tokens A–E with seeded random vectors drawn using the model configuration's `initializer_range`; all transformer layers and all other vocabulary rows remain learned. It recomputes all 28×28=784 head scores and reports Spearman rank correlation and top-4 overlap against the trained-readout ranking. The Qwen2.5-VL model configuration states `tie_word_embeddings=false`, and the script refuses to run if the loaded configuration says otherwise ([official model config](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)).

This is deliberately a limited output-readout perturbation, not a full layer-wise Adebayo-style randomization test, not a causal test, and not a faithfulness effect estimate. The random row scale may differ from the learned row distribution; additionally, changing only five answer-token rows leaves all hidden representations and the rest of the vocabulary readout intact. A large ranking change is evidence that this diagnostic is sensitive to the learned choice readout, but does not establish faithful explanations. A stable ranking is a warning to investigate, not by itself a rejection criterion. A separate layer-wise randomization check remains required before independent confirmation.

### Execution plan and publication

- Use the existing three-video validation smoke manifest outside Git; do not rebuild it or publish sample questions, paths, IDs, or video data.
- First run a one-sample/one-randomization engineering smoke. The publisher commits only the derived result JSON and a run manifest containing the sample-manifest SHA-256. `--require-success` makes the command return nonzero after publishing any failure, so `&&` prevents the follow-up run from proceeding.
- If the smoke succeeds, run the same three development videos with five independent seeded readout randomizations per sample. This remains a diagnostic on the validation development cohort, not an independent confirmation or inferential result. The smoke sample is included again in this diagnostic; no p-value or confidence interval is used to claim generalization.
- Expected workload: one GPU model load for smoke and another for the three-sample diagnostic; each sample requires one trained-readout gradient pass plus one gradient pass per randomization. Allow several minutes in addition to model download/loading; actual timing will be recorded, not assumed.
- The wrapper refuses existing result paths, validates requested/effective/failure accounting and anonymous fields, stages only the result and manifest, then commits/pushes. A published failure remains visible and stops chained work.

### Published results and integrity analysis

- GitHub `main` was fetched live at `f5546ca62d0a28a3203d65d5d45f323064f24beb`. The one-sample smoke and its manifest were published in `ea2b235`; the three-sample diagnostic and manifest were published in `f5546ca`. The two manifests have the same sample-manifest SHA-256 (`c60f9b9a...5834b8a`), so the smoke is a subset of the three-sample cohort, not an independent replication.
- Smoke: 1/1 completed, 0 failures, 1 randomization. Rank Spearman ρ=0.4865, top-4 overlap 0/4, and the selected answer changed. This establishes that the path ran on the target GPU, not that the attribution method is faithful.
- Three-sample diagnostic: 3/3 completed, 0 failures, five readout randomizations per sample (15 draws total). Per-sample mean Spearman correlations were 0.4514, 0.4834, and 0.4641; their average was 0.4663. Top-4 overlap fractions were 0.10, 0.10, and 0, averaging 0.0667 (3 overlapping head slots out of 60 possible). The randomized readout changed the predicted answer in 12/15 draws (80%). The measured GPU sample-loop runtime was 40.23 seconds, excluding model initialization/loading.
- Both used Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb`, RTX 5070 Ti, Torch 2.11.0+cu128, Transformers 5.17.0, 4-bit/bfloat16, SDPA, non-reentrant checkpointing, and 16 sampled frames. The three-sample diagnostic used seed 20261011.
- `scripts/analyze_faithfulness_readout_randomization.py` independently recomputes replicate/sample aggregates, prediction-change rate, accounting, and run-manifest agreement. Its tests reject count, aggregate, and manifest mismatches; CPU checks for the analyzer, randomization helpers, and publisher passed (11 tests). Python compilation and `git diff --check` passed.

### Interpretation, limits, and next action

The combination of moderate rank correlation (about 0.47) and very low top-4 overlap (about 0.07) is mixed: some ordering structure remains while exact candidate-head identity is unstable under this intervention. The selected answer changed in most randomized draws. These are descriptive observations from only three validation-development videos; the 15 randomizations are repeated perturbations nested within those three examples, not 15 independent videos. No confidence interval or p-value is warranted.

The test changed only five A–E output-readout rows and left every transformer layer intact. Random vectors used `initializer_range=0.02`, which is not guaranteed to match the trained readout rows' scale/distribution; the frequent answer changes mean scale effects may contribute to rank changes. This result therefore does not establish that the explanations depend on learned transformer computation, and it is not a causal-faithfulness result. The next methodological gate remains a full layer-wise randomization check, with an explicit design for the 4-bit model before GPU execution. Independent confirmation and temporal-grounding claims remain on hold.

## Cumulative decoder-layer randomization implementation (2026-10-08; GPU validation pending)

### Purpose and protocol

The readout-only diagnostic leaves all transformer blocks trained, so it cannot resolve whether the 784-head ranking reflects learned internal computation. Implemented `scripts/head_faithfulness_layer_randomization.py` to randomize complete language decoder blocks in cumulative last-to-first order (layer 27, then 26, …, layer 0). For each cumulative depth, it compares the 784 gradient×activation head scores with the trained-model baseline using Spearman rank correlation, top-4 overlap, and answer-choice change. These are sanity-check outcomes, not causal faithfulness effects.

Each affected linear weight is sampled from a zero-mean Gaussian with the model text config's `initializer_range`, then encoded with the same blocksize/compression/quant-type metadata as that loaded 4-bit layer. Linear biases are reset to zero and RMS/LayerNorm scales to one (biases to zero), matching standard module initialization conventions. The script randomizes in place and processes all samples at each depth before advancing; this avoids retaining a second copy of all original quantized decoder weights on the GPU. The same seeded layer perturbation is therefore shared across the sample cohort at each cumulative depth, allowing paired descriptive comparisons across videos. It records the exact layer order and reinitialization counts. This uses the supported bitsandbytes `Linear4bit`/`Params4bit` model and blockwise `quantize_4bit` interface ([module implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/modules.py), [quantization function](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)); actual compatibility remains to be established on the installed Ubuntu version.

### Guarded execution

- Run a one-video, one-layer smoke first (`max-randomized-layers=1`, which randomizes final decoder layer 27). The publishing wrapper validates anonymous fields, sample accounting, depth summaries, and commits only the result plus a manifest with the sample SHA-256. `--require-success` publishes a failure artifact but exits nonzero so the chained full run will not proceed.
- Only after the smoke has zero failures, run the three existing validation-development videos across all 28 layers. This is a development diagnostic, not an independent confirmation. The same videos are reused from the previous diagnostic, so no generalization claim follows.
- Work estimate from the published readout run: 18 gradient-attribution passes took 40.23 s of sample-loop time for three examples. The full layer test requires 3 baseline + 84 randomized attribution passes, plus re-quantization overhead for each decoder layer and model load. Budget roughly 4–8 minutes, not a guarantee; smoke measures actual compatibility first.
- The implementation is untested on the remote GPU. In particular, the in-place `bitsandbytes` 4-bit re-quantization, cumulative perturbation behavior, gradient-hook compatibility, and VRAM use require the one-layer smoke. Do not run the 28-layer job if the smoke has any failure or non-finite output.

### CPU checks and next action

CPU tests cover cumulative depth aggregation, last-to-first order, sample accounting, privacy fields, and failure conditions. They cannot validate the bitsandbytes kernels. Run the guarded sequence below; the full command runs only if the one-layer smoke has zero failures. Each publisher writes only the derived JSON and SHA-256 run manifest to GitHub. After publication, confirm that each completed sample has all 28 cumulative conditions before interpreting the rank trajectory. No statistical p-values or confidence intervals are planned for these three development videos.

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_head_faithfulness_layer_randomization_and_publish.py \
  --samples-file ~/nextgqa_sample/faithfulness_validation_smoke_20261008.json \
  --n-samples 1 --max-randomized-layers 1 --seed 20261012 \
  --require-success \
  --output results/faithfulness_layer_randomization_smoke_20261008.json &&
python scripts/run_head_faithfulness_layer_randomization_and_publish.py \
  --samples-file ~/nextgqa_sample/faithfulness_validation_smoke_20261008.json \
  --n-samples 3 --seed 20261013 --require-success \
  --output results/faithfulness_layer_randomization_all_layers_20261008.json
```

### Published GPU result and independent CPU verification (2026-10-08)

GitHub `main` contains both the one-layer smoke and all-layer artifact plus their run manifests (commits `e11fdbe` and `8804861`). The smoke completed 1/1 with no failure. The all-layer run completed 3/3 with no failure, tested all 28 decoder layers in the documented reverse order (27→0), and recomputed the ranking over 784 heads at each of 28 cumulative depths. The manifest matches the result for model, seed, quantization, sample counts, layer order, and sample-manifest SHA-256 `c60f9b9a…5834b8a`.

An independent CPU analyzer (`scripts/analyze_faithfulness_layer_randomization.py`) validates per-sample conditions, recomputes depth aggregates, checks top-4 count/fraction consistency, and compares the result to its manifest. It reports mean Spearman rho 0.627 after randomizing only layer 27, 0.127 after 14 cumulative layers, 0.087 after 16, first nonpositive at depth 18, and −0.286 after all 28 layers. Mean top-4 overlap is 0.167 at depth 1 and first reaches zero at depth 7; at depth 28 it is zero for all three videos, and the answer choice differs from baseline for all three. Sample-loop runtime was 180.95 seconds, excluding model loading.

Interpretation is restricted to an exploratory sanity check: this perturbation progressively replaces trained decoder parameters with newly initialized and requantized parameters, in one fixed last-to-first order. It shows that the ranking and predictions change substantially as the trained network is disrupted; it does **not** isolate any single layer, establish that the ranking is a valid explanation, or test causal faithfulness. The sample is only three reused validation-development videos; the 28 depths are repeated conditions, not independent observations. No inferential test or confidence interval is appropriate. The primary faithfulness question remains unresolved.

Following the current [research redesign protocol](../docs/research_redesign_protocol.md), do not launch another GPU search from this result. The next gate is a CPU design review that fixes one primary estimand, including the model-selected explanation target, target intervention, prediction- and perturbation-matched control, success criterion, independent confirmation split, and multiplicity policy. The new analyzer was checked against the published JSON and manifest; its CPU tests pass. The analyzer can be rerun with:

```bash
python scripts/analyze_faithfulness_layer_randomization.py \
  --input results/faithfulness_layer_randomization_all_layers_20261008.json \
  --manifest results/faithfulness_layer_randomization_all_layers_20261008_run_manifest.json
```

The follow-up CPU design review is recorded in [`2026-10-08_primary_estimand_decision.md`](2026-10-08_primary_estimand_decision.md). It retains the already-approved primary protocol without tuning and recommends one independent test-set confirmation after a fresh, disjoint manifest and a separate engineering smoke. This does not turn the validation pilot or randomization checks into faithfulness evidence. No GPU run was launched during the design review.

The guarded execution wrapper is `scripts/run_head_faithfulness_confirmation.py`; it uses the existing anonymous publisher, excludes historical and smoke videos, and enforces the 3/3 smoke and minimum 71 effective confirmation cases. Its expected workload and exact Ubuntu command are in the primary estimand decision log. The wrapper has only CPU/static validation so far; the remote GPU run has not started.
