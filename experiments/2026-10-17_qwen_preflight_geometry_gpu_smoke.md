# Qwen input-geometry guard: synthetic GPU integration smoke

Date: 2026-10-17 (Asia/Tokyo)

## Purpose

An independent source audit found that the NExT-GQA condition builder used
the vision utility's returned tensor shape as the expected resize target. A
different spatial resize could therefore validate itself. The builder was
changed to derive the expected dimensions from the fixed 32-frame input and
the installed `qwen_vl_utils.vision_process` resize policy, then compare the
exact reconstructed frame tensor. This smoke checks that the stricter builder
still feeds the existing nine-forward scorer path.

## Setup

- Data: deterministic generated RGB frames and invented five-choice prompts;
  no dataset video or annotation was read.
- Model: locally cached Qwen2.5-VL-7B-Instruct, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Precision: BF16 model with 4-bit weights and BF16 compute.
- Hardware/software: RTX 5070 Ti (16 GiB), CUDA 12.8, PyTorch 2.11.0+cu128.
- Requested/effective: 3 synthetic records, 9 condition batches, 9/9 forwards.
- Scoring command:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py
```

## Verified result

The run completed with status `passed`, 9/9 model forwards, 3/3 triplets, and
zero failed conditions. Scoring time was 30.94 seconds. Peak allocated and
reserved VRAM during the scoring phase were 10.744 GiB and 12.473 GiB; minimum
free VRAM immediately before a model call was 1.871 GiB. After process exit,
the read-only GPU check showed 14,867 MiB free. The runner reported
`faithfulness_result=false` and `execution_authorization=not_granted`.

The condition-builder CPU suite passed 8/8, the pinned local Qwen processor
CPU smoke built and verified 9/9 batches with CUDA uninitialized and zero
forwards, the repository test suite passed 1,325/1,325, and `compileall` plus
`git diff --check` passed. The processor emitted an existing deprecation
warning about `min_pixels`/`max_pixels`; execution completed successfully.

## Interpretation and limitations

This confirms that the revised geometry check and the current synthetic
Qwen/scorer path execute together on the local GPU. It is an engineering and
capacity check only. No real video, claim, human review, or faithfulness
intervention score was measured. There is no efficacy estimate, confidence
interval, hypothesis test, or multiplicity adjustment to report. GPU resource
values describe this one synthetic run and are not a general capacity bound.

The resize reconstruction matches the current fixed prompt contract, which
contains `video` and `sample_fps` metadata. If future code adds explicit
`resized_height`, `resized_width`, `min_pixels`, `max_pixels`, or
`total_pixels`, the independent reconstruction must bind those values too.

## Faithfulness status and next action

Direct claim-linked video faithfulness remains **0/3 independently
human-reviewed and 0/3 scored**, with no direct estimate, interval, or test.
The human claim review, authenticated custody/trust-root release, and real
browser-clock-to-decoder frame correspondence gates remain unresolved. Do not
run another synthetic GPU cohort for this same integration question. Resume
the direct branch only after the independent human-review and capture evidence
are supplied and pass their existing validators; then run the frozen three-row
production preflight and nine-forward test.

The implementation and test changes were committed as
`f96b6d7c89ce9feacb0496d1e81051d801c310ee`; live `origin/main` was verified at
that commit. The experiment summary contains aggregate synthetic results
only.
