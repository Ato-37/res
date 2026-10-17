# Synthetic nine-forward GPU capacity smoke

## Purpose

Check whether the pinned Qwen2.5-VL scoring path can execute the complete
three-row by three-condition (nine-forward) workload sequentially on the
available GPU. This is an engineering capacity test only. The inputs are
synthetic questions and frames; this is not a dataset experiment or a
faithfulness measurement.

## Setup

- Command: `PYTHONPATH=scripts .venv/bin/python scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py`
- Model: cached Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Precision: 4-bit weights, BF16 compute; PyTorch 2.11.0+cu128, CUDA 12.8.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB reported total memory.
- Inputs: three synthetic records, 32 synthetic frames per condition, five
  answer-token candidates, and the fixed conditions `full`,
  `claim_evidence_black`, and `temporal_control_black`.
- The scorer ran sequentially and returned only aggregate diagnostics; no
  question, answer, video, or item-level score was read or emitted.

## Verified result

The script exited successfully with status `passed`: requested/effective
records 3/3, requested/completed forwards 9/9, completed triplets 3/3, and
zero failures. Scoring took 30.158 s. Peak allocated VRAM was 10.744 GiB,
peak reserved VRAM was 12.473 GiB, and minimum free VRAM before a forward was
1.921 GiB. The GPU returned to 14,914 MiB free and 1% utilization after the
process exited. Focused CPU tests for the condition builder, atomic freezer,
and scorer passed 38/38 with `PYTHONPATH=scripts`.

Source SHA-256 values at execution:

- GPU smoke: `747fe4efe01183f9f41fbb4095d4d828bed6748d3f49652a9f7fbbdd20ef5c22`
- condition builder: `57fb033e153cfc3737ac7292be4adeb53768bafa771652438bcd36c45514ae1a`
- atomic freezer: `751c797f13316a5aba6e32561041c4e9a40718e4667d54b320707af018426eb5`
- scorer: `8d95a1a8ec39a56c55efb38267227d7c118368a01efb9c8243bc45cdbebbe69b`

The working tree was clean for these four source files; their source commit
was `2fa19e7e8948078a76983f5002f3ddb10de6193d`. An independent code audit
confirmed that the script constructs and freezes all nine synthetic batches
before loading the model, then validates hashes and runs the shared scorer
sequentially. The existing log contains a different historical smoke-script
hash; this entry binds the current run to the hash above and does not reuse
that older value.

## Interpretation and limits

The 30.158 s interval covers scorer execution through CUDA synchronization; it
excludes checkpoint loading and CPU batch construction. Peak allocation was
reset after model load and therefore includes resident model allocation. This
confirms that the synthetic nine-forward implementation path completed under
the observed GPU conditions and gives a narrow capacity diagnostic for later
scheduling. It does not establish that real video inputs will fit: prompt and
aspect-ratio token geometry, other GPU users, and warm/cold variability were
not measured. The reported 1.921 GiB minimum free memory is not a guaranteed
real-run safety margin; retain sequential execution and the live per-forward
memory guard. The result does not establish that upstream review/clock
evidence is valid or that any explanation is faithful. It does not grant
execution authorization. Direct generated-claim
faithfulness remains 0/3 human-reviewed and 0/3 scored, with no direct effect,
uncertainty interval, or test. The observed minimum GPU headroom was 1.921
GiB, so any future real run must remain sequential and retain the existing
per-forward memory guard.

## Next action

Do not repeat this smoke. Complete the independent human claim reviews, all 15
browser-to-decoder frame-pair decisions and clock calibration, trusted joint
freeze, and the production all-three-row atomic preflight. Then run the exact
three-item real-video score once, with nine forwards and fail-closed behavior.
Rights status remains unverified; protected data and derivatives stay local.
