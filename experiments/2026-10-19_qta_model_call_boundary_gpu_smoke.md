# QTA scorer root-model call-boundary GPU smoke (2026-10-19)

## Purpose

The three-row claim-mask scorer already froze all nine CPU processor batches
and checked their device digests before forwarding. It did not verify the
exact final kwargs received by the root model. This cycle adds a call seam and
a temporary `model.forward` wrapper to verify the processor-field set, batch
digest, and fixed scoring options after global and module pre-hooks have run,
immediately before the original forward body.

This is a software-path provenance test on synthetic data. It does not measure
faithfulness, establish claim truth, or authenticate the model/runtime.

## Change and protocol

- Added `scripts/nextgqa_model_call_boundary.py`. It accepts only the exact
  processor batch key set passed separately by the scorer, plus fixed
  `use_cache=False` and `logits_to_keep=1`. At the final forward entry it
  checks kwargs, zero positional arguments, and typed batch digest, invokes the
  original bound method, then restores any pre-existing instance method (or
  removes the temporary instance attribute) in `finally`.
- `score_preflighted_triplets` now supports an optional `model_call` seam and
  sends it the exact assembled kwargs, expected digest/key set, and row and
  condition positions. Existing callers without the seam retain their prior
  behavior.
- The synthetic GPU integration smoke routes all nine Qwen forwards through
  this guard and reports root-entry, verified, and completed-call counts
  separately. It prints only aggregate resource/call counters.
- Local deterministic test data only: 3 synthetic records × 3 conditions;
  pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
  BF16 activations, eager attention; RTX 5070 Ti, PyTorch 2.11.0+cu128,
  CUDA 12.8. No protected video or real QA row was loaded.

## Verified result

- Focused CPU integration suites: **39/39 passed** (scorer and atomic batch
  freeze). Coverage includes all-nine-call success with a real `torch.nn.Module`,
  post-pre-hook kwargs rewrite rejection, class/instance `forward` restoration
  on error and success, changed tensor rejection before the model body, extra
  option rejection, mutation, nonfinite outputs, and null aggregate on failure.
- Python compilation and scoped `git diff --check` passed.
- GPU smoke command: `.venv/bin/python
  scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py --output
  data/private_nextgqa/qta_model_call_boundary_smoke_20261019_0735.json`.
- Result: **3/3 synthetic records, 9/9 root entries, 9/9 verified root
  kwargs, 9/9 completed guarded calls and scorer forwards**; status `passed`,
  failures 0. Scoring took **30.113 s**. Peak allocated/reserved VRAM was
  **10.744/12.473 GiB**; free VRAM was **14.657 GiB** before model load.
  After process exit, free VRAM returned to **15,112 MiB** of 16,303 MiB.
- Aggregate-only result: `data/private_nextgqa/qta_model_call_boundary_smoke_20261019_0735.json`,
  mode `0600`, 1,625 bytes, SHA-256
  `4111bafabc25403ef3721759bd9ee48fb7596b793e793f2bce776713b7de2eeb`.
  Independent readback verified source hashes, exact 9/9 counters,
  `faithfulness_result=false`, `execution_authorization=not_granted`, and no
  row-level score, question, or claim fields. The same aggregate-only bytes
  are copied for repository handoff as
  `results/synthetic_qta_model_call_boundary_smoke_20261019.json`; its hash
  matches the private receipt and it contains no protected data.
- Code SHA-256 values used by the run:
  - `scripts/nextgqa_model_call_boundary.py`:
    `8ed9f6368522e69acb875d31fdcbd3bc5cf389483f529ffca9737f2309f3d1d8`
  - `scripts/nextgqa_claim_mask_scoring.py`:
    `c10bc817f6c50593ccefeed027baabe491088fe6e2b78fc06baa5a8395f9c706`
  - `scripts/smoke_nextgqa_qwen_gpu_atomic_scoring.py`:
    `7ee5fb384863a85f2fc46138498e527ce0ab6c00d858aad4ea6bb7c42fb62589`
  - `scripts/nextgqa_atomic_batch_freeze.py`:
    `b6e5c60bc51cdf913c91ec6a2d8acfdee2c4f715eadaf851d7d9ce7f1d569c92`
  - `scripts/nextgqa_condition_batch_builder.py`:
    `7486c6e959e507add3190b2af91b92ddf3d9dc84035a11cd44b3c0f36cee14e7`

## Interpretation and limitations

The observed test confirms that, in this pinned local Qwen synthetic smoke,
kwargs at the final Python `forward` entry after all pre-hooks match the
scorer's expected batch digests and fixed scoring options for all nine calls.
This improves software-path evidence before a future scored forward. It does not establish
that downstream kernels consume immutable bytes, defend against hostile or
globally registered hooks, authenticate checkpoint weights/code, verify real
video or generated-claim provenance, or show that any prediction depends on
the evidence described by an explanation. The smoke is explicitly
`faithfulness_result=false` and `execution_authorization=not_granted`.

## Faithfulness progress

- Direct claim-linked faithfulness: **0/3 reviewed, 0/3 scored**.
- Clock/player-to-decoder-frame correspondence: **0/15**.
- Stage-2 prediction-link judgments: **0/3**.
- Direct `ΔF`, uncertainty interval, and inferential test: **none**.
- The nickname-first impression page is an informal three-item check; no
  response JSON exists. Rights status remains unverified; local artifacts stay
  private.
- Previous real-data QTA annotation-dependence, QTA-v2, and QTA-v4 proxy
  outcomes remain closed exploratory branches. This synthetic run adds no
  real-data or direct-faithfulness evidence.

## Next action

Do not expand or rerun the closed proxy cohorts. Continue the simple local
human-check handoff and inspect any returned impression JSON as exploratory
feedback only. For direct scoring, first complete the actual formal claim and
frame review inputs and freeze the baseline/prediction-link bindings; then use
the new root-boundary call seam in a one-row real-video smoke before any
fixed-cohort intervention scoring. No real-data forward was run in this cycle.
