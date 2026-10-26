# Real-video question-token selector wiring smoke (2026-10-18)

## Purpose

The earlier real-video wiring smoke selected keys from the final score-query attention row, which shares the query used for any next-token score. A synthetic question-token integration showed that a separate, outcome-independent selector can be composed with the key-bias hook. This bounded real-video test verifies exact question-span mapping and temporal-group selection on the pinned Qwen processor, then applies selected keys at a separate final query row. This is engineering feasibility only.

## Prespecified workload and verified setup

- Requested 3 manifest-unseen local NExT-GQA test videos, 16 decoder-PTS frames each, 6 forwards total (baseline selector and one treatment pass per video). Expected runtime was under 2 minutes and peak VRAM under 10 GiB, based on the previous 16-frame real-video wiring smoke. No QA choices, answer labels, gold interval values, output text, or faithfulness score enter the selector or treatment.
- Before model load: RTX 5070 Ti, 14,864 MiB free / 16,303 MiB total VRAM; WSL reported 15 GiB RAM with 13 GiB available; CUDA available, PyTorch 2.11.0+cu128, CUDA 12.8, BF16 supported.
- Pinned model revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights, BF16 compute, eager attention; selector layers 5–16 (1-based), engineering budget `ceil(10% of temporal groups)`, positive attention-logit bias `ln(2)`. These settings are not frozen efficacy parameters.
- The exact rendered prompt and processed `input_ids` were checked by the fail-closed mapper. Temporal groups were derived from processor video-token positions and `video_grid_thw` after the processor merge size; only full contiguous groups were eligible. Selector attention came from question-token query rows. Selected keys were then biased at the separate final score query and reused for that video's treatment pass.
- Two earlier question-selector runs are superseded: the first omitted full provenance, and the second did not yet bind image/video processor modules, processor fingerprint, checkpoint config, or source-video bytes. An earlier final-query-selector smoke is a different procedure and is not pooled with this result.

## Verified result (fully source- and input-bound rerun)

- Status: **passed**, 3/3 videos, 16/16 frames per video, 6/6 finite baseline/treatment forwards, no failures.
- All prompts mapped 13–22 question tokens. All three processor outputs had 8 temporal groups; one group per video was selected by the fixed smoke budget. Selected key counts were 252–264.
- Mean RMS vocabulary-logit change was **1.1400878**; the baseline top vocabulary token stayed unchanged in **3/3** cases. This is a descriptive wiring diagnostic, not a QA-answer or claim score.
- Runtime **37.53 s**; peak allocated/reserved VRAM **7.24/8.08 GiB**; free VRAM after the run **8.61 GiB**.
- All **11/11** recorded source hashes (runner, intervention/selector/cohort/decoder/batch-hash code, and installed Qwen model, processor, image/video processor, and video utility implementation modules) independently match current files. The 3 annotation-source hashes, 14 sample-manifest hashes, checkpoint config hash, and processor fingerprint also independently match current bytes/configuration.
- All 3 typed-batch digests stayed unchanged before baseline, after baseline, and after treatment. Independent recomputation of the deterministic candidate selection, source-video SHA-256, decoded PTS sequence, and RGB fingerprints matched **3/3** rows.
- The processor fingerprint binds the tokenizer and image-processor configuration, while each full typed-batch hash binds the effective produced video tensors and metadata. The video processor's separate configuration object and model-weight bytes are not separately hashed; the checkpoint is identified by its pinned revision and config hash.
- Independent auditor separately verified the result schema, source/annotation/manifest hashes, processor fingerprint, source-video/PTS/RGB correspondence, private modes, batch immutability, and no answer/gold scoring. No protected values were included in the audit report.
- The scoped implementation and tests are published as commit `cc0bb6ab1b593780f34c0ac867aad5aa96541f40`; the protected result JSON and this private log were not committed.
- Result JSON is mode `0600` under a mode `0700` private directory. Result SHA-256: `813a1b26e40e3b9bfd00426242ce3492a1aadf64c244e71c34c9b6198c356228`. The result and every video-derived measurement remain local.
- Focused tests: **14/14 passed**. Full repository suite: **1,458/1,458 passed** in 30.55 s. `py_compile`, CLI help, and `git diff --check` passed.

## Interpretation and limitations

The run confirms that, for three selected local examples, the production video-first prompt maps to exact question-token positions, complete processor-derived temporal groups can be selected from untreated question attention, and fixed keys can be biased at a distinct final query without non-finite outputs or batch mutation. The `ceil(10% of 8 groups)` rule selects one group (12.5% of groups); selected key counts vary with each temporal group's token count. Do not describe this as exactly 10% of video keys.

It does not show that selected groups correspond to an independently reviewed explanation claim, that the answer depends on them, or that faithfulness improves. The 10% group budget, layers 5–16, and `ln(2)` are engineering constants; the three examples are not an authenticated fresh confirmation cohort. No effect estimate, confidence interval, statistical test, or multiplicity adjustment applies because no faithfulness metric was measured. Rights remain unverified; the protected data and derived artifact remain local-only.

## Faithfulness progress and next action

**Direct evaluation: not performed; engineering diagnostic only.** Current verified gates remain 0/3 independently human-reviewed claims, 0/3 claim-linked scores, and 0/15 browser-to-decoder correspondences. The rebuilt candidate review byte inventory is complete but explicitly does not authorize review or scoring. Before any direct internal-ablation evaluation, independently complete human claim review and decoder correspondence; then freeze treatment/sham/control, fixed baseline answer/foil tokenizer positions, actual branch batch hashes, estimand, and analysis rule behind an owned production gate. Do not use this smoke to tune settings or claim a faithfulness benefit.
