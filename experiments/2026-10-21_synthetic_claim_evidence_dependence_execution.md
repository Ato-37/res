# Synthetic oracle-claim evidence-dependence execution (2026-10-21)

## Purpose and frozen design

Run the pre-specified synthetic construct study in
`2026-10-21_synthetic_claim_evidence_dependence_protocol_v1.md`. It asks
whether claim-selected attention-key bias increases true-versus-false claim
support-margin dependence on retaining the transient event window, relative to
an independent equal-size hash sham, compared with matched non-event control
ablation. This is an engineering/construct sensitivity study, not a natural
video explanation-faithfulness evaluation.

The frozen cohort contains 48 procedurally generated 32-frame videos in three
event families × four onset cells × four independently seeded nuisance scenes.
The 2×2 object × answer-label assignment is balanced in every cell. Each video
has a true claim and a one-object false foil. Each claim received nine forwards
(baseline, selected bias, and hash sham under full/event/control key conditions),
for 96 prompts and 864 requested forwards. The primary is the equally weighted
12-cell mean of the selected-bias minus sham difference-in-differences in the
signed Supported-vs-Not-supported log-odds.

## Provenance and execution

- Source commit: `da704d3135b0a3f72c90cff5d54bf5bab07d775f`.
- The private run receipt binds the protocol, generator, preflight, scorer,
  attention-capture/intervention helpers, QA video processor, source commit,
  full checkpoint shard manifest, model config, processor and tokenizer hashes.
- Model: pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
  eager attention, layers 5–16, RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128,
  Transformers 5.18.0, qwen-vl-utils 0.0.14, bitsandbytes 0.50.2.
- CPU suite: 37/37 focused generator, scoring, intervention, and attention
  capture tests passed. The current full CPU processor preflight passed 48/48
  clips and 96/96 prompts, zero failures; all prompts had `video_grid_thw`
  `[16,26,42]`, with 12 cells of four and each target-object × label-order
  combination represented once per cell.
- One external GPU engineering smoke completed 18/18 forwards in 96.241 s,
  with zero failures. An independent audit verified the smoke receipt, source,
  assignment, processor/tokenizer, config, and all five checkpoint shard hashes.
- Full run completed 48/48 clips, 96/96 prompts, and 864/864 attempted and
  completed forwards with zero failures in 3,271.158 s (54 min 31 s). Peak
  allocated/reserved VRAM was 12.21/14.29 GB. The aggregate private receipt is
  `data/private_nextgqa/oracle_claim_gpu_full_20261021_2155.json`, mode `0600`,
  SHA-256 `b343712b549f76f4710c6d3639b43aaa39c7bbd7a60387bc74bcb4b6ec018846`.
- A separate calculation from the receipt's 12 cell sums and sums-of-squares
  reproduced the estimate, standard error, Welch–Satterthwaite degrees of
  freedom, confidence interval, and p-value to numerical tolerance.

The existing worktree contains unrelated user changes. This log and the private
aggregate receipt remain local; no protected video, claim text, prompt, or model
weights were published. Model/data rights status remains unverified.

## Observed results

| Outcome | Result |
|---|---:|
| Primary equal-weighted difference-in-differences | −0.0369 nat |
| Stratified SE; Welch–Satterthwaite df | 0.0382; 18.78 |
| 95% two-sided t interval | [−0.1169, +0.0432] nat |
| One-sample two-sided test | p = 0.347 |
| Positive / negative / zero scene contrasts | 22 / 26 / 0 of 48 |
| Selected-bias full-video A/B accuracy | 59/96 = 61.5% |

Descriptive primary-contrast means by event family were gate +0.0066 nat,
expansion +0.0156 nat, and rotation −0.1328 nat. These family summaries are
exploratory, unadjusted descriptions, not separate confirmatory tests. The
smoke's 1/2 accuracy is engineering-only and is not included in the cohort.
There is one pre-specified primary test; no multiplicity adjustment is needed
for that single endpoint. The interval includes zero and does not support a
positive attention-selection advantage over the sham on this benchmark.

## Interpretation and limitations

The result is inconclusive around zero and trends slightly negative; it is not
evidence of a harmful natural-video faithfulness effect. At most, this run
shows that the tested Qwen configuration completed the frozen internal
interventions and that the pre-specified benchmark did not detect a positive
selected-bias versus sham advantage. It does not validate generated
explanations, natural-video grounding, or improvement in direct faithfulness.
The small four-scene cells, procedural renderers/oracle labels, answer-language
priors, attention steering, and key ablation's possible distribution shift
limit interpretation. Family differences must not drive post-hoc tuning.

## Direct-faithfulness checkpoint and next action

Direct natural-video explanation faithfulness remains `未実施` in this run.
The independent local human-review inventory found 0 valid responses among 52
candidate files bound to the current three-item review artifact: claim support
0/3, sampled-frame correspondences 0/15, prediction links 0/3, and direct
intervention judgments 0/3. No human judgment is inferred. The current Japanese
launcher now points to the current review artifact; the review remains
exploratory and does not authenticate reviewer identity.

Do not tune this synthetic cohort or reopen the closed head/layer/alpha search.
Next, inspect the most recent real-video claim/video-content and native
QA-interval intervention results to select a non-duplicative, frozen real-video
measurement that can run without human labels. Keep such QA/annotation
dependence results labeled as proxy diagnostics. For direct explanation
faithfulness, retain the valid local review handoff and require actual
artifact-bound human decisions before using claim-support labels as evidence.
