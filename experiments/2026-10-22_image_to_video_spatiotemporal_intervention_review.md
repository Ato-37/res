# Image-to-video intervention transfer review (2026-10-22)

## Decision

**Promising direction, with a specific change in unit of intervention:** transfer
image-space region interventions as sparse space-time object-region interventions,
not as independent repeated image masks and not as whole-frame masking. Start with
an outcome-blind geometry/preprocessing pilot, then run a small GPU comparison on
a frozen, previously unexposed cohort. The first measurement is a construct-validity
proxy; it cannot be called direct generated-explanation faithfulness unless claims
and their support are validated.

## Concrete precedents

1. **Li et al., WACV 2021, perturbation explanations for video classifiers.**
   Their transfer explicitly considers the added time dimension: 2D-EP perturbs
   frames separately, EP-3D optimizes across the full clip, and STEP adds a
   spatiotemporal smoothness constraint. Their temporal-pointing results also
   report that unconstrained EP-3D can over-attribute the beginning/end of a clip;
   temporal smoothness improved localization in their tested CNN/RNN settings.
   This supports treating time as part of the intervention and control, rather
   than reusing an image mask independently without temporal checks. The study is
   about classifier attribution, not Video-LLM explanation faithfulness.
   Source: https://openaccess.thecvf.com/content/WACV2021/html/Li_Towards_Visually_Explaining_Video_Understanding_Networks_With_Perturbation_WACV_2021_paper.html

2. **Xiao & Zhou, ICML 2026, MACD.** Their Video-LLM inference method creates
   object-level counterfactual videos by masking task-relevant regions and
   contrasts original/counterfactual decoding. It reports hallucination and
   task-accuracy outcomes across video benchmarks. This demonstrates a concrete
   modern VLM path from image-like object-region editing to video inference.
   For this study, the useful part is object-localized counterfactual inputs and
   matched decoding. Their model-feedback-driven region selection must not be
   copied into a confirmatory evaluation, because the intervention region would
   be selected from the model's own behavior. Freeze the region from annotation
   metadata before observing scores instead.
   Sources: https://proceedings.mlr.press/v306/xiao26w.html and
   https://github.com/chloeqxq/MACD

3. **Jing et al., ACL 2026 Findings, FIFA.** This is not a pixel intervention;
   it is relevant to the outcome side. It decomposes free-form video text into
   facts, represents semantic dependencies with a spatiotemporal graph, and
   verifies facts with VideoQA models. This supports scoring explanation claims
   individually rather than treating one fluent paragraph or answer accuracy as
   a faithfulness score. Its automated verification should remain secondary to
   independent support labels in a direct claim-level study.
   Source: https://aclanthology.org/2026.findings-acl.555/

## What this repository already establishes

- `scripts/necessity_choice_margin_test.py` applies whole-frame black/blur
  perturbations to a video, comparing annotated grounding frames with nearest
  non-evidence frames. On 60 held-out NExT-GQA videos, the full-cohort selective
  margin intervals included zero for both fills; a preregistered baseline-correct
  subgroup (n=37) had positive selective estimates. The controls match temporal
  position only, not image content, object area, or answer-discriminating detail.
  The result is a video QA sensitivity measurement, not explanation faithfulness.
- The claim-support QTA scorer can combine pixel masking and a frozen internal
  visual-key bias, but its 2026-10-22 real-video run is a synthetic-map wiring
  smoke. It adds no human-reviewed or direct-faithfulness observations.
- HD-EPIC local `mask_info` currently supplies frame-indexed boxes, not local
  raster segmentation masks. The official annotation README defines the box as
  `[xmin, ymin, xmax, ymax]` and warns that boxes and mask annotations may differ.
  The local videos inspected are 1408x1408; some annotation boxes extend beyond
  image bounds, so clipping must be measured and failures retained.
- A candidate three-video cohort has been frozen before outcomes after excluding
  previously exposed video IDs (cohort SHA-256
  `ae45c65ed9a268302b37fc6561631add23190194cd6a4b9ecda52bed2660480a`). Its
  eligibility and hashes bind to local annotation sources. Only one selected
  video's media is presently available locally; do not score it until the exact
  mapping/processor preflight and complete-cohort execution decision are logged.

## Proposed experiment

### Research question

Does an internal visual-token intervention targeted to an annotation-linked
object region increase the fixed claim's selective dependence on that region,
relative to dose-matched spatial and temporal controls?

### Unit and cohort

- Use a new frozen HD-EPIC/Minerva-Ego cohort, excluding every video that already
  contributed a model outcome. Select rows using metadata alone and freeze the
  selector inputs, IDs, and source hashes before generation/scoring.
- For the initial feasibility run, use up to three videos. Keep every selected
  row in the requested denominator; geometry, decoding, or mapping failures are
  retained as failures, not replaced after seeing scores.
- Freeze one dataset-authored, object-linked, timestamped rationale line per row
  before intervention. It is a fixed target sentence, not a claim validated by
  this project. A later direct study must instead freeze generated atomic claims
  and bind their support via the existing review artifact or another independent
  source.

### Geometry and intervention

1. Decode each exact annotated source frame using the official frame/time map and
   verify frame pixels/PTS. Validate bbox coordinate order against the official
   schema, image bounds, clipped-area fraction, and the final Qwen processor grid.
2. Map the original-frame bbox to merged Qwen visual-token cells from the actual
   `video_grid_thw` and resize geometry. Do not interpolate a sparse box across
   unannotated frames. For the first version, intervene only on sampled frames
   with exact annotation support.
3. Freeze three equal-dose key-bias conditions on the unmodified video:
   annotation-linked object-region keys; a same-frame, same-area/shape spatial
   sham outside all recorded boxes when feasible; and an equal-budget
   temporally displaced sham. Use the existing `ln(2)` bias/layers 5–16 only as
   a fixed exploratory setting, not as an optimized choice.
4. Separately measure region necessity by applying the same fill operation to
   the annotated object box and matched sham boxes, without changing prompt,
   sampling, frame count, or processor settings. Use one primary fill; treat a
   second fill as a robustness check because masking can create OOD inputs.

### Outcomes

- **Feasibility endpoint:** fixed target-claim mean log-probability under the
  baseline, object-region bias, spatial-sham bias, and temporal-sham bias.
- **Selective dependence endpoint:** change in target-claim score caused by
  object-box masking minus the same change caused by spatial/temporal sham
  masking. Compare this selectivity under targeted internal bias against
  equal-dose sham bias. State the exact sign convention in code and receipt.
- Keep full answer accuracy and gold-vs-best-foil margin as secondary diagnostics;
  they measure benchmark answer behavior, not explanation correctness.
- Report per-video paired values, requested/effective counts, failures, mean and
  uncertainty interval only when sample size and design support it. A 3-video
  pilot is descriptive and does not justify a population claim or significance
  test.

### Smoke and stop rules

- CPU synthetic test: known 2D boxes map to the expected merged patch set; reject
  malformed, inverted, non-finite, empty, or excessively clipped boxes; confirm
  sham masks have identical area and are disjoint from recorded support boxes.
- Real-video CPU preflight: confirm raw-frame dimensions, exact source PTS,
  processor frame/pixel identity, `video_grid_thw`, mapped spatial patch count,
  and intervention-only tensor differences. No model load until every planned
  row passes this gate.
- One-video GPU smoke: no more than seven forwards for baseline, object/sham
  internal biases, and object/sham pixel masks; record peak VRAM/time and all
  hook/pixel audits. Smoke outcome is excluded from any later cohort.
- Stop this branch if bbox-to-processor mapping cannot be independently
  reproduced, if a suitable equal-dose sham cannot be formed, or if most selected
  boxes cover no merged visual token. Do not substitute a frame-wide mask and
  call it spatial evidence.

## Claims and current blocker

This approach is promising because it combines a concrete image-region
intervention with established video perturbation practice and the model's actual
spatiotemporal visual-token layout. Its strongest near-term result would be
object-region-specific prediction dependence under a fixed internal intervention.
It would **not** prove that generated explanations are true or more faithful.
The handoff for three claim/frame reviews remains usable but has no bound
responses; current direct measures remain 0/3 reviews, 0/15 verified frame-time
pairs, 0/3 prediction links, and 0/3 direct scores (formal inventory 0/25).
The user declined the latest review request; no judgments are inferred from that
decision. Rights status remains unverified; all protected data and receipts stay
local.

## Next action

Implement the CPU bbox-to-Qwen patch mapper and its geometry tests first. Run it
on the frozen cohort only after missing local media is obtained through the
existing official-checksum downloader; retain its local-only status. If the real
processor audit passes on all three, freeze the seven-forward exploratory GPU
smoke and run it on a selected row that will be excluded from follow-up claims.
Then decide whether the measurement warrants an untouched larger cohort or
whether its proxy limitations make direct claim review/label acquisition the
only useful next step.

## CPU mapper implementation checkpoint (2026-10-22)

Implemented `bbox_to_merged_patch_offsets` and
`bbox_to_video_visual_key_positions` in
`scripts/minerva_ego_anchor_utils.py`. The mapper validates finite XYXY boxes,
source/grid geometry, clipping fraction, merged-grid divisibility, frame rank,
and token-group positions. Boxes use half-open pixel edges; returned offsets
are row-major on the post-merge visual grid. Nine new tests exercise geometry,
boundaries, clipping, malformed inputs, rectangular multirow order, and
temporal-pair mapping.

Verification: `PYTHONPATH=scripts .venv/bin/python -m unittest
test_minerva_ego_anchor_utils -v` passed 37/37 tests. A CPU processor integration
check used Transformers 5.18.0's `Qwen2VLVideoProcessor` with the exact pilot
pixel cap (`128 * 28 * 28`) on two synthetic 1408x1408 frames. It produced
`video_grid_thw=[1,22,22]`; source bbox `[128,128,256,256]` mapped to the
expected merged row/column `(1,1)`, offset 12, on the 11x11 merged grid. Source
inspection confirmed that resize preserves the full frame aspect ratio and
patchify orders merged 2x2 windows spatially before flattening. The Qwen2.5-VL
vision code reverses its window-attention permutation before returning merged
visual features, so token positions return to that order.

An independent read-only audit confirmed resize, row-major order, temporal
pairing, and caller requirements directly against the installed processor and
model source. Its separate non-square synthetic check produced grid `[1,16,32]`
and the expected four-cell mapping. No geometry mismatch was found. The mapper
is not yet wired into the GPU runner; its token-group argument still depends on
the caller providing ordered visual positions. The helper now rejects
non-increasing positions, and the rectangular test checks multirow ordering.
End-to-end runner wiring remains outstanding.

Critical limitation confirmed: each temporal patch combines two adjacent input
frames. A box on one frame therefore selects a spatial token whose pixel input
includes its paired frame; this cannot support a frame-isolated causal claim.
The intervention must be described as region-targeted at a two-frame temporal
resolution, or the sampling/control design must isolate the paired frame by
using the same pairing in every condition. The synthetic CPU integration is
geometry evidence only. No media was decoded, no model/GPU forward was run, and
there is still no direct explanation-faithfulness outcome. Rights status remains
unverified and all dataset material stays local.

Code SHA-256: `scripts/minerva_ego_anchor_utils.py`
`18022e40e37342ebb2d46433e1d0759011cf8a87d557a119b9601488e1dc9b9b`;
`scripts/test_minerva_ego_anchor_utils.py`
`2f459cf25c9a488480b8c2261cef5750bc9c58849182414d51049fcd69832be0`.

Next: wire the mapper into a CPU-only preflight on the frozen cohort's actual
decoded anchor frames. The cohort has 3 rows, but only 1 currently has its video
and time map locally; retain the other two as unavailable failures rather than
replacing them. Preserve all geometry failures. Only proceed to a bounded GPU
smoke if the planned rows pass and the temporal-pairing limitation is explicitly
handled in its estimand and controls.

## Follow-up: real-video token mapping and GPU pilot (2026-10-22)

The mapper is now wired into the runner's `spatial_bbox` mode. Dynamic frame
counts accept the selected video's 56-frame budget; source annotations bind to
the frozen cohort hashes, and the preflight binds cohort, media, time map,
sampling indices, anchor, and bbox coordinates. The full prompt's actual
`input_ids` are checked against `video_grid_thw`; bbox cells are converted into
ordered visual-key positions. A second deterministic six-token sham stays in
the same temporal pair but outside the bbox cells, and a third moves the same
bbox offsets to a matched non-track temporal group. All conditions use the same
`+ln(2)` bias and layers.

Actual-video CPU geometry succeeded for P05: 56 sampled frames, exact annotation
anchor at rank 20, 28 Qwen temporal groups, grid `[28,22,22]`, six bbox cells.
The full-prompt CPU smoke found 3388 video-token positions and confirmed a
six-key dose in each nonbaseline condition. Focused helper/preflight tests pass
48/48. This remains one row from a frozen 3-row denominator; two missing videos
are not replaced and are being retrieved under official manifest MD5 checks.

The local GPU pilot has completed and is documented in
`experiments/2026-10-22_minerva_bbox_spatial_internal_bias_gpu_pilot.md`.
On the one available row, bbox support was −0.013686 nat/token versus baseline,
−0.020889 versus same-pair spatial sham, and +0.042195 versus the temporal
bbox sham. Only the last contrast is positive. This is single-video fixed-rationale
likelihood sensitivity, not an explanation-faithfulness gain. An independent
auditor reproduced the scores, dose counts, and layer hooks; the GPU receipt
still omits an explicit preflight-hash field and has a stale nominal-grid error
field, both disclosed in the result log. A later full-prompt CPU receipt records
the corrected 0.2 s maximum sampling-grid deviation and Qwen's
`second_per_grid_ts=[1.0]`.

Next: finish downloading and checksum-verify the two remaining frozen videos,
rerun the three-row CPU preflight and full-prompt token smoke, then execute the
same exploratory four-condition GPU design on all three with no row replacement.
Use descriptive paired effects only; direct faithfulness remains unmeasured.
