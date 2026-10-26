# Counterfactual explanation-faithfulness design audit (2026-10-22)

## Purpose

Identify a nonduplicate faithfulness endpoint that can eventually test whether
an internal video intervention improves the link between generated
explanations and prediction behavior. This is a literature/design audit only;
it adds no sample or GPU result.

## Primary-source finding

Menzat et al., *Counterfactual Tests for Measuring Chain-of-Thought
Faithfulness in Visual Language Models* (arXiv:2609.06704, accepted to
Findings of EMNLP 2026), introduce vCT and vCCT for image VLMs. Their continuous
vCCT is the point-biserial correlation between counterfactual answer
distribution shift (TVD over a fixed candidate set) and whether the explanation
mentions the changed object. The paper reports that binary vCT is often near
saturated, whereas vCCT distinguishes prompting conditions. Their image
counterfactual pipeline uses Qwen2.5-VL for object selection, Qwen-Image-Edit
for removal, and GroundingDINO/SAM filters; it also includes a no-op editing
control. A manual audit found 87.9% correctness for edits/change descriptions
and 92.2% for change-mention detection. The authors explicitly scope the main
results to images and leave direct action/relation edits as future work.

Primary sources:

- https://arxiv.org/html/2609.06704v1
- https://github.com/neuralisa/cct-vlm

## Fit to this project

The continuous alignment idea addresses a weakness already seen here: internal
attention/log-likelihood movement may occur without any answer-choice change
(the 27-row full-video QA diagnostic had 0/27 choice flips and 15/27 correct in
all conditions). A video adaptation could compare answer-distribution change
under a controlled event intervention with whether the generated explanation
describes that changed event, then compare vCCT under baseline, the frozen
internal intervention, and an equal-budget matched sham. This would measure
behavioral explanation–prediction alignment, not expose hidden reasoning and
not by itself prove causal mechanism.

This is a candidate design, not an approved or executed protocol. The local
project lacks a validated event-level video counterfactual pair set. Existing
NExT-GQA QA intervals are not claim-specific event labels; the CLEVRER pair
feasibility branch is closed (only 12 pairs passed the fixed surface-distance
screen, with no semantic validation); temporal reversal is already a closed
proxy and can be out of distribution. The current review artifact collects
claim support/frame choices, not whether a named counterfactual event is
mentioned. Current HF cache inspection found only Qwen2.5-VL-3B and 7B models,
not Qwen-Image-Edit, GroundingDINO, or SAM weights. A fresh outcome-blind
filesystem check found 533 `.mp4` files under the local NExTVideo sample tree;
the local `datasets/` folder contains CLEVRER question/program and human
annotation materials but no CLEVRER video clips; and no matching
counterfactual-video directories were found in the searched `~` and
`/mnt` roots. The cached model inventory resolves to only the two Qwen2.5-VL
checkpoints. This does not prove no such data exist outside the searched roots,
but confirms none are present in the current known research inputs. Therefore
no real-video GPU run is currently justified: constructing a valid, temporally
coherent edit pair and verifying its semantic scope is the missing
measurement/data work, not GPU compute.

## Prespecified next decision

The planned outcome-blind local inventory is complete for the searched
workspace, dataset folders, and `/mnt` roots: it found no prepared event-level
counterfactual video pair set with stable answer choices. Do not reuse the
closed CLEVRER, temporal-order, QA-interval, or frame-mean cohorts. Develop a
paired-video protocol only after a source provides a defensible changed-event
label and a no-op or matched-control edit. Any future metric must report the
continuous answer-shift/explanation-mention relationship, per-video clustering,
pair failures, independent/manual audit status, and baseline/intervention/sham
paired uncertainty. Keep human-confirmed semantic labels separate from
machine-judge labels.

## Faithfulness status

Directly scored claims remain 0/25 formally reviewed, and 0/3 for the current
three-item cohort; no direct effect or interval exists. This literature audit
changes a future design candidate only. It does not change the evidence count,
and no model/CUDA/GPU process was run.
