# EG-VQA fit to the thesis faithfulness question (2026-10-19)

## Purpose

Check whether a recent evidence-grounded VideoQA benchmark offers a distinct,
executable GPU experiment while the NExT-GQA human handoff is pending. This is
a primary-source and local-availability audit, not an experiment or a dataset
adoption decision.

## Sources and verified observations

- The EG-VQA v2 paper describes 2,067 videos and 11,838 open-ended QA pairs;
  each answer is associated with timestamped textual evidence. It reports a
  video-disjoint split with 1,731 training videos / 8,949 questions and 336
  test videos / 2,889 questions. Evidence-Grounded F1 combines temporal IoU
  and semantic similarity, then uses one-to-one optimal matching. The paper
  also describes an additional stratified human audit of 1,000 examples, with
  over 95% verified as valid; the full dataset screening uses two LLM judges
  and human adjudication for disagreements. Source: [EG-VQA paper, v2]
  (https://arxiv.org/html/2606.24797v2), §§3.1–3.3 and Appendix A.3.
- The official repository says original videos are **not included** and must
  be obtained from ActivityNet Captions, YouCook2, and HiREST source datasets.
  The Hugging Face dataset page currently labels its metadata/annotation
  dataset `cc-by-4.0`; that label alone does not establish rights to each
  original video. Sources: [official EG-VQA repository]
  (https://github.com/HCPLab-SYSU/EG-VQA) and [EG-VQA dataset card]
  (https://huggingface.co/datasets/lphuang33/EG-VQA).
- A local filename scan found **0** EG-VQA, ActivityNet, YouCook2, or HiREST
  dataset files under the repository's `data/` tree. The current local corpus
  is NExT-GQA; its `gsub` records QA-associated time intervals, as documented
  in `experiments/2026-10-19_qta_gsub_alignment_and_review_workspace.md`.
- EG-VQA's evidence annotations can support evaluation of a model's predicted
  temporal-semantic evidence. They do not, by themselves, establish whether a
  generated explanation is true, whether the model's prediction depends on
  that explanation's video evidence, or whether an internal intervention
  improves the thesis's target outcome. That is a scope inference from the
  benchmark's stated output and metric, not a result reported by this project.

## Decision and limits

Do not pivot or start an EG-VQA GPU run now. It would require acquiring a new
video corpus, checking source-video terms and local availability, freezing a
new task/estimand and output protocol, and implementing an evidence-generation
evaluation path. A small run on that benchmark would be exploratory and would
not unblock or replace the current NExT-GQA direct-faithfulness path. The
benchmark is a plausible *separate* future test of generated evidence
localization, subject to those prerequisites.

The current direct-faithfulness path remains the nearer executable route once
its actual human and trust inputs arrive. No GPU run was justified by this
literature audit. Rights for the current protected NExT-GQA materials remain
unverified; no data were downloaded or published during this audit.

## Faithfulness status and next action

Direct claim review remains **0/3**, verified player-clock/decoder-frame
correspondences **0/15**, prediction-link judgments **0/3**, and direct scores
**0/3**. There is no direct `ΔF`, confidence interval, or statistical test.
The informal quick-review payloads are test fixtures, not observations, as
documented in `experiments/2026-10-19_quick_review_response_mapping_audit.md`.
Continue the local human-review handoff and the fail-closed baseline transaction
audit; reconsider EG-VQA only if the research question is deliberately expanded
to include a separate evidence-localization benchmark.
