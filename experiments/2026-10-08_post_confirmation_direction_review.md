# Post-confirmation research-direction review

Date: 2026-10-08

## Purpose

Choose a scientifically useful next step after the independent head-faithfulness test did not meet its preregistered success criterion. Do not treat a repeated analyzer invocation as new evidence, and do not launch GPU work simply to maintain activity.

## Verified basis

- The head-faithfulness test is recorded in `results/head_faithfulness_confirm_20261009.json` and its run manifest: 79 requested, 79 effective, 0 failures; primary estimate `−0.0756 nat`, SD `0.3667`, 95% t CI `[−0.1578,+0.0065]`; the lower-bound criterion `>+0.25 nat` was not met.
- Previous grounding-mask work measures answer-margin sensitivity to removing evidence-like frames. It does not by itself validate an explanation; masking may create out-of-distribution inputs, and temporal-position controls are not matched on semantic information.
- The model was evaluated only on A–E choice margins in the head experiment; that run did not ask the model for a natural-language explanation. The current result therefore says nothing direct about whether a generated rationale is faithful.

## Literature scan (official records checked 2026-10-08)

- Findings of EMNLP 2025 studied text- and image-based bias effects on LVLM chain-of-thought and how those biases are articulated in explanations: [ACL Anthology record](https://aclanthology.org/2025.findings-emnlp.723/).
- EDCT proposes extracting visual concepts from an image-VLM's own explanation and testing them with generated image counterfactuals. The version found in this scan is an arXiv preprint, not treated here as peer-reviewed confirmation: [arXiv:2510.00047](https://arxiv.org/abs/2510.00047).
- A 2026 TU Wien diploma thesis studies counterfactual faithfulness of VLM natural-language explanations on SNLI-VE image data: [repository record](https://repositum.tuwien.at/handle/20.500.12708/226904).
- FaithLM (EACL 2026) tests explanations by contradicting their content with a hint and observing prediction changes, but is an LLM explanation framework rather than a video-specific evaluation: [ACL Anthology](https://aclanthology.org/2026.eacl-long.177/).
- Earlier ACL work already proposes counterfactual tests for claims in natural-language explanations and considers predicted probability shifts: [Atanasova et al. 2023](https://aclanthology.org/2023.acl-short.25/), [Siegel et al. 2024](https://aclanthology.org/2024.acl-short.49/).
- The official CLEVRER specification says train/validation counterfactual QA includes answer labels and functional programs for each answer choice, while test omits answer/program labels. The dataset also provides object trajectories and collision-event annotations; the original paper describes 37,253 counterfactual questions. Sources checked: [official README](https://data.csail.mit.edu/clevrer/README.txt), [original paper](https://arxiv.org/abs/1910.01442). The separate CLEVRER-Humans CEG annotations are human-written causal graphs and do not supply verified temporal intervals in the existing local audit; they are not a substitute for official QA programs.

These works make broad novelty claims such as “first counterfactual faithfulness test for VLM explanations” untenable. A defensible gap may require video-specific temporal-event counterfactuals and matched sham edits, but that gap has not yet been established by a systematic related-work search.

## Decision and next action

The first remote search returned no files, but it searched only paths matching `*/questions/{train,validation,val}.json`. This was too narrow: the public CLEVRER code repository keeps the question files at `executor/data/train.json` and `executor/data/validation.json`. The source files are 82.6 MB and 41.3 MB respectively at the inspected GitHub revision `98b842082ba4f7c18b6b9e3f39145871782a65ef`; no videos are needed for the schema audit. Thus the observed result means “not found in the searched local paths,” not “unavailable.”

Next, download only those two pinned JSON files to a directory outside the repository, then run the CPU-only audit. `scripts/audit_clevrer_counterfactual_programs.py` reports question/choice counts and answer/program coverage; `scripts/run_clevrer_program_audit_and_publish.py` publishes a counts-only summary without raw text or absolute paths. Confirm counterfactual rows with answer/program and QA-to-scene linkage. This is a schema/data feasibility gate, not a faithfulness test. NExT-GQA temporal evidence alone does not create a counterfactual. CLEVRER scene/causal structures may support controlled reconstruction, but the synthetic-video domain and rendering pipeline require verification. Until a source of counterfactual pairs, independent event labels, and edit-quality controls is identified, there is no new GPU command.

The proposed candidate question and explicit gates are in `docs/post_confirmation_direction_review.md`. This is a design candidate, not an approved experiment, novelty claim, external editing-API request, or GPU authorization. If the feasibility gate fails, close experimentation and frame the thesis around the limits and negative results of the evaluated faithfulness tests rather than manufacture a positive result.

## Workflow improvement

The published-result analyzer now prints only sample accounting, target/control effects, the primary CI, two relevant p-values, and the registered decision by default. Full JSON is available only with an explicit `--json` flag. More importantly, users should not be asked to rerun an already-verified artifact merely to display the same numbers; the default handoff is a short interpretation of repository-resident results.

## CLEVRER official-question schema audit (verified 2026-10-08)

The counts-only result is published at `results/clevrer_program_schema_audit_20261008.json`.
It covers the two expected official files (`train.json`, `validation.json`), whose SHA-256
digests are recorded in that artifact. This was a deterministic CPU census; no GPU inference,
statistical test, sample exclusion, or raw text publication was involved.

| Split | Questions | Counterfactual-type | Questions with answer/program-key/scene-link present | Choices with answer/program-key present |
|---|---:|---:|---:|---:|
| Train | 152,572 | 18,642 | 152,572 / 152,572 | 141,211 / 141,211 |
| Validation | 76,368 | 9,333 | 76,368 / 76,368 | 70,862 / 70,862 |
| Total | 228,940 | 27,975 | 228,940 / 228,940 | 212,073 / 212,073 |

Interpretation: these are field-presence counts, not evidence that program payloads are
non-empty or executable. The later field audit found question-level counterfactual `program`
values to be lists with no dictionary nodes, so the earlier 100% program-field coverage does
not establish usable functional programs. This is only schema availability. It does not test
whether a model-generated rationale matches a program, whether a program pair encodes a
valid isolated counterfactual, or whether the model's explanation is faithful.

### Initial within-scene signature census (superseded for program-pair interpretation)

The counts-only artifact is `results/clevrer_program_pairability_audit_20261008.json`.
Its train/validation SHA-256 values exactly match the earlier schema audit, so the same
source files were examined.

| Split | Counterfactual scenes | Scenes with exactly 2 CF questions | Those pairs with distinct program signatures | Max CF questions/scene |
|---|---:|---:|---:|---:|
| Train | 10,000 | 8,642 | 8,642 | 2 |
| Validation | 5,000 | 4,333 | 4,333 | 2 |
| Total | 15,000 | 12,975 | 12,975 | 2 |

Observed at the time: every scene containing a counterfactual question had at most two, and
the selected fields had distinct signatures in 12,975 two-question scenes. The helper
preferred the question-level `program` field even when choice-level programs were present,
so those original signature counts are invalid for answer-choice program comparison. The
scene/question counts remain valid. The corrected choice-level signature census is reported
below and supersedes the original signature interpretation.

### Corrected choice-program audit and operator-token interpretation (2026-10-08)

The corrected counts-only artifact is
`results/clevrer_program_choicelevel_corrected_audit_20261008.json`. The source hashes match
the earlier train/validation audits. All 27,975 counterfactual questions have non-empty
choice-level program payloads; the question-level payload contributes no non-empty program.
The payloads are lists of strings, not lists of node dictionaries. Of exactly-two-question
scenes, choice-program signatures differ in 8,641/8,642 train scenes and 4,333/4,333
validation scenes (12,974/12,975 combined). These hashes establish only structural
difference, not that videos differ in one isolated event or that the questions form valid
matched counterfactual pairs.

The published operator-token audit is
`results/clevrer_program_operator_trace_corrected_20261008.json` (GitHub commit `c429b88`).
It covers the same source files: their SHA-256 digests are unchanged. Among 12,975
exactly-two-counterfactual-question scenes, one pair had identical complete choice-program
signatures. Of the remaining 12,974 pairs, 158 (1.22%) had the same sorted multiset of
per-choice operator-token sequences (104 train, 54 validation); 12,816 (98.78%) differed
(8,537 train, 4,279 validation). Operator extraction was available for every distinct-signature
pair. This is a deterministic census, so no p-value or confidence interval applies.

The same-trace subset is small and post hoc; matching operator sequences do not prove that
the natural-language questions differ only in the intended intervention, that their choice
programs encode equivalent outcomes, or that video evidence is isolated.

The next CPU-only screen, published as
`results/clevrer_pair_question_token_distance_20261008.json` (commit `29373ea`), compared
tokenized question strings only within the 158 distinct-program/same-operator-trace pairs.
It published coarse edit-distance bins, not raw text or tokens. Train: 9/104 pairs had
distance 2–3 and 95/104 had distance 4+; validation: 3/54 and 51/54, respectively. Combined,
12/158 (7.6%) had distance 2–3 and 146/158 (92.4%) had distance 4+; none had distance 0 or
1, and none were unavailable. Source hashes still match the previous census. This is an
exhaustive deterministic count, not a statistical test. Token edit distance is only a crude
surface-form measure: the 12 closer pairs are not thereby established as changing one
intervention, and the 146 farther pairs are not proven semantically unmatched.

**Decision:** stop the CLEVRER structural-pair feasibility branch here; it has not established
a defensible set of matched video counterfactuals for a faithfulness experiment. Do not launch
GPU work or continue threshold-tuning on these post-hoc bins. Preserve the result as a negative
data-feasibility finding. Any revival would need a separately justified semantic pair
definition and blinded manual validation criteria before inspecting individual cases.

The earlier “operator trace unavailable” result is invalidated: the extractor looked for
dictionary nodes, while the official CLEVRER multiple-choice code concatenates choice-level
string-token programs with a question-level token sequence and its executor recognizes
registered module-name tokens as operators. Other strings are runtime values/control tokens.
See official [`run_mc.py`](https://github.com/chuangg/CLEVRER/blob/master/executor/run_mc.py)
and [`executor.py`](https://github.com/chuangg/CLEVRER/blob/master/executor/executor.py).
The local audit now extracts only the official registered operator tokens, discarding
literal strings, and includes a synthetic regression test for this token-list schema. No GPU
was needed. The resulting operator and wording audits are reported above. Even a trace match
is only a structural filter, not proof of an isolated semantic intervention or explanation
faithfulness.
