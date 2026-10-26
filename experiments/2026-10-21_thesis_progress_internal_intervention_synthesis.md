# Graduation-thesis progress: internal visual intervention and faithfulness (2026-10-21)

## Research question and feasible thesis contribution

The intended contribution is to measure whether an internal visual-attention
intervention makes a generated explanation depend more on the video evidence
it claims to describe, separately from answer accuracy or general temporal
sensitivity. A feasible thesis title is **“動画VLMの根拠依存性をどう測るか：内部視覚介入のmatched-control評価”**. Do not claim novelty for token-conditioned
internal video evidence intervention itself: recent work already proposes
STEAR's layer-aware spatiotemporal evidence intervention and SEASON's
self-diagnostic temporal/spatial contrastive decoding. A related VLM study also
tests visual-attention faithfulness through causal sufficiency and
comprehensiveness. The thesis contribution must instead focus on validated
claim-level support measurement and its relationship to internal interventions
on real NExT-GQA videos; novelty remains to be established. See [STEAR](https://arxiv.org/abs/2604.03045), [SEASON](https://arxiv.org/abs/2512.04643), and [visual attention faithfulness in VLMs](https://arxiv.org/abs/2609.00830).
The defensible thesis at the present stage is an empirical measurement/negative
or inconclusive result: generic global steering can change internal/model
scores, but the current pilots do not show a stable increase in evidence
dependence. A future claim-specific experiment can test improvement, but this
log does not claim that improvement has already been demonstrated.

## Verified evidence available

The local validation/development and test-split experiments use
Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
4-bit weights, BF16 compute, and real video files. Results below are exploratory
finite-cohort measurements; cohorts were small or sequentially developed and
should not be treated as confirmatory.

| Measurement | Observed result | What it supports / does not support |
|---|---:|---|
| Test-split internal visual-key bias × original vs per-frame RGB-mean likelihood (3 videos; 6 new forwards) | Difference-in-differences +0.061967, +0.034255, +0.021600; mean +0.039274 nat/token; 3/3 positive | A concrete positive internal-intervention signal: the fixed global bias increased claim-token sensitivity to intact spatial pixel structure versus a severe uniform-color control. Same rows' baseline was reused after earlier outcome exposure; the control is OOD and not a claim-support interval. It is not direct faithfulness improvement. See `2026-10-21_test_split_internal_bias_frame_mean_interaction.md`. |
| Fixed-prediction QTA internal key ablation, cohort 12 (3 videos; 15 forwards) | Mean ΔF +0.04167 nat; per row +0.125, +0.125, −0.125 | A small mixed mechanistic signal on a machine-selected unreviewed window; not direct faithfulness or efficacy. See `2026-10-21_qta_fixed_prediction_internal_key_ablation.md`. |
| Internal visual-key bias × temporal-block reversal, validation cohort (3 videos; 12 forwards) | Historical wrong-prompt values −0.005938, +0.046906, +0.074168 are invalid for v2_short. Correct prompt-bound rerun: +0.004429, +0.057039, +0.003445; mean +0.021637 nat/token; 3/3 positive | Post-hoc prompt repair on the same outcome-exposed rows, not an independent confirmation. Temporal-order likelihood sensitivity only; no direct support label. See `2026-10-21_temporal_order_prompt_binding_repair.md`. |
| New test-cohort global visual-key bias × temporal-block reversal (3 videos; 12 forwards) | −0.022759, +0.008539, −0.016455; mean −0.010225 nat/token; 1/3 positive | Independent known-history local cohort with mixed/negative result; does not reproduce the earlier validation direction. Confirms generic global bias has no stable positive proxy signal here. See `2026-10-21_fresh_test_cohort_temporal_order_global_bias.md`. |
| New val v2-short history-screened global visual-key bias × temporal-block reversal (3 videos; 12 forwards) | Historical wrong-prompt values −0.018254, +0.021063, −0.013389 are invalid for v2_short. Correct prompt-bound rerun: +0.011883, +0.063553, +0.050534; mean +0.041990 nat/token; 3/3 positive | Post-hoc prompt repair on unreviewed machine-generated claims; cannot establish semantic support or stable intervention effect. See `2026-10-21_temporal_order_prompt_binding_repair.md`. |
| Same new val cohort: QA-interval pixel mask × baseline / QA-key bias / disjoint sham (3 videos; 18 full forwards) | `DeltaF` QA−baseline +0.06046, +0.03528, −0.04124; mean +0.01817 nat/token, 2/3 positive. QA−sham +0.03617, +0.07699, −0.00986; mean +0.03444, 2/3 positive. | Small mixed positive QA-interval sensitivity proxy on three local-history-screened rows; this is post-hoc relative to the temporal-order measurement on the same claims. The interval is benchmark QA annotation, not reviewed support for the claim. It does not establish semantic faithfulness. See `2026-10-21_freshval_qa_interval_claim_mask_bias.md`. |
| Same frozen QA-interval mask/key-bias estimand on next 10-row local-history-screened val batch (6 scorable; 36/36 forwards) | `DeltaF` QA−baseline mean −0.00355 nat/token, median −0.00151, range [−0.05460,+0.05759], 3/6 positive. QA−sham mean −0.00878, median −0.00926, range [−0.03425,+0.01111], 2/6 positive. | The preceding 3-row positive proxy mean did not replicate; combined small cohorts do not show a stable positive effect. Attrition: 10 requested, 1 claim-generation failure, 3 geometry exclusions, 6 scored. QA intervals remain unreviewed proxies, not direct faithfulness. No CI/test at n=6. See `2026-10-21_freshval_qa_interval_claim_mask_bias_cohort10.md`. |
| Same estimand on the following seven-row outcome-blind val continuation (7 scorable; 42/42 forwards) | `DeltaF` QA−baseline mean −0.00368 nat/token, median −0.00064, range [−0.05295,+0.03293], 3/7 positive. QA−sham mean +0.01474, median +0.01038, range [−0.04326,+0.08636], 5/7 positive. | Baseline effect remains near zero/mixed; QA−sham is positive in this small batch but heterogeneous and does not establish support validity or stable improvement. Across the prior n=3, n=6, and n=7 batches, primary means are +0.01817, −0.00355, −0.00368 respectively. Keep strata separate; no inferential pooling. See `2026-10-21_freshval_qa_interval_claim_mask_bias_cohort7.md`. |
| Claim-token attention-selected key bias × temporal reversal vs equal-budget disjoint hash sham (cohort 08; 3 videos; 18 forwards) | Key DID −0.017952, +0.037386, −0.008962 (mean +0.003491; 1/3 positive); key-minus-sham specificity +0.023010, +0.033725, +0.018543 (mean +0.025093; 3/3 positive) | Historical non-authorizing diagnostic only: treatment order effect is mixed and near zero; positive contrast partly reflects negative sham interactions. Runner ignored false scoring-authorization flags. Do not use as approved faithfulness or efficacy evidence. Receipt audit was not an independent model rerun. |
| Same fixed claim-token key intervention on disjoint cohort 09 (3 videos; 18 forwards) | Key DID +0.008707, −0.000724, +0.006115 (mean +0.004699; 2/3 positive); specificity −0.006880, −0.015693, +0.003055 (mean −0.006506; 1/3 positive) | Non-authorizing diagnostic; cohort08 pattern did not replicate. Across six rows, key DID mean +0.004095 (3/6 positive), specificity +0.009294 (4/6 positive), with no population inference. Direct faithfulness is unmeasured. |
| Global visual-attention bias × gray-video saved-claim likelihood (3 videos) | Difference-in-differences mean −0.003816 nat/token; 2/3 positive; range [−0.050003,+0.034167] | Near-zero/mixed; no evidence of increased visual-content dependence. |
| Global visual-attention bias × QA-interval mask / saved-claim likelihood (3 videos; 12 forwards) | Mean interaction −0.018260 nat/token; all 3 negative; range [−0.037334,−0.000174] | Does not support improved differential dependence on benchmark QA intervals; QA interval is not verified claim evidence. |
| New local-history-screened 32-frame QA-interval claim likelihood, sham protocol v2 (3 videos; 18 forwards) | QA-bias−baseline +0.039396, −0.036819, −0.038841; mean −0.012088, descriptive 95% t CI [−0.122875,+0.098699], 1/3 positive. QA-bias−sham +0.019521, −0.022711, +0.039426; mean +0.012079, CI [−0.066742,+0.090899], 2/3 positive. | Sequential exploratory; 0 local scored-video overlap in a partial/incomplete history. No stable proxy improvement; QA intervals are not claim-reviewed support. See `2026-10-21_claim_text_qa_interval_key_bias.md`. |
| Question-attention-guided internal visual-key bias on five-choice QA accuracy (27 reused videos; 81 forwards) | Baseline/QTA/matched sham all 15/27 correct; 0 wrong→correct and 0 correct→wrong baseline/QTA flips; predictions identical on all 27 rows. | No finite-cohort accuracy improvement; post-hoc reused sample, so no independent or population claim. Score-margin movement up to 0.375 nat occurred under QTA and sham. See `2026-10-21_qta_full_video_accuracy_diagnostic.md`. |
| Original video vs per-frame RGB-mean control for saved-claim likelihood (same 3 rows) | Mean +0.899918 nat/token, all 3 positive | Generated-claim likelihood reacts to video pixel structure; same rows as related checks and no semantic-support conclusion. |
| Gold QA interval masking, separate test cohort (10 videos; 50 forwards) | Black control−interval mean −0.2625 nat, SD 1.2183; blur mean −0.0125, SD 0.8589; 8/10 full-video QA answers correct | Unstable/no robust gold-interval margin advantage; explicitly not generated-claim faithfulness. No population CI or test. |
| Direct claim-support review for claim-conditioned cohorts 08/09 | 0/6 reviewed; 0/6 direct scores | The central outcome remains unmeasured. User permission to proceed with local proxy work does not turn likelihood diagnostics into human evidence labels. |

These values were read from and are reproducible through the dated logs
`2026-10-21_qta_fixed_prediction_internal_key_ablation.md`,
`2026-10-21_attention_bias_video_content_interaction_plan.md`,
`2026-10-21_claim_qa_interval_likelihood_interaction.md`,
`2026-10-21_saved_claim_frame_mean_control_plan.md`, and
`2026-10-21_test_gold_interval_margin_n10.md`. No inferential tests or
population-level claims are warranted for the selected exploratory cohorts.

Across the three disjoint sequential val batches above (3+6+7=16 rows), a raw
row-weighted descriptive aggregation is `DeltaF` mean **+0.00046 nat/token**,
median +0.00475, range [−0.05460,+0.06046], positive 8/16; QA−sham mean
**+0.00961**, median +0.01031, range [−0.04326,+0.08636], positive 9/16.
This is provided only as a transparent arithmetic summary: candidate selection
was sequential and geometry-conditioned, the proxy interval is not reviewed
claim support, and the batches are exploratory. It is not an inferential pooled
estimate or evidence of direct faithfulness improvement.

## Interpretation

There are useful thesis observations: (1) a global internal visual-key bias
increased the original-vs-frame-mean claim-likelihood gap on three fixed real
videos; (2) fixed-prediction QTA steering showed small, mixed effects; and (3)
other global-steering endpoints were near-zero or negative. Those pilot runs
have separate identification limits and do not establish faithfulness
improvement. The claim-conditioned attention-selected contrast also failed to
replicate consistently across two disjoint three-row cohorts. In addition, an
independent protocol audit found that scoring authorization was false and the
runner ignored that gate. Cohort08/09 values are non-authorizing historical
diagnostics, not protocol-approved efficacy evidence. The details are in
`2026-10-21_claim_conditioned_temporal_internal_intervention.md`.

The new claim-conditioned contrast adds a sharper thesis hypothesis: internal
attention-selected steering may alter claim-likelihood sensitivity differently
from an equal-budget sham. The initial 3-row proxy lift did not replicate on a
disjoint 3-row cohort: specificity changed from +0.025093 to −0.006506 nat/token,
while pooled key-only order effects averaged +0.004095 with only 3/6 positive.
This is useful negative replication evidence against the first proxy lift being
stable. It remains a result about a likelihood proxy, not valid video evidence
or improved explanations, and its non-authorizing status bars its use as a
formal study score.

The main alternative explanation is measurement mismatch: automated QA/model
windows may not be the event that supports the generated claim. Masking can
also create out-of-distribution visual artifacts; black and blur results
already differ. The new 32-frame sham-v2 local-history-screened cohort also
failed to show a consistent positive QA-bias effect (1/3 positive rows) or
stable specificity benefit (2/3 positive rows, mean near zero with a broad
interval). Small cohorts and incomplete history limit generalization. No
closed head/layer/alpha search should be reopened.

## Next discriminating experiment

Use a frozen, outcome-blind cohort with a **machine-proposed claim-conditioned
interval**. Unless a human later validates it, this remains an unreviewed proxy,
not a claim-specific evidence label. Keep the explanation text fixed. For each
row, branch `b ∈ {baseline, intervention}`, and mask `m ∈ {candidate, sham}`,
score the fixed claim under full video and under the equal-budget masked video.
Define `F[b,m] = LL(claim | full video,b) − LL(claim | mask m,b)`, then define
the primary specificity effect as
`Δ = (F[intervention,candidate] − F[baseline,candidate]) − (F[intervention,sham] − F[baseline,sham])`.
Positive Δ means steering increased claim-likelihood sensitivity to the
machine-proposed interval more than to the matched sham; it does not by itself
mean the interval is valid or faithfulness improved. Also record fixed-answer
margin and output stability as secondary diagnostics. Freeze cohort, claim,
interval algorithm, hook layers, steering magnitude, token score, exclusion
rules, and practical threshold before scoring. The primary label must explicitly
state “machine-proposed unreviewed interval.” A new 32-frame sham-v2 cohort has
now completed as sequential exploratory work with incomplete exposure history.
Do not rerun or pool it with other frame-count/protocol strata. Direct
claim-support remains 0/3 reviewed for the current artifact. The next
experiment that can answer semantic faithfulness requires claim-specific
support frames and a frozen support-key versus matched-sham comparison; absent
those labels, keep the thesis conclusion scoped to the negative/inconclusive
QA-interval proxy and its measurement limits.

## Human-review handoff browser check

The local artifact `data/private_nextgqa/claim_support_frame_review_test_20261021_06.html`
now discloses that the 16 images are resized decoded RGB previews, not the full
video or model processor crop/patches. Focused unit tests pass 8/8. The current
Windows Chrome E2E used synthetic answers for two consecutive reviewers and
three items each; it verified artifact binding, answer JSON payload, old-Blob
revocation, schema inspection, and actual filesystem download. The saved file
matched the in-page Blob and exact artifact fingerprint. This verifies the
browser handoff only, not semantic judgments, reviewer identity, or a valid
human review. A no-content scan of 52 candidate JSON files found 0 responses
bound to this artifact; actual human review remains 0/3. The HTML and response
remain local, owner-only; dataset/model rights are unverified. This status
supersedes the earlier E2E note below that filesystem download was unverified.

## Validation and next actions

- `.venv/bin/python -m unittest -v test_build_test_split_claim_support_frame_review`: 8/8 passed.
- `.venv/bin/python scripts/test_claim_support_frame_review_browser_e2e.py`: passed with two synthetic reviewers, 3 items each; filesystem download was verified against the final Blob payload and artifact fingerprint. CSP enforcement remains untested because the harness bypasses CSP while blocking HTTP(S).
- Current direct faithfulness for claim-evidence support remains **unmeasured**; human review is 0/6. The two attention-selected likelihood runs are non-authorizing and cannot substitute for evidence-grounded scoring.
- Cohort08/09 provenance, branch arithmetic, equal budgets, sham disjointness, layer hooks, local exposure-set hashes, and cohort nonoverlap were independently audited. No independent GPU/model rerun was performed. Rights remain unverified; receipts stay private.
- Next: resolve upstream review/trust-root and authenticated-capture requirements, then use a versioned runner that checks authorization before model load. Keep all protected data/results local.
