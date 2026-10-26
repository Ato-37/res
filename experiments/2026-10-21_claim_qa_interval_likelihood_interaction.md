# QA interval mask × saved-claim likelihood interaction (2026-10-21)

## Question and scope

This post-result exploratory diagnostic asks whether one fixed internal visual
attention-logit bias changes saved-claim token-likelihood sensitivity to the
dataset's QA timing annotation versus an equal-count temporal-control mask.
For each row, the sensitivity is
`mean claim logprob(control mask) - mean claim logprob(QA interval mask)`;
the reported interaction is the biased sensitivity minus the baseline
sensitivity. Positive values would indicate an increase in this diagnostic
under the intervention.

The cohort is the same three frozen validation/development rows used in the
prior local sensitivity checks. The QA annotations are question-level spans,
not verified claim-specific evidence. Generated claims were elicited from the
same question/options, the claim was re-tokenized rather than scored from its
original generation token IDs, black fill is out-of-distribution, and the
global bias affects all expanded visual tokens including those from masked
frames. This does not establish claim truth, support, or direct explanation
faithfulness. This is a single selected-cohort exploratory measurement, not a
confirmatory replication or population estimate.

The user waived human-review collection as an experimental prerequisite for
local work. No human review was performed or recorded (0/3). Rights remain
unverified. Video, source packet, detailed receipts and derived results remain
local and are not suitable for publication.

## Frozen inputs, setup, and run

- Exactly three previously fixed unique validation rows; no substitutions.
- Per row, QA interval and nearest-rank equal-count disjoint control were
  selected from the same 16-frame decode using strict presentation-time
  inclusion. CPU preflight confirmed 2/2, 4/4 and 8/8 interval/control frames.
- Conditions: QA interval black-mask and equal-count temporal-control
  black-mask, each scored at baseline and with the existing `+ln(2)` bias on
  visual-key logits in layers 5–16 for all claim-token query rows.
- Four teacher-forced forwards per row; total 12. Existing pinned
  Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
  4-bit weights, BF16 compute, eager attention, seed 20261021, offline local
  checkpoint, RTX 5070 Ti. Runner enforces VRAM/RAM checks, checks exact frame
  pixel and temporal provenance, unchanged non-visual prompt tensors, finite
  score windows and per-layer hook masks.
- The independent pre-run protocol audit judged one narrowly labeled run
  feasible, while emphasizing the QA-span/claim mismatch and post-result
  selection. It recommended no further repeat on this cohort.
- New implementation: `scripts/run_claim_qa_interval_likelihood_interaction.py`
  and `scripts/analyze_claim_qa_interval_likelihood_interaction.py`.
  Regression coverage: 13 tests passed across the runner, analyzer and prior
  interaction join checks. The three media preflight decodes and interval
  control constructions also passed.

## Results

All three rows completed all four conditions: **3/3 rows, 12/12 forwards,
zero failed or excluded rows**. The independently recomputed interaction
values were:

| Frozen row | QA/control masked frames | Baseline control−interval | Biased control−interval | Interaction (nat/token) |
|---|---:|---:|---:|---:|
| smoke | 2 / 2 | −0.036400 | −0.036575 | −0.000174 |
| remaining 00 | 4 / 4 | +0.000842 | −0.016430 | −0.017272 |
| remaining 01 | 8 / 8 | +0.200079 | +0.162745 | −0.037334 |
| **Finite-cohort mean** | — | — | — | **−0.018260** |

Median interaction was −0.017272 nat/token; range [−0.037334, −0.000174].
No row was positive. No CI, hypothesis test, p-value, or population inference
was computed. Across these three fixed rows, the chosen global visual bias
did not increase differential saved-claim likelihood sensitivity to the QA
timing annotations; all observed interactions were slightly negative. This
does not show the bias reduces faithfulness, because this endpoint is not a
claim-grounded faithfulness metric.

Private receipt SHA-256 values:

- `claim_qa_interval_likelihood_interaction_20261021_smoke_01.json`:
  `17e219600855bbcbb10ba544d896ea866421aa89cfaece404b9b13d5ed577363`
- `claim_qa_interval_likelihood_interaction_20261021_remaining_00.json`:
  `c1e8e051f322f47778281b4c23f8158fdd68b76ca622c0a4111ed772e6a07365`
- `claim_qa_interval_likelihood_interaction_20261021_remaining_01.json`:
  `85387b4aedcb7a50470fc8473690904dec0287760ae18537487313f8ca694df1`

The analyzer re-computes each DID from the four raw condition means, rejects
bad condition sets or hook counts, and excludes incomplete receipts rather
than treating partial data as a score. The successful receipt set contains
three unique row tokens and no excluded rows. All receipts are private local
files; do not stage or publish them.

An independent read-only audit separately rejoined the three rows to their
private packets and verified video/claim/packet/row-token hashes, unique video
digests, 0/1 hook counts in all 12 targeted layers for baseline/biased
conditions, equal frame counts, source hashes, 0600 permissions and DID
arithmetic. No mismatch was found. The auditor also confirmed that the
attention hook checks actual mask values at every target-query × visual-key
cell, not only hook invocation counters.

## Interpretation and next action

Together with the earlier fixed-row gray-video diagnostic (mean original minus
gray `+0.889496` nat/token) and global-bias × gray interaction (mean
`−0.003816` nat/token), this confirms that the model's likelihood reacts to
visual content but gives no evidence that the tested global intervention
improves the tested content-dependence measures. The frame-mean control also
had a positive original-versus-control contrast (+0.899918 nat/token), but it
shares the same rows and original-video scores; it is not an independent
replication.

Do not continue outcome fishing on this three-row cohort, reopen the closed
head/layer/alpha search, or revive the already closed QA-margin branch. The
primary research gap remains a preregistered claim-specific target with an
independent source of claim-to-frame validity and a fresh, adequately sized
cohort. Human-validated direct faithfulness remains **0/3**, and there is no
direct `DeltaF`, uncertainty interval or confirmatory estimate. Until a fresh
target and cohort are available, useful next work is a construct/protocol
design audit and a broader power/measurement plan; no additional GPU score on
these same three rows is justified by this result.
