# QTA generated-text proxy v2 feasibility and descriptive scoring (2026-10-19)

## Purpose

Test whether the prospectively specified exploratory v2 three-way-disjoint
QTA/evidence/control selector can produce four eligible real-video rows from
its fixed candidate pool and pre-outcome eligibility rule. The selector-only
pass determined the realized smoke/descriptive rows before generation and
scoring. A separately implemented scorer then ran one engineering smoke and
three descriptive rows. Neither stage assesses hallucination or measures
direct explanation faithfulness.

## Frozen setup

- Protocol: [`qta_generated_text_annotation_dependence_protocol_v2_exploratory.md`](../docs/qta_generated_text_annotation_dependence_protocol_v2_exploratory.md).
- Candidate lock: `qta_generated_text_candidate_cohort_local_20261019_01.json`,
  417 ordered candidates; v1-exposed indices 0–3 excluded from the v2 pool.
- Engineering smoke: lock index 0 only. Pool: indices 4–15, at most 12; stop
  after the first four eligible rows. No backfill or pool expansion.
- Eligibility: fixed question-attention QTA top group(s), reject QTA/evidence
  overlap, choose equal-count rank-distance control groups outside QTA/evidence,
  and require equal expanded E/C key budgets and pairwise E/Q/C disjointness.
- Model/environment: local pinned Qwen2.5-VL-7B-Instruct, 4-bit weights,
  BF16 compute, eager attention, PyTorch 2.11.0+cu128, Transformers 5.18.0,
  CUDA 12.8, NVIDIA GeForce RTX 5070 Ti (15.89 GiB reported).
- Candidate lock, dataset, preflight, source, smoke receipt, and pool output
  stayed local. Rights status remains unverified. Private receipts use mode
  0600 under the mode-0700 local private directory.

## Verified outcomes

| Stage | Requested | Attempted | Completed / eligible | Exclusions | Runtime |
|---|---:|---:|---:|---|---:|
| Engineering smoke, index 0 | 1 | 1 | selector 1 / eligible 1 | 0 | 38.58 s |
| Fixed pool, indices 4–15 | up to 12; require 4 | 7 (indices 4–10) | geometry 7/7; selector 7/7; eligible 4 | 3 | 80.39 s |

The pool exclusions were two rows with no eligible equal-count disjoint control
and one row where QTA overlapped the evidence groups. It stopped at four
eligible rows, indices 7–10; indices 11–15 were not attempted. Peak allocated
VRAM was 7.35 GiB. No CUDA OOM or memory-guard failure occurred.

The primary agent independently checked the receipt permissions, source/data
hashes, lock/preflight/smoke linkage, candidate denominator, eligibility
selection, and reproduction of geometry hashes, QTA top-group selection,
rank-matched controls, and expanded key disjointness/budgets for all four pool
rows. A separate auditor verified both receipts, the same hash links, the
smoke row, and all four eligible pool rows. Both receipts state generation,
target scoring, direct faithfulness, and faithfulness result are false.

## Interpretation and limitations

The fixed pool had sufficient selector feasibility: four eligible rows were
found after seven attempts. Eligibility depends on both model attention and
annotation geometry and is therefore a selected subset, not a randomized
sample. The candidate history is not certified fresh because prior exposure
records are incomplete. `gsub` is QA timing rather than independent semantic
support for a generated claim. Thus this run is an engineering/protocol gate,
not evidence of faithfulness or efficacy. The earlier v1 annotation-dependence
results remain closed and unchanged.

## Faithfulness progress and next action

Direct faithfulness remains **0/3 claims reviewed and 0/3 directly scored**;
player-clock/decoder-frame correspondence remains **0/15**, and prediction-link
review **0/3**. There is no direct `ΔF`, confidence interval, or test. The
informal nickname-to-question artifact has no submitted response and does not
fill any formal review gate. The v2 document is a prospective exploratory
design; no external registration record was found in the inspected sources.
Its fixed candidate pool and endpoint are closed after the observed results.
The v2 protocol/log files are untracked in the current worktree, and no
immutable pre-score protocol hash or freeze receipt is recorded. The dated
prospective-design entry and execution sequence support that the selection
rule preceded the v2 scores, but do not cryptographically prove the exact
pre-score document bytes. The three realized rows were chosen by selector-only
eligibility from the fixed pool, not named as a fixed three-row sample before
that pass.

### Scoring smoke and descriptive rows

After the selector receipt passed independent audit, a source/data/receipt-bound
scorer froze one baseline continuation per row and scored the same raw token IDs
under baseline/QTA × evidence/control ablations. Smoke index 7 completed one
generation and 4/4 condition forwards (12 target tokens) in 34.88 s; its
`ΔA` was +0.013513 nat/token. The fixed descriptive rows 8–10 completed 3/3
generations and 12/12 forwards without failures in 69.51 s. Peak allocated VRAM
was 7.44 GiB. Their `ΔA` values were +0.033096, +0.004754, and +0.044470
nat/token: mean +0.027440, median +0.033096, range [+0.004754,+0.044470],
with 3 positive signs. These are exact descriptive values for a selected n=3
finite cohort; no inferential interval or test is warranted. Two rows had
negative evidence-vs-control margin under both model conditions, so positive
`ΔA` must not be summarized as an absolute gain in evidence dependence.

The primary agent independently recomputed token counts and all four condition
scores from the private receipts; the independent auditor checked the hash
chain, cohort counts, score vectors, and recomputed deltas. Receipts remain
local, mode 0600, and contain no raw question, generated text, or absolute
paths. Rights remain unverified.

This closes the fixed v2 proxy cohort; do not expand or tune this endpoint based
on its observed signs. The result is still only conditional likelihood
dependence on benchmark QA timing annotations under visual masking. It does not
validate generated claims, establish hallucination, or show direct faithfulness
improvement. A lightweight nickname-to-question impression artifact is
available, but no responses have been submitted and it cannot replace
claim-linked review. Direct progress therefore remains **0/3 claims reviewed,
0/15 clock/frame correspondences, 0/3 prediction-link reviews, and 0/3 direct
scores**, with no direct `ΔF`, CI, or test. The nickname-first impression
artifact was subsequently simplified and browser-smoke-tested, but no human
response has been received. Keep this v2 proxy endpoint closed. Direct `ΔF`
still requires independently trusted claim-linked video review, verified
clock/frame correspondence, baseline `p/q`, and prediction-link adjudication.

## Provenance wording correction (2026-10-19)

The original purpose line called v2 “preregistered.” The protocol file labels
itself a prospective exploratory design and is not an external registration;
that wording overstated the available provenance. Corrected the status to
“prospectively specified exploratory v2.” This changes no cohort, scores,
eligibility, or interpretation; the completed finite endpoint remains closed.

## Informal response status correction (2026-10-19)

A later local import audit found one unique informal respondent; two JSON exports
had the same nickname and ordered item judgments and are deduplicated. This
corrects the earlier statement that no preliminary response existed. The saved
response has three claim-truth impressions but no structured sufficiency choice.
The user's separate item-1 clarity and item-3 sufficiency comments are retained
as informal qualitative feedback. This does not change any formal review or
faithfulness count above, and the closed v2 proxy endpoint remains closed.
