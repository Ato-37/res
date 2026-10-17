# QTA internal intervention score-path smoke (2026-10-18)

## Purpose and setup

Integrate the question-conditioned attention bias, answer-query visual-key
ablation, stable SHA-256 sham selection, and frozen baseline answer/foil score
arithmetic in one GPU path before using reviewed NExT-GQA evidence. This was a
synthetic one-record smoke, not a real-data experiment. Qwen2.5-VL-7B-Instruct
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` ran in bf16 with 4-bit
weights/bf16 compute on an RTX 5070 Ti. The test used nine model forwards.

## Verified result

The smoke completed 1/1 synthetic record and 9/9 finite forwards in 36.57 s;
peak allocated/reserved VRAM was 10.75/12.44 GiB. Baseline-full froze option
indices D/E, and both target and sham branches completed the same fixed-label
margin calculation. The synthetic `ΔF_QTA` and `ΔF_sham` were each
`−5.96×10⁻⁸` nat, with difference 0. This is floating-point residue around
zero on a synthetic clip and has no scientific interpretation.

The sanitized, source-hash-bound summary is
[`results/synthetic_qta_internal_score_path_smoke_20261018.json`](../results/synthetic_qta_internal_score_path_smoke_20261018.json), with SHA-256
`abc38e33de74de3828f7d936eb10255b6bb878888e223099163291aee497eeb2`. It
contains no video, question/answer text,
review content, or local dataset path. The private source artifact's SHA-256 is
`84d3e9360d617b4cb15d0885b4f8b7b5fa2c6ef9e8a09b09a14a0cee8a10637d`. An
independent auditor recomputed the score arithmetic and verified all 21 source
hashes in the result. The primary independently recomputed the 16 repository
source hashes and sham assignment. Cached model weight bytes were not hashed.

Focused scorer/hook tests passed 6/6; the complete suite passed 1,465/1,465
in 30.55 s. Python compilation and `git diff --check` passed.

## Interpretation and limitations

This establishes a working synthetic numerical and hook-composition path only.
No human-validated evidence/control groups or real videos were scored; the
intervention is not frozen; review and browser-to-decoder time gates remain
incomplete. Direct faithfulness evidence remains **0/3 reviewed, 0/3 scored,
0/15 verified clock correspondences, with no real `ΔF`, confidence interval,
or test**. Rights remain unverified, and protected data/results stay local.

## Next action

Implement the claim-mapped real-data runner and its three cohort-wide gates.
Run no real score branch until every record has a review export containing
self-attested judgments, exact-byte/trusted-roster adoption verification, and
verified player-clock/decoder-PTS correspondence. Obtain independent audit
before treating any real run as evidence.
