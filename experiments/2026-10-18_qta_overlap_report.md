# QTA selector/evidence overlap reporting check (2026-10-18)

## Purpose

The nine-condition QTA engineering runner reported selector/sham overlap but
omitted overlap with the evidence and control groups. The QTA protocol requires
these intersections because overlap affects how an attention-bias treatment
composes with the internal key-ablation branches. This work adds an
outcome-independent overlap report and verifies its runner wiring on one
synthetic GPU fixture.

## Implementation and verification

`temporal_group_overlap()` now derives selected visual keys from one shared
temporal-group partition, then reports group and key intersection counts and
fractions over both sets. The runner emits QTA-selector/evidence,
QTA-selector/control, sham/evidence, and sham/control summaries. The helper
rejects empty, duplicate, unsorted, negative, overlapping, or out-of-range
group/key layouts. An independent audit confirmed that fractions use
nonempty denominators, that selector/evidence/control sets at the call site
come from the same partition, and that overlap bookkeeping does not enter
selection or scoring.

- Focused QTA runner suite: **18/18 passed**.
- Full scripts test suite on the final code: **1,561/1,561 passed** in
  **35.626 s**; Python compilation and `git diff --check` passed.
- Synthetic GPU smoke: **1/1 record, 9/9 forwards**, 45.034 s on the RTX
  5070 Ti, Torch 2.11.0+cu128, CUDA 12.8, Transformers 5.18.0,
  bf16/4-bit-bf16 eager attention. Peak allocated/reserved VRAM was
  **10.755/12.439 GiB**. The private run artifact is mode `0600` in a mode
  `0700` directory, SHA-256
  `527074d1e2759ef1ba9330a3a920ed13a5d09937e085b38fd677d5244bee2a7a`.
- The new output contains all four overlap summaries. In this synthetic
  fixture, QTA/evidence shared 1/2 groups and 256/512 keys; QTA/control shared
  0/2 groups and 0/512 keys. Sham/evidence shared 0/2 groups and 0/512 keys;
  sham/control shared 1/2 groups and 256/512 keys. These assignments have no
  annotation or semantic meaning.
- The sanitized public summary is
  [`results/synthetic_qta_overlap_audit_gpu_20261018_0728.json`](../results/synthetic_qta_overlap_audit_gpu_20261018_0728.json),
  SHA-256 `b186849c07159a79b02e7fee2613e892ec3217a51046a7ed70d35160a1de1ef6`.
  It excludes prompt/QA text, row IDs, token IDs, per-branch logits, local
  paths, and model-weight hashes. The private run hash and four source hashes
  are recorded for provenance.

## Interpretation and limits

This is GPU-backed execution of reporting and hook/scorer wiring on a synthetic
fixture. It does not measure claim-linked evidence dependence, validate human
review, or estimate treatment efficacy. The selected synthetic groups are
arbitrary and cannot be interpreted as real evidence. `faithfulness_result`
is `false`; there is no real `DeltaF`, interval, or hypothesis test. The
production cohort runner must use the same shared-partition overlap helper and
must keep overlap descriptive rather than use it for exclusions, subgroups, or
post-outcome treatment changes.

Direct research progress remains **0/3** independent claim reviews, **0/3**
prediction-link adjudications, **0/15** verified player-clock/decoder-frame
correspondences, and **0/3** direct scores; no empirical `DeltaF`, CI, or test
exists. Human/trust/timing gates and the production claim-mapped scorer remain
the blockers to a real-data run. Rights remain unverified; protected data and
review artifacts remain local.

## Provenance

- Runner: `c502992faece8bbf22873a1693eed6a7f02d4aeab3ea078f8fef9921ba8818cd`
- Focused tests: `a544f52b47e82c06733127224b5b63e164f22baf7f0eb4f050c9dccc2a4386ad`
- QTA selector: `3f56fe1e992ffca5a035864ced7472cbe1f12af9978c1559d1b82e0b938712f8`
- Fixed-prediction scorer: `84ba18241668fd659ddce0ab25a4e3dfa2cacc656ef43fb494fdc056a5c21744`
