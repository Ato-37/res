# Map all reviewed claim occurrences to Qwen temporal groups

## Purpose

The offline claim-review procedure asks a reviewer to mark every distinct
sufficient occurrence of a claim, including separated occurrences. The
existing processor mapper accepted only one interval, so it could not represent
the union without either losing occurrences or masking the entire gap between
them. This change adds an explicit multi-interval path while retaining the
existing single-interval API.

## Implementation and checks

- Added `map_review_intervals_to_qwen_groups()`. It maps each interval with the
  existing midpoint-cell positive-overlap rule, unions the resulting temporal
  groups, preserves each occurrence's individual group list (including repeated
  identical intervals), and leaves intervening groups outside the evidence set.
  Shared inputs are normalized once with field-specific error codes;
  `temporal_group_count` is explicitly aliased to `evidence_group_count`, while
  `total_temporal_group_count` reports the full sequence length.
- Added an equal-count control selector. It selects non-evidence groups whose
  order-paired absolute rank-distance sum is minimal; equal-cost solutions use
  the lexicographically earliest tuple. A dynamic program avoids combinatorial
  enumeration, and the function fails closed when there are too few non-evidence
  groups.
- Integrated the plural mapper into `map_review_interval_from_processor_batch()`.
  It now accepts exactly one of a legacy single interval or the complete
  multi-occurrence list, maps from the cloned CPU batch, retains occurrence
  rows, and binds the full ordered interval list into the sampling metadata
  digest. A synthetic four-group batch verifies two disjoint occurrences,
  their two-group union, equal-count control, and equal token budget.
- Combined mapper suites passed **16/16** and the complete script suite passed
  **1,456/1,456**. The dynamic program matched a brute-force subset oracle for
  every feasible evidence subset over temporal-group counts 2–8
  (296 evidence-set cases). Tests also cover disjoint occurrences, overlap
  union, no gap filling, duplicate occurrence preservation, iterable inputs,
  malformed-input classification, digest changes for modified/reordered
  occurrence lists, and no-control failure. An independent audit reproduced the
  16 focused passes. The sampling digest binds supplied metadata and batch bytes;
  helper/source version must remain in the higher-level run manifest.

## Bounded synthetic GPU integration

After the CPU checks, extended the existing synthetic internal-ablation runner
to pass two separated intervals through the actual pinned Qwen processor,
processor-batch snapshot, mapper, and GPU ablation hook. The GPU job ran one
synthetic record with a baseline and ablation forward (2/2 complete, finite
logits), no scoring or efficacy estimate. The two occurrences mapped to groups
`[2,3,4]` and `[12,13]`; their union was five groups and the equal-count
control was five disjoint groups. Each arm therefore covered 1,280 visual keys
in this synthetic processor layout. Runtime was 22.06 s; peak allocated/reserved
VRAM was 10.75/12.44 GiB on the RTX 5070 Ti. Four independently recomputed local
source hashes matched the result. The aggregate SHA-256 is
`3ec9c8bff79d650ca6abff115da00521357f6602be21a40c874826f76d461300`.

The full repository suite passed **1,456/1,456** both before and after the
runner fixture change; the changed fixture and both mapper suites also passed
**17/17** before GPU execution. The GPU run checks wiring only: synthetic intervals do not
validate reviewer evidence, claim truth, source-clock calibration, or direct
faithfulness. No direct outcome was scored; status remains **0/3 independently
reviewed, 0/3 scored, no real `ΔF`, CI, or test, and 0/15 browser-to-decoder
correspondences**.
  `py_compile` and `git diff --check` passed.

## Interpretation and limitations

This repairs the data structure needed to represent multiple reviewed temporal
occurrences and creates an outcome-blind, reproducible equal-budget control
selection rule. It establishes processor-layout geometry only. It does not
authenticate the reviewer's identity or viewing, validate claim semantics,
prove browser-clock to decoder-PTS correspondence, or measure prediction
dependence. The control rule is a proposed deterministic study rule and must be
explicitly bound into the eventual scoring protocol before outcomes are seen.

Faithfulness progress remains **direct evaluation not performed**: independent
claim review **0/3**, scored **0/3**, no effect estimate/interval/test;
verified browser-to-decoder correspondence **0/15**. Rights status remains
unverified and protected inputs stay local.

## Next action

The next useful step is a fresh complete-suite check and scoped publication of
this code plus the synthetic aggregate. Then bind the plural-interval metadata
into the six-branch fixed-prediction treatment runner without changing any
observed treatment or score; human review, frame-pair evidence, and the
intervention/scorer freeze remain required before real scoring.
