# QTA v4 union-mask geometry census (2026-10-18)

> Historical run `_02`. The later full audited run is `_03`, documented in
> [the `_03` census log](2026-10-18_qta_v4_union_geometry_census_03.md).

## Purpose

Check whether the frozen 147-video test-associated cohort has at least one
outcome-blind QID per video whose all-annotation union matches the Qwen
processor's actual-PTS temporal-group geometry. The protocol and four-condition
estimand remain frozen in `2026-10-18_qta_v4_union_treatment.md`.

## Setup and verified result

The full census used the repository `.venv`, local NExT-GQA metadata and video
bytes, the pinned local Qwen processor configuration, and no model weights or
CUDA. It considered all interval-valid candidate QIDs, excluded only candidates
whose processor-implied scalar-FPS assignment disagreed with actual PTS, and
selected the minimum predeclared SHA-256 feasible QID for every evaluation
video. It preserved the original 147-video cohort and used private output names
`nextgqa_tlvs_qta_v4_union_geometry_census_local_20261018_02.json` and
`nextgqa_tlvs_exploratory_split_v4_union_20261018_02.json`.

Verified counts: 147/147 videos completed; 702/702 interval-valid candidates
attempted and completed; 698 candidates were geometry-eligible; every one of
the 147 videos had an eligible candidate. The four ineligible candidates all
failed the same fixed actual-PTS/scalar-FPS grid-agreement condition. The
selected plan contains all 147 videos and all 147 min-hash selections were
independently reconstructed from the private census. Runtime was 615.28 s.
The census and plan are mode 0600 under the mode-0700 local private-data root.

Across the selected records, 727/2,352 temporal groups were annotation-covered
(mean fraction 0.3091; median 0.25; range 0.0625–1.0). Four rows have full-group
coverage and 143 are partial; no row has an empty union. Full coverage is
retained as a mask property and is not an exclusion rule.

Primary-agent checks and an independent audit verified all-or-none counts,
current 24-source and four-dataset hash matches, v2 plan/preflight linkage,
the exact evaluation/calibration membership and order, the census-to-plan hash,
and every selected QID/geometry/row binding. The scorer's `selected_items`
loader now successfully reconstructs all 147 media-pinned items after a prior
`media_sha256` lookup defect was repaired. The current focused v4 tests pass
13/13; compilation and `git diff --check` pass.

## Interpretation and limitations

This closes only the geometry-feasibility gate for the v4 exploratory
annotation-union dependence proxy. QID selection is conditional on processor
geometry feasibility, the data are test-associated with prior exposure history
incomplete, and four rows fully mask the model's temporal groups. These counts
do not establish annotation validity, claim-to-frame correspondence, human
review, source-event independence, or generated-explanation faithfulness. No
model score, effect estimate, uncertainty interval, significance test, or GPU
work was produced. Rights remain unverified; all protected inputs and derived
receipts remain local.

## Next action

Run the separate all-147 CPU scorer geometry preflight and require every row to
match this plan. If it passes, audit the one-row four-forward GPU smoke against
the same plan/source/data/processor bindings before considering the full
588-forward local exploratory score. Human review and the claim-linked primary
remain separate, unresolved gates.

## Provenance

- Census SHA-256: `48505228570bcecb8de844e1789974891f2b6d8b43fdb531abd53af0327ace3e`
- Plan SHA-256: `f77f9add25eb88e7f3a24e8c81f8a15a5c3c5cf59192af5c46c28f27050615cf`
- Updated census source SHA-256: `0babbf7980959b452fa2f6a9da83223d239a176659ec48971ef18412c18874db`
- Updated scorer source SHA-256: `cc93f19e3eff68e773f584608ea1bd705cf20473a2b17117a4b3ec0931b35afe`
- Exact input, dataset, and processor provenance remains in the owner-only
  receipt and plan; raw IDs, paths, and media hashes are intentionally omitted
  here.
