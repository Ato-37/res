# QTA v4 union-mask full geometry census `_03` (2026-10-18)

## Purpose and setup

Run the outcome-blind geometry eligibility census for the frozen, test-associated
147-video v2 evaluation partition. For every interval-valid QID in that
partition, compare annotation-union token groups under the actual-PTS video
processor geometry and scalar-FPS fallback. Select the minimum prespecified
SHA-256 eligible QID per video before model outcomes. This is the required
cohort gate for the v4 scorer, not a score or faithfulness experiment.

The run used the repository `.venv`, cached pinned Qwen2.5-VL processor
configuration, CPU decode/processing, seed-free deterministic hash selection,
and local protected NExT-GQA test media/annotations. No model weights were
loaded, no CUDA call or model forward was made, and no scores were produced.
Rights remain unverified; all receipts and derived records stay local.

Command executed:

```bash
.venv/bin/python scripts/run_nextgqa_qta_v4_union_geometry_census.py \
  --n 147 \
  --census-output-name nextgqa_tlvs_qta_v4_union_geometry_census_local_20261018_03.json \
  --plan-output-name nextgqa_tlvs_exploratory_split_v4_union_20261018_03.json \
  --user-authorized-local-only
```

## Observed result

- Requested/completed videos: **147/147**.
- Candidate QIDs attempted/completed: **702/702**; eligible: **698**.
- Ineligible candidates: **4**, all due to
  `processor_implied_grid_changes_group_assignment` (actual-PTS versus
  scalar-FPS grouping disagreement).
- All **147/147** videos had at least one eligible QID; minimum/median/maximum
  eligible candidates per video were **1/5/12**.
- Minimum-hash selection was reproduced for all 147 plan rows. Each selected
  geometry digest and selected-row binding matched the census. The scorer's
  selected-item loader also revalidated all 147 media hashes and row bindings.
- Selected annotation-union coverage was **727/2,352 temporal groups** in
  aggregate (mean row coverage **0.30910**, median **0.25**, range
  **0.0625–1.0**). Four rows had full union coverage and 143 partial coverage;
  no row was excluded for coverage.
- Runtime: **612.20 seconds** on CPU.

The independent auditor separately recomputed the video/QID denominators,
per-video feasibility, minimum-hash selection, geometry and row bindings,
partition ordering, and source/data hashes. No discrepancy was found.

## Provenance

- Private census: `data/private_nextgqa/nextgqa_tlvs_qta_v4_union_geometry_census_local_20261018_03.json`, mode `0600`, SHA-256 `bd4814b231c71b936e043f73e200941b0e623f37c51219b9d68c0cd7d30bdce4`.
- Private plan: `data/private_nextgqa/nextgqa_tlvs_exploratory_split_v4_union_20261018_03.json`, mode `0600`, SHA-256 `482ffbfc2eb3c8f260a5d70dd26428f29bebd91ff40c7948024a8efddebe8b99`.
- The plan binds v2 plan SHA-256 `18dd22aba614464db404555337ee126d45a7bccaf1e76f21e16a263d283317f9`, v2 preflight SHA-256 `e5b23566bb48e0e6f1f7fcd68b9b2bb261473985c5bd867df7f35c95de5158d5`, and the exact census digest above.
- The census receipt contains 24 source hashes and 4 dataset hashes; the primary agent recomputed these against current files, and the independent audit confirmed the full sets match.

## Interpretation and limitations

The geometry gate passes for the frozen 147-video membership without drops.
This establishes processor-geometry feasibility and a reproducible pre-outcome
QID selection for this cohort. It does not establish annotation validity,
exposure freshness, model behavior, treatment benefit, generated-claim
faithfulness, population generalization, or statistical significance. The
protocol remains a test-associated exploratory annotation-mask dependence
proxy; full annotation masking can create out-of-distribution inputs.

**Faithfulness progress:** direct claim-linked scoring remains **0/3**; no
generated explanation was reviewed, no prediction-link judgment was completed,
and no direct effect/uncertainty/test was computed. This census is a geometry
engineering gate only. Existing earlier annotation-anchor proxy and synthetic
arithmetic checks remain auxiliary diagnostics, not direct evidence.

## Next action

Run the frozen 147-row CPU geometry preflight using the current `_03` plan and
require all geometry/row bindings to match. After independent receipt review,
run the one-row four-forward CUDA/BF16 smoke with fresh checkpoint byte audit.
Proceed to the 147-row scorer only if that smoke receipt passes all exact
counter, census, source/data, processor, checkpoint, and plan-binding checks.
Keep every raw/result artifact local.
