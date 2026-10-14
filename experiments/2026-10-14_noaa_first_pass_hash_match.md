# NOAA first-pass export matching robustness (2026-10-14)

## Purpose

Reduce avoidable friction in the private NOAA human-review workflow while keeping
the first-pass export and separate gold-reference reveal in the same order.

## Change

The gold-reference page previously compared each source-hash object by comparing
its `JSON.stringify` output. JSON object member order is not semantically
significant, so a valid first-pass JSON file that had been reformatted or had its
keys sorted could be rejected even when all five source hashes matched. The page
now checks the exact five expected hash fields and values independent of key
order, while rejecting extra/missing fields. It still requires the first-pass
file before rendering keyed answers; the first-pass export itself is unchanged.

## Verification

Synthetic CPU-only packet and structural-contract tests pass: 16 tests. The packet
test confirms the generated gold page uses field-wise matching and no longer
compares serialized hash objects. No canonical NOAA dataset, annotation, ID, or
review packet was read or emitted for this change. No GPU or inference was run.

## Interpretation and limitations

This is a workflow robustness improvement only. It does not strengthen blinding:
the gold answers remain embedded in the separate local HTML file, and a human can
still open that file early or edit the first-pass JSON. It does not verify human
review, semantic correctness, chronology, reviewer identity, media rights, or
execution authorization. NOAA human review remains incomplete; direct faithfulness
progress remains 0 new samples and 0 new effect estimates.

## Next action

Keep the QA and rights gates closed until the actual human review and item-level
rights evidence are independently completed and validated. Do not treat this
synthetic test as evidence that a human review occurred.
