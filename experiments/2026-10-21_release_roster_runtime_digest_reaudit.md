# Review-role and baseline-run provenance re-audit — 2026-10-21

## Purpose

Verify the current local changes to the review release roster and baseline `p/q` candidate inputs before treating the refreshed source inventory as a meaningful freeze candidate.

## Verified implementation and tests

The release roster now requires a distinct `capture_operator` key alongside freeze owner, custodian, claim reviewer, frame reviewer, prediction-link reviewer, and release witness. The validator rejects capture-key reuse with each role; it still permits the documented freeze-owner/custodian alias while keeping the capture operator separate. Ceremony-signing CLI fixtures now exercise the additional role.

Baseline run digests now require `runtime_sha256` in addition to protocol, model, tokenizer, scorer, and processor hashes. The p/q candidate builder rejects duplicate source-video SHA-256 values even when review tokens differ. The associated focused review-roster, ceremony, provenance, and p/q suites passed **41/41** tests. These fixtures are synthetic and non-authorizing.

A separate fresh source inventory now registers the baseline scorer, p/q candidate, output provenance, and their local dependencies. The corresponding actual-data sampler and source-identity join passed **3/3 rows**, but those checks establish only local byte identity and decoder feasibility.

## Limits and direct faithfulness status

No real reviewer keys, signed rosters, external root fingerprint, witnessed handoff, or completed Stage-1 review were present in the private intake paths at this checkpoint. Synthetic signature tests do not authenticate actual people or a trust-root origin. Direct faithfulness remains **0/3 Stage-1 reviews, 0/15 frame-pair judgments, 0/3 Stage-2 prediction-link judgments, and 0/3 direct scores**. There is no effect estimate, uncertainty interval, hypothesis test, or multiplicity result. No model/GPU forward ran. Rights status remains unverified; private evidence stays local.

## Next action

Keep the current inventory candidate-only until real people and the out-of-band trust pin are established. Once the witnessed review-only handoff and Stage-1/frame evidence are complete, rebuild the inventory over the exact roster, receipt, reviewers' artifacts, baseline scorer, and runtime pins. Then implement the atomic production runner to validate all three rows before loading weights and permit exactly the three baseline forwards.
