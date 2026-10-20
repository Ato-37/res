# QTA baseline freeze source-closure repair — 2026-10-21

## Purpose

Audit whether the review-only inventory pins the exact code that will produce the three-row full-video baseline and freeze its prediction manifest. This is required so later review evidence and `p/q` predictions are tied to the intended implementation.

## Change and verification

The current local worktree includes changes adding `runtime_sha256` to baseline run digests and rejecting duplicate source-video hashes in the p/q candidate builder. The inventory had not registered the baseline scorer, output-provenance validator, or p/q candidate builder, so those current implementations were absent from the inventory's static source closure. Added the three entry modules plus their local dependencies: the model-call boundary, attention-forward capture, choice-token binding, and temporal-attention selector. The inventory-builder test now requires these roles.

The inventory-builder suite passed **21/21**. A fresh actual-data candidate inventory contains **107 roles, zero missing slots** and remains `candidate_only`, `execution_authorization=not_granted`. Its bound CPU sampler passed on **3/3 videos**, sampling **32 frames/video** (decoded counts 510/1,140/1,258; maximum grid deviations 16.145/16.145/16.355 ms). The local actual-QA video-byte join again matched **3/3** rows, with no model load, forwards, score calls, or authorization. Separately, current release-roster, ceremony-signing, baseline-provenance, and p/q-candidate changes passed **41/41** focused tests.

## Interpretation and limits

This closes an inventory omission for the current implementation; it does not freeze the inventory, prove reviewer identity, establish claim truth, authenticate source/sampler provenance, or authorize execution. There are still no authentic role/release keys or external root pin in the local intake paths. Direct faithfulness remains **0/3 Stage-1 claim reviews, 0/15 frame correspondences, 0/3 Stage-2 prediction-link reviews, and 0/3 direct scores**. There is no effect estimate, interval, test, or multiplicity result. No GPU inference ran. Rights status remains unverified; private evidence and detailed receipts remain local.

## Next action

After the real reviewer roster, external root pin, witnessed review-only freeze, and Stage-1/frame evidence exist, rebuild the inventory and verify it binds these exact current scorer and p/q modules. Then implement the production atomic baseline-only runner against that frozen inventory and authorize exactly three full-video baseline forwards; Stage-2 prediction-link review remains required before any treatment/control forward.
