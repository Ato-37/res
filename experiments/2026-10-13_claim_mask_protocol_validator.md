# Claim-mask protocol validator (2026-10-13)

## Purpose

The NOAA human-review gate is incomplete and NExT-GQA redistribution rights
remain unresolved. A claim-selective GPU efficacy run is therefore blocked. This
cycle implemented a separate schema-v1 validator so a future private protocol
cannot silently reuse the internal-activation contract.

## Verified implementation

`validate_claim_mask_protocol.py` checks exact schema keys, model/runtime
declarations, split-overlap and video-level counts, completed claim-review
references, whole-video sampling, explicit claim/answer hashes and triplet
manifest, claim-mask selector/control hashes, and non-authorizing runtime
declaration. It emits only generic failure codes and an optional input hash.
Three CPU tests pass, including valid structural input, incomplete review, and
claim-conditioned sampling failures. No private data, human attestation, or
GPU inference was used.

## Interpretation and limits

This is an engineering gate, not evidence of claim faithfulness. A structurally
valid file still requires independently verified review, data rights, semantic
mapping, implementation audit, and a bounded smoke run before efficacy analysis.

## Next action

Have an independent audit review the schema and then keep the real-data branch
closed until NOAA attestation and dataset-rights gates are resolved.
