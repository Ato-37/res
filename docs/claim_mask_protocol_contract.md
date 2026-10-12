# Claim-mask protocol contract (schema v1)

`scripts/validate_claim_mask_protocol.py` is a CPU-only, fail-closed structural
validator for a future claim-selective dependence run. It requires separate
hashes for claim extraction, claim-to-intervention mapping, completed human
review, target/answer text, triplet manifest, sampling, selector, and control.
It also requires whole-video sampling, explicit baseline correction, and a
runtime declaration with execution authorization set to false.

Passing means only that declarations are structurally complete. The validator
does not inspect private claims or videos, authenticate reviewers, verify hash
targets, establish semantic alignment, or authorize GPU inference. NOAA review
and NExT-GQA rights remain independent gates; this contract does not satisfy
either gate.

CPU check:

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest scripts/test_validate_claim_mask_protocol.py -v
```
