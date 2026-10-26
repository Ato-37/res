# Source-attributed prior-exposure ledger (2026-10-19)

## Purpose

The existing prior-exposure indexes kept HMAC identities only as global unions,
so a validation-video match could not be attributed to a listed source. Build a
private source-attributed ledger for the explicit historical roster and audit
the current NExT-GQA validation split by canonical `video_id`. Keep video-ID,
media-byte, and video/question identities in distinct HMAC domains. This is an
outcome-blind provenance repair, not a faithfulness experiment.

## Setup

- CPU only; no video frames decoded, no model loaded, and no GPU used.
- Processed the existing 45-source local roster plus the previously omitted
  30-row validation-pilot source (46 explicit source records total). Protected
  inputs and the HMAC key remained local. Rights status remains unverified.
- The ledger stores HMAC identities per source, source class/namespace, a keyed
  source fingerprint, and aggregate coverage flags. It does not store raw IDs,
  QIDs, media hashes, questions, answers, paths, or filenames. Ledger and roster
  are owner-only files (0600) in the private data directory (0700).
- Validation split input: `NExT-GQA/datasets/nextgqa/val.csv`; its canonical
  `video_id` column was HMAC-matched against sources whose declared namespace is
  `nextgqa`. Media-byte equality was not tested by this new ID audit.

## Observed results

- Ledger build: **46** source records; **44/46** had at least one usable
  identity. The union contains **242** canonical video-ID tokens; the prior
  45-source ledger's union contained 212, so the added pilot source contributes
  30 IDs. The video-ID, media-byte, and video/QID HMAC unions exactly reproduce
  the prior aggregate index for the original 45 sources.
- Current validation split: **567** distinct video IDs; **30** match the
  source-attributed union. All 30 matches come from one listed `sample-json`
  source; the remaining 537 IDs are unmatched only to these explicit sources.
- A second, existing aggregate-only HMAC auditor independently returned **30**
  video-ID matches and **30** video/QID-pair matches against the same
  validation-pilot supplement.
- Source-attribution and HMAC separation tests passed. Focused combined tests:
  **38/38** passed. Python compilation and scoped `git diff --check` passed.

## Interpretation and limitations

This repairs attribution for known local use and confirms that the added pilot
source overlaps the validation split. It does **not** establish complete
historical coverage, key/root authenticity, or freshness of unmatched videos.
The 537 non-matches remain unknown; they are not certified unused. The ledger
does not establish media-byte identity, event independence, reviewer validity,
clock/frame correspondence, or scoring authorization. No model score or direct
faithfulness estimate, interval, or test was produced.

Direct faithfulness progress remains **0/3** formal claim reviews, **0/15**
verified clock/frame correspondences, **0/3** prediction-link reviews, and
**0/3** direct scores. GPU execution remains ineligible pending trusted human
review and the frozen evidence gates.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python scripts/build_nextgqa_prior_exposure_source_ledger.py \
  --roster data/private_nextgqa/prior_exposure_inputs_source_ledger_run02.json \
  --secret-file data/private_nextgqa/prior_exposure_hmac_key_20261016_01.bin \
  --output data/private_nextgqa/prior_exposure_source_ledger_run02.json

PYTHONPATH=scripts .venv/bin/python scripts/audit_nextgqa_source_ledger_split.py \
  --split-csv NExT-GQA/datasets/nextgqa/val.csv \
  --source-ledger data/private_nextgqa/prior_exposure_source_ledger_run02.json \
  --hmac-key data/private_nextgqa/prior_exposure_hmac_key_20261016_01.bin
```

## Next action

Retain the source-attributed local ledger as a candidate-history aid, continue
reconciling additional explicit history sources, and keep the 537 unmatched
validation IDs freshness-unknown. In parallel, obtain the missing trusted
reviewer assignments and complete the fixed review/timing gates before any
direct real-video GPU scoring.
