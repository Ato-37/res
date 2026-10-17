# QTA local media byte-diversity and partial exposure audit (2026-10-18)

## Purpose

Before assembling any later confirmatory cohort, measure local test-media
availability, byte-identical duplicates, and matches against the existing
partial prior-exposure HMAC index. This is an outcome-blind data-integrity
diagnostic. It does not establish distinct source events, reviewed evidence,
claim eligibility, or faithfulness.

## Method and verified result

Added `scripts/audit_nextgqa_local_media_byte_diversity.py`. It streams mapped
local media in 1 MiB blocks for SHA-256, with bounded input sizes, stable-fd
checks, root-relative no-follow opens, and aggregate-only output. The prior
index and key are read locally and must be owner-only files. The audit does
not decode, display, copy, or emit media, identifiers, paths, question/answer
text, digests, or individual match records. The current supplied prior index is
known to be partial; its completeness is not authenticated.

Command:

```bash
PYTHONPATH=scripts ./.venv/bin/python scripts/audit_nextgqa_local_media_byte_diversity.py \
  --test-csv NExT-GQA/datasets/nextgqa/test.csv \
  --video-map-json NExT-GQA/datasets/nextgqa/map_vid_vidorID.json \
  --video-root /home/ato121/nextgqa_sample/NExTVideo \
  --prior-index data/private_nextgqa/prior_exposure_partial_20261016_03.json \
  --hmac-key data/private_nextgqa/prior_exposure_hmac_key_20261016_01.bin
```

Observed aggregate output from the final hardened implementation:

| Quantity | Result |
| --- | ---: |
| NExT-GQA test video-ID denominator | 990 |
| Mapped local media files found | 500 |
| Mapped local media files missing | 490 |
| Bytes streamed | 2,062,511,334 |
| Unique byte hashes among available files | 500 |
| Exact duplicate groups / extra copies | 0 / 0 |
| Available files matching supplied partial HMAC index | 85 |
| Distinct available media hashes matching that index | 85 |

The script reports the exact median file size as 3,421,173 bytes. Hashing
completed in approximately 1.2 seconds in this local run; this reflects the
current filesystem/cache state and is not a general throughput estimate.

Focused synthetic tests pass **7/7**, including duplicate accounting,
non-disclosure of IDs/paths/questions/answers/digests, private HMAC matching,
missing-file denominator preservation, unsafe path rejection, intermediate
symlink rejection, and stable/private fd-read limits. The independent auditor
reviewed the implementation without opening actual media or the actual key;
it independently checked component-wise no-follow opening, HMAC canonical
serialization, generic error output, even-count median logic, and the
synthetic suite. The final full `scripts` suite passed **1,620/1,620** in
**38.147 seconds** after stable-fd, strict-schema, and even-median changes.
Expected negative-path fixture messages and Transformers deprecation warnings
appeared. `git diff --check` and Python compilation passed, and the final
aggregate data audit reproduced the counts above.

## Interpretation and limitations

Of the 500 locally available mapped test media, none are byte-identical to
another available file under this test map. Eighty-five media bytes match
entries in the supplied partial HMAC index. Those 85 are defensible exclusion
candidates for a later cohort if the index's scope and matching policy are
adopted. The other 415 are **not** verified as unexposed because index
completeness is unknown. A byte hash cannot detect re-encoded copies, shared
events, overlapping source footage, or related scenes. The 490 missing local
files are local availability failures, not evidence that the dataset lacks
them or that their records are ineligible.

Rights status remains unverified. All media and prior-exposure materials
remained local; no protected material or per-item derivative was published.

## Faithfulness progress

- Direct faithfulness evidence remains **0/3** reviewed claims, **0/3**
  prediction-link judgments, **0/15** verified browser-player/decoder
  correspondences, and **0/3** real QTA scores. No empirical `DeltaF`,
  uncertainty interval, or hypothesis test exists.
- This audit adds a quantitative availability/exclusion diagnostic only:
  **500/990** mapped media available locally, **0/500** exact duplicates,
  and **85/500** matches to the supplied partial exposure index.
- It does not show that internal intervention improved selective prediction
  dependence on reviewed video evidence. It is not GPU work or a model
  experiment.

## Next action

Keep the 85 partial-index matches excluded from a new candidate roster unless
an outcome-blind protocol explicitly justifies otherwise. Do not treat the
remaining 415 as clean or independent. To advance the requested internal
intervention question, first complete the existing human review and
player-to-decoder correspondence requirements for the three feasibility
videos, then complete the outcome-blind phase/freeze gates. Continue useful
preparation of the production prebaseline readiness validator while those
independent review tasks remain outstanding; no real score is authorized by
this audit.

## Provenance

- Script SHA-256: `48dade10a60b71d6422d8e6697d59f5254db57dc86a1989a427ab470287b7ec8`
- Focused-test SHA-256: `5732cf70a47908880408c32c44a716f16f1859dd4fa27d4e64cbb8c10b59ed52`
- Inputs: local NExT-GQA test CSV and video-ID map; local mapped media tree;
  private partial exposure HMAC index and local HMAC key. Raw input hashes,
  media paths/IDs, per-item matches, and question/answer text are not recorded
  here.
- Environment: repository `.venv`, Python 3.12; CPU-only.
