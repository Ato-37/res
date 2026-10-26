# Claim-likelihood proxy review artifact v2 (2026-10-21)

## Purpose

Create a simple local artifact for informal human inspection of the three
videos/claims used in the 32-frame sham-v2 exploratory likelihood run. This is
not a formal ground-truth annotation instrument and does not change the
machine-scored experiment into a direct faithfulness result. The question is
whether the explanation roughly matches the video; reviewers may also mark
partial mismatch or unable to tell.

## Frozen inputs and binding

Builder: `scripts/build_claim_likelihood_human_review_v2.py`.
The builder requires the exact 3-record/32-frame packet bytes to match the
full result's `cohort_sha256`; it also checks the three row-level video and
claim hashes before extracting any images. Each response packet binds the
artifact fingerprint, source packet/result hashes, and each item's video,
claim, and sampled-PTS hashes. The page shows the full local video and the 32
model-input frames with indices 0–31 and sampled timestamps. Multiple reviewers
can complete the same artifact, entering a nickname before each person's run.

The artifact is
`data/private_nextgqa/claim_likelihood_human_review_v2_20261021_01.html`,
mode 0600, 14,838,963 bytes, SHA-256
`67abd73231518d3b0e9e60af1e97fae866f0c40e00697efe563db0da217fe173`.
It embeds protected video and claim material and must remain local/private;
dataset/model rights remain unverified. The builder rejects outputs outside
the private data root, refuses accidental overwrite by default, and writes
through an owner-only temporary file followed by atomic publication.

## Verification and status

The command

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest -v \
  test_build_claim_likelihood_human_review_v2 \
  test_inspect_claim_likelihood_human_review_v2_response
```

passed **9/9** tests. The builder tests
cover exact fingerprint derivation, frame-index display, row binding failure,
private-root output restriction, mode 0600, and refusal to overwrite. Python
bytecode compilation and scoped whitespace checks passed. The artifact was
regenerated from the frozen packet/result and the final mode and SHA-256 were
checked independently. The response inspector validates the artifact's
self-fingerprint, response fingerprint and exact source binding, all three
item IDs/order per reviewer session, and the permitted labels. It emits only
aggregate label counts and session counts; it never prints nicknames or notes.
Its tests cover artifact substitution, incomplete response, source-binding
mismatch, duplicate JSON keys, and PII-free aggregation.

The exact artifact browser check is
`.venv/bin/python scripts/test_claim_likelihood_human_review_v2_browser_e2e.py`.
It ran in isolated Windows Chrome with page HTTP(S) blocked. Two synthetic
review sessions each answered all three claims; all six item views loaded 32
decodable model-frame images, exactly one combined JSON download was verified,
and the launched Chrome PID and temporary profile cleanup were checked. The
first test draft saved after each reviewer and Chrome blocked the second
download; the final workflow tests one save after all reviewers complete.

No actual human review was performed in this cycle.
The synthetic E2E validates only the UI/export path, not the judgments. The
saved real review response remains absent, so direct claim-support evidence is
still **0/3 for this artifact** and the broader formal score remains **0/25**.
Any resulting judgments are informal exploratory observations only, not
authenticated or independent annotations. The UI intentionally avoids names
and external transmission; it does not authenticate reviewer identity.
Instructions are Japanese, while source benchmark questions and generated
claims remain in English; reviewers unable to read them should not submit
semantic judgments from this artifact.

After a response is downloaded, validate and summarize it with:

```sh
PYTHONPATH=scripts .venv/bin/python scripts/inspect_claim_likelihood_human_review_v2_response.py \
  --artifact data/private_nextgqa/claim_likelihood_human_review_v2_20261021_01.html \
  --response /path/to/claim_likelihood_human_review_responses.json
```

This command has not been run against a real response because none is present.

## Next action

The Japanese opening/responding guide is `docs/claim_likelihood_review_v2_ja.md`.
If a reviewer completes it later, inspect the downloaded JSON against the
artifact fingerprint and per-item bindings before using any judgment. Until then,
retain the current conclusion: the sham-v2 run measured a QA-interval
claim-likelihood proxy, not semantic explanation faithfulness.
