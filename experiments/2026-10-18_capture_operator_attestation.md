# Capture-operator artifact attestation (synthetic protocol check)

## Purpose

Add a machine-side attestation for the human-review runbook's capture operator
role. The protocol binds the exact candidate 15-anchor plan and exact bytes of
raw player observations, the screenshot/OS capture receipt, screenshots, crop
map, crop crosswalk, and crops. This closes a byte-provenance gap before any
real capture receipt is considered.

## Scope and implementation

New code is isolated in `scripts/nextgqa_capture_operator_attestation.py`, with
a private filesystem CLI in `scripts/run_nextgqa_capture_operator_attestation.py`
and synthetic tests in `scripts/test_nextgqa_capture_operator_attestation.py`. It
reuses the repository's root-signed release-roster validator, existing clock
plan/receipt validators, and Ed25519 receipt helpers. The attestation requires
the capture-operator key from the verified roster and emits ordered, pseudonymous
anchor slots plus exact hashes. It rejects missing/duplicate anchors, extra or
missing screenshot/crop files, changed bytes, a bad plan digest, bool anchor
counts, and role-key reuse.

The CLI requires an absolute mode-0700 owner-only root, exact relative input
paths beneath it, owner-owned regular mode-0600 files, exact screenshot/crop
directory membership, no-follow reads, per-file and aggregate size caps, and an
exclusive mode-0600 output followed by `fsync`. Directory reads traverse each
component relative to the open root descriptor with `O_NOFOLLOW`, check exact
membership, and recheck directory/file identities. The CLI calls the API's
builder/verifier and emits only a content-free status line. Tests create files under a
temporary directory with directory mode 0700 and file mode 0600; bytes are
synthetic strings, not images or real receipts. No protected review HTML, video,
identity, real capture receipt, or authorization action was accessed or created.

## Verified result

Twelve synthetic tests pass under the repository `.venv`:

```text
Ran 12 tests ... OK
```

The round trip confirms the rostered key's signature over the candidate artifact
hashes and that the verifier recomputes those hashes. Tampering tests cover the
plan digest, event payload/index, screenshot, crop, OS receipt, duplicate/missing
artifact slots, boolean denominator, capture role-key reuse, CLI containment and
intermediate/leaf symlink escape and extra-directory-artifact rejection, output
mode, overwrite refusal, and sanitized malformed receipt/crosswalk failures.
`py_compile` and scoped `git diff --check` pass for the API, CLI, tests, and log.

An independent audit found that the first CLI draft used a path-based directory
reader that no-followed only the leaf; an intermediate symlink could redirect
the read outside the private root. That draft was not used on real artifacts.
The CLI now uses root-fd-anchored componentwise no-follow traversal, exact set
checks, stable identity rechecks, and regression tests for both intermediate and
leaf symlinks. Malformed capture-receipt/crosswalk indexes now fail through a
sanitized content-free CLI status instead of leaking a traceback.

Source hashes at this check:

```text
6ba64d8c47ec1f26a65848f557efbcf9a3b9fcda091bd0d716876c09dae5354f  scripts/nextgqa_capture_operator_attestation.py
d9ab080d1ad7a2be931a0225ae43649078740dee64ae6edf568133b4fc389387  scripts/run_nextgqa_capture_operator_attestation.py
ac8433403a9aa57d025387415afa0d082e00bf186593b16934fa75f6040272aa  scripts/test_nextgqa_capture_operator_attestation.py
```

## Interpretation and limitations

The signature proves only that the private key corresponding to the
root-rostered `capture_operator` public key adopted these exact candidate bytes.
It does not prove who possessed the key, that a capture process ran, that the OS
receipt is truthful, that a screenshot corresponds to a decoder frame, or that
a crop contains video-only pixels. The runbook describes the OS receipt as
self-reported; no frame correspondence or crop-purity review is performed here.
The output fixes all authorization/readiness fields to false or `not_granted`.

The CLI does not open review HTML or video. Its actual invocation is deliberately
not run on real artifacts. The independent audit cleared only the synthetic API
and wrapper behavior. Before any real workflow, the role-to-key trust inputs and
a fresh private inventory must be established; the candidate attestation is not
integrated into release or score gates. If those prerequisites are later met, an
operator would provide the private root and explicit relative paths, for example:

```sh
.venv/bin/python scripts/run_nextgqa_capture_operator_attestation.py \
  --private-root /absolute/private/capture-root \
  --plan plan.json --plan-sha256 <frozen-plan-sha256> \
  --plan-provenance plan-provenance.json \
  --plan-provenance-sha256 <plan-provenance-sha256> \
  --observations observations.json --observation-index observation-index.json \
  --capture-receipt capture-receipt.json --crop-map crop-map.json \
  --crop-crosswalk crop-crosswalk.json --screenshots-dir screenshots \
  --crops-dir crops --integrity-roster integrity-roster.json \
  --release-roster release-roster.json --root-public-key root-public.pem \
  --root-public-der-sha256 <pinned-root-key-sha256> \
  --operator-private-key capture-operator-private.pem \
  --operator-public-key capture-operator-public.pem \
  --output capture-operator-attestation.json
```

The placeholders must be replaced with frozen digest values; output must be a
new path because creation is exclusive. This command has not been run against
real artifacts and must remain gated on independent audit and all human-review
gates.

## Next action

The implementation has passed synthetic independent audit only. The next gate
is human establishment of the trust inputs and a fresh inventory; this synthetic
result is not evidence that the capture protocol or screenshots are valid.

Runbook hash after documenting this candidate-byte capability:

```text
b71e78a72c2f61f8922e41d1a3da6a0f5f2709f39b51552cbd747514ca4842f4  docs/nextgqa_v6_human_review_runbook.md
```
