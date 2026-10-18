# Human-review release gate regression follow-up

## Purpose and setup

Recheck the non-authorizing human-review release, receipt, role separation,
capture-attestation, claim-rendering, frame-group mapping, and baseline-p/q
candidate code after the latest local edits. This is an engineering test run,
not a release ceremony or completed review. It used synthetic fixture bytes and
keys only; no real private review package, review export, video, model, or GPU
was opened or loaded.

Date: 2026-10-18 (Asia/Tokyo). Repository HEAD: `d6d92a3a9da422f6f1b7641b6748980501552496`.

Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_audit_nextgqa_review_release_order \
  test_nextgqa_review_ceremony_signing \
  test_nextgqa_capture_operator_attestation \
  test_nextgqa_prediction_link_attestation \
  test_render_nextgqa_claim_review_html \
  test_qwen_review_interval_token_groups \
  test_qta_baseline_pq_candidate -v
```

## Verified results

All **83/83** selected tests passed in 11.082 seconds. Test coverage included
distinct owner/reviewer/capture/witness keys, exact artifact and receipt-byte
binding, signature tampering, unsafe path/mode rejection, non-authorizing page
rendering, interval-to-group boundary handling, and fixed baseline answer/foil
integrity. A synthetic CLI negative case returned `blocked` with
`trust_root_key_file_invalid`, `package_opening_permitted=false`, and
`model_load_permitted=false`, as expected. These outputs validate fail-closed
software behavior only; no real signature or reviewer identity was provided.

The exact source and test hashes at execution are recorded below.

| File | SHA-256 |
|---|---|
| `scripts/audit_nextgqa_review_release_order.py` | `da9dcc6cc325ceb9a143b6bb828b9cd0cafebb53c95f071ec4d19d69587b680d` |
| `scripts/nextgqa_review_ceremony_signing.py` | `6fdc73cc4b95a0475d31d918cf4fcc2784407cc398853b2b844ff8e94ab89bc9` |
| `scripts/nextgqa_capture_operator_attestation.py` | `6ba64d8c47ec1f26a65848f557efbcf9a3b9fcda091bd0d716876c09dae5354f` |
| `scripts/nextgqa_prediction_link_attestation.py` | `4d8b625a588210c5a6434d2a6eb1dabbd9d162174c45e276fa5bcf1a31ba847b` |
| `scripts/render_nextgqa_claim_review_html.py` | `71da49d7eb4e611e61dbbc2579425763a2adf5e4746d16417e6648db03dfef92` |
| `scripts/qwen_review_interval_token_groups.py` | `4ce78c5a368c7ca0e011af1b4c0c7da2d3d7fbed5860b28a5174045c4f1660c4` |
| `scripts/qta_baseline_pq_candidate.py` | `e5afeb00bb3bf501b2ef0e90c2bddf4e7a4c364a08713c7a0897b897f4d2f3dd` |
| `scripts/test_audit_nextgqa_review_release_order.py` | `1a747b44631af33f11625dc8f016754f4bb1bb023eda84cbfcbd20fbd127efbd` |
| `scripts/test_nextgqa_review_ceremony_signing.py` | `3fd0953f6589cc330575ff1fdf1530a667d7bf7c36fad3c1363ba0f07e9de778` |
| `scripts/test_nextgqa_capture_operator_attestation.py` | `ac8433403a9aa57d025387415afa0d082e00bf186593b16934fa75f6040272aa` |
| `scripts/test_nextgqa_prediction_link_attestation.py` | `cf7727fe34ed85a7d9f0b8618b810d8dbcc847b17292729fe1d79673619b4979` |
| `scripts/test_render_nextgqa_claim_review_html.py` | `a46a8d6d71cfcea40c09e641631a169f502c9e75fb545c9c579027b7ee652bf6` |
| `scripts/test_qwen_review_interval_token_groups.py` | `c21f724f4c489299c8f5a18b473ade91e574886cd5808f37eee34b7265884c3c` |
| `scripts/test_qta_baseline_pq_candidate.py` | `0dbe622136b8d4ce75e8cd07dca363b718dc21003481685db1b0d3740918a871` |

## Faithfulness progress and limits

Direct generated-claim faithfulness remains **0/3** reviewed claims,
**0/15** player-clock/decoder-PTS correspondences, **0/3** prediction-link
judgments, and **0/3** claim-linked scores. There is no real `ΔF`, interval, or
test. Human-review readiness remains `preparation only`: there is no
independently pinned trust root/fingerprint, actual role-to-key roster, signed
review-only receipt, or witnessed unopened handoff. Rights remain unverified;
all real protected materials stay local.

Passing synthetic tests does not grant permission to open the review package,
run a model, or score data. Continue other justified human-free work while the
external identity/root and witnessed ceremony inputs remain unavailable.
