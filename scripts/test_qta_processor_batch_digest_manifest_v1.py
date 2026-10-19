from __future__ import annotations

import hashlib
import json
import unittest

from nextgqa_atomic_batch_freeze import batch_sha256
from qta_processor_batch_digest_manifest_v1 import (
    ManifestError, build_manifest, validate_manifest,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class QtaProcessorBatchDigestManifestV1Tests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.batches = []
        for i in range(3):
            batch = {"input_ids": [i, i + 1], "pixel_values": [[float(i + 1)]]}
            self.batches.append(batch)
            self.rows.append({
                "record_token": sha(f"opaque-token-{i}"),
                "base_processor_batch_sha256": batch_sha256(batch),
                "source_video_sha256": sha(f"video-{i}"),
                "prompt_sha256": sha(f"prompt-{i}"),
                "input_ids_sha256": sha(f"ids-{i}"),
                "processor_sha256": sha("processor"),
                "tokenizer_sha256": sha("tokenizer"),
                "runtime_sha256": sha("runtime"),
                "sampler_metadata_sha256": sha(f"sampler-{i}"),
                "video_grid_thw": [8 + i, 2, 2],
            })
        self.protocol_sha256 = sha("frozen protocol")
        self.raw = build_manifest(rows=self.rows, live_batches=self.batches,
                                  expected_row_pins=self.rows,
                                  protocol_sha256=self.protocol_sha256,
                                  expected_protocol_sha256=self.protocol_sha256)

    def bindings(self):
        return [{"record_token": row["record_token"],
                 "processor_batch_sha256": row["base_processor_batch_sha256"]}
                for row in self.rows]

    def validate(self, raw=None, bindings=None, expected_rows=None, protocol=None):
        return validate_manifest(
            self.raw if raw is None else raw,
            self.bindings() if bindings is None else bindings,
            expected_row_pins=self.rows if expected_rows is None else expected_rows,
            expected_protocol_sha256=self.protocol_sha256 if protocol is None else protocol)

    def test_builds_canonical_non_authorizing_manifest_and_aggregate_report(self):
        parsed = json.loads(self.raw)
        self.assertEqual(parsed["status"], "candidate_only")
        self.assertEqual(parsed["execution_authorization"], "not_granted")
        self.assertEqual(parsed["protocol_sha256"], self.protocol_sha256)
        self.assertEqual(self.raw, json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False).encode())
        self.assertEqual(self.validate(), {
            "status": "candidate_only", "authorization": "not_granted",
            "row_count": 3, "bindings_match": True,
        })

    def test_order_swap_rejected(self):
        with self.assertRaisesRegex(ManifestError, "order_or_digest"):
            self.validate(bindings=list(reversed(self.bindings())))
        with self.assertRaisesRegex(ManifestError, "pin_order"):
            build_manifest(rows=self.rows, live_batches=self.batches,
                           expected_row_pins=list(reversed(self.rows)),
                           protocol_sha256=self.protocol_sha256,
                           expected_protocol_sha256=self.protocol_sha256)

    def test_live_batch_tamper_and_external_pin_mismatch_rejected(self):
        changed = list(self.batches)
        changed[1] = {"input_ids": [99]}
        with self.assertRaisesRegex(ManifestError, "external_pin_mismatch"):
            build_manifest(rows=self.rows, live_batches=changed,
                           expected_row_pins=self.rows,
                           protocol_sha256=self.protocol_sha256,
                           expected_protocol_sha256=self.protocol_sha256)
        altered = [dict(row) for row in self.rows]
        altered[0]["prompt_sha256"] = sha("unexpected")
        with self.assertRaisesRegex(ManifestError, "external_pin_mismatch"):
            build_manifest(rows=self.rows, live_batches=self.batches,
                           expected_row_pins=altered,
                           protocol_sha256=self.protocol_sha256,
                           expected_protocol_sha256=self.protocol_sha256)

    def test_protocol_must_match_independent_expected_pin(self):
        with self.assertRaisesRegex(ManifestError, "protocol_pin_mismatch"):
            build_manifest(rows=self.rows, live_batches=self.batches,
                           expected_row_pins=self.rows,
                           protocol_sha256=sha("different protocol"),
                           expected_protocol_sha256=self.protocol_sha256)
        with self.assertRaisesRegex(ManifestError, "protocol_pin_mismatch"):
            self.validate(protocol=sha("different protocol"))

    def test_source_and_processor_metadata_must_match_external_row_pins(self):
        obj = json.loads(self.raw)
        obj["rows"][0]["source_video_sha256"] = sha("other source")
        tampered = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode()
        with self.assertRaisesRegex(ManifestError, "external_row_pin_mismatch"):
            self.validate(raw=tampered)

    def test_nonfinite_and_unsupported_live_batches_rejected(self):
        for invalid in ({"x": float("nan")}, {"x": object()}):
            changed = list(self.batches)
            changed[0] = invalid
            with self.subTest(value=type(next(iter(invalid.values()))).__name__):
                with self.assertRaisesRegex(ManifestError, "live_typed_batch_invalid"):
                    build_manifest(rows=self.rows, live_batches=changed,
                                   expected_row_pins=self.rows,
                                   protocol_sha256=self.protocol_sha256,
                                   expected_protocol_sha256=self.protocol_sha256)

    def test_duplicate_record_tokens_and_wrong_row_count_rejected(self):
        duplicate = [dict(row) for row in self.rows]
        duplicate[1]["record_token"] = duplicate[0]["record_token"]
        with self.assertRaisesRegex(ManifestError, "record_tokens_not_unique"):
            build_manifest(rows=duplicate, live_batches=self.batches,
                           expected_row_pins=duplicate,
                           protocol_sha256=self.protocol_sha256,
                           expected_protocol_sha256=self.protocol_sha256)
        with self.assertRaisesRegex(ManifestError, "exactly_three_rows"):
            build_manifest(rows=self.rows[:2], live_batches=self.batches[:2],
                           expected_row_pins=self.rows[:2],
                           protocol_sha256=self.protocol_sha256,
                           expected_protocol_sha256=self.protocol_sha256)

    def test_duplicate_source_video_bytes_rejected(self):
        duplicate = [dict(row) for row in self.rows]
        duplicate[1]["source_video_sha256"] = duplicate[0]["source_video_sha256"]
        for argument_name, value in (("rows", duplicate),
                                     ("expected_row_pins", duplicate)):
            arguments = {"rows": self.rows, "expected_row_pins": self.rows}
            arguments[argument_name] = value
            with self.subTest(argument=argument_name):
                with self.assertRaisesRegex(ManifestError, "source_video_digests"):
                    build_manifest(**arguments, live_batches=self.batches,
                                   protocol_sha256=self.protocol_sha256,
                                   expected_protocol_sha256=self.protocol_sha256)

    def test_duplicate_keys_nonfinite_and_unknown_fields_rejected(self):
        with self.assertRaisesRegex(ManifestError, "duplicate"):
            self.validate(raw=b'{"a":1,"a":2}')
        nonfinite = self.raw.replace(b'"video_grid_thw":[8,2,2]', b'"video_grid_thw":[8,NaN,2]')
        with self.assertRaises(ManifestError):
            self.validate(raw=nonfinite)
        obj = json.loads(self.raw)
        obj["unrecognized"] = True
        with self.assertRaisesRegex(ManifestError, "schema_invalid"):
            self.validate(raw=json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

    def test_noncanonical_serialization_and_binding_digest_tamper_rejected(self):
        with self.assertRaisesRegex(ManifestError, "canonical"):
            self.validate(raw=json.dumps(json.loads(self.raw), indent=2).encode())
        bad = self.bindings()
        bad[0] = dict(bad[0], processor_batch_sha256=sha("tampered"))
        with self.assertRaisesRegex(ManifestError, "order_or_digest"):
            self.validate(bindings=bad)

    def test_manifest_size_is_bounded(self):
        with self.assertRaisesRegex(ManifestError, "size_invalid"):
            self.validate(raw=self.raw + b" " * 1_000_000)


if __name__ == "__main__":
    unittest.main()
