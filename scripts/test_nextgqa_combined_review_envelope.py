import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from nextgqa_combined_review_envelope import (
    EnvelopeError, MAX_MEMBER_BYTES, _read_envelope_file, build_envelope,
    parse_envelope, split_envelope,
)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def make_session(reviewer="reviewer-a", *, clock=False, clock_schema=2):
    review = encode({
        "schema_version": 3,
        "purpose": "anonymous_local_claim_video_human_review",
        "status": "human_review_recorded_unvalidated",
        "execution_authorization": "not_granted",
        "human_reviewer_private_id": reviewer,
    })
    index = encode({
        "schema_version": 1,
        "purpose": "anonymous_unvalidated_human_review_handoff",
        "review_payload_sha256": hashlib.sha256(review).hexdigest(),
        "review_status": "unvalidated_requires_custodian_audit",
        "execution_authorization": "not_granted",
    })
    result = {"review_payload": review, "review_index": index}
    if clock:
        observations = encode({
            "schema_version": clock_schema,
            "purpose": "local_unvalidated_html5_player_clock_observations",
            "status": "raw_player_clock_observations_only",
            "execution_authorization": "not_granted",
        })
        clock_index = encode({
            "schema_version": 1,
            "purpose": "unvalidated_html5_clock_observation_integrity_index",
            "payload_sha256": hashlib.sha256(observations).hexdigest(),
            "status": "raw_observations_only",
            "execution_authorization": "not_granted",
        })
        result.update(clock_observations=observations, clock_index=clock_index)
    return result


class CombinedReviewEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.a = make_session("reviewer-a", clock=True)
        self.b = make_session("reviewer-b")
        self.raw = build_envelope([self.a, self.b])

    def test_round_trip_preserves_member_bytes_reviewer_order_and_clock_pair(self):
        recovered = parse_envelope(self.raw)
        self.assertEqual(recovered, [self.a, self.b])
        document = json.loads(self.raw)
        self.assertEqual(set(document), {
            "schema_version", "purpose", "status", "execution_authorization", "sessions"
        })
        self.assertEqual([item["sequence"] for item in document["sessions"]], [1, 2])
        self.assertEqual([member["role"] for member in document["sessions"][0]["members"]], [
            "review_payload", "review_index", "clock_observations", "clock_index"
        ])
        self.assertEqual(document["execution_authorization"], "not_granted")

    def test_single_reviewer_uses_same_envelope_contract(self):
        recovered = parse_envelope(build_envelope([self.b]))
        self.assertEqual(recovered, [self.b])

    def test_rejects_duplicate_reviewer_id(self):
        with self.assertRaisesRegex(EnvelopeError, "duplicate_reviewer_private_id"):
            build_envelope([self.a, make_session("reviewer-a")])

    def test_enforces_maximum_sixteen_sessions(self):
        sessions = [make_session(f"reviewer-{i:02d}") for i in range(17)]
        with self.assertRaisesRegex(EnvelopeError, "session_count_out_of_range"):
            build_envelope(sessions)

    def test_reviewer_id_scalar_limit_and_del_control_policy(self):
        with self.assertRaisesRegex(EnvelopeError, "reviewer_private_id_invalid"):
            build_envelope([make_session("r" * 257)])
        with self.assertRaisesRegex(EnvelopeError, "reviewer_private_id_invalid"):
            build_envelope([make_session("bad\x7fname")])
        # Python and the browser both count Unicode scalar values, not UTF-16 units.
        accepted = make_session("😀" * 256)
        self.assertEqual(parse_envelope(build_envelope([accepted]))[0], accepted)

    def test_rejects_wrong_session_role_set(self):
        bad = dict(self.b)
        bad.pop("review_index")
        with self.assertRaisesRegex(EnvelopeError, "session_member_role_set_invalid"):
            build_envelope([self.a, bad])

    def test_preserves_renderer_clock_v3_without_normalizing_bytes(self):
        session = make_session("renderer-clock-v3", clock=True, clock_schema=3)
        packed = build_envelope([self.a, session])
        recovered = parse_envelope(packed)
        self.assertEqual(recovered[1]["clock_observations"], session["clock_observations"])
        self.assertEqual(json.loads(recovered[1]["clock_observations"])["schema_version"], 3)

    def test_rejects_duplicate_json_keys(self):
        with self.assertRaisesRegex(EnvelopeError, "duplicate_json_key"):
            parse_envelope(b'{"schema_version":1,"schema_version":1}')

    def test_rejects_nonfinite_json(self):
        raw = self.raw.replace(b'"sequence":1', b'"sequence":NaN', 1)
        with self.assertRaisesRegex(EnvelopeError, "nonfinite_json_value"):
            parse_envelope(raw)

    def test_rejects_extra_root_or_session_or_member_fields(self):
        document = json.loads(self.raw)
        document["extra"] = True
        with self.assertRaisesRegex(EnvelopeError, "envelope_root_schema"):
            parse_envelope(encode(document))
        document = json.loads(self.raw)
        document["sessions"][0]["extra"] = True
        with self.assertRaisesRegex(EnvelopeError, "session_schema"):
            parse_envelope(encode(document))
        document = json.loads(self.raw)
        document["sessions"][0]["members"][0]["extra"] = True
        with self.assertRaisesRegex(EnvelopeError, "member_schema_invalid"):
            parse_envelope(encode(document))

    def test_rejects_out_of_order_or_duplicate_sequences(self):
        document = json.loads(self.raw)
        document["sessions"][0]["sequence"] = 2
        with self.assertRaisesRegex(EnvelopeError, "session_schema_or_sequence_invalid"):
            parse_envelope(encode(document))

    def test_rejects_noncanonical_base64_digest_and_size_tampering(self):
        for mutate, code in (
            (lambda m: m.update(content_base64=m["content_base64"] + "\n"),
             "member_base64_invalid"),
            (lambda m: m.update(sha256="0" * 64), "member_sha256_mismatch"),
            (lambda m: m.update(size_bytes=m["size_bytes"] + 1),
             "member_base64_or_size_mismatch"),
        ):
            with self.subTest(code=code):
                document = json.loads(self.raw)
                mutate(document["sessions"][0]["members"][0])
                with self.assertRaisesRegex(EnvelopeError, code):
                    parse_envelope(encode(document))

    def test_rejects_incomplete_or_wrong_inner_review_hash_and_authorization(self):
        document = json.loads(self.raw)
        document["sessions"][0]["members"].pop()
        with self.assertRaisesRegex(EnvelopeError, "session_member_count_invalid"):
            parse_envelope(encode(document))
        invalid = make_session("bad-index")
        invalid["review_index"] = encode({
            "schema_version": 1,
            "purpose": "anonymous_unvalidated_human_review_handoff",
            "review_payload_sha256": "0" * 64,
            "review_status": "unvalidated_requires_custodian_audit",
            "execution_authorization": "not_granted",
        })
        with self.assertRaisesRegex(EnvelopeError, "inner_review_payload_hash_mismatch"):
            build_envelope([self.a, invalid])
        invalid = make_session("authorized")
        review = json.loads(invalid["review_payload"])
        review["execution_authorization"] = "granted"
        invalid["review_payload"] = encode(review)
        invalid["review_index"] = encode({
            "schema_version": 1,
            "purpose": "anonymous_unvalidated_human_review_handoff",
            "review_payload_sha256": hashlib.sha256(invalid["review_payload"]).hexdigest(),
            "review_status": "unvalidated_requires_custodian_audit",
            "execution_authorization": "not_granted",
        })
        with self.assertRaisesRegex(EnvelopeError, "inner_review_payload_declaration_invalid"):
            build_envelope([self.a, invalid])

    def test_rejects_oversized_inner_member(self):
        too_large = b"x" * (MAX_MEMBER_BYTES + 1)
        with self.assertRaisesRegex(EnvelopeError, "member_bytes_invalid_or_too_large"):
            build_envelope([self.a, {"review_payload": too_large,
                                     "review_index": self.b["review_index"]}])

    def test_split_creates_private_byte_identical_reviewer_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            output = root / "combined_001"
            result = split_envelope(self.raw, output, private_root=root)
            self.assertEqual(result["status"], "split_unvalidated")
            self.assertEqual(result["execution_authorization"], "not_granted")
            self.assertEqual(result["session_count"], 2)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            for index, expected in enumerate((self.a, self.b), start=1):
                session_dir = output / f"session_{index:03d}"
                self.assertEqual(stat.S_IMODE(session_dir.stat().st_mode), 0o700)
                for role, raw in expected.items():
                    filename = {
                        "review_payload": "review_payload.json",
                        "review_index": "review_index.json",
                        "clock_observations": "clock_observations.json",
                        "clock_index": "clock_index.json",
                    }[role]
                    path = session_dir / filename
                    self.assertEqual(path.read_bytes(), raw)
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_split_rejects_outside_root_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "private"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            outside = base / "outside"
            with self.assertRaisesRegex(EnvelopeError, "output_must_be_new_private_root_child"):
                split_envelope(self.raw, outside, private_root=root)
            existing = root / "existing"
            existing.mkdir(mode=0o700)
            marker = existing / "keep"
            marker.write_bytes(b"keep")
            with self.assertRaisesRegex(EnvelopeError, "output_must_be_new_private_root_child"):
                split_envelope(self.raw, existing, private_root=root)
            self.assertEqual(marker.read_bytes(), b"keep")

    def test_split_write_failure_cleans_only_new_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            real_write = os.write
            calls = 0

            def fail_write(fd, raw):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(fd, raw[:3])
                raise OSError("synthetic failure")

            with mock.patch("nextgqa_combined_review_envelope.os.write", side_effect=fail_write):
                with self.assertRaisesRegex(EnvelopeError, "private_output_creation_failed"):
                    split_envelope(self.raw, root / "should_clean", private_root=root)
            self.assertEqual(list(root.iterdir()), [])

    def test_post_publish_root_fsync_failure_reports_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "private"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root_stat = root.stat()
            real_fsync = os.fsync

            def fail_root_fsync(fd):
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino) == (root_stat.st_dev, root_stat.st_ino):
                    raise OSError("synthetic post-publish fsync failure")
                return real_fsync(fd)

            output = root / "published"
            with mock.patch("nextgqa_combined_review_envelope.os.fsync",
                            side_effect=fail_root_fsync):
                result = split_envelope(self.raw, output, private_root=root)
            self.assertEqual(result["status"], "published_durability_unconfirmed")
            self.assertEqual(Path(result["output_dir"]), output)
            self.assertTrue((output / "session_001" / "review_payload.json").is_file())

    def test_input_reader_rejects_symlinks_and_reads_regular_file_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "envelope.json"
            source.write_bytes(self.raw)
            source.chmod(0o600)
            self.assertEqual(_read_envelope_file(source), self.raw)
            link = root / "link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(EnvelopeError, "input_file_invalid"):
                _read_envelope_file(link)

            insecure = root / "insecure.json"
            insecure.write_bytes(self.raw)
            insecure.chmod(0o644)
            with self.assertRaisesRegex(EnvelopeError, "input_file_invalid"):
                _read_envelope_file(insecure)

    def test_rejects_overflow_float_and_integer_tokens_as_sanitized_errors(self):
        for malformed in (
            self.raw.replace(b'"sequence":1', b'"sequence":1e999', 1),
            b'{"schema_version":' + b"9" * 5000 + b"}",
        ):
            with self.subTest(length=len(malformed)):
                with self.assertRaises(EnvelopeError):
                    parse_envelope(malformed)

    def test_rejects_reviewer_id_aliases_after_trim_casefold_and_nfkc(self):
        aliases = [
            ("Reviewer-X", "reviewer-x"),
            ("Cafe\u0301", "Café"),
            ("Ａｌｉｃｅ", "alice"),
            ("Straße", "STRASSE"),
            ("ΟΣ", "οσ"),
            ("Ꭰ", "ꭰ"),
            ("ǰ", "j\u030c"),
        ]
        for first, second in aliases:
            with self.subTest(alias=first):
                with self.assertRaisesRegex(EnvelopeError, "duplicate_reviewer_private_id"):
                    build_envelope([make_session(first), make_session(second)])
        with self.assertRaisesRegex(EnvelopeError, "reviewer_private_id_invalid"):
            build_envelope([make_session(" reviewer-x"), make_session("other")])

    def test_build_checks_aggregate_budget_before_base64_encoding(self):
        import nextgqa_combined_review_envelope as envelope_module

        with mock.patch.object(envelope_module, "MAX_ENVELOPE_BYTES", 128), \
                mock.patch.object(envelope_module.base64, "b64encode",
                                  side_effect=AssertionError("base64 called before size check")):
            with self.assertRaisesRegex(EnvelopeError, "envelope_too_large"):
                build_envelope([self.a, self.b])


if __name__ == "__main__":
    unittest.main()
