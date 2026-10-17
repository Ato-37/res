from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
import os
import stat
import tempfile
from unittest.mock import patch

import audit_nextgqa_claim_cohort_fingerprint as audit
from audit_nextgqa_claim_cohort_fingerprint import (
    AuditError, _canonical_sha, _evidence_indices, _matched_controls,
    _normalize_answer, _parse, _video_key, recompute,
)


class CohortFingerprintAuditUnitTests(unittest.TestCase):
    def test_canonical_sha_matches_declared_json_canonicalization(self):
        value = [{"x": "é", "a": [1, 2.5]}]
        expected = hashlib.sha256(b'[{"a":[1,2.5],"x":"\xc3\xa9"}]').hexdigest()
        self.assertEqual(_canonical_sha(value), expected)

    def test_answer_normalization_matches_selection_rule(self):
        self.assertEqual(_normalize_answer("  Ａ  B\nC "), "a b c")

    def test_video_key_uses_last_two_components_without_extension(self):
        self.assertEqual(_video_key("split/a/b/video.mp4"), "b/video")

    def test_evidence_intervals_use_exact_inclusive_pts(self):
        times = [Fraction(0), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)]
        self.assertEqual(_evidence_indices(times, [[1, 2]]), [1, 2, 3])

    def test_controls_minimize_rank_distance_with_lexical_ties(self):
        self.assertEqual(_matched_controls([1, 2], 6), [0, 3])

    def test_duplicate_json_keys_fail_closed(self):
        with self.assertRaises(AuditError):
            _parse(b'{"x":1,"x":2}')

    def test_nonfinite_json_fails_closed(self):
        with self.assertRaises(AuditError):
            _parse(b'{"x":NaN}')

    def test_read_private_requires_private_owned_bytes_without_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            private.chmod(0o700)
            target = private / "packet.json"
            target.write_bytes(b'{"ok":true}')
            target.chmod(0o600)
            outside = root / "outside.json"
            outside.write_bytes(b'{"secret":true}')
            outside.chmod(0o600)
            with patch.object(audit, "REPO_ROOT", root):
                self.assertEqual(audit._read_private(target), b'{"ok":true}')
                link = private / "linked.json"
                link.symlink_to(outside)
                with self.assertRaises(AuditError):
                    audit._read_private(link)
                broad = private / "broad.json"
                broad.write_bytes(b"{}")
                broad.chmod(0o644)
                with self.assertRaises(AuditError) as raised:
                    audit._read_private(broad)
                self.assertEqual(str(raised.exception), "private_input_mode_or_type_invalid")

    def test_recompute_binds_three_synthetic_video_sources_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, lineage_rows = [], []
            for index in range(3):
                relative = f"test/video-{index}.mp4"
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                content = f"synthetic-video-{index}".encode()
                path.write_bytes(content)
                video_sha = hashlib.sha256(content).hexdigest()
                qid = f"q-{index}"
                question = f"synthetic question {index}"
                choices = ["wrong A", f"gold {index}", "wrong C", "wrong D", "wrong E"]
                rows.append({
                    "dataset_question_id": qid,
                    "local_video_path": str(path),
                    "video_sha256": video_sha,
                    "question": question,
                    "choices": choices,
                    "gold_answer": choices[1],
                    "gold_interval_seconds": [[1, 2]],
                })
                lineage_rows.append({
                    "video_sha256": video_sha,
                    "video_id": "video-" + video_sha,
                    "dataset_question_id": qid,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "relative_video_path": relative,
                })

            generation = {
                "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
                "status": "passed", "requested_videos": 3, "effective_videos": 3,
                "frames_per_video": 4, "dataset_inputs_sha256": "d" * 64,
                "records": rows,
            }
            pts = [Fraction(0), Fraction(1), Fraction(2), Fraction(3)]
            fingerprint_rows = []
            for row in rows:
                video_sha = row["video_sha256"]
                local = Path(row["local_video_path"])
                evidence = [1, 2]
                fingerprint_rows.append({
                    "qid": row["dataset_question_id"],
                    "video_key": _video_key(local.relative_to(root).as_posix()),
                    "video_sha256": video_sha,
                    "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                    "choices_sha256": _canonical_sha(row["choices"]),
                    "gold": 1,
                    "spans": row["gold_interval_seconds"],
                    "sampled_pts": [[x.numerator, x.denominator] for x in pts],
                    "evidence_positions": evidence,
                    "control_positions": _matched_controls(evidence, 4),
                })
            generation["cohort_sha256"] = _canonical_sha(fingerprint_rows)
            generation_raw = (json.dumps(generation, sort_keys=True,
                                         separators=(",", ":"), ensure_ascii=False)
                             .encode("utf-8"))
            lineage = {
                "status": "metadata_lineage_crosschecked",
                "source_packet_sha256": hashlib.sha256(generation_raw).hexdigest(),
                "cohort_sha256": generation["cohort_sha256"],
                "dataset_inputs_sha256": generation["dataset_inputs_sha256"],
                "records": lineage_rows,
            }
            lineage_raw = json.dumps(lineage, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=False).encode("utf-8")

            def fake_decode(path, *, num_frames, uniform_tolerance_seconds):
                content = Path(path).read_bytes()
                video_sha = hashlib.sha256(content).hexdigest()
                decoded = {
                    "source_sha256": video_sha,
                    "sampled_frames": [
                        {"relative_presentation_time": [i, 1]} for i in range(num_frames)
                    ],
                }
                return [object()] * num_frames, decoded

            with (patch.object(audit, "decode_sampled_video", side_effect=fake_decode),
                  patch.object(audit, "verify_sampled_frame_pixels")):
                result = recompute(generation_raw=generation_raw,
                                   lineage_raw=lineage_raw, video_root=root)

            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["digest_match"])
            self.assertEqual(result["decoded_records"], 3)
            self.assertFalse(result["full_candidate_scan_replayed"])
            self.assertFalse(result["human_review_or_execution_authorized"])

            with (patch.object(audit, "_sha_file", return_value="0" * 64),
                  patch.object(audit, "decode_sampled_video", side_effect=fake_decode),
                  patch.object(audit, "verify_sampled_frame_pixels")):
                with self.assertRaises(AuditError) as raised:
                    recompute(generation_raw=generation_raw,
                              lineage_raw=lineage_raw, video_root=root)
            self.assertEqual(str(raised.exception), "source_video_hash_mismatch")

    def test_recompute_rejects_invalid_generation_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AuditError) as raised:
                recompute(generation_raw=b"{}", lineage_raw=b"{}", video_root=Path(tmp))
            self.assertEqual(str(raised.exception), "generation_packet_scope_invalid")


if __name__ == "__main__":
    unittest.main()
