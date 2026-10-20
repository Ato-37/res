from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_temporal_order_claim_dependence import (
    _expected_packet_status, _finite_cohort_descriptives, analyze,
)


def _write(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _receipt(media: str, packet_path: Path, partial: bool) -> dict:
    branches = {}
    for name, value in (("original_baseline", -1.0), ("permuted_baseline", -2.0),
                        ("original_visual_bias", -0.5), ("permuted_visual_bias", -1.6)):
        branches[name] = {
            "target_logprob_mean": value,
            "intervention_hook_invocations_by_layer_1_based": {
                str(layer): (0 if name.endswith("baseline") else 1)
                for layer in range(5, 17)},
        }
    raw = packet_path.read_bytes()
    return {
        "status": "completed_one_row_exploratory_sensitivity_smoke",
        "source_video_sha256": media,
        "requested_forwards": 4, "attempted_forwards": 4, "completed_forwards": 4,
        "direct_faithfulness": False, "faithfulness_result": False,
        "claim_truth_reviewed": False,
        "human_review": {"review_actually_performed": False},
        "private_input_packet_name": packet_path.name,
        "private_input_packet_sha256": hashlib.sha256(raw).hexdigest(),
        "partial_input_packet_allowed": partial,
        "source_sha256": {"scripts/run_saved_claim_frame_permutation_attention_smoke.py":
            hashlib.sha256(Path(__file__).with_name(
                "run_saved_claim_frame_permutation_attention_smoke.py").read_bytes()).hexdigest()},
        "branch_target_scores": branches,
        "primary_descriptive_interaction": {
            "bias_by_frame_order_interaction_nats_per_token": 0.1},
    }


class TemporalOrderAnalyzerTests(unittest.TestCase):
    def test_packet_status_reflects_valid_count_for_smoke_and_remaining_batches(self):
        self.assertEqual(_expected_packet_status(1, 1), "passed")
        self.assertEqual(_expected_packet_status(2, 2), "passed")
        self.assertEqual(_expected_packet_status(1, 2), "partial")
        with self.assertRaises(ValueError):
            _expected_packet_status(3, 2)

    def test_complete_cohort_reports_only_finite_sample_descriptives(self):
        result = _finite_cohort_descriptives([-1.0, 0.0, 2.0], complete=True)
        self.assertAlmostEqual(result["mean"], 1.0 / 3.0)
        self.assertEqual(result["median"], 0.0)
        self.assertEqual(result["range"], [-1.0, 2.0])
        self.assertEqual(_finite_cohort_descriptives([1.0], complete=False),
                         {"mean": None, "median": None, "range": None})

    def test_incomplete_locked_cohort_keeps_row_scores_unpooled(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            qids = ["q0", "q1", "q2"]
            media = [hashlib.sha256(f"m{i}".encode()).hexdigest() for i in range(3)]
            lock = {"status": "frozen_before_claim_generation_and_scoring",
                    "dataset_split": "test",
                    "effective_videos": 3,
                    "assignments": [{"media_sha256": media[i],
                        "qid_sha256": hashlib.sha256(qids[i].encode()).hexdigest()}
                        for i in range(3)]}
            pair_hashes = [hashlib.sha256(f"p{i}".encode()).hexdigest() for i in range(3)]
            smoke_filter = {"schema_version": 1, "pair_sha256": pair_hashes[:1]}
            remaining_filter = {"schema_version": 1, "pair_sha256": pair_hashes[1:]}
            smoke_filter_path = root / "smoke-filter.json"
            remaining_filter_path = root / "remaining-filter.json"
            _write(smoke_filter_path, smoke_filter); _write(remaining_filter_path, remaining_filter)
            smoke_filter_hash = hashlib.sha256(smoke_filter_path.read_bytes()).hexdigest()
            remaining_filter_hash = hashlib.sha256(remaining_filter_path.read_bytes()).hexdigest()
            repo = Path(__file__).resolve().parents[1]
            generator_hash = hashlib.sha256(
                (repo / "scripts/generate_nextgqa_claim_review_pilot.py").read_bytes()).hexdigest()
            pilot_hash = hashlib.sha256(
                (repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py").read_bytes()).hexdigest()
            smoke = {"dataset_split": "test", "status": "passed", "effective_videos": 1,
                "requested_videos": 1,"fully_structured_claim_records": 1,
                "runner_sha256": generator_hash, "pilot_runner_sha256": pilot_hash,
                "candidate_pair_filter_sha256": smoke_filter_hash,
                "candidate_pair_filter_digest": hashlib.sha256(json.dumps(sorted(pair_hashes[:1]),separators=(",", ":")).encode()).hexdigest(),
                "records": [{"video_sha256": media[0], "dataset_question_id": qids[0],
                    "candidate_pair_sha256":pair_hashes[0],
                    "generated_parsed": {"valid": True, "claim": "claim one"}}]}
            remaining = {"dataset_split": "test", "status": "partial", "requested_videos": 2,
                "effective_videos": 2,"fully_structured_claim_records": 1,
                "runner_sha256": generator_hash, "pilot_runner_sha256": pilot_hash,
                "candidate_pair_filter_sha256": remaining_filter_hash,
                "candidate_pair_filter_digest": hashlib.sha256(json.dumps(sorted(pair_hashes[1:]),separators=(",", ":")).encode()).hexdigest(),
                "records": [
                    {"video_sha256": media[1], "dataset_question_id": qids[1],
                     "candidate_pair_sha256":pair_hashes[1],
                     "generated_parsed": {"valid": False, "claim": None,
                                           "claim_text_quality_failure": "claim_missing"}},
                    {"video_sha256": media[2], "dataset_question_id": qids[2],
                     "candidate_pair_sha256":pair_hashes[2],
                     "generated_parsed": {"valid": True, "claim": "claim three"}}]}
            lock_path = root / "lock.json"; smoke_path = root / "smoke.json"
            remaining_path = root / "remaining.json"
            _write(lock_path, lock); _write(smoke_path, smoke); _write(remaining_path, remaining)
            receipt_paths = []
            for name, row_media, packet, partial in (
                    ("r0.json", media[0], smoke_path, False),
                    ("r2.json", media[2], remaining_path, True)):
                path = root / name; _write(path, _receipt(row_media, packet, partial))
                receipt_paths.append(path)
            result = analyze(lock_path=lock_path, smoke_packet_path=smoke_path,
                remaining_packet_path=remaining_path,
                smoke_pair_filter_path=smoke_filter_path,
                remaining_pair_filter_path=remaining_filter_path,
                receipt_paths=tuple(receipt_paths))
            self.assertEqual(result["status"], "incomplete_cohort_no_effect_estimate")
            self.assertEqual(result["scored_rows"], 2)
            self.assertEqual(result["provenance_unverified_rows"], [])
            self.assertEqual(result["claim_generation_failed_rows"], 1)
            self.assertIsNone(result["cohort_mean"])
            self.assertEqual([row["locked_row"] for row in result[
                "verified_scored_rows_individual_descriptives_only"]], [0, 2])

            # A receipt whose recorded runner cannot be reproduced remains
            # visible but cannot contribute to the verified scored count.
            r0 = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
            r0["source_sha256"]["scripts/run_saved_claim_frame_permutation_attention_smoke.py"] = "0" * 64
            receipt_paths[0].write_text(json.dumps(r0), encoding="utf-8")
            audited = analyze(lock_path=lock_path, smoke_packet_path=smoke_path,
                remaining_packet_path=remaining_path,
                smoke_pair_filter_path=smoke_filter_path,
                remaining_pair_filter_path=remaining_filter_path,
                receipt_paths=tuple(receipt_paths))
            self.assertEqual(audited["scored_rows"], 1)
            self.assertEqual(audited["provenance_unverified_rows"], [0])
            self.assertEqual([row["locked_row"] for row in audited[
                "verified_scored_rows_individual_descriptives_only"]], [2])
            self.assertEqual([row["locked_row"] for row in audited[
                "provenance_unverified_receipt_diagnostics"]], [0])

            # A packet with a mismatched generator/pilot source must be rejected
            # before any linked receipts can contribute to an estimate.
            smoke_payload = json.loads(smoke_path.read_text(encoding="utf-8"))
            smoke_payload["runner_sha256"] = "0" * 64
            smoke_path.write_text(json.dumps(smoke_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim_generation_source_hash_mismatch"):
                analyze(lock_path=lock_path, smoke_packet_path=smoke_path,
                    remaining_packet_path=remaining_path,
                    smoke_pair_filter_path=smoke_filter_path,
                    remaining_pair_filter_path=remaining_filter_path,
                    receipt_paths=tuple(receipt_paths))
            smoke_payload["runner_sha256"] = generator_hash
            smoke_path.write_text(json.dumps(smoke_payload), encoding="utf-8")

            # The same analyzer contract also accepts a separately frozen
            # validation-split cohort, without mixing its packets with test.
            for path in (lock_path, smoke_path, remaining_path):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["dataset_split"] = "val"
                path.write_text(json.dumps(payload), encoding="utf-8")
            for receipt_path, packet_path in ((receipt_paths[0], smoke_path),
                                               (receipt_paths[1], remaining_path)):
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["private_input_packet_sha256"] = hashlib.sha256(
                    packet_path.read_bytes()).hexdigest()
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            validation = analyze(lock_path=lock_path, smoke_packet_path=smoke_path,
                remaining_packet_path=remaining_path,
                smoke_pair_filter_path=smoke_filter_path,
                remaining_pair_filter_path=remaining_filter_path,
                receipt_paths=tuple(receipt_paths))
            self.assertEqual(validation["status"], "incomplete_cohort_no_effect_estimate")
            self.assertEqual(validation["provenance_unverified_rows"], [0])
            self.assertIsNone(audited["cohort_mean"])


if __name__ == "__main__":
    unittest.main()
