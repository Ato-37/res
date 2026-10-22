from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_temporal_order_global_bias_cohort as analyzer
import run_nextgqa_gold_interval_margin_pilot as pilot
import run_saved_claim_frame_permutation_attention_smoke as scorer


class CohortAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        private = self.root / "data/private_nextgqa"
        private.mkdir(parents=True)
        private.chmod(0o700)
        source_dir = self.root / "scripts"
        source_dir.mkdir()
        source_names = (analyzer.LOCK_SOURCES | analyzer.GENERATOR_SOURCES
                        | analyzer.RECEIPT_SOURCES)
        source_hashes = {}
        for name in source_names:
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"frozen {name}\n")
            source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.packet_rel = Path("data/private_nextgqa/packet.json")
        self.lock_rel = Path("data/private_nextgqa/lock.json")
        self.filter_rel = Path("data/private_nextgqa/filter.json")
        pairs = [f"{i + 1:064x}" for i in range(3)]
        pair_payload = {"schema_version": 1, "pair_sha256": pairs}
        pair_raw = (json.dumps(pair_payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        pair_path = self.root / self.filter_rel
        pair_path.write_bytes(pair_raw); pair_path.chmod(0o600)
        records, assignments = [], []
        for i, pair in enumerate(pairs):
            media = hashlib.sha256(f"video-{i}".encode()).hexdigest()
            qid = f"qid-{i}"
            qid_hash = hashlib.sha256(qid.encode()).hexdigest()
            claim = f"A person performs action {i}."
            records.append({"video_sha256": media, "dataset_question_id": qid,
                "candidate_pair_sha256": pair, "generation_prompt_version": "v1",
                "generated_parsed": {"valid": True, "claim": claim}})
            assignments.append({"media_sha256": media, "qid_sha256": qid_hash})
        lock = {"effective_videos": 3, "dataset_split": "test",
            "assignments": assignments, "cohort_sha256": "frozen-cohort",
            "selected_pair_filter_sha256": hashlib.sha256(pair_raw).hexdigest(),
            "source_sha256": {name: source_hashes[name] for name in analyzer.LOCK_SOURCES}}
        packet = {"status": "passed", "dataset_split": "test",
            "model_revision": scorer.MODEL_REVISION,
            "generation_prompt_version": "v1", "effective_videos": 3,
            "fully_structured_claim_records": 3, "records": records,
            "candidate_pair_filter_sha256": hashlib.sha256(pair_raw).hexdigest(),
            "candidate_pair_filter_digest": pilot._hash(sorted(pairs)),
            "cohort_sha256": "frozen-cohort",
            "runner_sha256": source_hashes["scripts/generate_nextgqa_claim_review_pilot.py"],
            "pilot_runner_sha256": source_hashes["scripts/run_nextgqa_gold_interval_margin_pilot.py"]}
        for rel, payload in ((self.packet_rel, packet), (self.lock_rel, lock)):
            path = self.root / rel
            path.write_text(json.dumps(payload)); path.chmod(0o600)
        self.packet_hash = hashlib.sha256((self.root / self.packet_rel).read_bytes()).hexdigest()
        self.receipt_stem = "data/private_nextgqa/receipt_row"
        branch_values = {
            "original_baseline": 0.2,
            "permuted_baseline": 0.1,
            "original_visual_bias": 0.3,
            "permuted_visual_bias": 0.1,
        }
        for i, record in enumerate(records):
            branches = {key: {"target_logprob_mean": value}
                        for key, value in branch_values.items()}
            inter = scorer.compute_bias_frame_order_interaction(
                original_baseline=0.2, permuted_baseline=0.1,
                original_biased=0.3, permuted_biased=0.1)
            qid = record["dataset_question_id"]
            receipt = {"status": "completed_one_row_exploratory_sensitivity_smoke",
                "packet_record_index_zero_based": i,
                "private_input_packet_sha256": self.packet_hash,
                "source_video_sha256": record["video_sha256"],
                "saved_claim_text_sha256": hashlib.sha256(
                    record["generated_parsed"]["claim"].encode()).hexdigest(),
                "row_token_sha256": hashlib.sha256(
                    (record["video_sha256"] + qid).encode()).hexdigest(),
                "requested_forwards": 4, "completed_forwards": 4,
                "direct_faithfulness": False, "faithfulness_result": False,
                "human_review": {"review_actually_performed": False},
                "model_revision": scorer.MODEL_REVISION,
                "precision": "4-bit weights; BF16 compute; eager attention",
                "branch_target_scores": branches,
                "primary_descriptive_interaction": inter,
                "source_sha256": {name: source_hashes[name]
                                  for name in analyzer.RECEIPT_SOURCES}}
            path = self.root / f"{self.receipt_stem}{i}.json"
            path.write_text(json.dumps(receipt)); path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_cohort_recomputes_all_rows_and_counts(self):
        result = analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                                  self.filter_rel, self.receipt_stem)
        self.assertEqual(result["completed_forwards"], 12)
        for value in result["row_interactions_nats_per_token"]:
            self.assertAlmostEqual(value, 0.1)
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertTrue(result["row_assignment_and_pair_bindings_verified"])

    def test_rejects_wrong_row_receipt_binding(self):
        path = self.root / f"{self.receipt_stem}1.json"
        receipt = json.loads(path.read_text())
        receipt["packet_record_index_zero_based"] = 0
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "receipt_cohort_binding"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)

    def test_rejects_changed_interaction_arithmetic(self):
        path = self.root / f"{self.receipt_stem}2.json"
        receipt = json.loads(path.read_text())
        receipt["primary_descriptive_interaction"][
            "bias_by_frame_order_interaction_nats_per_token"] = 0.2
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "receipt_interaction_arithmetic"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)

    def test_rejects_private_path_traversal_and_parent_symlink(self):
        outside = self.root / "outside.json"
        outside.write_text("{}")
        with self.assertRaisesRegex(ValueError, "private_input_path_invalid"):
            analyzer._private_bytes(self.root,
                Path("data/private_nextgqa/../outside.json"))
        nested = self.root / "data/private_nextgqa/linked"
        nested.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "private_input_path_invalid"):
            analyzer._private_bytes(self.root,
                Path("data/private_nextgqa/linked/outside.json"))

    def test_rejects_private_root_with_permissive_mode(self):
        private = self.root / "data/private_nextgqa"
        private.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "private_root_permissions_invalid"):
            analyzer._private_bytes(self.root, self.packet_rel)

    def test_rejects_malformed_nested_assignment_and_review_schema(self):
        lock_path = self.root / self.lock_rel
        lock = json.loads(lock_path.read_text())
        lock["assignments"][0] = None
        lock_path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, "frozen_cohort_row_schema_invalid"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)
        self.tearDown()
        self.setUp()
        receipt_path = self.root / f"{self.receipt_stem}0.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["human_review"] = []
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "receipt_nested_schema_invalid"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)

    def test_rejects_incomplete_or_unrelated_source_map(self):
        path = self.root / f"{self.receipt_stem}0.json"
        receipt = json.loads(path.read_text())
        receipt["source_sha256"] = {"scripts/unrelated.py": "0" * 64}
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "receipt_source_hash_mismatch"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)

    def test_rejects_lock_source_mutation(self):
        path = self.root / "scripts/freeze_temporal_order_claim_cohort.py"
        path.write_text("tampered source\n")
        with self.assertRaisesRegex(ValueError, "lock_source_hash_mismatch"):
            analyzer.analyze(self.root, self.packet_rel, self.lock_rel,
                              self.filter_rel, self.receipt_stem)

    def test_rejects_duplicate_keys_and_nonfinite_json_numbers(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                analyzer._json(raw)

    def test_source_hash_rejects_symlink_parent(self):
        real = self.root / "real_scripts"
        real.mkdir()
        (real / "source.py").write_text("source\n")
        scripts = self.root / "linked_scripts"
        scripts.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "source_open_failed"):
            analyzer._source_sha256(self.root, "scripts/source.py")


if __name__ == "__main__":
    unittest.main()
