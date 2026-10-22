import json
import os
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
import hashlib
from unittest.mock import patch

import load_frozen_claim_packet_samples as replay
from generate_nextgqa_claim_review_pilot import validate_replay_dataset_split
from generate_nextgqa_claim_review_pilot import parse_claim_output
from run_nextgqa_gold_interval_margin_pilot import _hash, candidate_pair_sha256


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/private_nextgqa"


class PrivateSnapshotTests(unittest.TestCase):
    def test_private_reader_rejects_symlink_and_wrong_mode(self):
        fd, name = tempfile.mkstemp(prefix=".snapshot-test-", suffix=".json", dir=PRIVATE)
        source = Path(name)
        linked = PRIVATE / f"{source.name}.link"
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write('{"schema_version":1}')
            source.chmod(0o600)
            linked.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "frozen_input_path_invalid"):
                replay._private_payload(linked, PRIVATE)
            source.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "frozen_input_permissions_invalid"):
                replay._private_payload(source, PRIVATE)
        finally:
            source.unlink(missing_ok=True)
            linked.unlink(missing_ok=True)


class SyntheticReplayFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "data/private_nextgqa"
        self.private.mkdir(parents=True)
        self.dataset = self.root / "NExT-GQA/datasets/nextgqa"
        self.dataset.mkdir(parents=True)
        self.video_root = self.root / "videos"
        self.video_root.mkdir()
        self.video_paths = []
        self.selected = []
        self.assignments = []
        self.records = []
        self.pairs = []
        media_hashes = []
        for i in range(3):
            media = hashlib.sha256(f"media-{i}".encode()).hexdigest()
            qid = f"qid-{i}"
            qid_hash = hashlib.sha256(qid.encode()).hexdigest()
            pair = candidate_pair_sha256(f"video-{i}", qid)
            self.pairs.append(pair)
            media_hashes.append(media)
            folder = self.video_root / f"{i:04d}"
            folder.mkdir()
            video = folder / f"{i}.mp4"
            video.write_bytes(b"placeholder")
            self.video_paths.append(video)
            choices = ["red", "blue", "green", "gray", "white"]
            decode = {"source_sha256": media,
                      "sampled_frames": [{"relative_presentation_time": [j, 1]}
                                         for j in range(16)]}
            spans = [[2.0, 4.0]]
            sample = {"video_id": f"video-{i}", "qid": qid,
                      "question": f"Question {i}?", "choices": choices,
                      "gold": i, "spans": spans, "video_path": str(video),
                      "frames": [], "decode": decode,
                      "sampled_times": [Fraction(j) for j in range(16)]}
            self.selected.append(sample)
            self.assignments.append({"media_sha256": media, "qid_sha256": qid_hash})
            self.records.append({
                "candidate_pair_sha256": pair, "choices": choices,
                "dataset_question_id": qid,
                "generated_parsed": parse_claim_output("Visible event.", 16),
                "generated_raw_text": "Visible event.",
                "generation_prompt_version": "v3_sequence_complete",
                "gold_answer": choices[i], "gold_interval_seconds": spans,
                "human_review": None, "local_video_path": str(video),
                "question": f"Question {i}?", "review_index": i,
                "sampled_frame_relative_pts_seconds": list(range(16)),
                "video_sha256": media,
            })
        self.exclusions = [hashlib.sha256(b"excluded").hexdigest()]
        self.dataset_hashes = {}
        for name in replay.SPLIT_DATA_FILES["test"]:
            path = self.dataset / name
            path.write_bytes(f"dataset:{name}".encode())
            self.dataset_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        locked_hashes = {name: self.dataset_hashes[name]
                         for name in ("test.csv", "gsub_test.json", "map_vid_vidorID.json")}
        cohort = "a" * 64
        pair_payload = {"schema_version": 1, "pair_sha256": self.pairs}
        pair_raw = json.dumps(pair_payload).encode()
        exclusion_payload = {"schema_version": 1, "media_sha256": self.exclusions}
        exclusion_raw = json.dumps(exclusion_payload).encode()
        lock = {
            "assignments": self.assignments,
            "candidate_pair_filter_sha256": "b" * 64,
            "candidate_pairs_available": 3, "candidate_pool_summary_sha256": "c" * 64,
            "candidate_videos_scanned": 3, "cohort_sha256": cohort,
            "dataset_input_sha256": locked_hashes, "dataset_split": "test",
            "effective_videos": 3, "full_split_candidate_video_count": 3,
            "limitations": [], "local_claim_exposure_manifest_sha256": "d" * 64,
            "local_claim_exposure_media_hash_count": 0,
            "local_claim_exposure_source_file_count": 0,
            "local_claim_exposure_source_inventory_sha256": "e" * 64,
            "local_sample_exclusion_manifest_count": 0,
            "local_sample_exclusion_manifest_digest": "f" * 64,
            "local_scored_claim_overlap_count": 0, "prior_media_exclusion_count": 0,
            "prior_media_exclusion_sha256": "1" * 64,
            "prior_scored_claim_digest_count": 0,
            "prior_scored_claim_digest_set_sha256": "2" * 64,
            "purpose": "exploratory_claim_likelihood_selected_window_prediction_dependence",
            "requested_videos": 3, "schema_version": 1,
            "selected_pair_filter_sha256": hashlib.sha256(pair_raw).hexdigest(),
            "selection_rule": "fixed", "source_sha256": {"source": "3" * 64},
            "status": "frozen_before_claim_generation_and_scoring",
        }
        packet = {
            "authorization_scope": "local", "candidate_pair_filter_digest": _hash(sorted(self.pairs)),
            "candidate_pair_filter_entries": 3,
            "candidate_pair_filter_sha256": hashlib.sha256(pair_raw).hexdigest(),
            "candidate_video_count": 3, "candidate_videos_scanned": 3,
            "candidate_videos_unscanned": 0, "claim_review_completed": False,
            "cohort_sha256": cohort, "compute_dtype": "bfloat16",
            "dataset_inputs_sha256": self.dataset_hashes, "dataset_split": "test",
            "effective_videos": 3, "elapsed_seconds": 1.0,
            "faithfulness_effect_estimate": None, "frames_per_video": 16,
            "fully_structured_claim_records": 3,
            "generation_prompt_version": "v3_sequence_complete", "gpu": "synthetic",
            "gpu_free_before_load_gib": 1.0, "gpu_total_gib": 2.0,
            "host_available_before_load_gib": 1.0, "model": "synthetic",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "peak_allocated_gib": 1.0, "pilot_runner_sha256": "3" * 64,
            "privacy": "synthetic", "private_exclusion_manifest_sha256": hashlib.sha256(exclusion_raw).hexdigest(),
            "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
            "quantization": "4bit", "records": self.records,
            "requested_videos": 3, "runner_sha256": "4" * 64,
            "schema_version": 1, "seed": 1, "source_commit": "5" * 40,
            "started_utc": "2026-01-01T00:00:00+00:00", "status": "complete",
            "uniform_grid_tolerance_seconds": 0.05, "versions": {},
        }
        self.paths = {
            "packet_path": self.private / "packet.json",
            "lock_path": self.private / "lock.json",
            "pair_filter_path": self.private / "pairs.json",
            "exclusion_path": self.private / "exclude.json",
            "video_root": self.video_root, "repo_root": self.root,
            "frame_count": 16,
        }
        for path, value in ((self.paths["packet_path"], packet),
                            (self.paths["lock_path"], lock),
                            (self.paths["pair_filter_path"], pair_payload),
                            (self.paths["exclusion_path"], exclusion_payload)):
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
        self.packet = packet
        self.pair_raw = pair_raw

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, packet=None):
        with patch.object(replay, "freeze_cohort", return_value=(
                self.selected, {"candidate_scan_count": 3}, {}, "a" * 64)), \
             patch.object(replay, "verify_sampled_frame_pixels"):
            if packet is not None:
                packet_path = self.paths["packet_path"]
                packet_path.write_text(json.dumps(packet), encoding="utf-8")
                packet_path.chmod(0o600)
            return replay.reconstruct_samples(**self.paths)

    def test_valid_synthetic_replay_binds_three_locked_rows(self):
        rows, binding = self._run()
        self.assertEqual(len(rows), 3)
        self.assertEqual(binding["cohort_sha256"], "a" * 64)

    def test_rejects_packet_question_mutation_and_row_swap(self):
        for mutate in (lambda p: p["records"][0].update(question="tampered"),
                       lambda p: p["records"].__setitem__(slice(0, 2),
                                                           reversed(p["records"][:2]))):
            packet = json.loads(json.dumps(self.packet))
            mutate(packet)
            with self.assertRaisesRegex(ValueError, "frozen_packet_dataset_row_binding_invalid"):
                self._run(packet)

    def test_rejects_parsed_claim_and_prompt_version_tampering(self):
        packet = json.loads(json.dumps(self.packet))
        packet["records"][0]["generated_parsed"]["claim"] = "Different claim."
        with self.assertRaisesRegex(ValueError, "frozen_packet_parsed_claim_binding_invalid"):
            self._run(packet)
        packet = json.loads(json.dumps(self.packet))
        packet["records"][0]["generation_prompt_version"] = "v2_short"
        with self.assertRaisesRegex(ValueError, "frozen_packet_generation_binding_invalid"):
            self._run(packet)

    def test_rejects_missing_dataset_digest_and_wrong_split(self):
        packet = json.loads(json.dumps(self.packet))
        del packet["dataset_inputs_sha256"]["test.csv"]
        with self.assertRaisesRegex(ValueError, "frozen_dataset_hash_manifest_schema_invalid"):
            self._run(packet)
        with self.assertRaisesRegex(ValueError, "frozen_packet_dataset_split_must_match_cli"):
            validate_replay_dataset_split({"dataset_split": "val"}, "test")

    def test_rejects_symlink_private_packet(self):
        link = self.private / "link.json"
        link.symlink_to(self.paths["packet_path"])
        self.paths["packet_path"] = link
        with self.assertRaisesRegex(ValueError, "frozen_input_path_invalid"):
            self._run()


@unittest.skipUnless(
    (PRIVATE / "claim_generation_cohort11_gpu_full_20261021_01.json").is_file()
    and (PRIVATE / "claim_cohort_lock_20261021_11.json").is_file()
    and (PRIVATE / "claim_cohort_pair_filter_20261021_11.json").is_file()
    and (PRIVATE / "claim_cohort_media_exclusions_20261021_11.json").is_file()
    and (ROOT / "NExT-GQA/datasets/nextgqa/test.csv").is_file(),
    "local protected cohort/data not installed",
)
class FrozenCohortBindingIntegrationTests(unittest.TestCase):
    def _reconstruct(self, packet):
        return replay.reconstruct_samples(
            packet_path=packet,
            lock_path=PRIVATE / "claim_cohort_lock_20261021_11.json",
            pair_filter_path=PRIVATE / "claim_cohort_pair_filter_20261021_11.json",
            exclusion_path=PRIVATE / "claim_cohort_media_exclusions_20261021_11.json",
            video_root=None, repo_root=ROOT, frame_count=16)

    def test_rebinds_all_rows_to_dataset_and_recomputed_cohort(self):
        samples, binding = self._reconstruct(
            PRIVATE / "claim_generation_cohort11_gpu_full_20261021_01.json")
        self.assertEqual(len(samples), 3)
        self.assertEqual(binding["effective_videos"], 3)
        self.assertEqual(len({row["candidate_pair_sha256"] for row in samples}), 3)
        self.assertEqual(binding["cohort_sha256"],
                         "5ca7c89b4607c21fd717ebf4a9a6658e740e9df5441c3829848a1c9138f1e637")

    def test_rejects_changed_dataset_input_digest_before_video_decode(self):
        source = PRIVATE / "claim_generation_cohort11_gpu_full_20261021_01.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["dataset_inputs_sha256"]["test.csv"] = "0" * 64
        fd, name = tempfile.mkstemp(prefix=".replay-tamper-", suffix=".json", dir=PRIVATE)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, "frozen_dataset_input_hash_mismatch"):
                self._reconstruct(Path(name))
        finally:
            Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
