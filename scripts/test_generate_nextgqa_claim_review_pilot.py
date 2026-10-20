import unittest
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_nextgqa_claim_review_pilot import (
    PROMPT_SUFFIX_BY_VERSION, candidate_pair_digest_for_sample, parse_claim_output,
    read_candidate_pair_filter,
)
from run_nextgqa_gold_interval_margin_pilot import candidate_pair_sha256, load_candidate_rows


class ClaimOutputParserTests(unittest.TestCase):
    def test_short_prompt_version_is_explicit_and_legacy_v1_stays_unchanged(self):
        self.assertEqual(set(PROMPT_SUFFIX_BY_VERSION), {"v1", "v2_short"})
        self.assertIn("25 words or fewer", PROMPT_SUFFIX_BY_VERSION["v2_short"])
        self.assertTrue(PROMPT_SUFFIX_BY_VERSION["v2_short"].startswith(
            PROMPT_SUFFIX_BY_VERSION["v1"]))

    def test_parses_one_plain_claim_line_without_model_frame_selection(self):
        parsed = parse_claim_output("CLAIM=A person opens the cabinet.", 16)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["claim"], "A person opens the cabinet.")
        self.assertIsNone(parsed["answer_letter"])
        self.assertEqual(parsed["evidence_frame_indices"], [])

    def test_packet_pair_digest_binds_sample_video_and_question(self):
        self.assertEqual(
            candidate_pair_digest_for_sample({"video_id": "vid", "qid": "qid"}),
            candidate_pair_sha256("vid", "qid"))
        self.assertNotEqual(
            candidate_pair_digest_for_sample({"video_id": "other", "qid": "qid"}),
            candidate_pair_sha256("vid", "qid"))


class SplitAwareClaimCohortTests(unittest.TestCase):
    def test_validation_split_uses_its_own_csv_and_gsub_with_test_default_unchanged(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset = repo / "NExT-GQA/datasets/nextgqa"
            dataset.mkdir(parents=True)
            video_root = repo / "videos"
            (video_root / "abc").mkdir(parents=True)
            (video_root / "abc/v.mp4").write_bytes(b"placeholder")
            (dataset / "map_vid_vidorID.json").write_text(
                json.dumps({"vid": "abc/v.mp4"}), encoding="utf-8")
            for split, qid in (("val", "val-q"), ("test", "test-q")):
                (dataset / f"gsub_{split}.json").write_text(json.dumps({
                    "vid": {"duration": 10, "location": {qid: [[1, 2]]}}
                }), encoding="utf-8")
                with (dataset / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "video_id", "qid", "question", "answer",
                        "a0", "a1", "a2", "a3", "a4"])
                    writer.writeheader()
                    writer.writerow({"video_id": "vid", "qid": qid,
                        "question": f"Which event in {split}?", "answer": "yes",
                        "a0": "yes", "a1": "no", "a2": "maybe", "a3": "other", "a4": "none"})
            val, _ = load_candidate_rows(repo, video_root, dataset_split="val")
            default, _ = load_candidate_rows(repo, video_root)
            explicit_test, _ = load_candidate_rows(repo, video_root, dataset_split="test")
            self.assertEqual([q["qid"] for _, qs in val for q in qs], ["val-q"])
            val_record = val[0][1][0]
            self.assertEqual(val_record["video_id"], "vid")
            self.assertEqual(candidate_pair_sha256(val_record["video_id"], val_record["qid"]),
                             candidate_pair_sha256("vid", "val-q"))
            self.assertEqual([q["qid"] for _, qs in default for q in qs], ["test-q"])
            self.assertEqual(default, explicit_test)

    def test_pair_allowlist_filters_candidates_and_records_stable_digest(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dataset = repo / "NExT-GQA/datasets/nextgqa"
            dataset.mkdir(parents=True)
            video_root = repo / "videos"
            (video_root / "abc").mkdir(parents=True)
            (video_root / "abc/v.mp4").write_bytes(b"placeholder")
            (dataset / "map_vid_vidorID.json").write_text(
                json.dumps({"vid": "abc/v.mp4"}), encoding="utf-8")
            (dataset / "gsub_val.json").write_text(json.dumps({
                "vid": {"duration": 10, "location": {
                    "val-q": [[1, 2]], "other-q": [[2, 3]]}}
            }), encoding="utf-8")
            with (dataset / "val.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "video_id", "qid", "question", "answer",
                    "a0", "a1", "a2", "a3", "a4"])
                writer.writeheader()
                for qid in ("val-q", "other-q"):
                    writer.writerow({"video_id": "vid", "qid": qid,
                        "question": f"Which event {qid}?", "answer": "yes",
                        "a0": "yes", "a1": "no", "a2": "maybe", "a3": "other", "a4": "none"})
            allowed = candidate_pair_sha256("vid", "other-q")
            filtered, _ = load_candidate_rows(
                repo, video_root, [allowed], dataset_split="val")
            self.assertEqual([q["qid"] for _, qs in filtered for q in qs], ["other-q"])

            filter_path = repo / "pairs.json"
            filter_path.write_text(json.dumps({"schema_version": 1,
                "pair_sha256": [allowed]}), encoding="utf-8")
            pairs, source_digest, semantic_digest = read_candidate_pair_filter(filter_path)
            self.assertEqual(pairs, [allowed])
            self.assertEqual(source_digest, hashlib.sha256(
                filter_path.read_bytes()).hexdigest())
            self.assertEqual(semantic_digest,
                hashlib.sha256(json.dumps([allowed], ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest())

    def test_pair_filter_rejects_invalid_manifest(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text('{"schema_version":1,"pair_sha256":["raw-id"]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "private_candidate_pair_filter_invalid"):
                read_candidate_pair_filter(path)

    def test_invalid_dataset_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "nextgqa_dataset_split_invalid"):
            load_candidate_rows(Path("/unused"), Path("/unused"), dataset_split="dev")

    def test_rejects_multiline_or_extra_output(self):
        for text in ("CLAIM=A person opens the cabinet.\nF=1,2",
                     "CLAIM=\nA person opens the cabinet."):
            parsed = parse_claim_output(text, 16)
            self.assertFalse(parsed["valid"])

    def test_rejects_template_and_special_token_artifacts(self):
        for text in ("CLAIM=A addCriterion fragment", "CLAIM=<|assistant|>event"):
            parsed = parse_claim_output(text, 16)
            self.assertFalse(parsed["valid"])
            self.assertEqual(parsed["claim_text_quality_failure"],
                             "claim_template_or_token_artifact")

    def test_empty_or_unlabeled_output_is_invalid(self):
        self.assertFalse(parse_claim_output("", 16)["valid"])
        self.assertTrue(parse_claim_output("A person opens the cabinet.", 16)["valid"])


if __name__ == "__main__":
    unittest.main()
