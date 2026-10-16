import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from PIL import Image

import build_nextgqa_diagnostic_target_candidate as candidate


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


class FakeTokenizer:
    def __call__(self, label, add_special_tokens=False):
        return {"input_ids": [ord(label)]}


class FakeProcessor:
    tokenizer = FakeTokenizer()
    video_processor = type("VP", (), {"temporal_patch_size": 2})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = messages[0]["content"][1]["text"]
        return "rendered-prefix:" + text

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[5, 7, 11]]),
                "video_grid_thw": torch.tensor([[16, 1, 1]]),
                "second_per_grid_ts": torch.tensor([2.0])}


def make_fixture(root: Path):
    video_root = root / "videos"
    video_root.mkdir()
    rows, lineage_rows, worksheet_rows, packet_rows = [], [], [], []
    csv_stream = io.StringIO(newline="")
    fields = ["video_id", "qid", "question", "answer", "a0", "a1", "a2", "a3", "a4", "type"]
    writer = csv.DictWriter(csv_stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i in range(3):
        media_raw = f"synthetic-video-{i}".encode()
        (video_root / f"v{i}.mp4").write_bytes(media_raw)
        media_sha = sha(media_raw)
        question = f"synthetic question {i}?"
        claim = f"synthetic claim {i}."
        qid = f"qid-{i}"
        dataset_video = f"dataset-video-{i}"
        csv_row = {"video_id": dataset_video, "qid": qid, "question": question,
                   "answer": f"gold-{i}", "a0": f"gold-{i}",
                   "a1": f"foil-{i}-1", "a2": f"foil-{i}-2",
                   "a3": f"foil-{i}-3", "a4": f"foil-{i}-4", "type": "TN"}
        writer.writerow(csv_row)
        claim_sha = sha(claim.encode())
        question_sha = sha(question.encode())
        video_id = "video-" + media_sha
        claim_id = "claim-" + sha((media_sha + "\0" + qid + "\0" + claim_sha).encode())
        packet_video = root / f"packet-{i}.mp4"
        packet_video.write_bytes(media_raw)
        packet_rows.append({"dataset_question_id": qid, "video_sha256": media_sha,
                            "question": question, "local_video_path": str(packet_video),
                            "generated_parsed": {"claim": claim}})
        lineage_rows.append({"video_id": video_id, "claim_id": claim_id,
                             "dataset_question_id": qid, "dataset_video_id": dataset_video,
                             "video_key": f"v{i}", "relative_video_path": f"v{i}.mp4",
                             "video_sha256": media_sha, "question_sha256": question_sha,
                             "claim_text_sha256": claim_sha})
        worksheet_rows.append({"video_id": video_id, "claim_id": claim_id,
                               "video_sha256": media_sha, "claim_text_sha256": claim_sha,
                               "claim_text": claim, "question": question,
                               "review": {"review_status": "not_started"}})
        rows.append(csv_row)
    csv_raw = csv_stream.getvalue().encode()
    dataset_inputs = {"test.csv": sha(csv_raw), "gsub_test.json": sha(b"gsub"),
                      "map_vid_vidorID.json": sha(b"map"),
                      "frame2time_test.json": sha(b"times")}
    generation = {"schema_version": 1, "purpose": candidate.GENERATION_PURPOSE,
                  "status": "passed", "requested_videos": 3, "effective_videos": 3,
                  "cohort_sha256": sha(b"cohort"),
                  "dataset_inputs_sha256": dataset_inputs, "records": packet_rows}
    generation_raw = canonical(generation)
    lineage = {"schema_version": 1, "purpose": candidate.LINEAGE_PURPOSE,
               "created_by": "synthetic fixture", "source_packet_sha256": sha(generation_raw),
               "cohort_sha256": generation["cohort_sha256"],
               "dataset_inputs_sha256": dataset_inputs, "requested_records": 3,
               "effective_records": 3, "verification_scope": "synthetic",
               "status": "metadata_lineage_crosschecked", "records": lineage_rows}
    lineage_raw = canonical(lineage)
    worksheet = {"schema_version": 1, "purpose": candidate.WORKSHEET_PURPOSE,
                 "status": "draft_not_reviewed", "source_packet_sha256": sha(generation_raw),
                 "cohort_sha256": generation["cohort_sha256"],
                 "dataset_inputs_sha256": dataset_inputs,
                 "requested_records": 3, "effective_records": 3,
                 "claim_records_available": 3, "review_completed": False,
                 "execution_authorization": "not_granted", "records": worksheet_rows}
    worksheet_raw = canonical(worksheet)
    protocol_raw = (Path(__file__).resolve().parents[1] / "docs"
                    / "nextgqa_finite_claim_descriptive_protocol_v2.md").read_bytes()
    return {"csv_raw": csv_raw, "generation_raw": generation_raw,
            "lineage_raw": lineage_raw, "worksheet_raw": worksheet_raw,
            "protocol_raw": protocol_raw, "video_root": video_root,
            "records": rows, "lineage": lineage, "generation": generation,
            "worksheet": worksheet}


def make_model_snapshot(root: Path) -> Path:
    model = root / candidate.MODEL_REVISION
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    (model / "model-00001-of-00001.safetensors").write_bytes(b"synthetic weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.synthetic": "model-00001-of-00001.safetensors"}}))
    (model / "tokenizer.json").write_text("{}")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "preprocessor_config.json").write_text("{}")
    return model


def synthetic_decoder(path, *, num_frames, uniform_tolerance_seconds):
    raw_sha = sha(path.read_bytes())
    frames = [Image.new("RGB", (2, 2), (i % 255, 1, 2)) for i in range(num_frames)]
    return frames, {"source_sha256": raw_sha, "effective_sample_fps": (1, 1)}


def synthetic_vision(messages, return_video_metadata=True):
    import numpy as np
    frames = messages[0]["content"][0]["video"]
    pixels = torch.stack([
        torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1)
        for frame in frames])
    return None, [(pixels,
                   {"frames_indices": list(range(len(frames))), "fps": 1.0})]


class DiagnosticTargetCandidateTests(unittest.TestCase):
    def build(self, root, fixture, *, decoder_fn=None):
        kwargs = dict(
            **{key: fixture[key] for key in (
                "csv_raw", "generation_raw", "lineage_raw", "worksheet_raw", "protocol_raw",
                "video_root")},
            model_path=make_model_snapshot(root), processor=FakeProcessor(), frame_count=32,
            verify_frames_fn=lambda frames, meta: None, vision_info_fn=synthetic_vision)
        if decoder_fn is not None:
            kwargs["decoder_fn"] = decoder_fn
        return candidate.build_candidate(**kwargs)

    def test_builds_diagnostic_rows_with_production_prompt_processor_and_no_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = make_fixture(root)
            payload = self.build(root, fixture, decoder_fn=synthetic_decoder)
        self.assertEqual(payload["status"], "diagnostic_candidate_unvalidated")
        self.assertEqual(payload["effective_records"], 3)
        self.assertEqual(payload["source_row_bindings"], 3)
        self.assertFalse(payload["claim_input_manifest_present"])
        self.assertIsNone(payload["claim_input_manifest_sha256"])
        self.assertEqual(payload["model_weights_loaded"], False)
        self.assertEqual(payload["cuda_used"], False)
        self.assertEqual(payload["score_calls"], 0)
        self.assertEqual(payload["execution_authorization"], "not_granted")
        self.assertEqual(len(payload["records"]), 3)
        self.assertEqual(payload["records"][0]["answer_label_token_ids"], list(map(ord, "ABCDE")))
        self.assertEqual(payload["records"][0]["answer_label_token_piece_counts"], [1] * 5)
        self.assertTrue(all(len(row["rendered_scoring_prefix_sha256"]) == 64
                            and len(row["processor_input_ids_sha256"]) == 64
                            for row in payload["records"]))
        self.assertTrue(all(row["processor_input_ids_hash_scope"]
                            == "input_ids_only_not_pixels_or_video_metadata"
                            for row in payload["records"]))
        encoded = json.dumps(payload)
        self.assertNotIn("synthetic question", encoded)
        self.assertNotIn("synthetic claim", encoded)
        self.assertNotIn("gold-0", encoded)

    def test_generation_claim_and_worksheet_claim_tampering_fail_before_decode(self):
        for target in ("generation", "worksheet"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture = make_fixture(root)
                if target == "generation":
                    generation = json.loads(fixture["generation_raw"])
                    generation["records"][0]["generated_parsed"]["claim"] = "tampered claim."
                    fixture["generation_raw"] = canonical(generation)
                else:
                    worksheet = json.loads(fixture["worksheet_raw"])
                    worksheet["records"][0]["claim_text"] = "tampered claim."
                    fixture["worksheet_raw"] = canonical(worksheet)
                with mock.patch.object(candidate, "decode_sampled_video") as decode:
                    with self.assertRaises(candidate.CandidateError):
                        self.build(root, fixture)
                    decode.assert_not_called()

    def test_protocol_v2_identity_status_and_frame_spec_fail_closed_before_decode(self):
        mutations = (
            (b"protocol v2", b"protocol v1"),
            (b"design draft; not frozen", b"approved; frozen"),
            (b"Use 32 endpoint-inclusive", b"Use 31 endpoint-inclusive"),
        )
        for old, new in mutations:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture = make_fixture(root)
                fixture["protocol_raw"] = fixture["protocol_raw"].replace(old, new, 1)
                with mock.patch.object(candidate, "decode_sampled_video") as decode:
                    with self.assertRaises(candidate.CandidateError):
                        self.build(root, fixture, decoder_fn=synthetic_decoder)
                    decode.assert_not_called()

    def test_qwen_pixel_signature_mismatch_or_frame_reorder_fails(self):
        for reorder in (False, True):
            with self.subTest(reorder=reorder), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture = make_fixture(root)

                def wrong_pixels(messages, return_video_metadata=True):
                    import numpy as np
                    frames = messages[0]["content"][0]["video"]
                    if reorder:
                        frames = list(reversed(frames))
                    pixels = torch.stack([
                        torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1)
                        for frame in frames])
                    if not reorder:
                        pixels.zero_()
                    return None, [(pixels, {"frames_indices": list(range(len(frames))),
                                            "fps": 1.0})]

                with self.assertRaisesRegex(candidate.CandidateError,
                                            "qwen_frame_pixel_identity_or_order_mismatch"):
                    candidate.build_candidate(
                        **{key: fixture[key] for key in (
                            "csv_raw", "generation_raw", "lineage_raw", "worksheet_raw",
                            "protocol_raw", "video_root")},
                        model_path=make_model_snapshot(root), processor=FakeProcessor(),
                        frame_count=32, verify_frames_fn=lambda frames, meta: None,
                        decoder_fn=synthetic_decoder, vision_info_fn=wrong_pixels)

    def test_tokenizer_fingerprint_includes_added_tokens_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text("{}")
            optional = ("vocab.json", "merges.txt", "special_tokens_map.json",
                        "added_tokens.json")
            before = candidate._hash_file_set(
                root, ("tokenizer.json", "tokenizer_config.json"), optional,
                "tokenizer_files_missing")
            (root / "added_tokens.json").write_text('{"extra":1}')
            after = candidate._hash_file_set(
                root, ("tokenizer.json", "tokenizer_config.json"), optional,
                "tokenizer_files_missing")
        self.assertNotEqual(before, after)

    def test_changed_csv_choice_order_fails_closed_against_frozen_dataset_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = make_fixture(root)
            changed = fixture["csv_raw"].replace(b"foil-0-1", b"CHANGED-0-1")
            fixture["csv_raw"] = changed
            with self.assertRaisesRegex(candidate.CandidateError,
                                        "cohort_or_source_manifest_mismatch"):
                candidate._validate_upstream(
                    csv_raw=fixture["csv_raw"], generation_raw=fixture["generation_raw"],
                    lineage_raw=fixture["lineage_raw"], worksheet_raw=fixture["worksheet_raw"],
                    video_root=fixture["video_root"])

    def test_exclusive_private_write_sets_mode_and_rejects_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "data").mkdir()
            private = repo / "data" / "private_nextgqa"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            output = private / "candidate.json"
            digest = candidate._write_exclusive_private(
                output, {"status": "diagnostic_candidate_unvalidated"}, repo)
            self.assertEqual(digest, sha(output.read_bytes()))
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                candidate._write_exclusive_private(
                    output, {"status": "changed"}, repo)
            self.assertEqual(json.loads(output.read_text())["status"],
                             "diagnostic_candidate_unvalidated")

    def test_cli_stdout_is_aggregate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = make_fixture(root)
            paths = {}
            for arg, key, name in (
                ("--csv", "csv_raw", "test.csv"),
                ("--generation-packet", "generation_raw", "generation.json"),
                ("--lineage", "lineage_raw", "lineage.json"),
                ("--worksheet", "worksheet_raw", "worksheet.json"),
                ("--protocol", "protocol_raw", "protocol.md"),
            ):
                path = root / name
                path.write_bytes(fixture[key])
                paths[arg] = path
            output_path = root / "candidate.json"
            expected_payload = {
                "status": "diagnostic_candidate_unvalidated", "requested_records": 3,
                "effective_records": 3, "source_row_bindings": 3,
                "score_calls": 0, "execution_authorization": "not_granted",
            }
            with mock.patch.object(candidate, "_load_local_processor", return_value=FakeProcessor()), \
                    mock.patch.object(candidate, "build_candidate", return_value=expected_payload), \
                    mock.patch.object(candidate, "_write_exclusive_private", return_value=sha(b"artifact")), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = candidate.main([
                    "--csv", str(paths["--csv"]),
                    "--generation-packet", str(paths["--generation-packet"]),
                    "--lineage", str(paths["--lineage"]),
                    "--worksheet", str(paths["--worksheet"]),
                    "--protocol", str(paths["--protocol"]),
                    "--video-root", str(fixture["video_root"]),
                    "--model-path", str(root / candidate.MODEL_REVISION),
                    "--frames", "32", "--output", str(output_path),
                ])
            self.assertEqual(code, 0)
            emitted = stdout.getvalue()
            self.assertIn('"effective_records": 3', emitted)
            self.assertNotIn("synthetic question", emitted)
            self.assertNotIn("synthetic claim", emitted)
            self.assertNotIn("gold-0", emitted)
            self.assertNotIn("dataset-video-0", emitted)


if __name__ == "__main__":
    unittest.main()
