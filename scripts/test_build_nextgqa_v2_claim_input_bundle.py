import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_nextgqa_v2_claim_input_bundle import (
    ALL_RELEVANT_OCCURRENCES,
    V2_COHORT_SHA256,
    V6_POLICY_SHA256,
    build_v2_claim_input_bundle,
    main,
    write_private_pair_exclusive,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def fixture(testcase=None):
    policy_raw = (Path(__file__).resolve().parents[1] / "docs"
                  / "nextgqa_claim_review_policy_v6.json").read_bytes()
    temp = tempfile.TemporaryDirectory()
    video_root = Path(temp.name) / "videos"
    video_root.mkdir()
    csv_lines = ["qid,video_id,question"]
    video_map = {}
    media_values = []
    for i in range(3):
        media = f"synthetic media {i}".encode()
        path = video_root / "synthetic" / f"{i}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(media)
        media_values.append(media)
        csv_lines.append(f"synthetic-qid-{i},dataset-video-{i},Synthetic question {i}?")
        video_map[f"dataset-video-{i}"] = f"synthetic/{i}.mp4"
    test_csv_raw = ("\n".join(csv_lines) + "\n").encode()
    video_map_raw = encode(video_map)
    dataset_inputs = {
        "test.csv": sha(test_csv_raw),
        "gsub_test.json": sha(b"synthetic gsub"),
        "map_vid_vidorID.json": sha(video_map_raw),
        "frame2time_test.json": sha(b"synthetic frame times"),
    }
    generation_records = []
    lineage_records = []
    eligible = []
    for i in range(3):
        media_sha = sha(media_values[i])
        qid = f"synthetic-qid-{i}"
        dataset_video_id = f"dataset-video-{i}"
        question = f"Synthetic question {i}?"
        claim = f"A visible synthetic event occurs in scene {i}."
        claim_sha = sha(claim.encode("utf-8"))
        question_sha = sha(question.encode("utf-8"))
        video_id = "video-" + media_sha
        claim_id = "claim-" + sha(
            (media_sha + "\0" + qid + "\0" + claim_sha).encode("utf-8"))
        generation_records.append({
            "review_index": i,
            "local_video_path": str(video_root / "synthetic" / f"{i}.mp4"),
            "video_sha256": media_sha,
            "dataset_question_id": qid,
            "question": question,
            "choices": [f"Synthetic answer {i}"],
            "gold_answer": f"Synthetic answer {i}",
            "gold_interval_seconds": [[0.1, 0.2]],
            "generated_raw_text": "PRIVATE SYNTHETIC RAW QA",
            "generated_parsed": {"claim": claim},
        })
        lineage_records.append({
            "video_id": video_id,
            "claim_id": claim_id,
            "dataset_question_id": qid,
            "dataset_video_id": dataset_video_id,
            "video_key": f"synthetic/{i}",
            "relative_video_path": f"synthetic/{i}.mp4",
            "video_sha256": media_sha,
            "question_sha256": question_sha,
            "claim_text_sha256": claim_sha,
        })
        eligible.append({"video_id": video_id})
    generation = {
        "schema_version": 1,
        "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
        "status": "passed",
        "source_commit": "a" * 40,
        "runner_sha256": sha(b"synthetic runner"),
        "pilot_runner_sha256": sha(b"synthetic pilot runner"),
        "dataset_inputs_sha256": dataset_inputs,
        "private_exclusion_manifest_sha256": sha(b"synthetic exclusions"),
        "cohort_sha256": V2_COHORT_SHA256,
        "requested_videos": 3,
        "effective_videos": 3,
        "fully_structured_claim_records": 3,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "records": generation_records,
    }
    generation_raw = encode(generation)
    lineage = {
        "schema_version": 1,
        "purpose": "private_source_lineage_crosscheck_for_claim_review",
        "created_by": "synthetic lineage builder",
        "source_packet_sha256": sha(generation_raw),
        "cohort_sha256": V2_COHORT_SHA256,
        "dataset_inputs_sha256": dataset_inputs,
        "requested_records": 3,
        "effective_records": 3,
        "verification_scope": "synthetic row binding only",
        "status": "metadata_lineage_crosschecked",
        "records": lineage_records,
    }
    if testcase is not None:
        testcase.addCleanup(temp.cleanup)
    return {
        "eligible": eligible, "eligible_raw": encode(eligible),
        "generation": generation, "generation_raw": generation_raw,
        "lineage": lineage, "lineage_raw": encode(lineage),
        "policy_raw": policy_raw, "dataset_id": "synthetic-nextgqa",
        "test_csv_raw": test_csv_raw, "video_map_raw": video_map_raw,
        "video_root": video_root, "temp": temp,
    }


def build(f):
    return build_v2_claim_input_bundle(
        eligible_manifest_raw=f["eligible_raw"],
        generation_export_raw=f["generation_raw"],
        lineage_manifest_raw=f["lineage_raw"],
        policy_raw=f["policy_raw"], dataset_id=f["dataset_id"],
        test_csv_raw=f["test_csv_raw"], video_map_raw=f["video_map_raw"],
        video_root=f["video_root"],
        worksheet_raw=f.get("worksheet_raw"),
    )


def worksheet_fixture(f):
    records = []
    for i, lineage in enumerate(f["lineage"]["records"]):
        claim = f["generation"]["records"][i]["generated_parsed"]["claim"]
        question = f["generation"]["records"][i]["question"]
        records.append({
            "video_path": f["generation"]["records"][i]["local_video_path"],
            "video_sha256": lineage["video_sha256"],
            "video_id": lineage["video_id"], "claim_id": lineage["claim_id"],
            "review_token": f"token_{i:021d}", "question": question,
            "claim_text": claim, "claim_text_sha256": lineage["claim_text_sha256"],
            "review": {
                key: ("not_started" if key == "review_status" else None)
                for key in {
                    "source_identity_verified", "full_source_video_reviewed",
                    "question_and_claim_only_context_shown",
                    "gold_and_seed_hidden_until_mapping_frozen", "outcome_scores_not_accessed",
                    "occurrence_policy_applied", "support_decision", "evidence_intervals_seconds",
                    "rationale", "reviewer_role", "review_record_reference",
                    "first_played_at_utc", "reviewed_at_utc", "review_status",
                }
            },
        })
    worksheet = {
        "schema_version": 1, "purpose": "private_nextgqa_claim_review_worksheet",
        "status": "draft_not_reviewed", "created_at_utc": "synthetic",
        "source_packet_sha256": sha(f["generation_raw"]), "cohort_sha256": V2_COHORT_SHA256,
        "model_revision": f["generation"]["model_revision"], "source_commit": "a" * 40,
        "claim_packet_runner_sha256": sha(b"runner"),
        "cohort_selector_runner_sha256": sha(b"selector"),
        "dataset_inputs_sha256": f["generation"]["dataset_inputs_sha256"],
        "occurrence_policy": ALL_RELEVANT_OCCURRENCES,
        "review_procedure_status": "frozen_pre_review", "requested_records": 3,
        "effective_records": 3, "claim_records_available": 3,
        "review_completed": False, "execution_authorization": "not_granted",
        "records": records, "claim_review_policy_id": "nextgqa_claim_review_v6",
        "claim_review_policy_sha256": V6_POLICY_SHA256,
    }
    return encode(worksheet)


class BuildNextgqaV2ClaimInputBundleTests(unittest.TestCase):
    def test_permuted_eligible_order_is_retained_in_v1_manifest_and_sidecar(self):
        f = fixture(self)
        order = [2, 0, 1]
        f["eligible"] = [f["eligible"][i] for i in order]
        f["eligible_raw"] = encode(f["eligible"])
        manifest, manifest_raw, sidecar, sidecar_raw = build(f)
        expected_ids = [row["video_id"] for row in f["eligible"]]
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual([row["video_id"] for row in manifest["claims"]], expected_ids)
        self.assertEqual([row["video_id"] for row in sidecar["records"]], expected_ids)
        self.assertEqual(manifest["occurrence_policy"], ALL_RELEVANT_OCCURRENCES)
        self.assertEqual(manifest["claim_policy_sha256"], V6_POLICY_SHA256)
        self.assertEqual(sidecar["status"], "candidate_unreviewed")
        self.assertEqual(sidecar["scope"], "structural_only")
        self.assertFalse(sidecar["review_completed"])
        self.assertFalse(sidecar["scoring_ready"])
        self.assertEqual(sidecar["execution_authorization"], "not_granted")
        self.assertEqual(sidecar["claim_input_manifest_sha256"], sha(manifest_raw))
        self.assertEqual(sidecar["input_sha256"], {
            "eligible_manifest": sha(f["eligible_raw"]),
            "generation_packet": sha(f["generation_raw"]),
            "lineage_manifest": sha(f["lineage_raw"]),
            "policy": V6_POLICY_SHA256,
            "test_csv": sha(f["test_csv_raw"]),
            "video_map": sha(f["video_map_raw"]),
        })
        self.assertTrue(all(row["generation_run_sha256"] == sha(f["generation_raw"])
                            for row in manifest["claims"]))
        for forbidden in ("Synthetic question", "A visible synthetic event", "Synthetic answer",
                          "local_video_path", "relative_video_path", "synthetic/",
                          "PRIVATE SYNTHETIC RAW QA"):
            self.assertNotIn(forbidden, sidecar_raw.decode("utf-8"))
        self.assertEqual(json.loads(manifest_raw), manifest)
        self.assertEqual(json.loads(sidecar_raw), sidecar)

    def test_detached_or_mutated_lineage_fails_closed(self):
        for mutation in ("detached", "claim", "question", "dataset_video"):
            with self.subTest(mutation=mutation):
                f = fixture(self)
                lineage = copy.deepcopy(f["lineage"])
                if mutation == "detached":
                    lineage["source_packet_sha256"] = "0" * 64
                elif mutation == "claim":
                    lineage["records"][0]["claim_id"] = "claim-" + "0" * 64
                elif mutation == "question":
                    lineage["records"][0]["question_sha256"] = "0" * 64
                else:
                    lineage["records"][0]["dataset_video_id"] = "detached-video-id"
                f["lineage"] = lineage
                f["lineage_raw"] = encode(lineage)
                with self.assertRaises(ValueError):
                    build(f)

    def test_mutated_claim_text_or_claim_id_binding_fails_closed(self):
        for mutation in ("claim_text", "claim_id"):
            with self.subTest(mutation=mutation):
                f = fixture(self)
                generation = copy.deepcopy(f["generation"])
                if mutation == "claim_text":
                    generation["records"][0]["generated_parsed"]["claim"] = "Changed claim text."
                else:
                    f["lineage"]["records"][0]["claim_id"] = "claim-" + "f" * 64
                    f["lineage_raw"] = encode(f["lineage"])
                f["generation"] = generation
                f["generation_raw"] = encode(generation)
                if mutation == "claim_text":
                    # The lineage source digest must bind the exact generation bytes.
                    f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
                    f["lineage_raw"] = encode(f["lineage"])
                with self.assertRaises(ValueError):
                    build(f)

    def test_generation_raw_bytes_are_bound_by_lineage_source_hash(self):
        f = fixture(self)
        f["generation_raw"] += b" "
        with self.assertRaisesRegex(ValueError, "lineage_manifest_binding_invalid"):
            build(f)

    def test_source_mapping_and_media_bytes_are_required(self):
        f = fixture(self)
        f["video_map_raw"] = encode({"dataset-video-0": "wrong/0.mp4",
                                     "dataset-video-1": "synthetic/1.mp4",
                                     "dataset-video-2": "synthetic/2.mp4"})
        with self.assertRaises(ValueError):
            build(f)
        f = fixture(self)
        (f["video_root"] / "synthetic" / "0.mp4").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "source_video_hash_or_path_mismatch"):
            build(f)

    def test_media_symlink_escape_is_rejected(self):
        f = fixture(self)
        source = f["video_root"] / "synthetic" / "0.mp4"
        outside = f["video_root"].parent / "outside.mp4"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaises(ValueError):
            build(f)

    def test_media_path_escape_and_bad_numeric_schema_fail_closed(self):
        f = fixture(self)
        generation = copy.deepcopy(f["generation"])
        escaped = f["video_root"].parent / "escape.mp4"
        escaped.write_bytes(b"synthetic escape")
        generation["records"][0]["local_video_path"] = str(escaped)
        f["generation_raw"] = encode(generation)
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        with self.assertRaises(ValueError):
            build(f)
        f = fixture(self)
        generation = copy.deepcopy(f["generation"])
        generation["schema_version"] = True
        f["generation_raw"] = encode(generation)
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        with self.assertRaisesRegex(ValueError, "generation_export_scope_invalid"):
            build(f)

    def test_generator_model_and_revision_are_bound_as_emitted(self):
        f = fixture(self)
        manifest, _, sidecar, _ = build(f)
        expected_revision = "Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5"
        self.assertTrue(all(row["model_revision"] == expected_revision
                            for row in manifest["claims"]))
        self.assertEqual(sidecar["model"], "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(sidecar["model_revision"], "cc594898137f460bfe9f0759e9844b3ce807cfb5")
        generation = copy.deepcopy(f["generation"])
        generation["model"] = "unexpected-model"
        f["generation_raw"] = encode(generation)
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        with self.assertRaisesRegex(ValueError, "generation_export_scope_invalid"):
            build(f)
        f = fixture(self)
        generation = copy.deepcopy(f["generation"])
        generation["model_revision"] = "c" * 40
        f["generation_raw"] = encode(generation)
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        with self.assertRaisesRegex(ValueError, "generation_export_scope_invalid"):
            build(f)

    def test_optional_worksheet_crosscheck_is_hash_only_in_output(self):
        f = fixture(self)
        f["worksheet_raw"] = worksheet_fixture(f)
        _, _, sidecar, sidecar_raw = build(f)
        self.assertEqual(
            sidecar["worksheet_binding_check"],
            "claim_and_identity_hashes_matched_only_not_review_or_custody_validation",
        )
        self.assertEqual(sidecar["input_sha256"]["worksheet"], sha(f["worksheet_raw"]))
        self.assertNotIn(b"Synthetic question", sidecar_raw)
        self.assertNotIn(b"A visible synthetic event", sidecar_raw)
        worksheet = json.loads(f["worksheet_raw"])
        worksheet["records"][0]["claim_text"] = "changed claim"
        f["worksheet_raw"] = encode(worksheet)
        with self.assertRaisesRegex(ValueError, "worksheet_claim_hash_binding_mismatch"):
            build(f)
        f = fixture(self)
        worksheet = json.loads(worksheet_fixture(f))
        worksheet["records"][0]["video_path"] = f["generation"]["records"][1]["local_video_path"]
        f["worksheet_raw"] = encode(worksheet)
        with self.assertRaisesRegex(ValueError, "worksheet_video_path_binding_invalid"):
            build(f)

    def test_reused_qid_requires_unique_video_key_pair_and_private_pair_is_exclusive(self):
        f = fixture(self)
        csv_raw = f["test_csv_raw"].decode().replace(
            "synthetic-qid-2,dataset-video-2,Synthetic question 2?",
            "synthetic-qid-0,dataset-video-2,Synthetic question 2?",
        ).encode()
        f["test_csv_raw"] = csv_raw
        f["generation"]["records"][2]["dataset_question_id"] = "synthetic-qid-0"
        f["generation"]["dataset_inputs_sha256"]["test.csv"] = sha(csv_raw)
        f["generation_raw"] = encode(f["generation"])
        reused_qid = "synthetic-qid-0"
        claim = f["generation"]["records"][2]["generated_parsed"]["claim"]
        media_sha = f["generation"]["records"][2]["video_sha256"]
        claim_sha = sha(claim.encode())
        f["lineage"]["records"][2]["dataset_question_id"] = reused_qid
        f["lineage"]["records"][2]["claim_id"] = "claim-" + sha(
            (media_sha + "\0" + reused_qid + "\0" + claim_sha).encode())
        f["lineage"]["dataset_inputs_sha256"]["test.csv"] = sha(csv_raw)
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        build(f)  # Same QID is valid when each (QID, video_key) pair is unique.

        f = fixture(self)
        duplicate_row = b"synthetic-qid-0,dataset-video-0,Synthetic question 0?\n"
        f["test_csv_raw"] += duplicate_row
        f["generation"]["dataset_inputs_sha256"]["test.csv"] = sha(f["test_csv_raw"])
        f["generation_raw"] = encode(f["generation"])
        f["lineage"]["dataset_inputs_sha256"]["test.csv"] = sha(f["test_csv_raw"])
        f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
        f["lineage_raw"] = encode(f["lineage"])
        with self.assertRaisesRegex(ValueError, "dataset_video_question_binding_mismatch"):
            build(f)

        f = fixture(self)
        _, manifest_raw, _, sidecar_raw = build(f)
        repo = Path(f["temp"].name) / "repo"
        repo.mkdir()
        private_dir = repo / "data" / "private_nextgqa"
        private_dir.mkdir(parents=True)
        private_dir.chmod(0o700)
        manifest_path = private_dir / "manifest.json"
        sidecar_path = private_dir / "sidecar.json"
        write_private_pair_exclusive(
            manifest_path=manifest_path, manifest_raw=manifest_raw,
            sidecar_path=sidecar_path, sidecar_raw=sidecar_raw, repo_root=repo,
        )
        self.assertEqual(manifest_path.read_bytes(), manifest_raw)
        self.assertEqual(sidecar_path.read_bytes(), sidecar_raw)
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(sidecar_path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_private_pair_exclusive(
                manifest_path=manifest_path, manifest_raw=manifest_raw,
                sidecar_path=sidecar_path, sidecar_raw=sidecar_raw, repo_root=repo,
            )

        failing_repo = Path(f["temp"].name) / "rollback_repo"
        (failing_repo / "data").mkdir(parents=True)
        failing_private = failing_repo / "data" / "private_nextgqa"
        failing_private.mkdir(mode=0o700)
        path_a, path_b = failing_private / "a.json", failing_private / "b.json"
        import build_nextgqa_v2_claim_input_bundle as builder_module
        real_link = builder_module.os.link
        call_count = 0

        def fail_second_link(src, dst, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("synthetic second-link failure")
            return real_link(src, dst, **kwargs)

        with patch.object(builder_module.os, "link", side_effect=fail_second_link):
            with self.assertRaisesRegex(OSError, "synthetic second-link failure"):
                write_private_pair_exclusive(
                    manifest_path=path_a, manifest_raw=manifest_raw,
                    sidecar_path=path_b, sidecar_raw=sidecar_raw, repo_root=failing_repo,
                )
        self.assertFalse(path_a.exists())
        self.assertFalse(path_b.exists())

    def test_cli_synthetic_inputs_and_no_overwrite(self):
        f = fixture(self)
        root = Path(f["temp"].name) / "repo"
        root.mkdir()
        (root / "data").mkdir()
        input_dir = Path(f["temp"].name) / "inputs"
        input_dir.mkdir()
        named = {
            "eligible": f["eligible_raw"], "generation": f["generation_raw"],
            "lineage": f["lineage_raw"], "policy": f["policy_raw"],
            "test": f["test_csv_raw"], "map": f["video_map_raw"],
        }
        input_paths = {}
        for name, raw in named.items():
            path = input_dir / f"{name}.input"
            path.write_bytes(raw)
            input_paths[name] = str(path)
        private_dir = root / "data" / "private_nextgqa"
        args = [
            "--eligible-manifest", input_paths["eligible"],
            "--generation-packet", input_paths["generation"],
            "--lineage-manifest", input_paths["lineage"],
            "--policy", input_paths["policy"], "--test-csv", input_paths["test"],
            "--video-map", input_paths["map"], "--video-root", str(f["video_root"]),
            "--dataset-id", f["dataset_id"],
            "--manifest-output", str(private_dir / "manifest.json"),
            "--sidecar-output", str(private_dir / "sidecar.json"),
            "--repo-root", str(root), "--user-authorized-local-only",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        sidecar = json.loads((private_dir / "sidecar.json").read_bytes())
        self.assertEqual(sidecar["status"], "candidate_unreviewed")
        self.assertIn("execution_authorization=not_granted", stdout.getvalue())
        self.assertNotIn("Synthetic question", stdout.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)

    def test_mutated_policy_or_wrong_cohort_size_fails_closed(self):
        f = fixture(self)
        f["policy_raw"] = f["policy_raw"].replace(b"all_relevant_occurrences",
                                                  b"one_sufficient_occurrence", 1)
        with self.assertRaises(ValueError):
            build(f)
        for mode in ("eligible", "generation", "lineage"):
            with self.subTest(mode=mode):
                f = fixture(self)
                if mode == "eligible":
                    f["eligible"] = f["eligible"][:2]
                    f["eligible_raw"] = encode(f["eligible"])
                elif mode == "generation":
                    f["generation"]["records"] = f["generation"]["records"][:2]
                    f["generation"]["effective_videos"] = 2
                    f["generation"]["fully_structured_claim_records"] = 2
                    f["generation_raw"] = encode(f["generation"])
                else:
                    f["lineage"]["records"] = f["lineage"]["records"][:2]
                    f["lineage"]["effective_records"] = 2
                    f["lineage"]["source_packet_sha256"] = sha(f["generation_raw"])
                    f["lineage_raw"] = encode(f["lineage"])
                with self.assertRaises(ValueError):
                    build(f)


if __name__ == "__main__":
    unittest.main()
