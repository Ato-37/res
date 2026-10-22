from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from summarize_minerva_visible_object_v2 import build_summary


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_hash(data: bytes):
    return _sha(data), hashlib.md5(data, usedforsecurity=False).hexdigest()


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.private = self.root / "data" / "private_minerva"
        self.private.mkdir(parents=True)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        source_names = (
            "run_minerva_generated_object_pixel_pilot.py",
            "preflight_minerva_grounding_cohort.py",
            "run_minerva_ego_anchor_gpu_pilot.py",
            "run_minerva_ego_generated_grounding_pilot.py",
            "minerva_ego_anchor_utils.py")
        self.source_hashes = {}
        for rank, name in enumerate(source_names):
            code = f"synthetic source {rank}\n".encode()
            (self.scripts / name).write_bytes(code)
            self.source_hashes[name] = _sha(code)
        self.cohort, self.cohort_raw, self.final, self.final_raw = self._make_cohort()
        self.entries, self.items_raw = self._make_entries()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_private(self, name, payload):
        path = self.private / name
        raw = payload if isinstance(payload, bytes) else json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()
        path.write_bytes(raw)
        os.chmod(path, 0o600)
        return path, raw

    def _make_cohort(self):
        selection, records = [], []
        for index in range(3):
            video_id = f"P0{index + 1}-private-video"
            row_key = f"private-row-key-{index}"
            row_hash = _sha(row_key.encode())
            video = f"video-{index}-bytes".encode()
            time_map = f"pts-{index}-bytes".encode()
            video_sha, video_md5 = _media_hash(video)
            time_sha, time_md5 = _media_hash(time_map)
            selection.append({"video_id": video_id, "row_key": row_key,
                "row_key_sha256": row_hash})
            records.append({"video_id": video_id, "row_key_sha256": row_hash,
                "status": "pass", "video_sha256": video_sha,
                "video_md5": video_md5, "time_map_sha256": time_sha,
                "time_map_md5": time_md5})
            (self.private / f"video{index}.mp4").write_bytes(video)
            (self.private / f"map{index}.csv").write_bytes(time_map)
        cohort = {"status": "frozen_pre_outcome_exploratory_cohort",
            "requested_rows": 3, "selection": selection,
            "source_sha256": {"minerva_json": "a" * 64,
                "assoc_info": "c" * 64, "mask_info": "e" * 64}}
        cohort_raw = json.dumps(cohort, sort_keys=True).encode()
        source_fields = {"preflight_source_sha256": self.source_hashes[
                "preflight_minerva_grounding_cohort.py"],
            "decoder_source_sha256": self.source_hashes["run_minerva_ego_anchor_gpu_pilot.py"],
            "generated_runner_source_sha256": self.source_hashes[
                "run_minerva_ego_generated_grounding_pilot.py"],
            "bbox_mapper_source_sha256": self.source_hashes["minerva_ego_anchor_utils.py"],
            "qwen_video_processor": {"name": "synthetic"},
            "human_review_performed": False, "rights_status": "local_only"}
        final = {"status": "completed_frozen_cohort_cpu_preflight",
            "requested_rows": 3, "effective_rows": 3,
            "cohort_sha256": _sha(cohort_raw), "manifest_sha256": "m" * 64,
            "records": records, **source_fields}
        final_raw = json.dumps(final, sort_keys=True).encode()
        return cohort, cohort_raw, final, final_raw

    def _make_entries(self):
        items, preflights = [], []
        final_sha = _sha(self.final_raw)
        annotation_sha = {"minerva_json": "a" * 64,
            "assoc_info": "c" * 64, "mask_info": "e" * 64}
        for index, selected in enumerate(self.cohort["selection"]):
            preflight_sha = final_sha
            if index == 0:
                partial = dict(self.final)
                partial.update(requested_rows=1, effective_rows=1,
                    records=[self.final["records"][0]])
                partial_raw = json.dumps(partial, sort_keys=True).encode()
                _, partial_raw = self._write_private("partial.json", partial_raw)
                preflight_sha = _sha(partial_raw)
            preflights.append(preflight_sha)
            video_rel, map_rel = f"data/private_minerva/video{index}.mp4", f"data/private_minerva/map{index}.csv"
            video = (self.root / video_rel).read_bytes()
            time_map = (self.root / map_rel).read_bytes()
            video_sha, video_md5 = _media_hash(video)
            time_sha, time_md5 = _media_hash(time_map)
            row_key_hash = selected["row_key_sha256"]
            source = self.source_hashes
            cpu = {"status": "completed_minerva_generated_object_pixel_cpu_preflight",
                "cohort_sha256": _sha(self.cohort_raw),
                "cohort_cpu_preflight_sha256": preflight_sha,
                "row_index": index, "row_key_sha256": row_key_hash,
                "video_id_sha256": _sha(selected["video_id"].encode()),
                "video_sha256": video_sha, "video_md5": video_md5,
                "time_map_sha256": time_sha, "time_map_md5": time_md5,
                "manifest_sha256": self.final["manifest_sha256"],
                "annotation_source_sha256": annotation_sha,
                "source_sha256": source,
                "runner_sha256": source["run_minerva_generated_object_pixel_pilot.py"],
                "pixel_tensor_sha256": f"{index + 1}" * 64,
                "condition_key_counts": {"bbox": 25 - index,
                    "spatial_sham": 25 - index}}
            cpu_path, cpu_raw = self._write_private(f"cpu{index}.json", cpu)
            if index < 2:
                ids = [100 + index, 200 + index]
                token_hash = _sha(json.dumps(ids, separators=(",", ":")).encode())
                scores = {
                    "support_none": -0.4 - index * 0.02,
                    "support_bbox_bias": -0.41 - index * 0.02,
                    "support_spatial_sham_bias": -0.405 - index * 0.02,
                    "sham_none": -0.38 + index * 0.01,
                    "sham_bbox_bias": -0.39 + index * 0.01,
                    "sham_spatial_sham_bias": -0.40 + index * 0.01}
                d = {k: scores[f"sham_{k}"] - scores[f"support_{k}"]
                     for k in ("none", "bbox_bias", "spatial_sham_bias")}
                contrasts = {"D_no_bias_nats_per_token": d["none"],
                    "D_bbox_bias_nats_per_token": d["bbox_bias"],
                    "D_spatial_sham_bias_nats_per_token": d["spatial_sham_bias"],
                    "primary_bbox_minus_spatial_sham_interaction_nats_per_token": d["bbox_bias"] - d["spatial_sham_bias"],
                    "secondary_bbox_minus_no_bias_interaction_nats_per_token": d["bbox_bias"] - d["none"]}
                audits = {}
                for fill in ("support", "sham"):
                    for condition in ("none", "bbox_bias", "spatial_sham_bias"):
                        branch = f"{fill}_{condition}"
                        audits[branch] = {"target_token_ids_sha256": token_hash,
                            "target_token_count": 2, "target_query_count": 2,
                            "biased_key_count": 0 if condition == "none" else 25 - index,
                            "deleted_key_count": 0, "logit_bias_nats": math.log(2),
                            "intervention": "none" if condition == "none"
                                else "QTA visual-key logit bias",
                            "layer_hook_calls": {str(layer): 1 for layer in range(5, 17)}}
                gpu = {"status": "completed_single_video_generated_object_pixel_exploratory_pilot",
                    "direct_generated_explanation_faithfulness": False,
                    "human_review_performed": False,
                    "cohort_sha256": _sha(self.cohort_raw),
                    "cohort_preflight_sha256": preflight_sha,
                    "cpu_pixel_receipt_sha256": _sha(cpu_raw),
                    "row_index": index, "video_id_sha256": _sha(selected["video_id"].encode()),
                    "row_key_sha256": row_key_hash, "video_sha256": video_sha,
                    "video_md5": video_md5, "time_map_sha256": time_sha,
                    "manifest_sha256": self.final["manifest_sha256"],
                    "annotation_source_sha256": annotation_sha,
                    "runner_sha256": source["run_minerva_generated_object_pixel_pilot.py"],
                    "model_revision": "synthetic-model-revision",
                    "source_sha256": source,
                    "pixel_tensor_sha256": cpu["pixel_tensor_sha256"],
                    "condition_key_counts": cpu["condition_key_counts"],
                    "generation": {"generated_text": "synthetic private claim",
                        "generated_text_sha256": _sha(b"synthetic private claim"),
                        "generated_token_ids": ids,
                        "generated_token_ids_sha256": token_hash,
                        "target_token_count": 2,
                        "branch_mean_logprob_nats_per_token": scores,
                        "branch_audit": audits, "contrasts": contrasts,
                        "model_call_count": {"generation_calls": 1,
                            "teacher_forced_score_calls": 6,
                            "completed_teacher_forced_score_calls": 6}}}
            else:
                gpu = {"status": "failed_single_video_generated_object_pixel_exploratory_pilot",
                    "direct_generated_explanation_faithfulness": False,
                    "human_review_performed": False,
                    "cohort_sha256": _sha(self.cohort_raw),
                    "cohort_preflight_sha256": preflight_sha,
                    "cpu_pixel_receipt_sha256": _sha(cpu_raw),
                    "row_index": index, "row_key_sha256": row_key_hash,
                    "video_sha256": video_sha,
                    "runner_sha256": source["run_minerva_generated_object_pixel_pilot.py"],
                    "failure_code": "generated_target_not_short_single_line",
                    "progress": {"generation_attempted": True,
                        "generation_completed": True,
                        "teacher_forced_score_attempted": 0,
                        "teacher_forced_score_completed": 0}}
            gpu_path, _ = self._write_private(f"gpu{index}.json", gpu)
            entry = {"row_index": index, "status": "review" if index < 2 else "gpu_failed",
                "cpu_receipt": str(cpu_path.relative_to(self.root)),
                "video": video_rel, "time_map": map_rel}
            entry["gpu_receipt" if index < 2 else "gpu_failure_receipt"] = str(
                gpu_path.relative_to(self.root))
            if index == 0:
                entry["cohort_preflight"] = "data/private_minerva/partial.json"
            items.append(entry)
        manifest = {"schema_version": 1, "items": items}
        items_raw = json.dumps(manifest, sort_keys=True).encode()
        return items, items_raw

    def _build(self, *, cohort_raw=None, final_raw=None, items_raw=None):
        return build_summary(cohort_raw or self.cohort_raw,
            final_raw or self.final_raw, items_raw or self.items_raw,
            root=self.root, private_root=self.private, script_root=self.scripts)

    def test_recomputes_two_row_contrasts_ci_and_retains_attrition_privately(self):
        result = self._build()
        self.assertEqual(result["requested_rows"], 3)
        self.assertEqual(result["scored_rows"], 2)
        self.assertEqual(result["structural_attrition_rows"], 1)
        self.assertEqual(result["rows"][2]["row_index"], 2)
        self.assertEqual(result["rows"][2]["score_forward_count"], 0)
        self.assertEqual(result["rows"][0]["branch_count"], 6)
        self.assertEqual(result["rows"][0]["hook_calls_total"], 72)
        self.assertEqual(result["rows"][0]["cohort_preflight_sha256"],
            _sha((self.private / "partial.json").read_bytes()))
        self.assertEqual(result["primary_interaction_summary_nats_per_token"]["n"], 2)
        self.assertEqual(result["primary_interaction_summary_nats_per_token"]["df"], 1)
        self.assertAlmostEqual(result["primary_interaction_summary_nats_per_token"]["mean"],
            sum(r["primary_interaction_nats_per_token"] for r in result["rows"][:2]) / 2)

    def test_summary_omits_video_ids_and_claim_text(self):
        result = self._build()
        rendered = json.dumps(result)
        self.assertNotIn("P01-private-video", rendered)
        self.assertNotIn("P02-private-video", rendered)
        self.assertNotIn("synthetic private claim", rendered)
        self.assertNotIn(str(self.root), rendered)
        for row in result["rows"]:
            self.assertNotIn("video_sha256", row)
            self.assertNotIn("time_map_sha256", row)
            self.assertNotIn("row_key_sha256", row)

    def test_rejects_alternate_preflight_record_or_source_tampering(self):
        partial_path = self.private / "partial.json"
        partial = json.loads(partial_path.read_bytes())
        partial["records"][0]["time_map_sha256"] = "0" * 64
        partial_path.write_text(json.dumps(partial))
        os.chmod(partial_path, 0o600)
        with self.assertRaisesRegex(ValueError, "row_preflight_record_mismatch"):
            self._build()

    def test_rejects_receipt_hash_mismatch_and_hook_change(self):
        manifest = json.loads(self.items_raw)
        cpu_path = self.root / manifest["items"][0]["cpu_receipt"]
        cpu = json.loads(cpu_path.read_bytes())
        cpu["pixel_tensor_sha256"] = "0" * 64
        cpu_path.write_text(json.dumps(cpu))
        os.chmod(cpu_path, 0o600)
        with self.assertRaisesRegex(ValueError, "gpu_cpu_receipt_binding"):
            self._build()

        self.tearDown()
        self.setUp()
        manifest = json.loads(self.items_raw)
        gpu_path = self.root / manifest["items"][0]["gpu_receipt"]
        gpu = json.loads(gpu_path.read_bytes())
        gpu["generation"]["branch_audit"]["support_none"]["layer_hook_calls"]["5"] = 0
        gpu_path.write_text(json.dumps(gpu))
        os.chmod(gpu_path, 0o600)
        with self.assertRaisesRegex(ValueError, "branch_target_dose_or_hook"):
            self._build()

    def test_rejects_wrong_structural_failure_code(self):
        manifest = json.loads(self.items_raw)
        gpu_path = self.root / manifest["items"][2]["gpu_failure_receipt"]
        gpu = json.loads(gpu_path.read_bytes())
        gpu["failure_code"] = "different_failure"
        gpu_path.write_text(json.dumps(gpu))
        os.chmod(gpu_path, 0o600)
        with self.assertRaisesRegex(ValueError, "row_two_structural_attrition"):
            self._build()

    def test_rejects_attrition_row_annotation_source_drift(self):
        manifest = json.loads(self.items_raw)
        cpu_path = self.root / manifest["items"][2]["cpu_receipt"]
        cpu = json.loads(cpu_path.read_bytes())
        cpu["annotation_source_sha256"]["mask_info"] = "0" * 64
        _, new_cpu_raw = self._write_private(cpu_path.name, cpu)
        gpu_path = self.root / manifest["items"][2]["gpu_failure_receipt"]
        gpu = json.loads(gpu_path.read_bytes())
        gpu["cpu_pixel_receipt_sha256"] = _sha(new_cpu_raw)
        self._write_private(gpu_path.name, gpu)
        with self.assertRaisesRegex(ValueError,
                                    "cpu_annotation_source_cohort_binding_invalid"):
            self._build()


if __name__ == "__main__":
    unittest.main()
