import copy
import hashlib
import json
from pathlib import Path
import unittest

from preflight_nextgqa_v2_synthetic import (
    FIXTURE_TYPE,
    PROTOCOL_V2_SHA256,
    REQUIRED_HASH_NAMES,
    preflight_synthetic_v2,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    protocol = (Path(__file__).resolve().parents[1] / "docs"
                / "nextgqa_finite_claim_descriptive_protocol_v2.md").read_bytes()
    pts = list(range(32))
    time_base = [1, 1]
    pts_payload = json.dumps([[value, 1, 1] for value in pts],
                             separators=(",", ":")).encode("ascii")
    sample_indices = list(range(32))
    rows = []
    for index in range(3):
        media = f"synthetic media {index}".encode()
        decoded_pixels = [sha(f"frame {index} {frame}".encode()) for frame in pts]
        pixel_payload = json.dumps(decoded_pixels, separators=(",", ":")).encode("ascii")
        sampled_pixels = [decoded_pixels[frame] for frame in sample_indices]
        rows.append({
            "video_id": f"synthetic-video-{index}",
            "source_media_bytes": media,
            "source_media_sha256": sha(media),
            "pts": pts.copy(),
            "time_base": time_base.copy(),
            "num_frames": len(pts),
            "decoded_pts_sequence_sha256": sha(pts_payload),
            "decoded_rgb_pixel_sha256": decoded_pixels,
            "decoded_rgb_sequence_sha256": sha(pixel_payload),
            "sample_indices": sample_indices.copy(),
            "sampled_pts": [pts[frame] for frame in sample_indices],
            "sampled_pixel_sha256": sampled_pixels,
            "processor_pixel_sha256": sampled_pixels.copy(),
            "evidence_intervals": [[4.8, 5.2]],
            "clock_slope": [1, 1],
            "clock_intercept_seconds": [0, 1],
            "clock_guard_band_seconds": [1, 100],
        })
    clock_config = json.dumps({
        "rule": "mapped_interval_boundary_guard_inclusive_v1",
        "rows": [{
            "video_id": row["video_id"], "slope": row["clock_slope"],
            "intercept_seconds": row["clock_intercept_seconds"],
            "guard_band_seconds": row["clock_guard_band_seconds"],
        } for row in rows],
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    contents = {name: f"synthetic {name}".encode() for name in REQUIRED_HASH_NAMES}
    contents["clock_mapping_config"] = clock_config
    declarations = {name: sha(raw) for name, raw in contents.items()}
    return {
        "fixture_type": FIXTURE_TYPE,
        "synthetic_fixture_attestation": {
            "status": "synthetic_only", "generator": "synthetic_fixture_encoder",
            "no_real_media": True,
        },
        "protocol_raw": protocol,
        "expected_video_ids": [row["video_id"] for row in rows],
        "rows": rows,
        "code_config_contents": contents,
        "declared_code_config_sha256": declarations,
    }


def run(f):
    return preflight_synthetic_v2(
        fixture_type=f["fixture_type"], protocol_raw=f["protocol_raw"],
        expected_video_ids=f["expected_video_ids"], rows=f["rows"],
        code_config_contents=f["code_config_contents"],
        declared_code_config_sha256=f["declared_code_config_sha256"],
        synthetic_fixture_attestation=f["synthetic_fixture_attestation"],
    )


def rebind_clock_config(f):
    material = [{
        "video_id": row["video_id"], "slope": row["clock_slope"],
        "intercept_seconds": row["clock_intercept_seconds"],
        "guard_band_seconds": row["clock_guard_band_seconds"],
    } for row in f["rows"]]
    raw = json.dumps({"rule": "mapped_interval_boundary_guard_inclusive_v1",
                      "rows": material}, sort_keys=True,
                     separators=(",", ":")).encode("ascii")
    f["code_config_contents"]["clock_mapping_config"] = raw
    f["declared_code_config_sha256"]["clock_mapping_config"] = sha(raw)


class SyntheticV2PreflightTests(unittest.TestCase):
    def assert_denied(self, result, expected_code):
        self.assertIn(expected_code, result["failure_codes"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertTrue(result["synthetic_only"])
        self.assertFalse(result["real_media_eligible"])
        self.assertEqual(result["scope"], "synthetic_structure_only")
        self.assertNotIn("synthetic_fixture_declaration_matched", result)
        self.assertFalse(result["real_media_eligible"])
        self.assertNotIn("ready", result["status"])

    def test_three_row_32_frame_k_positive_synthetic_contract_passes_unvalidated(self):
        result = run(fixture())
        self.assertEqual(result["status"], "synthetic_structural_preflight_passed_unvalidated")
        self.assertEqual(result["protocol_sha256"], PROTOCOL_V2_SHA256)
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["structurally_checked_records"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["protocol_status"], "design_draft_not_frozen")
        self.assertEqual(result["scope"], "synthetic_structure_only")
        self.assertFalse(result["real_media_eligible"])
        self.assertTrue(result["synthetic_fixture_declaration_matched"])
        self.assertEqual(result["synthetic_fixture_declaration_authentication"],
                         "caller_supplied_not_authenticated")
        self.assertNotIn("source_media_bytes", result)
        self.assertNotIn("ready", result["status"])

    def test_wrong_protocol_identity_fails_closed(self):
        f = fixture()
        f["protocol_raw"] = f["protocol_raw"].replace(b"protocol v2", b"protocol v1", 1)
        self.assert_denied(run(f), "protocol_v2_identity_mismatch")

    def test_missing_or_malformed_synthetic_attestation_fails_without_matched_flag(self):
        for attestation in (None, {
                "status": "synthetic_only", "generator": "synthetic_fixture_encoder",
                "no_real_media": 1}):
            with self.subTest(attestation=attestation):
                f = fixture()
                f["synthetic_fixture_attestation"] = attestation
                result = run(f)
                self.assert_denied(result, "synthetic_fixture_attestation_invalid")
                self.assertNotIn("synthetic_fixture_declaration_matched", result)

    def test_missing_or_extra_cohort_row_fails_closed_at_denominator_three(self):
        f = fixture()
        f["rows"].pop()
        self.assert_denied(run(f), "cohort_row_count_mismatch")

    def test_non_32_or_non_endpoint_sample_grid_fails_closed(self):
        f = fixture()
        f["rows"][1]["sample_indices"] = list(range(1, 32))
        f["rows"][1]["sampled_pts"] = list(range(1, 32))
        f["rows"][1]["sampled_pixel_sha256"] = f["rows"][1][
            "decoded_rgb_pixel_sha256"][1:32]
        f["rows"][1]["processor_pixel_sha256"] = f["rows"][1]["sampled_pixel_sha256"].copy()
        self.assert_denied(run(f), "sample_indices_not_exact_32_frame_grid")

    def test_zero_evidence_overlap_on_any_row_blocks_whole_cohort(self):
        for row_index in range(3):
            with self.subTest(row_index=row_index):
                f = fixture()
                f["rows"][row_index]["evidence_intervals"] = [[100, 101]]
                rebind_clock_config(f)
                self.assert_denied(run(f), "zero_sampled_evidence_overlap")

    def test_insufficient_equal_count_controls_blocks_whole_cohort(self):
        f = fixture()
        f["rows"][0]["evidence_intervals"] = [[0.5, 30.5]]
        rebind_clock_config(f)
        self.assert_denied(run(f), "insufficient_equal_count_controls")

    def test_pts_source_media_and_pixel_hash_incoherence_fail_closed(self):
        mutations = (
            ("source_media_bytes", b"different media", "source_media_digest_invalid"),
            ("decoded_pts_sequence_sha256", "0" * 64, "source_pts_digest_mismatch"),
            ("decoded_rgb_sequence_sha256", "0" * 64,
             "decoded_pixel_map_digest_mismatch"),
            ("processor_pixel_sha256", ["0" * 64] * 32,
             "processor_pixel_identity_mismatch"),
        )
        for key, value, code in mutations:
            with self.subTest(key=key):
                f = fixture()
                f["rows"][1][key] = copy.deepcopy(value)
                self.assert_denied(run(f), code)

    def test_missing_or_mismatched_code_config_hash_fails_closed(self):
        missing = fixture()
        missing["declared_code_config_sha256"].pop("frame_sampler")
        self.assert_denied(run(missing), "code_config_hash_set_invalid")
        mismatch = fixture()
        mismatch["code_config_contents"]["frame_sampler"] = b"modified synthetic sampler"
        self.assert_denied(run(mismatch), "code_config_hash_set_invalid")

    def test_guard_band_classification_change_blocks_and_is_hash_bound(self):
        f = fixture()
        f["rows"][1]["evidence_intervals"] = [[4.99, 5.2]]
        rebind_clock_config(f)
        self.assert_denied(run(f), "sample_classification_uncertain_at_boundary")

        inconsistent = fixture()
        inconsistent["rows"][1]["clock_guard_band_seconds"] = [1, 10]
        self.assert_denied(run(inconsistent), "clock_mapping_config_content_mismatch")


if __name__ == "__main__":
    unittest.main()
