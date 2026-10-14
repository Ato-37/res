import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path

from build_claim_input_manifest import build, main
from audit_claim_review_evidence_index import audit as audit_claim_review
from test_audit_claim_review_evidence_index import fixture as claim_review_fixture, raw as review_raw

H = "a" * 64
REV = "Qwen/Qwen2.5-VL-7B-Instruct@" + "b" * 40
RUN = "c" * 64


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def fixtures():
    eligible = [{"video_id": "synthetic-a"}, {"video_id": "synthetic-b"}]
    generation = {"schema_version": 1, "claims": [
        {"video_id": "synthetic-b", "claim_id": "claim-b", "claim_text": "A second synthetic claim.",
         "model_revision": REV, "generation_run_sha256": RUN},
        {"video_id": "synthetic-a", "claim_id": "claim-a", "claim_text": "Café, synthetic.",
         "model_revision": REV, "generation_run_sha256": RUN},
    ]}
    return eligible, raw(eligible), generation


class BuildClaimInputManifestTests(unittest.TestCase):
    def test_manifest_is_complete_ordered_and_hashes_exact_utf8_text(self):
        eligible, eligible_raw, generation = fixtures()
        manifest, output_raw = build(eligible, eligible_raw, generation, "synthetic-dataset", H,
                                     "one_sufficient_occurrence")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["eligible_manifest_sha256"], hashlib.sha256(eligible_raw).hexdigest())
        self.assertEqual([row["video_id"] for row in manifest["claims"]], ["synthetic-a", "synthetic-b"])
        self.assertEqual(manifest["claims"][0]["claim_text_sha256"],
                         hashlib.sha256("Café, synthetic.".encode("utf-8")).hexdigest())
        self.assertEqual(json.loads(output_raw), manifest)

    def test_incomplete_duplicate_or_out_of_cohort_generation_fails(self):
        eligible, eligible_raw, generation = fixtures()
        for rows, code in (
            (generation["claims"][:1], "generation_rows_missing"),
            (generation["claims"] + [generation["claims"][0]], "generation_video_id_duplicate"),
            (generation["claims"] + [{**generation["claims"][0], "video_id": "outside"}],
             "generation_rows_outside_eligible_cohort"),
        ):
            with self.subTest(code=code):
                candidate = {"schema_version": 1, "claims": rows}
                with self.assertRaisesRegex(ValueError, code):
                    build(eligible, eligible_raw, candidate, "synthetic-dataset", H,
                          "one_sufficient_occurrence")

    def test_mixed_model_revisions_and_unpaired_surrogates_fail(self):
        eligible, eligible_raw, generation = fixtures()
        mixed = json.loads(json.dumps(generation))
        mixed["claims"][0]["model_revision"] = "Qwen/model@" + "d" * 40
        with self.assertRaisesRegex(ValueError, "generation_model_revision_mixed"):
            build(eligible, eligible_raw, mixed, "synthetic-dataset", H, "one_sufficient_occurrence")
        invalid_unicode = json.loads(json.dumps(generation))
        invalid_unicode["claims"][0]["claim_text"] = "bad\ud800text"
        with self.assertRaisesRegex(ValueError, "claim_text_invalid_unicode"):
            build(eligible, eligible_raw, invalid_unicode, "synthetic-dataset", H,
                  "one_sufficient_occurrence")

    def test_direct_api_rejects_detached_eligible_object(self):
        eligible, eligible_raw, generation = fixtures()
        detached = [{"video_id": "other-a"}, {"video_id": "other-b"}]
        with self.assertRaisesRegex(ValueError, "eligible_manifest_bytes_invalid"):
            build(detached, eligible_raw, generation, "synthetic-dataset", H,
                  "one_sufficient_occurrence")

    def test_direct_api_rejects_unhashable_occurrence_policy(self):
        eligible, eligible_raw, generation = fixtures()
        with self.assertRaisesRegex(ValueError, "occurrence_policy_invalid"):
            build(eligible, eligible_raw, generation, "synthetic-dataset", H, [])

    def test_built_manifest_is_accepted_by_claim_review_hash_checker(self):
        parts = list(claim_review_fixture())
        protocol, eligible, eligible_raw, _claim_input, _input_raw, ledger, ledger_raw, review, review_bytes = parts
        generation = {"schema_version": 1, "claims": [
            {key: row[key] for key in ("video_id", "claim_id", "claim_text", "model_revision", "generation_run_sha256")}
            for row in _claim_input["claims"]
        ]}
        manifest, manifest_raw = build(
            eligible, eligible_raw, generation, _claim_input["eligible_dataset_id"],
            _claim_input["claim_policy_sha256"], _claim_input["occurrence_policy"],
        )
        protocol = dict(protocol)
        protocol["claim_review"] = dict(protocol["claim_review"])
        protocol["claim_review"]["claim_manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
        report = audit_claim_review(
            protocol, review_raw(protocol), eligible, eligible_raw, manifest, manifest_raw,
            ledger, ledger_raw, review, review_bytes,
        )
        self.assertEqual(report["structural_status"], "passed")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_cli_creates_private_artifact_without_echoing_claim_or_ids(self):
        eligible, eligible_raw, generation = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            eligible_path = Path(tmp) / "eligible.json"
            generation_path = Path(tmp) / "generation.json"
            output_path = Path(tmp) / "claim-input.json"
            eligible_path.write_bytes(eligible_raw)
            generation_path.write_bytes(raw(generation))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main([
                    "--eligible-manifest", str(eligible_path),
                    "--generation-export", str(generation_path),
                    "--eligible-dataset-id", "synthetic-dataset",
                    "--claim-policy-sha256", H,
                    "--occurrence-policy", "one_sufficient_occurrence",
                    "--output", str(output_path),
                ])
            self.assertEqual(status, 0)
            self.assertTrue(output_path.exists())
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertIn('"execution_authorization": "not_granted"', output.getvalue())
            self.assertNotIn("Café", output.getvalue())
            self.assertNotIn("synthetic-a", output.getvalue())
            self.assertEqual(json.loads(output_path.read_bytes())["schema_version"], 1)
            with contextlib.redirect_stdout(output):
                second_status = main([
                    "--eligible-manifest", str(eligible_path),
                    "--generation-export", str(generation_path),
                    "--eligible-dataset-id", "synthetic-dataset",
                    "--claim-policy-sha256", H,
                    "--occurrence-policy", "one_sufficient_occurrence",
                    "--output", str(output_path),
                ])
            self.assertEqual(second_status, 1)
            self.assertEqual(json.loads(output_path.read_bytes())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
