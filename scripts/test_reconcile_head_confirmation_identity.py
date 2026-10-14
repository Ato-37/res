import unittest

from reconcile_head_confirmation_identity import reconcile

H = "0" * 64
G = "0" * 40


def fixtures():
    result = {
        "experiment": "gradient_activation_head_faithfulness",
        "n_requested": 2, "n_effective": 2, "n_failures": 0,
        "records": [{"sample_key": "sample_000"}, {"sample_key": "sample_001"}],
    }
    manifest = {
        "experiment": result["experiment"], "n_requested": 2,
        "n_effective": 2, "n_failures": 0, "samples_sha256": H,
    }
    identity = {
        "status": "complete", "input_manifest_sha256": H,
        "requested_sample_records": 2, "effective_sample_records_with_hashed_video": 2,
        "unique_nonempty_normalized_video_ids": 2,
        "effective_unique_nonempty_normalized_video_ids": 2,
        "duplicate_video_id_excess_records": 0,
        "effective_duplicate_video_id_excess_records": 0,
        "unique_resolved_video_paths": 2, "files_hashed_successfully": 2,
        "missing_video_path_records": 0, "missing_files": 0, "read_failed_files": 0,
        "duplicate_resolved_path_excess_records": 0, "duplicate_content_excess_files": 0,
        "source_commit": G, "runner_sha256": H,
    }
    return result, manifest, identity


def run(result, manifest, identity, **updates):
    args = {"result_sha256": H, "manifest_sha256": H, "identity_sha256": H,
            "audited_runner_sha256": H, "source_runner_sha256": H,
            "source_commit_available": True}
    args.update(updates)
    return reconcile(result, manifest, identity, **args)


class TestHeadConfirmationIdentityReconciliation(unittest.TestCase):
    def test_same_hash_complete_unique_audit_and_complete_ordinals_pass(self):
        result, manifest, identity = fixtures()
        report = run(result, manifest, identity)
        self.assertEqual(report["status"], "cohort_identity_verified")
        self.assertTrue(report["manifest_hash_matches_identity_audit"])
        self.assertTrue(report["result_sample_ordinals_complete"])
        self.assertFalse(report["result_run_manifest_crosshash_binding"])
        self.assertFalse(report["independence_established"])

    def test_manifest_hash_mismatch_fails(self):
        result, manifest, identity = fixtures(); identity["input_manifest_sha256"] = "1" * 64
        report = run(result, manifest, identity)
        self.assertIn("identity_audit_manifest_hash_mismatch", report["failure_codes"])

    def test_duplicate_or_missing_result_ordinal_fails(self):
        result, manifest, identity = fixtures(); result["records"][1]["sample_key"] = "sample_000"
        report = run(result, manifest, identity)
        self.assertIn("result_sample_ordinals_incomplete_or_duplicate", report["failure_codes"])
        result, manifest, identity = fixtures(); result["records"][1]["sample_key"] = 7
        report = run(result, manifest, identity)
        self.assertIn("result_sample_ordinals_incomplete_or_duplicate", report["failure_codes"])

    def test_duplicate_ids_or_content_fail(self):
        result, manifest, identity = fixtures(); identity["duplicate_video_id_excess_records"] = 1
        report = run(result, manifest, identity)
        self.assertIn("identity_audit_counts_incomplete_or_nonunique", report["failure_codes"])
        result, manifest, identity = fixtures(); identity["duplicate_content_excess_files"] = 1
        report = run(result, manifest, identity)
        self.assertIn("identity_audit_counts_incomplete_or_nonunique", report["failure_codes"])

    def test_boolean_counts_do_not_pass_as_integer_counts(self):
        result, manifest, identity = fixtures(); identity["requested_sample_records"] = True
        report = run(result, manifest, identity)
        self.assertIn("identity_audit_counts_incomplete_or_nonunique", report["failure_codes"])
        result, manifest, identity = fixtures(); manifest["n_effective"] = True
        report = run(result, manifest, identity)
        self.assertIn("run_manifest_result_mismatch", report["failure_codes"])

    def test_runner_source_mismatch_fails(self):
        result, manifest, identity = fixtures()
        report = run(result, manifest, identity, source_runner_sha256="1" * 64)
        self.assertIn("identity_audit_runner_provenance_unverified", report["failure_codes"])
        self.assertFalse(report["identity_audit_runner_source_match"])
        report = run(result, manifest, identity, audited_runner_sha256="",
                     source_runner_sha256="")
        self.assertIn("identity_audit_runner_provenance_unverified", report["failure_codes"])
        self.assertFalse(report["identity_audit_runner_source_match"])

    def test_missing_accounting_does_not_claim_complete_ordinals(self):
        result, manifest, identity = fixtures(); result["records"] = None
        report = run(result, manifest, identity)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["result_sample_ordinals_complete"])

    def test_result_manifest_crosshash_is_reported_separately(self):
        result, manifest, identity = fixtures(); manifest["result_sha256"] = "1" * 64
        report = run(result, manifest, identity)
        self.assertEqual(report["status"], "cohort_identity_verified")
        self.assertFalse(report["result_run_manifest_crosshash_binding"])

    def test_no_identity_values_are_copied_into_summary(self):
        result, manifest, identity = fixtures()
        identity["video_id"] = "private identifier"
        report = run(result, manifest, identity)
        self.assertNotIn("video_id", report)
        self.assertNotIn("private identifier", str(report))


if __name__ == "__main__":
    unittest.main()
