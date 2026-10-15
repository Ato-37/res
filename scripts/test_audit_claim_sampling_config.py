import hashlib
import json
import unittest

from audit_claim_sampling_config import audit


class SamplingConfigAuditTests(unittest.TestCase):
    def fixture(self):
        config = {
            "policy": "whole_video_fixed",
            "uses_claim_for_selection": False,
            "num_frames": 16,
            "frame_size": 1024,
            "max_pixels": 200704,
            "cap_pixels_per_frame": True,
        }
        raw = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
        protocol = {"schema_version": 13, "sampling": {
            **config, "config_sha256": hashlib.sha256(raw).hexdigest(),
        }}
        return protocol, config, raw

    def test_exact_bytes_and_sampling_values_pass_without_authorizing(self):
        protocol, config, raw = self.fixture()
        report = audit(protocol, config, raw)
        self.assertEqual(report["structural_status"], "passed")
        self.assertEqual(report["failure_codes"], [])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_changed_raw_bytes_fail_even_when_parsed_values_match(self):
        protocol, config, raw = self.fixture()
        report = audit(protocol, config, raw + b" ")
        self.assertIn("sampling_config_hash_mismatch", report["failure_codes"])

    def test_declared_value_mismatch_fails_even_with_matching_digest(self):
        protocol, config, raw = self.fixture()
        config["num_frames"] = 32
        changed_raw = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
        protocol["sampling"]["config_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        report = audit(protocol, config, changed_raw)
        self.assertIn("sampling_config_values_mismatch", report["failure_codes"])
        self.assertNotIn("sampling_config_hash_mismatch", report["failure_codes"])

    def test_boolean_and_integer_are_not_interchangeable(self):
        protocol, config, raw = self.fixture()
        config["uses_claim_for_selection"] = 0
        report = audit(protocol, config, raw)
        self.assertIn("sampling_config_values_mismatch", report["failure_codes"])

    def test_wrong_protocol_version_fails_closed(self):
        protocol, config, raw = self.fixture()
        protocol["schema_version"] = 12
        report = audit(protocol, config, raw)
        self.assertIn("protocol_sampling_declaration_invalid", report["failure_codes"])
        self.assertEqual(report["execution_authorization"], "not_granted")


if __name__ == "__main__":
    unittest.main()
