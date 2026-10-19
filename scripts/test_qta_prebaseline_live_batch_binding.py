import json
import unittest

import torch

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from qta_actual_qa_source_contract import bind_qta_actual_qa_source
from qta_actual_prebaseline_diagnostic import diagnose_qta_prebaseline
from test_qta_actual_prebaseline_diagnostic import combined_inputs


class LiveProcessorBatchBindingTests(unittest.TestCase):
    def setUp(self):
        self.source, self.review, _ = combined_inputs()
        bound = bind_qta_actual_qa_source(**self.source)
        self.crosswalk = json.loads(
            bound["private_bundle"]["raw_json"].decode("utf-8")
        )["review_source_crosswalk"]
        source_lineage = json.loads(self.source["lineage_manifest_raw"])["records"]
        self.batches = [
            FrozenBatch({
                "input_ids": torch.tensor([[index + 1, index + 2]], dtype=torch.int64),
                "pixel_values_videos": torch.arange(12, dtype=torch.float32).reshape(6, 2)
                                      + index,
            })
            for index in range(3)
        ]
        self.expected = []
        self.live = []
        for row, line, batch in zip(self.crosswalk, source_lineage, self.batches, strict=True):
            digest = batch_sha256(batch)
            identity = {"review_token": row["review_token"],
                        "source_video_sha256": line["video_sha256"]}
            self.expected.append({**identity, "batch_sha256": digest})
            self.live.append({**identity, "batch_sha256": digest, "batch": batch})

    def diagnose(self, expected=None, live=None):
        return diagnose_qta_prebaseline(
            actual_source_inputs=self.source,
            review_bundle_inputs=self.review,
            expected_processor_rows=self.expected if expected is None else expected,
            live_processor_rows=self.live if live is None else live,
        )

    def test_exact_live_batches_join_all_three_rows_but_remain_non_authorizing(self):
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["validated_rows"], 3)
        self.assertIs(report["model_load_permitted"], False)
        self.assertIs(report["scoring_authorized"], False)
        self.assertIs(report["faithfulness_scored"], False)
        serialized = json.dumps(report, ensure_ascii=False)
        for row in self.expected:
            self.assertNotIn(row["review_token"], serialized)
            self.assertNotIn(row["batch_sha256"], serialized)

    def test_reordered_live_rows_fail_atomically(self):
        report = self.diagnose(live=list(reversed(self.live)))
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["validated_rows"], 0)

    def test_mutated_batch_tensor_fails_live_digest_check(self):
        self.live[1]["batch"]["pixel_values_videos"][0, 0] += 1
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_live_digest_mismatch")
        self.assertEqual(gate["validated_rows"], 0)

    def test_nonfinite_batch_is_rejected_before_hash_acceptance(self):
        self.live[1]["batch"]["pixel_values_videos"][0, 0] = float("inf")
        report = self.diagnose()
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_nonfinite_tensor")
        self.assertEqual(gate["validated_rows"], 0)

    def test_duplicate_source_token_is_rejected_atomically(self):
        duplicate = [dict(row) for row in self.expected]
        duplicate[2]["review_token"] = duplicate[1]["review_token"]
        report = self.diagnose(expected=duplicate)
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "processor_batch_duplicate_source_or_token")
        self.assertEqual(gate["validated_rows"], 0)

    def test_upstream_mismatch_blocks_processor_binding(self):
        changed_review = dict(self.review)
        changed_review["expected_lineage_sha256"] = "f" * 64
        report = diagnose_qta_prebaseline(
            actual_source_inputs=self.source,
            review_bundle_inputs=changed_review,
            expected_processor_rows=self.expected,
            live_processor_rows=self.live,
        )
        gate = report["gates"]["processor_batch_binding"]
        self.assertEqual(gate["failure_code"], "upstream_source_review_join_blocked")
        self.assertEqual(gate["validated_rows"], 0)


if __name__ == "__main__":
    unittest.main()
