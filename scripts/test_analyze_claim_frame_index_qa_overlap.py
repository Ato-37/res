import unittest

from analyze_claim_frame_index_qa_overlap import analyze_rows, span_jaccard, summarize


class FrameIndexQAOverlapTests(unittest.TestCase):
    @staticmethod
    def fixture():
        records = [{
            "video_sha256": f"{i:064x}",
            "sampled_frame_relative_pts_seconds": [i / 15 for i in range(16)],
            "gold_interval_seconds": [[0.2, 0.4]],
        } for i in range(3)]
        row = {"row_index": 0, "start_frame_index": 3, "end_frame_index": 5}
        return ({"status": "passed", "records": records},
                {"status": "passed", "mode": "full", "effective_rows": 3,
                 "requested_scorer_forwards": 15, "completed_scorer_forwards": 15,
                 "source_video_sha256": [record["video_sha256"] for record in records],
                 "rows": [dict(row, row_index=i) for i in range(3)]})

    def test_frame_set_jaccard_and_rank(self):
        self.assertAlmostEqual(span_jaccard([2, 3, 4], [3, 4, 5]), 0.5)
        packet, result = self.fixture()
        rows = analyze_rows(packet, result)
        self.assertEqual([row["qa_interval_frame_count"] for row in rows], [4] * 3)
        self.assertEqual([row["jaccard"] for row in rows], [0.75] * 3)
        self.assertEqual(summarize(rows)["n"], 3)

    def test_rejects_packet_result_source_mismatch_and_partial_result(self):
        packet, result = self.fixture()
        result["source_video_sha256"] = ["b" * 64] * 3
        with self.assertRaisesRegex(ValueError, "source_binding"):
            analyze_rows(packet, result)
        packet, result = self.fixture()
        result["completed_scorer_forwards"] = 14
        with self.assertRaisesRegex(ValueError, "incomplete_pilot_pair"):
            analyze_rows(packet, result)

    def test_rejects_invalid_machine_interval(self):
        packet, result = self.fixture()
        result["rows"][0]["end_frame_index"] = 8
        with self.assertRaisesRegex(ValueError, "localized_span_invalid"):
            analyze_rows(packet, result)


if __name__ == "__main__":
    unittest.main()
