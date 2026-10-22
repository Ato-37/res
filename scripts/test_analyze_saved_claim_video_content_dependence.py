from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analyze_saved_claim_video_content_dependence import analyze


def receipt(token: str, contrast: float) -> dict:
    return {"status":"completed_one_row_exploratory_video_content_sensitivity",
        "direct_faithfulness":False,"faithfulness_result":False,
        "human_review":{"review_actually_performed":False},
        "requested_forwards":2,"attempted_forwards":2,"completed_forwards":2,
        "row_token_sha256":token*64,
        "branch_target_scores":{"original_video":{"target_logprob_mean":contrast-1},
            "constant_gray_video":{"target_logprob_mean":-1}},
        "primary_descriptive_contrast_original_minus_gray_nats_per_token":contrast}


class AnalyzeSavedClaimVideoContentTests(unittest.TestCase):
    def test_finite_cohort_statistics(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for idx,value in enumerate((0.5,-0.2,1.1)):
                path=Path(tmp)/f"{idx}.json"; path.write_text(json.dumps(receipt(str(idx+1),value)))
                paths.append(path)
            result=analyze(tuple(paths))
        self.assertEqual(result["row_contrasts_original_minus_gray_nats_per_token"],[0.5,-0.2,1.1])
        self.assertAlmostEqual(result["finite_cohort_mean_nats_per_token"],0.4666666666666666)
        self.assertEqual(result["finite_cohort_range_nats_per_token"],[-0.2,1.1])
        self.assertEqual(result["positive_rows"],2)
        self.assertFalse(result["direct_faithfulness"])

    def test_rejects_duplicate_and_incomplete_rows(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for idx in range(3):
                path=Path(tmp)/f"{idx}.json"; path.write_text(json.dumps(receipt(str(idx+1),0.1)))
                paths.append(path)
            paths[2].write_text(json.dumps(receipt("1",0.1)))
            with self.assertRaisesRegex(ValueError,"duplicated"):
                analyze(tuple(paths))
            paths[2].write_text(json.dumps(receipt("3",0.1)))
            broken=receipt("c",0.2); broken["completed_forwards"]=1
            paths[2].write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,"complete"):
                analyze(tuple(paths))

    def test_rejects_stored_arithmetic_mismatch(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for idx in range(3):
                item=receipt(str(idx+1),0.2)
                if idx == 0:
                    item["primary_descriptive_contrast_original_minus_gray_nats_per_token"]=0.3
                path=Path(tmp)/f"{idx}.json"; path.write_text(json.dumps(item)); paths.append(path)
            with self.assertRaisesRegex(ValueError,"arithmetic_mismatch"):
                analyze(tuple(paths))


if __name__ == "__main__":
    unittest.main()
