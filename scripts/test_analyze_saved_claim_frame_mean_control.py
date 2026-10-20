from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from analyze_saved_claim_frame_mean_control import analyze

def make(token,value):
    return {"status":"completed_one_row_posthoc_frame_mean_sensitivity","post_result_exploration":True,
        "direct_faithfulness":False,"human_review":{"review_actually_performed":False},
        "requested_forwards":2,"attempted_forwards":2,"completed_forwards":2,
        "row_token_sha256":token*64,"branch_target_scores":{"original_video":{"target_logprob_mean":value-1},
            "per_frame_rgb_mean_video":{"target_logprob_mean":-1}},
        "primary_descriptive_contrast_original_minus_frame_mean_nats_per_token":value}

class FrameMeanSummaryTests(unittest.TestCase):
    def test_statistics_and_direct_faithfulness_flag(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for i,v in enumerate((0.3,-0.1,0.8)):
                p=Path(tmp)/f"{i}.json";p.write_text(json.dumps(make(str(i+1),v)));paths.append(p)
            result=analyze(tuple(paths))
        self.assertAlmostEqual(result["finite_cohort_mean_nats_per_token"],1/3)
        self.assertEqual(result["finite_cohort_range_nats_per_token"],[-0.1,0.8])
        self.assertFalse(result["direct_faithfulness"])

    def test_incomplete_or_duplicate_receipts_rejected(self):
        with TemporaryDirectory() as tmp:
            paths=[]
            for i in range(3):
                p=Path(tmp)/f"{i}.json";p.write_text(json.dumps(make(str(i+1),0.2)));paths.append(p)
            paths[2].write_text(json.dumps(make("1",0.2)))
            with self.assertRaisesRegex(ValueError,"duplicated"): analyze(tuple(paths))
            paths[2].write_text(json.dumps(make("3",0.2)))
            x=make("3",0.2);x["completed_forwards"]=1;paths[2].write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,"complete"): analyze(tuple(paths))

if __name__=="__main__":unittest.main()
