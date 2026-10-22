from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from analyze_claim_qa_interval_likelihood_interaction import summarize


def receipt():
    scores={}
    for name,value,hook in (("qa_interval_baseline",-1.5,0),("control_baseline",-1,0),
            ("qa_interval_biased",-1.6,1),("control_biased",-.8,1)):
        scores[name]={"target_logprob_mean":value,
            "score_semantics":"mean conditional logprob of frozen target token IDs",
            "target_token_count":5,"visual_key_position_count":10,
            "intervention_hook_invocations_by_layer_1_based":{str(i):hook for i in range(5,17)}}
    return {"status":"completed_one_row_exploratory_qa_interval_claim_likelihood_interaction",
        "direct_faithfulness":False,"human_review":{"review_actually_performed":False},
        "requested_records":1,"effective_records":1,"requested_forwards":4,
        "attempted_forwards":4,"completed_forwards":4,"row_token_sha256":"r"*64,
        "qa_interval_frame_count":2,"control_frame_count":2,"target_token_count":5,
        "condition_scores":scores,"primary_descriptive_interaction":{
            "baseline_control_minus_qa_interval_nats_per_token":.5,
            "biased_control_minus_qa_interval_nats_per_token":.8,
            "bias_by_qa_interval_interaction_nats_per_token":.3}}


class ClaimIntervalAnalysisTests(unittest.TestCase):
    def test_recomputes_and_excludes_failure(self):
        with TemporaryDirectory() as d:
            root=Path(d); good=root/"good.json";good.write_text(json.dumps(receipt()))
            bad=root/"failed.json";bad.write_text(json.dumps({"status":"failed","completed_forwards":2}))
            result=summarize([good,bad]);self.assertAlmostEqual(result["interaction_mean"],.3)
            self.assertEqual(result["completed_rows"],1);self.assertEqual(len(result["excluded_receipts"]),1)

    def test_rejects_hook_mismatch(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"bad.json";data=receipt()
            data["condition_scores"]["control_biased"]["intervention_hook_invocations_by_layer_1_based"]["5"]=0
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"hook_audit_invalid"):summarize([p])

    def test_rejects_bad_interaction_arithmetic(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"bad.json";data=receipt()
            data["primary_descriptive_interaction"]["bias_by_qa_interval_interaction_nats_per_token"]=.31
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"arithmetic_mismatch"):summarize([p])


if __name__=="__main__":unittest.main()
