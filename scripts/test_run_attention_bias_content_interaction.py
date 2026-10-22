from __future__ import annotations
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_attention_bias_content_interaction as module


def make_scores(value,hook=0):
    return {"target_logprob_mean":value,"score_semantics":"mean conditional logprob of frozen target token IDs",
        "target_token_count":3,"visual_key_position_count":20,"visual_key_positions_sha256":"v"*64,
        "first_query_position":100,"last_target_query_position":102,
        "intervention_hook_invocations_by_layer_1_based":{str(i):hook for i in range(5,17)}}

def paired_receipts():
    shared={"row_token_sha256":"r"*64,"source_video_sha256":"v"*64,"saved_claim_text_sha256":"c"*64,
        "private_input_packet_sha256":"p"*64,"model_revision":module.MODEL_REVISION,
        "reencoded_claim_token_ids_sha256":"t"*64,"retokenized_claim_ids_sha256":"t"*64,"target_token_count":3,
        "direct_faithfulness":False,"human_review":{"review_actually_performed":False},
        "source_sha256":{name:"s"*64 for name in ("scripts/qta_multiquery_video_key_intervention.py",
            "scripts/claim_mask_video_decoder.py","scripts/generate_nextgqa_claim_review_pilot.py",
            "scripts/run_saved_claim_frame_permutation_attention_smoke.py")}}
    attention={**shared,"source_sha256":dict(shared["source_sha256"]),"status":"completed_one_row_exploratory_sensitivity_smoke","completed_forwards":4,
        "attention_intervention":{"layers_1_based":list(module.LAYERS_1_BASED),"logit_bias":module.LOGIT_BIAS},
        "branch_target_scores":{"original_baseline":make_scores(-1.0),"original_visual_bias":make_scores(-0.8,hook=1)}}
    content={**shared,"source_sha256":dict(shared["source_sha256"]),"status":"completed_one_row_exploratory_video_content_sensitivity","completed_forwards":2,
        "branch_target_scores":{"original_video":make_scores(-1.0),"constant_gray_video":make_scores(-1.5)}}
    return attention,content

def prepare_sources(tmp,a,c):
    names=set(a["source_sha256"])|set(c["source_sha256"])
    digest=hashlib.sha256(b"src").hexdigest()
    for name in names:
        path=Path(tmp)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"src")
        a["source_sha256"][name]=digest;c["source_sha256"][name]=digest
    for item in ("scripts/run_saved_claim_frame_permutation_attention_smoke_20261021_v1.py",
                 "scripts/run_saved_claim_video_content_dependence.py"):
        path=Path(tmp)/item;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"src")


class AttentionBiasContentInteractionTests(unittest.TestCase):
    def test_packet_local_rows_map_to_frozen_global_rows(self):
        self.assertEqual(module.fixed_global_index(module.PACKET_REL,0),0)
        self.assertEqual(module.fixed_global_index(module.REMAINING_PACKET_REL,0),1)
        self.assertEqual(module.fixed_global_index(module.REMAINING_PACKET_REL,1),2)
        with self.assertRaisesRegex(ValueError,"index_invalid"):
            module.fixed_global_index(module.PACKET_REL,1)

    def test_interaction_difference_in_differences(self):
        result=module.interaction(-1.0,-1.5,-0.8,-1.6)
        self.assertAlmostEqual(result["baseline_original_minus_gray_nats_per_token"],0.5)
        self.assertAlmostEqual(result["biased_original_minus_gray_nats_per_token"],0.8)
        self.assertAlmostEqual(result["bias_by_video_content_interaction_nats_per_token"],0.3)

    def test_cross_receipts_join_on_frozen_inputs(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"data"/"private_nextgqa";root.mkdir(parents=True)
            a,c=paired_receipts();prepare_sources(tmp,a,c);(root/"a.json").write_text(json.dumps(a));(root/"c.json").write_text(json.dumps(c))
            previous=module.RECEIPTS
            try:
                module.RECEIPTS={0:("a.json","c.json")}
                joined=module.load_joined_receipts(Path(tmp),0)
                self.assertEqual(joined[0]["row_token_sha256"],"r"*64)
            finally: module.RECEIPTS=previous

    def test_mismatched_row_and_visual_geometry_rejected(self):
        for mutate,expected in (
            (lambda a,c:a.update(row_token_sha256="x"*64),"row_token_sha256"),
            (lambda a,c:c["branch_target_scores"]["constant_gray_video"].update(first_query_position=99),"visual_query_positions"),
            (lambda a,c:c["source_sha256"].update({"scripts/claim_mask_video_decoder.py":"x"*64}),"source_hash")):
            with self.subTest(expected=expected),TemporaryDirectory() as tmp:
                root=Path(tmp)/"data"/"private_nextgqa";root.mkdir(parents=True)
                a,c=paired_receipts();prepare_sources(tmp,a,c);mutate(a,c)
                (root/"a.json").write_text(json.dumps(a));(root/"c.json").write_text(json.dumps(c))
                previous=module.RECEIPTS
                try:
                    module.RECEIPTS={0:("a.json","c.json")}
                    with self.assertRaisesRegex(ValueError,expected):module.load_joined_receipts(Path(tmp),0)
                finally:module.RECEIPTS=previous

if __name__=="__main__":unittest.main()
