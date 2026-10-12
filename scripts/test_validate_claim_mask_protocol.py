import copy, hashlib, json, unittest
from validate_claim_mask_protocol import validate

H = "0" * 64
def fixture():
    return {"schema_version":1,"protocol_stage":"confirmatory","model":{"revision":"m","tokenizer_revision":"t","quant":"4bit","dtype":"bfloat16","attention_implementation":"sdpa"},"cohort":{"split_manifest_sha256":H,"overlap_audit_sha256":H,"video_count":2,"analysis_unit_count":2,"split_overlap_count":0},"claim_review":{"claim_manifest_sha256":H,"mapping_sha256":H,"review_record_sha256":H,"review_status":"completed"},"sampling":{"policy":"whole_video_fixed","config_sha256":H,"uses_claim_for_selection":False,"num_frames":16,"frame_size":1024,"max_pixels":200704,"cap_pixels_per_frame":True},"estimand":{"target_text_sha256":H,"answer_text_sha256":H,"score_kind":"teacher_forced_logprob","baseline_correction":"selected_minus_control_drop","triplet_manifest_sha256":H},"intervention":{"kind":"claim_mask","selector_sha256":H,"control_sha256":H,"selected_condition":"claim tokens masked","control_condition":"matched nonclaim tokens masked"},"runtime":{"seed":1,"source_commit":"abc","execution_authorization":False}}

class TestClaimMaskProtocol(unittest.TestCase):
    def test_valid_is_structural_only(self):
        r=validate(fixture(),H); self.assertTrue(r["structurally_valid"]); self.assertEqual(r["execution_authorization"],"not_granted")
    def test_incomplete_review_fails_closed(self):
        x=fixture(); x["claim_review"]["review_status"]="incomplete"; r=validate(x); self.assertIn("claim_review_incomplete",r["failure_codes"])
    def test_annotation_sampling_fails(self):
        x=fixture(); x["sampling"]["uses_claim_for_selection"]=True; r=validate(x); self.assertIn("sampling_fields_invalid",r["failure_codes"])
