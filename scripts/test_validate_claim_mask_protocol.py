import contextlib, copy, hashlib, io, json, tempfile, unittest
from pathlib import Path
from validate_claim_mask_protocol import main, validate

H = "0" * 64
G = "0" * 40
def fixture():
    return {"schema_version":1,"protocol_stage":"confirmatory","model":{"revision":G,"tokenizer_revision":G,"quant":"4bit","dtype":"bfloat16","attention_implementation":"sdpa"},"cohort":{"split_manifest_sha256":H,"overlap_audit_sha256":H,"video_count":2,"analysis_unit_count":2,"split_overlap_count":0},"claim_review":{"claim_manifest_sha256":H,"mapping_sha256":H,"review_record_sha256":H,"review_status":"completed"},"sampling":{"policy":"whole_video_fixed","config_sha256":H,"uses_claim_for_selection":False,"num_frames":16,"frame_size":1024,"max_pixels":200704,"cap_pixels_per_frame":True},"estimand":{"target_text_sha256":H,"answer_text_sha256":H,"score_kind":"teacher_forced_logprob","baseline_correction":"selected_minus_control_drop","triplet_manifest_sha256":H},"intervention":{"kind":"claim_mask","selector_sha256":H,"control_sha256":H,"selected_condition":"claim tokens masked","control_condition":"matched nonclaim tokens masked"},"runtime":{"seed":1,"source_commit":"0123456789abcdef0123456789abcdef01234567","execution_authorization":False}}

class TestClaimMaskProtocol(unittest.TestCase):
    def test_valid_is_structural_only(self):
        r=validate(fixture(),H); self.assertTrue(r["structurally_valid"]); self.assertEqual(r["execution_authorization"],"not_granted")

    def test_invalid_caller_supplied_input_digest_is_not_echoed(self):
        r=validate(fixture(), "not-a-hash")
        self.assertFalse(r["structurally_valid"])
        self.assertIn("input_sha256_invalid",r["failure_codes"])
        self.assertNotIn("input_sha256",r)
    def test_incomplete_review_fails_closed(self):
        x=fixture(); x["claim_review"]["review_status"]="incomplete"; r=validate(x); self.assertIn("claim_review_incomplete",r["failure_codes"])
    def test_annotation_sampling_fails(self):
        x=fixture(); x["sampling"]["uses_claim_for_selection"]=True; r=validate(x); self.assertIn("sampling_fields_invalid",r["failure_codes"])

    def test_boolean_schema_version_is_not_integer_one(self):
        x=fixture(); x["schema_version"]=True; r=validate(x)
        self.assertFalse(r["structurally_valid"]); self.assertIn("schema_version_invalid",r["failure_codes"])

    def test_unhashable_protocol_stage_returns_failure_instead_of_raising(self):
        for bad in ([], {}, None):
            with self.subTest(protocol_stage=bad):
                x=fixture(); x["protocol_stage"]=bad; r=validate(x)
                self.assertFalse(r["structurally_valid"])
                self.assertIn("protocol_stage_invalid",r["failure_codes"])

    def test_model_and_tokenizer_revisions_require_full_immutable_ids(self):
        for field, bad in (("revision", "main"), ("tokenizer_revision", "abc"),
                           ("revision", "g" * 40), ("tokenizer_revision", "0" * 39)):
            with self.subTest(field=field, value=bad):
                x=fixture(); x["model"][field]=bad; r=validate(x)
                self.assertIn("model_field_missing",r["failure_codes"])

    def test_runtime_requires_full_git_commit_hash(self):
        for bad in ("abc", "g" * 40, "0" * 39):
            with self.subTest(source_commit=bad):
                x=fixture(); x["runtime"]["source_commit"]=bad; r=validate(x)
                self.assertIn("runtime_fields_invalid",r["failure_codes"])

    def test_overlap_count_cannot_exceed_video_count(self):
        x=fixture(); x["cohort"]["split_overlap_count"]=3; r=validate(x)
        self.assertIn("split_overlap_exceeds_video_count",r["failure_codes"])

    def test_selected_and_control_conditions_must_be_distinct(self):
        x=fixture(); x["intervention"]["control_condition"]=x["intervention"]["selected_condition"]
        r=validate(x); self.assertIn("intervention_fields_invalid",r["failure_codes"])

    def test_runtime_cannot_grant_execution_authorization(self):
        x=fixture(); x["runtime"]["execution_authorization"]=True; r=validate(x)
        self.assertIn("runtime_fields_invalid",r["failure_codes"])

    def test_duplicate_json_key_is_rejected_without_echoing_input(self):
        raw=json.dumps(fixture()).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"duplicate.json"; path.write_text(raw, encoding="utf-8")
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
        self.assertEqual(code,2)
        result=json.loads(stdout.getvalue())
        self.assertFalse(result["structurally_valid"])
        self.assertIn("input_unreadable_or_malformed",result["failure_codes"])
        self.assertNotIn("claim tokens masked",stdout.getvalue())

    def test_cli_rejects_nonfinite_json_and_valid_input_never_authorizes_execution(self):
        valid=json.dumps(fixture())
        nonfinite=valid.replace('"seed": 1', '"seed": NaN')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"protocol.json"; stdout=io.StringIO()
            path.write_text(nonfinite, encoding="utf-8")
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
            self.assertEqual(code,2)
            failed=json.loads(stdout.getvalue())
            self.assertIn("input_unreadable_or_malformed",failed["failure_codes"])
            stdout=io.StringIO(); path.write_text(valid, encoding="utf-8")
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
        self.assertEqual(code,0)
        accepted=json.loads(stdout.getvalue())
        self.assertTrue(accepted["structurally_valid"])
        self.assertEqual(accepted["execution_authorization"],"not_granted")
