"""Fail-closed CPU validator for a future claim-mask evaluation contract.

This checks declarations and hashes only. It does not select claims, inspect
videos, authenticate review, or authorize inference.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re, sys
from typing import Any

SHA = re.compile(r"^[a-f0-9]{64}$")
GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")

def _obj(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise ValueError("duplicate key")
        out[k] = v
    return out

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def _sha(x): return isinstance(x, str) and SHA.fullmatch(x) is not None
def _text(x): return isinstance(x, str) and bool(x.strip())
def _keys(x, required): return isinstance(x, dict) and set(x) == set(required)

def validate(value: Any, input_sha256: str | None = None) -> dict:
    f = set()
    req = {"schema_version", "protocol_stage", "model", "cohort", "claim_review",
           "sampling", "estimand", "intervention", "runtime"}
    if not isinstance(value, dict): return _summary({"protocol_not_object"}, input_sha256)
    if set(value) != req: f.add("protocol_schema_keys_invalid")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1: f.add("schema_version_invalid")
    stage = value.get("protocol_stage")
    if not isinstance(stage, str) or stage not in {"exploratory", "confirmatory"}: f.add("protocol_stage_invalid")
    model = value.get("model"); mk = {"revision", "tokenizer_revision", "quant", "dtype", "attention_implementation"}
    if not _keys(model, mk): f.add("model_schema_invalid")
    elif any(not isinstance(model.get(k), str) or GIT_COMMIT.fullmatch(model[k]) is None
             for k in ("revision", "tokenizer_revision")) or not all(_text(model.get(k)) for k in ("quant", "dtype", "attention_implementation")): f.add("model_field_missing")
    cohort = value.get("cohort"); ck = {"split_manifest_sha256", "overlap_audit_sha256", "video_count", "analysis_unit_count", "split_overlap_count"}
    if not _keys(cohort, ck): f.add("cohort_schema_invalid")
    elif (not all(_sha(cohort.get(k)) for k in ("split_manifest_sha256", "overlap_audit_sha256"))
          or any(type(cohort.get(k)) is not int or cohort[k] < 0 for k in ("video_count", "analysis_unit_count", "split_overlap_count"))):
        f.add("cohort_fields_invalid")
    elif cohort["video_count"] == 0 or cohort["analysis_unit_count"] != cohort["video_count"]: f.add("analysis_unit_not_video_level")
    if isinstance(cohort, dict) and all(type(cohort.get(k)) is int for k in ("video_count", "split_overlap_count")) and cohort["split_overlap_count"] > cohort["video_count"]: f.add("split_overlap_exceeds_video_count")
    if isinstance(cohort, dict) and stage == "confirmatory" and cohort.get("split_overlap_count") != 0: f.add("confirmatory_split_overlap")
    cr = value.get("claim_review"); rk = {"claim_manifest_sha256", "mapping_sha256", "review_record_sha256", "review_status"}
    if not _keys(cr, rk): f.add("claim_review_schema_invalid")
    elif not all(_sha(cr.get(k)) for k in ("claim_manifest_sha256", "mapping_sha256", "review_record_sha256")): f.add("claim_review_hash_invalid")
    elif cr.get("review_status") != "completed": f.add("claim_review_incomplete")
    sm = value.get("sampling"); sk = {"policy", "config_sha256", "uses_claim_for_selection", "num_frames", "frame_size", "max_pixels", "cap_pixels_per_frame"}
    if not _keys(sm, sk): f.add("sampling_schema_invalid")
    elif (sm.get("policy") != "whole_video_fixed" or not _sha(sm.get("config_sha256")) or sm.get("uses_claim_for_selection") is not False
          or any(type(sm.get(k)) is not int or sm[k] <= 0 for k in ("num_frames", "frame_size", "max_pixels")) or type(sm.get("cap_pixels_per_frame")) is not bool): f.add("sampling_fields_invalid")
    es = value.get("estimand"); ek = {"target_text_sha256", "answer_text_sha256", "score_kind", "baseline_correction", "triplet_manifest_sha256"}
    if not _keys(es, ek): f.add("estimand_schema_invalid")
    elif (not _sha(es.get("target_text_sha256")) or not _sha(es.get("answer_text_sha256")) or not _sha(es.get("triplet_manifest_sha256"))
          or es.get("score_kind") != "teacher_forced_logprob" or es.get("baseline_correction") != "selected_minus_control_drop"): f.add("estimand_fields_invalid")
    iv = value.get("intervention"); ik = {"kind", "selector_sha256", "control_sha256", "selected_condition", "control_condition"}
    if not _keys(iv, ik): f.add("intervention_schema_invalid")
    elif (iv.get("kind") != "claim_mask" or not _sha(iv.get("selector_sha256")) or not _sha(iv.get("control_sha256"))
          or not _text(iv.get("selected_condition")) or not _text(iv.get("control_condition"))
          or iv.get("selected_condition") == iv.get("control_condition")): f.add("intervention_fields_invalid")
    rt = value.get("runtime"); tk = {"seed", "source_commit", "execution_authorization"}
    if not _keys(rt, tk): f.add("runtime_schema_invalid")
    elif type(rt.get("seed")) is not int or not isinstance(rt.get("source_commit"), str) or GIT_COMMIT.fullmatch(rt["source_commit"]) is None or rt.get("execution_authorization") is not False: f.add("runtime_fields_invalid")
    return _summary(f, input_sha256)

def _summary(f, digest):
    failures = set(f)
    if digest is not None and not _sha(digest):
        failures.add("input_sha256_invalid")
        digest = None
    return {"schema_version": 1, "structurally_valid": not failures, "failure_codes": sorted(failures),
            "semantic_validation": "not_machine_verifiable", "execution_authorization": "not_granted",
            **({"input_sha256": digest} if digest else {})}

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("path"); a = ap.parse_args(argv)
    try:
        with open(a.path, "rb") as stream:
            raw = stream.read()
        value = json.loads(raw, object_pairs_hook=_obj, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite")))
        out = validate(value, hashlib.sha256(raw).hexdigest())
    except Exception:
        out = _summary({"input_unreadable_or_malformed"}, None)
    print(json.dumps(out, sort_keys=True)); return 0 if out["structurally_valid"] else 2
if __name__ == "__main__": raise SystemExit(main())
