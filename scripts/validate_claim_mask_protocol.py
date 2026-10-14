"""Fail-closed CPU validator for a future claim-mask evaluation contract.

This checks declarations and hashes only. It does not select claims, inspect
videos, authenticate review, or authorize inference.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re, sys
from datetime import datetime
from typing import Any

SHA = re.compile(r"^[a-f0-9]{64}$")
GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

def _obj(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise ValueError("duplicate key")
        out[k] = v
    return out

def _finite(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool): return False
    try: return math.isfinite(x)
    except (OverflowError, TypeError, ValueError): return False

def _sha(x): return isinstance(x, str) and SHA.fullmatch(x) is not None
def _text(x): return isinstance(x, str) and bool(x.strip())
def _enum(x, choices): return isinstance(x, str) and x in choices
def _keys(x, required): return isinstance(x, dict) and set(x) == set(required)

def compute_primary_estimand_sha256(value: dict) -> str | None:
    """Hash the canonical, outcome-independent definition of the primary contrast."""
    try:
        m, c, s, e, i = (value[k] for k in ("model", "cohort", "sampling", "estimand", "intervention"))
        payload = {
            "model": {k: m[k] for k in ("revision", "tokenizer_revision", "quant", "dtype", "attention_implementation")},
            "eligible_manifest_sha256": c["eligible_manifest_sha256"],
            "primary_population": c["primary_population"],
            "inference_design": value["inference_design"],
            "sampling_config_sha256": s["config_sha256"],
            "claim_review": {k: value["claim_review"][k] for k in ("claim_manifest_sha256", "mapping_sha256", "review_record_sha256", "mapping_unit")},
            "answer_candidate_protocol_sha256": e["answer_candidate_protocol_sha256"],
            "gold_choice_manifest_sha256": e["gold_choice_manifest_sha256"],
            "triplet_manifest_sha256": e["triplet_manifest_sha256"],
            "score_kind": e["score_kind"],
            "baseline_correction": e["baseline_correction"],
            "target_definition": e["target_definition"],
            "margin_definition": e["margin_definition"],
            "intervention": {k: i[k] for k in ("kind", "selector_sha256", "control_sha256", "mask_transform_sha256", "minimum_selected_frame_count", "selected_condition", "control_condition", "selected_unit", "control_matching")},
        }
        if not _valid_hash_payload(payload):
            return None
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except (KeyError, TypeError, UnicodeError):
        return None

def compute_inference_plan_sha256(value: dict) -> str | None:
    """Hash uncertainty assumptions and analysis choices separately from the point estimand."""
    try:
        analysis = value["analysis"]
        payload = {
            "dependence_plan": value["dependence_plan"],
            "primary_ci_method": analysis["primary_ci_method"],
            "primary_contrast_count": analysis["primary_contrast_count"],
            "primary_multiplicity": analysis["primary_multiplicity"],
            "secondary_families": analysis["secondary_families"],
            "secondary_adjustment": analysis["secondary_adjustment"],
            "secondary_hypothesis_manifest_sha256": analysis["secondary_hypothesis_manifest_sha256"],
        }
        if not _valid_hash_payload(payload): return None
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except (KeyError, TypeError, UnicodeError):
        return None

def _valid_hash_payload(value):
    if isinstance(value, dict):
        return all(_text(key) and _valid_hash_payload(item) for key, item in value.items())
    elif isinstance(value, list):
        return all(_valid_hash_payload(item) for item in value)
    elif value is None:
        return True
    elif isinstance(value, str):
        return _text(value)
    return type(value) is int and value >= 0

def validate(value: Any, input_sha256: str | None = None) -> dict:
    f = set()
    req = {"schema_version", "protocol_stage", "model", "cohort", "inference_design", "dependence_plan", "claim_review",
           "data_rights", "sampling", "estimand", "intervention", "analysis", "runtime"}
    if not isinstance(value, dict): return _summary({"protocol_not_object"}, input_sha256)
    if set(value) != req: f.add("protocol_schema_keys_invalid")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 10: f.add("schema_version_invalid")
    stage = value.get("protocol_stage")
    # The estimand direction and secondary Holm families are selected, but the
    # SESOI is not fixed. This schema must not label a run confirmatory yet.
    if stage != "exploratory": f.add("protocol_stage_invalid")
    model = value.get("model"); mk = {"revision", "tokenizer_revision", "quant", "dtype", "attention_implementation"}
    if not _keys(model, mk): f.add("model_schema_invalid")
    elif any(not isinstance(model.get(k), str) or GIT_COMMIT.fullmatch(model[k]) is None
             for k in ("revision", "tokenizer_revision")) or not all(_text(model.get(k)) for k in ("quant", "dtype", "attention_implementation")): f.add("model_field_missing")
    cohort = value.get("cohort"); ck = {"candidate_manifest_sha256", "eligible_manifest_sha256", "exclusion_manifest_sha256", "overlap_audit_sha256", "candidate_video_count", "eligible_video_count", "prerun_exclusion_count", "analysis_unit_count", "split_overlap_count", "primary_population", "secondary_population"}
    if not _keys(cohort, ck): f.add("cohort_schema_invalid")
    elif (not all(_sha(cohort.get(k)) for k in ("candidate_manifest_sha256", "eligible_manifest_sha256", "exclusion_manifest_sha256", "overlap_audit_sha256"))
          or any(type(cohort.get(k)) is not int or cohort[k] < 0 for k in ("candidate_video_count", "eligible_video_count", "prerun_exclusion_count", "analysis_unit_count", "split_overlap_count"))
          or cohort.get("candidate_video_count") != cohort.get("eligible_video_count", -1) + cohort.get("prerun_exclusion_count", -1)
          or cohort.get("primary_population") != "all_eligible_videos"
          or cohort.get("secondary_population") != "baseline_correctness_strata"):
        f.add("cohort_fields_invalid")
    elif cohort["eligible_video_count"] == 0 or cohort["analysis_unit_count"] != cohort["eligible_video_count"]: f.add("analysis_unit_not_video_level")
    inference = value.get("inference_design")
    infk = {"target_kind", "target_definition_sha256", "selection_design", "sampling_frame_sha256", "selection_record_sha256"}
    if not _keys(inference, infk): f.add("inference_design_schema_invalid")
    elif (not _enum(inference.get("target_kind"), {"unresolved", "frozen_eligible_video_cohort", "eligible_video_superpopulation"})
          or not _sha(inference.get("target_definition_sha256"))
          or not _enum(inference.get("selection_design"), {"unknown", "census_of_frame", "probability_sample", "nonprobability_sample"})
          or not _sha(inference.get("sampling_frame_sha256")) or not _sha(inference.get("selection_record_sha256"))):
        f.add("inference_design_fields_invalid")
    dep = value.get("dependence_plan")
    dpk = {"uncertainty_target", "independence_unit", "cluster_audit_status", "cluster_unit", "independent_cluster_count", "cluster_audit_sha256", "method_status", "method"}
    if not _keys(dep, dpk): f.add("dependence_plan_schema_invalid")
    elif (not _enum(dep.get("uncertainty_target"), {"unresolved", "finite_cohort_description", "model_based_video_superpopulation", "design_based_population"})
          or dep.get("independence_unit") != "video"
          or not _enum(dep.get("cluster_audit_status"), {"unresolved", "reviewed_no_known_cluster_fields", "clusters_identified"})
          or not _enum(dep.get("cluster_unit"), {"unresolved", "video_source", "creator", "recording_event", "series", "multiway"})
          or (dep.get("independent_cluster_count") is not None and (type(dep.get("independent_cluster_count")) is not int or dep["independent_cluster_count"] < 1))
          or not _sha(dep.get("cluster_audit_sha256"))
          or not _enum(dep.get("method_status"), {"unresolved", "resolved"})
          or not _enum(dep.get("method"), {"unresolved", "two_sided_student_t_df_n_minus_1"})):
        f.add("dependence_plan_fields_invalid")
    elif ((dep["cluster_audit_status"] in {"unresolved", "reviewed_no_known_cluster_fields"} and (dep["cluster_unit"] != "unresolved" or dep["independent_cluster_count"] is not None))
          or (dep["cluster_audit_status"] == "unresolved" and dep["method_status"] != "unresolved")
          or dep["cluster_audit_status"] == "clusters_identified"
          or (dep["method_status"] == "resolved" and (dep["uncertainty_target"] != "model_based_video_superpopulation" or dep["method"] == "unresolved" or dep["cluster_audit_status"] != "reviewed_no_known_cluster_fields"))
          or (dep["method_status"] == "resolved" and isinstance(inference, dict) and inference.get("selection_design") == "unknown")
          or (dep["uncertainty_target"] == "design_based_population" and isinstance(inference, dict) and inference.get("selection_design") != "probability_sample")
          or (isinstance(inference, dict) and ((inference.get("target_kind") == "unresolved") != (dep["uncertainty_target"] == "unresolved")))
          or (isinstance(inference, dict) and inference.get("target_kind") == "frozen_eligible_video_cohort" and dep["uncertainty_target"] != "finite_cohort_description")
          or (isinstance(inference, dict) and inference.get("target_kind") == "eligible_video_superpopulation" and dep["uncertainty_target"] not in {"model_based_video_superpopulation", "design_based_population"})):
        f.add("dependence_plan_inconsistent")
    if isinstance(cohort, dict) and all(type(cohort.get(k)) is int for k in ("eligible_video_count", "split_overlap_count")) and cohort["split_overlap_count"] > cohort["eligible_video_count"]: f.add("split_overlap_exceeds_video_count")
    if isinstance(cohort, dict) and stage == "confirmatory" and cohort.get("split_overlap_count") != 0: f.add("confirmatory_split_overlap")
    cr = value.get("claim_review"); rk = {"claim_manifest_sha256", "mapping_sha256", "review_record_sha256", "review_status", "review_timing", "mapping_unit"}
    if not _keys(cr, rk): f.add("claim_review_schema_invalid")
    elif not all(_sha(cr.get(k)) for k in ("claim_manifest_sha256", "mapping_sha256", "review_record_sha256")): f.add("claim_review_hash_invalid")
    elif cr.get("review_status") != "completed": f.add("claim_review_incomplete")
    elif cr.get("review_timing") != "before_scoring": f.add("claim_review_timing_invalid")
    elif cr.get("mapping_unit") != "claim_to_evidence_frames": f.add("claim_review_mapping_unit_invalid")
    rights = value.get("data_rights"); dk = {"scope", "status", "evidence_sha256", "item_rights_manifest_sha256", "covered_eligible_video_count", "unresolved_eligible_video_count"}
    if not _keys(rights, dk): f.add("data_rights_schema_invalid")
    elif (not _text(rights.get("scope")) or not isinstance(rights.get("status"), str)
          or rights.get("status") not in {"unresolved", "verified"}
          or not _sha(rights.get("evidence_sha256"))
          or not _sha(rights.get("item_rights_manifest_sha256"))
          or any(type(rights.get(k)) is not int or rights[k] < 0
                 for k in ("covered_eligible_video_count", "unresolved_eligible_video_count"))): f.add("data_rights_fields_invalid")
    elif isinstance(cohort, dict) and type(cohort.get("eligible_video_count")) is int:
        covered = rights["covered_eligible_video_count"]
        unresolved = rights["unresolved_eligible_video_count"]
        if covered + unresolved != cohort["eligible_video_count"]: f.add("data_rights_coverage_count_invalid")
        if rights["status"] == "verified" and (covered != cohort["eligible_video_count"] or unresolved != 0):
            f.add("verified_rights_incomplete")
    sm = value.get("sampling"); sk = {"policy", "config_sha256", "uses_claim_for_selection", "num_frames", "frame_size", "max_pixels", "cap_pixels_per_frame"}
    if not _keys(sm, sk): f.add("sampling_schema_invalid")
    elif (sm.get("policy") != "whole_video_fixed" or not _sha(sm.get("config_sha256")) or sm.get("uses_claim_for_selection") is not False
          or any(type(sm.get(k)) is not int or sm[k] <= 0 for k in ("num_frames", "frame_size", "max_pixels")) or type(sm.get("cap_pixels_per_frame")) is not bool): f.add("sampling_fields_invalid")
    es = value.get("estimand"); ek = {"answer_candidate_protocol_sha256", "gold_choice_manifest_sha256", "primary_estimand_sha256", "score_kind", "baseline_correction", "triplet_manifest_sha256", "target_definition", "margin_definition"}
    if not _keys(es, ek): f.add("estimand_schema_invalid")
    elif (not _sha(es.get("answer_candidate_protocol_sha256")) or not _sha(es.get("gold_choice_manifest_sha256")) or not _sha(es.get("primary_estimand_sha256")) or not _sha(es.get("triplet_manifest_sha256"))
          or es.get("score_kind") != "teacher_forced_gold_vs_best_foil_logprob_margin"
          or es.get("baseline_correction") != "selected_minus_control_margin_drop"
          or es.get("target_definition") != "fixed_gold_choice_from_annotation"
          or es.get("margin_definition") != "gold_label_logprob_minus_max_foil_label_logprob_same_prompt"): f.add("estimand_fields_invalid")
    computed_estimand_sha = compute_primary_estimand_sha256(value)
    if computed_estimand_sha is None or not isinstance(es, dict) or es.get("primary_estimand_sha256") != computed_estimand_sha:
        f.add("primary_estimand_hash_mismatch")
    iv = value.get("intervention"); ik = {"kind", "selector_sha256", "control_sha256", "mask_transform_sha256", "minimum_selected_frame_count", "selected_condition", "control_condition", "selected_unit", "control_matching"}
    if not _keys(iv, ik): f.add("intervention_schema_invalid")
    elif (iv.get("kind") != "claim_linked_evidence_frame_mask" or not _sha(iv.get("selector_sha256")) or not _sha(iv.get("control_sha256")) or not _sha(iv.get("mask_transform_sha256"))
          or not _text(iv.get("selected_condition")) or not _text(iv.get("control_condition"))
          or type(iv.get("minimum_selected_frame_count")) is not int or iv.get("minimum_selected_frame_count") != 1
          or iv.get("selected_condition") == iv.get("control_condition")
          or iv.get("selected_unit") != "sampled_frames_overlapping_reviewed_claim_evidence_interval"
          or iv.get("control_matching") != "equal_count_minimum_total_temporal_rank_distance_lexicographic_tie_break"): f.add("intervention_fields_invalid")
    an = value.get("analysis"); ak = {"primary_contrast_count", "primary_multiplicity", "primary_ci_method", "secondary_families", "secondary_adjustment", "secondary_hypothesis_manifest_sha256", "inference_plan_sha256", "sesoi"}
    if not _keys(an, ak): f.add("analysis_schema_invalid")
    elif (type(an.get("primary_contrast_count")) is not int or an.get("primary_contrast_count") != 1
          or an.get("primary_multiplicity") != "none_single_primary_contrast"
          or an.get("primary_ci_method") != "two_sided_student_t_df_n_minus_1"
          or an.get("secondary_families") != ["baseline_correctness_strata", "question_type_strata"]
          or an.get("secondary_adjustment") != "holm_within_each_predeclared_family"
          or not _sha(an.get("secondary_hypothesis_manifest_sha256"))
          or not _sha(an.get("inference_plan_sha256"))):
        f.add("analysis_fields_invalid")
    else:
        computed_inference_sha = compute_inference_plan_sha256(value)
        if computed_inference_sha is None or an.get("inference_plan_sha256") != computed_inference_sha:
            f.add("inference_plan_hash_mismatch")
        if isinstance(dep, dict) and dep.get("method_status") == "resolved" and dep.get("method") != an.get("primary_ci_method"):
            f.add("dependence_plan_method_mismatch")
        sesoi = an.get("sesoi")
        sk = {"status", "value_nat", "decision_rule", "estimand_sha256", "rationale_category",
              "inference_plan_sha256",
              "rationale_sha256", "fixed_at_utc", "review_status", "review_timing",
              "independent_review_sha256", "independent_reviewed_at_utc",
              "outcome_access_status", "outcome_access_record_sha256"}
        if not _keys(sesoi, sk): f.add("sesoi_schema_invalid")
        elif sesoi.get("status") == "unresolved":
            if any(sesoi.get(k) is not None for k in sk - {"status"}): f.add("sesoi_unresolved_fields_not_null")
        elif sesoi.get("status") == "preoutcome_fixed":
            if (not _finite(sesoi.get("value_nat")) or sesoi["value_nat"] <= 0
                    or sesoi.get("decision_rule") != "primary_95pct_ci_lower_gt_sesoi"
                    or not _sha(sesoi.get("estimand_sha256"))
                    or not isinstance(es, dict) or sesoi.get("estimand_sha256") != es.get("primary_estimand_sha256")
                    or sesoi.get("inference_plan_sha256") != an.get("inference_plan_sha256")
                    or not isinstance(inference, dict) or inference.get("target_kind") == "unresolved"
                    or not isinstance(dep, dict) or dep.get("method_status") != "resolved"
                    or not isinstance(sesoi.get("rationale_category"), str)
                    or sesoi.get("rationale_category") not in {"annotation_granularity", "output_reproducibility", "measurement_interpretability", "other"}
                    or not _sha(sesoi.get("rationale_sha256"))
                    or not isinstance(sesoi.get("fixed_at_utc"), str) or not UTC_TIMESTAMP.fullmatch(sesoi["fixed_at_utc"])
                    or not _valid_utc_timestamp(sesoi["fixed_at_utc"])
                    or sesoi.get("review_status") != "completed"
                    or sesoi.get("review_timing") != "before_scoring"
                    or not _sha(sesoi.get("independent_review_sha256"))
                    or not isinstance(sesoi.get("independent_reviewed_at_utc"), str)
                    or not UTC_TIMESTAMP.fullmatch(sesoi["independent_reviewed_at_utc"])
                    or not _valid_utc_timestamp(sesoi["independent_reviewed_at_utc"])
                    or datetime.strptime(sesoi["independent_reviewed_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
                       > datetime.strptime(sesoi["fixed_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
                    or sesoi.get("outcome_access_status") != "not_accessed_at_freeze"
                    or not _sha(sesoi.get("outcome_access_record_sha256"))):
                f.add("sesoi_fixed_fields_invalid")
        else: f.add("sesoi_status_invalid")
    rt = value.get("runtime"); tk = {"seed", "source_commit", "execution_authorization"}
    if not _keys(rt, tk): f.add("runtime_schema_invalid")
    elif type(rt.get("seed")) is not int or not isinstance(rt.get("source_commit"), str) or GIT_COMMIT.fullmatch(rt["source_commit"]) is None or rt.get("execution_authorization") is not False: f.add("runtime_fields_invalid")
    return _summary(f, input_sha256)

def _summary(f, digest):
    failures = set(f)
    if digest is not None and not _sha(digest):
        failures.add("input_sha256_invalid")
        digest = None
    return {"schema_version": 10, "structurally_valid": not failures, "failure_codes": sorted(failures),
            "semantic_validation": "not_machine_verifiable", "execution_authorization": "not_granted",
            **({"input_sha256": digest} if digest else {})}

def _valid_utc_timestamp(value):
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False

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
