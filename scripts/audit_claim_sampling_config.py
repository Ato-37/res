#!/usr/bin/env python3
"""Check that exact sampling-config bytes match a schema-v13 protocol declaration.

This structural helper never inspects media or authorizes inference. It binds
the raw configuration digest and compares every declared sampling value with
strict JSON types (so booleans cannot stand in for integers).
"""
from __future__ import annotations

import hashlib
from typing import Any


CONFIG_FIELDS = (
    "policy", "uses_claim_for_selection", "num_frames", "frame_size",
    "max_pixels", "cap_pixels_per_frame",
)


def audit(protocol: Any, config: Any, config_raw: bytes) -> dict:
    failures: set[str] = set()
    sampling = protocol.get("sampling") if isinstance(protocol, dict) else None
    if (not isinstance(protocol, dict) or type(protocol.get("schema_version")) is not int
            or protocol.get("schema_version") != 13 or not isinstance(sampling, dict)):
        failures.add("protocol_sampling_declaration_invalid")
    if not isinstance(config, dict) or set(config) != set(CONFIG_FIELDS):
        failures.add("sampling_config_schema_invalid")
    if not isinstance(config_raw, bytes):
        failures.add("sampling_config_bytes_invalid")
    elif isinstance(sampling, dict) and isinstance(sampling.get("config_sha256"), str):
        if hashlib.sha256(config_raw).hexdigest() != sampling["config_sha256"]:
            failures.add("sampling_config_hash_mismatch")
    else:
        failures.add("sampling_config_hash_invalid")
    if isinstance(sampling, dict) and isinstance(config, dict) and set(config) == set(CONFIG_FIELDS):
        for field in CONFIG_FIELDS:
            expected, actual = sampling.get(field), config.get(field)
            if type(actual) is not type(expected) or actual != expected:
                failures.add("sampling_config_values_mismatch")
                break
    return {
        "schema_version": 1,
        "purpose": "claim_sampling_config_binding_audit",
        "structural_status": "passed" if not failures else "failed",
        "failure_codes": sorted(failures),
        "execution_authorization": "not_granted",
    }
