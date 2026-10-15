"""Fail closed while every current preflight is non-authorizing.

The current claim-mask cross-gate emits ``blocked``/``not_granted`` by design.
This consumer prevents a missing or malformed report from being treated as
permission. A report does not authenticate its declared human or legal evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def require_execution_preflight(path: Path) -> None:
    """Fail closed: current schema v12 has no authenticated executable report."""
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("execution preflight report is missing")
    raw = path.read_bytes()
    try:
        report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ValueError("non_finite_json")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("execution preflight report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("execution preflight report must be an object")
    if (isinstance(report, dict) and type(report.get("schema_version")) is int
            and report.get("schema_version") == 1
            and report.get("purpose") == "claim_mask_structural_cross_gate_preflight"):
        raise ValueError(
            "schema-v12 cross-gate deliberately never grants execution; "
            "an authenticated, run-bound future gate is not implemented")
    raise ValueError("unsupported execution preflight schema; refusing inference")
