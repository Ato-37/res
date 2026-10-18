"""Canonical, privacy-minimal runtime descriptor for QTA freeze inputs.

The descriptor describes execution settings only. It is not remote attestation,
model provenance, or proof that a forward used the reported runtime.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
PURPOSE = "qta_sampler_model_runtime_descriptor_non_authorizing_v1"
_KEYS = {
    "schema_version", "purpose", "python_version", "torch_version",
    "cuda_runtime_version", "transformers_version", "bitsandbytes_version",
    "device_type", "precision", "quantization", "deterministic_algorithms",
    "allow_tf32",
}
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")


def _fail(code: str) -> None:
    raise ValueError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def validate_qta_sampler_model_runtime_descriptor(raw: bytes) -> dict[str, Any]:
    """Validate exact canonical descriptor bytes; no authenticity assertion."""
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= 16_384:
        _fail("sampler_runtime_descriptor_bytes_invalid")

    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                _fail("sampler_runtime_descriptor_duplicate_key")
            output[key] = value
        return output

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=unique,
                           parse_constant=lambda _x: _fail(
                               "sampler_runtime_descriptor_nonfinite"))
    except ValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("sampler_runtime_descriptor_json_invalid")
    if (not isinstance(value, dict) or set(value) != _KEYS
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
            or value.get("purpose") != PURPOSE):
        _fail("sampler_runtime_descriptor_schema_invalid")
    for key in ("python_version", "torch_version", "transformers_version"):
        version = value.get(key)
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            _fail(f"sampler_runtime_descriptor_{key}_invalid")
    for key in ("cuda_runtime_version", "bitsandbytes_version"):
        version = value.get(key)
        if version is not None and (not isinstance(version, str)
                                    or not _VERSION.fullmatch(version)):
            _fail(f"sampler_runtime_descriptor_{key}_invalid")
    if value.get("device_type") not in {"cpu", "cuda"}:
        _fail("sampler_runtime_descriptor_device_type_invalid")
    if value.get("precision") not in {"bfloat16", "float32"}:
        _fail("sampler_runtime_descriptor_precision_invalid")
    if value.get("quantization") not in {"none", "4bit_nf4"}:
        _fail("sampler_runtime_descriptor_quantization_invalid")
    if (type(value.get("deterministic_algorithms")) is not bool
            or type(value.get("allow_tf32")) is not bool):
        _fail("sampler_runtime_descriptor_runtime_flags_invalid")
    if value["quantization"] == "4bit_nf4" and value["device_type"] != "cuda":
        _fail("sampler_runtime_descriptor_quantization_device_mismatch")
    if value["quantization"] == "4bit_nf4" and value["bitsandbytes_version"] is None:
        _fail("sampler_runtime_descriptor_quantization_dependency_missing")
    if _canonical(value) != raw:
        _fail("sampler_runtime_descriptor_not_canonical")
    return {"runtime_descriptor_valid": True,
            "runtime_descriptor_sha256": hashlib.sha256(raw).hexdigest(),
            "runtime_descriptor_semantics_authenticated": False,
            "forward_source_authenticated": False,
            "execution_authorization": "not_granted"}
