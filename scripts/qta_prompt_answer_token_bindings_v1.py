"""Canonical private prompt/answer-token bindings for QTA phase-1 receipts.

The validator independently rebuilds the fixed QA prompt from ephemeral source
text/frame inputs, retokenizes it, derives A-E continuations, and hashes the live
processor input IDs. This is structural recomputation, not authentication of the
processor/tokenizer artifacts or their provenance and never grants authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

ROW_COUNT = 3
PURPOSE = "qta_prompt_answer_token_bindings_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = {"record_token", "question_sha256", "options_sha256", "prompt_sha256",
             "tokenizer_sha256", "prompt_token_ids_sha256",
             "choice_token_binding_sha256", "input_ids_sha256", "answer_token_ids"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(code)
    return value


def _token_ids(tokenizer: Any, text: str, code: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ValueError("prompt_answer_tokenizer_invalid")
    try:
        values = encode(text, add_special_tokens=False)
    except Exception:
        raise ValueError(code) from None
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value < 0 for value in values)):
        raise ValueError("prompt_answer_token_ids_invalid")
    return values


def _input_ids(batch: Any) -> list[int]:
    if not isinstance(batch, Mapping) or "input_ids" not in batch:
        raise ValueError("prompt_answer_live_batch_input_ids_missing")
    values = batch["input_ids"]
    try:
        if callable(getattr(values, "detach", None)):
            values = values.detach().cpu().tolist()
        elif callable(getattr(values, "tolist", None)):
            values = values.tolist()
    except Exception:
        raise ValueError("prompt_answer_live_batch_input_ids_invalid") from None
    if (not isinstance(values, list) or len(values) != 1
            or not isinstance(values[0], list) or not values[0]
            or any(type(value) is not int or value < 0 for value in values[0])):
        raise ValueError("prompt_answer_live_batch_input_ids_invalid")
    return values[0]


def _rgb_hashes(frames: Any) -> list[str]:
    if not isinstance(frames, (list, tuple)) or len(frames) != 32:
        raise ValueError("prompt_answer_frame_material_invalid")
    result = []
    try:
        for frame in frames:
            rgb = frame.convert("RGB").tobytes()
            if not isinstance(rgb, bytes) or not rgb:
                raise ValueError
            result.append(hashlib.sha256(rgb).hexdigest())
    except Exception:
        raise ValueError("prompt_answer_frame_material_invalid") from None
    return result


def _reconstruct_prompt(processor: Any, source: Any) -> tuple[str, str, str, str]:
    """Rebuild fixed prompt independently; raw inputs are caller-owned/transient."""
    if (not isinstance(source, Mapping) or set(source) != {
            "review_token", "question", "choices", "frames", "sample_fps"}):
        raise ValueError("prompt_answer_reconstruction_input_invalid")
    token, question, choices = (source["review_token"], source["question"],
                                source["choices"])
    fps = source["sample_fps"]
    frames = source["frames"]
    if (not isinstance(token, str) or not token
            or not isinstance(question, str) or not question.strip()
            or not isinstance(choices, (list, tuple)) or len(choices) != 5
            or any(not isinstance(value, str) or not value.strip() for value in choices)
            or not isinstance(fps, (list, tuple)) or len(fps) != 2
            or any(type(value) is not int or value <= 0 for value in fps)
            or not isinstance(frames, (list, tuple)) or not frames):
        raise ValueError("prompt_answer_reconstruction_input_invalid")
    options = "\n".join(f"{label}. {choice}"
                        for label, choice in zip("ABCDE", choices, strict=True))
    prompt_text = (f"{question}\nOptions:\n{options}\n"
                   "Choose the best answer. Respond with one letter only.\nAnswer:")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "sample_fps": fps[0] / fps[1]},
        {"type": "text", "text": prompt_text},
    ]}]
    render = getattr(processor, "apply_chat_template", None)
    if not callable(render):
        raise ValueError("prompt_answer_processor_renderer_invalid")
    try:
        rendered = render(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        raise ValueError("prompt_answer_processor_render_failed") from None
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("prompt_answer_processor_render_invalid")
    return (token, hashlib.sha256(question.encode("utf-8")).hexdigest(),
            hashlib.sha256(_canonical(list(choices))).hexdigest(), rendered)


def _row(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
        raise ValueError("prompt_answer_row_schema_invalid")
    row = {key: _sha(value[key], f"prompt_answer_{key}_invalid")
           for key in sorted(_ROW_KEYS - {"answer_token_ids"})}
    ids = value["answer_token_ids"]
    if (not isinstance(ids, (list, tuple)) or len(ids) != 5
            or any(type(item) is not int or item < 0 for item in ids)
            or len(set(ids)) != 5):
        raise ValueError("prompt_answer_token_ids_invalid")
    row["answer_token_ids"] = list(ids)
    return row


def build_qta_prompt_answer_token_bindings(candidate_rows: Any) -> bytes:
    """Build canonical privacy-minimal bytes from actual processor candidates."""
    if not isinstance(candidate_rows, (list, tuple)) or len(candidate_rows) != ROW_COUNT:
        raise ValueError("prompt_answer_three_rows_required")
    rows = []
    for candidate in candidate_rows:
        if not isinstance(candidate, Mapping):
            raise ValueError("prompt_answer_candidate_row_invalid")
        token = candidate.get("review_token")
        if not isinstance(token, str) or not token:
            raise ValueError("prompt_answer_review_token_invalid")
        try:
            record_token = hashlib.sha256(token.encode("ascii", errors="strict")).hexdigest()
        except UnicodeError:
            raise ValueError("prompt_answer_review_token_invalid") from None
        rows.append(_row({
            "record_token": record_token,
            "question_sha256": candidate.get("question_sha256"),
            "options_sha256": candidate.get("options_sha256"),
            "prompt_sha256": candidate.get("rendered_prompt_sha256"),
            "tokenizer_sha256": candidate.get("tokenizer_sha256"),
            "prompt_token_ids_sha256": candidate.get("prompt_token_ids_sha256"),
            "choice_token_binding_sha256": candidate.get("choice_token_binding_sha256"),
            "input_ids_sha256": candidate.get("input_ids_sha256"),
            "answer_token_ids": candidate.get("answer_token_ids"),
        }))
    if len({row["record_token"] for row in rows}) != ROW_COUNT:
        raise ValueError("prompt_answer_record_tokens_not_unique")
    return _canonical({"schema_version": 1, "purpose": PURPOSE,
        "status": "candidate_only", "execution_authorization": "not_granted",
        "rows": rows})


def validate_qta_prompt_answer_token_bindings(raw: bytes, *, expected_rows: Any,
        processor_rows: Any, live_processor_rows: Any,
        sampler_rows: Any, prompt_reconstruction_inputs: Any,
        processor: Any, tokenizer: Any) -> dict[str, Any]:
    """Independently reconstruct and cross-check every signed prompt/token pin.

    ``prompt_reconstruction_inputs`` are ephemeral: ordered mappings with
    review_token/question/choices/frames/sample_fps. They are never returned or
    serialized. Processor/tokenizer authenticity remains out of scope.
    """
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= 1_000_000:
        raise ValueError("prompt_answer_bytes_invalid")
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("prompt_answer_duplicate_key")
            obj[key] = value
        return obj
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=lambda _x: (_ for _ in ()).throw(
                             ValueError("prompt_answer_nonfinite")))
    except ValueError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("prompt_answer_json_invalid") from None
    if (not isinstance(obj, dict)
            or set(obj) != {"schema_version", "purpose", "status",
                            "execution_authorization", "rows"}
            or type(obj["schema_version"]) is not int or obj["schema_version"] != 1
            or obj["purpose"] != PURPOSE or obj["status"] != "candidate_only"
            or obj["execution_authorization"] != "not_granted"
            or not isinstance(obj["rows"], list) or len(obj["rows"]) != ROW_COUNT
            or not isinstance(expected_rows, (list, tuple)) or len(expected_rows) != ROW_COUNT
            or not isinstance(processor_rows, (list, tuple)) or len(processor_rows) != ROW_COUNT
            or not isinstance(live_processor_rows, (list, tuple)) or len(live_processor_rows) != ROW_COUNT
            or not isinstance(sampler_rows, (list, tuple)) or len(sampler_rows) != ROW_COUNT
            or not isinstance(prompt_reconstruction_inputs, (list, tuple))
            or len(prompt_reconstruction_inputs) != ROW_COUNT
            or processor is None or tokenizer is None
            or not callable(getattr(tokenizer, "encode", None))
            or _canonical(obj) != raw):
        raise ValueError("prompt_answer_manifest_schema_or_canonical_invalid")
    processor_tokenizer = getattr(processor, "tokenizer", None)
    if (processor_tokenizer is None or processor_tokenizer is not tokenizer
            or not callable(getattr(processor, "apply_chat_template", None))):
        raise ValueError("prompt_answer_processor_tokenizer_mismatch")
    clean = [_row(row) for row in obj["rows"]]
    if len({row["record_token"] for row in clean}) != ROW_COUNT:
        raise ValueError("prompt_answer_record_tokens_not_unique")
    for binding, source, processor_pin, live, sampler, reconstruction in zip(
            clean, expected_rows, processor_rows, live_processor_rows, sampler_rows,
            prompt_reconstruction_inputs, strict=True):
        if (not isinstance(source, Mapping) or not isinstance(processor_pin, Mapping)
                or not isinstance(live, Mapping) or not isinstance(sampler, Mapping)
                or not isinstance(live.get("batch"), Mapping)):
            raise ValueError("prompt_answer_source_or_processor_join_mismatch")
        if not isinstance(reconstruction, Mapping):
            raise ValueError("prompt_answer_reconstruction_input_invalid")
        frames_before = _rgb_hashes(reconstruction.get("frames"))
        token, question_sha, options_sha, rendered = _reconstruct_prompt(processor, reconstruction)
        try:
            token_digest = hashlib.sha256(token.encode("ascii", errors="strict")).hexdigest()
        except UnicodeError:
            raise ValueError("prompt_answer_review_token_invalid") from None
        fps = reconstruction.get("sample_fps") if isinstance(reconstruction, Mapping) else None
        expected_fps = sampler.get("effective_sample_fps")
        frames = reconstruction.get("frames") if isinstance(reconstruction, Mapping) else None
        frame_hashes = _rgb_hashes(frames)
        try:
            sampler_sha = hashlib.sha256(_canonical(dict(sampler))).hexdigest()
        except (TypeError, ValueError, OverflowError):
            raise ValueError("prompt_answer_sampler_material_invalid") from None
        if (not isinstance(expected_fps, (list, tuple)) or len(expected_fps) != 2
                or list(fps) != list(expected_fps)
                or frames_before != frame_hashes
                or sampler.get("source_sha256") != source.get("source_video_sha256")
                or sampler_sha != source.get("sampler_row_sha256")
                or sampler.get("sampled_frame_count") != len(frames)
                or sampler.get("sampled_rgb_sha256") != frame_hashes):
            raise ValueError("prompt_answer_sampler_frame_or_fps_mismatch")
        tokenizer_sha = source.get("tokenizer_sha256")
        _sha(tokenizer_sha, "prompt_answer_tokenizer_pin_invalid")
        prompt_ids = _token_ids(tokenizer, rendered, "prompt_answer_prompt_tokenization_failed")
        answer_ids = []
        for label in "ABCDE":
            extended = _token_ids(tokenizer, rendered + label,
                                  "prompt_answer_choice_tokenization_failed")
            if extended[:len(prompt_ids)] != prompt_ids:
                raise ValueError("prompt_answer_prompt_prefix_retokenized")
            suffix = extended[len(prompt_ids):]
            if len(suffix) != 1:
                raise ValueError("prompt_answer_choice_not_single_token")
            answer_ids.append(suffix[0])
        if len(set(answer_ids)) != 5:
            raise ValueError("prompt_answer_choice_ids_not_distinct")
        prompt_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        prompt_ids_sha = hashlib.sha256(json.dumps(
            prompt_ids, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        choice_body = {"labels": list("ABCDE"), "prompt_sha256": prompt_sha,
            "tokenizer_sha256": tokenizer_sha,
            "prompt_token_ids_sha256": prompt_ids_sha,
            "answer_token_ids": answer_ids}
        choice_sha = hashlib.sha256(_canonical(choice_body)).hexdigest()
        batch_ids = _input_ids(live["batch"])
        input_sha = hashlib.sha256(_canonical(batch_ids)).hexdigest()
        video_token_id = getattr(processor, "video_token_id", None)
        if type(video_token_id) is not int:
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                try:
                    video_token_id = convert("<|video_pad|>")
                except Exception:
                    video_token_id = None
        if type(video_token_id) is not int or video_token_id < 0:
            raise ValueError("prompt_answer_video_token_id_missing")
        encoded_prompt = prompt_ids
        if encoded_prompt.count(video_token_id) != 1:
            raise ValueError("prompt_answer_video_placeholder_invalid")
        placeholder = encoded_prompt.index(video_token_id)
        occurrences = [i for i, value in enumerate(batch_ids) if value == video_token_id]
        if (not occurrences or occurrences != list(range(occurrences[0], occurrences[-1] + 1))
                or batch_ids != (encoded_prompt[:placeholder]
                    + [video_token_id] * len(occurrences)
                    + encoded_prompt[placeholder + 1:])):
            raise ValueError("prompt_answer_live_processor_ids_mismatch")
        if (token != source.get("review_token") or token != live.get("review_token")
                or token_digest != binding["record_token"]
                or question_sha != source.get("question_sha256")
                or options_sha != source.get("options_sha256")
                or binding["question_sha256"] != question_sha
                or binding["options_sha256"] != options_sha
                or prompt_sha != binding["prompt_sha256"]
                or prompt_sha != processor_pin.get("prompt_sha256")
                or tokenizer_sha != binding["tokenizer_sha256"]
                or tokenizer_sha != processor_pin.get("tokenizer_sha256")
                or prompt_ids_sha != binding["prompt_token_ids_sha256"]
                or choice_sha != binding["choice_token_binding_sha256"]
                or answer_ids != binding["answer_token_ids"]
                or input_sha != binding["input_ids_sha256"]
                or input_sha != processor_pin.get("input_ids_sha256")):
            raise ValueError("prompt_answer_independent_recomputation_mismatch")
    return {"status": "prompt_token_declarations_independently_recomputed_unvalidated",
            "validated_rows": ROW_COUNT,
            "prompt_rendering_independently_reconstructed": True,
            "prompt_renderer_origin_authenticated": False,
            "tokenizer_authenticity_verified": False,
            "processor_authenticity_verified": False,
            "model_load_permitted": False,
            "scoring_authorized": False,
            "execution_authorization": "not_granted"}
