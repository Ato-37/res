#!/usr/bin/env python3
"""Pair raw HTML5 player observations with an externally frozen PTS anchor plan.

This emits only unvalidated calibration inputs. It cannot prove capture
provenance, reviewer identity, anchor correctness, or grant scoring authority.
All inputs/outputs contain private opaque tokens and must remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

from private_output_paths import validate_private_output

SHA256 = re.compile(r"^[a-f0-9]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{24}$")
OBSERVATION_PURPOSE = "local_unvalidated_html5_player_clock_observations"
PLAN_PURPOSE = "predeclared_html5_clock_pts_anchor_plan"
PRESENTED_FRAME_KEYS = {"media_time_seconds", "presented_frames", "presentation_time_ms",
                        "expected_display_time_ms", "width", "height", "callback_now_ms"}


def _strict_json(raw: bytes) -> Any:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError("nonfinite_json")))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    except ValueError:
        return False


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field + "_invalid")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(field + "_invalid")
    return result


def prepare_capture_bundle(observations_raw: bytes, index_raw: bytes,
                           plan_raw: bytes, expected_plan_sha256: str) -> dict:
    """Validate exact observation/plan bindings and prepare raw per-video inputs."""
    if any(not isinstance(x, bytes) or len(x) > 2_000_000
           for x in (observations_raw, index_raw, plan_raw)):
        raise ValueError("input_bytes_invalid")
    if not isinstance(expected_plan_sha256, str) or not SHA256.fullmatch(expected_plan_sha256):
        raise ValueError("expected_plan_hash_invalid")
    if _sha(plan_raw) != expected_plan_sha256:
        raise ValueError("frozen_plan_hash_mismatch")
    obs, index, plan = map(_strict_json, (observations_raw, index_raw, plan_raw))
    index_keys = {"schema_version", "purpose", "payload_sha256", "created_at_utc",
                  "status", "execution_authorization"}
    if (not isinstance(index, dict) or set(index) != index_keys
            or type(index.get("schema_version")) is not int or index["schema_version"] != 1
            or index.get("purpose") != "unvalidated_html5_clock_observation_integrity_index"
            or index.get("status") != "raw_observations_only"
            or index.get("execution_authorization") != "not_granted"
            or index.get("payload_sha256") != _sha(observations_raw)):
        raise ValueError("observation_index_invalid")
    obs_keys = {"schema_version", "purpose", "status", "player_clock",
                "presented_frame_metadata", "decoder_frame_identity",
                "created_at_utc", "browser_user_agent", "anchor_id_validation",
                "predeclared_plan_binding", "fit_heldout_split", "decoder_pts_pairing",
                "execution_authorization", "records"}
    if (not isinstance(obs, dict) or set(obs) != obs_keys
            or type(obs.get("schema_version")) is not int or obs["schema_version"] != 2
            or obs.get("purpose") != OBSERVATION_PURPOSE
            or obs.get("status") != "raw_player_clock_observations_only"
            or obs.get("player_clock") != "html5_video_currentTime_seconds"
            or obs.get("presented_frame_metadata") != "requestVideoFrameCallback_when_available_not_decoder_matched"
            or obs.get("decoder_frame_identity") != "not_verified"
            or obs.get("anchor_id_validation") != "format_and_per_video_uniqueness_only_unverified"
            or obs.get("predeclared_plan_binding") != "not_checked"
            or obs.get("fit_heldout_split") != "not_assigned"
            or obs.get("decoder_pts_pairing") != "not_provided"
            or obs.get("execution_authorization") != "not_granted"
            or not isinstance(obs.get("browser_user_agent"), str)
            or not obs["browser_user_agent"].strip()
            or not _utc(obs.get("created_at_utc"))
            or not isinstance(obs.get("records"), list)):
        raise ValueError("raw_observation_schema_invalid")
    plan_keys = {"schema_version", "purpose", "status", "execution_authorization", "records"}
    if (not isinstance(plan, dict) or set(plan) != plan_keys
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("purpose") != PLAN_PURPOSE
            or plan.get("status") != "predeclared_frozen"
            or plan.get("execution_authorization") != "not_granted"
            or not isinstance(plan.get("records"), list)):
        raise ValueError("anchor_plan_schema_invalid")
    if len(obs["records"]) != 3 or len(plan["records"]) != 3:
        raise ValueError("record_count_invalid")

    captures = []
    seen_tokens = set()
    for observed, planned in zip(obs["records"], plan["records"], strict=True):
        if (not isinstance(observed, dict) or set(observed) != {"review_token", "events"}
                or not isinstance(planned, dict)
                or set(planned) != {"review_token", "source_media_sha256", "fit_anchors", "heldout_anchors"}):
            raise ValueError("record_schema_invalid")
        token = observed.get("review_token")
        if (not isinstance(token, str) or not TOKEN.fullmatch(token)
                or token != planned.get("review_token") or token in seen_tokens):
            raise ValueError("record_identity_invalid")
        seen_tokens.add(token)
        media_sha = planned.get("source_media_sha256")
        if not isinstance(media_sha, str) or not SHA256.fullmatch(media_sha):
            raise ValueError("planned_media_hash_invalid")
        fit, heldout = planned.get("fit_anchors"), planned.get("heldout_anchors")
        if (not isinstance(fit, list) or len(fit) < 2
                or not isinstance(heldout, list) or not heldout):
            raise ValueError("planned_anchor_counts_invalid")
        plan_by_id = {}
        for anchors in (fit, heldout):
            for anchor in anchors:
                if not isinstance(anchor, dict) or set(anchor) != {"anchor_id", "decoder_pts_seconds"}:
                    raise ValueError("planned_anchor_schema_invalid")
                anchor_id = anchor.get("anchor_id")
                pts = anchor.get("decoder_pts_seconds")
                if (not isinstance(anchor_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", anchor_id)
                        or anchor_id in plan_by_id or not isinstance(pts, str)):
                    raise ValueError("planned_anchor_invalid")
                # Fraction-compatible string validation is repeated by the calibrator.
                from fractions import Fraction
                try:
                    if Fraction(pts) < 0:
                        raise ValueError("planned_anchor_invalid")
                except (ValueError, ZeroDivisionError, OverflowError):
                    raise ValueError("planned_anchor_invalid") from None
                plan_by_id[anchor_id] = pts
        if set(a["anchor_id"] for a in fit) & set(a["anchor_id"] for a in heldout):
            raise ValueError("fit_heldout_plan_overlap")
        events = observed.get("events")
        if not isinstance(events, list) or len(events) != len(plan_by_id):
            raise ValueError("observed_anchor_count_mismatch")
        event_by_id = {}
        sequences = set()
        for event in events:
            event_keys = {"anchor_id", "sequence", "html5_current_time_seconds",
                          "request_video_frame_callback_metadata", "frame_match_status",
                          "performance_now_ms", "media_duration_seconds", "playback_rate",
                          "muted", "volume", "text_tracks_disabled", "paused", "seeking", "ready_state",
                          "current_src_is_staged_media", "visibility_state", "captured_at_utc"}
            if not isinstance(event, dict) or set(event) != event_keys:
                raise ValueError("observed_event_schema_invalid")
            anchor_id = event.get("anchor_id")
            playback_rate = _number(event.get("playback_rate"), "playback_rate", positive=True)
            volume = _number(event.get("volume"), "volume")
            if (not isinstance(anchor_id, str) or anchor_id not in plan_by_id
                    or anchor_id in event_by_id or event.get("current_src_is_staged_media") is not True
                    or playback_rate != 1 or event.get("muted") is not True
                    or volume != 0 or event.get("text_tracks_disabled") is not True
                    or event.get("paused") is not True or event.get("seeking") is not False
                    or event.get("frame_match_status") != "last_presented_frame_metadata_unverified"
                    or not isinstance(event.get("visibility_state"), str)
                    or event.get("visibility_state") != "visible"
                    or type(event.get("ready_state")) is not int or event["ready_state"] < 2
                    or not _utc(event.get("captured_at_utc"))):
                raise ValueError("observed_event_state_invalid")
            if type(event.get("sequence")) is not int or event["sequence"] < 1:
                raise ValueError("observed_event_sequence_invalid")
            if event["sequence"] in sequences:
                raise ValueError("observed_event_sequence_invalid")
            sequences.add(event["sequence"])
            current = _number(event.get("html5_current_time_seconds"), "player_time")
            duration = _number(event.get("media_duration_seconds"), "duration", positive=True)
            _number(event.get("performance_now_ms"), "performance_now")
            if current < 0 or current > duration:
                raise ValueError("observed_player_time_out_of_range")
            frame = event.get("request_video_frame_callback_metadata")
            if not isinstance(frame, dict) or set(frame) != PRESENTED_FRAME_KEYS:
                raise ValueError("presented_frame_metadata_invalid")
            media_time = _number(frame.get("media_time_seconds"), "presented_media_time")
            presented_frames = frame.get("presented_frames")
            width, height = frame.get("width"), frame.get("height")
            if (media_time < 0 or media_time > duration
                    or type(presented_frames) is not int or presented_frames < 1
                    or type(width) is not int or width < 1
                    or type(height) is not int or height < 1):
                raise ValueError("presented_frame_metadata_invalid")
            _number(frame.get("presentation_time_ms"), "presentation_time")
            _number(frame.get("expected_display_time_ms"), "expected_display_time")
            _number(frame.get("callback_now_ms"), "callback_now")
            event_by_id[anchor_id] = {"anchor_id": anchor_id,
                                      "decoder_pts_seconds": plan_by_id[anchor_id],
                                      "html5_media_time_seconds": str(media_time)}
        if set(event_by_id) != set(plan_by_id):
            raise ValueError("observed_anchor_plan_mismatch")
        if sequences != set(range(1, len(events) + 1)):
            raise ValueError("observed_event_sequence_invalid")
        captures.append({
            "review_token": token,
            "capture": {
                "schema_version": 2,
                "purpose": "local_html5_player_clock_anchor_capture",
                "player_clock": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
                "browser_user_agent": obs["browser_user_agent"],
                "current_src_matches_staged_media": True,
                "source_media_sha256": media_sha,
                "playback_rate": "1",
                "muted": True,
                "text_tracks_disabled": True,
                "fit_anchors": [event_by_id[a["anchor_id"]] for a in fit],
                "heldout_anchors": [event_by_id[a["anchor_id"]] for a in heldout],
            },
        })
    return {
        "schema_version": 1,
        "purpose": "unvalidated_html5_clock_calibration_input_bundle",
        "status": "raw_observations_paired_with_declared_plan_unvalidated",
        "observations_sha256": _sha(observations_raw),
        "observation_index_sha256": _sha(index_raw),
        "anchor_plan_sha256": expected_plan_sha256,
        "captures": captures,
        "execution_authorization": "not_granted",
        "limitations": [
            "the plan digest must be independently frozen before capture; this tool cannot authenticate its timing",
            "requestVideoFrameCallback metadata may be stale at capture time; paused/currentTime state does not prove which compositor frame the reviewer selected",
            "the tool cannot authenticate browser telemetry, visual anchor correspondence, decoder-frame identity, or reviewer identity",
            "calibration results remain empirical held-out diagnostics and do not authorize scoring",
        ],
    }


def _write_private_exclusive(path: Path, obj: dict) -> None:
    raw = (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            os.fchmod(fd, 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        for path in (args.observations, args.index, args.plan):
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("private_input_file_invalid")
        output = validate_private_output(args.output, repo)
        bundle = prepare_capture_bundle(args.observations.read_bytes(), args.index.read_bytes(),
                                        args.plan.read_bytes(), args.expected_plan_sha256)
        _write_private_exclusive(output, bundle)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "capture_bundle_invalid")
    print(json.dumps({"status": bundle["status"], "video_count": len(bundle["captures"]),
                      "execution_authorization": bundle["execution_authorization"],
                      "output_mode": "0600", "private_output": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
