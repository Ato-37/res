#!/usr/bin/env python3
"""Build a private, outcome-blind local NExT-GQA ZIP candidate pool.

This CPU-only preparation tool extracts only candidates that pass the auxiliary
gold-interval pilot's interval and five-choice eligibility checks. It never
loads a model or prints candidate identities, paths, or QA text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 100 * 1024**3
SHA = re.compile(r"^[0-9a-f]{64}$")


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate_json_key")
        out[k] = v
    return out


def _json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("bad_json")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json") from exc


def _token(key: bytes, domain: str, parts: tuple[str, ...]) -> str:
    payload = json.dumps(["nextgqa-exposure-v1", domain, *parts], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def candidate_pair_sha256(video_id: str, qid: str) -> str:
    payload = json.dumps(["nextgqa-gold-interval-pair-v1", video_id, qid],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _video_key(value: str) -> str:
    p = [x for x in value.replace("\\", "/").split("/") if x]
    if p:
        p[-1] = p[-1].rsplit(".", 1)[0]
    return "/".join(p[-2:])


def _norm(value: str) -> str:
    import unicodedata
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _safe_member(name: str, is_dir: bool = False) -> bool:
    if "\\" in name or name.startswith("/") or "\x00" in name:
        return False
    if is_dir and name.endswith("/"):
        name = name[:-1]
    parts = name.split("/")
    return bool(name) and all(part not in ("", ".", "..") for part in parts)


def _secure_read(path: Path, *, secret: bool = False) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("input_unavailable_or_symlink") from exc
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("input_must_be_regular_file")
        if secret and stat.S_IMODE(st.st_mode) != 0o600:
            raise ValueError("secret_must_be_regular_mode_0600")
        raw = stream.read()
    return raw


def _preflight_file(path: Path, *, secret: bool = False) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("input_unavailable_or_symlink") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("input_must_be_regular_file")
        if secret and stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("secret_must_be_regular_mode_0600")


def _read_secret(path: Path) -> bytes:
    key = _secure_read(path, secret=True)
    if len(key) < 32:
        raise ValueError("secret_too_short")
    return key


def _load_index(raw: bytes):
    obj = _json(raw)
    if (not isinstance(obj, dict) or type(obj.get("schema_version")) is not int
            or obj.get("schema_version") != 1
            or obj.get("purpose") != "outcome_blind_prior_exposure_index"
            or obj.get("canonicalization_version") != "explicit-identifiers-v1-no-path-stems"
            or not isinstance(obj.get("sources"), list)
            or not any(isinstance(row, dict) and row.get("identity_namespace") == "nextgqa"
                       for row in obj["sources"])
            or not isinstance(obj.get("identities"), dict)):
        raise ValueError("prior_index_schema_invalid")
    ids = obj["identities"]
    names = {"dataset_video_hmac_sha256": "video", "media_sha256_hmac_sha256": "media",
             "video_question_pair_hmac_sha256": "pair"}
    ret = {}
    for field, kind in names.items():
        values = ids.get(field)
        if (not isinstance(values, list) or any(not isinstance(x, str) or not SHA.fullmatch(x) for x in values)
                or len(values) != len(set(values))):
            raise ValueError("prior_index_identities_invalid")
        ret[kind] = set(values)
    return ret


def _sample_paths(root: Path, pattern: str):
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("sample_manifest_dir_invalid")
    paths = sorted(root.glob(pattern))
    if not paths:
        raise ValueError("sample_manifest_pattern_empty")
    for path in paths:
        _preflight_file(path)
    excluded = set(); excluded_keys = set()
    digests = []
    for path in paths:
        raw = _secure_read(path); digests.append(hashlib.sha256(raw).hexdigest())
        value = _json(raw)
        rows = value if isinstance(value, list) else value.get("records") if isinstance(value, dict) else None
        if not isinstance(rows, list):
            raise ValueError("sample_manifest_shape_invalid")
        for row in rows:
            if isinstance(row, dict):
                for field in ("video", "video_path", "path"):
                    if isinstance(row.get(field), str):
                        normalized = row[field].replace("\\", "/").lstrip("./")
                        excluded.add(normalized)
                        if normalized:
                            excluded_keys.add(_video_key(normalized))
    return excluded, excluded_keys, digests


def _write_exclusive(path: Path, data: bytes):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.chmod(path, 0o600)


def _extract_member(zf, info, dest: Path, expected_sha: str,
                    max_member_bytes: int, remaining_total_bytes: int) -> int:
    h = hashlib.sha256(); written = 0
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            with zf.open(info) as src:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > max_member_bytes or written > remaining_total_bytes:
                        raise ValueError("zip_actual_extraction_size_limit")
                    h.update(block); out.write(block)
            out.flush(); os.fsync(out.fileno())
        if h.hexdigest() != expected_sha:
            raise ValueError("media_hash_changed_during_extraction")
        os.chmod(dest, 0o600)
        return written
    except Exception:
        try:
            dest.unlink()
        except OSError:
            pass
        raise


def _private_input(path: Path) -> None:
    _secure_read(path)


def build(zip_path: Path, csv_path: Path, gsub_path: Path, map_path: Path,
          sample_dir: Path, sample_pattern: str, index_path: Path | list[Path], secret_path: Path,
          output_dir: Path, *, max_member_bytes=MAX_MEMBER_BYTES,
          max_total_bytes=MAX_TOTAL_BYTES, expected_secret_sha256: str | None = None,
          dataset_split: str = "test", max_unique_media: int | None = None) -> dict:
    if dataset_split not in {"train", "val", "test"}:
        raise ValueError("dataset_split_invalid")
    if (max_unique_media is not None
            and (type(max_unique_media) is not int or max_unique_media < 1)):
        raise ValueError("max_unique_media_invalid")
    index_paths = [index_path] if isinstance(index_path, Path) else list(index_path)
    if not index_paths:
        raise ValueError("prior_index_required")
    # Preflight every explicit source path before consuming any source bytes.
    _preflight_file(secret_path, secret=True)
    for source in (zip_path, csv_path, gsub_path, map_path, *index_paths):
        _preflight_file(source)
    sample_root_info = sample_dir.lstat()
    if not stat.S_ISDIR(sample_root_info.st_mode) or stat.S_ISLNK(sample_root_info.st_mode):
        raise ValueError("sample_manifest_dir_invalid")
    matched_sample_files = sorted(sample_dir.glob(sample_pattern))
    if not matched_sample_files:
        raise ValueError("sample_manifest_pattern_empty")
    for source in matched_sample_files:
        _preflight_file(source)
    key = _read_secret(secret_path)
    key_sha = hashlib.sha256(key).hexdigest()
    if expected_secret_sha256 is None or not SHA.fullmatch(expected_secret_sha256):
        raise ValueError("expected_secret_sha256_required")
    if not hmac.compare_digest(key_sha, expected_secret_sha256):
        raise ValueError("secret_sha256_mismatch")
    index_bytes = [_secure_read(path) for path in index_paths]
    input_bytes = {"split_csv": _secure_read(csv_path), "gsub": _secure_read(gsub_path),
                   "video_map": _secure_read(map_path)}
    prior_indexes = [_load_index(raw) for raw in index_bytes]
    prior = {kind: set().union(*(index[kind] for index in prior_indexes))
             for kind in ("video", "media", "pair")}
    excluded_paths, excluded_keys, sample_digests = _sample_paths(sample_dir, sample_pattern)
    gsub, vidmap = _json(input_bytes["gsub"]), _json(input_bytes["video_map"])
    if not isinstance(gsub, dict) or not isinstance(vidmap, dict):
        raise ValueError("dataset_mapping_schema_invalid")
    # Resolve ZIP safety and all archive members before any output is created.
    try:
        zip_fd = os.open(zip_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("input_unavailable_or_symlink") from exc
    zip_stream = os.fdopen(zip_fd, "rb")
    if not stat.S_ISREG(os.fstat(zip_stream.fileno()).st_mode):
        zip_stream.close()
        raise ValueError("input_must_be_regular_file")
    zip_hasher = hashlib.sha256()
    for block in iter(lambda: zip_stream.read(1024 * 1024), b""):
        zip_hasher.update(block)
    zip_sha = zip_hasher.hexdigest()
    zip_stream.seek(0)
    with zip_stream, zipfile.ZipFile(zip_stream) as zf:
        infos = zf.infolist(); by_key = {}; total = 0
        for info in infos:
            if not _safe_member(info.filename, info.is_dir()):
                raise ValueError("unsafe_zip_member")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("zip_symlink_rejected")
            if info.is_dir():
                continue
            if info.file_size < 0 or info.file_size > max_member_bytes:
                raise ValueError("zip_member_size_limit")
            total += info.file_size
            if total > max_total_bytes:
                raise ValueError("zip_total_size_limit")
            k = _video_key(info.filename)
            by_key.setdefault(k, []).append(info)
        mapped = {}
        for vid, loc in vidmap.items():
            if not isinstance(vid, str) or not isinstance(loc, str):
                raise ValueError("video_map_invalid")
            k = _video_key(loc)
            if k in mapped and mapped[k] != vid:
                raise ValueError("duplicate_mapped_video_key")
            mapped[k] = vid
        # Input digests bind exact local inputs without disclosing names or content.
        digests = {"zip_sha256": zip_sha, f"{dataset_split}_csv_sha256": hashlib.sha256(input_bytes["split_csv"]).hexdigest(),
                   "gsub_sha256": hashlib.sha256(input_bytes["gsub"]).hexdigest(), "video_map_sha256": hashlib.sha256(input_bytes["video_map"]).hexdigest(),
                   "prior_index_sha256": hashlib.sha256(index_bytes[0]).hexdigest(), "secret_sha256": key_sha,
                   "sample_manifest_set_sha256": hashlib.sha256("".join(sorted(sample_digests)).encode()).hexdigest()}
        digests["prior_index_sha256s"] = [hashlib.sha256(raw).hexdigest()
                                             for raw in index_bytes]
        candidates = []; exclusion_raw_sha = []; media_cache = {}; hashed_total = 0
        counts = {"csv_rows": 0, "mapped_rows": 0,
            "sample_path_excluded": 0, "ambiguous_or_missing_zip_media": 0,
            "prior_video_excluded": 0, "prior_pair_excluded": 0, "prior_media_excluded": 0,
            "invalid_interval": 0, "choice_ineligible": 0, "eligible": 0}
        with io.StringIO(input_bytes["split_csv"].decode("utf-8-sig"), newline="") as f:
            reader = csv.DictReader(f)
            required = {"video_id", "qid", "answer", "a0", "a1", "a2", "a3", "a4"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("nextgqa_mcqa_schema_invalid")
            for row in reader:
                counts["csv_rows"] += 1
                vid, qid = row["video_id"], row["qid"]
                if vid not in vidmap:
                    continue
                counts["mapped_rows"] += 1; loc = vidmap[vid]; k = _video_key(loc)
                if (loc.replace("\\", "/").lstrip("./") in excluded_paths
                        or k in excluded_keys):
                    counts["sample_path_excluded"] += 1; continue
                if len(by_key.get(k, [])) != 1:
                    counts["ambiguous_or_missing_zip_media"] += 1; continue
                vtok = _token(key, "dataset-video-id", ("nextgqa", vid))
                ptok = _token(key, "video-question-pair", ("nextgqa", "dataset-video-id", "nextgqa", vid, qid))
                if vtok in prior["video"]:
                    counts["prior_video_excluded"] += 1; continue
                if ptok in prior["pair"]:
                    counts["prior_pair_excluded"] += 1; continue
                info = gsub.get(vid); locs = info.get("location") if isinstance(info, dict) else None
                spans = locs.get(qid) if isinstance(locs, dict) else None
                try:
                    import math
                    duration = float(info["duration"]); spans2 = [[float(x[0]), float(x[1])] for x in spans]
                    valid = (math.isfinite(duration) and duration > 0 and bool(spans2)
                             and all(math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= duration
                                     for a, b in spans2))
                except (TypeError, ValueError, KeyError, IndexError, OverflowError):
                    valid = False; spans2 = []
                if not valid:
                    counts["invalid_interval"] += 1; continue
                choices = [str(row[f"a{i}"]).strip() for i in range(5)]
                gold = [i for i, x in enumerate(choices) if _norm(x) == _norm(str(row["answer"]))]
                if any(not x for x in choices) or len(gold) != 1:
                    counts["choice_ineligible"] += 1; continue
                zi = by_key[k][0]
                cache_key = zi.header_offset
                if cache_key not in media_cache:
                    media_cache[cache_key] = _sha_member(zf, zi, max_member_bytes)
                    hashed_total += media_cache[cache_key][1]
                    if hashed_total > max_total_bytes:
                        raise ValueError("zip_actual_total_size_limit")
                media_sha, _media_size = media_cache[cache_key]
                mtok = _token(key, "media-sha256", (media_sha,))
                if mtok in prior["media"]:
                    counts["prior_media_excluded"] += 1; exclusion_raw_sha.append(media_sha); continue
                pairsha = candidate_pair_sha256(vid, qid)
                candidates.append((zi, pairsha, media_sha)); counts["eligible"] += 1
        # Rank only after every mapped candidate has been scanned and its media
        # bytes hashed for exposure. The cap affects extraction, never exposure
        # checking, and is independent of answer/model outcomes.
        unique_media = {}
        for zi, _pairsha, media_sha in candidates:
            unique_media.setdefault(zi.header_offset, (zi, media_sha))
        ranked_media = sorted(
            unique_media.values(),
            key=lambda item: hashlib.sha256(json.dumps(
                ["nextgqa-zip-candidate-selection-v1", dataset_split, item[1]],
                separators=(",", ":")).encode("utf-8")).digest())
        if max_unique_media is not None:
            selected_offsets = {zi.header_offset for zi, _ in ranked_media[:max_unique_media]}
        else:
            selected_offsets = set(unique_media)
        selected_candidates = [row for row in candidates if row[0].header_offset in selected_offsets]
        selected_unique_media = {offset: value for offset, value in unique_media.items()
                                 if offset in selected_offsets}
        # Create all outputs exclusively after validation. Roll back only this new directory on failure.
        parent_info = output_dir.parent.lstat()
        if (not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode)
                or parent_info.st_uid != os.getuid() or stat.S_IMODE(parent_info.st_mode) != 0o700):
            raise ValueError("output_parent_must_be_owned_mode_0700_directory")
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("output_directory_already_exists")
        output_dir.mkdir(mode=0o700, parents=False)
        try:
            os.chmod(output_dir, 0o700)
            pair_hashes = [pairsha for _zi, pairsha, _sha in selected_candidates]
            extracted_total = 0
            for zi, expected_sha in selected_unique_media.values():
                # Preserve the archive-relative media key expected by the downstream runner.
                dest = output_dir.joinpath(*PurePosixPath(zi.filename).parts)
                dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(dest.parent, 0o700)
                written = _extract_member(zf, zi, dest, expected_sha, max_member_bytes,
                                          max_total_bytes - extracted_total)
                extracted_total += written
            _write_exclusive(output_dir / "eligible_pairs.json",
                (json.dumps({"schema_version": 1, "pair_sha256": pair_hashes}, sort_keys=True, indent=2)+"\n").encode())
            _write_exclusive(output_dir / "prior_media_sha256_exclusions.json",
                (json.dumps({"schema_version": 1, "media_sha256": sorted(set(exclusion_raw_sha))}, indent=2)+"\n").encode())
            summary = {"schema_version": 1, "purpose": "outcome_blind_local_candidate_pool",
                       "dataset_split": dataset_split,
                       "counts": counts, "input_sha256": digests,
                       "eligible_pair_count": len(candidates),
                       "selected_eligible_pair_count": len(selected_candidates),
                       "eligible_unique_media_count": len(unique_media),
                       "selected_unique_media_count": len(selected_unique_media),
                       "extracted_media_count": len(selected_unique_media),
                       "extracted_unique_media_count": len(selected_unique_media),
                       "extracted_bytes": extracted_total,
                       "selection_rule": "lowest SHA-256 rank of [nextgqa-zip-candidate-selection-v1, dataset_split, media_sha256], after complete exposure scan; all eligible pairs for selected media are retained",
                       "max_unique_media": max_unique_media,
                       "private_outputs": ["eligible_pairs.json", "prior_media_sha256_exclusions.json"],
                       "max_member_bytes": max_member_bytes, "max_total_bytes": max_total_bytes}
            _write_exclusive(output_dir / "summary.json",
                (json.dumps(summary, sort_keys=True, indent=2)+"\n").encode())
            return summary
        except Exception:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
            raise


def _sha_member(zf, info, max_member_bytes):
    h = hashlib.sha256()
    size = 0
    with zf.open(info) as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            size += len(block)
            if size > max_member_bytes:
                raise ValueError("zip_actual_member_size_limit")
            h.update(block)
    return h.hexdigest(), size


def _sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("zip", "video-map", "sample-manifest-dir", "secret-file", "output-dir"):
        p.add_argument("--"+flag, required=True, type=Path)
    p.add_argument("--test-csv", "--split-csv", dest="split_csv", required=True, type=Path)
    p.add_argument("--gsub-test", "--gsub-json", dest="gsub_json", required=True, type=Path)
    p.add_argument("--dataset-split", choices=("train", "val", "test"), default="test")
    p.add_argument("--max-unique-media", type=int, default=None,
                   help="extract at most this many uniquely ranked eligible media files")
    p.add_argument("--prior-exposure-index", required=True, action="append", type=Path,
                   help="validated HMAC prior index; may be repeated and is conservatively unioned")
    p.add_argument("--sample-manifest-pattern", required=True)
    p.add_argument("--expected-secret-sha256", required=True,
                   help="out-of-band SHA-256 pin for the local mode-0600 HMAC key")
    a=p.parse_args(argv)
    try:
        result=build(a.zip,a.split_csv,a.gsub_json,a.video_map,a.sample_manifest_dir,a.sample_manifest_pattern,
                     a.prior_exposure_index,a.secret_file,a.output_dir,
                     expected_secret_sha256=a.expected_secret_sha256,
                     dataset_split=a.dataset_split, max_unique_media=a.max_unique_media)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        p.error(str(exc).split(":",1)[0])
    print(json.dumps({"counts":result["counts"],"extracted_media_count":result["extracted_media_count"]},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
