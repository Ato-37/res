"""Preflight candidate samples for strict grounding/control frame coverage.

This CPU/video-only check runs before loading Qwen. It selects a fixed number
of candidates for which the experiment can construct strict evidence and an
equal-dose, non-grounding control. It does not change or infer the labels.

Usage:
    python scripts/preflight_necessity_choice_manifest.py \
        --samples-file data/necessity_candidates.json \
        --n-valid 60 \
        --output-samples data/necessity_valid.json \
        --n-smoke 3 \
        --smoke-samples-output data/necessity_smoke.json \
        --audit-output data/necessity_preflight.json
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from necessity_choice_margin_utils import (assign_evidence_control_frames,
                                           inspect_video_capture,
                                           uniform_sample_indices)


def decodable_video_info(video_path):
    # Keep OpenCV lazy-loaded so --help and static checks work without the
    # remote video/GPU environment installed.
    import cv2

    cap = cv2.VideoCapture(video_path)
    return inspect_video_capture(cap, cv2.CAP_PROP_FPS)


def sample_indices(total, num_frames):
    # Keep the CLI-level preflight API while sharing the runner implementation.
    return uniform_sample_indices(total, num_frames)


def audit_samples(samples, num_frames):
    selected, records = [], []
    candidate_video_ids = [sample['video_id'] for sample in samples
                           if isinstance(sample, dict)
                           and isinstance(sample.get('video_id'), str)
                           and sample['video_id'].strip()
                           and sample['video_id'] == sample['video_id'].strip()]
    video_id_counts = Counter(candidate_video_ids)
    duplicated_video_ids = {video_id for video_id, count in video_id_counts.items()
                            if count > 1}
    for sample_idx, sample in enumerate(samples):
        if (not isinstance(sample, dict)
                or not isinstance(sample.get('video_id'), str)
                or not sample['video_id'].strip()
                or sample['video_id'] != sample['video_id'].strip()
                or not isinstance(sample.get('video'), str)
                or not sample['video']):
            records.append({'sample_id': f'{sample_idx}:{sample_idx}',
                            'status': 'invalid_manifest_record',
                            'error': 'sample row must have an exact video_id and nonempty video path'})
            continue
        sample_id = f"{sample.get('video_id', sample_idx)}:{sample.get('qid', sample_idx)}"
        if sample['video_id'] in duplicated_video_ids:
            records.append({'sample_id': sample_id, 'status': 'duplicate_video_id',
                            'video': sample['video'],
                            'error': 'multiple candidate rows share a video_id'})
            continue
        try:
            video = os.path.expanduser(sample['video'])
            total, fps = decodable_video_info(video)
            try:
                coverage = assign_evidence_control_frames(
                    total, num_frames, fps, sample.get('grounding'))
            except (TypeError, ValueError) as exc:
                record = {'sample_id': sample_id,
                          'status': 'invalid_assignment_geometry',
                          'video': sample.get('video'), 'error': repr(exc)}
                records.append(record)
                continue
            record = {
                'sample_id': sample_id,
                'status': coverage['status'],
                'video': sample['video'],
                'requested_frame_count': coverage['requested_frame_count'],
                'sampled_frame_count': coverage['sampled_frame_count'],
                'evidence_frame_count': len(coverage['evidence_indices']),
                'control_frame_count': len(coverage['control_indices']),
                'sampled_indices': coverage['sampled_indices'],
                'evidence_indices': coverage['evidence_indices'],
                'control_indices': coverage['control_indices'],
                'matched_rank_distance': coverage['matched_rank_distance'],
            }
            if coverage['status'] == 'ok':
                selected.append(sample)
        except Exception as exc:
            record = {'sample_id': sample_id, 'status': 'video_or_metadata_error',
                      'video': sample.get('video'), 'error': repr(exc)}
        records.append(record)
    return selected, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-valid', type=int, required=True)
    parser.add_argument('--n-smoke', type=int, default=3)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--output-samples', required=True)
    parser.add_argument('--smoke-samples-output', required=True)
    parser.add_argument('--audit-output', required=True)
    args = parser.parse_args()
    if args.n_valid < 1 or args.n_smoke < 0 or args.num_frames < 3:
        parser.error('--n-valid must be positive, --n-smoke nonnegative, and --num-frames >=3')
    source = Path(args.samples_file).resolve()
    selected_path = Path(args.output_samples).resolve()
    smoke_path = Path(args.smoke_samples_output).resolve()
    audit_path = Path(args.audit_output).resolve()
    if not source.is_file():
        parser.error(f'Missing candidate manifest: {source}')
    if len({selected_path, smoke_path, audit_path}) != 3:
        parser.error('Smoke, heldout, and audit outputs must be different files')
    if any(path.exists() for path in (selected_path, smoke_path, audit_path)):
        parser.error('Refusing to overwrite an existing output')
    samples = json.loads(source.read_text())
    if not isinstance(samples, list):
        parser.error('Candidate manifest must contain a JSON list')
    candidates, records = audit_samples(samples, args.num_frames)
    counts = Counter(record['status'] for record in records)
    required = args.n_smoke + args.n_valid
    report = {
        'script': 'preflight_necessity_choice_manifest.py',
        'samples_file': str(source),
        'n_candidates': len(samples),
        'n_valid_requested': args.n_valid,
        'n_smoke_requested': args.n_smoke,
        'num_frames': args.num_frames,
        'status_counts': dict(sorted(counts.items())),
        'n_valid_available': len(candidates),
        'smoke_sample_ids': [f"{s.get('video_id')}:{s.get('qid')}"
                             for s in candidates[:args.n_smoke]],
        'heldout_sample_ids': [f"{s.get('video_id')}:{s.get('qid')}"
                               for s in candidates[args.n_smoke:required]],
        'records': records,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + '\n')
    if len(candidates) < required:
        print(f"preflight insufficient: requested={required} valid={len(candidates)}; "
              f"audit={audit_path}")
        raise SystemExit(2)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(json.dumps(candidates[:args.n_smoke], indent=2) + '\n')
    selected_path.write_text(json.dumps(candidates[args.n_smoke:required], indent=2) + '\n')
    print(f"preflight PASS candidates={len(samples)} valid={len(candidates)} "
          f"smoke={args.n_smoke} heldout={args.n_valid} "
          f"coverage_failures={len(samples)-len(candidates)}")
    print(f"wrote smoke={smoke_path} heldout={selected_path} audit={audit_path}")


if __name__ == '__main__':
    main()
