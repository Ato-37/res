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

from necessity_choice_margin_utils import assess_frame_coverage, uniform_sample_indices


def decodable_video_info(video_path):
    # Keep OpenCV lazy-loaded so --help and static checks work without the
    # remote video/GPU environment installed.
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    if count <= 0:
        raise RuntimeError(f'Could not decode any frames from {video_path}')
    return count, fps


def sample_indices(total, num_frames):
    # Keep the CLI-level preflight API while sharing the runner implementation.
    return uniform_sample_indices(total, num_frames)


def audit_samples(samples, num_frames):
    selected, records = [], []
    for sample_idx, sample in enumerate(samples):
        sample_id = f"{sample.get('video_id', sample_idx)}:{sample.get('qid', sample_idx)}"
        try:
            video = os.path.expanduser(sample['video'])
            total, fps = decodable_video_info(video)
            indices = sorted(set(sample_indices(total, num_frames)))
            coverage = assess_frame_coverage(indices, fps, sample['grounding'])
            record = {
                'sample_id': sample_id,
                'status': coverage['status'],
                'video': sample['video'],
                'sampled_frame_count': len(indices),
                'evidence_frame_count': len(coverage['evidence']),
                'control_frame_count': len(coverage['control']),
                'sampled_indices': indices,
                'evidence_indices': coverage['evidence'],
                'control_indices': coverage['control'],
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
