"""Build disjoint test manifests, run a guarded smoke, then one confirmation."""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import subprocess
import sys

from run_faithfulness_validation_pilot import ensure_git_ready, run
from closed_research_branch import reject_inference


def validate_result(result, requested, stage, minimum_effective=None):
    if result.get('n_requested') != requested:
        raise SystemExit(f'{stage}: requested-count mismatch; stopping')
    effective = result.get('n_effective')
    failures = result.get('n_failures')
    if (not isinstance(effective, int) or not isinstance(failures, int)
            or effective + failures != requested):
        raise SystemExit(f'{stage}: sample accounting mismatch; stopping')
    required = requested if minimum_effective is None else minimum_effective
    if effective < required or (minimum_effective is None and failures != 0):
        raise SystemExit(
            f'{stage}: completed {effective}/{requested}, failures={failures}; '
            f'required effective n >= {required}; stopping without replacement'
        )


def build_manifest(builder, repo, csv_path, gsub_path, map_path, video_root,
                   zip_path, n, seed, exclusions, output):
    command = [sys.executable, builder,
               '--test-csv', csv_path,
               '--gsub-json', gsub_path,
               '--map-json', map_path,
               '--video-root', video_root,
               '--zip-path', zip_path,
               '--n', str(n), '--seed', str(seed),
               '--exclude', *map(str, exclusions),
               '--out', str(output)]
    run(command, repo)
    samples = json.loads(output.read_text())
    if len(samples) != n:
        raise SystemExit(f'sample builder returned {len(samples)}/{n}; stopping before inference')
    video_ids = [str(sample.get('video_id')) for sample in samples]
    if any(not value for value in video_ids) or len(video_ids) != len(set(video_ids)):
        raise SystemExit('sample manifest has missing or duplicate video IDs; stopping')


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True,
                        help='unique filename tag, for example 20261009; never reused')
    parser.add_argument('--test-csv', default='NExT-GQA/datasets/nextgqa/test.csv')
    parser.add_argument('--gsub-json', default='NExT-GQA/datasets/nextgqa/gsub_test.json')
    parser.add_argument('--map-json', default='NExT-GQA/datasets/nextgqa/map_vid_vidorID.json')
    parser.add_argument('--video-root', default='~/nextgqa_sample/NExTVideo')
    parser.add_argument('--zip-path', default='~/NExTVideo.zip')
    parser.add_argument('--exclude-manifest', action='append', default=[],
                        help='additional historical sample JSON to exclude; repeatable')
    parser.add_argument('--historical-test-smoke',
                        default='~/nextgqa_sample/faithfulness_smoke_samples_20261008.json')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--smoke-n', type=int, default=3)
    parser.add_argument('--confirm-n', type=int, default=79)
    parser.add_argument('--minimum-effective', type=int, default=71)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    args = parser.parse_args()
    reject_inference(parser, "closed_head_faithfulness",
                     "the independent confirmation was completed and did not meet its practical threshold")
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag may contain only letters, numbers, underscore, and hyphen')
    if args.smoke_n != 3:
        parser.error('protocol requires a three-video engineering smoke')
    if args.confirm_n < args.minimum_effective or args.minimum_effective < 2:
        parser.error('confirm-n must be >= minimum-effective >= 2')

    test_csv = (repo / args.test_csv).expanduser().resolve()
    gsub_json = (repo / args.gsub_json).expanduser().resolve()
    map_json = (repo / args.map_json).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    zip_path = Path(args.zip_path).expanduser().resolve()
    historical = [Path(path).expanduser().resolve() for path in args.exclude_manifest]
    historical_test_smoke = Path(args.historical_test_smoke).expanduser().resolve()
    required_inputs = (test_csv, gsub_json, map_json, zip_path, historical_test_smoke)
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        parser.error('required input(s) missing: ' + ', '.join(missing))
    missing_extra = [str(path) for path in historical if not path.is_file()]
    if missing_extra:
        parser.error('exclusion manifest(s) missing: ' + ', '.join(missing_extra))
    if not zip_path.stat().st_size:
        parser.error(f'video archive is empty: {zip_path}')

    data_manifests = sorted((repo / 'data').glob('nextgqa_samples*.json'))
    exclusions = list(data_manifests) + historical + [historical_test_smoke]
    if len({path.resolve() for path in exclusions}) != len(exclusions):
        exclusions = list(dict.fromkeys(path.resolve() for path in exclusions))

    manifest_dir = Path.home() / 'nextgqa_sample'
    smoke_manifest = manifest_dir / f'head_faithfulness_confirm_smoke_{args.tag}.json'
    confirm_manifest = manifest_dir / f'head_faithfulness_confirm_{args.tag}.json'
    smoke_output = f'results/head_faithfulness_confirm_smoke_{args.tag}.json'
    confirm_output = f'results/head_faithfulness_confirm_{args.tag}.json'
    outputs = [smoke_manifest, confirm_manifest,
               repo / smoke_output,
               repo / 'results' / f'head_faithfulness_confirm_smoke_{args.tag}_run_manifest.json',
               repo / confirm_output,
               repo / 'results' / f'head_faithfulness_confirm_{args.tag}_run_manifest.json']
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error('refusing to overwrite existing output(s): ' + ', '.join(existing))

    ensure_git_ready(repo)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    builder = str(repo / 'scripts' / 'build_samples.py')
    publisher = str(repo / 'scripts' / 'run_head_faithfulness_and_publish.py')

    # Smoke is excluded from confirmation. The older test-side smoke is excluded
    # from both stages; all tracked/local sample manifests are excluded as well.
    build_manifest(builder, str(repo), str(test_csv), str(gsub_json), str(map_json),
                   str(video_root), str(zip_path), args.smoke_n, args.seed,
                   exclusions, smoke_manifest)
    run([sys.executable, publisher, '--samples-file', str(smoke_manifest),
         '--n-samples', str(args.smoke_n), '--layers', 'all',
         '--quant', args.quant, '--seed', str(args.seed),
         '--output', smoke_output], str(repo))
    smoke_result = json.loads((repo / smoke_output).read_text())
    validate_result(smoke_result, args.smoke_n, 'engineering smoke')

    confirm_exclusions = exclusions + [smoke_manifest]
    build_manifest(builder, str(repo), str(test_csv), str(gsub_json), str(map_json),
                   str(video_root), str(zip_path), args.confirm_n, args.seed + 1,
                   confirm_exclusions, confirm_manifest)
    run([sys.executable, publisher, '--samples-file', str(confirm_manifest),
         '--n-samples', str(args.confirm_n), '--layers', 'all',
         '--quant', args.quant, '--seed', str(args.seed + 1),
         '--output', confirm_output], str(repo))
    confirm_result = json.loads((repo / confirm_output).read_text())
    validate_result(confirm_result, args.confirm_n, 'independent confirmation',
                    args.minimum_effective)
    print('Published smoke and locked independent confirmation; no post-inference replacement.',
          flush=True)


if __name__ == '__main__':
    main()
