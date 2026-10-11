"""Build a disjoint validation smoke and 30-video faithfulness pilot.

The smoke artifact is published first. The pilot is not started unless all
three smoke videos complete successfully. Both result artifacts are published
by the existing privacy-scoped publisher; raw manifests remain outside Git.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_head_faithfulness_and_publish import git_identity


def run(command, repo):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=repo, check=True)


def validate_effective_result(result, requested, stage):
    if result.get('n_requested') != requested:
        raise SystemExit(f'{stage} result requested-count mismatch; stopping')
    if result.get('n_effective', 0) + result.get('n_failures', 0) != requested:
        raise SystemExit(f'{stage} result accounting mismatch; stopping')
    if result.get('n_effective') != requested or result.get('n_failures') != 0:
        raise SystemExit(
            f'{stage} completed {result.get("n_effective")}/{requested} with '
            f'{result.get("n_failures")} failures; pilot/next stage will not start'
        )


def ensure_git_ready(repo):
    git_identity(str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('detached HEAD; refusing to start validation job')
    tracked = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                             cwd=repo, text=True, capture_output=True,
                             check=True).stdout.strip()
    staged = subprocess.run(['git', 'diff', '--cached', '--name-only'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if tracked or staged:
        raise SystemExit('tracked worktree/index must be clean before starting')


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-csv', default='NExT-GQA/datasets/nextgqa/val.csv')
    parser.add_argument('--gsub-val-json', default='NExT-GQA/datasets/nextgqa/gsub_val.json')
    parser.add_argument('--map-json', default='NExT-GQA/datasets/nextgqa/map_vid_vidorID.json')
    parser.add_argument('--video-root', default='~/nextgqa_sample/NExTVideo')
    parser.add_argument('--zip-path', default='~/NExTVideo.zip')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--smoke-n', type=int, default=3)
    parser.add_argument('--pilot-n', type=int, default=30)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    args = parser.parse_args()
    if args.smoke_n < 1 or args.pilot_n < 2:
        parser.error('--smoke-n must be >=1 and --pilot-n must be >=2')

    ensure_git_ready(repo)
    val_csv = (repo / args.val_csv).expanduser().resolve()
    gsub = (repo / args.gsub_val_json).expanduser().resolve()
    map_json = (repo / args.map_json).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    zip_path = Path(args.zip_path).expanduser().resolve()
    for path in (val_csv, gsub, map_json, zip_path):
        if not path.is_file():
            parser.error(f'required validation input does not exist: {path}')
    if not zip_path.stat().st_size:
        parser.error(f'video archive is empty: {zip_path}')

    date_tag = '20261008'
    manifest_root = Path.home() / 'nextgqa_sample'
    manifest_root.mkdir(parents=True, exist_ok=True)
    smoke_manifest = manifest_root / f'faithfulness_validation_smoke_{date_tag}.json'
    pilot_manifest = manifest_root / f'faithfulness_validation_pilot_{date_tag}.json'
    smoke_output = f'results/faithfulness_validation_smoke_{date_tag}.json'
    pilot_output = f'results/faithfulness_validation_pilot_{date_tag}.json'
    paths_to_preserve = (smoke_manifest, pilot_manifest,
                         repo / smoke_output,
                         repo / 'results' / f'faithfulness_validation_smoke_{date_tag}_run_manifest.json',
                         repo / pilot_output,
                         repo / 'results' / f'faithfulness_validation_pilot_{date_tag}_run_manifest.json')
    collisions = [str(path) for path in paths_to_preserve if path.exists()]
    if collisions:
        parser.error('refusing to overwrite existing output(s): ' + ', '.join(collisions))

    existing_sample_manifests = sorted((repo / 'data').glob('nextgqa_samples*.json'))
    builder = repo / 'scripts' / 'build_samples.py'
    publisher = repo / 'scripts' / 'run_head_faithfulness_and_publish.py'

    def build_manifest(n, seed, output, exclusions):
        command = [sys.executable, builder,
                   '--test-csv', val_csv,
                   '--gsub-json', gsub,
                   '--map-json', map_json,
                   '--video-root', video_root,
                   '--zip-path', zip_path,
                   '--n', str(n), '--seed', str(seed),
                   '--exclude', *exclusions,
                   '--out', output]
        run(command, repo)
        manifest = json.loads(output.read_text())
        if len(manifest) != n:
            raise SystemExit(f'builder returned {len(manifest)}/{n} videos; stopping before inference')

    build_manifest(args.smoke_n, args.seed, smoke_manifest,
                   [str(path) for path in existing_sample_manifests])
    run([sys.executable, publisher,
         '--samples-file', smoke_manifest,
         '--n-samples', str(args.smoke_n), '--layers', 'all',
         '--quant', args.quant, '--seed', str(args.seed),
         '--output', smoke_output], repo)
    smoke_result = json.loads((repo / smoke_output).read_text())
    validate_effective_result(smoke_result, args.smoke_n, 'validation smoke')

    # The pilot uses different videos from both the historical manifests and the smoke.
    pilot_exclusions = [str(path) for path in existing_sample_manifests]
    pilot_exclusions.append(str(smoke_manifest))
    build_manifest(args.pilot_n, args.seed + 1, pilot_manifest, pilot_exclusions)
    run([sys.executable, publisher,
         '--samples-file', pilot_manifest,
         '--n-samples', str(args.pilot_n), '--layers', 'all',
         '--quant', args.quant, '--seed', str(args.seed + 1),
         '--output', pilot_output], repo)
    pilot_result = json.loads((repo / pilot_output).read_text())
    validate_effective_result(pilot_result, args.pilot_n, 'validation pilot')
    print(f'completed and published validation smoke ({args.smoke_n}) and pilot ({args.pilot_n})',
          flush=True)
    print('Pilot results are for runtime/variance calibration only; they are not confirmatory.',
          flush=True)


if __name__ == '__main__':
    main()
