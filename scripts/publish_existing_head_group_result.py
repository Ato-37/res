"""Analyze and publish an already-completed head-group result.

Use this after a direct GPU command has finished. It does not rerun the model.
The input, analysis, and manifest are force-added because results/ is ignored.

Usage:
    python scripts/publish_existing_head_group_result.py \
        --input results/head_group_all_layers_heldout.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def run(command, cwd):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def result_path(value, repo):
    path = (repo / value).resolve()
    root = (repo / 'results').resolve()
    if root not in path.parents:
        raise ValueError('Path must be inside results/')
    return path


def git_identity(repo):
    """Return the local Git identity or fail with an actionable message."""
    values = {}
    for key in ('user.name', 'user.email'):
        result = subprocess.run(
            ['git', 'config', '--local', '--get', key],
            cwd=str(repo), text=True, capture_output=True, check=False,
        )
        values[key] = result.stdout.strip() if result.returncode == 0 else ''
    missing = [key for key in ('user.name', 'user.email') if not values[key]]
    if missing:
        missing_text = ', '.join(missing)
        raise SystemExit(
            f'Missing local Git config: {missing_text}. '
            'Set it before publishing, for example: '
            "git config --local user.name 'Your Name' && "
            "git config --local user.email 'you@example.com'"
        )
    return values['user.name'], values['user.email']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--analysis')
    parser.add_argument('--commit-message', default='Publish completed head-group result')
    parser.add_argument('--no-push', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    input_path = result_path(args.input, repo)
    if not input_path.is_file():
        parser.error(f'Missing result: {input_path}')
    analysis = result_path(args.analysis, repo) if args.analysis else input_path.with_name(
        input_path.stem + '_analysis.json')
    manifest = input_path.with_name(input_path.stem + '_publish_manifest.json')
    git_identity(repo)
    run([sys.executable, str(repo / 'scripts/analyze_head_group_patching.py'),
         '--input', str(input_path), '--output', str(analysis)], repo)
    manifest.write_text(json.dumps({
        'script': 'publish_existing_head_group_result.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(input_path.relative_to(repo)),
        'analysis': str(analysis.relative_to(repo)),
    }, indent=2) + '\n')
    relative = [str(path.relative_to(repo)) for path in (input_path, analysis, manifest)]
    run(['git', 'add', '-f', *relative], repo)
    run(['git', 'commit', '--only', '-m', args.commit_message, *relative], repo)
    if not args.no_push:
        branch = subprocess.run(['git', 'branch', '--show-current'], cwd=str(repo),
                                text=True, capture_output=True, check=True).stdout.strip()
        if not branch:
            raise SystemExit('Detached HEAD: refusing to push')
        run(['git', 'push', 'origin', branch], repo)
    print(f'Published {input_path.relative_to(repo)}')


if __name__ == '__main__':
    main()
