"""Run a head-group patch screen, analyze it, and publish its outputs.

This wrapper is intended for the remote GPU machine. It prevents the common
failure mode where ``results/`` is ignored and a completed experiment never
reaches Git. It checks Git identity before starting the expensive run.

Usage:
    python scripts/run_all_layer_and_publish.py \
        --samples-file data/nextgqa_samples_confirm_heldout.json \
        --n-samples 25 --layers all --groups all

For a whole-model singleton-head screen, use ``--layers all --groups singletons``.
This evaluates 28 layers x 28 heads and is substantially more expensive than
``--groups all`` (one all-head group per layer). A selected-layer screen can
reduce confirmation cost, for example ``--layers 3,5,7,9,12 --groups singletons``.
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


def repo_path(value, repo):
    path = (repo / value).resolve()
    results_root = (repo / 'results').resolve()
    if results_root not in path.parents:
        raise ValueError('Output must be inside results/')
    return path


def git_identity(repo):
    """Return the local Git identity or fail before starting a GPU run."""
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
            'Set it before starting the GPU run, for example: '
            "git config --local user.name 'Your Name' && "
            "git config --local user.email 'you@example.com'"
        )
    return values['user.name'], values['user.email']


def push_with_rebase(repo, branch, max_attempts=3):
    """Push the just-created commit, rebasing it if origin advanced meanwhile."""
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(['git', 'push', 'origin', branch], cwd=str(repo), check=False)
        if result.returncode == 0:
            return
        print(f'Push attempt {attempt} failed; fetching origin/{branch} to check for concurrent commits.',
              flush=True)
        run(['git', 'fetch', 'origin', branch], repo)

        status = subprocess.run(['git', 'status', '--porcelain'], cwd=str(repo),
                                text=True, capture_output=True, check=True).stdout.strip()
        if status:
            raise SystemExit(
                'Push was rejected and the worktree is not clean. The experiment commit remains local; '
                'preserve these changes, then fetch/rebase/push manually.'
            )

        remote_ref = f'origin/{branch}'
        remote_has_head = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', 'HEAD', remote_ref],
            cwd=str(repo), check=False,
        ).returncode == 0
        if remote_has_head:
            run(['git', 'merge', '--ff-only', remote_ref], repo)
            print('The experiment commit is already contained in origin; synchronized the local branch.',
                  flush=True)
            return

        remote_is_ancestor = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', remote_ref, 'HEAD'],
            cwd=str(repo), check=False,
        ).returncode == 0
        if not remote_is_ancestor:
            rebase = subprocess.run(['git', 'rebase', remote_ref], cwd=str(repo), check=False)
            if rebase.returncode != 0:
                raise SystemExit(
                    'Automatic rebase hit a conflict. The experiment commit is preserved locally; '
                    'resolve the rebase, then push. Do not rerun the GPU experiment.'
                )
    raise SystemExit(
        'Push still failed after synchronization attempts. The experiment commit remains local; '
        'inspect git status/log and push it after resolving the remote state.'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int, default=25)
    parser.add_argument('--layers', default='all',
                        help='Layer selector passed to head_group_activation_patching.py')
    parser.add_argument('--groups', default='all',
                        help='Head groups accepted by head_group_activation_patching.py '
                             '(for example: all or singletons)')
    parser.add_argument('--output')
    parser.add_argument('--commit-message')
    parser.add_argument('--no-push', action='store_true')
    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error('--n-samples must be positive')

    repo = Path(__file__).resolve().parents[1]
    samples = (repo / args.samples_file).resolve()
    if not samples.is_file():
        parser.error(f'Missing samples file: {samples}')
    try:
        samples_relative = samples.relative_to(repo)
    except ValueError:
        parser.error('--samples-file must be inside the repository so it can be published')
    group_label = args.groups.replace(',', '_').replace('+', '_')
    layer_label = args.layers.replace(',', '_').replace('+', '_')
    if args.layers == 'all':
        default_name = ('head_group_all_layers_heldout.json' if args.groups == 'all'
                        else f'head_group_all_layers_{group_label}_heldout.json')
    else:
        default_name = f'head_group_layers_{layer_label}_{group_label}_heldout.json'
    output_value = args.output or f'results/{default_name}'
    output = repo_path(output_value, repo)
    analysis = output.with_name(output.stem + '_analysis.json')
    manifest = output.with_name(output.stem + '_run_manifest.json')

    name, email = git_identity(repo)

    gpu_command = [sys.executable, str(repo / 'scripts/head_group_activation_patching.py'),
                   '--samples-file', str(samples), '--n-samples', str(args.n_samples),
                   '--layers', args.layers, '--groups', args.groups, '--patch-scope', 'all',
                   '--self-check', '--strict-grounding', '--output', str(output)]
    run(gpu_command, repo)
    run([sys.executable, str(repo / 'scripts/analyze_head_group_patching.py'),
         '--input', str(output), '--output', str(analysis)], repo)

    manifest.write_text(json.dumps({
        'script': 'run_all_layer_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(samples_relative),
        'n_samples': args.n_samples,
        'layers': args.layers,
        'groups': args.groups,
        'gpu_command': gpu_command,
        'analysis': str(analysis.relative_to(repo)),
        'git_user_name': name,
        'git_user_email': email,
    }, indent=2) + '\n')

    relative = [str(path.relative_to(repo)) for path in (output, analysis, manifest)]
    sample_is_tracked = subprocess.run(
        ['git', 'ls-files', '--error-unmatch', str(samples_relative)],
        cwd=str(repo), text=True, capture_output=True, check=False,
    )
    if sample_is_tracked.returncode != 0:
        relative.insert(0, str(samples_relative))
    run(['git', 'add', '-f', *relative], repo)
    commit_message = args.commit_message or f'Publish layer-{layer_label} {group_label} head patch analysis'
    run(['git', 'commit', '--only', '-m', commit_message, *relative], repo)
    if not args.no_push:
        branch = subprocess.run(['git', 'branch', '--show-current'], cwd=str(repo),
                                text=True, capture_output=True, check=True).stdout.strip()
        if not branch:
            raise SystemExit('Detached HEAD: refusing to push')
        push_with_rebase(repo, branch)
    print(f'Published {output.relative_to(repo)} and {analysis.relative_to(repo)}')


if __name__ == '__main__':
    main()
