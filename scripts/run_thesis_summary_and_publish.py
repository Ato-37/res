"""Regenerate the necessity summary, run CPU tests, and publish only its artifacts.

Designed for the Ubuntu research checkout. No model/GPU inference is performed.
The worktree must be clean so unrelated user changes can never enter the commit.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from run_all_layer_and_publish import git_identity, push_with_rebase, run


INPUTS = (
    'results/necessity_choice_margin_replication_heldout.json',
    'results/necessity_choice_margin_replication_blur_heldout.json',
    'results/necessity_choice_margin_replication_mask_fill_comparison.json',
)
TESTS = (
    'scripts/test_plot_necessity_choice_results.py',
    'scripts/test_necessity_choice_margin_utils.py',
    'scripts/test_thesis_summary_publisher.py',
)
SUFFIXES = ('_estimates.csv', '_forest.svg', '_metadata.json')
RUN_PREFIX = 'results/necessity_choice_summary_ubuntu_'


def require_no_tracked_changes(status):
    # User-owned untracked datasets (for example NExT-GQA/) must not block a
    # scoped commit; git add/commit below names only our generated artifacts.
    tracked = [line for line in status.splitlines() if line and not line.startswith('?? ')]
    if tracked:
        raise ValueError('Tracked worktree/index changes exist; commit or stash them before publishing')


def recoverable_staged_batch(status):
    """Return paths for one complete staged artifact batch from a failed prior run."""
    tracked = [line for line in status.splitlines() if line and not line.startswith('?? ')]
    if len(tracked) != len(SUFFIXES):
        return None
    paths = []
    for line in tracked:
        if line[:2] != 'A ':
            return None
        path = line[3:]
        if not path.startswith(RUN_PREFIX):
            return None
        paths.append(path)
    prefixes = set()
    for path in paths:
        suffix = next((suffix for suffix in SUFFIXES if path.endswith(suffix)), None)
        if suffix is None:
            return None
        prefixes.add(path[:-len(suffix)])
    if len(prefixes) != 1 or {p.rsplit('_', 1)[-1] for p in paths} != {
            'estimates.csv', 'forest.svg', 'metadata.json'}:
        return None
    return paths


def output_paths(repo, prefix):
    path = (repo / prefix).resolve()
    results = (repo / 'results').resolve()
    if path.parent != results:
        raise ValueError('--output-prefix must be a filename prefix directly inside results/')
    outputs = [Path(str(path) + suffix) for suffix in SUFFIXES]
    existing = [p for p in outputs if p.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite: ' + ', '.join(map(str, existing)))
    return outputs


def test_environment(repo, base=None):
    env = dict(os.environ if base is None else base)
    paths = [str(repo / 'scripts')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    return env


def git_output(repo, *command):
    return subprocess.run(['git', *command], cwd=str(repo), text=True,
                          capture_output=True, check=True).stdout.strip()


def synchronize_clean_branch(repo, branch):
    """Fetch first and only fast-forward a clean branch; refuse divergence."""
    run(['git', 'fetch', 'origin', branch], repo)
    remote_ref = f'origin/{branch}'
    exists = subprocess.run(['git', 'rev-parse', '--verify', remote_ref], cwd=str(repo),
                            text=True, capture_output=True, check=False)
    if exists.returncode:
        raise ValueError(f'Missing remote branch {remote_ref}')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', remote_ref, 'HEAD'],
                       cwd=str(repo), check=False).returncode == 0:
        return  # Current branch is at or ahead of origin.
    if subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', remote_ref],
                      cwd=str(repo), check=False).returncode == 0:
        run(['git', 'merge', '--ff-only', remote_ref], repo)
        return
    raise ValueError(f'Local branch and {remote_ref} diverged; refusing to run or publish')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix',
                        help='Unique path under results/, without file suffix. Defaults to a UTC timestamp.')
    parser.add_argument('--no-push', action='store_true',
                        help='Create the scoped commit but do not push it.')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    try:
        status = git_output(repo, 'status', '--porcelain')
        previous_batch = recoverable_staged_batch(status)
        if previous_batch:
            run(['git', 'restore', '--staged', '--', *previous_batch], repo)
            print('Recovered a complete staged artifact batch from the prior failed run; '
                  'kept its files and will generate a fresh batch.', flush=True)
            status = git_output(repo, 'status', '--porcelain')
        require_no_tracked_changes(status)
        branch = git_output(repo, 'branch', '--show-current')
        if not branch:
            raise ValueError('Detached HEAD: refusing to publish')
        if 'origin' not in git_output(repo, 'remote').splitlines():
            raise ValueError("Git remote 'origin' is not configured")
        git_identity(repo)
        synchronize_clean_branch(repo, branch)
        for relative in INPUTS:
            if not (repo / relative).is_file():
                raise FileNotFoundError(f'Missing required result: {relative}')

        prefix = args.output_prefix or (
            'results/necessity_choice_summary_ubuntu_' +
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        outputs = output_paths(repo, prefix)
    except (ValueError, FileNotFoundError, FileExistsError, SystemExit) as exc:
        parser.error(str(exc))

    test_command = [sys.executable, '-m', 'unittest', *TESTS]
    print('Running:', ' '.join(test_command), flush=True)
    subprocess.run(test_command, cwd=str(repo), env=test_environment(repo), check=True)
    run([
        sys.executable, str(repo / 'scripts/plot_necessity_choice_results.py'),
        '--black', str(repo / INPUTS[0]),
        '--blur', str(repo / INPUTS[1]),
        '--comparison', str(repo / INPUTS[2]),
        '--output-prefix', str(outputs[0])[:-len(SUFFIXES[0])],
    ], repo)

    relative_outputs = [str(path.relative_to(repo)) for path in outputs]
    run(['git', 'add', '-f', *relative_outputs], repo)
    staged = git_output(repo, 'diff', '--cached', '--name-only').splitlines()
    if set(staged) != set(relative_outputs):
        raise SystemExit(f'Refusing unexpected staged paths: {staged}')
    whitespace_check = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=str(repo),
                                       text=True, capture_output=True, check=False)
    if whitespace_check.returncode:
        details = (whitespace_check.stdout + whitespace_check.stderr).strip()
        raise SystemExit(f'Staged output whitespace check failed:\n{details}')
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    run(['git', 'commit', '--only', '-m', f'Publish thesis summary artifacts ({stamp})',
         *relative_outputs], repo)
    if not args.no_push:
        push_with_rebase(repo, branch)
    print('Published: ' + ', '.join(relative_outputs), flush=True)
    print(f'branch={branch} commit={git_output(repo, "rev-parse", "--short", "HEAD")}', flush=True)


if __name__ == '__main__':
    main()
