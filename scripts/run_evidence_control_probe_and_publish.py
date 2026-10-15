"""Run the evidence-vs-placebo input probe, analyze, and safely publish it.

Unlike activation-patching sweeps, this evaluates video-input perturbations only;
no head/layer intervention is performed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from closed_research_branch import reject_inference


def run(command, repo):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def git_identity(repo):
    values = []
    for key in ('user.name', 'user.email'):
        result = subprocess.run(['git', 'config', '--local', '--get', key], cwd=repo,
                                text=True, capture_output=True, check=False)
        if result.returncode or not result.stdout.strip():
            raise SystemExit(f'Missing local Git {key}; configure it before starting the run.')
        values.append(result.stdout.strip())
    return values


def push_with_rebase(repo, branch):
    for attempt in range(1, 4):
        pushed = subprocess.run(['git', 'push', 'origin', branch], cwd=repo, check=False)
        if pushed.returncode == 0:
            return
        print(f'Push attempt {attempt} failed; fetching origin/{branch}.', flush=True)
        run(['git', 'fetch', 'origin', branch], repo)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                               text=True, capture_output=True, check=True).stdout.strip()
        if dirty:
            raise SystemExit('Push rejected with a dirty worktree; result commit remains local. '
                             'Preserve changes, then fetch/rebase/push manually.')
        remote = f'origin/{branch}'
        if subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', remote],
                          cwd=repo, check=False).returncode == 0:
            run(['git', 'merge', '--ff-only', remote], repo)
            return
        if subprocess.run(['git', 'merge-base', '--is-ancestor', remote, 'HEAD'],
                          cwd=repo, check=False).returncode != 0:
            rebased = subprocess.run(['git', 'rebase', remote], cwd=repo, check=False)
            if rebased.returncode:
                raise SystemExit('Rebase conflict; local result commit is preserved. '
                                 'Resolve it manually; do not rerun the GPU probe.')
    raise SystemExit('Push retries exhausted; inspect git status/log. Do not rerun the probe.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--n-samples', type=int, default=40)
    parser.add_argument('--placebo-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--output', default='results/evidence_control_probe.json')
    parser.add_argument('--commit-message', default='Publish evidence versus placebo probe')
    args = parser.parse_args()
    reject_inference(parser, "legacy_evidence_control",
                     "this exploratory single-frame protocol is not the current preregistered estimand")
    repo = Path(__file__).resolve().parents[1]
    samples = (repo / args.samples_file).resolve()
    output = (repo / args.output).resolve()
    if not samples.is_file():
        parser.error(f'Missing sample file: {samples}')
    if repo not in samples.parents or repo not in output.parents:
        parser.error('Sample and output paths must be inside the repository')
    if not output.is_relative_to(repo / 'results'):
        parser.error('Output must be inside results/')
    if output.exists():
        parser.error(f'Refusing to overwrite existing result: {output}')
    name, email = git_identity(str(repo))
    command = [sys.executable, str(repo / 'scripts/evidence_control_probe.py'),
               '--samples-file', str(samples), '--n-samples', str(args.n_samples),
               '--placebo-repeats', str(args.placebo_repeats), '--seed', str(args.seed),
               '--quant', args.quant, '--output', str(output)]
    run(command, str(repo))
    manifest = output.with_name(output.stem + '_run_manifest.json')
    manifest.write_text(json.dumps({
        'script': 'run_evidence_control_probe_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(samples.relative_to(repo)),
        'n_samples': args.n_samples, 'placebo_repeats': args.placebo_repeats,
        'seed': args.seed, 'quant': args.quant, 'command': command,
        'git_user_name': name, 'git_user_email': email,
    }, indent=2) + '\n')
    files = [output, manifest]
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch',
                              str(samples.relative_to(repo))], cwd=str(repo),
                             text=True, capture_output=True, check=False)
    if tracked.returncode:
        files.insert(0, samples)
    relative = [str(path.relative_to(repo)) for path in files]
    run(['git', 'add', '-f', *relative], str(repo))
    run(['git', 'commit', '--only', '-m', args.commit_message, *relative], str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=str(repo),
                            text=True, capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD: refusing to push')
    push_with_rebase(str(repo), branch)
    print(f'Published {output.relative_to(repo)} and {manifest.relative_to(repo)}')


if __name__ == '__main__':
    main()
