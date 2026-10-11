"""Run the NExT-GQA multiple-choice task-validity audit and publish it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def run(command, repo):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def git_identity(repo):
    result = []
    for key in ('user.name', 'user.email'):
        item = subprocess.run(['git', 'config', '--local', '--get', key], cwd=repo,
                              text=True, capture_output=True, check=False)
        if item.returncode or not item.stdout.strip():
            raise SystemExit(f'Missing local Git {key}; configure it before starting the audit.')
        result.append(item.stdout.strip())
    return result


def push_with_rebase(repo, branch):
    for attempt in range(1, 4):
        if subprocess.run(['git', 'push', 'origin', branch], cwd=repo, check=False).returncode == 0:
            return
        print(f'Push attempt {attempt} failed; fetching origin/{branch}.', flush=True)
        run(['git', 'fetch', 'origin', branch], repo)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                               text=True, capture_output=True, check=True).stdout.strip()
        if dirty:
            raise SystemExit('Push rejected while worktree is dirty. Audit commit remains local; '
                             'preserve changes and rebase/push manually.')
        remote = f'origin/{branch}'
        if subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD', remote],
                          cwd=repo, check=False).returncode == 0:
            run(['git', 'merge', '--ff-only', remote], repo)
            return
        if subprocess.run(['git', 'merge-base', '--is-ancestor', remote, 'HEAD'],
                          cwd=repo, check=False).returncode != 0:
            if subprocess.run(['git', 'rebase', remote], cwd=repo, check=False).returncode:
                raise SystemExit('Rebase conflict; audit commit is preserved. Resolve manually; '
                                 'do not rerun model inference.')
    raise SystemExit('Push retries exhausted; audit commit remains local. Do not rerun inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--probe-results', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--map-json', required=True)
    parser.add_argument('--gsub-json', required=True)
    parser.add_argument('--output', default='results/evidence_probe_task_validity_audit.json')
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = {name: (repo / value).resolve() for name, value in {
        'samples': args.samples_file, 'probe': args.probe_results,
        'csv': args.test_csv, 'map': args.map_json, 'gsub': args.gsub_json,
        'output': args.output,
    }.items()}
    if any(repo not in path.parents for path in paths.values()):
        parser.error('All input/output paths must be inside the repository')
    if not paths['output'].is_relative_to(repo / 'results'):
        parser.error('Output must be inside results/')
    for name, path in paths.items():
        if name != 'output' and not path.is_file():
            parser.error(f'Missing {name}: {path}')
    if paths['output'].exists():
        parser.error(f'Refusing to overwrite existing output: {paths["output"]}')
    name, email = git_identity(str(repo))
    command = [sys.executable, str(repo / 'scripts/audit_evidence_probe_task_validity.py'),
               '--samples-file', str(paths['samples']), '--probe-results', str(paths['probe']),
               '--test-csv', str(paths['csv']), '--map-json', str(paths['map']),
               '--gsub-json', str(paths['gsub']), '--output', str(paths['output']),
               '--quant', args.quant, '--seed', str(args.seed)]
    if args.n_samples is not None:
        command.extend(['--n-samples', str(args.n_samples)])
    run(command, str(repo))
    manifest = paths['output'].with_name(paths['output'].stem + '_run_manifest.json')
    manifest.write_text(json.dumps({
        'script': 'run_task_validity_audit_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(paths['samples'].relative_to(repo)),
        'probe_results': str(paths['probe'].relative_to(repo)),
        'test_csv': str(paths['csv'].relative_to(repo)),
        'map_json': str(paths['map'].relative_to(repo)),
        'gsub_json': str(paths['gsub'].relative_to(repo)),
        'n_samples': args.n_samples, 'quant': args.quant, 'seed': args.seed,
        'command': command, 'git_user_name': name, 'git_user_email': email,
    }, indent=2) + '\n')
    relative = [str(path.relative_to(repo)) for path in (paths['output'], manifest)]
    run(['git', 'add', '-f', *relative], str(repo))
    run(['git', 'commit', '--only', '-m', 'Publish evidence probe task-validity audit',
         *relative], str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=str(repo),
                            text=True, capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD: refusing to push')
    push_with_rebase(str(repo), branch)
    print(f'Published {paths["output"].relative_to(repo)} and {manifest.relative_to(repo)}')


if __name__ == '__main__':
    main()
