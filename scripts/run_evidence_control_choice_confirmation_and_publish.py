"""Run and publish a fresh-sample choice-margin confirmation experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from run_task_validity_audit_and_publish import git_identity, push_with_rebase


def run(command, repo):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--placebo-repeats', type=int, default=5)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sample_path, output_path = (repo / args.samples_file).resolve(), (repo / args.output).resolve()
    if repo not in sample_path.parents or repo not in output_path.parents:
        parser.error('All paths must be inside the repository')
    if not output_path.is_relative_to(repo / 'results'):
        parser.error('Output must be inside results/')
    if not sample_path.is_file():
        parser.error(f'Missing sample manifest: {sample_path}')
    if output_path.exists():
        parser.error(f'Refusing to overwrite existing output: {output_path}')
    name, email = git_identity(str(repo))
    command = [sys.executable, str(repo / 'scripts/evidence_control_choice_confirmation.py'),
               '--samples-file', str(sample_path), '--output', str(output_path),
               '--quant', args.quant, '--seed', str(args.seed),
               '--placebo-repeats', str(args.placebo_repeats)]
    run(command, str(repo))
    manifest = output_path.with_name(output_path.stem + '_run_manifest.json')
    manifest.write_text(json.dumps({
        'script': 'run_evidence_control_choice_confirmation_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(sample_path.relative_to(repo)),
        'output': str(output_path.relative_to(repo)),
        'quant': args.quant, 'seed': args.seed,
        'placebo_repeats': args.placebo_repeats, 'command': command,
        'git_user_name': name, 'git_user_email': email,
    }, indent=2) + '\n')
    files = [str(p.relative_to(repo)) for p in (sample_path, output_path, manifest)]
    run(['git', 'add', '-f', *files], str(repo))
    run(['git', 'commit', '--only', '-m', 'Publish fresh choice-margin confirmation', *files], str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD: refusing to push')
    push_with_rebase(str(repo), branch)
    print(f'Published {sample_path.relative_to(repo)}, {output_path.relative_to(repo)}, '
          f'and {manifest.relative_to(repo)}')


if __name__ == '__main__':
    main()
