"""Run the paired choice-margin intervention probe and publish its outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from closed_research_branch import reject_inference

from run_task_validity_audit_and_publish import git_identity, push_with_rebase


def run(command, repo):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--probe-results', required=True)
    parser.add_argument('--choice-audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    reject_inference(parser, "legacy_evidence_control",
                     "this exploratory single-frame protocol is not the current preregistered estimand")
    repo = Path(__file__).resolve().parents[1]
    paths = {key: (repo / value).resolve() for key, value in {
        'samples': args.samples_file, 'probe': args.probe_results,
        'audit': args.choice_audit, 'output': args.output}.items()}
    if any(repo not in p.parents for p in paths.values()):
        parser.error('All paths must be inside the repository')
    if not paths['output'].is_relative_to(repo / 'results'):
        parser.error('Output must be inside results/')
    for key, path in paths.items():
        if key != 'output' and not path.is_file():
            parser.error(f'Missing {key}: {path}')
    if paths['output'].exists():
        parser.error(f'Refusing to overwrite existing output: {paths["output"]}')
    name, email = git_identity(str(repo))
    command = [sys.executable, str(repo / 'scripts/evidence_control_choice_probe.py'),
               '--samples-file', str(paths['samples']), '--probe-results', str(paths['probe']),
               '--choice-audit', str(paths['audit']), '--output', str(paths['output']),
               '--quant', args.quant, '--seed', str(args.seed)]
    if args.n_samples is not None:
        command += ['--n-samples', str(args.n_samples)]
    run(command, str(repo))
    manifest = paths['output'].with_name(paths['output'].stem + '_run_manifest.json')
    manifest.write_text(json.dumps({
        'script': 'run_evidence_control_choice_probe_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(paths['samples'].relative_to(repo)),
        'probe_results': str(paths['probe'].relative_to(repo)),
        'choice_audit': str(paths['audit'].relative_to(repo)),
        'output': str(paths['output'].relative_to(repo)),
        'n_samples': args.n_samples, 'quant': args.quant, 'seed': args.seed,
        'command': command, 'git_user_name': name, 'git_user_email': email,
    }, indent=2) + '\n')
    files = [str(p.relative_to(repo)) for p in (paths['output'], manifest)]
    run(['git', 'add', '-f', *files], str(repo))
    run(['git', 'commit', '--only', '-m', 'Publish choice-margin evidence probe', *files], str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo, text=True,
                            capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD: refusing to push')
    push_with_rebase(str(repo), branch)
    print(f'Published {paths["output"].relative_to(repo)} and {manifest.relative_to(repo)}')


if __name__ == '__main__':
    main()
