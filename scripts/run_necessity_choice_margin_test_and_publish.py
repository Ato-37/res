"""Run and publish the multiple-choice temporal necessity-mask experiment."""
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


def analysis_command(repo, samples, output, args):
    command = [sys.executable, str(repo / 'scripts/necessity_choice_margin_test.py'),
               '--samples-file', str(samples), '--output', str(output),
               '--quant', args.quant, '--seed', str(args.seed),
               '--bootstrap-draws', str(args.bootstrap_draws),
               '--num-frames', str(args.num_frames),
               '--mask-fill', args.mask_fill, '--blur-radius', str(args.blur_radius)]
    if args.n_samples is not None:
        command.extend(['--n-samples', str(args.n_samples)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-report')
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--mask-fill', choices=['black', 'blur'], default='black')
    parser.add_argument('--blur-radius', type=float, default=12.0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    samples, output = (repo / args.samples_file).resolve(), (repo / args.output).resolve()
    preflight = (repo / args.preflight_report).resolve() if args.preflight_report else None
    if repo not in samples.parents or repo not in output.parents:
        parser.error('All paths must be inside the repository')
    if preflight is not None and (repo not in preflight.parents or not preflight.is_file()):
        parser.error('Preflight report must exist inside the repository')
    if not output.is_relative_to(repo / 'results'):
        parser.error('Output must be inside results/')
    if not samples.is_file():
        parser.error(f'Missing sample manifest: {samples}')
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    name, email = git_identity(str(repo))
    command = analysis_command(repo, samples, output, args)
    run(command, str(repo))
    manifest = output.with_name(output.stem + '_run_manifest.json')
    manifest.write_text(json.dumps({
        'script': 'run_necessity_choice_margin_test_and_publish.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_file': str(samples.relative_to(repo)),
        'preflight_report': str(preflight.relative_to(repo)) if preflight else None,
        'output': str(output.relative_to(repo)), 'n_samples': args.n_samples,
        'quant': args.quant, 'seed': args.seed,
        'bootstrap_draws': args.bootstrap_draws, 'num_frames': args.num_frames,
        'mask_fill': args.mask_fill, 'blur_radius': args.blur_radius,
        'command': command, 'git_user_name': name, 'git_user_email': email,
    }, indent=2) + '\n')
    publish_paths = [samples, output, manifest]
    if preflight is not None:
        publish_paths.append(preflight)
    files = [str(p.relative_to(repo)) for p in publish_paths]
    run(['git', 'add', '-f', *files], str(repo))
    run(['git', 'commit', '--only', '-m', 'Publish choice-margin necessity test', *files], str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo, text=True,
                            capture_output=True, check=True).stdout.strip()
    if not branch:
        raise SystemExit('Detached HEAD: refusing to push')
    push_with_rebase(str(repo), branch)
    print(f'Published {output.relative_to(repo)}, {manifest.relative_to(repo)}'
          + (f', and {preflight.relative_to(repo)}' if preflight else ''))


if __name__ == '__main__':
    main()
