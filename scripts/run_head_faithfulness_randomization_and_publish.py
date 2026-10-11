"""Run and publish the limited output-readout randomization sanity check."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_task_validity_audit_and_publish import git_identity, push_with_rebase


FORBIDDEN_KEYS = {'video', 'video_id', 'question', 'choices', 'answer', 'qid', 'prompt'}


def reject_private_fields(value, location='result'):
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            if name in FORBIDDEN_KEYS or 'path' in name:
                raise SystemExit(f'refusing publication of sensitive field {location}.{key}')
            reject_private_fields(child, f'{location}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private_fields(child, f'{location}[{index}]')


def validate_counts(result, requested):
    if (result.get('n_requested') != requested
            or result.get('n_effective', 0) + result.get('n_failures', 0) != requested
            or len(result.get('records', [])) != result.get('n_effective')):
        raise SystemExit('result completeness check failed; artifact preserved locally')


def run(command, repo):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=repo, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--randomizations', type=int, default=1)
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--require-success', action='store_true',
                        help='exit nonzero after publishing if any sample failed')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sample_arg = Path(args.samples_file).expanduser()
    sample_path = (sample_arg if sample_arg.is_absolute() else repo / sample_arg).resolve()
    output_path = (repo / args.output).resolve()
    if not sample_path.is_file():
        parser.error(f'missing sample manifest: {sample_path}')
    if not output_path.is_relative_to(repo / 'results'):
        parser.error('output must be under results/')
    if output_path.exists():
        parser.error(f'refusing to overwrite existing output: {args.output}')
    samples = json.loads(sample_path.read_text())
    if args.n_samples < 1 or args.n_samples > len(samples):
        parser.error('requested sample count is outside the manifest')
    if args.randomizations < 1:
        parser.error('--randomizations must be positive')
    manifest_path = output_path.with_name(output_path.stem + '_run_manifest.json')
    if manifest_path.exists():
        parser.error(f'refusing to overwrite existing run manifest: {manifest_path.name}')

    git_identity(str(repo))
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    tracked = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                             cwd=repo, text=True, capture_output=True,
                             check=True).stdout.strip()
    staged = subprocess.run(['git', 'diff', '--cached', '--name-only'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if not branch or tracked or staged:
        parser.error('requires a named branch and clean tracked worktree/index')

    command = [sys.executable, repo / 'scripts/head_faithfulness_readout_randomization.py',
               '--samples-file', sample_path, '--n-samples', str(args.n_samples),
               '--randomizations', str(args.randomizations), '--quant', args.quant,
               '--seed', str(args.seed), '--model', args.model, '--output', output_path]
    run(command, repo)
    result = json.loads(output_path.read_text())
    reject_private_fields(result)
    validate_counts(result, args.n_samples)
    manifest = {
        'experiment': 'head_attribution_output_readout_randomization_sanity',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'samples_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        'model': args.model, 'quant': args.quant, 'seed': args.seed,
        'n_requested': args.n_samples, 'n_effective': result['n_effective'],
        'n_failures': result['n_failures'],
        'randomizations_per_sample': args.randomizations,
        'num_frames': 16, 'layers': 'all',
        'randomized_parameter_scope': 'five A-E output-readout rows only',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    files = [str(path.relative_to(repo)) for path in (output_path, manifest_path)]
    run(['git', 'add', '-f', *files], repo)
    run(['git', 'diff', '--cached', '--check', '--', *files], repo)
    run(['git', 'commit', '--only', '-m', 'Publish head attribution randomization sanity check', *files], repo)
    push_with_rebase(str(repo), branch)
    print('Published anonymous randomization diagnostic:', ', '.join(files), flush=True)
    if args.require_success and result['n_failures']:
        raise SystemExit('diagnostic was published, but failures were recorded; stopping chained work')


if __name__ == '__main__':
    main()
