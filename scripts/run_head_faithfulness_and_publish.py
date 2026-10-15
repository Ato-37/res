"""Run the head-faithfulness experiment and publish only derived anonymous data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_task_validity_audit_and_publish import git_identity, push_with_rebase
from closed_research_branch import reject_inference


def run(command, repo):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def reject_sensitive_keys(value, path='result'):
    forbidden = {'video', 'video_id', 'question', 'choices', 'answer', 'qid', 'prompt'}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden or 'path' in str(key).lower():
                raise SystemExit(f'Refusing publication: sensitive field at {path}.{key}')
            reject_sensitive_keys(child, f'{path}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_sensitive_keys(child, f'{path}[{index}]')


def validate_result_completeness(result, requested):
    if (result.get('n_requested') != requested
            or result.get('n_effective', 0) + result.get('n_failures', 0) != requested):
        raise SystemExit('result completeness check failed; artifact preserved locally, not published')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--quant', choices=['4bit', '8bit', 'none'], default='4bit')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--control-repeats', type=int, default=5)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--frame-size', type=int, default=224)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--exploratory', action='store_true',
                        help='label non-primary head settings as exploratory')
    args = parser.parse_args()
    reject_inference(parser, "closed_head_faithfulness",
                     "the preregistered head confirmation failed its practical threshold")
    repo = Path(__file__).resolve().parents[1]
    sample_arg = Path(args.samples_file).expanduser()
    sample_path = (sample_arg if sample_arg.is_absolute() else repo / sample_arg).resolve()
    output_path = (repo / args.output).resolve()
    if repo not in output_path.parents:
        parser.error('output path must be inside the repository')
    if not sample_path.is_file():
        parser.error(f'missing sample manifest: {args.samples_file}')
    if not output_path.is_relative_to(repo / 'results'):
        parser.error('output must be inside results/')
    if output_path.exists():
        parser.error(f'refusing to overwrite existing output: {args.output}')
    samples = json.loads(sample_path.read_text())
    if args.n_samples is not None and args.n_samples > len(samples):
        parser.error(f'requested {args.n_samples} samples but manifest has only {len(samples)}')
    requested = len(samples) if args.n_samples is None else args.n_samples
    if requested < 1:
        parser.error('no samples selected')
    git_identity(str(repo))
    source_commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True,
        capture_output=True, check=True).stdout.strip()
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo, text=True,
                            capture_output=True, check=True).stdout.strip()
    if not branch:
        parser.error('detached HEAD: refusing to run a job that cannot be published')
    tracked_changes = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                     cwd=repo, text=True, capture_output=True,
                                     check=True).stdout.strip()
    staged = subprocess.run(['git', 'diff', '--cached', '--name-only'], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if tracked_changes or staged:
        parser.error('tracked worktree and index must be clean before GPU inference')

    manifest_path = output_path.with_name(output_path.stem + '_run_manifest.json')
    if manifest_path.exists():
        parser.error(f'refusing to overwrite existing run manifest: {manifest_path.name}')
    command = [sys.executable, str(repo / 'scripts/head_faithfulness_test.py'),
               '--samples-file', str(sample_path), '--output', str(output_path),
               '--n-samples', str(requested), '--layers', args.layers,
               '--quant', args.quant, '--seed', str(args.seed), '--top-k', str(args.top_k),
               '--alpha', str(args.alpha), '--control-repeats', str(args.control_repeats),
               '--num-frames', str(args.num_frames), '--model', args.model]
    command += ['--frame-size', str(args.frame_size)]
    if args.exploratory:
        command.append('--exploratory')
    run(command, str(repo))
    result = json.loads(output_path.read_text())
    result_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    reject_sensitive_keys(result)
    validate_result_completeness(result, requested)
    manifest_path.write_text(json.dumps({
        'experiment': 'gradient_activation_head_faithfulness',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': source_commit,
        'result_sha256': result_sha256,
        'command_args': {
            'script': 'scripts/head_faithfulness_test.py',
            'n_samples': requested,
            'layers': args.layers,
            'quant': args.quant,
            'seed': args.seed,
            'top_k': args.top_k,
            'alpha': args.alpha,
            'control_repeats': args.control_repeats,
            'num_frames': args.num_frames,
            'frame_size': args.frame_size,
            'exploratory': args.exploratory,
        },
        'samples_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        'model': args.model,
        'quant': args.quant,
        'seed': args.seed,
        'layers': args.layers,
        'top_k': args.top_k,
        'alpha': args.alpha,
        'control_repeats': args.control_repeats,
        'exploratory': args.exploratory,
        'num_frames': args.num_frames,
        'frame_size': args.frame_size,
        'cap_pixels_per_frame': True,
        'max_pixels': 256 * 28 * 28,
        'n_requested': requested,
        'n_effective': result['n_effective'],
        'n_failures': result['n_failures'],
    }, indent=2) + '\n')
    relative_files = [str(path.relative_to(repo)) for path in (output_path, manifest_path)]
    run(['git', 'add', '-f', *relative_files], str(repo))
    run(['git', 'diff', '--cached', '--check', '--', *relative_files], str(repo))
    run(['git', 'commit', '--only', '-m', 'Publish head faithfulness experiment',
         *relative_files], str(repo))
    push_with_rebase(str(repo), branch)
    print(f'Published anonymous derived artifacts: {", ".join(relative_files)}', flush=True)


if __name__ == '__main__':
    main()
