"""Run fixed 3-scope audit and paired analysis without overwriting old results.

Usage: python scripts/run_patch_scope_audit.py \
    --reference results/activation_patching_confirm_heldout.json \
    --output-dir results/patch_scope_audit_25

Reads the sample manifest path from the previous result, not a guessed filename.
Run from the repository root. GPU required for sweep subprocesses.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(samples, output_dir, n_samples):
    scripts = Path(__file__).resolve().parent
    paths = {s: str(output_dir / (s + '.json')) for s in ('all', 'prompt', 'answer')}
    runs = [[sys.executable, str(scripts / 'activation_patching_sweep.py'),
             '--samples-file', str(samples), '--n-samples', str(n_samples),
             '--num-frames', '16', '--layers', '0,6,12,24,27',
             '--patch-scope', scope, '--self-check', '--strict-grounding',
             '--output', path] for scope, path in paths.items()]
    runs.append([sys.executable, str(scripts / 'compare_activation_patch_scopes.py'),
                 '--all', paths['all'], '--prompt', paths['prompt'], '--answer', paths['answer'],
                 '--output', str(output_dir / 'comparison.json')])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-samples', type=int, default=25)
    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error('--n-samples must be positive')
    reference = json.loads(Path(args.reference).read_text())
    samples = Path(reference['samples_file']).expanduser()
    manifest = json.loads(samples.read_text())
    if len(manifest) < args.n_samples:
        parser.error(f'Manifest contains only {len(manifest)} samples')
    missing = [r['video'] for r in manifest[:args.n_samples]
               if not Path(r['video']).expanduser().is_file()]
    if missing:
        parser.error(f'Missing video files: {missing}')
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error('Output directory already exists; choose a new directory to preserve results')
    output_dir.mkdir(parents=True)
    for command in commands(samples, output_dir, args.n_samples):
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)
    print(f'Completed: {output_dir / "comparison.json"}')


if __name__ == '__main__':
    main()
