"""Create a reproducible CSV and SVG summary of necessity mask-fill results.

The figure shows pointwise bootstrap intervals for black/blur conditions.
The companion CSV also includes paired blur-minus-black contrasts. No plotting
package is required; the SVG is emitted with the Python standard library.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

COHORTS = [
    ('all', 'All samples'),
    ('baseline_correct_primary', 'Baseline correct (primary)'),
    ('baseline_incorrect_secondary', 'Baseline incorrect (secondary)'),
]


def validate_pairing(black, blur, comparison):
    black_records = black.get('records', [])
    blur_records = blur.get('records', [])
    black_by_id = {row.get('sample_id'): row for row in black_records}
    blur_by_id = {row.get('sample_id'): row for row in blur_records}
    if None in black_by_id or None in blur_by_id:
        raise ValueError('Each record must have a sample_id')
    if len(black_by_id) != len(black_records) or len(blur_by_id) != len(blur_records):
        raise ValueError('Duplicate sample_id in input records')
    if black_by_id.keys() != blur_by_id.keys():
        raise ValueError('Black and blur results do not contain identical sample IDs')
    match_fields = ('video_id', 'qid', 'sampled_indices', 'evidence_mask_indices',
                    'control_mask_indices', 'masked_frame_count',
                    'baseline_choice_correct', 'gold_choice_index')
    for sample_id in black_by_id:
        for field in match_fields:
            if black_by_id[sample_id].get(field) != blur_by_id[sample_id].get(field):
                raise ValueError(f'Paired inputs disagree for {sample_id}: {field}')
    if comparison.get('n_paired') != len(black_records):
        raise ValueError('Comparison report does not cover all paired records')
    return len(black_records)


def csv_text(black, blur, comparison):
    stream = io.StringIO(newline='')
    # Git's diff --check treats CR from csv.writer's default CRLF as trailing
    # whitespace in generated artifacts; write portable LF-only text instead.
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['cohort', 'condition', 'n', 'mean', 'sd', 'ci95_low',
                     'ci95_high', 'signflip_p_two_sided', 'positive_fraction'])
    sources = [('black', black['summary']), ('blur', blur['summary'])]
    for cohort, _ in COHORTS:
        for condition, summaries in sources:
            s = summaries[cohort]
            ci = s.get('ci95_pointwise') or [None, None]
            writer.writerow([cohort, condition, s.get('n'), s.get('mean'), s.get('sd'),
                             ci[0], ci[1], s.get('signflip_p_two_sided'),
                             s.get('positive_fraction')])
        s = comparison['comparisons'][cohort]['blur_minus_black']
        ci = s.get('ci95_pointwise') or [None, None]
        writer.writerow([cohort, 'blur_minus_black', s.get('n'), s.get('mean'), s.get('sd'),
                         ci[0], ci[1], s.get('signflip_p_two_sided'),
                         s.get('positive_fraction')])
    return stream.getvalue()


def svg_text(black, blur, comparison):
    chart_rows = []
    all_bounds = [0.0]
    for cohort, label in COHORTS:
        for condition, result, y_offset, color in (
            ('black', black, -10, '#34495e'),
            ('blur', blur, 10, '#d35400'),
        ):
            s = result['summary'][cohort]
            ci = s.get('ci95_pointwise')
            if s.get('n', 0) and ci:
                chart_rows.append((label, condition, s, y_offset, color))
                all_bounds.extend([ci[0], ci[1]])
    if not chart_rows:
        raise ValueError('No cohort has an interval available to plot')
    xmin, xmax = min(all_bounds), max(all_bounds)
    pad = max((xmax - xmin) * 0.1, 0.05)
    xmin, xmax = xmin - pad, xmax + pad

    width, height = 1000, 430
    left, right, top, bottom = 300, 940, 95, 333
    scale = lambda x: left + (x - xmin) * (right - left) / (xmax - xmin)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<title>Necessity choice-margin mask-fill results</title>',
        '<desc>Pointwise 95% bootstrap intervals for black and Gaussian blur fill conditions. '
        'The paired fill difference is tabulated in the companion CSV.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}'
        '.small{font-size:13px}.axis{font-size:12px}.cohort{font-size:15px;font-weight:600}'
        '.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}</style>',
        '<text x="40" y="38" class="title">Grounding-mask choice-margin sensitivity</text>',
        '<text x="40" y="61" class="subtitle">NExT-GQA · Qwen2.5-VL-7B-Instruct · '
        '4-bit BF16 · 16 sampled frames</text>',
    ]
    # Vertical axis grid and zero reference.
    tick_count = 6
    ticks = [xmin + i * (xmax - xmin) / (tick_count - 1) for i in range(tick_count)]
    for tick in ticks:
        x = scale(tick)
        stroke = '#9ca3af' if abs(tick) < (xmax - xmin) / 1000 else '#e5e7eb'
        dash = ' stroke-dasharray="5,4"' if abs(tick) < (xmax - xmin) / 1000 else ''
        chunks.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
                      f'stroke="{stroke}"{dash}/>')
        chunks.append(f'<text x="{x:.2f}" y="{bottom+20}" text-anchor="middle" '
                      f'class="axis">{tick:.2f}</text>')
    zero_x = scale(0.0)
    chunks.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{bottom}" '
                  'stroke="#111827" stroke-width="1.5" stroke-dasharray="5,4"/>')
    row_y = {label: 142 + i * 74 for i, (_, label) in enumerate(COHORTS)}
    for _, label in COHORTS:
        y = row_y[label]
        chunks.append(f'<text x="40" y="{y+5}" class="cohort">{escape(label)}</text>')
    for label, condition, summary, y_offset, color in chart_rows:
        y = row_y[label] + y_offset
        mean = summary['mean']
        lo, hi = summary['ci95_pointwise']
        x1, x2, xm = scale(lo), scale(hi), scale(mean)
        chunks.append(f'<line x1="{x1:.2f}" y1="{y}" x2="{x2:.2f}" y2="{y}" '
                      f'stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<line x1="{x1:.2f}" y1="{y-5}" x2="{x1:.2f}" y2="{y+5}" '
                      f'stroke="{color}" stroke-width="2"/>')
        chunks.append(f'<line x1="{x2:.2f}" y1="{y-5}" x2="{x2:.2f}" y2="{y+5}" '
                      f'stroke="{color}" stroke-width="2"/>')
        chunks.append(f'<circle cx="{xm:.2f}" cy="{y}" r="5" fill="{color}"/>')
        chunks.append(f'<text x="{right+10}" y="{y+4}" class="small">n={summary["n"]}</text>')
    chunks.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#6b7280"/>',
        f'<text x="{(left+right)/2:.0f}" y="{bottom+46}" text-anchor="middle" class="small">'
        'Selective necessity margin drop (log-prob margin units)</text>',
        '<circle cx="640" cy="390" r="5" fill="#34495e"/><text x="652" y="394" '
        'class="small">Black mask</text>',
        '<circle cx="765" cy="390" r="5" fill="#d35400"/><text x="777" y="394" '
        'class="small">Gaussian blur (radius 12)</text>',
        '<text x="40" y="418" class="axis">Intervals are pointwise percentile bootstrap 95% CIs; '
        'paired blur−black contrasts and p-values are in the CSV.</text>',
        '</svg>',
    ])
    return '\n'.join(chunks) + '\n'


def artifact_metadata(input_paths, black, blur, comparison):
    hashes = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in input_paths}
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        'script': 'plot_necessity_choice_results.py',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_git_commit': commit,
        'input_sha256': hashes,
        'model': black.get('model'),
        'quant': black.get('quant'),
        'dtype': black.get('dtype'),
        'num_frames': black.get('num_frames'),
        'n_paired': comparison.get('n_paired'),
        'blur_intervention': blur.get('mask_intervention'),
        'interval': 'pointwise percentile bootstrap 95%',
        'multiplicity': 'No correction across cohorts or fill comparisons.',
        'limitations': [
            'Black and blur conditions are perturbations and may be out of distribution.',
            'The primary estimate is conditioned on baseline-correct samples.',
            'The fill comparison uses the same videos and is not an independent replication.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--black', required=True)
    parser.add_argument('--blur', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()
    inputs = [Path(args.black), Path(args.blur), Path(args.comparison)]
    for path in inputs:
        if not path.is_file():
            parser.error(f'Missing input JSON: {path}')
    prefix = Path(args.output_prefix)
    outputs = [Path(str(prefix) + suffix) for suffix in
               ('_estimates.csv', '_forest.svg', '_metadata.json')]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error('Refusing to overwrite: ' + ', '.join(existing))
    black, blur, comparison = [json.loads(path.read_text()) for path in inputs]
    try:
        n_paired = validate_pairing(black, blur, comparison)
    except ValueError as exc:
        parser.error(str(exc))
    csv_data = csv_text(black, blur, comparison)
    svg_data = svg_text(black, blur, comparison)
    metadata = artifact_metadata(inputs, black, blur, comparison)
    for path, content in zip(outputs, (csv_data, svg_data,
                                        json.dumps(metadata, indent=2) + '\n')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f'created={datetime.now(timezone.utc).isoformat()}')
    print('outputs=' + ', '.join(map(str, outputs)))
    print(f"source_git_commit={metadata['source_git_commit']} paired_n={n_paired}")


if __name__ == '__main__':
    main()
