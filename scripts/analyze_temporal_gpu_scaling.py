"""Summarize privacy-safe temporal GPU scaling artifacts (CPU only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def analyze(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        d = json.loads(path.read_text())
        required = {'num_frames','status','n_requested','n_effective','n_failures','peak_cuda_memory_bytes'}
        if not required.issubset(d): raise ValueError('missing scaling fields')
        if type(d['num_frames']) is not int or d['num_frames'] < 1: raise ValueError('invalid frame count')
        if d['status'] not in {'engineering_smoke_only','engineering_smoke_failure'}: raise ValueError('invalid status')
        if d['n_effective'] + d['n_failures'] != d['n_requested']: raise ValueError('count mismatch')
        rows.append({k:d[k] for k in ('num_frames','status','n_requested','n_effective','n_failures','peak_cuda_memory_bytes')})
    rows.sort(key=lambda r:r['num_frames'])
    return {'schema_version':1,'analysis':'temporal_gpu_scaling_summary','status':'engineering_only','conditions':rows,
            'limitations':['Memory values are allocator observations, not a faithfulness metric.','OOM conditions have no effect estimate.']}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('artifacts',nargs='+',type=Path); p.add_argument('--output',required=True,type=Path); a=p.parse_args(argv)
    if a.output.exists(): raise SystemExit(f'refusing to overwrite {a.output}')
    out=analyze(a.artifacts); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
