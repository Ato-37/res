"""Record nvidia-smi telemetry while a bounded BF16 CUDA workload runs."""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import torch

def sample_gpu() -> dict:
    cmd=['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free,power.draw','--format=csv,noheader,nounits']
    raw=subprocess.check_output(cmd,text=True).strip().split(',')
    return {'gpu_util_percent':int(raw[0].strip()),'memory_used_mib':int(raw[1].strip()),'memory_free_mib':int(raw[2].strip()),'power_watts':float(raw[3].strip())}

def run(seconds: float, size: int, interval: float) -> dict:
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    a=torch.randn((size,size),device='cuda',dtype=torch.bfloat16); b=torch.randn_like(a)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter(); samples=[]; iters=0; next_sample=start
    while time.perf_counter()-start < seconds:
        _=a@b; iters+=1
        now=time.perf_counter()
        if now >= next_sample:
            try: samples.append({'elapsed_seconds':now-start,**sample_gpu()})
            except (OSError, subprocess.SubprocessError, ValueError): pass
            next_sample=now+interval
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    try: source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError, subprocess.SubprocessError): source_commit=None
    return {'schema_version':1,'diagnostic':'cuda_bf16_utilization_probe','source_commit':source_commit,'size':size,'target_seconds':seconds,'elapsed_seconds':elapsed,'iterations':iters,'dtype':'bfloat16','torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),'telemetry_samples':samples,'status':'engineering_gpu_path_only','limitations':['Synthetic dense matmul only; no model, video, QA, or faithfulness claim.']}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--seconds',type=float,default=5); p.add_argument('--size',type=int,default=8192); p.add_argument('--interval',type=float,default=.5); p.add_argument('--output',type=Path,required=True); a=p.parse_args(argv)
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(Path.cwd()/'results'): raise SystemExit('new output under results/ required')
    out=run(a.seconds,a.size,a.interval); output.write_text(json.dumps(out,indent=2)+'\n'); output.chmod(0o600); print(json.dumps({'status':out['status'],'iterations':out['iterations'],'telemetry_samples':len(out['telemetry_samples']),'output':str(output)}))
if __name__=='__main__': raise SystemExit(main())
