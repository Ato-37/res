"""Synthetic BF16 causal-attention shape diagnostic for eager versus SDPA."""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
import torch

SEED = 20261012
LENGTHS = (584, 585, 586)
HEADS, DIM = 8, 64

def _manual(q, k, v):
    scale = 1.0 / math.sqrt(DIM)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)

def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

def run(output: Path, dtype_name: str = "bfloat16") -> dict:
    if output.exists() or not output.is_relative_to(Path.cwd() / "results"):
        raise ValueError("new output under results/ required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    max_len = max(LENGTHS)
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    q = torch.randn((1, HEADS, max_len, DIM), device=device, dtype=dtype)
    k = torch.randn_like(q); v = torch.randn_like(q)
    records = []
    torch.cuda.synchronize(); start = time.monotonic()
    for backend, fn in (("manual_eager", _manual), ("sdpa", _sdpa)):
        outputs = {}
        for length in LENGTHS:
            outputs[length] = fn(q[:, :, :length], k[:, :, :length], v[:, :, :length]).detach()
        for left, right in ((584, 585), (584, 586), (585, 586)):
            delta = (outputs[left][:, :, :min(left, right)] - outputs[right][:, :, :min(left, right)]).float()
            records.append({"backend": backend, "left_length": left, "right_length": right,
                            "raw_max_abs": float(delta.abs().max()), "raw_mean_abs": float(delta.abs().mean()),
                            "finite": bool(torch.isfinite(delta).all())})
    torch.cuda.synchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "purpose": "synthetic_causal_attention_backend_shape_diagnostic",
              "seed": SEED, "lengths": list(LENGTHS), "heads": HEADS, "head_dim": DIM,
              "dtype": dtype_name, "records": records, "status": "complete", "failure_count": 0,
              "elapsed_seconds": time.monotonic() - start,
              "gpu": torch.cuda.get_device_name(0), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "limitations": ["Synthetic Q/K/V only; no Qwen weights, video, QA or faithfulness claim.",
                              "Measures this runtime's causal-attention shape sensitivity, not universal model behavior."]}
    with output.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2); stream.write("\n")
    os.chmod(output, 0o600)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16"); args = parser.parse_args()
    result = run((Path.cwd() / args.output).resolve(), args.dtype)
    print(json.dumps({"status": result["status"], "records": len(result["records"]), "output": args.output}))

if __name__ == "__main__": main()
