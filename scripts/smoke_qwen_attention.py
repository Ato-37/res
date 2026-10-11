"""Run one Qwen attention forward on a generated synthetic MP4.

This is an execution/VRAM smoke only. The synthetic overlap score is not a
grounding or faithfulness result.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quant", choices=["4bit"], default="4bit")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qwen_attention_smoke_") as tmp:
        root = Path(tmp)
        video = root / "synthetic.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (224, 224)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create synthetic MP4")
        try:
            for index in range(24):
                color = ((index * 13) % 256, (index * 29) % 256, (index * 47) % 256)
                writer.write(np.full((224, 224, 3), color, dtype=np.uint8))
        finally:
            writer.release()

        sample_file = root / "synthetic_sample.json"
        sample_file.write_text(json.dumps([{
            "video": str(video),
            "question": "Describe only the synthetic color changes.",
            "answer": "The synthetic colors change.",
            "grounding": [[0.0, 2.875]],
        }]), encoding="utf-8")
        command = [
            sys.executable, str(repo / "scripts/grounding_check.py"),
            "--samples-file", str(sample_file),
            "--model", args.model,
            "--quant", args.quant,
            "--num-frames", "4",
            "--frame-size", "224",
            "--max-pixels", str(256 * 28 * 28),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(
                "Synthetic model forward failed.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if "frames_requested=4, frames_effective=4" not in result.stdout:
            raise RuntimeError("Smoke did not evaluate exactly four requested frames")
        if "attention-overlap proxy (peak temporal-chunk intersection; not faithfulness):" not in result.stdout or "/1 =" not in result.stdout:
            raise RuntimeError("Smoke did not complete exactly one model forward")

        selected = [
            line.strip() for line in result.stdout.splitlines()
            if line.strip().startswith(("device=", "t_chunks=", "gpu_peak_memory_allocated_gib="))
        ]
        print(
            "Qwen synthetic attention forward passed: requested/effective frames=4/4; "
            "synthetic overlap score intentionally discarded."
        )
        for line in selected:
            print(line)


if __name__ == "__main__":
    main()
