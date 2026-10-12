"""Analyze the fixed eager-vs-SDPA Qwen diagnostic contrast."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def analyze(eager_path: Path, sdpa_path: Path) -> dict:
    eager, sdpa = json.loads(eager_path.read_text()), json.loads(sdpa_path.read_text())
    for name, report, expected in (("eager", eager, "eager"), ("sdpa", sdpa, "sdpa")):
        if report.get("status") != "diagnostic_complete" or report.get("failures") != 0:
            raise ValueError(f"{name} run incomplete")
        if report.get("attention_implementation") != expected:
            raise ValueError(f"{name} attention implementation mismatch")
        if report.get("completed_model_calls") != 5 or report.get("effective_comparison_pairs") != 19:
            raise ValueError(f"{name} fixed counts mismatch")
        if report.get("effective_hidden_row_comparisons") != 9 or report.get("numeric_crosscheck_passed") is not True:
            raise ValueError(f"{name} fixed numeric gate mismatch")
        before, after = report["checkpoint_shards_before_model_load"], report["checkpoint_shards_after_model_load"]
        if before["shards"] != after["shards"] or len(before["shards"]) != 5:
            raise ValueError(f"{name} checkpoint witness mismatch")
    for key in ("prompt_ids_sha256", "frozen_target_sha256", "media_sha256", "video_grid_thw"):
        if eager.get(key) != sdpa.get(key):
            raise ValueError(f"fixed input mismatch: {key}")
    contrasts = []
    for left, right in zip(eager["comparisons"], sdpa["comparisons"]):
        if (left["comparison_type"], left["left"], left["right"]) != (right["comparison_type"], right["left"], right["right"]):
            raise ValueError("comparison plan mismatch")
        for value in (left["raw_max_abs"], right["raw_max_abs"]):
            if not math.isfinite(value): raise ValueError("nonfinite comparison")
        contrasts.append({"comparison_type": left["comparison_type"], "left": left["left"], "right": left["right"], "eager_raw_max_abs": left["raw_max_abs"], "sdpa_raw_max_abs": right["raw_max_abs"]})
    return {"analysis": "qwen_attention_implementation_exploratory_control", "fixed_input_equal": True, "comparison_count": len(contrasts), "contrasts": contrasts, "interpretation_limit": "exploratory implementation contrast; no causal-mask or faithfulness claim"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("eager", type=Path); parser.add_argument("sdpa", type=Path); args = parser.parse_args()
    print(json.dumps(analyze(args.eager, args.sdpa), indent=2, sort_keys=True))


if __name__ == "__main__": main()
