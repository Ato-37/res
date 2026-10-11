#!/usr/bin/env python3
"""Run and publish the counts-only temporal-grounding data audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(args, repo, check=True):
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", default="NExT-GQA/datasets/nextgqa/val.csv")
    parser.add_argument("--gsub-val-json", default="NExT-GQA/datasets/nextgqa/gsub_val.json")
    parser.add_argument("--test-csv", default="NExT-GQA/datasets/nextgqa/test.csv")
    parser.add_argument("--gsub-test-json", default="NExT-GQA/datasets/nextgqa/gsub_test.json")
    parser.add_argument("--video-map", default="NExT-GQA/datasets/nextgqa/map_vid_vidorID.json")
    parser.add_argument("--frame2time-val-json", default="NExT-GQA/datasets/nextgqa/frame2time_val.json")
    parser.add_argument("--frame2time-test-json", default="NExT-GQA/datasets/nextgqa/frame2time_test.json")
    parser.add_argument("--exclude-samples", nargs="*", default=None)
    parser.add_argument("--output", default="results/grounding_plan_data_audit_20261009.json")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    values = {
        "val_csv": args.val_csv, "gsub_val_json": args.gsub_val_json,
        "test_csv": args.test_csv, "gsub_test_json": args.gsub_test_json,
        "video_map": args.video_map, "frame2time_val_json": args.frame2time_val_json,
        "frame2time_test_json": args.frame2time_test_json,
    }
    inputs = {key: (repo / value).resolve() for key, value in values.items()}
    if args.exclude_samples is None:
        exclusions = sorted((repo / "data").glob("nextgqa_samples*.json"))
    else:
        exclusions = [(repo / value).resolve() for value in args.exclude_samples]
    output = (repo / args.output).resolve()
    if any(repo not in path.parents for path in [*inputs.values(), *exclusions, output]):
        parser.error("All inputs and output must be inside the repository")
    if not output.is_relative_to(repo / "results"):
        parser.error("Output must be inside results/")
    if any(not path.is_file() for path in [*inputs.values(), *exclusions]):
        parser.error("An input or exclusion manifest is missing")
    manifest = output.with_name(output.stem + "_run_manifest.json")
    if output.exists() or manifest.exists():
        parser.error("Refusing to overwrite an existing result or manifest")
    if git(["diff", "--quiet"], repo, check=False).returncode or git(
        ["diff", "--cached", "--quiet"], repo, check=False
    ).returncode:
        parser.error("Tracked worktree/index changes exist; commit scoped code before publishing")
    branch = git(["branch", "--show-current"], repo).stdout.strip()
    if not branch:
        parser.error("Detached HEAD: refusing to publish")

    command = [
        sys.executable, str(repo / "scripts/audit_grounding_plan_data.py"),
        "--val-csv", str(inputs["val_csv"]),
        "--gsub-val-json", str(inputs["gsub_val_json"]),
        "--test-csv", str(inputs["test_csv"]),
        "--gsub-test-json", str(inputs["gsub_test_json"]),
        "--video-map", str(inputs["video_map"]),
        "--frame2time-val-json", str(inputs["frame2time_val_json"]),
        "--frame2time-test-json", str(inputs["frame2time_test_json"]),
        "--exclude-samples", *[str(path) for path in exclusions],
        "--output", str(output),
    ]
    subprocess.run(command, cwd=repo, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({
        "script": "run_grounding_plan_audit_and_publish.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git(["rev-parse", "HEAD"], repo).stdout.strip(),
        "input_sha256": {key: sha256(path) for key, path in inputs.items()},
        "exclusion_manifest_count": len(exclusions),
        "audit_output_sha256": sha256(output),
        "privacy_check": result["privacy"],
    }, indent=2) + "\n", encoding="utf-8")

    relative = [str(path.relative_to(repo)) for path in (output, manifest)]
    subprocess.run(["git", "add", "-f", *relative], cwd=repo, check=True)
    subprocess.run([
        "git", "commit", "--only", "-m", "Publish temporal grounding data audit", *relative
    ], cwd=repo, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=repo, check=True)
    print(f"Published {relative[0]} and {relative[1]}")


if __name__ == "__main__":
    main()
