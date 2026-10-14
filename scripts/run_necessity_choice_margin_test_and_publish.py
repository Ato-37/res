"""Run locally; publish only a privacy-filtered aggregate necessity summary."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

from derive_necessity_choice_margin_summary import derive, publish_manifest
from run_task_validity_audit_and_publish import push_with_rebase


def run(command, repo):
    subprocess.run(command, cwd=repo, check=True)


def run_analysis(command, repo, private_log):
    with private_log.open("w") as log:
        os.chmod(private_log, 0o600)
        try:
            subprocess.run(command, cwd=repo, check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"necessity analysis failed with exit status {exc.returncode}; "
                "private diagnostics were retained locally"
            ) from None


def analysis_command(repo, samples, output, args):
    command = [sys.executable, str(repo / "scripts/necessity_choice_margin_test.py"),
               "--samples-file", str(samples), "--output", str(output),
               "--quant", args.quant, "--seed", str(args.seed),
               "--bootstrap-draws", str(args.bootstrap_draws),
               "--num-frames", str(args.num_frames),
               "--mask-fill", args.mask_fill, "--blur-radius", str(args.blur_radius)]
    if args.n_samples is not None:
        command.extend(["--n-samples", str(args.n_samples)])
    return command


def validate_sample_units(samples, n_samples=None):
    """Require one independent analysis unit per video before model loading."""
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample manifest must be a nonempty list")
    if n_samples is not None and (type(n_samples) is not int or n_samples < 1
                                  or len(samples) < n_samples):
        raise ValueError("requested sample count is unavailable")
    selected = samples if n_samples is None else samples[:n_samples]
    video_ids = [row.get("video_id") if isinstance(row, dict) else None
                 for row in selected]
    if any(not isinstance(video_id, str) or not video_id for video_id in video_ids):
        raise ValueError("every sample must declare a video analysis unit")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("samples must contain one analysis unit per video")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output", required=True,
                        help="unique derived aggregate path inside results/")
    parser.add_argument("--preflight-report",
                        help="optional local-only preflight file; never published")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--mask-fill", choices=["black", "blur"], default="black")
    parser.add_argument("--blur-radius", type=float, default=12.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    samples = (repo / args.samples_file).resolve()
    output = (repo / args.output).resolve()
    preflight = (repo / args.preflight_report).resolve() if args.preflight_report else None
    if repo not in samples.parents or repo not in output.parents:
        parser.error("sample/output paths must be inside the repository")
    if preflight is not None and (repo not in preflight.parents or not preflight.is_file()):
        parser.error("preflight report must exist inside the repository")
    if not output.is_relative_to(repo / "results"):
        parser.error("derived output must be inside results/")
    manifest = output.with_name(output.stem + "_run_manifest.json")
    if output.exists() or manifest.exists():
        parser.error("refusing to overwrite an existing derived artifact or manifest")
    if not samples.is_file():
        parser.error("missing sample manifest")
    try:
        sample_rows = json.loads(samples.read_text())
        requested = validate_sample_units(sample_rows, args.n_samples)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid sample manifest: {exc}")

    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
        text=True, capture_output=True, check=True).stdout.strip()
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repo,
        text=True, capture_output=True, check=True).stdout.strip()
    if tracked or staged:
        parser.error("tracked worktree and index must be clean before inference")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
        capture_output=True, check=True).stdout.strip()
    if not branch:
        parser.error("detached HEAD: refusing to publish")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True).stdout.strip()

    # Raw per-item records are retained in a private temporary directory and
    # never staged. The sample manifest and optional preflight report also stay local.
    private_dir = Path(tempfile.mkdtemp(prefix="res_necessity_private_"))
    os.chmod(private_dir, 0o700)
    raw_output = private_dir / "raw_result.json"
    private_log = private_dir / "runner.log"
    command = analysis_command(repo, samples, raw_output, args)
    print("Running bounded necessity analysis; item-level console output is kept local.",
          flush=True)
    run_analysis(command, str(repo), private_log)
    os.chmod(raw_output, 0o600)
    derived = derive(raw_output)
    if derived["source_commit"] != source_commit:
        parser.error("runner source commit changed during execution")
    if derived["n_requested"] != requested:
        parser.error("runner requested count does not match the frozen selection")

    output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = (json.dumps(derived, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.write_bytes(output_bytes)
    output.chmod(0o644)
    manifest.write_text(json.dumps(publish_manifest(derived, output_bytes),
                                   indent=2, sort_keys=True) + "\n")
    manifest.chmod(0o644)

    files = [str(path.relative_to(repo)) for path in (output, manifest)]
    run(["git", "add", "-f", *files], str(repo))
    run(["git", "diff", "--cached", "--check", "--", *files], str(repo))
    run(["git", "commit", "--only", "-m",
         "Publish choice-margin necessity aggregate", *files], str(repo))
    push_with_rebase(str(repo), branch)
    run(["git", "fetch", "origin", branch], str(repo))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                          capture_output=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "rev-parse", f"origin/{branch}"], cwd=repo,
                            text=True, capture_output=True, check=True).stdout.strip()
    if head != remote:
        raise SystemExit("published summary is not verified on origin; local commit preserved")
    print(f"Published derived aggregate {output.relative_to(repo)} and {manifest.relative_to(repo)}")
    print("Private raw per-item output and runner log remain in a restricted temporary directory.")


if __name__ == "__main__":
    main()
