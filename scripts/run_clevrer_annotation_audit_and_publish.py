#!/usr/bin/env python3
"""Audit official CLEVRER-Humans annotations and publish only the safe summary.

The source annotations stay outside the repository. Only the metadata-only audit
JSON is staged, committed, and pushed, so the result can be inspected from GitHub.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_all_layer_and_publish import git_identity, run


def safe_results_path(repo: Path, value: str) -> Path:
    path = (repo / value).resolve()
    results = (repo / "results").resolve()
    if path.parent != results or path.suffix.lower() != ".json":
        raise ValueError("--output must be a .json filename directly inside results/")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    return path


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()


def tracked_status_lines(status: str) -> list[str]:
    return [line for line in status.splitlines() if line and not line.startswith("?? ")]


def synchronize_clean_branch(repo: Path, branch: str) -> None:
    run(["git", "fetch", "origin", branch], repo)
    remote_ref = f"origin/{branch}"
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", remote_ref, "HEAD"], cwd=repo, check=False
    ).returncode == 0:
        return
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", remote_ref], cwd=repo, check=False
    ).returncode == 0:
        run(["git", "merge", "--ff-only", remote_ref], repo)
        return
    raise ValueError(f"Local branch and {remote_ref} diverged; refusing to run/publish")


def require_no_tracked_changes(repo: Path) -> None:
    status = git_output(repo, "status", "--porcelain")
    tracked_changes = tracked_status_lines(status)
    if tracked_changes:
        raise ValueError(
            "Tracked/index changes exist; preserve and commit them separately before publishing: "
            + "; ".join(tracked_changes)
        )


def push_with_safe_rebase(repo: Path, branch: str, max_attempts: int = 3) -> None:
    """Retry a rejected push after safe sync, ignoring unrelated untracked user data."""
    for attempt in range(1, max_attempts + 1):
        push = subprocess.run(["git", "push", "origin", branch], cwd=repo, check=False)
        if push.returncode == 0:
            return
        print(f"Push attempt {attempt} failed; checking remote advancement.", flush=True)
        run(["git", "fetch", "origin", branch], repo)
        status = git_output(repo, "status", "--porcelain")
        if tracked_status_lines(status):
            raise SystemExit(
                "Push rejected and tracked changes appeared; preserving local commit and stopping."
            )

        remote_ref = f"origin/{branch}"
        remote_has_head = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", remote_ref], cwd=repo, check=False
        ).returncode == 0
        if remote_has_head:
            run(["git", "merge", "--ff-only", remote_ref], repo)
            return
        remote_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", remote_ref, "HEAD"], cwd=repo, check=False
        ).returncode == 0
        if not remote_is_ancestor:
            rebase = subprocess.run(["git", "rebase", remote_ref], cwd=repo, check=False)
            if rebase.returncode != 0:
                raise SystemExit(
                    "Safe rebase conflicted; local commit is preserved. Resolve and push without rerunning audit."
                )
    raise SystemExit("Push failed after safe synchronization attempts; local commit is preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        default="results/clevrer_humans_annotation_audit_v7_20261008.json",
        help="unique metadata-only report path directly inside results/",
    )
    parser.add_argument(
        "--trust-official-pickle",
        action="store_true",
        help="explicitly allow loading the pickle files from the official Dryad package",
    )
    parser.add_argument("--commit-message", default="Publish CLEVRER-Humans annotation audit")
    parser.add_argument("--no-push", action="store_true", help="commit locally without pushing")
    args = parser.parse_args()
    if not args.trust_official_pickle:
        parser.error("confirm the official Dryad source with --trust-official-pickle")

    repo = Path(__file__).resolve().parents[1]
    annotation_dir = args.annotation_dir.expanduser().resolve()
    try:
        annotation_dir.relative_to(repo)
    except ValueError:
        pass
    else:
        parser.error("keep source annotations outside the Git repository")
    if not annotation_dir.is_dir():
        parser.error(f"annotation directory not found: {annotation_dir}")

    try:
        output = safe_results_path(repo, args.output)
        require_no_tracked_changes(repo)
        branch = git_output(repo, "branch", "--show-current")
        if not branch:
            raise ValueError("Detached HEAD; refusing to publish")
        if "origin" not in git_output(repo, "remote").splitlines():
            raise ValueError("Git remote 'origin' is not configured")
        git_identity(repo)
        synchronize_clean_branch(repo, branch)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))

    audit_script = repo / "scripts/audit_clevrer_humans_annotations.py"
    run(
        [
            sys.executable,
            str(audit_script),
            "--annotation-dir",
            str(annotation_dir),
            "--output",
            str(output),
            "--trust-official-pickle",
        ],
        repo,
    )

    relative_output = str(output.relative_to(repo))
    run(["git", "add", "-f", "--", relative_output], repo)
    staged = git_output(repo, "diff", "--cached", "--name-only").splitlines()
    if staged != [relative_output]:
        raise SystemExit(f"Refusing to commit unexpected staged paths: {staged}")
    check = subprocess.run(
        ["git", "diff", "--cached", "--check"], cwd=repo, text=True, capture_output=True
    )
    if check.returncode:
        raise SystemExit("Staged audit report failed git diff --check:\n" + check.stdout + check.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run(["git", "commit", "--only", "-m", f"{args.commit_message} ({stamp})", relative_output], repo)
    if not args.no_push:
        push_with_safe_rebase(repo, branch)
    print(f"published={relative_output}", flush=True)
    print(f"branch={branch} commit={git_output(repo, 'rev-parse', '--short', 'HEAD')}", flush=True)


if __name__ == "__main__":
    main()
