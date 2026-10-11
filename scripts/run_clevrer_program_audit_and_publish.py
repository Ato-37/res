#!/usr/bin/env python3
"""Run the CPU-only CLEVRER QA schema audit and publish counts only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from audit_clevrer_counterfactual_programs import audit_payload, discover_json_files, safe_report
from run_all_layer_and_publish import git_identity, run
from run_clevrer_annotation_audit_and_publish import (
    git_output,
    push_with_safe_rebase,
    require_no_tracked_changes,
    safe_results_path,
    synchronize_clean_branch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], type=Path,
                        help="official CLEVRER question JSON; repeat for multiple splits")
    parser.add_argument("--root", action="append", default=[], type=Path,
                        help="search root; repeatable")
    parser.add_argument("--output", required=True,
                        help="unique JSON report filename directly inside results/")
    parser.add_argument("--commit-message", default="Publish CLEVRER program schema audit")
    parser.add_argument("--no-push", action="store_true",
                        help="commit locally without pushing")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    candidates = {path.expanduser().resolve() for path in args.input}
    candidates.update(discover_json_files(path.expanduser() for path in args.root))
    paths = sorted(candidates)
    if not paths:
        parser.error("no candidate CLEVRER QA JSON found; no report was created")

    try:
        output = safe_results_path(repo, args.output)
        require_no_tracked_changes(repo)
        branch = git_output(repo, "branch", "--show-current")
        if not branch:
            raise ValueError("detached HEAD; refusing to publish")
        if "origin" not in git_output(repo, "remote").splitlines():
            raise ValueError("Git remote 'origin' is not configured")
        synchronize_clean_branch(repo, branch)
        git_identity(repo)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))

    reports = []
    for path in paths:
        if not path.is_file():
            parser.error(f"input JSON does not exist: {path.name}")
        with path.open(encoding="utf-8") as source:
            reports.append((path, audit_payload(json.load(source))))

    output.write_text(json.dumps(safe_report(reports), indent=2) + "\n",
                      encoding="utf-8")
    relative_output = str(output.relative_to(repo))
    run(["git", "add", "-f", "--", relative_output], repo)
    staged = git_output(repo, "diff", "--cached", "--name-only").splitlines()
    if staged != [relative_output]:
        raise SystemExit(f"Refusing to commit unexpected staged paths: {staged}")
    check = subprocess.run(["git", "diff", "--cached", "--check", "--", relative_output],
                           cwd=repo, text=True, capture_output=True)
    if check.returncode:
        raise SystemExit("Staged summary failed git diff --check; report preserved locally")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run(["git", "commit", "--only", "-m", f"{args.commit_message} ({stamp})",
         relative_output], repo)
    if not args.no_push:
        push_with_safe_rebase(repo, branch)
    print(f"Published counts-only audit: {relative_output}")
    print(f"Commit: {git_output(repo, 'rev-parse', '--short', 'HEAD')}")


if __name__ == "__main__":
    main()
