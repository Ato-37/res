"""Safely allow private review artifacts in one ignored repository directory."""
from __future__ import annotations

import os
from pathlib import Path
import stat


def _lstat(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_private_output(path: str | Path, repo_root: str | Path) -> Path:
    """Return an absolute output path, rejecting unsafe in-repository targets.

    External destinations preserve the existing CLI behavior. Within the
    repository, only direct children of ``data/private_nextgqa`` are allowed.
    That directory must be a current-user-owned, non-symlink directory with
    exact 0700 permissions. It is created with 0700 when absent.
    """
    repo = Path(repo_root).resolve(strict=True)
    raw = Path(path)
    lexical = Path(os.path.abspath(raw if raw.is_absolute() else Path.cwd() / raw))
    resolved = lexical.resolve(strict=False)
    private = repo / "data" / "private_nextgqa"

    # Reject external aliases that resolve back into the repository.
    if not _is_within(lexical, repo):
        if _is_within(resolved, repo):
            raise ValueError("private_output_symlink_into_repository")
        return lexical

    if lexical.parent != private or lexical.name in {"", ".", ".."}:
        raise ValueError("repository_output_not_private_allowlisted_path")

    data_info = _lstat(repo / "data")
    if data_info is None or not stat.S_ISDIR(data_info.st_mode) or stat.S_ISLNK(data_info.st_mode):
        raise ValueError("private_output_data_parent_invalid")

    private_info = _lstat(private)
    if private_info is None:
        try:
            os.mkdir(private, 0o700)
        except FileExistsError:
            pass
        private_info = _lstat(private)
    if (private_info is None or not stat.S_ISDIR(private_info.st_mode)
            or stat.S_ISLNK(private_info.st_mode)):
        raise ValueError("private_output_directory_invalid")
    if private_info.st_uid != os.geteuid():
        raise ValueError("private_output_directory_owner_invalid")
    if stat.S_IMODE(private_info.st_mode) != 0o700:
        raise ValueError("private_output_directory_mode_invalid")

    if os.path.lexists(lexical):
        raise FileExistsError("private_output_already_exists")
    # Resolve only after lstat checks: this catches redirection through either
    # the target directory or an unexpected path component.
    if lexical.resolve(strict=False) != lexical:
        raise ValueError("private_output_symlink_component")
    return lexical
