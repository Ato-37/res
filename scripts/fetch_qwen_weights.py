"""Download a pinned (or explicitly selected) Qwen checkpoint into the HF cache."""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--min-free-gib", type=float, default=24.0)
    parser.add_argument("--cache-dir", type=Path, default=Path(HF_HUB_CACHE))
    args = parser.parse_args()

    cache_root = args.cache_dir.expanduser()
    disk_root = cache_root if cache_root.exists() else cache_root.parent
    free_gib = shutil.disk_usage(disk_root).free / (1024 ** 3)
    if free_gib < args.min_free_gib:
        parser.error(
            f"Need {args.min_free_gib:.1f} GiB free for a 16.6 GB checkpoint; "
            f"only {free_gib:.1f} GiB is available"
        )

    print(
        f"Downloading {args.repo_id}@{args.revision}; upstream repository size "
        f"approximately 16.6 GB; destination is the Hugging Face cache."
    )
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=str(cache_root),
    )
    print(f"Download complete: resolved_snapshot={Path(path).name}")


if __name__ == "__main__":
    main()
