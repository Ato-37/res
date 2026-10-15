"""Explicit stop for retired/closed real-media inference branches."""


def reject_inference(parser, branch_code: str, reason: str) -> None:
    """Stop a CLI before it reads sample data, media, or initializes a model."""
    parser.error(f"blocked[{branch_code}]: {reason}; no inference was launched")
