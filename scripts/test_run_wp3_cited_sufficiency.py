"""CPU checks for WP3 helpers (no GPU, no model)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load():
    import importlib.util
    src = (Path(__file__).resolve().parent / "run_wp3_cited_sufficiency.py").read_text()
    ns = {}
    start = src.index("def complement_keys")
    end = src.index("def self_margin")
    exec(src[start:end], ns)
    return ns


def test_complement_keys_is_exact_complement():
    ns = _load()
    groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert ns["complement_keys"](groups, (1, 3)) == (0, 1, 4, 5)
    assert ns["complement_keys"](groups, ()) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert ns["complement_keys"](groups, (0, 1, 2, 3)) == ()


def test_runner_declares_keep_bias_and_arms():
    src = (Path(__file__).resolve().parent / "run_wp3_cited_sufficiency.py").read_text()
    assert "BIAS = -math.log(256.0)" in src
    for arm in ("keep_cited", "keep_sham", "keep_annotation"):
        assert arm in src
    assert "wp3_cited_sufficiency" in src
