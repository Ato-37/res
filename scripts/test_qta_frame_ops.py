"""CPU checks for the WP5 frame operations (frames are tagged integers wrapped in a copy-able stub)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qta_frame_ops as fo


class F:
    def __init__(self, tag):
        self.tag = tag

    def copy(self):
        return F(self.tag)


def tags(frames):
    return [f.tag for f in frames]


def test_replace_uses_nearest_outside_frame_earlier_on_ties():
    frames = [F(i) for i in range(8)]
    out = fo.replace_groups(frames, [1], 2)  # frames 2,3
    assert tags(out) == [0, 1, 1, 4, 4, 5, 6, 7]


def test_keep_only_is_complement_and_geometry_preserved():
    frames = [F(i) for i in range(8)]
    out = fo.keep_only_groups(frames, [1], 2)  # keep frames 2,3
    assert len(out) == 8 and tags(out) == [2, 2, 2, 3, 3, 3, 3, 3]


def test_all_static_and_errors():
    frames = [F(i) for i in range(6)]
    assert tags(fo.all_static(frames)) == [3] * 6
    try:
        fo.replace_indices(frames, set(range(6)))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        fo.replace_groups(frames, [9], 2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_replace_uses_nearest_outside_frame_earlier_on_ties(); test_keep_only_is_complement_and_geometry_preserved(); test_all_static_and_errors()
    print("tests ok")
