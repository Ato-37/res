import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qta_evidence_edge_knockout as ev

GROUPS = tuple(tuple(range(2 + 3 * g, 5 + 3 * g)) for g in range(4))
SEQ = 19


class EvidenceBlockTests(unittest.TestCase):
    def test_v2t_blocks_text_rows_from_selected_groups_only(self):
        blocks = ev.evidence_blocks("video_to_text", video_groups=GROUPS, seq_len=SEQ, key_groups=(1, 3))
        self.assertEqual(blocks, [((14, 15, 16, 17), (5, 6, 7, 11, 12, 13))])

    def test_v2l_blocks_last_row(self):
        blocks = ev.evidence_blocks("video_to_last", video_groups=GROUPS, seq_len=SEQ, key_groups=[0])
        self.assertEqual(blocks, [((18,), (2, 3, 4))])

    def test_equal_group_count_gives_equal_cell_budget(self):
        a = ev.evidence_blocks("video_to_text", video_groups=GROUPS, seq_len=SEQ, key_groups=(0, 1))
        b = ev.evidence_blocks("video_to_text", video_groups=GROUPS, seq_len=SEQ, key_groups=(2, 3))
        self.assertEqual(ev.block_cell_count(a), ev.block_cell_count(b))

    def test_all_video_control_covers_every_key(self):
        blocks = ev.all_video_blocks("video_to_text", video_groups=GROUPS, seq_len=SEQ)
        self.assertEqual(blocks[0][1], tuple(range(2, 14)))

    def test_invalid_inputs(self):
        for kwargs in (dict(edge="cross_frame", key_groups=(0,)), dict(edge="video_to_text", key_groups=()),
                       dict(edge="video_to_text", key_groups=(9,))):
            with self.assertRaises(ValueError):
                ev.evidence_blocks(kwargs["edge"], video_groups=GROUPS, seq_len=SEQ, key_groups=kwargs["key_groups"])


if __name__ == "__main__":
    unittest.main()
