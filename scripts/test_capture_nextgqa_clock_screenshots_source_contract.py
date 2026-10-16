"""Source-contract checks for the Windows-only capture row mapping/prompt."""
from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name("capture_nextgqa_clock_screenshots.ps1")
SOURCE = SCRIPT.read_text(encoding="utf-8")
ANCHOR_ORDER = ("fit_begin", "heldout_quarter", "fit_midpoint",
                "heldout_three_quarter", "fit_end")
FIT_IDS = {"fit_begin", "fit_midpoint", "fit_end"}
HELDOUT_IDS = {"heldout_quarter", "heldout_three_quarter"}


def ordered_rows(records):
    """Pure synthetic contract model for the PS plan-to-row mapping."""
    if len(records) != 3:
        raise ValueError("record_count")
    tokens = [row["review_token"] for row in records]
    if len(set(tokens)) != 3:
        raise ValueError("duplicate_token")
    rows = []
    for record in records:
        fit = record["fit_anchors"]
        heldout = record["heldout_anchors"]
        fit_ids = [x["anchor_id"] for x in fit]
        heldout_ids = [x["anchor_id"] for x in heldout]
        if len(fit_ids) != len(set(fit_ids)) or set(fit_ids) != FIT_IDS:
            raise ValueError("fit_ids")
        if len(heldout_ids) != len(set(heldout_ids)) or set(heldout_ids) != HELDOUT_IDS:
            raise ValueError("heldout_ids")
        anchors = {x["anchor_id"]: x["decoder_pts_seconds"] for x in fit + heldout}
        rows.extend((record["review_token"], anchor_id, anchors[anchor_id])
                    for anchor_id in ANCHOR_ORDER)
    return rows


class CapturePromptContractTests(unittest.TestCase):
    def test_explicit_chronological_anchor_order_is_used_per_record(self):
        order = re.search(r"\$anchorOrder\s*=\s*@\((.*?)\)", SOURCE, re.S)
        self.assertIsNotNone(order)
        actual = re.findall(r"'([a-z_]+)'", order.group(1))
        self.assertEqual(actual, ["fit_begin", "heldout_quarter", "fit_midpoint",
                                  "heldout_three_quarter", "fit_end"])
        self.assertIn("foreach ($anchorId in $anchorOrder)", SOURCE)
        self.assertIn("$recordAnchors[$anchorId]", SOURCE)
        self.assertIn("candidate_anchor_ids_invalid", SOURCE)
        self.assertIn("$fitAnchorIds = @('fit_begin', 'fit_midpoint', 'fit_end')", SOURCE)
        self.assertIn("$heldoutAnchorIds = @('heldout_quarter', 'heldout_three_quarter')", SOURCE)
        self.assertIn("candidate_anchor_split_ids_invalid", SOURCE)
        self.assertIn("candidate_token_duplicate", SOURCE)

    def test_interactive_prompt_names_current_token_anchor_and_planned_pts(self):
        prompt = re.search(r'Write-Host \("\[\{0\}/15\].*?\n\s*\(\$index \+ 1\), \$anchor\.review_token, \$anchor\.anchor_id, \$anchor\.decoder_pts_seconds\)', SOURCE, re.S)
        self.assertIsNotNone(prompt)
        line = prompt.group(0)
        self.assertIn("$anchor.review_token", line)
        self.assertIn("$anchor.anchor_id", line)
        self.assertIn("$anchor.decoder_pts_seconds", line)
        self.assertNotIn("Follow the next planned anchor", SOURCE)

    def test_candidate_authorization_remains_ungranted(self):
        self.assertIn("$plan.execution_authorization -ne 'not_granted'", SOURCE)
        self.assertIn("execution_authorization = 'not_granted'", SOURCE)

    def test_synthetic_plan_mapping_and_split_or_token_rejections(self):
        records = []
        for i in range(3):
            records.append({
                "review_token": f"token_{i:020d}",
                "fit_anchors": [{"anchor_id": aid, "decoder_pts_seconds": str(j)}
                                for j, aid in enumerate(("fit_end", "fit_begin", "fit_midpoint"))],
                "heldout_anchors": [{"anchor_id": aid, "decoder_pts_seconds": str(j + 3)}
                                    for j, aid in enumerate(("heldout_three_quarter", "heldout_quarter"))],
            })
        rows = ordered_rows(records)
        self.assertEqual(len(rows), 15)
        self.assertEqual([x[1] for x in rows[:5]], list(ANCHOR_ORDER))
        swapped = [dict(row) for row in records]
        swapped[0]["fit_anchors"] = [
            {"anchor_id": "fit_begin", "decoder_pts_seconds": "0"},
            {"anchor_id": "fit_midpoint", "decoder_pts_seconds": "1"},
            {"anchor_id": "heldout_quarter", "decoder_pts_seconds": "2"},
        ]
        with self.assertRaisesRegex(ValueError, "fit_ids"):
            ordered_rows(swapped)
        duplicated = [dict(row) for row in records]
        duplicated[1]["review_token"] = duplicated[0]["review_token"]
        with self.assertRaisesRegex(ValueError, "duplicate_token"):
            ordered_rows(duplicated)


if __name__ == "__main__":
    unittest.main()
