import unittest

from freeze_minerva_generated_grounding_cohort import select_cohort


def _row(video_id, suffix, tags=None, name="ladle", question_id=None):
    oid = "abcdef0123456789"
    return {"video_id": video_id, "key": question_id or f"{video_id}:{suffix}",
        "question_type": tags or ["Action Recognition"],
        "referenced_object_masks": [f"{name}-{oid}"],
        "reasoning": f"At 00:08 the {name} moves ({oid})."}


def _metadata(videos):
    assoc = {video: {"abcdef0123456789": {"name": "ladle",
        "tracks": [{"time_segment": [7.5, 9.0], "masks": ["mask1"]}]}}
        for video in videos}
    masks = {video: {"mask1": {"frame_number": 10}} for video in videos}
    return assoc, masks


class CohortSelectionTests(unittest.TestCase):
    def test_selects_one_deterministic_row_per_distinct_video(self):
        rows = [_row("v1", "b"), _row("v1", "a"), _row("v2", "c")]
        assoc, masks = _metadata(["v1", "v2"])
        first = select_cohort(rows, assoc, masks, 2)
        second = select_cohort(list(reversed(rows)), assoc, masks, 2)
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(len({x["video_id"] for x in first["selection"]}), 2)
        self.assertEqual(first["eligible_rows"], 3)

    def test_excludes_counterfactual_and_name_mismatch(self):
        rows = [_row("v1", "a", ["Action Recognition", "Counterfactual"]),
                _row("v2", "b", name="spoon")]
        assoc, masks = _metadata(["v1", "v2"])
        with self.assertRaisesRegex(ValueError, "eligible_video_count_below_requested_rows"):
            select_cohort(rows, assoc, masks, 1)

    def test_fails_closed_when_requested_video_count_unavailable(self):
        assoc, masks = _metadata(["v1"])
        with self.assertRaisesRegex(ValueError, "eligible_video_count_below_requested_rows"):
            select_cohort([_row("v1", "a")], assoc, masks, 2)

    def test_excludes_previously_exposed_videos_before_stable_selection(self):
        rows = [_row(f"v{i}", "a") for i in range(4)]
        assoc, masks = _metadata([f"v{i}" for i in range(4)])
        first = select_cohort(rows, assoc, masks, 2,
                              excluded_video_ids={"v0", "v1"})
        second = select_cohort(rows, assoc, masks, 2,
                               excluded_video_ids={"v0", "v1"})
        selected = {item["video_id"] for item in first["selection"]}
        self.assertEqual(first, second)
        self.assertEqual(selected, {"v2", "v3"})
        self.assertEqual(first["eligible_rows"], 4)
        self.assertEqual(first["eligible_videos"], 2)


if __name__ == "__main__":
    unittest.main()
