from __future__ import annotations

import unittest

from freeze_minerva_metadata_anchor_cohort_v2 import select_metadata_candidates


def _fixture(video_id: str, frame_number: int = 303):
    row = {
        "video_id": video_id,
        "key": f"{video_id}-question",
        "referenced_object_masks": ["cup-42"],
        "question_type": ["Action Recognition"],
        "reasoning": "1. cup 42 is moved at 00:10.",
    }
    assoc = {video_id: {"42": {
        "name": "cup",
        "tracks": [{"time_segment": [10.0, 11.0], "masks": ["mask-a"]}],
    }}}
    masks = {video_id: {"mask-a": {"frame_number": frame_number}}}
    return row, assoc, masks


def _select(fixtures, *, excluded=None, rows=1):
    rows_in, assoc, masks = [], {}, {}
    for row, row_assoc, row_masks in fixtures:
        rows_in.append(row)
        assoc.update(row_assoc)
        masks.update(row_masks)
    return select_metadata_candidates(rows_in, assoc, masks,
        requested_rows=rows, excluded_video_ids=excluded or set())


class MetadataAnchorSelectorTests(unittest.TestCase):
  def test_safe_interior_proxy_frame_is_eligible_and_deterministic(self):
    fixtures = [_fixture("P01-a"), _fixture("P02-b")]
    first = _select(fixtures, rows=2)
    second = _select(fixtures, rows=2)
    self.assertEqual(first["selection"], second["selection"])
    self.assertEqual(first["candidate_counts_before_selection"][
        "proxy_interval_anchor_rows"], 2)
    self.assertEqual(first["candidate_counts_before_selection"][
        "eligible_distinct_videos_before_exposure_exclusion"], 2)
    self.assertEqual(len({item["video_id"] for item in first["selection"]}), 2)
    self.assertTrue(all(item["proxy_anchor"]["frame_number"] == 303
                        for item in first["selection"]))


  def test_frame_outside_strict_interior_or_track_is_ineligible(self):
    for frame_number in (300, 301, 330, 360):
      with self.subTest(frame_number=frame_number):
        with self.assertRaisesRegex(ValueError, "eligible_distinct_video_count"):
          _select([_fixture("P01-a", frame_number)], rows=1)


  def test_exposure_excluded_after_proxy_count_with_no_replacement(self):
    fixtures = [_fixture("P01-a"), _fixture("P02-b")]
    result = _select(fixtures, excluded={"P01-a"}, rows=1)
    counts = result["candidate_counts_before_selection"]
    self.assertEqual(counts["eligible_rows_before_exposure_exclusion"], 2)
    self.assertEqual(counts["exposed_rows_excluded"], 1)
    self.assertEqual(counts["eligible_distinct_videos_before_exposure_exclusion"], 2)
    self.assertEqual(result["selection"][0]["video_id"], "P02-b")


  def test_duplicate_rows_still_select_one_per_video(self):
    fixtures = [_fixture("P01-a"), _fixture("P01-a", 306), _fixture("P02-b")]
    result = _select(fixtures, rows=2)
    self.assertEqual(result["candidate_counts_before_selection"][
        "eligible_rows_after_exposure_exclusion"], 3)
    self.assertEqual(result["eligible_distinct_videos"], 2)
    self.assertEqual(len({item["video_id"] for item in result["selection"]}), 2)


  def test_distinct_participants_are_selected_before_gpu_outcomes(self):
    fixtures = [_fixture("P01-a"), _fixture("P01-b"),
                _fixture("P02-c"), _fixture("P03-d")]
    result = _select(fixtures, rows=3)
    participants = {item["video_id"].split("-", 1)[0]
                    for item in result["selection"]}
    self.assertEqual(participants, {"P01", "P02", "P03"})
    self.assertEqual(result["candidate_counts_before_selection"][
        "eligible_distinct_participants_after_exposure_exclusion"], 3)


  def test_malformed_reference_and_question_types_are_skipped(self):
    good = _fixture("P01-a")
    bad = {**good[0], "video_id": "P03-c", "referenced_object_masks": [None]}
    with self.assertRaisesRegex(ValueError, "eligible_distinct_video_count"):
        select_metadata_candidates([bad], good[1], good[2], requested_rows=1)


if __name__ == "__main__":
    unittest.main()
