from fractions import Fraction
import hashlib
import unittest

from PIL import Image

from nextgqa_claim_mask_conditions import build_claim_mask_conditions
from presentation_time_utils import assign_evidence_control_frames_from_times


def frames32():
    return [Image.new("RGB", (3, 2), (i + 1, 80 + i, 150 + i)) for i in range(32)]


def sampled_map(total=63):
    # Strictly increasing, variable-duration exact PTS values.
    times = [Fraction(0)]
    for index in range(1, total):
        times.append(times[-1] + Fraction(1 if index % 3 else 2, 10))
    sampled = [round(i * (total - 1) / 31) for i in range(32)]
    return times, sampled


def pixel_sha(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


class ClaimMaskConditionsTests(unittest.TestCase):
    def test_endpoint_inclusive_evidence_assigns_equal_count_disjoint_controls(self):
        times, sampled = sampled_map()
        source = frames32()
        before = [pixel_sha(frame) for frame in source]
        # Select sampled ranks 4, 5, 6 exactly by their boundary PTS values.
        intervals = [[times[sampled[4]], times[sampled[6]]]]
        result = build_claim_mask_conditions(
            frames=source, sampled_indices=sampled, presentation_times=times,
            intervals=intervals, include_metadata=True)
        metadata = result["metadata"]
        self.assertEqual(metadata["status"], "assignment_constructed_unvalidated")
        self.assertEqual(metadata["evidence_ranks"], [4, 5, 6])
        self.assertEqual(len(metadata["control_ranks"]), 3)
        self.assertFalse(set(metadata["evidence_ranks"]) & set(metadata["control_ranks"]))
        self.assertEqual(before, [pixel_sha(frame) for frame in source])
        self.assertEqual(metadata["input_frame_sha256"], before)
        self.assertEqual(result["conditions"]["full"][0].getpixel((0, 0)), source[0].getpixel((0, 0)))
        result["conditions"]["full"][0].putpixel((0, 0), (0, 0, 0))
        self.assertEqual(pixel_sha(source[0]), before[0])
        self.assertEqual(metadata["clock_calibration_authenticated"], False)
        self.assertEqual(metadata["review_provenance_authenticated"], False)
        self.assertEqual(metadata["execution_authorization"], "not_granted")

    def test_single_evidence_tie_uses_lexicographically_smaller_control_rank(self):
        total = 63
        times = [Fraction(i, 2) for i in range(total)]
        sampled = [round(i * (total - 1) / 31) for i in range(32)]
        evidence_rank = 10
        evidence_time = times[sampled[evidence_rank]]
        assignment = assign_evidence_control_frames_from_times(
            total, 32, times, [[evidence_time, evidence_time]])
        self.assertEqual(assignment["status"], "ok")
        self.assertEqual(assignment["control_indices"], [sampled[9]])
        self.assertEqual(assignment["evidence_indices"], [sampled[evidence_rank]])

        result = build_claim_mask_conditions(
            frames=frames32(), sampled_indices=sampled,
            presentation_times=times, intervals=[[evidence_time, evidence_time]],
            include_metadata=True)
        self.assertEqual(result["metadata"]["control_ranks"], [9])

    def test_only_assigned_whole_frames_are_black_and_all_other_pixels_match(self):
        times, sampled = sampled_map()
        source = frames32()
        result = build_claim_mask_conditions(
            frames=source, sampled_indices=sampled, presentation_times=times,
            intervals=[[times[sampled[12]], times[sampled[13]]]],
            include_metadata=True)
        conditions = result["conditions"]
        metadata = result["metadata"]
        for condition, selected_ranks in (
                ("claim_evidence_black", set(metadata["evidence_ranks"])),
                ("temporal_control_black", set(metadata["control_ranks"]))):
            for rank, (original, transformed) in enumerate(zip(
                    source, conditions[condition], strict=True)):
                if rank in selected_ranks:
                    self.assertEqual(transformed.tobytes(), bytes(transformed.width * transformed.height * 3))
                else:
                    self.assertEqual(pixel_sha(transformed), pixel_sha(original))

    def test_no_evidence_and_insufficient_controls_fail_closed(self):
        total = 63
        times = [Fraction(i, 2) for i in range(total)]
        sampled = [round(i * (total - 1) / 31) for i in range(32)]
        with self.assertRaisesRegex(ValueError, "frame_assignment_no_evidence"):
            build_claim_mask_conditions(
                frames=frames32(), sampled_indices=sampled,
                presentation_times=times, intervals=[[Fraction(1, 100), Fraction(1, 100)]])
        with self.assertRaisesRegex(ValueError, "frame_assignment_insufficient_control"):
            build_claim_mask_conditions(
                frames=frames32(), sampled_indices=sampled,
                presentation_times=times,
                intervals=[[times[sampled[15]], times[sampled[31]]]])

    def test_interval_may_extend_past_final_pts_without_duration_input(self):
        times, sampled = sampled_map()
        end = times[-1]
        # Final PTS is not necessarily the media duration, so this helper cannot
        # reject trailing display-time intervals without a separately bound duration.
        result = build_claim_mask_conditions(
            frames=frames32(), sampled_indices=sampled,
            presentation_times=times,
            intervals=[[end, end + Fraction(1, 10)]], include_metadata=True)
        self.assertEqual(result["metadata"]["evidence_ranks"], [31])

    def test_wrong_sampler_input_or_invalid_rgb_frame_fails(self):
        times, sampled = sampled_map()
        wrong_indices = list(sampled)
        wrong_indices[0] = 1
        with self.assertRaisesRegex(ValueError, "sampled_indices_do_not_match_frozen_sampler"):
            build_claim_mask_conditions(
                frames=frames32(), sampled_indices=wrong_indices,
                presentation_times=times,
                intervals=[[times[sampled[2]], times[sampled[3]]]])
        invalid = frames32()
        invalid[0] = Image.new("L", (3, 2), 0)
        with self.assertRaisesRegex(ValueError, "sampled_frames_must_be_rgb_images"):
            build_claim_mask_conditions(
                frames=invalid, sampled_indices=sampled,
                presentation_times=times,
                intervals=[[times[sampled[2]], times[sampled[3]]]])


if __name__ == "__main__":
    unittest.main()
