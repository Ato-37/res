from __future__ import annotations
import unittest
from fractions import Fraction
from pathlib import Path
import run_claim_qa_interval_likelihood_interaction as module
from presentation_time_utils import strict_evidence_indices_from_times
from necessity_choice_margin_utils import nearest_control_indices
from run_saved_claim_video_content_dependence import resolve_claim_prompt_suffix
from generate_nextgqa_claim_review_pilot import PROMPT_SUFFIX_BY_VERSION


class ClaimQAIntervalInteractionTests(unittest.TestCase):
    def test_fixed_rows_map_without_replacement(self):
        self.assertEqual(module.index_map(module.PACKET_REL,0),0)
        self.assertEqual(module.index_map(module.REMAINING_PACKET_REL,0),1)
        self.assertEqual(module.index_map(module.REMAINING_PACKET_REL,1),2)
        with self.assertRaises(ValueError): module.index_map(module.PACKET_REL,1)

    def test_difference_in_differences(self):
        value=module.difference_in_differences(-1.0,-1.5,-.8,-1.6)
        self.assertAlmostEqual(value["baseline_control_minus_qa_interval_nats_per_token"],.5)
        self.assertAlmostEqual(value["biased_control_minus_qa_interval_nats_per_token"],.8)
        self.assertAlmostEqual(value["bias_by_qa_interval_interaction_nats_per_token"],.3)

    def test_qa_spans_yield_equal_sized_disjoint_controls(self):
        pts=[Fraction(i,15) for i in range(16)]
        evidence=strict_evidence_indices_from_times(list(range(16)),pts,[[.2,.4]])
        control=nearest_control_indices(list(range(16)),evidence)
        self.assertEqual(len(control),len(evidence));self.assertFalse(set(control)&set(evidence))

    def test_nonfinite_scores_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            module.difference_in_differences(0,float("nan"),0,0)

    def test_prompt_suffix_is_bound_to_recorded_generation_version(self):
        version, suffix = resolve_claim_prompt_suffix(
            {"generation_prompt_version": "v2_short"}, PROMPT_SUFFIX_BY_VERSION)
        self.assertEqual(version, "v2_short")
        self.assertIn("25", suffix)
        with self.assertRaisesRegex(ValueError, "prompt_version_invalid"):
            resolve_claim_prompt_suffix({"generation_prompt_version": "unknown"}, PROMPT_SUFFIX_BY_VERSION)


if __name__=="__main__":unittest.main()
