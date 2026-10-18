from __future__ import annotations

import unittest

from audit_qta_v4_mask_specificity_exhaustive import run_audit


class ExhaustivePlaceboSelectionAuditTests(unittest.TestCase):
    def test_complete_synthetic_domain_matches_independent_reference(self):
        self.assertEqual(run_audit(), {
            "count_vectors": 126,
            "masks": 5334,
            "eligible_offsets": 4116,
            "no_candidate_masks": 2812,
        })


if __name__ == "__main__":
    unittest.main()
