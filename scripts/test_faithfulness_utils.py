"""CPU-only tests for head selection and matched-control construction."""

import math
import unittest

from faithfulness_utils import (
    build_matched_control_groups,
    format_choice_prompt,
    gradient_activation_scores,
    mean_t_interval,
    select_prediction_and_foil,
    select_top_heads,
    signflip_p_against,
)


class FaithfulnessUtilsTests(unittest.TestCase):
    def test_top_k_is_deterministic_and_validated(self):
        rows = [
            {'layer': 1, 'head': 2, 'score': 0.4},
            {'layer': 0, 'head': 1, 'score': 0.4},
            {'layer': 0, 'head': 0, 'score': 0.9},
        ]
        self.assertEqual([(x['layer'], x['head']) for x in select_top_heads(rows, 2)],
                         [(0, 0), (0, 1)])
        with self.assertRaises(ValueError):
            select_top_heads(rows + [rows[0]], 2)
        with self.assertRaises(ValueError):
            select_top_heads([{'layer': 0, 'head': 0, 'score': math.nan}], 1)

    def test_choice_margin_freezes_prediction_but_updates_best_foil(self):
        prompt = format_choice_prompt('What happens?', ['a', 'b', 'c', 'd', 'e'])
        self.assertIn('A. a', prompt)
        self.assertTrue(prompt.endswith('Answer:'))
        prediction, foil, margin = select_prediction_and_foil([0.1, 0.4, 0.2, 0.0, -0.1])
        self.assertEqual((prediction, foil), (1, 2))
        self.assertAlmostEqual(margin, 0.2)
        # Keep option B as the selected answer, but recompute its strongest foil.
        prediction, foil, margin = select_prediction_and_foil([0.2, 0.3, 0.29, 0.0, -0.1], 1)
        self.assertEqual((prediction, foil), (1, 2))
        self.assertAlmostEqual(margin, 0.01)
        with self.assertRaises(ValueError):
            format_choice_prompt('Q', ['only one'])

    def test_gradient_times_activation_is_absolute_dot_product(self):
        scores, norms = gradient_activation_scores(
            [[2.0, -3.0], [1.0, 1.0]],
            [[-1.0, 2.0], [0.5, -0.5]],
        )
        self.assertEqual(scores, [8.0, 0.0])
        self.assertAlmostEqual(norms[0], math.sqrt(13))
        with self.assertRaises(ValueError):
            gradient_activation_scores([[1.0]], [[float('inf')]])

    def test_matched_controls_keep_layer_histogram_and_are_disjoint(self):
        selected = [
            {'layer': 0, 'head': 2, 'score': 1.0},
            {'layer': 0, 'head': 5, 'score': 0.8},
            {'layer': 2, 'head': 1, 'score': 0.6},
            {'layer': 4, 'head': 9, 'score': 0.4},
        ]
        norms = {(layer, head): float(1 + layer * 0.1 + head * 0.01)
                 for layer in range(5) for head in range(28)}
        groups, distances = build_matched_control_groups(
            selected, norms, num_heads=28, repeats=5, seed=17)
        expected = {layer: sum(item['layer'] == layer for item in selected)
                    for layer in {item['layer'] for item in selected}}
        self.assertEqual(len(groups), 5)
        self.assertEqual(len(distances), 20)
        all_controls = set()
        for group in groups:
            self.assertEqual(len(group), len(selected))
            histogram = {layer: sum(item['layer'] == layer for item in group)
                         for layer in expected}
            self.assertEqual(histogram, expected)
            keys = {(item['layer'], item['head']) for item in group}
            self.assertFalse(keys.intersection((item['layer'], item['head']) for item in selected))
            self.assertEqual(len(keys), len(group))
            self.assertFalse(keys.intersection(all_controls))
            all_controls.update(keys)

    def test_matched_controls_fail_if_too_many_disjoint_controls_requested(self):
        selected = [{'layer': 0, 'head': head, 'score': 1.0} for head in range(4)]
        norms = {(0, head): 1.0 for head in range(5)}
        with self.assertRaises(ValueError):
            build_matched_control_groups(selected, norms, num_heads=5, repeats=2)

    def test_t_interval_and_signflip_are_finite(self):
        result = mean_t_interval([0.3, 0.5, 0.4])
        self.assertAlmostEqual(result['mean'], 0.4)
        self.assertEqual(len(result['ci']), 2)
        self.assertLess(result['ci'][0], result['ci'][1])
        p = signflip_p_against([0.5, 0.6, 0.7], null=0.25, draws=1000, seed=4)
        self.assertGreaterEqual(p, 0)
        self.assertLessEqual(p, 1)


if __name__ == '__main__':
    unittest.main()
