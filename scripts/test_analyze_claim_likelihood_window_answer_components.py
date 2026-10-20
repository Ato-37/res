import unittest
from analyze_claim_likelihood_window_answer_components import decompose

class AnswerComponentTests(unittest.TestCase):
    def test_margin_difference_reconstructs_from_gold_and_best_foil(self):
        probs={"full":[-4,-3,-2,-.1,-4],
               "3b_likelihood_selected_window_black":[-3,-2,-1,-1,-4],
               "temporal_control_black":[-4,-3,-2,-.2,-3]}
        result=decompose(3,probs)
        self.assertAlmostEqual(result["control_minus_selected_margin_reconstructed"],
            result["control_minus_selected_gold_component"]-result["control_minus_selected_best_foil_component"])
        self.assertAlmostEqual(result["control_minus_selected_margin_reconstructed"],1.8)

    def test_invalid_gold_or_answer_vector_rejected(self):
        with self.assertRaises(ValueError):decompose(5,{})
        with self.assertRaises(ValueError):decompose(0,{"full":[1]})

if __name__=="__main__":unittest.main()
