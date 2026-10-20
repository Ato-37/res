from __future__ import annotations
import unittest
from claim_likelihood_window_ranking import (
    partition_window_indices, select_max_mean_logprob, selected_interval,
)

class ClaimLikelihoodWindowTests(unittest.TestCase):
    def test_four_equal_disjoint_windows_partition_all_frames(self):
        windows=partition_window_indices(32)
        self.assertEqual(tuple(map(len,windows)),(8,8,8,8))
        self.assertEqual(tuple(i for window in windows for i in window),tuple(range(32)))

    def test_highest_score_selected_and_tie_breaks_earliest(self):
        self.assertEqual(select_max_mean_logprob((-2,-1,-3,-4)),1)
        self.assertEqual(select_max_mean_logprob((-1,-1,-2,-3)),0)

    def test_selected_window_binds_exact_sampled_times(self):
        times=tuple(i/2 for i in range(32))
        indices,interval=selected_interval(times,2)
        self.assertEqual(indices,tuple(range(16,24)))
        self.assertEqual(interval,(8.0,11.5))

    def test_bad_dimensions_scores_and_times_rejected(self):
        for args in ((31,4),(32,1),(0,4)):
            with self.assertRaises(ValueError):partition_window_indices(*args)
        for scores in ((1,2,3), (1,2,float('nan'),4), (1,2,True,4)):
            with self.assertRaises(ValueError):select_max_mean_logprob(scores)
        with self.assertRaises(ValueError):selected_interval([0,1,0],0)

if __name__=="__main__":unittest.main()
