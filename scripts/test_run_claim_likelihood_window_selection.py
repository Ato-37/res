import unittest
from run_claim_likelihood_window_selection import _selection_status

class SelectionStatusTests(unittest.TestCase):
    def test_partial_smoke_never_claims_cohort_completion(self):
        row={"row_token_sha256":"a","window_mean_claim_logprobs_nats_per_token":[-1]*4}
        self.assertEqual(_selection_status([row],4),"partial_window_likelihood_smoke")
        self.assertEqual(_selection_status([row,row],8),"partial_window_likelihood_smoke")
        self.assertEqual(_selection_status([row,row,{**row,"row_token_sha256":"b"}],12),
                         "partial_window_likelihood_smoke")

    def test_only_three_unique_rows_and_twelve_windows_complete(self):
        rows=[{"row_token_sha256":str(i),"window_mean_claim_logprobs_nats_per_token":[-1]*4}
              for i in range(3)]
        self.assertEqual(_selection_status(rows,12),"completed_window_likelihood_selection")
        self.assertEqual(_selection_status(rows,11),"partial_window_likelihood_smoke")
        self.assertEqual(_selection_status(rows[:2],12),"partial_window_likelihood_smoke")

if __name__=="__main__":unittest.main()
