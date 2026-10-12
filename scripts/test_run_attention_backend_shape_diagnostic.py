import unittest
import torch
from run_attention_backend_shape_diagnostic import _manual, _manual_fp32_accum, _sdpa


class AttentionBackendTests(unittest.TestCase):
    def test_shapes_and_finiteness(self):
        q = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        for fn in (_manual, _manual_fp32_accum, _sdpa):
            out = fn(q, k, v)
            self.assertEqual(tuple(out.shape), tuple(q.shape))
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_prefix_append_does_not_change_previous_sdpa_rows(self):
        q = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        short = _sdpa(q[:, :, :4], k[:, :, :4], v[:, :, :4])
        long = _sdpa(q, k, v)
        self.assertTrue(torch.equal(short, long[:, :, :4]))


if __name__ == "__main__": unittest.main()
