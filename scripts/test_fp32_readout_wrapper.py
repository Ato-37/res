import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nextgqa_annotation_key_bias as base
import run_nextgqa_annotation_key_bias_fp32_readout as wrapper


class BiasOptionTests(unittest.TestCase):
    def setUp(self):
        self.saved = base.BIAS_NATS
        self.saved_layers = base.LAYERS_1_BASED

    def tearDown(self):
        base.BIAS_NATS = self.saved
        base.LAYERS_1_BASED = self.saved_layers

    def test_layers_all_option(self):
        out = wrapper._consume_layers_option(["prog", "--layers", "all", "--n", "3"])
        self.assertEqual(out, ["prog", "--n", "3"])
        self.assertEqual(base.LAYERS_1_BASED, tuple(range(1, 29)))
        with self.assertRaises(SystemExit):
            wrapper._consume_layers_option(["prog", "--layers", "5-16"])

    def test_option_sets_base_module_and_is_removed(self):
        argv = ["prog", "--mode", "full", "--bias-nats", str(math.log(4)), "--n", "3"]
        out = wrapper._consume_bias_option(list(argv))
        self.assertEqual(out, ["prog", "--mode", "full", "--n", "3"])
        self.assertAlmostEqual(base.BIAS_NATS, math.log(4))

    def test_absent_option_leaves_default(self):
        out = wrapper._consume_bias_option(["prog", "--mode", "full"])
        self.assertEqual(out, ["prog", "--mode", "full"])
        self.assertAlmostEqual(base.BIAS_NATS, math.log(2))

    def test_out_of_range_rejected(self):
        with self.assertRaises(SystemExit):
            wrapper._consume_bias_option(["prog", "--bias-nats", "9"])
        with self.assertRaises(SystemExit):
            wrapper._consume_bias_option(["prog", "--bias-nats", "0"])
        with self.assertRaises(SystemExit):
            wrapper._consume_bias_option(["prog", "--bias-nats", "-9"])

    def test_negative_suppression_bias_accepted(self):
        out = wrapper._consume_bias_option(["prog", "--bias-nats", str(-math.log(16))])
        self.assertEqual(out, ["prog"])
        self.assertAlmostEqual(base.BIAS_NATS, -math.log(16))


if __name__ == "__main__":
    unittest.main()
