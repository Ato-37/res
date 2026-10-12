import json
import tempfile
import unittest
from pathlib import Path

from derive_necessity_smoke_summary import derive


class Tests(unittest.TestCase):
    def _raw(self):
        return {"n_requested": 3, "n_effective": 3, "failures": [], "cap_pixels_per_frame": True,
                "frame_size": 1024, "max_pixels": 200704,
                "records": [{"input_metadata": {"full": {"input_ids_shape": [1, 99], "video_grid_thw": [[8, 40, 26]]}}}],
                "mask_intervention": {"fill": "black"}, "summary": {
                    "all": {"mean": 0.1, "ci95_pointwise": [-0.2, 0.4]},
                    "baseline_correct_primary": {"n": 2, "mean": 0.2, "ci95_pointwise": [-0.1, 0.5]}}}

    def test_derives_safe_fields(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.json"; p.write_text(json.dumps(self._raw()))
            out = derive(p)
            self.assertEqual(out["n_effective"], 3)
            self.assertEqual(out["mask_fill"], "black")
            self.assertEqual(out["unique_video_grid_thw"], [[8, 40, 26]])
            self.assertEqual(out["frame_size"], 1024)
            self.assertNotIn("records", out)

    def test_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.json"; x = self._raw(); x["n_effective"] = 2; p.write_text(json.dumps(x))
            with self.assertRaises(ValueError): derive(p)


if __name__ == "__main__": unittest.main()
