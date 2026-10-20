from __future__ import annotations

import unittest
from PIL import Image

from run_saved_claim_frame_mean_control import make_frame_mean_control


class FrameMeanControlTests(unittest.TestCase):
    def test_framewise_rgb_mean_and_dimensions_are_preserved_as_documented(self):
        first=Image.new("RGB",(2,1));first.putdata([(0,10,255),(1,11,254)])
        second=Image.new("RGB",(1,2));second.putdata([(10,20,30),(30,40,50)])
        original_pixels=[list(x.getdata()) for x in (first,second)]
        result,means=make_frame_mean_control((first,second))
        self.assertEqual(means,[(1,11,255),(20,30,40)])
        self.assertEqual([x.size for x in result],[(2,1),(1,2)])
        self.assertEqual([x.mode for x in result],["RGB","RGB"])
        self.assertEqual([list(x.getdata()) for x in result],[[(1,11,255)]*2,[(20,30,40)]*2])
        self.assertEqual([list(x.getdata()) for x in (first,second)],original_pixels)

    def test_empty_frames_rejected(self):
        with self.assertRaisesRegex(ValueError,"frames_invalid"):
            make_frame_mean_control([])


if __name__=="__main__":unittest.main()
