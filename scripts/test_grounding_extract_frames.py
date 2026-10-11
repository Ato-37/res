"""Mocked CPU tests for frame-extraction cleanup and returned ownership."""

import shutil
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import grounding_check


class _FakeCapture:
    def __init__(self, timestamps_ms, fail_index=None):
        self.timestamps_ms = timestamps_ms
        self.fail_index = fail_index
        self.current_index = 0
        self.released = False
        self.read_indices = []

    def get(self, prop):
        if prop == grounding_check.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.timestamps_ms)
        if prop == grounding_check.cv2.CAP_PROP_POS_MSEC:
            return self.timestamps_ms[self.current_index]
        raise AssertionError(f"unexpected capture property: {prop}")

    def set(self, prop, value):
        if prop != grounding_check.cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(f"unexpected set property: {prop}")
        self.current_index = int(value)
        return True

    def read(self):
        self.read_indices.append(self.current_index)
        if self.current_index == self.fail_index:
            return False, None
        return True, np.zeros((20, 30, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class GroundingExtractFramesTests(unittest.TestCase):
    def _patched_capture(self, timestamps_ms, fail_index=None):
        captures = []

        def create_capture(_path):
            capture = _FakeCapture(timestamps_ms, fail_index=fail_index)
            captures.append(capture)
            return capture

        return captures, mock.patch.object(
            grounding_check.cv2, "VideoCapture", side_effect=create_capture
        )

    def _track_temp_dirs(self):
        created = []
        original = tempfile.mkdtemp

        def make_temp_dir(*args, **kwargs):
            path = original(*args, **kwargs)
            created.append(Path(path))
            return path

        return created, mock.patch.object(grounding_check.tempfile, "mkdtemp", side_effect=make_temp_dir)

    def test_nonuniform_timestamps_reject_and_remove_temp_frames(self):
        captures, capture_patch = self._patched_capture([0, 1000, 2100, 3000])
        temp_dirs, temp_patch = self._track_temp_dirs()
        with capture_patch, temp_patch:
            with self.assertRaisesRegex(ValueError, "nonuniform timestamps"):
                grounding_check.extract_frames("unused.mp4", 4, 15)

        self.assertEqual(len(temp_dirs), 1)
        self.assertFalse(temp_dirs[0].exists())
        self.assertTrue(captures[0].released)
        self.assertEqual(captures[0].read_indices, [0, 1, 2, 3])

    def test_decode_failure_removes_partial_frames_and_releases_capture(self):
        captures, capture_patch = self._patched_capture([0, 1000, 2000, 3000], fail_index=2)
        temp_dirs, temp_patch = self._track_temp_dirs()
        with capture_patch, temp_patch:
            with self.assertRaisesRegex(RuntimeError, "Could not read frame 2"):
                grounding_check.extract_frames("unused.mp4", 4, 15)

        self.assertEqual(len(temp_dirs), 1)
        self.assertFalse(temp_dirs[0].exists())
        self.assertTrue(captures[0].released)
        self.assertEqual(captures[0].read_indices, [0, 1, 2])

    def test_success_returns_existing_temp_directory_owned_by_caller(self):
        captures, capture_patch = self._patched_capture([0, 1000, 2000, 3000])
        temp_dirs, temp_patch = self._track_temp_dirs()
        try:
            with capture_patch, temp_patch:
                paths, temp_dir, timestamps, fps = grounding_check.extract_frames(
                    "unused.mp4", 4, 15
                )
            self.assertEqual(len(paths), 4)
            self.assertEqual(Path(temp_dir), temp_dirs[0])
            self.assertTrue(Path(temp_dir).is_dir())
            self.assertTrue(all(Path(path).is_file() for path in paths))
            self.assertEqual(timestamps, [0.0, 1.0, 2.0, 3.0])
            self.assertEqual(fps, 1.0)
            self.assertTrue(captures[0].released)
            self.assertEqual(captures[0].read_indices, [0, 1, 2, 3])
        finally:
            for path in temp_dirs:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
