"""Synthetic tests for candidate peak and cutout-boundary screening."""

import sys
import unittest
from pathlib import Path

import numpy as np


# Add the project root to Python's module search path
# so tests can import the reorganized source packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import source detection and screening utilities.
from src.preprocessing.detect_sources import (
    count_segment_peaks,
    edge_distance
)


class DetectionScreeningTests(unittest.TestCase):
    def test_one_peak_is_not_a_multi_peak_candidate(self):
        y, x = np.mgrid[:64, :64]
        image = 12.0 * np.exp(-((x - 25) ** 2 + (y - 25) ** 2) / 18)
        mask = np.ones_like(image, dtype=bool)
        self.assertEqual(count_segment_peaks(image, mask, min_height=4), 1)

    def test_two_separated_peaks_in_one_segment(self):
        y, x = np.mgrid[:64, :64]
        image = (12.0 * np.exp(-((x - 18) ** 2 + (y - 30) ** 2) / 18)
                 + 10.0 * np.exp(-((x - 48) ** 2 + (y - 30) ** 2) / 18))
        mask = np.ones_like(image, dtype=bool)
        self.assertEqual(count_segment_peaks(image, mask, min_height=4,
                                             separation=4), 2)

    def test_close_equal_height_peaks_are_counted_once(self):
        image = np.zeros((32, 32), dtype=float)
        image[16, 15] = 10
        image[16, 18] = 10
        mask = np.ones_like(image, dtype=bool)
        self.assertEqual(count_segment_peaks(image, mask, min_height=4,
                                             separation=4), 1)

    def test_mask_and_min_height_are_respected(self):
        image = np.zeros((32, 32), dtype=float)
        image[8, 8] = 10
        image[24, 24] = 6
        mask = np.ones_like(image, dtype=bool)
        mask[24, 24] = False
        self.assertEqual(count_segment_peaks(image, mask), 1)
        self.assertEqual(count_segment_peaks(image, mask, min_height=11), 0)

    def test_edge_screening(self):
        self.assertAlmostEqual(edge_distance(123.468, 30.991, 128), 3.532)
        self.assertGreater(edge_distance(64, 64, 128), 16)
        self.assertLess(edge_distance(78.023, 113.2, 128), 16)

    def test_invalid_peak_inputs(self):
        image = np.ones((8, 8), dtype=float)
        with self.assertRaises(ValueError):
            count_segment_peaks(image, np.ones((7, 8), dtype=bool))
        with self.assertRaises(ValueError):
            count_segment_peaks(image, np.ones_like(image, dtype=bool), separation=0)


if __name__ == "__main__":
    unittest.main()
