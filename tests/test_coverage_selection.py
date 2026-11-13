"""Synthetic checks for reproducible, spatially spread region selection."""

import sys
import unittest
from pathlib import Path


# Add the project root to Python's module search path
# so tests can import the reorganized source packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import coverage selection from the preprocessing package.
from src.preprocessing.find_coverage_regions import select_spread_regions


class CoverageSelectionTests(unittest.TestCase):
    def test_selected_regions_cover_the_field(self):
        candidates = [(1.0, x, y) for y in range(0, 1025, 256)
                      for x in range(0, 1025, 256)]
        selected = select_spread_regions(candidates, max_regions=6, min_spacing=384)
        self.assertEqual(len(selected), 6)
        self.assertEqual(min(row[1] for row in selected), 0)
        self.assertEqual(max(row[1] for row in selected), 1024)
        self.assertEqual(min(row[2] for row in selected), 0)
        self.assertEqual(max(row[2] for row in selected), 1024)
        for index, (_, x1, y1) in enumerate(selected):
            for _, x2, y2 in selected[index + 1:]:
                self.assertGreaterEqual((x1 - x2) ** 2 + (y1 - y2) ** 2,
                                        384 ** 2)

    def test_order_independent_and_prefers_full_coverage_first(self):
        candidates = [(.99, 500, 500), (1.0, 0, 0), (1.0, 0, 1000),
                      (1.0, 1000, 0), (1.0, 1000, 1000)]
        forward = select_spread_regions(candidates, 3, 100)
        backward = select_spread_regions(list(reversed(candidates)), 3, 100)
        self.assertEqual(forward, backward)
        self.assertEqual(forward[0][0], 1.0)

    def test_spacing_may_limit_count(self):
        candidates = [(1.0, 0, 0), (1.0, 10, 10), (1.0, 20, 20)]
        self.assertEqual(len(select_spread_regions(candidates, 10, 100)), 1)
        self.assertEqual(select_spread_regions([], 10, 100), [])

    def test_invalid_selection_arguments(self):
        with self.assertRaises(ValueError):
            select_spread_regions([(1.0, 0, 0)], 0, 100)
        with self.assertRaises(ValueError):
            select_spread_regions([(1.0, 0, 0)], 1, -1)


if __name__ == "__main__":
    unittest.main()
