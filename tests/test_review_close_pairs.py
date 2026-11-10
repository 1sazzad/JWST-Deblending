"""Small, dependency-free tests for pair selection and input validation."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_close_pairs import find_close_pairs, get_region_catalog, save_pair_figure


def row(region, label, x, y, flag="close_neighbor"):
    return {"region_id": region, "source_label": str(label),
            "x_local": str(x), "y_local": str(y), "qc_flags": flag}


class PairReviewTests(unittest.TestCase):
    def test_detects_exactly_two_pairs(self):
        rows = [
            row("region_008", 1, 123.468, 30.991),
            row("region_008", 2, 121.308, 39.693),
            row("region_010", 5, 78.023, 113.2),
            row("region_010", 7, 73.831, 118.339),
        ]
        pairs = find_close_pairs(rows)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["region_id"], "region_010")
        self.assertAlmostEqual(pairs[0]["distance_px"], 6.632, places=2)
        self.assertAlmostEqual(pairs[1]["distance_px"], 8.966, places=2)

    def test_no_cross_region_or_unflagged_pairs(self):
        rows = [row("a", 1, 30, 30), row("b", 2, 30, 30),
                row("a", 3, 31, 30, flag="near_edge")]
        self.assertEqual(find_close_pairs(rows), [])

    def test_duplicate_labels_raise(self):
        with self.assertRaises(ValueError):
            find_close_pairs([row("a", 1, 10, 10), row("a", 1, 12, 10)])

    def test_threshold_is_strict(self):
        sources = [row("a", 1, 10, 10), row("a", 2, 22, 10)]
        self.assertEqual(find_close_pairs(sources, 12.0), [])
        self.assertEqual(len(find_close_pairs(sources, 12.1)), 1)

    def test_region_catalog_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "regions.csv"
            with filename.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream,
                                        fieldnames=("region_id", "x0", "y0", "size"))
                writer.writeheader()
                writer.writerow({"region_id": "region_010", "x0": 7936,
                                 "y0": 256, "size": 128})
            regions = get_region_catalog(filename)
            self.assertEqual(regions["region_010"],
                             {"x0": 7936, "y0": 256, "size": 128})

    def test_review_figure_is_generated(self):
        import numpy as np

        first = row("region_001", 1, 10, 10)
        second = row("region_001", 2, 16, 10)
        pair = find_close_pairs([first, second])[0]
        science = np.zeros((32, 32), dtype=np.float32)
        science[10, 10] = 10
        science[10, 16] = 8
        cutouts = [(science, np.ones_like(science), np.ones_like(science))] * 3
        labels = np.zeros_like(science, dtype=int)
        labels[9:12, 9:12] = 1
        labels[9:12, 15:18] = 2
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "review.png"
            save_pair_figure(pair, cutouts, SimpleNamespace(data=labels),
                             ("f150w", "f277w", "f444w"), destination)
            self.assertGreater(destination.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
