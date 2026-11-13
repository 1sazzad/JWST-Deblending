"""Regression tests for preliminary detection quality screening."""

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


# Add the project root to Python's module search path
# so tests can import the reorganized source packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import source quality assessment utilities.
from src.preprocessing.source_quality import (QualityCriteria, assess_sources, main, read_csv,
                            read_region_sizes, write_quality_csv)


def source(label, x, y, area, region="region_001"):
    return {
        "region_id": region, "source_label": str(label),
        "x_local": str(x), "y_local": str(y),
        "x_global": str(x), "y_global": str(y),
        "area_pixels": str(area),
    }


class QualityTests(unittest.TestCase):
    def test_close_neighbors_are_flagged_without_deletion(self):
        rows = [source(1, 50, 50, 50), source(2, 56, 50, 45)]
        checked = assess_sources(rows, {"region_001": 128})
        self.assertEqual(len(checked), len(rows))
        self.assertTrue(all("close_neighbor" in r["qc_flags"] for r in checked))
        self.assertEqual(checked[0]["nearest_neighbor_px"], "6.000")
        self.assertEqual(rows[0].get("qc_flags"), None)

    def test_edge_tiny_and_large_are_independent_flags(self):
        rows = [source(1, 2, 50, 5), source(2, 64, 64, 5000)]
        checked = assess_sources(rows, {"region_001": 128})
        self.assertEqual(checked[0]["qc_flags"], "near_edge;tiny_segment")
        self.assertEqual(checked[1]["qc_flags"], "large_segment")

    def test_isolated_detection_remains_unflagged(self):
        checked = assess_sources([source(1, 60, 60, 100)], {"region_001": 128})
        self.assertEqual(checked[0]["nearest_neighbor_px"], "")
        self.assertEqual(checked[0]["needs_review"], "no")

    def test_duplicate_and_unknown_sources_raise(self):
        with self.assertRaises(ValueError):
            assess_sources([source(1, 40, 40, 100), source(1, 60, 60, 100)],
                           {"region_001": 128})
        with self.assertRaises(ValueError):
            assess_sources([source(1, 40, 40, 100, "region_unknown")],
                           {"region_001": 128})

    def test_invalid_coordinates_raise(self):
        with self.assertRaises(ValueError):
            assess_sources([source(1, -1, 40, 100)], {"region_001": 128})
        with self.assertRaises(ValueError):
            assess_sources([source(1, 40, 40, float("nan"))],
                           {"region_001": 128})

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            assess_sources([source(1, 50, 50, 100)], {"region_001": 128},
                           QualityCriteria(close_neighbor_px=-1))

    def test_csv_round_trip_and_region_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            regions_file = directory / "regions.csv"
            with regions_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("region_id", "size"))
                writer.writeheader()
                writer.writerow({"region_id": "region_001", "size": "128"})
            sizes = read_region_sizes(regions_file)
            checked = assess_sources([source(1, 40, 40, 100)], sizes)
            output = directory / "quality.csv"
            write_quality_csv(output, checked)
            loaded = read_csv(output, ("region_id", "qc_flags", "needs_review"))
            self.assertEqual(loaded, checked)

    def test_additional_source_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            row = source(1, 40, 40, 100)
            row["flux_estimate"] = "12.5"
            checked = assess_sources([row], {"region_001": 128})
            output = Path(temp) / "quality.csv"
            write_quality_csv(output, checked)
            loaded = read_csv(output, ("flux_estimate",))
            self.assertEqual(loaded[0]["flux_estimate"], "12.5")

    def test_command_line_writes_expected_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sources = folder / "sources.csv"
            regions = folder / "regions.csv"
            output = folder / "quality.csv"
            with sources.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(source(1, 1, 1, 5)))
                writer.writeheader()
                writer.writerow(source(1, 1, 1, 5))
            with regions.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("region_id", "size"))
                writer.writeheader()
                writer.writerow({"region_id": "region_001", "size": 128})
            with redirect_stdout(io.StringIO()):
                main(["--sources", str(sources), "--regions", str(regions),
                      "--output", str(output)])
            checked = read_csv(output, ("qc_flags", "needs_review"))
            self.assertEqual(checked[0]["qc_flags"], "near_edge;tiny_segment")


if __name__ == "__main__":
    unittest.main()
