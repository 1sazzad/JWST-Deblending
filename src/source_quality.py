r"""Quality-screen preliminary CEERS detections without removing any sources.

Run from the JWST-Deblending project root:
    .\.venv\Scripts\python.exe src\source_quality.py

Reads existing source_candidates.csv and coverage_regions.csv. Writes a separate
source_quality.csv with reproducible, conservative review flags. A close pair
is NOT evidence that the sources are galaxies or physically blended.
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COLUMNS = (
    "region_id", "source_label", "x_local", "y_local",
    "x_global", "y_global", "area_pixels",
)
QC_COLUMNS = (
    "nearest_neighbor_px", "border_distance_px", "qc_flags", "needs_review",
)


@dataclass(frozen=True)
class QualityCriteria:
    edge_margin_px: float = 8.0
    tiny_area_px: float = 8.0
    large_area_fraction: float = 0.20
    close_neighbor_px: float = 12.0

    def validate(self):
        if self.edge_margin_px < 0 or self.tiny_area_px < 0:
            raise ValueError("Edge and area thresholds must be nonnegative")
        if not (0 < self.large_area_fraction <= 1):
            raise ValueError("Large-area fraction must be in (0, 1]")
        if self.close_neighbor_px < 0:
            raise ValueError("Neighbor distance must be nonnegative")


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = set(required) - columns
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        return list(reader)


def read_region_sizes(path):
    regions = read_csv(path, ("region_id", "size"))
    sizes = {}
    for row in regions:
        region = row["region_id"].strip()
        if not region or region in sizes:
            raise ValueError(f"Invalid or repeated region ID: {region!r}")
        size = int(row["size"])
        if size <= 0:
            raise ValueError(f"Non-positive size for region {region}")
        sizes[region] = size
    return sizes


def assess_sources(sources, region_sizes, criteria=QualityCriteria()):
    """Return annotated copies of all input rows, in their original order."""
    criteria.validate()
    by_region = defaultdict(list)
    unique_ids = set()
    parsed = []

    for index, source in enumerate(sources):
        region = source["region_id"].strip()
        if region not in region_sizes:
            raise ValueError(f"Unknown region {region!r} in source catalog")
        source_id = (region, source["source_label"].strip())
        if not source_id[1] or source_id in unique_ids:
            raise ValueError(f"Missing or repeated source identifier {source_id}")
        unique_ids.add(source_id)

        x = float(source["x_local"])
        y = float(source["y_local"])
        area = float(source["area_pixels"])
        size = region_sizes[region]
        if not all(math.isfinite(v) for v in (x, y, area)):
            raise ValueError(f"Non-finite measurements for {source_id}")
        if not (0 <= x < size and 0 <= y < size and area >= 0):
            raise ValueError(f"Out-of-range measurements for {source_id}")

        parsed.append((region, x, y, area, size))
        by_region[region].append(index)

    annotated = []
    for index, source in enumerate(sources):
        region, x, y, area, size = parsed[index]
        neighbors = (
            math.hypot(x - parsed[other][1], y - parsed[other][2])
            for other in by_region[region] if other != index
        )
        nearest = min(neighbors, default=None)
        border = min(x, y, size - 1 - x, size - 1 - y)

        flags = []
        if border < criteria.edge_margin_px:
            flags.append("near_edge")
        if area < criteria.tiny_area_px:
            flags.append("tiny_segment")
        if area > criteria.large_area_fraction * size * size:
            flags.append("large_segment")
        if nearest is not None and nearest < criteria.close_neighbor_px:
            flags.append("close_neighbor")

        annotated.append({
            **source,
            "nearest_neighbor_px": "" if nearest is None else f"{nearest:.3f}",
            "border_distance_px": f"{border:.3f}",
            "qc_flags": ";".join(flags),
            "needs_review": "yes" if flags else "no",
        })
    return annotated


def write_quality_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = ([key for key in rows[0]
                      if key not in SOURCE_COLUMNS + QC_COLUMNS] if rows else [])
    fieldnames = SOURCE_COLUMNS + tuple(extra_columns) + QC_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=PROJECT_ROOT / "data/catalog/source_candidates.csv")
    parser.add_argument("--regions", type=Path,
                        default=PROJECT_ROOT / "data/processed/coverage_regions.csv")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data/catalog/source_quality.csv")
    parser.add_argument("--edge-margin", type=float, default=8.0)
    parser.add_argument("--tiny-area", type=float, default=8.0)
    parser.add_argument("--large-fraction", type=float, default=0.20)
    parser.add_argument("--near-distance", type=float, default=12.0)
    args = parser.parse_args(argv)

    criteria = QualityCriteria(args.edge_margin, args.tiny_area,
                               args.large_fraction, args.near_distance)
    sources = read_csv(args.sources, SOURCE_COLUMNS)
    regions = read_region_sizes(args.regions)
    checked = assess_sources(sources, regions, criteria)
    write_quality_csv(args.output, checked)

    counts = Counter(flag for row in checked
                     for flag in row["qc_flags"].split(";") if flag)
    reviewed = sum(row["needs_review"] == "yes" for row in checked)
    print(f"Assessed {len(checked)} sources; {reviewed} flagged for review.")
    for flag in ("near_edge", "tiny_segment", "large_segment", "close_neighbor"):
        print(f"  {flag}: {counts[flag]}")
    print(f"Saved: {args.output}")
    print("Flags are screening aids, not verified galaxy/blend classifications.")


if __name__ == "__main__":
    main()
