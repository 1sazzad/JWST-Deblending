r"""Visual review of close source pairs in existing CEERS cutouts.

Run from the project root:
    .\.venv\Scripts\python.exe src\review_close_pairs.py

Uses the existing source-quality catalog. Shows SCI_BKSUB images, detection
centroids, and contours from the same provisional Photutils segmentation
settings as detect_sources.py. These figures are diagnostic, not blend labels.
"""

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_close_pairs(rows, maximum_distance=12.0):
    """Return each flagged same-region close pair once, nearest first."""
    if maximum_distance <= 0:
        raise ValueError("maximum_distance must be positive")
    groups = defaultdict(list)
    for row in rows:
        flags = set(row.get("qc_flags", "").split(";"))
        if "close_neighbor" not in flags:
            continue
        region = row["region_id"].strip()
        label = row["source_label"].strip()
        x, y = float(row["x_local"]), float(row["y_local"])
        if not region or not label or not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Invalid flagged source coordinates or identifier")
        groups[region].append({**row, "x": x, "y": y, "label": label})

    found = []
    for region, detections in groups.items():
        for first, second in combinations(detections, 2):
            if first["label"] == second["label"]:
                raise ValueError(f"Repeated source label in {region}")
            distance = math.hypot(first["x"] - second["x"],
                                  first["y"] - second["y"])
            if distance < maximum_distance:
                found.append({"region_id": region, "first": first,
                              "second": second, "distance_px": distance})
    return sorted(found, key=lambda pair: (pair["distance_px"], pair["region_id"]))


def get_region_catalog(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    regions = {}
    for row in rows:
        name = row["region_id"].strip()
        if name in regions:
            raise ValueError(f"Repeated region: {name}")
        regions[name] = {key: int(row[key]) for key in ("x0", "y0", "size")}
    return regions


def compute_segmentation(cutouts, threshold=2.5, min_pixels=5):
    """Reproduce the detector's provisional 3-band SNR segmentation."""
    import numpy as np
    from photutils.segmentation import detect_sources
    from scipy.ndimage import gaussian_filter

    standardized = []
    shared_valid = None
    for science, error, weight in cutouts:
        valid = ((weight > 0) & np.isfinite(science) &
                 np.isfinite(error) & (error > 0))
        shared_valid = valid if shared_valid is None else shared_valid & valid
        z = np.zeros_like(science, dtype=np.float32)
        if valid.any():
            background = np.median(science[valid])
            z[valid] = (science[valid] - background) / error[valid]
        standardized.append(np.clip(z, -10, 25))

    combined = np.sum(standardized, axis=0) / math.sqrt(len(standardized))
    combined[~shared_valid] = 0
    smooth = gaussian_filter(combined, sigma=1.0)
    smooth[~shared_valid] = 0
    return detect_sources(smooth, threshold=threshold, n_pixels=min_pixels,
                          mask=~shared_valid)


def save_pair_figure(pair, cutouts, segmentation, bands, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    first, second = pair["first"], pair["second"]
    colors = ("cyan", "magenta")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for axis, band, (science, error, weight) in zip(axes, bands, cutouts):
        valid = science[(weight > 0) & np.isfinite(science)]
        lower, upper = (np.percentile(valid, (5, 99.5))
                        if valid.size else (0.0, 1.0))
        if upper <= lower:
            upper = lower + 1.0
        axis.imshow(science, origin="lower", cmap="gray", vmin=lower, vmax=upper)
        if segmentation is not None:
            for item, color in zip((first, second), colors):
                mask = segmentation.data == int(item["label"])
                if np.any(mask) and not np.all(mask):
                    axis.contour(mask.astype(float), levels=[0.5], colors=[color],
                                 linewidths=1.0)
        for item, color in zip((first, second), colors):
            axis.plot(item["x"], item["y"], marker="+", color=color,
                      markersize=11, markeredgewidth=1.5)
            axis.text(item["x"] + 3, item["y"] + 3, item["label"],
                      color=color, fontsize=8)
        axis.set_title(band.upper())

    fig.suptitle(f"{pair['region_id']}: source pair ({first['label']}, "
                 f"{second['label']}) — {pair['distance_px']:.2f} px")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path,
                        default=PROJECT_ROOT / "data/catalog/source_quality.csv")
    parser.add_argument("--regions", type=Path,
                        default=PROJECT_ROOT / "data/processed/coverage_regions.csv")
    parser.add_argument("--near-distance", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--min-pixels", type=int, default=5)
    args = parser.parse_args(argv)

    if args.threshold <= 0 or args.min_pixels < 1:
        raise ValueError("Detection parameters must be positive")
    with args.quality.open(newline="", encoding="utf-8-sig") as stream:
        quality_rows = list(csv.DictReader(stream))
    pairs = find_close_pairs(quality_rows, args.near_distance)
    if not pairs:
        print("No close-neighbor pairs in the quality catalog")
        return

    from detect_sources import BANDS, get_band_files, read_region
    regions = get_region_catalog(args.regions)
    paths = get_band_files()

    for pair in pairs:
        region = pair["region_id"]
        if region not in regions:
            raise ValueError(f"Missing coverage region: {region}")
        x0, y0, size = (regions[region][key] for key in ("x0", "y0", "size"))
        cutouts = [read_region(paths[band], x0, y0, size) for band in BANDS]
        segmentation = compute_segmentation(cutouts, args.threshold, args.min_pixels)
        labels = (pair["first"]["label"], pair["second"]["label"])
        filename = f"pair_review_{region}_{labels[0]}_{labels[1]}.png"
        destination = PROJECT_ROOT / "results" / filename
        save_pair_figure(pair, cutouts, segmentation, BANDS, destination)
        print(f"{region}: {labels[0]}–{labels[1]} "
              f"({pair['distance_px']:.2f} px) → {destination}")
    print("Contour overlap/shape requires human review; no blend classification made.")


if __name__ == "__main__":
    main()
