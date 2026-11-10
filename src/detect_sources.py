r"""Detect preliminary sources in CEERS cutouts shared by three NIRCam bands.

Run from the project root:
    .\.venv\Scripts\python.exe src\detect_sources.py

Requires coverage_regions.csv created by find_coverage_regions.py.
The output is a detection catalog, NOT a verified galaxy or blend catalog.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter


ROOT = Path(__file__).resolve().parents[1]
BANDS = ("f150w", "f277w", "f444w")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Provisional threshold on smoothed combined SNR")
    parser.add_argument("--min-pixels", type=int, default=5)
    parser.add_argument("--regions", type=Path,
                        default=ROOT / "data" / "processed" / "coverage_regions.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "catalog" / "source_candidates.csv")
    parser.add_argument("--preview-output", type=Path,
                        default=ROOT / "results" / "source_detection_preview.png")
    parser.add_argument("--edge-margin", type=float, default=16.0,
                        help="Minimum centroid distance from a cutout edge, in pixels")
    parser.add_argument("--peak-threshold", type=float, default=4.0,
                        help="Minimum local peak height in the exploratory SNR proxy")
    parser.add_argument("--min-peak-separation", type=int, default=4,
                        help="Minimum separation scale for distinct segment peaks, in pixels")
    return parser.parse_args()


def get_band_files():
    directory = ROOT / "data" / "raw" / "fits"
    paths = {}
    for band in BANDS:
        matches = list(directory.glob(f"*_{band}_*_i2d.fits"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {band} FITS file in {directory}")
        paths[band] = matches[0]
    return paths


def read_region(path, x0, y0, size):
    from astropy.io import fits

    region = np.s_[y0:y0 + size, x0:x0 + size]
    with fits.open(path, memmap=True) as hdus:
        science = hdus["SCI_BKSUB"].data[region].copy()
        error = hdus["ERR"].data[region].copy()
        weight = hdus["WHT"].data[region].copy()
    return science, error, weight


def numeric(value):
    return float(value.value if hasattr(value, "value") else value)


def edge_distance(x, y, size):
    """Distance of a source centroid to the nearest cutout boundary."""
    return min(x, y, size - 1 - x, size - 1 - y)


def count_segment_peaks(image, segment_mask, min_height=4.0, separation=4):
    """Count distinct bright local maxima within one detected segment.

    Peaks are screening signals, not evidence of distinct galaxies.
    """
    if image.ndim != 2 or segment_mask.shape != image.shape:
        raise ValueError("Expected a 2-D image and same-shaped segment mask")
    if not np.isfinite(min_height) or min_height <= 0 or separation < 1:
        raise ValueError("Peak threshold must be positive and separation >= 1")
    masked = np.where(segment_mask & np.isfinite(image), image, -np.inf)
    maxima = maximum_filter(masked, size=3, mode="constant", cval=-np.inf)
    peaks = (masked >= min_height) & (masked == maxima)
    plateaus, count = label(peaks, structure=np.ones((3, 3), dtype=bool))
    candidates = []
    for idx in range(1, count + 1):
        # Represent flat-topped peaks once; sorting makes ties deterministic.
        y, x = np.argwhere(plateaus == idx)[0]
        candidates.append((float(masked[y, x]), int(y), int(x)))

    accepted = []
    for _, y, x in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if all((x - px) ** 2 + (y - py) ** 2 >= separation ** 2
               for py, px in accepted):
            accepted.append((y, x))
    return len(accepted)


def main():
    args = get_args()

    import matplotlib.pyplot as plt
    from photutils.segmentation import SourceCatalog, detect_sources

    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.min_pixels < 1:
        raise ValueError("Detection threshold and minimum pixel count must be positive")
    if (args.edge_margin < 0 or not np.isfinite(args.edge_margin)
            or args.peak_threshold <= 0 or not np.isfinite(args.peak_threshold)
            or args.min_peak_separation < 1):
        raise ValueError("Invalid edge or peak screening settings")

    region_csv = args.regions
    with region_csv.open(newline="", encoding="utf-8") as stream:
        regions = list(csv.DictReader(stream))
    if not regions:
        raise RuntimeError("coverage_regions.csv has no regions")

    files = get_band_files()
    detected = []
    preview = None
    max_sources = -1

    for region in regions:
        region_id = region["region_id"]
        x0, y0, size = (int(region[key]) for key in ("x0", "y0", "size"))
        images = {}
        standardized = []
        shared_valid = np.ones((size, size), dtype=bool)

        for band in BANDS:
            science, error, weight = read_region(files[band], x0, y0, size)
            if science.shape != (size, size):
                raise RuntimeError(f"Unexpected crop size for {band} in {region_id}")
            images[band] = science
            valid = ((weight > 0) & np.isfinite(science)
                     & np.isfinite(error) & (error > 0))
            shared_valid &= valid
            z = np.zeros_like(science, dtype=np.float32)
            if valid.any():
                background = np.median(science[valid])
                z[valid] = (science[valid] - background) / error[valid]
            standardized.append(np.clip(z, -10, 25))

        # An exploratory multi-band SNR proxy; not a calibrated significance map.
        combined = np.sum(standardized, axis=0) / np.sqrt(len(BANDS))
        combined[~shared_valid] = 0
        smooth = gaussian_filter(combined, sigma=1.0)
        smooth[~shared_valid] = 0

        segments = detect_sources(
            smooth, threshold=args.threshold, n_pixels=args.min_pixels,
            mask=~shared_valid,
        )
        centroids = []
        if segments is not None:
            catalog = SourceCatalog(np.maximum(smooth, 0), segments, mask=~shared_valid)
            for source in catalog:
                x, y = numeric(source.x_centroid), numeric(source.y_centroid)
                if not np.isfinite(x + y):
                    continue
                centroids.append((x, y))
                boundary_distance = edge_distance(x, y, size)
                peak_count = count_segment_peaks(
                    smooth, segments.data == int(source.label),
                    min_height=args.peak_threshold,
                    separation=args.min_peak_separation,
                )
                detected.append({
                    "region_id": region_id,
                    "source_label": int(source.label),
                    "x_local": round(x, 3),
                    "y_local": round(y, 3),
                    "x_global": round(x0 + x, 3),
                    "y_global": round(y0 + y, 3),
                    "area_pixels": round(numeric(source.area), 2),
                    "edge_distance_px": round(boundary_distance, 3),
                    "interior_candidate": int(boundary_distance >= args.edge_margin),
                    "peak_count": peak_count,
                    "multi_peak_candidate": int(peak_count >= 2),
                })

        print(f"{region_id}: {len(centroids)} preliminary sources")
        if len(centroids) > max_sources:
            preview = (region_id, images, centroids)
            max_sources = len(centroids)

    catalog_file = args.output
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    fields = ("region_id", "source_label", "x_local", "y_local",
              "x_global", "y_global", "area_pixels", "edge_distance_px",
              "interior_candidate", "peak_count", "multi_peak_candidate")
    with catalog_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detected)

    region_id, images, centroids = preview
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, band in zip(axes, BANDS):
        image = images[band]
        finite = image[np.isfinite(image)]
        lo, hi = np.percentile(finite, (5, 99.5)) if finite.size else (0, 1)
        if hi <= lo:
            hi = lo + 1
        ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        if centroids:
            xy = np.asarray(centroids)
            ax.scatter(xy[:, 0], xy[:, 1], facecolors="none",
                       edgecolors="cyan", s=80, linewidths=1)
        ax.set_title(band.upper())
    fig.suptitle(f"Preliminary source detections: {region_id}")
    fig.tight_layout()

    preview_file = args.preview_output
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_file, dpi=160)
    plt.close(fig)

    print(f"Total preliminary sources: {len(detected)}")
    print(f"Saved: {catalog_file}")
    print(f"Saved: {preview_file}")
    print("Peak/edge flags are screening aids, not verified galaxy/blend labels.")


if __name__ == "__main__":
    main()
