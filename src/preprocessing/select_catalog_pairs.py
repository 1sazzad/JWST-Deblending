"""
Module: select_catalog_pairs.py

Purpose:
    Identify nearby source pairs from the official CEERS DR1
    photometric catalog for galaxy deblending experiments.

Inputs:
    - Official CEERS DR1 photometric catalog
    - JWST NIRCam F150W science image header

Outputs:
    - CSV file containing candidate source pairs

Important:
    Spatial proximity alone does not confirm that two sources
    are blended. Further validation is required.
"""

from pathlib import Path
import csv

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from contextlib import ExitStack

# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

# Resolve paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = PROJECT_ROOT / "data/catalog/ceers_dr1/ceers_cat_v1.0.fits.gz"

IMAGE_PATH = (
    PROJECT_ROOT / "data/raw/fits" / "hlsp_ceers_jwst_nircam_nircam1_f150w_v1_i2d.fits"
)

OUTPUT_PATH = PROJECT_ROOT / "data/catalog/ceers_candidate_pairs.csv"

# Initial experimental source separation range in pixels.
# These thresholds are configurable research parameters,
# not physically validated blending boundaries.
MIN_SEPARATION = 4.0
MAX_SEPARATION = 12.0

# Keep sources away from image boundaries so that
# 128 x 128 pixel cutouts can be extracted later.
EDGE_MARGIN = 64

# NIRCam filters used throughout the deblending study.
BANDS = ("f150w", "f277w", "f444w")

# Spatial dimensions of each experimental image cutout.
CUTOUT_SIZE = 128

# Minimum fraction of pixels that must be valid
# simultaneously across all three filters.
MIN_SHARED_COVERAGE = 0.98


def load_catalog_sources():
    """
    Load official catalog sources and transform their
    celestial coordinates into NIRCam image coordinates.

    Returns:
        source_ids: Array of eligible catalog source IDs.
        coordinates: Array of corresponding (x, y) positions.
    """

    catalog = fits.getdata(CATALOG_PATH, 1)

    header = fits.getheader(IMAGE_PATH, "SCI")
    wcs = WCS(header).celestial

    width = header["NAXIS1"]
    height = header["NAXIS2"]

    # Convert sky coordinates (RA, DEC) into zero-based
    # pixel coordinates in the reference NIRCam image.
    x, y = wcs.world_to_pixel_values(catalog["RA"], catalog["DEC"])

    # -----------------------------------------------------
    # Catalog quality flag diagnostics
    # -----------------------------------------------------

    # Select sources located inside the usable image bounds,
    # before applying any catalog quality flags.
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= EDGE_MARGIN)
        & (x < width - EDGE_MARGIN)
        & (y >= EDGE_MARGIN)
        & (y < height - EDGE_MARGIN)
    )

    print("\n--- Catalog Diagnostics ---")
    print("Total catalog sources:", len(catalog))
    print("Sources inside image margin:", np.count_nonzero(inside))

    # Inspect the actual values and frequencies of each flag
    # rather than assuming their encoding.
    for flag_name in ["STAR_FLAG", "BAD_REGION_FLAG"]:

        values, counts = np.unique(catalog[flag_name][inside], return_counts=True)

        print(f"\n{flag_name} distribution:")

        for value, count in zip(values, counts):
            print(f"  Value {value}: {count} sources")

    # Examine all combinations of binary quality flags.
    # This identifies which filtering condition is responsible
    # for rejecting most of the catalog sources.
    print("\nCombined flag distribution:")

    for star_flag in [0, 1]:
        for bad_flag in [0, 1]:

            mask = (
                inside
                & (catalog["STAR_FLAG"] == star_flag)
                & (catalog["BAD_REGION_FLAG"] == bad_flag)
            )

            print(
                f"STAR_FLAG={star_flag}, "
                f"BAD_REGION_FLAG={bad_flag}: "
                f"{np.count_nonzero(mask)} sources"
            )

    print("---------------------------\n")

        # -----------------------------------------------------
    # Apply official CEERS DR1 quality filtering
    # -----------------------------------------------------

    # STAR_FLAG == 0:
    # Source is outside the catalog's star-masked regions.
    #
    # BAD_REGION_FLAG == 0:
    # Source is outside regions flagged for unreliable
    # detections due to incomplete filter coverage.
    #
    # Reference:
    # data/catalog/ceers_dr1/ceers_cat.readme.md
    # Lines 133-136.
    #
    # Note:
    # These quality flags do not guarantee valid coverage
    # in every selected band. Band-specific validation
    # will be performed during cutout preparation.

    valid = (
        inside
        & (catalog["STAR_FLAG"] == 0)
        & (catalog["BAD_REGION_FLAG"] == 0)
    )

    source_ids = catalog["NUMBER"][valid]

    coordinates = np.column_stack((x[valid], y[valid]))

    return source_ids, coordinates


def find_candidate_pairs(source_ids, coordinates):
    """
    Identify source pairs within the configured
    pixel separation range.

    Uses a KD-tree for efficient spatial neighbor search
    instead of comparing every possible source combination.

    Returns:
        List of candidate pair records.
    """

    if len(coordinates) < 2:
        return []

    tree = cKDTree(coordinates)

    # Each unordered neighboring pair is returned once.
    nearby_pairs = tree.query_pairs(r=MAX_SEPARATION)

    results = []

    for i, j in nearby_pairs:

        # Euclidean distance between source centroids.
        distance = np.linalg.norm(coordinates[i] - coordinates[j])

        if distance < MIN_SEPARATION:
            continue

        x1, y1 = coordinates[i]
        x2, y2 = coordinates[j]

        results.append(
            {
                "source_id_1": int(source_ids[i]),
                "source_id_2": int(source_ids[j]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "separation_px": float(distance),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
            }
        )

    # Ensure deterministic output ordering.
    results.sort(
        key=lambda row: (row["separation_px"], row["source_id_1"], row["source_id_2"])
    )

    return results

def validate_pair_coverage(results):
    """
    Validate multi-band image coverage for candidate pairs.

    A pixel is considered valid when:
        - Science intensity is finite.
        - Measurement uncertainty is finite and positive.
        - Exposure weight is finite and positive.

    A candidate passes when:
        - At least 98% of its cutout has shared valid coverage.
        - Both source positions have valid pixels in all bands.

    Parameters:
        results: List of candidate pair dictionaries.

    Returns:
        Updated list containing coverage measurements and
        a coverage validation flag for each candidate.
    """

    half_size = CUTOUT_SIZE // 2

    # ExitStack manages multiple open FITS files and
    # ensures that all files are closed after processing.
    with ExitStack() as stack:

        images = {}

        # Open each mosaic only once.
        # Memory mapping avoids loading entire images into RAM.
        for band in BANDS:

            path = (
                PROJECT_ROOT
                / "data/raw/fits"
                / f"hlsp_ceers_jwst_nircam_nircam1_{band}_v1_i2d.fits"
            )

            images[band] = stack.enter_context(
                fits.open(path, memmap=True)
            )

        # Process each candidate independently.
        for row in results:

            # Calculate the integer midpoint of the source pair.
            cx = int(np.floor(row["center_x"] + 0.5))
            cy = int(np.floor(row["center_y"] + 0.5))

            # Construct a fixed-size cutout around the midpoint.
            x0 = cx - half_size
            x1 = x0 + CUTOUT_SIZE

            y0 = cy - half_size
            y1 = y0 + CUTOUT_SIZE

            masks = []

            for band in BANDS:

                hdul = images[band]

                # Read only the required image sections,
                # rather than loading the complete FITS arrays.
                science = hdul["SCI_BKSUB"].section[y0:y1, x0:x1]
                error = hdul["ERR"].section[y0:y1, x0:x1]
                weight = hdul["WHT"].section[y0:y1, x0:x1]

                # Validate image dimensions.
                expected_shape = (CUTOUT_SIZE, CUTOUT_SIZE)

                if any(
                    arr.shape != expected_shape
                    for arr in (science, error, weight)
                ):
                    raise ValueError(
                        f"Invalid cutout dimensions in {band}"
                    )

                # Construct a valid-pixel mask.
                # Negative science intensities are allowed
                # because background-subtracted data may be negative.
                valid = (
                    np.isfinite(science)
                    & np.isfinite(error)
                    & np.isfinite(weight)
                    & (error > 0)
                    & (weight > 0)
                )

                masks.append(valid)

                # Store the valid-pixel fraction for each band.
                row[f"coverage_{band}"] = float(valid.mean())

            # A shared pixel must be valid in every filter.
            shared_mask = np.logical_and.reduce(masks)

            shared_coverage = float(shared_mask.mean())

            row["shared_coverage"] = shared_coverage

            # Convert global source positions into local
            # cutout coordinates.
            source_positions = [
                (row["x1"], row["y1"]),
                (row["x2"], row["y2"])
            ]

            sources_valid = True

            for sx, sy in source_positions:

                local_x = int(np.floor(sx + 0.5)) - x0
                local_y = int(np.floor(sy + 0.5)) - y0

                # Validate source coordinate boundaries.
                if not (
                    0 <= local_x < CUTOUT_SIZE
                    and 0 <= local_y < CUTOUT_SIZE
                ):
                    sources_valid = False
                    break

                # Verify the nearest pixel to each source
                # is valid across all three filters.
                if not shared_mask[local_y, local_x]:
                    sources_valid = False
                    break

            row["sources_valid"] = int(sources_valid)

            # Preserve every candidate but mark whether
            # it passes the coverage requirements.
            row["coverage_pass"] = int(
                shared_coverage >= MIN_SHARED_COVERAGE
                and sources_valid
            )

    return results


def save_results(results):
    """
    Save candidate pairs to CSV for subsequent
    cutout extraction and visual validation.
    """

    fields = [
        "source_id_1",
        "source_id_2",
        "x1",
        "y1",
        "x2",
        "y2",
        "separation_px",
        "center_x",
        "center_y",
        # Multi-band data quality measurements.
        "coverage_f150w",
        "coverage_f277w",
        "coverage_f444w",
        "shared_coverage",
        "sources_valid",
        "coverage_pass",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()
        writer.writerows(results)


def main():
    """Execute the catalog-based pair selection pipeline."""

    source_ids, coordinates = load_catalog_sources()

    results = find_candidate_pairs(source_ids, coordinates)

        # Validate image coverage before saving candidates.
    results = validate_pair_coverage(results)

    # Count candidates that satisfy the coverage criteria.
    passed = sum(
        row["coverage_pass"] for row in results
    )

    print(f"Coverage-passed pairs: {passed}")
    print(f"Coverage-failed pairs: {len(results) - passed}")

    # Save candidates together with validation results.
    save_results(results)

    print(f"Eligible sources: {len(source_ids)}")
    print(f"Candidate pairs: {len(results)}")
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nFirst 10 candidate pairs:")

    for row in results[:10]:
        print(
            f"Sources: {row['source_id_1']}, "
            f"{row['source_id_2']} | "
            f"Distance: {row['separation_px']:.2f} px"
        )


if __name__ == "__main__":
    main()
