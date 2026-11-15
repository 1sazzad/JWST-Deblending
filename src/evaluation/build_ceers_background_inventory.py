
"""Phase 11.2.1: CEERS background cutout inventory.

Identify non-overlapping multi-band background cutouts
with valid coverage and clean central injection regions.

Preserve spatially separated development, calibration,
and test regions. Never modify the original FITS files.
"""

import csv

from pathlib import Path
from contextlib import ExitStack
from collections import Counter

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "raw" / "fits"

OUTPUT = (
    ROOT / "results"
    / "ceers_background_inventory_phase11_2_1.csv"
)

BANDS = ("f150w", "f277w", "f444w")

PARENT = 256
SIZE = 128

MIN_BLANK = 0.50
PREFERRED_CLEARANCE = 24
MIN_CLEARANCE = 16

OFFSETS = (0, 64, 128)

# Verified photometric conversion.
NJY_PER_UNIT = 21.153987485

# Coordinates are stored as (x, y).
# Exclude all previous background audit tiles.
AUDIT_TILES = {
    (9728, 1024),
    (3072, 2560),
    (8704, 3840),
    (1024, 2048),
    (3840, 1792),
    (6400, 3840),
    (256, 2560),
    (768, 1280),
    (9216, 1280),
    (3328, 2816),
    (8704, 4096),
    (512, 4096),
    (7680, 3072),
    (1792, 1280),
    (8192, 1792),
    (7680, 4096),
    (2304, 2560),
    (7936, 512),
    (6912, 2816),
    (3328, 512),
    (2048, 3840),
    (1280, 3584),
    (512, 3584),
    (7424, 768),
    (7680, 3584),
    (6912, 4096),
    (3584, 1792),
    (2560, 256),
    (3584, 3840),
    (6400, 512),
    (2304, 3840),
    (2816, 1024),
}


# ---------------------------------------------------------
# Spatial split
# ---------------------------------------------------------

def spatial_split(x):

    # Development: x = 0 to 3839
    if x + PARENT <= 3840:
        return "development"

    # Guard gap: x = 3840 to 4095
    # Calibration: x = 4096 to 7167
    if x >= 4096 and x + PARENT <= 7168:
        return "calibration"

    # Guard gap: x = 7168 to 7423
    # Test: x >= 7424
    if x >= 7424:
        return "test"

    return None


# ---------------------------------------------------------
# Background statistics
# ---------------------------------------------------------

def robust_sigma(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    median = np.median(values)

    return float(
        1.4826022185
        * np.median(
            np.abs(values - median)
        )
    )


# ---------------------------------------------------------
# Select one cutout per parent tile
# ---------------------------------------------------------

def choose_cutout(valid, blank):

    # Distance to nearest masked or invalid pixel.
    distance = distance_transform_edt(
        np.pad(
            blank,
            1,
            constant_values=False,
        )
    )[1:-1, 1:-1]

    best = None

    for dy in OFFSETS:

        for dx in OFFSETS:

            region = np.s_[
                dy:dy + SIZE,
                dx:dx + SIZE,
            ]

            # Require complete coverage.
            if not valid[region].all():
                continue

            blank_fraction = float(
                blank[region].mean()
            )

            center_y = dy + SIZE // 2
            center_x = dx + SIZE // 2

            clearance = float(
                distance[center_y, center_x]
            )

            if blank_fraction < MIN_BLANK:
                continue

            if clearance < MIN_CLEARANCE:
                continue

            if blank_fraction == 1.0:

                quality = "A"

            elif clearance >= PREFERRED_CLEARANCE:

                quality = "B"

            else:

                quality = "C"

            # Prefer full blank cutouts, then
            # larger clearance and blank fraction.
            rank = {
                "A": 3,
                "B": 2,
                "C": 1,
            }[quality]

            key = (
                rank,
                clearance,
                blank_fraction,
            )

            if best is None or key > best["key"]:

                best = dict(
                    key=key,

                    dx=dx,
                    dy=dy,

                    quality=quality,

                    blank_fraction=blank_fraction,

                    clearance_px=clearance,
                )

    return best


# ---------------------------------------------------------
# Read one horizontal strip per band
# ---------------------------------------------------------

def read_strip(hdul, y, width):

    region = np.s_[
        y:y + PARENT,
        :width,
    ]

    science = np.asarray(
        hdul["SCI_BKSUB"].section[region],
        dtype=np.float32,
    )

    error = np.asarray(
        hdul["ERR"].section[region],
        dtype=np.float32,
    )

    weight = np.asarray(
        hdul["WHT"].section[region],
        dtype=np.float32,
    )

    mask = np.asarray(
        hdul["BKGMASK"].section[region]
    )

    valid = (
        np.isfinite(science)
        & np.isfinite(error)
        & np.isfinite(weight)
        & (error > 0)
        & (weight > 0)
    )

    blank = (
        valid
        & np.isfinite(mask)
        & (mask == 0)
    )

    return (
        science,
        error,
        valid,
        blank,
    )


# ---------------------------------------------------------
# Main inventory generation
# ---------------------------------------------------------

def main():

    if OUTPUT.exists():

        raise FileExistsError(
            f"Inventory already exists: {OUTPUT}"
        )

    print(
        "Phase 11.2.1 — "
        "CEERS background cutout inventory"
    )

    print(f"Parent tile: {PARENT} x {PARENT}")
    print(f"Cutout size: {SIZE} x {SIZE}")

    print(
        f"Previous audit tiles excluded: "
        f"{len(AUDIT_TILES)}"
    )

    records = []

    rejected = Counter()

    with ExitStack() as stack:

        hduls = {}

        shapes = []

        reference_wcs = None

        for band in BANDS:

            path = DATA / (
                f"hlsp_ceers_jwst_nircam_nircam1_"
                f"{band}_v1_i2d.fits"
            )

            # Use memmap=False because the
            # scaled FITS extensions require it.
            hdul = stack.enter_context(
                fits.open(
                    path,
                    mode="readonly",
                    memmap=False,
                )
            )

            hduls[band] = hdul

            required = (
                "SCI_BKSUB",
                "ERR",
                "WHT",
                "BKGMASK",
            )

            for name in required:

                if name not in hdul:

                    raise ValueError(
                        f"{band}: missing {name}"
                    )

            header = hdul["SCI_BKSUB"].header

            shape = (
                header["NAXIS2"],
                header["NAXIS1"],
            )

            shapes.append(shape)

            for name in required:

                h = hdul[name].header

                if (
                    h["NAXIS2"],
                    h["NAXIS1"],
                ) != shape:

                    raise ValueError(
                        f"{band}: {name} shape mismatch"
                    )

            for name in ("SCI_BKSUB", "ERR"):

                if hdul[name].header.get(
                    "BUNIT"
                ) != "MJy/sr":

                    raise ValueError(
                        f"{band}: unexpected {name} unit"
                    )

            if band == "f150w":

                reference_wcs = WCS(
                    header
                ).celestial

            print(
                f"{band.upper()}: "
                f"shape={shape}",
                flush=True,
            )

        if len(set(shapes)) != 1:

            raise ValueError(
                "Cross-band image shapes differ"
            )

        height, width = shapes[0]

        usable_width = (
            width // PARENT
        ) * PARENT

        number_of_strips = height // PARENT

        total_tiles = (
            number_of_strips
            * (usable_width // PARENT)
        )

        print(
            f"\nTotal parent tiles: {total_tiles}"
        )

        print(
            f"Horizontal strips: {number_of_strips}",
            flush=True,
        )

        for strip_id, y in enumerate(
            range(
                0,
                height - PARENT + 1,
                PARENT,
            ),
            start=1,
        ):

            common_valid = np.ones(
                (PARENT, usable_width),
                dtype=bool,
            )

            common_blank = common_valid.copy()

            strip_science = {}

            strip_error = {}

            # Read one strip at a time instead
            # of loading complete mosaics.
            for band in BANDS:

                science, error, valid, blank = (
                    read_strip(
                        hduls[band],
                        y,
                        usable_width,
                    )
                )

                strip_science[band] = science

                strip_error[band] = error

                common_valid &= valid

                common_blank &= blank

            for x in range(
                0,
                usable_width,
                PARENT,
            ):

                if (x, y) in AUDIT_TILES:

                    rejected["previous_audit"] += 1

                    continue

                split = spatial_split(x)

                if split is None:

                    rejected["guard_gap"] += 1

                    continue

                valid_tile = common_valid[
                    :,
                    x:x + PARENT,
                ]

                blank_tile = common_blank[
                    :,
                    x:x + PARENT,
                ]

                # Skip tiles with insufficient
                # coverage even for one cutout.
                if valid_tile.sum() < SIZE**2:

                    rejected["coverage"] += 1

                    continue

                selected = choose_cutout(
                    valid_tile,
                    blank_tile,
                )

                if selected is None:

                    rejected["no_suitable_cutout"] += 1

                    continue

                dx = selected["dx"]
                dy = selected["dy"]

                cutout_x = x + dx
                cutout_y = y + dy

                center_x = cutout_x + SIZE // 2
                center_y = cutout_y + SIZE // 2

                sky = reference_wcs.pixel_to_world(
                    center_x,
                    center_y,
                )

                region = np.s_[
                    dy:dy + SIZE,
                    cutout_x:cutout_x + SIZE,
                ]

                blank_cutout = common_blank[
                    region
                ]

                row = dict(
                    cutout_id=(
                        f"ceers1_x{cutout_x:05d}"
                        f"_y{cutout_y:05d}"
                    ),

                    split=split,

                    quality=selected["quality"],

                    parent_x=x,
                    parent_y=y,

                    x=cutout_x,
                    y=cutout_y,

                    width=SIZE,
                    height=SIZE,

                    center_x=center_x,
                    center_y=center_y,

                    ra_deg=float(sky.ra.deg),
                    dec_deg=float(sky.dec.deg),

                    coverage_fraction=1.0,

                    blank_fraction=selected[
                        "blank_fraction"
                    ],

                    clearance_px=selected[
                        "clearance_px"
                    ],

                    n_blank_pixels=int(
                        blank_cutout.sum()
                    ),

                    njy_per_unit=NJY_PER_UNIT,
                )

                for band in BANDS:

                    science = strip_science[
                        band
                    ][region]

                    error = strip_error[
                        band
                    ][region]

                    values = science[
                        blank_cutout
                    ]

                    err_values = error[
                        blank_cutout
                    ]

                    background = float(
                        np.median(values)
                    )

                    sigma = robust_sigma(
                        values
                    )

                    median_err = float(
                        np.median(err_values)
                    )

                    standardized_sigma = (
                        robust_sigma(
                            (values - background)
                            / err_values
                        )
                    )

                    row.update({
                        f"{band}_background":
                            background,

                        f"{band}_robust_sigma":
                            sigma,

                        f"{band}_median_err":
                            median_err,

                        f"{band}_standardized_sigma":
                            standardized_sigma,

                        f"{band}_sigma_njy_per_pixel":
                            sigma * NJY_PER_UNIT,
                    })

                records.append(row)

            print(
                f"Strip {strip_id:02d}/"
                f"{number_of_strips:02d} | "
                f"Accepted cutouts: {len(records)}",
                flush=True,
            )

    # -----------------------------------------------------
    # Inventory consistency checks
    # -----------------------------------------------------

    if not records:

        raise RuntimeError(
            "No suitable background cutouts found"
        )

    ids = {
        row["cutout_id"]
        for row in records
    }

    if len(ids) != len(records):

        raise RuntimeError(
            "Duplicate cutout IDs"
        )

    parents = {
        (
            row["parent_x"],
            row["parent_y"],
        )
        for row in records
    }

    if len(parents) != len(records):

        raise RuntimeError(
            "Multiple cutouts in the same parent"
        )

    if parents & AUDIT_TILES:

        raise RuntimeError(
            "Previous audit tile leakage"
        )

    if (
        len(records)
        + sum(rejected.values())
        != total_tiles
    ):

        raise RuntimeError(
            "Parent tile accounting mismatch"
        )

    for row in records:

        if spatial_split(
            row["parent_x"]
        ) != row["split"]:

            raise RuntimeError(
                "Spatial split mismatch"
            )

    # -----------------------------------------------------
    # Save inventory
    # -----------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT.with_suffix(
        ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
        )

        writer.writeheader()

        writer.writerows(records)

    temporary.replace(OUTPUT)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        "\n--- Background inventory summary ---"
    )

    print(
        f"Total parent tiles: {total_tiles}"
    )

    print(
        f"Accepted cutouts: {len(records)}"
    )

    print(
        f"Rejected parent tiles: "
        f"{sum(rejected.values())}"
    )

    print("\nRejection reasons:")

    for reason, count in sorted(
        rejected.items()
    ):

        print(
            f"  {reason}: {count}"
        )

    counts = Counter(
        (r["split"], r["quality"])
        for r in records
    )

    for split in (
        "development",
        "calibration",
        "test",
    ):

        print(f"\n{split.upper()}")

        group = [
            r for r in records
            if r["split"] == split
        ]

        print(f"  Total: {len(group)}")

        for quality in ("A", "B", "C"):

            print(
                f"  Quality {quality}: "
                f"{counts[(split, quality)]}"
            )

        if group:

            print(
                f"  Median blank fraction: "
                f"{np.median([r['blank_fraction'] for r in group]):.4f}"
            )

            print(
                f"  Median clearance: "
                f"{np.median([r['clearance_px'] for r in group]):.2f} px"
            )

            for band in BANDS:

                key = f"{band}_robust_sigma"

                values = np.array([
                    r[key]
                    for r in group
                ])

                print(
                    f"  {band.upper()} noise: "
                    f"median={np.median(values):.6e}, "
                    f"P10={np.percentile(values, 10):.6e}, "
                    f"P90={np.percentile(values, 90):.6e}"
                )

    quality_counts = Counter(
        r["quality"]
        for r in records
    )

    print("\n--- Overall quality ---")

    for quality in ("A", "B", "C"):

        print(
            f"Quality {quality}: "
            f"{quality_counts[quality]}"
        )

    print(
        "\nRecommended initial inventory (A + B): "
        f"{quality_counts['A'] + quality_counts['B']}"
    )

    print(f"\nResults saved: {OUTPUT}")

    print("No FITS files modified.")


if __name__ == "__main__":
    main()
