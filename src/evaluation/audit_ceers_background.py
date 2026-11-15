
from pathlib import Path
from contextlib import ExitStack
from itertools import combinations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

ROOT = Path(r"H:\JWST-Deblending")
DATA = ROOT / "data/raw/fits"
PSF = ROOT / "data/psf/ceers_dr1/ceers_psfs_v1.0"

BANDS = ("f150w", "f277w", "f444w")

SIZE = 256
TARGET = 32
SEED = 11012026

LAGS = (1, 2, 4, 8)
APERTURES = (2, 4, 8, 16)

rng = np.random.default_rng(SEED)


def robust_sigma(values):

    values = np.asarray(values, dtype=float)

    median = np.median(values)

    return 1.4826022185 * np.median(
        np.abs(values - median)
    )


def correlation(a, b):

    if len(a) < 100:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def spatial_correlation(z, valid, lag, axis):

    if axis == 0:

        a = z[:-lag, :]
        b = z[lag:, :]

        keep = (
            valid[:-lag, :]
            & valid[lag:, :]
        )

    else:

        a = z[:, :-lag]
        b = z[:, lag:]

        keep = (
            valid[:, :-lag]
            & valid[:, lag:]
        )

    return correlation(
        a[keep],
        b[keep],
    )


def aperture_samples(
    image,
    error,
    valid,
    background,
    size,
):

    # Non-overlapping square apertures.
    # Use only apertures with all pixels unmasked.

    def blocks(array):

        return array.reshape(
            SIZE // size,
            size,
            SIZE // size,
            size,
        ).transpose(0, 2, 1, 3)

    mask_blocks = blocks(valid)

    keep = mask_blocks.all(axis=(2, 3))

    residual = np.where(
        valid,
        image - background,
        0.0,
    )

    variance = np.where(
        valid,
        error**2,
        0.0,
    )

    observed_sum = blocks(
        residual
    ).sum(axis=(2, 3))

    predicted_sigma = np.sqrt(
        blocks(variance).sum(axis=(2, 3))
    )

    keep &= predicted_sigma > 0

    return (
        observed_sum[keep]
        / predicted_sigma[keep]
    )


def read_patch(hdul, y, x):

    region = np.s_[
        y:y + SIZE,
        x:x + SIZE
    ]

    return {
        name: np.array(
            hdul[name].section[region],
            dtype=float,
            copy=True,
        )

        for name in (
            "SCI_BKSUB",
            "ERR",
            "WHT",
            "BKGMASK",
        )
    }


def coverage_valid(patch):

    return (
        np.isfinite(patch["SCI_BKSUB"])

        & np.isfinite(patch["ERR"])

        & np.isfinite(patch["WHT"])

        & (patch["ERR"] > 0)

        & (patch["WHT"] > 0)
    )


print("=" * 65)
print("PHASE 11.1.2 - CEERS BACKGROUND AUDIT")
print("=" * 65)

stats = {b: [] for b in BANDS}

aperture_data = {
    b: {
        size: []
        for size in APERTURES
    }
    for b in BANDS
}

cross_band = {
    (a, b): []
    for a, b in combinations(BANDS, 2)
}

locations = []

with ExitStack() as stack:

    hduls = {}

    shapes = []

    for band in BANDS:

        path = DATA / (
            f"hlsp_ceers_jwst_nircam_nircam1_"
            f"{band}_v1_i2d.fits"
        )

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

                raise RuntimeError(
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

        wcs = WCS(header).celestial

        area_deg2 = abs(
            np.linalg.det(
                wcs.pixel_scale_matrix
            )
        )

        pixel_sr = (
            area_deg2 * u.deg**2
        ).to_value(u.sr)

        science_unit = header.get(
            "BUNIT",
            "",
        )

        error_unit = hdul["ERR"].header.get(
            "BUNIT",
            "",
        )

        print(f"\n{band.upper()} METADATA")
        print(f"Shape: {shape}")
        print(f"SCI unit: {science_unit}")
        print(f"ERR unit: {error_unit}")
        print(f"Pixel solid angle: {pixel_sr:.10e} sr")

        if science_unit == "MJy/sr":

            print(
                f"1 summed image unit = "
                f"{1e15 * pixel_sr:.6f} nJy"
            )

        psf_path = PSF / f"{band}_psf.fits"

        with fits.open(psf_path) as psf_hdul:

            kernel = next(
                np.asarray(hdu.data, dtype=float)

                for hdu in psf_hdul

                if hdu.data is not None
                and hdu.data.ndim == 2
            )

            print(f"PSF shape: {kernel.shape}")
            print(f"PSF sum: {kernel.sum():.10f}")
            print(f"PSF finite: {np.isfinite(kernel).all()}")

    if len(set(shapes)) != 1:

        raise RuntimeError(
            "Cross-band dimensions differ"
        )

    height, width = shapes[0]

    # Non-overlapping candidate tiles.

    candidates = [
        (y, x)

        for y in range(
            0, height - SIZE + 1, SIZE
        )

        for x in range(
            0, width - SIZE + 1, SIZE
        )
    ]

    rng.shuffle(candidates)

    print(
        f"\nCandidate tiles: {len(candidates)}"
    )

    for y, x in candidates:

        if len(locations) >= TARGET:
            break

        patches = {}

        common_valid = np.ones(
            (SIZE, SIZE),
            dtype=bool,
        )

        for band in BANDS:

            patch = read_patch(
                hduls[band],
                y,
                x,
            )

            good = coverage_valid(patch)

            patches[band] = patch

            common_valid &= good

        # Require nearly complete common coverage.
        if common_valid.mean() < 0.95:
            continue

        common_blank = common_valid.copy()

        for band in BANDS:

            source_mask = patches[band]["BKGMASK"]

            # Assumption: zero denotes unmasked.
            common_blank &= (
                np.isfinite(source_mask)
                & (source_mask == 0)
            )

        # Require enough unmasked pixels to estimate noise.
        if common_blank.mean() < 0.10:
            continue

        locations.append((y, x))

        patch_id = len(locations)

        print(
            f"Patch {patch_id:02d}/{TARGET} | "
            f"x={x} y={y} | "
            f"coverage={common_valid.mean():.3f} | "
            f"blank={common_blank.mean():.3f}",
            flush=True,
        )

        standardized = {}

        for band in BANDS:

            patch = patches[band]

            image = patch["SCI_BKSUB"]

            error = patch["ERR"]

            valid = common_blank

            values = image[valid]

            err_values = error[valid]

            background = float(
                np.median(values)
            )

            empirical = robust_sigma(values)

            err_median = float(
                np.median(err_values)
            )

            z = np.zeros_like(image)

            z[valid] = (
                values - background
            ) / err_values

            standardized[band] = z

            row = {
                "background": background,

                "empirical_sigma": empirical,

                "ordinary_std": float(
                    np.std(values, ddof=1)
                ),

                "err_median": err_median,

                "sigma_ratio": (
                    empirical / err_median
                ),

                "standardized_sigma": robust_sigma(
                    z[valid]
                ),

                "valid_pixels": int(valid.sum()),

                "blank_fraction": valid.mean(),
            }

            for lag in LAGS:

                row[f"corr_x_{lag}"] = (
                    spatial_correlation(
                        z,
                        valid,
                        lag,
                        axis=1,
                    )
                )

                row[f"corr_y_{lag}"] = (
                    spatial_correlation(
                        z,
                        valid,
                        lag,
                        axis=0,
                    )
                )

            for size in APERTURES:

                samples = aperture_samples(
                    image,
                    error,
                    valid,
                    background,
                    size,
                )

                if samples.size:

                    aperture_data[
                        band
                    ][size].append(samples)

            stats[band].append(row)

            if patch_id == 1:

                mask = patch["BKGMASK"]

                finite = mask[
                    np.isfinite(mask)
                ]

                unique, counts = np.unique(
                    finite,
                    return_counts=True,
                )

                print(
                    f"  {band} mask values: "
                    f"{list(zip(unique.tolist(), counts.tolist()))}"
                )

        for a, b in cross_band:

            cross_band[(a, b)].append(
                correlation(
                    standardized[a][common_blank],
                    standardized[b][common_blank],
                )
            )


print("\n" + "=" * 65)
print("BACKGROUND AND NOISE SUMMARY")
print("=" * 65)

print(
    f"Accepted non-overlapping patches: "
    f"{len(locations)}/{TARGET}"
)

if not locations:

    raise RuntimeError(
        "No valid background patches found. "
        "Inspect BKGMASK semantics and coverage."
    )


def median_field(rows, key):

    return float(
        np.nanmedian([
            r[key]
            for r in rows
        ])
    )


for band in BANDS:

    rows = stats[band]

    print(f"\n--- {band.upper()} ---")

    print(
        f"Background median: "
        f"{median_field(rows, 'background'):.8e}"
    )

    print(
        f"Empirical robust sigma: "
        f"{median_field(rows, 'empirical_sigma'):.8e}"
    )

    print(
        f"Ordinary pixel standard deviation: "
        f"{median_field(rows, 'ordinary_std'):.8e}"
    )

    print(
        f"Median ERR: "
        f"{median_field(rows, 'err_median'):.8e}"
    )

    print(
        f"Empirical sigma / median ERR: "
        f"{median_field(rows, 'sigma_ratio'):.6f}"
    )

    print(
        f"Robust sigma of standardized residuals: "
        f"{median_field(rows, 'standardized_sigma'):.6f}"
    )

    print(
        f"Total unmasked pixels: "
        f"{sum(r['valid_pixels'] for r in rows)}"
    )

    print(
        f"Median blank fraction: "
        f"{median_field(rows, 'blank_fraction'):.4f}"
    )

    print("\nSpatial correlation:")

    for lag in LAGS:

        print(
            f"  Lag {lag:2d} px | "
            f"x={median_field(rows, f'corr_x_{lag}'):.6f} | "
            f"y={median_field(rows, f'corr_y_{lag}'):.6f}"
        )

    print("\nSquare-aperture noise scaling:")

    for size in APERTURES:

        chunks = aperture_data[band][size]

        if not chunks:

            print(
                f"  {size}x{size}: no valid apertures"
            )

            continue

        samples = np.concatenate(chunks)

        if samples.size < 30:

            print(
                f"  {size}x{size}: "
                f"insufficient samples "
                f"(N={samples.size})"
            )

            continue

        print(
            f"  {size}x{size} | "
            f"factor={robust_sigma(samples):.6f} | "
            f"N={samples.size}"
        )


print("\n--- Cross-band background correlation ---")

for (a, b), values in cross_band.items():

    print(
        f"{a.upper()} vs {b.upper()}: "
        f"{np.nanmedian(values):.6f}"
    )


print("\n--- Background patch coordinates ---")

for i, (y, x) in enumerate(
    locations, start=1
):

    print(
        f"{i:02d}: x={x}, y={y}"
    )

print("\nBackground audit completed.")
print("No FITS files modified.")

