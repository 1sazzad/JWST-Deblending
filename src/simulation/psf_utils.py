"""
Module: psf_utils.py

Purpose:
    Load and inspect empirical JWST/NIRCam PSF models
    released with the official CEERS DR1 catalog.

Reference:
    Cox et al., The CEERS Photometric and Physical
    Parameter Catalog.

Usage:
    python src/psf_utils.py

The original PSF data are preserved without modification.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from time import perf_counter


# Resolve paths independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PSF_DIR = (
    PROJECT_ROOT
    / "data/psf/ceers_dr1/ceers_psfs_v1.0"
)

# Maintain a consistent band order across the research.
BANDS = ("f150w", "f277w", "f444w")

# Published CEERS catalog PSF specifications.
EXPECTED_SHAPE = (101, 101)
PIXEL_SCALE_ARCSEC = 0.03


def load_psf(band):
    """
    Load an original empirical PSF from its FITS file.

    Parameters:
        band (str): NIRCam filter identifier.

    Returns:
        psf (numpy.ndarray): Two-dimensional PSF array.
        header: FITS header associated with the PSF.

    Raises:
        ValueError: Unsupported band or invalid PSF data.
        FileNotFoundError: Required PSF file is missing.
    """

    band = band.lower()

    if band not in BANDS:
        raise ValueError(f"Unsupported band: {band}")

    path = PSF_DIR / f"{band}_psf.fits"

    if not path.is_file():
        raise FileNotFoundError(path)

    # Read the PSF and its associated metadata.
    data, header = fits.getdata(path, header=True)

    psf = np.asarray(data, dtype=np.float64).squeeze()

    # A convolution kernel must be a two-dimensional array.
    if psf.ndim != 2:
        raise ValueError(f"{band}: PSF must be 2D")

    # Reject undefined or infinite pixel values.
    if not np.all(np.isfinite(psf)):
        raise ValueError(f"{band}: Non-finite PSF values")

    # A positive total response is required for subsequent
    # flux-preserving normalization.
    if psf.sum() <= 0:
        raise ValueError(f"{band}: Invalid PSF total")

    return psf, header


def inspect_psf(band):
    """
    Report PSF properties before applying normalization,
    cropping, resampling, or convolution.
    """

    psf, header = load_psf(band)

    # Locate the brightest pixel in the empirical PSF.
    peak_y, peak_x = np.unravel_index(
        np.argmax(psf), psf.shape
    )

    # Determine the geometric center of the PSF array.
    center_y = (psf.shape[0] - 1) / 2
    center_x = (psf.shape[1] - 1) / 2

    # Measure peak displacement from the array center.
    peak_offset = np.hypot(
        peak_x - center_x,
        peak_y - center_y
    )

    # Empirical PSFs may contain negative values caused
    # by noise or background estimation.
    # Report them rather than silently clipping them.
    negative_count = np.count_nonzero(psf < 0)

    print(f"\n--- {band.upper()} PSF ---")
    print(f"Shape: {psf.shape}")
    print(f"Expected shape: {psf.shape == EXPECTED_SHAPE}")
    print(f"Raw PSF sum: {psf.sum():.8f}")
    print(f"Minimum value: {psf.min():.8e}")
    print(f"Maximum value: {psf.max():.8e}")
    print(f"Negative pixels: {negative_count}")
    print(f"Peak position (x, y): ({peak_x}, {peak_y})")
    print(f"Peak offset: {peak_offset:.3f} pixels")

    # Inspect available pixel-scale metadata.
    # Missing keywords do not imply an invalid PSF.
    for key in ("PIXSCALE", "PIXSCAL", "CDELT1", "CD1_1"):
        if key in header:
            print(f"{key}: {header[key]}")


def normalize_psf(psf, nonnegative=False):
    """
    Normalize a PSF kernel to preserve integrated flux.

    Parameters:
        psf (numpy.ndarray):
            Original two-dimensional empirical PSF.

        nonnegative (bool):
            If True, explicitly remove negative values
            before normalization.

            If False, preserve the original signed
            empirical PSF structure.

    Returns:
        numpy.ndarray:
            Normalized PSF with a total sum of one.

    Notes:
        The original input array is never modified.
        Negative-value removal is optional because it
        may alter the empirical PSF characteristics.
    """

    kernel = np.array(psf, dtype=np.float64, copy=True)

    # Validate kernel dimensions and numerical values.
    if kernel.ndim != 2:
        raise ValueError("PSF must be two-dimensional")

    if not np.all(np.isfinite(kernel)):
        raise ValueError("PSF contains non-finite values")

    # Apply non-negativity only when explicitly requested.
    if nonnegative:
        kernel = np.maximum(kernel, 0.0)

    # Normalize the integrated response to unity.
    total = kernel.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid PSF normalization factor")

    kernel /= total

    return kernel

def validate_psf(band):
    """
    Evaluate normalized PSF properties and verify
    integrated flux preservation through convolution.

    Two kernel variants are examined:
        1. Signed empirical PSF.
        2. Non-negative empirical PSF.

    No changes are saved to the original FITS files.
    """

    psf, _ = load_psf(band)

    # Quantify the negative contribution rather than
    # relying only on the number of negative pixels.
    negative_mass = -psf[psf < 0].sum()
    positive_mass = psf[psf > 0].sum()

    negative_fraction = negative_mass / positive_mass

    # Prepare two explicitly defined normalization modes.
    signed_kernel = normalize_psf(
        psf, nonnegative=False
    )

    positive_kernel = normalize_psf(
        psf, nonnegative=True
    )

    # Measure how much clipping changes the normalized PSF.
    kernel_difference = np.abs(
        signed_kernel - positive_kernel
    ).sum()

    print(f"\n--- {band.upper()} Validation ---")
    print(f"Negative mass fraction: {negative_fraction:.8e}")
    print(f"Signed PSF sum: {signed_kernel.sum():.12f}")
    print(f"Positive PSF sum: {positive_kernel.sum():.12f}")
    print(f"Kernel L1 difference: {kernel_difference:.8e}")

    # Construct an artificial point source with known flux.
    # A sufficiently large image keeps the centered PSF
    # support inside the output image.
    image_size = 129
    center = image_size // 2
    true_flux = 100.0

    source = np.zeros(
        (image_size, image_size), dtype=np.float64
    )

    source[center, center] = true_flux

    # Validate both kernel variants independently.
    for name, kernel in (
        ("Signed", signed_kernel),
        ("Non-negative", positive_kernel),
    ):

        # Full convolution retains the complete
        # finite-support convolution output.
        observed = fftconvolve(
            source, kernel, mode="full"
        )

        recovered_flux = observed.sum()

        relative_error = abs(
            recovered_flux - true_flux
        ) / true_flux

        # Verify normalization and flux conservation.
        np.testing.assert_allclose(
            kernel.sum(),
            1.0,
            rtol=0,
            atol=1e-10
        )

        np.testing.assert_allclose(
            recovered_flux,
            true_flux,
            rtol=0,
            atol=1e-8
        )

        # Check the location of the brightest pixel
        # in the full convolution output.
        peak_y, peak_x = np.unravel_index(
            np.argmax(observed), observed.shape
        )

        expected_y = center + kernel.shape[0] // 2
        expected_x = center + kernel.shape[1] // 2

        assert (peak_y, peak_x) == (
            expected_y, expected_x
        ), f"{band}: Unexpected peak displacement"

        print(f"\nKernel type: {name}")
        print(f"Input flux: {true_flux:.8f}")
        print(f"Output flux: {recovered_flux:.8f}")
        print(f"Relative flux error: {relative_error:.8e}")
        print("Flux preservation test: PASSED")
        print("Peak alignment test: PASSED")


def main():
    """
    Execute PSF inspection and numerical validation
    for every selected NIRCam filter.
    """

    # Measure the total execution time.
    start_time = perf_counter()

    print("CEERS DR1 PSF Validation Pipeline")
    print(f"Total filters: {len(BANDS)}")

    for index, band in enumerate(BANDS, start=1):

        # Display progress during execution.
        print(
            f"\nProcessing {index}/{len(BANDS)}: "
            f"{band.upper()}",
            flush=True
        )

        inspect_psf(band)
        validate_psf(band)

    elapsed = perf_counter() - start_time

    print("\nAll PSF validation tests passed.")
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()