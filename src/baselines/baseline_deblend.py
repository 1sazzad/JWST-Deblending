"""
Module: baseline_deblend.py

Purpose:
    Establish a simple PSF-aware photometric baseline
    for controlled multi-band galaxy deblending.

Method:
    Non-negative flux fitting with fixed Gaussian
    morphology and known source positions.

Important:
    Ground-truth flux is used only for evaluation.
    This is a preliminary baseline, not SCARLET or
    the proposed scene-adaptive architecture.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import lsq_linear


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data/processed/synthetic/pilot_scene_0001.npz"
)

# Fixed circular Gaussian width assumed by the baseline.
# This is not fitted using ground-truth galaxy morphology.
TEMPLATE_SIGMA = 2.5


def create_template(shape, position, psf):
    """
    Construct a unit-flux galaxy template and convolve
    it with the PSF of the corresponding filter.

    Parameters:
        shape: Image dimensions (height, width).
        position: Known source position (x, y).
        psf: Normalized point spread function.

    Returns:
        PSF-convolved galaxy template.
    """

    y, x = np.indices(shape, dtype=np.float64)

    x0, y0 = position

    # Assume a fixed circular Gaussian morphology.
    template = np.exp(
        -0.5 * (
            (x - x0) ** 2 + (y - y0) ** 2
        ) / TEMPLATE_SIGMA ** 2
    )

    # Normalize the intrinsic galaxy to unit flux.
    template /= template.sum()

    # Apply the wavelength-dependent telescope blur.
    return fftconvolve(
        template, psf, mode="same"
    )


def fit_band(observed, psf, positions, noise_sigma):
    """
    Estimate individual source fluxes in one filter.

    The observation is modeled as the sum of
    PSF-convolved source templates and a constant
    background term.

    Source fluxes must be non-negative.
    Background intensity may be positive or negative.

    Returns:
        Estimated source fluxes, reconstructed image,
        background estimate and residual RMSE.
    """

    if noise_sigma <= 0:
        raise ValueError("Noise sigma must be positive")

    # Build one template for each catalog source.
    templates = [
        create_template(
            observed.shape, position, psf
        )
        for position in positions
    ]

    # Add a constant background component.
    background = np.ones_like(observed)

    # Each column represents an independent
    # contribution to the observed image.
    design_matrix = np.column_stack(
        [t.ravel() for t in templates]
        + [background.ravel()]
    )

    # Weight the fitting problem by measurement noise.
    weighted_matrix = design_matrix / noise_sigma
    weighted_observed = observed.ravel() / noise_sigma

    n_sources = len(positions)

    # Enforce non-negative galaxy fluxes while allowing
    # an unrestricted constant background.
    lower_bounds = np.array(
        [0.0] * n_sources + [-np.inf]
    )

    upper_bounds = np.full(
        n_sources + 1, np.inf
    )

    result = lsq_linear(
        weighted_matrix,
        weighted_observed,
        bounds=(lower_bounds, upper_bounds)
    )

    if not result.success:
        raise RuntimeError(
            f"Flux fitting failed: {result.message}"
        )

    # The first parameters represent integrated
    # source fluxes. The last is the background.
    estimated_flux = result.x[:n_sources]
    estimated_background = result.x[-1]

    # Reconstruct the blended image from fitted sources.
    reconstructed = (
        design_matrix @ result.x
    ).reshape(observed.shape)

    residual = observed - reconstructed

    # Quantify remaining image reconstruction error.
    rmse = np.sqrt(np.mean(residual ** 2))

    return (
        estimated_flux,
        reconstructed,
        estimated_background,
        rmse
    )


def evaluate_flux(estimated_flux, true_flux, bands):
    """
    Compare recovered flux with simulation ground truth.

    Ground truth is accessed only after fitting.
    """

    # Calculate signed and absolute relative errors.
    relative_error = (
        estimated_flux - true_flux
    ) / true_flux

    absolute_error = np.abs(relative_error)

    print("\n--- Photometric Evaluation ---")

    for band_index, band in enumerate(bands):

        for source_index in range(len(true_flux)):

            truth = true_flux[source_index, band_index]
            estimate = estimated_flux[
                source_index, band_index
            ]

            error = absolute_error[
                source_index, band_index
            ] * 100

            print(
                f"{band.upper()} | "
                f"Source {source_index + 1} | "
                f"True: {truth:.4f} | "
                f"Estimated: {estimate:.4f} | "
                f"Error: {error:.2f}%"
            )

    print("\n--- Summary Metrics ---")

    print(
        f"Mean absolute relative flux error: "
        f"{absolute_error.mean() * 100:.4f}%"
    )

    print(
        f"Median absolute relative flux error: "
        f"{np.median(absolute_error) * 100:.4f}%"
    )

    print(
        f"Mean signed flux error: "
        f"{relative_error.mean() * 100:.4f}%"
    )


def main():
    """
    Load the pilot dataset, perform source fitting,
    and evaluate recovered photometric measurements.
    """

    start_time = perf_counter()

    with np.load(DATASET_PATH, allow_pickle=False) as data:

        # Load only information available during fitting.
        observed = data["observed"]
        psfs = data["psfs"]
        positions = data["positions"]
        noise_sigma = data["noise_sigma"]
        bands = data["bands"]

        n_sources = len(positions)
        n_bands = len(bands)

        estimated_flux = np.zeros(
            (n_sources, n_bands)
        )

        print("PSF-Aware Fixed-Morphology Baseline")

        for i, band in enumerate(bands):

            print(
                f"\nProcessing {i + 1}/{n_bands}: "
                f"{band.upper()}",
                flush=True
            )

            flux, reconstruction, background, rmse = fit_band(
                observed[i],
                psfs[i],
                positions,
                noise_sigma[i]
            )

            estimated_flux[:, i] = flux

            print(f"Estimated flux: {flux}")
            print(f"Background: {background:.8f}")
            print(f"Residual RMSE: {rmse:.8f}")

        # Access true flux only after all fitting is complete.
        true_flux = data["input_flux"]

        evaluate_flux(
            estimated_flux,
            true_flux,
            bands
        )

    elapsed = perf_counter() - start_time

    print(f"\nScenes processed: 1")
    print(f"Sources processed: {n_sources}")
    print(f"Flux measurements: {n_sources * n_bands}")
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()