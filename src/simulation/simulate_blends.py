"""
Module: simulate_blends.py

Purpose:
    Generate controlled multi-band galaxy blends with
    known source positions, morphology and integrated flux.

Research application:
    Quantitative evaluation of galaxy deblending methods.

Important:
    This initial simulator uses analytical Gaussian
    galaxies and independent Gaussian noise. It is not
    yet a realistic JWST population simulator.

Output:
    data/processed/synthetic/pilot_scene_0001.npz
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

# Import shared PSF loading and normalization utilities
# from the simulation package.
from src.simulation.psf_utils import (
    load_psf,
    normalize_psf,
    BANDS
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/synthetic/pilot_scene_0001.npz"
)

IMAGE_SIZE = 128
PADDING = 64
RANDOM_SEED = 42

# Source centers in zero-based cutout coordinates.
# The horizontal separation between sources is 6 pixels.
POSITIONS = np.array([
    [60.5, 63.5],
    [66.5, 63.5]
])

# Known integrated source flux in arbitrary units.
# Rows represent galaxies; columns represent filters.
TRUE_FLUX = np.array([
    [100.0, 80.0, 60.0],
    [25.0, 45.0, 80.0]
])

# Independent Gaussian noise standard deviation per pixel.
# These are illustrative values, not measured CEERS noise.
NOISE_SIGMA = np.array([0.02, 0.03, 0.04])


def gaussian_galaxy(
    size, x0, y0, sigma_x, sigma_y, angle
):
    """
    Create a normalized elliptical Gaussian galaxy.

    Parameters:
        size: Square image size in pixels.
        x0, y0: Galaxy center coordinates.
        sigma_x, sigma_y: Gaussian widths in pixels.
        angle: Rotation angle in degrees.

    Returns:
        A 2D galaxy morphology with integrated sum of one.
    """

    y, x = np.indices(
        (size, size), dtype=np.float64
    )

    theta = np.deg2rad(angle)

    # Translate coordinates relative to the galaxy center.
    dx = x - x0
    dy = y - y0

    # Rotate coordinates to produce elliptical morphology.
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)

    # Evaluate the elliptical Gaussian intensity profile.
    galaxy = np.exp(
        -0.5 * (
            (xr / sigma_x) ** 2
            + (yr / sigma_y) ** 2
        )
    )

    # Normalize morphology before applying source flux.
    return galaxy / galaxy.sum()


def simulate_scene(
    positions=None,
    true_flux=None,
    noise_sigma=None,
    morphology=None,
    seed=RANDOM_SEED
):
    """
    Generate a configurable multi-band synthetic galaxy scene.

    Parameters:
        positions:
            Source centers with shape (n_sources, 2).
            Coordinates follow the (x, y) convention.

        true_flux:
            Integrated source flux with shape
            (n_sources, n_bands).

        noise_sigma:
            Gaussian noise standard deviation for each band.

        morphology:
            Gaussian parameters for each source:
            (sigma_x, sigma_y, rotation_angle).

        seed:
            Random seed controlling the noise realization.

    Returns:
        Dictionary containing observations, ground truth,
        PSF kernels and simulation metadata.

    Default parameters reproduce the original pilot scene.
    """

    # Preserve the original pilot configuration when
    # no custom experimental parameters are supplied.
    positions = np.asarray(
        POSITIONS if positions is None else positions,
        dtype=np.float64
    )

    true_flux = np.asarray(
        TRUE_FLUX if true_flux is None else true_flux,
        dtype=np.float64
    )

    noise_sigma = np.asarray(
        NOISE_SIGMA if noise_sigma is None else noise_sigma,
        dtype=np.float64
    )

    if morphology is None:
        morphology = [
            (2.5, 1.8, 25.0),
            (3.2, 2.0, -35.0)
        ]

    morphology = np.asarray(
        morphology, dtype=np.float64
    )

    n_sources = len(positions)
    n_bands = len(BANDS)

    # Validate parameter dimensions and physical constraints.
    if positions.shape != (n_sources, 2):
        raise ValueError("Invalid source positions")

    if true_flux.shape != (n_sources, n_bands):
        raise ValueError("Invalid flux array dimensions")

    if noise_sigma.shape != (n_bands,):
        raise ValueError("Invalid noise dimensions")

    if morphology.shape != (n_sources, 3):
        raise ValueError("Invalid morphology dimensions")

    for array in (
        positions, true_flux, noise_sigma, morphology
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError("Non-finite simulation parameters")

    if (
        np.any(true_flux <= 0)
        or np.any(noise_sigma <= 0)
        or np.any(morphology[:, :2] <= 0)
    ):
        raise ValueError("Invalid simulation parameter values")

    rng = np.random.default_rng(seed)

    # Use a padded canvas to minimize convolution
    # boundary effects.
    padded_size = IMAGE_SIZE + 2 * PADDING
    crop = slice(PADDING, PADDING + IMAGE_SIZE)

    shape = (
        n_sources, n_bands,
        IMAGE_SIZE, IMAGE_SIZE
    )

    components = np.zeros(shape, dtype=np.float64)
    intrinsic = np.zeros_like(components)

    profiles = []

    # Construct the intrinsic morphology of each galaxy.
    for i, (x0, y0) in enumerate(positions):

        sx, sy, angle = morphology[i]

        profile = gaussian_galaxy(
            padded_size,
            x0 + PADDING,
            y0 + PADDING,
            sx, sy, angle
        )

        profiles.append(profile)

    psfs = []

    # Apply the appropriate empirical PSF to every band.
    for b, band in enumerate(BANDS):

        raw_psf, _ = load_psf(band)

        psf = normalize_psf(
            raw_psf, nonnegative=True
        )

        psfs.append(psf)

        for s in range(n_sources):

            source = profiles[s] * true_flux[s, b]

            intrinsic[s, b] = source[crop, crop]

            blurred = fftconvolve(
                source, psf, mode="same"
            )

            components[s, b] = blurred[crop, crop]

    # Generate the blended observation.
    noiseless = components.sum(axis=0)

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma[:, None, None],
        size=noiseless.shape
    )

    observed = noiseless + noise

    # Check that the simulation contains valid data.
    if not np.all(np.isfinite(observed)):
        raise ValueError("Invalid simulated observation")

    # Confirm that intrinsic source flux is preserved.
    np.testing.assert_allclose(
        intrinsic.sum(axis=(-2, -1)),
        true_flux,
        rtol=1e-8,
        atol=1e-8
    )

    return {
        "observed": observed,
        "noiseless": noiseless,
        "components": components,
        "intrinsic": intrinsic,
        "input_flux": true_flux,
        "visible_flux": components.sum(axis=(-2, -1)),
        "psfs": np.stack(psfs),
        "positions": positions,
        "bands": np.array(BANDS),
        "noise_sigma": noise_sigma,
        "morphology": morphology,
        "seed": seed
    }


def validate_scene(scene):
    """
    Verify numerical consistency before saving the scene.

    This validates the simulator, not deblending accuracy.
    """

    expected_shape = (
        len(BANDS), IMAGE_SIZE, IMAGE_SIZE
    )

    assert scene["observed"].shape == expected_shape
    assert np.all(np.isfinite(scene["observed"]))

    # Confirm that the noiseless blend equals the sum
    # of the individual convolved galaxy components.
    np.testing.assert_allclose(
        scene["noiseless"],
        scene["components"].sum(axis=0),
        atol=1e-10
    )

    # Confirm that each unconvolved source retains its
    # specified integrated input flux inside the cutout.
    np.testing.assert_allclose(
        scene["intrinsic"].sum(axis=(-2, -1)),
        scene["input_flux"],
        rtol=1e-8,
        atol=1e-8
    )

    print("\n--- Flux validation ---")

    for index, band in enumerate(BANDS):

        for source in range(len(POSITIONS)):

            true_flux = scene["input_flux"][source, index]
            visible = scene["visible_flux"][source, index]

            relative_loss = (
                abs(true_flux - visible) / true_flux
            )

            print(
                f"{band.upper()} | Source {source + 1} | "
                f"Input: {true_flux:.4f} | "
                f"Visible: {visible:.4f} | "
                f"Relative loss: {relative_loss:.3e}"
            )

    print("\nSimulation consistency tests: PASSED")


def preview_scene(scene):
    """
    Visualize individual convolved galaxies and their blend.

    Each row represents one filter.
    Each column represents a different image component.
    """

    fig, axes = plt.subplots(
        len(BANDS), 3, figsize=(11, 10)
    )

    for i, band in enumerate(BANDS):

        images = [
            scene["components"][0, i],
            scene["components"][1, i],
            scene["observed"][i]
        ]

        titles = [
            "Galaxy A — Ground truth",
            "Galaxy B — Ground truth",
            "Observed blend + noise"
        ]

        # Use the same display scale across each band
        # to make relative source brightness comparable.
        vmax = scene["noiseless"][i].max()
        vmin = -3 * NOISE_SIGMA[i]

        for j, image in enumerate(images):

            ax = axes[i, j]

            ax.imshow(
                image,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(titles[j], fontsize=10)

            # Zoom into the central galaxy region.
            ax.set_xlim(40, 87)
            ax.set_ylim(40, 87)

            ax.set_xticks([])
            ax.set_yticks([])

        axes[i, 0].set_ylabel(band.upper())

    fig.suptitle(
        "Controlled JWST/NIRCam Galaxy Blend",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def main():
    """
    Generate, validate and save the pilot simulation.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display the generated multi-band scene."
    )

    args = parser.parse_args()

    start_time = perf_counter()

    scene = simulate_scene()

    validate_scene(scene)

    # Save observations, ground truth and experimental
    # metadata together in one reproducible dataset file.
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(OUTPUT_PATH, **scene)

    elapsed = perf_counter() - start_time

    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print(f"Scenes generated: 1")
    print(f"Observed band images: {len(BANDS)}")
    print(f"Ground-truth sources: {len(POSITIONS)}")
    print(f"Execution time: {elapsed:.2f} seconds")

    if args.preview:
        preview_scene(scene)


if __name__ == "__main__":
    main()