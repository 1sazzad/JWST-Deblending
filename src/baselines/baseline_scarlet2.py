"""
Module: baseline_scarlet2.py

Purpose:
    Evaluate the official scarlet2 implementation on
    controlled JWST/NIRCam galaxy blends.

Baseline configuration:
    - Joint multi-band reconstruction.
    - Fixed source positions.
    - Optimizable source morphology and spectrum.
    - Empirical CEERS PSFs.
    - No neural morphology prior.

Reference:
    https://scarlet2.readthedocs.io/en/v0.5.1/0-example.html

Ground-truth flux is used only during evaluation.
"""

from functools import partial, lru_cache
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scarlet2 as sc2

from numpyro.distributions import constraints

# Reuse the existing photometric evaluation function.
# This ensures consistent metrics across baselines.
# Reuse the same photometric evaluation metrics
# across all baseline methods.
# Reuse the existing baseline for observation-based
# initialization and consistent photometric evaluation.
from src.baselines.baseline_deblend import (
    evaluate_flux,
    fit_band,
    TEMPLATE_SIGMA
)


# Load pretrained galaxy morphology models.
from galaxygrad import get_prior

# Connect the score-based neural model to scarlet2.
from scarlet2.nn import ScorePrior


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data/processed/synthetic/pilot_scene_0001.npz"
)

MAX_ITERATIONS = 1000
RELATIVE_TOLERANCE = 1e-4
# Fixed spatial support for source initialization.
# Use a larger odd-sized box to allow extended
# galaxy morphology to be represented.
INITIAL_BOX_SIZE = 31
# Select the initialization strategy.
# Supported modes:
#   "moments": Official Gaussian-moment initialization.
#   "warm_start": Observation-based Gaussian initialization.
INITIALIZATION_MODE = "warm_start"
# Prespecified amount of equal-share mixing for the initialization ablation.
# A 10% mixture gives each of two sources at least 5% of the band total
# (before source placement), while preserving the total estimated spectrum.
INITIALIZATION_MIX = 0.10


def balance_initial_spectra(spectra, fraction=INITIALIZATION_MIX):
    """Mix the observation-based spectra with an equal-share allocation.

    This changes only the initial flux allocation, not the morphology, PSFs,
    or optimizer. The input has shape (source, band). No truth is used.
    """
    values = np.asarray(spectra)
    if (values.ndim != 2 or values.shape[0] < 1
            or not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError("Expected finite positive initial spectra")
    if not 0 <= fraction <= 1:
        raise ValueError("Initialization mixture must lie in [0, 1]")
    equal_share = values.mean(axis=0, keepdims=True)
    return (1 - fraction) * values + fraction * equal_share


# Enable or disable neural morphology regularization
# without changing the underlying fitting pipeline.
USE_NEURAL_PRIOR = True

# Pretrained morphology model based on HSC galaxies.
PRIOR_MODEL_NAME = "hsc32"

# Diffusion temperature used by the score-based prior.
# The initial value follows the official scarlet2 tutorial.
PRIOR_TEMPERATURE = 0.01

def initialize_from_baseline(
    observed, psfs, positions, noise_sigma, box_size=INITIAL_BOX_SIZE
):
    """
    Initialize scarlet2 using observation-based flux
    estimates from the fixed-morphology baseline.

    No ground-truth flux or morphology is accessed.

    Returns:
        spectra: Initial source amplitudes per band.
        morphology: Shared initial Gaussian profile.
    """

    n_sources = len(positions)
    n_bands = len(observed)

    initial_flux = np.zeros((n_sources, n_bands))

    # Estimate source flux independently in each band
    # using the existing PSF-aware baseline.
    for band_index in range(n_bands):

        flux, _, _, _ = fit_band(
            observed[band_index],
            psfs[band_index],
            positions,
            noise_sigma[band_index]
        )

        initial_flux[:, band_index] = flux

    # Construct a generic centered Gaussian morphology.
    # Use the same spatial support for every source.
    size = box_size

    y, x = np.indices((size, size), dtype=np.float64)

    center = (size - 1) / 2

    morphology = np.exp(
        -0.5 * (
            (x - center) ** 2 +
            (y - center) ** 2
        ) / TEMPLATE_SIGMA ** 2
    )

    # Keep pixel values strictly inside the unit interval
    # required by scarlet2's morphology constraint.
    morphology = np.clip(
        morphology, 1e-6, 1.0 - 1e-6
    )

    # scarlet2 represents each source using the product
    # of its spectrum and spatial morphology.
    #
    # Convert integrated flux into spectrum amplitudes
    # using the sum of the initial morphology.
    spectra = np.maximum(
        initial_flux, 1e-6
    ) / morphology.sum()

    return (
        jnp.asarray(spectra),
        jnp.asarray(morphology)
    )

@lru_cache(maxsize=1)
def load_neural_model():
    """
    Load and cache the pretrained morphology network.

    The same model is reused across benchmark scenes,
    avoiding repeated model loading and download checks.
    """
    return get_prior(PRIOR_MODEL_NAME)

def fit_scarlet2(
    observed,
    psfs,
    positions,
    noise_sigma,
    bands,
    use_neural_prior=USE_NEURAL_PRIOR,
    max_iterations=MAX_ITERATIONS,
    initialization_mode=INITIALIZATION_MODE,
    collect_diagnostics=False,
    box_size=INITIAL_BOX_SIZE
):
    """
    Fit a multi-band galaxy scene using scarlet2.

    Parameters:
        observed:
            Observed image cube (band, height, width).

        psfs:
            Normalized empirical PSF for each band.

        positions:
            Known source coordinates in (x, y) order.

        noise_sigma:
            Gaussian noise standard deviation per band.

        bands:
            Ordered list of photometric filter names.
        initialization_mode:
            "warm_start" (unchanged default), "balanced_warm_start" (10%
            equal-share mixing), or the existing "moments" initialization.
        collect_diagnostics:
            Record convergence and flux/morphology summaries using the official
            scarlet2 callback. Does not change the objective or update rule.
        box_size:
            Odd spatial support for each source. Default 31 preserves the
            existing baseline; larger boxes are for vanilla scarlet2 only.

    Returns:
        estimated_flux:
            Integrated model flux for each source and band.

        reconstruction:
            Model rendered into the observation frame.

        diagnostics:
            Initial and final goodness-of-fit measurements.
    """

    # Validate the observation and its uncertainty.
    if not np.all(np.isfinite(observed)):
        raise ValueError("Invalid observed image")

    if not np.all(np.isfinite(noise_sigma)):
        raise ValueError("Invalid noise uncertainty")

    if np.any(noise_sigma <= 0):
        raise ValueError("Noise must be positive")

    # Convert the simulator's (x, y) coordinates into
    # scarlet2's required (y, x) coordinate convention.
    centers = jnp.asarray(positions[:, ::-1].copy())

    # Statistical weights are inverse noise variance.
    # Broadcast one variance value across each band.
    weights = np.broadcast_to(
        1.0 / noise_sigma[:, None, None] ** 2,
        observed.shape
    ).copy()

    print("\nCreating multi-band observation...", flush=True)

    observation = sc2.Observation(
        jnp.asarray(observed),
        weights=jnp.asarray(weights),
        psf=jnp.asarray(psfs),
        channels=[str(band) for band in bands]
    )

    # Construct a common model frame with a narrower
    # reference PSF than the observed images.
    model_frame = sc2.Frame.from_observations(
        observation
    )

    # Configure the transformation from model space
    # to the wavelength-dependent observation space.
    observation.match(model_frame)

    print("\nInitializing galaxy sources...", flush=True)

        # Prepare observation-based initial flux estimates
    # when the warm-start strategy is selected.
    if initialization_mode not in ("warm_start", "balanced_warm_start", "moments"):
        raise ValueError("Unsupported initialization mode")
    if not isinstance(box_size, (int, np.integer)) or box_size < 3 or box_size % 2 != 1:
        raise ValueError("box_size must be an odd integer >= 3")
    if use_neural_prior and box_size != INITIAL_BOX_SIZE:
        raise ValueError("Larger boxes are not validated for the HSC32 prior")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    warm_start_flux = None
    if initialization_mode in ("warm_start", "balanced_warm_start"):

        initial_spectra, initial_morphology = (
            initialize_from_baseline(
                observed,
                psfs,
                positions,
                noise_sigma,
                box_size=box_size
            )
        )

        if collect_diagnostics:
            warm_start_flux = np.asarray(initial_spectra) * float(initial_morphology.sum())
        if initialization_mode == "balanced_warm_start":
            initial_spectra = jnp.asarray(balance_initial_spectra(initial_spectra))

    print(f"Initialization mode: {initialization_mode} | Support: {box_size}x{box_size}")

    # Initialize each galaxy independently.
    with sc2.Scene(model_frame) as scene:

        for index, center in enumerate(centers):

            if initialization_mode in ("warm_start", "balanced_warm_start"):

                spectrum = initial_spectra[index]
                morphology = initial_morphology.copy()

            elif initialization_mode == "moments":

                spectrum, morphology = (
                    sc2.init.from_gaussian_moments(
                        observation,
                        center,
                        box_sizes=[box_size]
                    )
                )

            else:
                raise ValueError(
                    "Unsupported initialization mode"
                )

            # Create a source with fixed position and
            # optimizable spectrum and morphology.
            sc2.Source(
                center,
                spectrum,
                morphology
            )

            print(f"\nSource {index + 1} initialization:")
            print(f"Morphology shape: {morphology.shape}")
            print(f"Spectrum: {np.asarray(spectrum)}")

    # Measure initial integrated flux before optimization.
    # This helps determine whether optimization improves
    # or degrades the observation-based initialization.
    print("\n--- Initial Integrated Flux ---")

    for index, source in enumerate(scene.sources):

        print(
            f"Source {index + 1}: "
            f"{np.asarray(sc2.measure.flux(source))}"
        )
    # Small bounding boxes may restrict flux recovery.
    for index, source in enumerate(scene.sources):

        print(
            f"Source {index + 1} bounding box: "
            f"{source.bbox}"
        )

    initial_flux = None
    if collect_diagnostics:
        initial_flux = np.stack([
            np.asarray(sc2.measure.flux(source)) for source in scene.sources
        ])

    # Record the goodness of fit before optimization.
    initial_chi2 = float(
        observation.goodness_of_fit(scene())
    )

    print(f"Initial mean chi-square: {initial_chi2:.6f}")

    # Use relative optimization step sizes based on
    # the official scarlet2 example configuration.
    spectrum_step = partial(
        sc2.relative_step, factor=0.05
    )

    morphology_step = partial(
        sc2.relative_step, factor=0.001
    )
    
        # Load the pretrained morphology prior only when enabled.
    # Reuse the downloaded model from the local HF cache.
    morphology_prior = None

    # Load the neural prior only when requested.
    if use_neural_prior:

        print(
            "\nLoading pretrained galaxy morphology prior...",
            flush=True
        )

        # Reuse the cached pretrained network.
        neural_model = load_neural_model()

        prior_shape = neural_model.shape()

        # Verify that every source morphology fits within
        # the supported neural network input dimensions.
        for index, source in enumerate(scene.sources):

            shape = source.morphology.shape

            if any(
                actual > maximum
                for actual, maximum in zip(shape, prior_shape)
            ):
                raise ValueError(
                    f"Source {index + 1}: "
                    f"Morphology {shape} exceeds "
                    f"prior input shape {prior_shape}"
                )

        # ScorePrior automatically pads smaller morphology
        # arrays and supplies neural prior gradients.
        morphology_prior = ScorePrior(
            neural_model,
            prior_shape,
            t=PRIOR_TEMPERATURE
        )

        print(f"Prior model: {PRIOR_MODEL_NAME}")
        print(f"Prior shape: {prior_shape}")
        print(f"Prior temperature: {PRIOR_TEMPERATURE}")

    # Define the trainable model parameters.
    # Source positions remain fixed for consistency
    # with the previous fixed-Gaussian baseline.
    with sc2.Parameters(scene):

        for index, source in enumerate(scene.sources):

            # Enforce physically non-negative spectra.
            sc2.Parameter(
                source.spectrum,
                name=f"spectrum:{index}",
                constraint=constraints.positive,
                stepsize=spectrum_step
            )

            # Select the morphology regularization strategy.
            # The spectrum parameter remains unchanged.
            # Select neural regularization for this experiment.
            # Select neural regularization for this experiment.
            if use_neural_prior:

                # Apply the learned galaxy morphology prior.
                # Follow the official scarlet2 neural-prior
                # API instead of the unit-interval constraint.
                sc2.Parameter(
                    source.morphology,
                    name=f"morphology:{index}",
                    prior=morphology_prior,
                    stepsize=morphology_step
                )

            else:

                # Preserve the original non-parametric
                # baseline for controlled comparisons.
                sc2.Parameter(
                    source.morphology,
                    name=f"morphology:{index}",
                    constraint=constraints.unit_interval,
                    stepsize=morphology_step
                )

    print("\nStarting scarlet2 optimization...", flush=True)

    # Record the same convergence statistic used by scarlet2 0.5.1.
    trace = []

    def record_iteration(current_scene, convergence, loss):
        max_change = float(jax.tree_util.tree_reduce(lambda a, b: max(a, b), convergence))
        trace.append((float(loss), max_change))

    fit_options = {"callback": record_iteration} if collect_diagnostics else {}

    # Jointly optimize all sources across all filters.
    fitted_scene = scene.fit(
        observation,
        # Use the requested optimization budget.
        max_iter=max_iterations,
        e_rel=RELATIVE_TOLERANCE,
        **fit_options
    )
    
    print("\n--- Fitted Morphology Diagnostics ---")
    morphology_stats = []

    for index, source in enumerate(fitted_scene.sources):

        morphology = np.asarray(source.morphology)

        # Reject undefined morphology values.
        if not np.all(np.isfinite(morphology)):
            raise RuntimeError(
                f"Source {index + 1}: Invalid morphology"
            )

        # Measure negative pixels introduced during fitting.
        negative_count = np.count_nonzero(morphology < 0)

        # Locate the brightest morphology pixel.
        peak_y, peak_x = np.unravel_index(
            np.argmax(morphology),
            morphology.shape
        )

        # Compute the geometric center of the source box.
        center_y = (morphology.shape[0] - 1) / 2
        center_x = (morphology.shape[1] - 1) / 2

        # Measure peak displacement from the box center.
        # This is a diagnostic, not a physical centroid.
        peak_offset = np.hypot(
            peak_x - center_x,
            peak_y - center_y
        )

        print(f"\nSource {index + 1}")
        print(f"Morphology shape: {morphology.shape}")
        print(f"Minimum: {morphology.min():.8f}")
        print(f"Maximum: {morphology.max():.8f}")
        print(f"Negative pixels: {negative_count}")
        print(f"Peak offset: {peak_offset:.4f} pixels")

        if collect_diagnostics:
            # Raw morphology coordinates are relative to the box center.
            # Use positive weights to avoid cancellation by negative pixels.
            positive = np.maximum(morphology.astype(float), 0)
            total_abs = np.abs(morphology.astype(float)).sum()
            negative_mass = np.maximum(-morphology.astype(float), 0).sum()
            yy, xx = np.indices(morphology.shape)
            centroid_offset = None
            if positive.sum() > 0:
                cy = (yy * positive).sum() / positive.sum()
                cx = (xx * positive).sum() / positive.sum()
                centroid_offset = float(np.hypot(cy - center_y, cx - center_x))
            # Record the actual model-frame bounding box and edge mass.
            # Edge mass uses raw positive morphology, not intrinsic truth.
            edge = np.ones(morphology.shape, dtype=bool)
            edge[2:-2, 2:-2] = False
            bbox = source.bbox
            morphology_stats.append({
                "bbox_height": int(bbox.shape[-2]),
                "bbox_width": int(bbox.shape[-1]),
                "bbox_y": int(bbox.origin[-2]),
                "bbox_x": int(bbox.origin[-1]),
                "edge_positive_mass_fraction": (float(positive[edge].sum() / positive.sum())
                                                if positive.sum() > 0 else None),
                "peak_offset_px": float(peak_offset),
                "positive_centroid_offset_px": centroid_offset,
                "negative_mass_fraction": float(negative_mass / total_abs) if total_abs > 0 else None,
            })

    # Measure integrated source flux in the model frame.
    # Do not use the spectrum amplitude alone because
    # the morphology may not have a unit integrated sum.
    estimated_flux = np.stack([
        np.asarray(sc2.measure.flux(source))
        for source in fitted_scene.sources
    ])

    # Render the complete fitted model into the
    # observation frame for residual analysis.
    reconstruction = np.asarray(
        observation.render(fitted_scene())
    )

    final_chi2 = float(
        observation.goodness_of_fit(fitted_scene())
    )

    # Reject numerically invalid reconstruction results.
    if not np.all(np.isfinite(estimated_flux)):
        raise RuntimeError("Non-finite recovered flux")

    if not np.all(np.isfinite(reconstruction)):
        raise RuntimeError("Non-finite reconstruction")

    diagnostics = {
        "initial_chi2": initial_chi2,
        "final_chi2": final_chi2
    }

    if collect_diagnostics:
        if not trace or not np.all(np.isfinite(trace)):
            raise RuntimeError("Missing or nonfinite convergence trace")
        if not np.all(np.isfinite(initial_flux)):
            raise RuntimeError("Nonfinite initial model flux")
        last_change = trace[-1][1]
        if last_change < RELATIVE_TOLERANCE:
            stopping_reason = "relative_tolerance"
        elif len(trace) == max_iterations:
            stopping_reason = "iteration_limit"
        else:
            stopping_reason = "unexpected_early_stop"
        diagnostics.update({
            "warm_start_flux": warm_start_flux,
            "initial_flux": initial_flux,
            "iterations": len(trace),
            "stopping_reason": stopping_reason,
            "first_loss": trace[0][0],
            "last_loss": trace[-1][0],
            "last_max_change": last_change,
            "morphology_stats": morphology_stats,
            "trace": trace,
        })

    return estimated_flux, reconstruction, diagnostics


def main():
    """
    Execute the scarlet2 pilot experiment and compare
    the recovered flux against simulation ground truth.
    """

    start_time = perf_counter()

    print("Official scarlet2 Pilot Benchmark")
    print(f"scarlet2 version: {version('scarlet2')}")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")

    with np.load(DATASET_PATH, allow_pickle=False) as data:

        # Load only the information required for fitting.
        observed = data["observed"]
        psfs = data["psfs"]
        positions = data["positions"]
        noise_sigma = data["noise_sigma"]
        bands = data["bands"]

        estimated_flux, reconstruction, diagnostics = (
            fit_scarlet2(
                observed,
                psfs,
                positions,
                noise_sigma,
                bands
            )
        )

        print("\n--- Reconstruction Diagnostics ---")

        print(
            "Initial mean chi-square: "
            f"{diagnostics['initial_chi2']:.6f}"
        )

        print(
            "Final mean chi-square: "
            f"{diagnostics['final_chi2']:.6f}"
        )

        # Evaluate reconstruction residuals per band.
        for index, band in enumerate(bands):

            residual = (
                observed[index]
                - reconstruction[index]
            )

            rmse = np.sqrt(
                np.mean(residual ** 2)
            )

            print(
                f"{band.upper()} residual RMSE: "
                f"{rmse:.8f}"
            )

        # Access ground truth only after optimization.
        true_flux = data["input_flux"]

        evaluate_flux(
            estimated_flux,
            true_flux,
            bands
        )

    print("\n--- Experiment Accounting ---")
    print("Scenes processed: 1")
    print(f"Sources processed: {len(positions)}")
    print(f"Flux measurements: {estimated_flux.size}")
    print(
        f"Total execution time: "
        f"{perf_counter() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()