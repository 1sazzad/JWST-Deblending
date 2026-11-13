"""
Module: benchmark_baseline.py

Purpose:
    Evaluate the fixed-morphology deblending baseline
    across a controlled grid of synthetic galaxy scenes.

Experimental design:
    5 separations x 4 flux ratios x 5 repetitions
    = 100 synthetic scenes.

Output:
    A CSV containing individual flux measurements,
    relative errors and experimental metadata.

This is a development benchmark, not a held-out test.
"""

import csv
from pathlib import Path
from itertools import product
from time import perf_counter

import numpy as np

# Reuse the synthetic scene generator for reproducible experiments.
from src.simulation.simulate_blends import simulate_scene

# Import the fixed-morphology baseline for flux recovery.
from src.baselines.baseline_deblend import fit_band


# ---------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/benchmark_baseline.csv"
)

SEPARATIONS = [2, 4, 6, 8, 12]
FLUX_RATIOS = [0.1, 0.25, 0.5, 1.0]
REPETITIONS = 5
BASE_SEED = 10000

BANDS = ("f150w", "f277w", "f444w")


def create_configuration(separation, ratio, seed):
    """
    Generate reproducible galaxy properties for one scene.

    The requested flux ratio is defined in F277W.
    Source colors vary between bands.
    """

    rng = np.random.default_rng(seed)

    # Randomize the orientation of the source pair
    # to avoid restricting all blends to one direction.
    angle = rng.uniform(0, 2 * np.pi)

    offset = separation / 2 * np.array([
        np.cos(angle),
        np.sin(angle)
    ])

    center = np.array([63.5, 63.5])

    positions = np.stack([
        center - offset,
        center + offset
    ])

    # Define the brighter reference galaxy.
    flux_a = np.array([100.0, 80.0, 60.0])

    # Give the second galaxy a different spectral shape.
    # Its F277W flux ratio remains exactly the requested ratio.
    flux_b = ratio * flux_a * np.array([
        0.8, 1.0, 1.5
    ])

    true_flux = np.stack([flux_a, flux_b])

    # Randomize galaxy dimensions and orientations.
    morphology = np.column_stack([
        rng.uniform(1.5, 4.0, size=2),
        rng.uniform(1.2, 3.0, size=2),
        rng.uniform(-90, 90, size=2)
    ])

    # Vary the noise amplitude across scenes.
    noise_scale = rng.uniform(0.5, 2.0)

    noise_sigma = (
        np.array([0.02, 0.03, 0.04])
        * noise_scale
    )

    return {
        "positions": positions,
        "true_flux": true_flux,
        "noise_sigma": noise_sigma,
        "morphology": morphology,
        "seed": seed + 100000
    }


def evaluate_scene(scene_id, separation, ratio, seed):
    """
    Simulate one scene and recover flux independently
    in each band using the existing baseline.

    Ground truth is used only after flux fitting.
    """

    config = create_configuration(
        separation, ratio, seed
    )

    scene = simulate_scene(**config)

    estimated_flux = np.zeros((2, len(BANDS)))

    # Fit using only observations, PSFs, source
    # positions and noise information.
    for b in range(len(BANDS)):

        flux, _, _, _ = fit_band(
            scene["observed"][b],
            scene["psfs"][b],
            scene["positions"],
            scene["noise_sigma"][b]
        )

        estimated_flux[:, b] = flux

    if not np.all(np.isfinite(estimated_flux)):
        raise ValueError("Non-finite recovered flux")

    # Ground truth is accessed for evaluation only.
    true_flux = scene["input_flux"]

    relative_error = (
        estimated_flux - true_flux
    ) / true_flux

    rows = []

    # Store each source-band measurement separately.
    for s in range(2):
        for b, band in enumerate(BANDS):

            rows.append({
                "scene_id": scene_id,
                "seed": seed,
                "separation_px": separation,
                "flux_ratio_f277w": ratio,
                "source": s + 1,
                "band": band,
                "true_flux": true_flux[s, b],
                "estimated_flux": estimated_flux[s, b],
                "signed_error": relative_error[s, b],
                "absolute_error": abs(relative_error[s, b]),
                "status": "success",
                "failure_reason": ""
            })

    return rows


def main():
    """
    Execute the complete development benchmark.

    Failed scenes are recorded rather than silently
    excluded from the experiment.
    """

    start = perf_counter()

    combinations = list(product(
        SEPARATIONS,
        FLUX_RATIOS,
        range(REPETITIONS)
    ))

    total = len(combinations)

    rows = []
    failed_scenes = 0

    print("Fixed-Morphology Baseline Benchmark")
    print(f"Total planned scenes: {total}")

    for index, (separation, ratio, repeat) in enumerate(
        combinations, start=1
    ):

        seed = BASE_SEED + index

        try:
            measurements = evaluate_scene(
                index, separation, ratio, seed
            )

            rows.extend(measurements)

        except Exception as error:

            failed_scenes += 1

            # Preserve the expected measurement count
            # even if reconstruction fails.
            for source in (1, 2):
                for band in BANDS:

                    rows.append({
                        "scene_id": index,
                        "seed": seed,
                        "separation_px": separation,
                        "flux_ratio_f277w": ratio,
                        "source": source,
                        "band": band,
                        "true_flux": "",
                        "estimated_flux": "",
                        "signed_error": "",
                        "absolute_error": "",
                        "status": "failed",
                        "failure_reason": str(error)
                    })

            print(f"\nScene {index} failed: {error}")

        # Report progress without printing every measurement.
        if index % 10 == 0 or index == total:

            elapsed = perf_counter() - start

            print(
                f"Processed {index}/{total} scenes | "
                f"Failed: {failed_scenes} | "
                f"Elapsed: {elapsed:.2f}s",
                flush=True
            )

    # Save all results, including failures.
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    successful = [
        row for row in rows
        if row["status"] == "success"
    ]

    print("\n--- Benchmark Accounting ---")
    print(f"Total scenes: {total}")
    print(f"Successful scenes: {total - failed_scenes}")
    print(f"Failed scenes: {failed_scenes}")
    print(f"Expected flux measurements: {total * 6}")
    print(f"Successful measurements: {len(successful)}")

    if successful:

        errors = np.array([
            row["absolute_error"]
            for row in successful
        ])

        print("\n--- Overall Metrics ---")

        print(
            f"Mean absolute flux error: "
            f"{errors.mean() * 100:.4f}%"
        )

        print(
            f"Median absolute flux error: "
            f"{np.median(errors) * 100:.4f}%"
        )

        print(
            f"90th percentile error: "
            f"{np.percentile(errors, 90) * 100:.4f}%"
        )

        # Analyze performance as blending becomes harder.
        print("\n--- Error by Source Separation ---")

        for separation in SEPARATIONS:

            group = [
                row["absolute_error"]
                for row in successful
                if row["separation_px"] == separation
            ]

            if group:
                print(
                    f"{separation:2d} px: "
                    f"{np.mean(group) * 100:.2f}%"
                )

        # Analyze performance for faint neighboring sources.
        print("\n--- Error by Flux Ratio (Source B) ---")

        for ratio in FLUX_RATIOS:

            group = [
                row["absolute_error"]
                for row in successful
                if row["flux_ratio_f277w"] == ratio
                and row["source"] == 2
            ]

            if group:
                print(
                    f"Ratio {ratio:.2f}: "
                    f"{np.mean(group) * 100:.2f}%"
                )

    print(f"\nResults saved: {OUTPUT_PATH}")
    print(f"Total runtime: {perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()