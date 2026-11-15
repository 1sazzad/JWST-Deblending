"""
Module: benchmark_scarlet2.py

Purpose:
    Compare fixed Gaussian fitting, unregularized scarlet2,
    and neural-prior scarlet2 on identical simulated scenes.

Experimental design:
    Select one repetition from every separation and flux
    ratio combination in the existing development benchmark.

    5 separations x 4 flux ratios = 20 scenes.

Ground truth is used only for evaluation.
Failed measurements are retained in the output CSV.
Fresh runs require a new output path or explicit --fresh.
Resume validates existing records and skips complete experiments.
Each experiment is committed with an atomic CSV replacement.

Morphology extension:
    Gaussian retains the original CSV format and output path.
    Sersic n=1, 2 and 4 use separate CSV files with morphology,
    visible-flux and intrinsic-support diagnostics. The n=4
    experiment is explicitly labeled as a stress test.
"""

import argparse
import csv
import os
import tempfile

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from src.simulation.simulate_blends import simulate_scene

from src.baselines.baseline_deblend import fit_band

from src.baselines.baseline_scarlet2 import fit_scarlet2

from src.evaluation.benchmark_baseline import (
    create_configuration,
    SEPARATIONS,
    FLUX_RATIOS,
    REPETITIONS,
    BASE_SEED
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT / "results/benchmark_scarlet2.csv"
)

# Separate files prevent one morphology run from overwriting another.
MORPHOLOGY_CONFIG = {
    "gaussian": (None, "main", None),
    "sersic1": (1.0, "main", 8),
    "sersic2": (2.0, "main", 8),
    "sersic4": (4.0, "stress", 32),
}
SUPPORT_SIZES = (31, 49, 65)

MAX_ITERATIONS = 1000

METHODS = (
    "fixed_gaussian",
    "scarlet2",
    "scarlet2_neural"
)

FIELDS = [
    "scene_id",
    "seed",
    "method",
    "separation_px",
    "flux_ratio_f277w",
    "source",
    "band",
    "true_flux",
    "estimated_flux",
    "absolute_error",
    "signed_error",
    "chi2",
    "runtime_seconds",
    "status",
    "failure_reason"
]


# Keep the original Gaussian checkpoint schema unchanged.
SERSIC_FIELDS = FIELDS + [
    "morphology", "sersic_n", "track", "sampling_resolution",
    "visible_flux", "support_31_fraction", "support_49_fraction",
    "support_65_fraction"
]


def output_path_for(morphology):
    """Return an isolated checkpoint path for each morphology family."""
    if morphology == "gaussian":
        return OUTPUT_PATH
    return OUTPUT_PATH.with_name(f"benchmark_scarlet2_{morphology}.csv")


def fields_for(morphology):
    """Preserve compatibility with the previously generated Gaussian CSV."""
    return FIELDS if morphology == "gaussian" else SERSIC_FIELDS


def scene_diagnostics(scene, morphology):
    """
    Compute evaluation-only intrinsic support and visible-flux diagnostics.

    Reuse the visualization module's illustrative centered-box definition.
    Fractions use input flux on the finite simulation canvas as denominator.
    The boxes are not exact scarlet2 internal bounding boxes.
    These ground-truth diagnostics must never be passed to the fitters.
    """
    if morphology == "gaussian":
        return {}

    from src.visualization.visualize_morphologies import (
        calculate_support_fraction
    )

    n, track, sampling = MORPHOLOGY_CONFIG[morphology]
    # Older local simulator revisions may omit optional metadata.
    # When present, verify it rather than silently accepting a mismatch.
    if ("morphology_family" in scene and scene["morphology_family"] != "sersic"
        or "sersic_n" in scene and float(scene["sersic_n"]) != n):
        raise ValueError("Simulator morphology metadata does not match request")

    diagnostics = {}
    for source, position in enumerate(scene["positions"]):
        for band_index, band in enumerate(scene["bands"]):
            truth = float(scene["input_flux"][source, band_index])
            visible = float(scene["visible_flux"][source, band_index])
            if not np.isfinite(truth) or truth <= 0:
                raise ValueError("Invalid diagnostic input flux")
            if not np.isfinite(visible) or not 0 < visible <= truth * (1 + 1e-6):
                raise ValueError("Invalid visible flux")
            fractions = [
                calculate_support_fraction(
                    scene["intrinsic"][source, band_index], position,
                    size, truth
                )
                for size in SUPPORT_SIZES
            ]
            if (
                not all(np.isfinite(v) and 0 <= v <= 1 + 1e-6 for v in fractions)
                or any(a > b + 1e-8 for a, b in zip(fractions, fractions[1:]))
            ):
                raise ValueError("Invalid intrinsic support fractions")

            diagnostics[(source + 1, str(band).upper())] = {
                "morphology": morphology,
                "sersic_n": n,
                "track": track,
                "sampling_resolution": sampling,
                "visible_flux": visible,
                **{f"support_{size}_fraction": float(value)
                   for size, value in zip(SUPPORT_SIZES, fractions)}
            }

    return diagnostics


def fit_method(scene, method):
    """
    Run one baseline using observation-side information.

    All methods receive the same observed image cube,
    PSF kernels, source positions and noise estimates.

    Returns:
        Estimated flux and reconstructed observation.
    """

    observed = scene["observed"]
    psfs = scene["psfs"]
    positions = scene["positions"]
    noise_sigma = scene["noise_sigma"]
    bands = scene["bands"]

    if method == "fixed_gaussian":

        estimated_flux = np.zeros(
            (len(positions), len(bands))
        )

        reconstruction = np.zeros_like(observed)

        # Fit each band independently.
        for b in range(len(bands)):

            flux, model, _, _ = fit_band(
                observed[b],
                psfs[b],
                positions,
                noise_sigma[b]
            )

            estimated_flux[:, b] = flux
            reconstruction[b] = model

        return estimated_flux, reconstruction

    if method in ("scarlet2", "scarlet2_neural"):

        # Both scarlet2 variants use the same iteration
        # budget and observation-based initialization.
        flux, reconstruction, _ = fit_scarlet2(
            observed,
            psfs,
            positions,
            noise_sigma,
            bands,
            use_neural_prior=(
                method == "scarlet2_neural"
            ),
            max_iterations=MAX_ITERATIONS
        )

        return flux, reconstruction

    raise ValueError(f"Unknown method: {method}")


def evaluate_method(
    scene, scene_id, seed, separation, ratio, method
):
    """
    Evaluate one method and retain individual source-band
    measurements, reconstruction error and runtime.

    Ground-truth flux is never passed to the fitter.
    """

    start = perf_counter()

    try:

        estimated, reconstruction = fit_method(
            scene, method
        )

        # Reject invalid model outputs.
        if not np.all(np.isfinite(estimated)):
            raise ValueError("Non-finite estimated flux")

        if not np.all(np.isfinite(reconstruction)):
            raise ValueError("Non-finite reconstruction")

        truth = scene["input_flux"]

        if estimated.shape != truth.shape:
            raise ValueError("Unexpected flux dimensions")

        # Prevent accidental NumPy broadcasting from hiding shape errors.
        if reconstruction.shape != scene["observed"].shape:
            raise ValueError("Unexpected reconstruction dimensions")

        if not np.all(np.isfinite(truth)) or np.any(truth <= 0):
            raise ValueError("Ground-truth flux must be finite and positive")

        # Apply the same goodness-of-fit calculation
        # to all methods for consistent comparison.
        residual = (
            scene["observed"] - reconstruction
        )

        chi2 = np.mean(
            (
                residual
                / scene["noise_sigma"][:, None, None]
            ) ** 2
        )

        relative_error = (
            estimated - truth
        ) / truth

        # A completed numerical fit is not necessarily a converged fit.
        if not np.isfinite(chi2) or not np.all(np.isfinite(relative_error)):
            raise ValueError("Non-finite evaluation metrics")

        status = "success"
        failure_reason = ""

    except Exception as error:

        # Preserve failed scenes in benchmark accounting.
        status = "failed"
        failure_reason = f"{type(error).__name__}: {error}"

        print(
            f"Method failed: {failure_reason}",
            flush=True
        )

    elapsed = perf_counter() - start

    rows = []

    # Store six measurements per method and scene.
    for s in range(len(scene["positions"])):

        for b, band in enumerate(scene["bands"]):

            success = status == "success"

            rows.append({
                "scene_id": scene_id,
                "seed": seed,
                "method": method,
                "separation_px": separation,
                "flux_ratio_f277w": ratio,
                "source": s + 1,
                "band": str(band),
                "true_flux": scene["input_flux"][s, b],
                "estimated_flux": (
                    estimated[s, b] if success else ""
                ),
                "absolute_error": (
                    abs(relative_error[s, b])
                    if success else ""
                ),
                "signed_error": (
                    relative_error[s, b]
                    if success else ""
                ),
                "chi2": chi2 if success else "",
                "runtime_seconds": elapsed,
                "status": status,
                "failure_reason": failure_reason
            })

    if status == "success":

        print(
            f"{method}: "
            f"Flux error = "
            f"{np.mean(np.abs(relative_error))*100:.2f}% | "
            f"Chi-square = {chi2:.4f} | "
            f"Runtime = {elapsed:.2f}s",
            flush=True
        )

    return rows


def summarize_results(rows):
    """
    Report performance and failure accounting separately
    for each baseline method.
    """

    print("\n--- Comparative Benchmark Summary ---")

    for method in METHODS:

        group = [
            row for row in rows
            if row["method"] == method
        ]

        successful = [
            row for row in group
            if row["status"] == "success"
        ]

        print(f"\nMethod: {method}")

        print(
            f"Successful measurements: "
            f"{len(successful)}/{len(group)}"
        )

        if not successful:
            continue

        errors = np.array([
            row["absolute_error"]
            for row in successful
        ])

        print(
            f"Mean flux error: "
            f"{errors.mean()*100:.4f}%"
        )

        print(
            f"Median flux error: "
            f"{np.median(errors)*100:.4f}%"
        )

        print(
            f"90th percentile error: "
            f"{np.percentile(errors, 90)*100:.4f}%"
        )

        # Report the faintest neighboring source group.
        faint_errors = [
            row["absolute_error"]
            for row in successful
            if row["source"] == 2
            and row["flux_ratio_f277w"] == 0.1
        ]

        if faint_errors:

            print(
                f"Faint-neighbor error: "
                f"{np.mean(faint_errors)*100:.4f}%"
            )

# ---------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------

# The development simulator generates exactly two sources
# and three bands for each scene.
EXPECTED_MEASUREMENTS = {
    (source, band)
    for source in (1, 2)
    for band in ("F150W", "F277W", "F444W")
}


def save_results(rows, path, fields=FIELDS):
    """
    Commit the complete CSV atomically.

    Write into a temporary file in the same directory, flush
    it to disk, then replace the destination. An interruption
    during writing leaves the previous committed CSV intact.

    Only the main process should write this result file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())

        # Both files are on the same filesystem.
        os.replace(temporary_path, path)

    finally:
        # Remove temporary output on handled failures.
        # A hard process termination may leave an orphan .tmp file.
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_existing_results(path, selected, morphology="gaussian"):
    """
    Restore a checkpoint and validate its measurement identities.

    Only complete experiments matching the selected development
    scene IDs, seeds, separations, flux ratios and methods are
    accepted. Failed fits remain recorded, not silently discarded.

    The CSV does not encode a software/configuration fingerprint;
    resume only with the same simulator and fitting configuration.
    """
    expected_scenes = {
        scene_id: (separation, ratio)
        for scene_id, separation, ratio in selected
    }

    rows = []
    grouped = {}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != fields_for(morphology):
            raise ValueError("Unexpected benchmark CSV structure")

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row")

            for field in ("scene_id", "seed", "source"):
                row[field] = int(row[field])

            for field in (
                "separation_px", "flux_ratio_f277w", "true_flux",
                "estimated_flux", "absolute_error", "signed_error",
                "chi2", "runtime_seconds"
            ):
                if row[field] != "":
                    row[field] = float(row[field])
                    if not np.isfinite(row[field]):
                        raise ValueError(f"Non-finite saved field: {field}")

            scene_id = row["scene_id"]
            method = row["method"]
            key = (scene_id, method)

            if scene_id not in expected_scenes or method not in METHODS:
                raise ValueError(f"Unexpected saved experiment: {key}")

            separation, ratio = expected_scenes[scene_id]
            if (
                row["seed"] != BASE_SEED + scene_id
                or row["separation_px"] != separation
                or row["flux_ratio_f277w"] != ratio
            ):
                raise ValueError(f"Saved scene configuration mismatch: {key}")

            if (row["source"], row["band"].upper()) not in EXPECTED_MEASUREMENTS:
                raise ValueError(f"Unexpected measurement identity: {key}")

            if row["true_flux"] == "" or row["true_flux"] <= 0:
                raise ValueError(f"Invalid saved ground truth: {key}")

            if row["runtime_seconds"] == "" or row["runtime_seconds"] < 0:
                raise ValueError(f"Invalid saved runtime: {key}")

            metric_fields = (
                "estimated_flux", "absolute_error", "signed_error", "chi2"
            )
            if row["status"] == "success":
                if any(row[field] == "" for field in metric_fields):
                    raise ValueError(f"Missing successful metrics: {key}")
                if row["absolute_error"] < 0 or row["chi2"] < 0:
                    raise ValueError(f"Invalid saved metrics: {key}")
                expected_error = (
                    row["estimated_flux"] - row["true_flux"]
                ) / row["true_flux"]
                if not (
                    np.isclose(row["signed_error"], expected_error, rtol=1e-5, atol=1e-7)
                    and np.isclose(row["absolute_error"], abs(expected_error), rtol=1e-5, atol=1e-7)
                ):
                    raise ValueError(f"Inconsistent saved flux metrics: {key}")
            elif row["status"] == "failed":
                if any(row[field] != "" for field in metric_fields):
                    raise ValueError(f"Unexpected failed-fit metrics: {key}")
            else:
                raise ValueError(f"Unknown saved status: {key}")

            if morphology != "gaussian":
                n, track, sampling = MORPHOLOGY_CONFIG[morphology]
                if (
                    row["morphology"] != morphology
                    or float(row["sersic_n"]) != n
                    or row["track"] != track
                    or int(row["sampling_resolution"]) != sampling
                ):
                    raise ValueError(f"Saved morphology configuration mismatch: {key}")
                row["sersic_n"] = n
                row["sampling_resolution"] = sampling
                row["visible_flux"] = float(row["visible_flux"])
                if (not np.isfinite(row["visible_flux"])
                    or not 0 < row["visible_flux"] <= row["true_flux"] * (1 + 1e-6)):
                    raise ValueError(f"Invalid saved visible flux: {key}")
                fractions = []
                for size in SUPPORT_SIZES:
                    field = f"support_{size}_fraction"
                    row[field] = float(row[field])
                    fractions.append(row[field])
                if (
                    not all(np.isfinite(v) and 0 <= v <= 1 + 1e-6 for v in fractions)
                    or any(a > b + 1e-8 for a, b in zip(fractions, fractions[1:]))
                ):
                    raise ValueError(f"Invalid saved support fractions: {key}")

            grouped.setdefault(key, []).append(row)
            rows.append(row)

    for key, group in grouped.items():
        identities = {(row["source"], row["band"].upper()) for row in group}
        statuses = {row["status"] for row in group}
        if (
            len(group) != len(EXPECTED_MEASUREMENTS)
            or identities != EXPECTED_MEASUREMENTS
            or len(statuses) != 1
        ):
            raise ValueError(f"Incomplete or duplicate experiment: {key}")

    return rows, set(grouped)


def main():
    """
    Run identical synthetic scenes through all three methods.

    Modes:
        --smoke: Run one scene without writing a CSV.
        --fresh: Start the full benchmark, explicitly replacing the CSV.
        --resume: Continue the full benchmark from a saved CSV.
        no flag: Start only if the output CSV does not already exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--smoke", action="store_true",
        help="Test all three methods on one scene; do not save results."
    )
    modes.add_argument(
        "--fresh", action="store_true",
        help="Start the full benchmark and replace existing results."
    )
    modes.add_argument(
        "--resume", action="store_true",
        help="Continue the full benchmark without repeating saved experiments."
    )

    parser.add_argument(
        "--morphology", choices=tuple(MORPHOLOGY_CONFIG), default="gaussian",
        help="Choose the matched Gaussian or Sersic morphology family."
    )

    # Register all arguments before parsing.
    args = parser.parse_args()
    morphology = args.morphology
    sersic_n, track, sampling = MORPHOLOGY_CONFIG[morphology]
    output_path = output_path_for(morphology)
    fields = fields_for(morphology)

    # Keep the original scene IDs and seeds of the 100-scene benchmark.
    combinations = list(product(
        SEPARATIONS, FLUX_RATIOS, range(REPETITIONS)
    ))
    selected = [
        (index, separation, ratio)
        for index, (separation, ratio, repeat)
        in enumerate(combinations, start=1)
        if repeat == 0
    ]

    if args.smoke:
        selected = selected[:1]

    start = perf_counter()
    all_rows = []
    completed = set()
    planned = len(selected) * len(METHODS)
    planned_measurements = planned * len(EXPECTED_MEASUREMENTS)

    print(f"Morphology: {morphology} | Track: {track}", flush=True)
    if sersic_n is not None:
        print(f"Sersic n: {sersic_n} | Expected sampling: {sampling}x{sampling}", flush=True)
    if not args.smoke:
        print(f"Output: {output_path}", flush=True)
    print(f"Planned scenes: {len(selected)}", flush=True)
    print(f"Planned method-scene experiments: {planned}", flush=True)
    print(f"Planned flux measurements: {planned_measurements}", flush=True)
    print(f"scarlet2 maximum iterations: {MAX_ITERATIONS}", flush=True)

    if not args.smoke:
        if args.resume:
            if not output_path.exists():
                parser.error(f"No checkpoint found: {output_path}")

            all_rows, completed = load_existing_results(output_path, selected, morphology)
            print(f"Restored measurements: {len(all_rows)}", flush=True)
            print(f"Completed experiments: {len(completed)}", flush=True)
        else:
            if output_path.exists() and not args.fresh:
                parser.error("CSV already exists. Use --resume or explicitly --fresh.")

            # A fresh run deliberately starts with a header-only CSV.
            save_results([], output_path, fields)

    try:
        for count, (scene_id, separation, ratio) in enumerate(
            selected, start=1
        ):
            print(
                f"\nScene {count}/{len(selected)} | Original ID: {scene_id} | "
                f"Separation: {separation} px | Ratio: {ratio}",
                flush=True
            )

            pending = [
                method for method in METHODS
                if (scene_id, method) not in completed
            ]
            for method in METHODS:
                if method not in pending:
                    print(f"Skipping completed: Scene {scene_id} | {method}", flush=True)

            if not pending:
                continue

            seed = BASE_SEED + scene_id

            # Generate each scene once; all pending methods share its input.
            config = create_configuration(separation, ratio, seed)
            # Preserve the original call for Gaussian runs. Only the
            # intrinsic morphology family changes in the Sersic runs.
            if sersic_n is None:
                scene = simulate_scene(**config)
            else:
                scene = simulate_scene(**config, sersic_n=sersic_n)

            # Validate the simulator's source/band layout before running fits.
            identities = {
                (source + 1, str(band).upper())
                for source in range(len(scene["positions"]))
                for band in scene["bands"]
            }
            if identities != EXPECTED_MEASUREMENTS:
                raise ValueError("Unexpected simulator source/band layout")

            diagnostics = scene_diagnostics(scene, morphology)
            if diagnostics:
                worst_support = min(
                    entry["support_31_fraction"] for entry in diagnostics.values()
                )
                print(f"Minimum intrinsic 31x31 support: {worst_support*100:.4f}%", flush=True)

            for method in pending:
                print(
                    f"Running experiment {len(completed)+1}/{planned}: {method}",
                    flush=True
                )
                rows = evaluate_method(
                    scene, scene_id, seed, separation, ratio, method
                )

                # Store identical truth-side diagnostics for all methods.
                # They are computed independently of each fitted result.
                if diagnostics:
                    for row in rows:
                        row.update(diagnostics[(row["source"], row["band"].upper())])

                # Commit the new complete experiment before marking it done.
                updated_rows = all_rows + rows
                if not args.smoke:
                    save_results(updated_rows, output_path, fields)

                all_rows = updated_rows
                completed.add((scene_id, method))
                print(
                    f"Progress: {len(completed)}/{planned} experiments | "
                    f"{len(all_rows)}/{planned_measurements} measurements",
                    flush=True
                )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", flush=True)
        if not args.smoke:
            print(f"Committed results are saved in: {output_path}", flush=True)
            print(f"Use --morphology {morphology} --resume to continue.", flush=True)
        # Do not print the normal completion summary for an interrupted run.
        raise SystemExit(130)

    if len(completed) != planned or len(all_rows) != planned_measurements:
        raise RuntimeError("Final benchmark accounting mismatch")

    summarize_results(all_rows)
    if morphology != "gaussian":
        # Each source-band appears once per method; avoid triple counting.
        unique = {(row["scene_id"], row["source"], row["band"].upper()): row
                  for row in all_rows}
        print("\n--- Intrinsic Support Summary (all recorded scenes) ---")
        for size in SUPPORT_SIZES:
            excluded = np.array([
                1 - row[f"support_{size}_fraction"] for row in unique.values()
            ]) * 100
            print(f"{size}x{size}: mean excluded={excluded.mean():.6f}% | "
                  f"max excluded={excluded.max():.6f}%")
    successful = sum(row["status"] == "success" for row in all_rows)
    print(f"\nCompleted experiments: {len(completed)}/{planned}")
    print(f"Recorded measurements: {len(all_rows)}/{planned_measurements}")
    print(f"Successful measurements: {successful}/{planned_measurements}")
    print(f"Failed measurements: {len(all_rows) - successful}")
    print(f"Current invocation runtime: {perf_counter()-start:.2f}s")

    if not args.smoke:
        print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
