
"""Phase 10.3.2: local chromatic evidence feasibility.

Reuse 20 archived shared-morphology fits.
Fit eight new composite observations.

No nonlinear chromatic refinement is performed.
"""

import csv
from pathlib import Path
from collections import defaultdict
from time import perf_counter

import numpy as np

from scipy.linalg import block_diag
from scipy.stats import chi2

from src.baselines.baseline_flexible_gaussian import (
    profile_linear,
    MIN_SIGMA,
    MAX_SIGMA,
    MAX_RHO,
)

from src.baselines.baseline_flexible_sersic import (
    SersicObjective,
    fit_flexible_sersic,
    MIN_N,
    MAX_N,
)

from src.evaluation.benchmark_baseline import (
    create_configuration,
)

from src.evaluation.benchmark_composite_morphology import (
    create_composite_components,
    create_composite_scene,
)

from src.evaluation.benchmark_risk_refinement import (
    shape_vector,
)

from src.evaluation.benchmark_uncertainty_calibration import (
    generate_observation,
)


ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = (
    ROOT / "results"
    / "benchmark_flexible_sersic_phase8_4.csv"
)

OUTPUT = (
    ROOT / "results"
    / "benchmark_chromatic_evidence_phase10_3_2.csv"
)

BANDS = ("f150w", "f277w", "f444w")

WAVELENGTHS = np.array([1.50, 2.77, 4.44])

T = np.log(WAVELENGTHS)
T = (T - T[0]) / (T[-1] - T[0])
T = T - T.mean()

RANK_TOLERANCE = 1e-7

REFERENCE_THRESHOLD = float(
    chi2.ppf(0.95, 4)
)


# ---------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------

def center(matrix):
    return matrix - matrix.mean(
        axis=0, keepdims=True
    )


def column_basis(matrix, scales=None):
    """Stable basis with explicit rank diagnostics."""

    if scales is None:
        scales = np.linalg.norm(matrix, axis=0)

    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 1e-12)
    ):
        return None, 0

    U, singular, _ = np.linalg.svd(
        matrix / scales,
        full_matrices=False,
    )

    keep = singular > (
        RANK_TOLERANCE * singular[0]
    )

    return U[:, keep], int(keep.sum())


def projected_evidence(residual, J, Z):
    """Project chromatic directions off the null model."""

    Q, null_rank = column_basis(J)

    if Q is None or null_rank != J.shape[1]:
        return np.nan, null_rank, 0, False

    Z_perp = Z - Q @ (Q.T @ Z)

    # Normalize using original column norms.
    # Do not inflate nearly vanished directions.
    P, chromatic_rank = column_basis(
        Z_perp,
        scales=np.linalg.norm(Z, axis=0),
    )

    if (
        P is None
        or chromatic_rank != Z.shape[1]
    ):
        return (
            np.nan,
            null_rank,
            chromatic_rank,
            False,
        )

    score = float(
        np.sum((P.T @ residual)**2)
    )

    return (
        score,
        null_rank,
        chromatic_rank,
        np.isfinite(score),
    )


# ---------------------------------------------------------
# PSF-aware chromatic evidence
# ---------------------------------------------------------

def chromatic_evidence(scene, theta):

    observed = scene["observed"]
    noise = scene["noise_sigma"]

    objective = SersicObjective(
        observed,
        scene["psfs"],
        scene["positions"],
        noise,
    )

    blurred = np.stack([
        objective.convolve(
            quad.evaluate(theta[s])
        )
        for s, quad in enumerate(
            objective.quad
        )
    ])

    flux = np.empty((2, 3))
    residuals = []

    X_blocks = []
    shared_blocks = []
    chromatic_blocks = []

    total_chi2 = 0.0

    for b in range(3):

        templates = blurred[:, 0, b]

        flux[:, b], _, model = (
            profile_linear(
                templates,
                observed[b],
            )
        )

        residual = (
            observed[b] - model
        ) / noise[b]

        total_chi2 += float(
            np.sum(residual**2)
        )

        residuals.append(
            residual.ravel() - residual.mean()
        )

        X = center(
            templates.reshape(2, -1).T
            / noise[b]
        )

        X_blocks.append(X)

        shared = []
        chromatic = []

        for s in range(2):

            derivatives = (
                blurred[s, 1:, b]
                .reshape(4, -1)
                .T
                * flux[s, b]
                / noise[b]
            )

            shared.append(derivatives)

            # Additional wavelength slopes:
            # radial log-width and log-Sersic index.
            chromatic.append(
                T[b]
                * (
                    derivatives[:, 0]
                    + derivatives[:, 1]
                )
            )

            chromatic.append(
                T[b] * derivatives[:, 3]
            )

        shared_blocks.append(
            center(np.column_stack(shared))
        )

        chromatic_blocks.append(
            center(np.column_stack(chromatic))
        )

    # Six flux parameters and eight shared
    # morphology parameters.
    # Constant backgrounds are profiled out.
    J = np.column_stack([
        block_diag(*X_blocks),
        np.vstack(shared_blocks),
    ])

    # Four additional chromatic slopes.
    Z = np.vstack(chromatic_blocks)

    r = np.concatenate(residuals)

    score, rank0, rank1, valid = (
        projected_evidence(r, J, Z)
    )

    # Detect flux boundary solutions.
    flux_boundary = bool(
        np.any(
            flux <= 1e-8 * max(
                float(flux.max()), 1e-12
            )
        )
    )

    # Shape boundaries invalidate a simple
    # regular asymptotic test interpretation.
    lower = np.array([
        np.log(MIN_SIGMA),
        np.log(MIN_SIGMA),
        -MAX_RHO,
        np.log(MIN_N),
    ])

    upper = np.array([
        np.log(MAX_SIGMA),
        np.log(MAX_SIGMA),
        MAX_RHO,
        np.log(MAX_N),
    ])

    shape_boundary = bool(
        np.any(theta - lower <= 1e-4)
        or np.any(upper - theta <= 1e-4)
    )

    valid = bool(valid and not flux_boundary)

    # This is a screening heuristic,
    # not a calibrated hypothesis test.
    route = bool(
        not valid
        or score > REFERENCE_THRESHOLD
    )

    return dict(
        flux=flux,
        chi2=total_chi2 / observed.size,
        evidence_score=score,
        null_rank=rank0,
        chromatic_rank=rank1,
        evidence_valid=valid,
        flux_boundary=flux_boundary,
        shape_boundary=shape_boundary,
        route_evidence=route,
    )


# ---------------------------------------------------------
# Reuse archived shared-morphology solutions
# ---------------------------------------------------------

def archived_cases():

    groups = defaultdict(list)

    with ARCHIVE.open(
        newline="", encoding="utf-8-sig"
    ) as stream:

        for row in csv.DictReader(stream):

            key = (
                row["morphology"],
                int(row["scene_id"]),
            )

            groups[key].append(row)

    if len(groups) != 20:
        raise ValueError(
            "Expected 20 archived scenes"
        )

    for (family, scene_id), rows in groups.items():

        if len(rows) != 6:
            raise ValueError(
                "Incomplete archived scene"
            )

        metadata = rows[0]

        n = (
            None if family == "gaussian"
            else float(family.replace("sersic", ""))
        )

        case = dict(
            scene_id=scene_id,
            morphology=family,
            seed=int(metadata["seed"]),
            separation=float(
                metadata["separation_px"]
            ),
            ratio=float(
                metadata["flux_ratio_f277w"]
            ),
            sersic_n=n,
            fractions=None,
        )

        theta = np.empty((2, 4))
        expected_flux = np.empty((2, 3))

        for s in range(2):

            source_rows = {
                row["band"].lower(): row
                for row in rows
                if int(row["source"]) == s + 1
            }

            if set(source_rows) != set(BANDS):
                raise ValueError(
                    "Invalid source-band inventory"
                )

            first = source_rows[BANDS[0]]

            theta[s] = [
                np.log(float(first["sigma_x"])),
                np.log(float(first["sigma_y"])),
                float(first["rho"]),
                np.log(
                    float(first["fitted_sersic_n"])
                ),
            ]

            expected_flux[s] = [
                float(
                    source_rows[b]["estimated_flux"]
                )
                for b in BANDS
            ]

        yield (
            case,
            theta,
            expected_flux,
            float(metadata["chi2"]),
        )


# ---------------------------------------------------------
# New paired composite observations
# ---------------------------------------------------------

def new_cases():

    # Four independent base configurations.
    # Each produces an achromatic/chromatic pair.
    specifications = [
        (2, 0.1, 0.04),
        (4, 0.25, 0.08),
        (8, 0.5, 0.15),
        (12, 1.0, 0.22),
    ]

    middle = np.array([0.55, 0.45])

    for i, (separation, ratio, slope) in enumerate(
        specifications, start=1
    ):

        seed = 2_110_000_000 + i

        config = create_configuration(
            separation,
            ratio,
            seed,
        )

        bulge, disk = create_composite_components(
            config
        )

        for family in ("achromatic", "chromatic"):

            if family == "achromatic":

                fractions = np.repeat(
                    middle[:, None],
                    3,
                    axis=1,
                )

            else:

                # Allow opposite color gradients
                # in the two galaxies.
                change = np.array([
                    slope, -slope
                ])

                fractions = np.column_stack([
                    middle - change,
                    middle,
                    middle + change,
                ])

            scene = create_composite_scene(
                bulge,
                disk,
                fractions,
            )

            case = dict(
                scene_id=i,
                morphology=family,
                seed=seed,
                separation=separation,
                ratio=ratio,
                gradient_strength=(
                    0.0
                    if family == "achromatic"
                    else slope
                ),
            )

            yield case, scene


# ---------------------------------------------------------
# Record results
# ---------------------------------------------------------

def make_row(
    case,
    scene,
    result,
    origin,
    fit_seconds,
    diagnostic_seconds,
    optimizer_success,
):

    # Ground truth is accessed only for evaluation.
    truth = scene["input_flux"]

    mean_error = float(
        np.mean(
            np.abs(
                result["flux"] / truth - 1
            )
        )
    )

    row = dict(
        origin=origin,
        scene_id=case["scene_id"],
        morphology=case["morphology"],
        seed=case["seed"],
        separation_px=case["separation"],
        flux_ratio_f277w=case["ratio"],
        gradient_strength=case.get(
            "gradient_strength", 0.0
        ),
        initial_error=mean_error,
        initial_chi2=result["chi2"],
        evidence_score=result["evidence_score"],
        reference_threshold=REFERENCE_THRESHOLD,
        null_rank=result["null_rank"],
        chromatic_rank=result["chromatic_rank"],
        evidence_valid=result["evidence_valid"],
        flux_boundary=result["flux_boundary"],
        shape_boundary=result["shape_boundary"],
        route_evidence=result["route_evidence"],
        optimizer_success=optimizer_success,
        fit_seconds=fit_seconds,
        diagnostic_seconds=diagnostic_seconds,
    )

    print(
        f"{origin:8s} | "
        f"{case['morphology']:10s} | "
        f"scene={case['scene_id']:3d} | "
        f"score={result['evidence_score']:8.3f} | "
        f"route={result['route_evidence']} | "
        f"valid={result['evidence_valid']} | "
        f"diagnostic={diagnostic_seconds:.3f}s",
        flush=True,
    )

    return row


def summarize(rows):

    print("\n--- Evidence screening results ---")

    groups = {
        "single_profile": [
            r for r in rows
            if r["origin"] == "archive"
        ],
        "achromatic": [
            r for r in rows
            if r["morphology"] == "achromatic"
        ],
        "chromatic": [
            r for r in rows
            if r["morphology"] == "chromatic"
        ],
    }

    for name, group in groups.items():

        selected = sum(
            r["route_evidence"] for r in group
        )

        valid = [
            r["evidence_score"]
            for r in group
            if r["evidence_valid"]
        ]

        print(
            f"\n{name}"
            f"\n  Scenes: {len(group)}"
            f"\n  Refinement candidates: "
            f"{selected}/{len(group)}"
            f"\n  Valid diagnostics: "
            f"{len(valid)}/{len(group)}"
            f"\n  Shape boundary cases: "
            f"{sum(r['shape_boundary'] for r in group)}"
        )

        if valid:

            print(
                f"  Minimum score: {min(valid):.4f}"
                f"\n  Median score: {np.median(valid):.4f}"
                f"\n  Maximum score: {max(valid):.4f}"
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    print("Phase 10.3.2 — Chromatic evidence")
    print(f"Reference threshold: {REFERENCE_THRESHOLD:.6f}")
    print("Archived initial fits: 20")
    print("New initial fits: 8")
    print("New chromatic refinement fits: 0")
    print("Expected scene records: 28", flush=True)

    rows = []

    for case, theta, expected_flux, expected_chi2 in (
        archived_cases()
    ):

        scene = generate_observation(case)

        start = perf_counter()

        result = chromatic_evidence(
            scene, theta
        )

        diagnostic_seconds = (
            perf_counter() - start
        )

        # Verify archive reconstruction.
        np.testing.assert_allclose(
            result["flux"],
            expected_flux,
            rtol=5e-5,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            result["chi2"],
            expected_chi2,
            rtol=1e-5,
        )

        rows.append(
            make_row(
                case,
                scene,
                result,
                "archive",
                0.0,
                diagnostic_seconds,
                True,
            )
        )

    for case, scene in new_cases():

        start = perf_counter()

        initial_flux, _, diagnostics = (
            fit_flexible_sersic(
                scene["observed"],
                scene["psfs"],
                scene["positions"],
                scene["noise_sigma"],
            )
        )

        fit_seconds = (
            perf_counter() - start
        )

        theta = np.stack([
            shape_vector(shape)
            for shape in diagnostics["shapes"]
        ])

        start = perf_counter()

        result = chromatic_evidence(
            scene, theta
        )

        diagnostic_seconds = (
            perf_counter() - start
        )

        np.testing.assert_allclose(
            result["flux"],
            initial_flux,
            rtol=5e-5,
            atol=1e-5,
        )

        # Nonconverged initial solutions require
        # explicit review rather than confidence.
        success = bool(
            diagnostics["optimizer_success"]
        )

        if not success:
            result["evidence_valid"] = False
            result["route_evidence"] = True

        rows.append(
            make_row(
                case,
                scene,
                result,
                "new",
                fit_seconds,
                diagnostic_seconds,
                success,
            )
        )

    if len(rows) != 28:
        raise RuntimeError(
            "Incomplete benchmark"
        )

    with OUTPUT.open(
        "x", newline="", encoding="utf-8"
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)

    print(f"\nResults saved: {OUTPUT}")


if __name__ == "__main__":
    main()
