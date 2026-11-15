
"""Phase 10.1: morphology-aware photometric uncertainty.

Compare six uncertainty scales using the same flexible Sersic
point estimator and scene-level split conformal calibration.

Calibration and test observations are independently generated
from a prespecified simulation distribution.

No previous benchmark fits are repeated.
"""

import argparse
import csv
import math

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.linalg import block_diag
from scipy.stats import binomtest

from src.baselines.baseline_flexible_sersic import (
    SersicObjective,
    fit_flexible_sersic,
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
    fit_templates,
    prepare_band,
    evaluate_alpha,
    center_columns,
)

from src.simulation.simulate_blends import (
    simulate_scene,
)


# ---------------------------------------------------------
# Frozen experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "results"
    / "benchmark_uncertainty_calibration_phase10_1.csv"
)

N_CALIBRATION = 39
N_TEST = 24

RANDOM_SEED = 4102026

BANDS = (
    "f150w",
    "f277w",
    "f444w",
)

LEVELS = (
    0.68,
    0.95,
)

METHODS = (
    "relative",
    "fixed",
    "shared",
    "chromatic",
    "fixed_ambiguity",
    "proposed",
)

RANK_TOLERANCE = 1e-7

FIELDS = [
    "scene_id",
    "split",
    "seed",
    "morphology",
    "separation_px",
    "flux_ratio_f277w",
    "source",
    "band",
    "true_flux",
    "estimated_flux",
    "absolute_error",
    "chi2",
    "linear_ambiguity",
    "shared_valid",
    "chromatic_valid",
    "diagnostic_reason",
    "optimizer_success",
    "fit_seconds",
    "diagnostic_seconds",
] + [
    f"scale_{name}"
    for name in METHODS
]


# ---------------------------------------------------------
# Independent simulation design
# ---------------------------------------------------------

def generate_cases():
    """Freeze all scene settings before fitting."""

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    cases = []

    used_seeds = set()

    for i in range(
        N_CALIBRATION + N_TEST
    ):

        seed = int(
            rng.integers(
                1_000_000,
                1_000_000_000
            )
        )

        if seed in used_seeds:
            raise RuntimeError(
                "Duplicate simulation seed"
            )

        used_seeds.add(seed)

        separation = int(
            rng.choice(
                [2, 4, 6, 8, 12]
            )
        )

        ratio = float(
            rng.choice(
                [0.1, 0.25, 0.5, 1.0]
            )
        )

        # Half of the expected population
        # contains single-profile galaxies.
        composite = bool(
            rng.random() < 0.5
        )

        if composite:

            family = "chromatic"

            # Independently vary the bulge
            # fraction of both galaxies.
            middle = rng.uniform(
                0.25,
                0.75,
                size=2,
            )

            slope = rng.uniform(
                0.10,
                0.25,
                size=2,
            )

            fractions = np.column_stack([
                middle - slope,
                middle,
                middle + slope,
            ])

            n = None

        else:

            n = float(
                rng.choice(
                    [0.5, 1.0, 2.0, 4.0]
                )
            )

            family = (
                "gaussian"
                if n == 0.5
                else f"sersic{int(n)}"
            )

            fractions = None

        cases.append(dict(
            scene_id=i + 1,

            split=(
                "calibration"
                if i < N_CALIBRATION
                else "test"
            ),

            seed=seed,
            separation=separation,
            ratio=ratio,
            morphology=family,
            sersic_n=n,
            fractions=fractions,
        ))

    return cases


def generate_observation(case):
    """Use the frozen simulator configuration."""

    config = create_configuration(
        case["separation"],
        case["ratio"],
        case["seed"],
    )

    if case["morphology"] == "chromatic":

        bulge, disk = (
            create_composite_components(
                config
            )
        )

        return create_composite_scene(
            bulge,
            disk,
            case["fractions"],
        )

    if case["morphology"] == "gaussian":

        return simulate_scene(
            **config
        )

    return simulate_scene(
        **config,
        sersic_n=case["sersic_n"],
    )


# ---------------------------------------------------------
# Stable covariance approximation
# ---------------------------------------------------------

def covariance_std(J):
    """Return local parameter standard deviations.

    J must be a whitened design matrix.
    Reject rank-deficient systems explicitly.
    """

    norms = np.linalg.norm(
        J,
        axis=0,
    )

    if (
        not np.isfinite(norms).all()
        or np.any(norms <= 1e-12)
    ):
        return None

    scaled = J / norms

    _, d, vt = np.linalg.svd(
        scaled,
        full_matrices=False,
    )

    if (
        len(d) != J.shape[1]
        or d[0] <= 0
        or d[-1] <= RANK_TOLERANCE * d[0]
    ):
        return None

    # Diagonal of the inverse Fisher matrix.
    variance = (
        np.sum(
            (vt.T / d)**2,
            axis=1,
        )
        / norms**2
    )

    sigma = np.sqrt(
        np.maximum(
            variance,
            0.0,
        )
    )

    if (
        not np.isfinite(sigma).all()
        or np.any(sigma <= 0)
    ):
        return None

    return sigma


# ---------------------------------------------------------
# Morphology-aware uncertainty scales
# ---------------------------------------------------------

def uncertainty_features(
    observed,
    psfs,
    positions,
    noise,
    theta,
):
    """Compute scales using observation information only."""

    objective = SersicObjective(
        observed,
        psfs,
        positions,
        noise,
    )

    # Shape:
    # sources x profile/derivatives x bands x H x W
    blurred = np.stack([
        objective.convolve(
            quadrature.evaluate(theta[s])
        )
        for s, quadrature in enumerate(
            objective.quad
        )
    ])

    templates = np.stack([
        blurred[:, 0, b]
        for b in range(len(noise))
    ])

    flux, _, chi2 = fit_templates(
        objective,
        templates,
    )

    fixed_sigma = np.empty_like(
        flux
    )

    local_sigma = np.empty_like(
        flux
    )

    ambiguity = np.full_like(
        flux,
        np.nan,
    )

    chromatic_valid = np.ones(
        len(noise),
        dtype=bool,
    )

    reasons = [
        ""
        for _ in noise
    ]

    X_blocks = []
    Z_shared_blocks = []

    for b in range(len(noise)):

        # Profile out the constant background.
        X = (
            templates[b]
            .reshape(2, -1)
            .T / noise[b]
        )

        X = center_columns(X)

        X_blocks.append(X)

        sigma = covariance_std(X)

        if sigma is None:
            raise RuntimeError(
                "Fixed flux design is rank deficient"
            )

        fixed_sigma[:, b] = sigma

        # Shared-morphology covariance:
        # all four shape parameters per source.
        shared_derivatives = []

        for s in range(2):

            shared_derivatives.append(
                blurred[s, 1:, b]
                .reshape(4, -1)
                .T
                * flux[s, b]
                / noise[b]
            )

        Z_shared_blocks.append(
            center_columns(
                np.column_stack(
                    shared_derivatives
                )
            )
        )

        # Band-specific radial and concentration
        # perturbations used in Phase 9.3.1.
        derivatives = []

        for s in range(2):

            derivatives.append(
                blurred[s, 1, b]
                + blurred[s, 2, b]
            )

            derivatives.append(
                blurred[s, 4, b]
            )

        try:

            system = prepare_band(
                templates[b],
                np.stack(derivatives),
                observed[b],
                noise[b],
                flux[:, b],
            )

            full = evaluate_alpha(
                system,
                0.0,
            )

            local_sigma[:, b] = (
                full["linear_sigma"]
            )

            ambiguity[:, b] = (
                full["ambiguity"]
            )

            if (
                not np.isfinite(
                    local_sigma[:, b]
                ).all()

                or np.any(
                    local_sigma[:, b] <= 0
                )

                or not np.isfinite(
                    ambiguity[:, b]
                ).all()
            ):
                raise ValueError(
                    "Invalid local covariance"
                )

        except (
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:

            chromatic_valid[b] = False

            reasons[b] = str(exc)

            # Prespecified finite fallback scale.
            # This is not a guaranteed error bound.
            fallback = max(
                float(
                    np.abs(
                        flux[:, b]
                    ).sum()
                ),
                float(
                    fixed_sigma[:, b].max()
                ),
                1e-12,
            )

            local_sigma[:, b] = fallback

            ambiguity[:, b] = np.nan

    # Joint shared morphology covariance.
    #
    # Six flux parameters and eight shared
    # nonlinear morphology parameters.
    J_shared = np.column_stack([
        block_diag(*X_blocks),
        np.vstack(
            Z_shared_blocks
        ),
    ])

    shared_std = covariance_std(
        J_shared
    )

    shared_valid = (
        shared_std is not None
    )

    fallback = np.maximum(
        np.abs(flux).sum(
            axis=0,
            keepdims=True,
        ),
        fixed_sigma,
    )

    if shared_valid:

        shared_sigma = (
            shared_std[:6]
            .reshape(3, 2)
            .T
        )

    else:

        shared_sigma = fallback.copy()

    # Common relative scale for evaluation.
    relative = np.maximum(
        np.abs(flux),
        fixed_sigma,
    )

    fixed_ambiguity = np.hypot(
        fixed_sigma,
        ambiguity,
    )

    proposed = np.hypot(
        local_sigma,
        ambiguity,
    )

    # Handle invalid bands without dropping
    # difficult observations.
    for b in range(len(noise)):

        if not chromatic_valid[b]:

            fixed_ambiguity[:, b] = (
                local_sigma[:, b]
            )

            proposed[:, b] = (
                local_sigma[:, b]
            )

    scales = {
        "relative": relative,
        "fixed": fixed_sigma,
        "shared": shared_sigma,
        "chromatic": local_sigma,
        "fixed_ambiguity": fixed_ambiguity,
        "proposed": proposed,
    }

    for name, scale in scales.items():

        if (
            not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise RuntimeError(
                f"Invalid scale: {name}"
            )

    return dict(
        flux=flux,
        chi2=chi2,
        scales=scales,
        ambiguity=ambiguity,
        shared_valid=shared_valid,
        chromatic_valid=chromatic_valid,
        reasons=reasons,
    )


# ---------------------------------------------------------
# Evaluate one new observation
# ---------------------------------------------------------

def evaluate_scene(case):

    scene = generate_observation(
        case
    )

    observed = scene["observed"]
    psfs = scene["psfs"]
    positions = scene["positions"]
    noise = scene["noise_sigma"]

    bands = tuple(
        str(b).lower()
        for b in scene["bands"]
    )

    if bands != BANDS:
        raise ValueError(
            "Unexpected band order"
        )

    start = perf_counter()

    fitted_flux, _, diagnostics = (
        fit_flexible_sersic(
            observed,
            psfs,
            positions,
            noise,
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

    result = uncertainty_features(
        observed,
        psfs,
        positions,
        noise,
        theta,
    )

    diagnostic_seconds = (
        perf_counter() - start
    )

    flux = result["flux"]

    # Check that reconstructing the fitted
    # morphology preserves the point estimate.
    np.testing.assert_allclose(
        flux,
        fitted_flux,
        rtol=5e-5,
        atol=1e-5,
    )

    # Ground truth is accessed only here.
    truth = scene["input_flux"]

    if (
        truth.shape != (2, 3)
        or not np.isfinite(truth).all()
        or np.any(truth <= 0)
    ):
        raise ValueError(
            "Invalid simulation truth"
        )

    error = np.abs(
        flux / truth - 1
    )

    rows = []

    for s in range(2):

        for b, band in enumerate(BANDS):

            row = dict(
                scene_id=case["scene_id"],
                split=case["split"],
                seed=case["seed"],
                morphology=case["morphology"],

                separation_px=case[
                    "separation"
                ],

                flux_ratio_f277w=case[
                    "ratio"
                ],

                source=s + 1,
                band=band,

                true_flux=truth[s, b],
                estimated_flux=flux[s, b],
                absolute_error=error[s, b],

                chi2=result["chi2"],

                linear_ambiguity=result[
                    "ambiguity"
                ][s, b],

                shared_valid=result[
                    "shared_valid"
                ],

                chromatic_valid=result[
                    "chromatic_valid"
                ][b],

                diagnostic_reason=result[
                    "reasons"
                ][b],

                optimizer_success=bool(
                    diagnostics[
                        "optimizer_success"
                    ]
                ),

                fit_seconds=fit_seconds,

                diagnostic_seconds=(
                    diagnostic_seconds
                ),
            )

            for name in METHODS:

                row[f"scale_{name}"] = (
                    result["scales"][name][s, b]
                )

            rows.append(row)

    print(
        f"{case['split']:11s} | "
        f"scene {case['scene_id']:3d} | "
        f"{case['morphology']:10s} | "
        f"error={error.mean()*100:6.2f}% | "
        f"converged={diagnostics['optimizer_success']} | "
        f"fit={fit_seconds:.2f}s | "
        f"diagnostic={diagnostic_seconds:.2f}s",
        flush=True,
    )

    return rows


# ---------------------------------------------------------
# Calibration and evaluation
# ---------------------------------------------------------

def grouped_rows(rows):

    groups = defaultdict(list)

    for row in rows:

        groups[int(
            row["scene_id"]
        )].append(row)

    for scene_id, group in groups.items():

        keys = {
            (
                int(r["source"]),
                r["band"],
            )
            for r in group
        }

        expected = {
            (s, b)
            for s in (1, 2)
            for b in BANDS
        }

        if (
            len(group) != 6
            or keys != expected
        ):
            raise ValueError(
                f"Incomplete or duplicate scene: "
                f"{scene_id}"
            )

    return groups


def validate_inventory(rows, cases):

    groups = grouped_rows(rows)

    if set(groups) != {
        c["scene_id"]
        for c in cases
    }:
        raise ValueError(
            "Unexpected scene inventory"
        )

    for case in cases:

        for row in groups[
            case["scene_id"]
        ]:

            if (
                row["split"] != case["split"]
                or int(row["seed"]) != case["seed"]
                or row["morphology"]
                != case["morphology"]
                or float(row["separation_px"])
                != case["separation"]
                or float(row["flux_ratio_f277w"])
                != case["ratio"]
            ):
                raise ValueError(
                    "Scene metadata mismatch"
                )

    return groups


def conformal_multiplier(
    calibration_groups,
    method,
    level,
):
    """Use one maximum residual score per scene."""

    scores = []

    for group in calibration_groups:

        score = max(
            abs(
                float(r["true_flux"])
                - float(r["estimated_flux"])
            )
            / float(r[f"scale_{method}"])

            for r in group
        )

        scores.append(score)

    n = len(scores)

    k = math.ceil(
        (n + 1) * level
    )

    if k > n:
        return np.inf

    return float(
        np.sort(scores)[k - 1]
    )


def interval_metrics(
    test_groups,
    method,
    level,
    q,
):
    """Evaluate coverage, width and interval score."""

    covered_scenes = []

    all_covered = []
    widths = []
    interval_scores = []

    chromatic_covered = []
    faint_covered = []

    alpha = 1.0 - level

    for group in test_groups:

        scene_covered = []

        for row in group:

            truth = float(
                row["true_flux"]
            )

            estimate = float(
                row["estimated_flux"]
            )

            scale = float(
                row[f"scale_{method}"]
            )

            normalizer = float(
                row["scale_relative"]
            )

            lower = max(
                0.0,
                estimate - q * scale,
            )

            upper = (
                estimate + q * scale
            )

            covered = (
                lower <= truth <= upper
            )

            width = upper - lower

            # Proper interval score:
            # narrower is better, but misses
            # receive an explicit penalty.
            score = (
                width

                + (2 / alpha)
                * max(
                    lower - truth,
                    0.0,
                )

                + (2 / alpha)
                * max(
                    truth - upper,
                    0.0,
                )
            )

            widths.append(
                width / normalizer
            )

            interval_scores.append(
                score / normalizer
            )

            all_covered.append(
                covered
            )

            scene_covered.append(
                covered
            )

            if (
                row["morphology"]
                == "chromatic"
            ):

                chromatic_covered.append(
                    covered
                )

            if (
                int(row["source"]) == 2

                and float(
                    row["flux_ratio_f277w"]
                ) == 0.1
            ):

                faint_covered.append(
                    covered
                )

        covered_scenes.append(
            all(scene_covered)
        )

    count = sum(
        covered_scenes
    )

    total = len(
        covered_scenes
    )

    ci = binomtest(
        count,
        total,
    ).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )

    def mean_or_nan(values):

        return (
            float(np.mean(values))
            if values
            else np.nan
        )

    return dict(
        scene_coverage=count / total,

        covered_scenes=count,

        total_scenes=total,

        coverage_ci_low=ci.low,

        coverage_ci_high=ci.high,

        marginal_coverage=mean_or_nan(
            all_covered
        ),

        mean_width=mean_or_nan(
            widths
        ),

        interval_score=mean_or_nan(
            interval_scores
        ),

        chromatic_coverage=mean_or_nan(
            chromatic_covered
        ),

        faint_coverage=mean_or_nan(
            faint_covered
        ),
    )


def summarize(rows, cases):

    groups = validate_inventory(
        rows,
        cases,
    )

    calibration = [
        g for g in groups.values()
        if g[0]["split"] == "calibration"
    ]

    test = [
        g for g in groups.values()
        if g[0]["split"] == "test"
    ]

    if (
        len(calibration)
        != N_CALIBRATION

        or len(test)
        != N_TEST
    ):
        raise ValueError(
            "Invalid calibration/test split"
        )

    print(
        "\n--- Diagnostic validity ---"
    )

    print(
        f"Calibration scenes: "
        f"{len(calibration)}"
    )

    print(
        f"Test scenes: {len(test)}"
    )

    test_rows = [
        r
        for group in test
        for r in group
    ]

    def is_true(value):

        return str(value).lower() == "true"

    converged = sum(
        is_true(
            g[0]["optimizer_success"]
        )
        for g in test
    )

    shared_valid = sum(
        is_true(
            g[0]["shared_valid"]
        )
        for g in test
    )

    chromatic_valid = sum(
        is_true(
            r["chromatic_valid"]
        )
        for r in test_rows
    )

    print(
        f"Converged test fits: "
        f"{converged}/{len(test)}"
    )

    print(
        f"Valid shared diagnostics: "
        f"{shared_valid}/{len(test)}"
    )

    print(
        f"Valid chromatic measurements: "
        f"{chromatic_valid}/{len(test_rows)}"
    )

    print(
        f"Test point estimation error: "
        f"{np.mean([float(r['absolute_error']) for r in test_rows])*100:.2f}%"
    )

    for level in LEVELS:

        print(
            f"\n--- Nominal joint coverage "
            f"{level*100:.0f}% ---"
        )

        for method in METHODS:

            q = conformal_multiplier(
                calibration,
                method,
                level,
            )

            metrics = interval_metrics(
                test,
                method,
                level,
                q,
            )

            print(
                f"\n{method}"
            )

            print(
                f"  Multiplier: {q:.6f}"
            )

            print(
                f"  Joint scene coverage: "
                f"{metrics['scene_coverage']*100:.2f}% "
                f"({metrics['covered_scenes']}/"
                f"{metrics['total_scenes']})"
            )

            print(
                f"  Coverage 95% CI: "
                f"{metrics['coverage_ci_low']*100:.2f}% "
                f"to "
                f"{metrics['coverage_ci_high']*100:.2f}%"
            )

            print(
                f"  Marginal flux coverage: "
                f"{metrics['marginal_coverage']*100:.2f}%"
            )

            print(
                f"  Mean normalized width: "
                f"{metrics['mean_width']*100:.2f}%"
            )

            print(
                f"  Normalized interval score: "
                f"{metrics['interval_score']:.6f}"
            )

            print(
                f"  Chromatic flux coverage: "
                f"{metrics['chromatic_coverage']*100:.2f}%"
            )

            print(
                f"  Faint companion coverage: "
                f"{metrics['faint_coverage']*100:.2f}%"
            )


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
        help=(
            "Analyze an existing complete CSV "
            "without running new fits."
        ),
    )

    args = parser.parse_args()

    cases = generate_cases()

    if args.summarize:

        with OUTPUT.open(
            newline="",
            encoding="utf-8-sig",
        ) as stream:

            rows = list(
                csv.DictReader(stream)
            )

        summarize(
            rows,
            cases,
        )

        return

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    planned = len(cases)

    print(
        "Phase 10.1 — "
        "Morphology-aware uncertainty"
    )

    print(
        f"Calibration scenes: {N_CALIBRATION}"
    )

    print(
        f"Test scenes: {N_TEST}"
    )

    print(
        f"Planned new fits: {planned}"
    )

    print(
        f"Expected measurements: "
        f"{planned * 6}",
        flush=True,
    )

    rows = []

    # Save each completed scene immediately.
    # An unexpected failure stops the benchmark
    # instead of silently excluding observations.
    with OUTPUT.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for case in cases:

            batch = evaluate_scene(
                case
            )

            writer.writerows(batch)

            stream.flush()

            rows.extend(batch)

    summarize(
        rows,
        cases,
    )

    print(
        f"\nCompleted scenes: "
        f"{len(rows)//6}/{planned}"
    )

    print(
        f"Results saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
