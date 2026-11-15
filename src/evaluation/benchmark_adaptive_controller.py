
"""Phase 9.2.2: held-out adaptive controller validation.

Evaluate sensitivity-based routing under a fixed refinement
budget on new simulation configurations and noise seeds.

All expensive fits are executed offline to obtain paired
counterfactual outcomes. Routing uses observation-derived
features only, except for the explicitly labelled oracle.
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from src.evaluation.benchmark_baseline import (
    create_configuration,
)

from src.evaluation.benchmark_composite_morphology import (
    create_composite_components,
    create_composite_scene,
    BULGE_FRACTIONS,
    fit_method,
)

from src.evaluation.benchmark_flux_sensitivity import (
    flux_sensitivity,
)

from src.simulation.simulate_blends import (
    simulate_scene,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "results"
    / "benchmark_adaptive_controller_phase9_2_2.csv"
)

# New configurations, distinct from the five
# previously selected development configurations.
# Columns: separation, F277W flux ratio, Sersic index.

CASES = (
    (2, 0.25, 0.5),
    (4, 0.10, 1.0),
    (4, 0.50, 2.0),
    (6, 1.00, 4.0),
    (8, 0.10, 0.5),
    (8, 0.50, 1.0),
    (12, 0.25, 2.0),
    (12, 0.50, 4.0),
)

BASE_SEED = 21000

TRACKS = (
    "single_profile",
    "chromatic_composite",
)

# Prespecified routing budget.
REFINEMENT_FRACTION = 2 / 3


def evaluate_scene(scene, track, case_id, seed, sep, ratio):
    """Measure diagnostic features and paired fit outcomes."""

    truth = scene["input_flux"]

    # Initial lightweight estimation.
    start = perf_counter()

    fixed, fixed_model, _ = fit_method(
        scene,
        "fixed_gaussian",
    )

    fixed_seconds = perf_counter() - start

    # Observation-derived sensitivity.
    start = perf_counter()

    features = flux_sensitivity(
        scene["observed"],
        scene["psfs"],
        scene["positions"],
        scene["noise_sigma"],
        fixed.T,
    )

    diagnostic_seconds = perf_counter() - start

    # Validate initial reconstruction consistency.
    chi2 = float(np.mean(
        (
            (scene["observed"] - fixed_model)
            / scene["noise_sigma"][:, None, None]
        ) ** 2
    ))

    np.testing.assert_allclose(
        features["fixed_chi2"],
        chi2,
        rtol=1e-5,
        atol=1e-6,
    )

    # Evaluate the expensive alternative offline.
    # Its result is not used to compute routing features.
    start = perf_counter()

    refined, _, diagnostics = fit_method(
        scene,
        "flexible_sersic",
    )

    refined_seconds = perf_counter() - start

    # Validate fit outputs.
    for flux in (fixed, refined):

        if (
            flux.shape != truth.shape
            or not np.isfinite(flux).all()
            or np.any(flux < 0)
        ):
            raise ValueError(
                "Invalid flux estimates"
            )

    # Ground truth is used exclusively for evaluation.
    fixed_error = np.abs(
        fixed / truth - 1
    )

    refined_error = np.abs(
        refined / truth - 1
    )

    sensitivity = features[
        "shared_relative_shift"
    ]

    # Route unresolved local systems to refinement.
    # Do not silently drop invalid diagnostics.
    unresolved = not np.isfinite(sensitivity)

    priority = (
        np.inf if unresolved
        else sensitivity
    )

    result = {
        "track": track,
        "case_id": case_id,
        "seed": seed,
        "separation_px": sep,

        **features,

        "sensitivity_priority": priority,
        "unresolved_shared_system": unresolved,

        # Evaluation-only quantities.
        "true_flux_ratio": ratio,
        "fixed_mean_error": float(
            fixed_error.mean()
        ),
        "fixed_max_error": float(
            fixed_error.max()
        ),
        "refined_mean_error": float(
            refined_error.mean()
        ),
        "refined_max_error": float(
            refined_error.max()
        ),
        "refinement_benefit": float(
            fixed_error.mean()
            - refined_error.mean()
        ),
        "refined_optimizer_success": bool(
            diagnostics["optimizer_success"]
        ),
        "fixed_seconds": fixed_seconds,
        "diagnostic_seconds": diagnostic_seconds,
        "refined_seconds": refined_seconds,
    }

    print(
        f"{track} | case {case_id} | "
        f"priority={priority:.4f} | "
        f"fixed={fixed_error.mean()*100:.2f}% | "
        f"refined={refined_error.mean()*100:.2f}%",
        flush=True,
    )

    return result


def select_top(rows, key, budget):
    """Rank observations using the specified feature."""

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            -float(rows[i][key]),
            i,
        ),
    )

    return set(order[:budget])


def evaluate_routing(rows):

    count = len(rows)

    budget = int(np.ceil(
        count * REFINEMENT_FRACTION
    ))

    strategies = {
        "sensitivity": (
            "sensitivity_priority"
        ),
        "relative_uncertainty": (
            "fixed_relative_sigma"
        ),
        "chi2": (
            "fixed_chi2"
        ),

        # Upper bound only: unavailable at inference.
        "oracle": (
            "refinement_benefit"
        ),
    }

    fixed = np.array([
        r["fixed_mean_error"]
        for r in rows
    ])

    refined = np.array([
        r["refined_mean_error"]
        for r in rows
    ])

    print(
        f"\nRefinement budget: "
        f"{budget}/{count}"
    )

    print(
        f"Fixed only: "
        f"{fixed.mean()*100:.2f}%"
    )

    print(
        f"Refine everything: "
        f"{refined.mean()*100:.2f}%"
    )

    expected_random = (
        fixed.mean()
        - (budget / count)
        * (fixed - refined).mean()
    )

    print(
        f"Random routing, expected: "
        f"{expected_random*100:.2f}%"
    )

    selections = {}

    for name, key in strategies.items():

        selected = select_top(
            rows,
            key,
            budget,
        )

        selections[name] = selected

        routed_error = np.array([
            refined[i]
            if i in selected
            else fixed[i]
            for i in range(count)
        ])

        print(
            f"{name}: "
            f"{routed_error.mean()*100:.2f}%"
        )

        for i, row in enumerate(rows):

            row[f"route_{name}"] = (
                i in selected
            )

            row[f"error_{name}"] = (
                routed_error[i]
            )

    # Estimated sequential processing costs.
    # Exclude simulation and archive I/O.
    fixed_cost = sum(
        r["fixed_seconds"]
        for r in rows
    )

    diagnostic_cost = sum(
        r["diagnostic_seconds"]
        for r in rows
    )

    all_refined_cost = sum(
        r["refined_seconds"]
        for r in rows
    )

    adaptive_cost = (
        fixed_cost
        + diagnostic_cost
        + sum(
            rows[i]["refined_seconds"]
            for i in selections["sensitivity"]
        )
    )

    print("\n--- Processing costs ---")

    print(
        f"Initial fits: "
        f"{fixed_cost:.2f}s"
    )

    print(
        f"Diagnostic overhead: "
        f"{diagnostic_cost:.2f}s"
    )

    print(
        f"Always-refine pipeline: "
        f"{fixed_cost + all_refined_cost:.2f}s"
    )

    print(
        f"Adaptive pipeline estimate: "
        f"{adaptive_cost:.2f}s"
    )


def main():

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    rows = []

    print(
        f"Planned new scenes: "
        f"{len(CASES) * len(TRACKS)}"
    )

    print(
        f"Planned expensive fits: "
        f"{len(CASES) * len(TRACKS)}"
    )

    for case_id, (sep, ratio, n) in enumerate(
        CASES,
        start=1,
    ):

        seed = BASE_SEED + case_id

        config = create_configuration(
            sep,
            ratio,
            seed,
        )

        # Single-profile observation.
        if n == 0.5:

            single = simulate_scene(
                **config
            )

        else:

            single = simulate_scene(
                **config,
                sersic_n=n,
            )

        rows.append(
            evaluate_scene(
                single,
                "single_profile",
                case_id,
                seed,
                sep,
                ratio,
            )
        )

        # Matched chromatic composite observation.
        bulge, disk = (
            create_composite_components(
                config
            )
        )

        composite = create_composite_scene(
            bulge,
            disk,
            BULGE_FRACTIONS["chromatic"],
        )

        rows.append(
            evaluate_scene(
                composite,
                "chromatic_composite",
                case_id,
                seed,
                sep,
                ratio,
            )
        )

    evaluate_routing(rows)

    converged = sum(
        r["refined_optimizer_success"]
        for r in rows
    )

    print(
        f"\nConverged refined fits: "
        f"{converged}/{len(rows)}"
    )

    with OUTPUT.open(
        "x",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Results saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
