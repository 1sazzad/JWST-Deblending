
"""Phase 10.3.1: frozen reliability-guided controller.

Validate the revised policy on 32 new observations.

Reuse existing initial fitting, uncertainty estimation
and nonlinear refinement implementations.

No threshold tuning on validation outcomes.
"""

import csv
from pathlib import Path

import numpy as np

from src.evaluation.benchmark_uncertainty_adaptive import (
    evaluate_case,
    load_controller,
)

from src.evaluation.benchmark_uncertainty_calibration import (
    grouped_rows,
)

# ---------------------------------------------------------
# Frozen experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results"

OUTPUT = (
    RESULTS
    / "benchmark_reliability_controller_phase10_3_1.csv"
)

PILOT_ARCHIVE = (
    RESULTS
    / "benchmark_uncertainty_adaptive_phase10_2.csv"
)

N_SCENES = 32

RANDOM_SEED = 6102026

EXPECTED_THRESHOLD = 0.16573900249573076

EXPECTED_Q95 = 2.880923563626555

# ---------------------------------------------------------
# Freeze the revised policy
# ---------------------------------------------------------

def frozen_policy():

    controller, used_seeds = (
        load_controller()
    )

    q95 = float(
        controller["q95"]
    )

    # The old controller's u0 is the median
    # calibrated scene-level uncertainty.
    threshold = (
        controller["u0"] / q95
    )

    np.testing.assert_allclose(
        threshold,
        EXPECTED_THRESHOLD,
        rtol=1e-8,
    )

    np.testing.assert_allclose(
        q95,
        EXPECTED_Q95,
        rtol=1e-8,
    )

    # Prevent overlap with the previous pilot.
    with PILOT_ARCHIVE.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        for row in csv.DictReader(stream):

            used_seeds.add(
                int(row["seed"])
            )

    return (
        controller,
        threshold,
        used_seeds,
    )

# ---------------------------------------------------------
# Independent observation generation
# ---------------------------------------------------------

def generate_cases(used_seeds):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    cases = []

    for i in range(N_SCENES):

        seed = int(
            rng.integers(
                2_000_000_000,
                2_100_000_000,
            )
        )

        if seed in used_seeds:

            raise ValueError(
                "Duplicate simulation seed"
            )

        used_seeds.add(seed)

        separation = int(
            rng.choice([
                2, 4, 6, 8, 12
            ])
        )

        ratio = float(
            rng.choice([
                0.1, 0.25, 0.5, 1.0
            ])
        )

        # Preserve the Phase 10.1
        # simulation distribution.
        if rng.random() < 0.5:

            family = "chromatic"

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
                rng.choice([
                    0.5, 1.0, 2.0, 4.0
                ])
            )

            family = (
                "gaussian"
                if n == 0.5
                else f"sersic{int(n)}"
            )

            fractions = None

        cases.append(dict(
            scene_id=i + 1,

            split="validation",

            seed=seed,

            separation=separation,

            ratio=ratio,

            morphology=family,

            sersic_n=n,

            fractions=fractions,
        ))

    return cases

# ---------------------------------------------------------
# Revised observation-only controller
# ---------------------------------------------------------

def apply_reliability_policy(
    rows,
    threshold,
):

    # Calculate the maximum relative
    # morphology-aware uncertainty.
    scores = np.array([
        float(row["scale_proposed"])
        / float(row["scale_relative"])

        for row in rows
    ])

    valid = (
        all(
            as_bool(
                row["diagnostic_valid"]
            )
            for row in rows
        )

        and np.isfinite(scores).all()
    )

    score = (
        float(scores.max())
        if np.isfinite(scores).all()
        else np.inf
    )

    # Invalid diagnostics explicitly trigger
    # an attempt at physical refinement.
    selected = (
        not valid
        or score >= threshold
    )

    reason = (
        "invalid_diagnostic"
        if not valid
        else (
            "high_uncertainty"
            if selected
            else "retain_initial"
        )
    )

    # The counterfactual refinement has
    # already been evaluated offline.
    for row in rows:

        final_flux = float(
            row["refined_flux"]
            if selected
            else row["initial_flux"]
        )

        row.update(
            reliability_score=score,

            reliability_threshold=threshold,

            reliability_valid=valid,

            route_reliability=selected,

            reliability_reason=reason,

            reliability_flux=final_flux,
        )

        # Ground truth is used exclusively
        # for evaluating the selected result.
        truth = float(
            row["true_flux"]
        )

        row["reliability_error"] = abs(
            final_flux / truth - 1.0
        )

    mean_error = np.mean([
        r["reliability_error"]
        for r in rows
    ])

    print(
        f"Revised controller | "
        f"score={score:.5f} | "
        f"route={selected} | "
        f"reason={reason} | "
        f"error={mean_error*100:.2f}%",
        flush=True,
    )

    return rows

def as_bool(value):

    return (
        str(value).strip().lower()
        == "true"
    )

# ---------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------

def summarize(rows):

    groups = sorted(
        grouped_rows(rows).values(),

        key=lambda g: int(
            g[0]["scene_id"]
        ),
    )

    count = len(groups)

    def numeric(key):

        return np.array([
            float(g[0][key])
            for g in groups
        ])

    def mean_error(key):

        return np.array([
            np.mean([
                float(r[key])
                for r in g
            ])

            for g in groups
        ])

    initial = mean_error(
        "initial_error"
    )

    refined = mean_error(
        "refined_error"
    )

    benefit = (
        initial - refined
    )

    selected = np.array([
        as_bool(
            g[0]["route_reliability"]
        )
        for g in groups
    ])

    invalid = np.array([
        not as_bool(
            g[0]["reliability_valid"]
        )
        for g in groups
    ])

    budget = int(
        selected.sum()
    )

    # All primary equal-budget comparators
    # receive the same invalid-system guard.
    def top_budget(
        scores,
        force_invalid=True,
    ):

        scores = np.asarray(
            scores,
            dtype=float,
        )

        scores = np.where(
            np.isfinite(scores),
            scores,
            -np.inf,
        )

        chosen = (
            invalid.copy()
            if force_invalid
            else np.zeros(
                count,
                dtype=bool,
            )
        )

        order = np.argsort(
            -scores,
            kind="stable",
        )

        for i in order:

            if chosen.sum() >= budget:
                break

            chosen[i] = True

        return chosen

    strategies = {
        "initial_only": np.zeros(
            count,
            dtype=bool,
        ),

        "always_refine": np.ones(
            count,
            dtype=bool,
        ),

        "reliability_controller": selected,

        "chi2": top_budget(
            numeric("chi2_score")
        ),

        "chromatic_covariance": top_budget(
            numeric("covariance_score")
        ),

        "ambiguity_only": top_budget(
            numeric("ambiguity_score")
        ),

        "ambiguity_noise": top_budget(
            numeric("ambiguity_noise_score")
        ),

        "previous_combined_score": top_budget(
            numeric("priority")
        ),

        # Ground-truth oracle:
        # evaluation only.
        "oracle": top_budget(
            benefit,
            force_invalid=False,
        ),
    }

    # Invalid-guard ablation:
    # keep the same frozen threshold.
    strategies["without_invalid_guard"] = (
        ~invalid
        & (
            numeric("reliability_score")
            >= numeric("reliability_threshold")
        )
    )

    chromatic = np.array([
        g[0]["morphology"]
        == "chromatic"

        for g in groups
    ])

    initial_time = numeric(
        "initial_seconds"
    )

    diagnostic_time = numeric(
        "diagnostic_seconds"
    )

    refinement_time = numeric(
        "refinement_seconds"
    )

    def safe_mean(values):

        return (
            float(np.mean(values))
            if len(values)
            else np.nan
        )

    print("\n--- Experiment inventory ---")

    print(
        f"Completed scenes: "
        f"{count}/{N_SCENES}"
    )

    print(
        f"Chromatic scenes: "
        f"{chromatic.sum()}"
    )

    print(
        f"Single-profile scenes: "
        f"{(~chromatic).sum()}"
    )

    print(
        f"Invalid diagnostics: "
        f"{invalid.sum()}"
    )

    print(
        f"Proposed refinement budget: "
        f"{budget}/{count}"
    )

    # Expected random routing with
    # the same forced-invalid policy.
    remaining = ~invalid

    extra_budget = (
        budget - invalid.sum()
    )

    random_probability = (
        extra_budget / remaining.sum()
        if remaining.sum() > 0
        else 0.0
    )

    random_error = (
        initial.sum()
        - benefit[invalid].sum()
        - random_probability
        * benefit[remaining].sum()
    ) / count

    print(
        f"Expected guarded random error: "
        f"{100*random_error:.2f}%"
    )

    print("\n--- Method comparison ---")

    for name, route in strategies.items():

        error = np.where(
            route,
            refined,
            initial,
        )

        cost = (
            initial_time.sum()
            + refinement_time[route].sum()
        )

        # Charge full diagnostic overhead
        # to controllers that use it.
        if name not in (
            "initial_only",
            "always_refine",
            "oracle",
        ):

            cost += diagnostic_time.sum()

        faint_errors = []

        measurement_errors = []

        for i, group in enumerate(groups):

            key = (
                "refined_error"
                if route[i]
                else "initial_error"
            )

            for row in group:

                value = float(
                    row[key]
                )

                measurement_errors.append(
                    value
                )

                if (
                    int(row["source"]) == 2

                    and float(
                        row["flux_ratio_f277w"]
                    ) == 0.1
                ):

                    faint_errors.append(
                        value
                    )

        print(
            f"\n{name}"
            f"\n  Refinements: {route.sum()}/{count}"
            f"\n  Mean error: {error.mean()*100:.2f}%"
            f"\n  Median error: {np.median(measurement_errors)*100:.2f}%"
            f"\n  P90 error: {np.percentile(measurement_errors,90)*100:.2f}%"
            f"\n  Single-profile error: {safe_mean(error[~chromatic])*100:.2f}%"
            f"\n  Chromatic error: {safe_mean(error[chromatic])*100:.2f}%"
            f"\n  Faint companion error: {safe_mean(faint_errors)*100:.2f}%"
            f"\n  Estimated processing cost: {cost:.2f}s"
        )

    # Check whether improvement depends
    # on a single catastrophic scene.
    final_error = np.where(
        selected,
        refined,
        initial,
    )

    improvement = (
        initial - final_error
    )

    largest = int(
        np.argmax(improvement)
    )

    if count > 1:

        keep = np.ones(
            count,
            dtype=bool,
        )

        keep[largest] = False

        print(
            "\n--- Robustness check ---"
        )

        print(
            f"Largest improvement scene: "
            f"{groups[largest][0]['scene_id']}"
        )

        print(
            f"Initial error excluding scene: "
            f"{initial[keep].mean()*100:.2f}%"
        )

        print(
            f"Adaptive error excluding scene: "
            f"{final_error[keep].mean()*100:.2f}%"
        )

    print(
        f"\nImproved scenes: "
        f"{np.sum(improvement > 1e-10)}"
    )

    print(
        f"Degraded scenes: "
        f"{np.sum(improvement < -1e-10)}"
    )

    print(
        f"Converged refinement fits: "
        f"{sum(as_bool(g[0]['refinement_converged']) for g in groups)}/{count}"
    )

# ---------------------------------------------------------
# Preserve completed results if interrupted
# ---------------------------------------------------------

def load_completed(cases, threshold):

    if not OUTPUT.exists():

        return [], None

    with OUTPUT.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        reader = csv.DictReader(
            stream
        )

        fields = reader.fieldnames

        rows = list(reader)

    groups = grouped_rows(
        rows
    )

    expected = {
        c["scene_id"]: c
        for c in cases
    }

    for scene_id, group in groups.items():

        if scene_id not in expected:

            raise ValueError(
                "Unexpected saved scene"
            )

        case = expected[scene_id]

        for row in group:

            if (
                int(row["seed"])
                != case["seed"]

                or row["morphology"]
                != case["morphology"]

                or float(
                    row["reliability_threshold"]
                ) != threshold
            ):

                raise ValueError(
                    "Saved experiment mismatch"
                )

    return rows, fields

# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    (
        controller,
        threshold,
        used_seeds,
    ) = frozen_policy()

    cases = generate_cases(
        used_seeds
    )

    rows, fields = load_completed(
        cases,
        threshold,
    )

    completed = set(
        grouped_rows(rows)
    )

    print(
        "Phase 10.3.1 — "
        "Reliability-guided adaptive deblending"
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.12f}"
    )

    print(
        f"Planned scenes: "
        f"{N_SCENES}"
    )

    print(
        f"Previously completed: "
        f"{len(completed)}"
    )

    print(
        f"Remaining initial fits: "
        f"{N_SCENES-len(completed)}"
    )

    print(
        f"Remaining refinement fits: "
        f"{N_SCENES-len(completed)}"
    )

    print(
        f"Expected measurements: "
        f"{N_SCENES*6}",
        flush=True,
    )

    for case in cases:

        if case["scene_id"] in completed:

            continue

        # Existing Phase 10.2 evaluation.
        # No new nonlinear fitter is needed.
        batch = evaluate_case(
            case,
            controller,
        )

        batch = apply_reliability_policy(
            batch,
            threshold,
        )

        grouped_rows(batch)

        current_fields = list(
            batch[0]
        )

        new_file = not OUTPUT.exists()

        if fields is None:

            fields = current_fields

        if fields != current_fields:

            raise ValueError(
                "CSV schema mismatch"
            )

        # Write one complete scene at a time.
        with OUTPUT.open(
            "x" if new_file else "a",
            newline="",
            encoding="utf-8",
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
            )

            if new_file:

                writer.writeheader()

            writer.writerows(
                batch
            )

        rows.extend(
            batch
        )

    if len(rows) != N_SCENES * 6:

        raise RuntimeError(
            "Incomplete experiment"
        )

    summarize(
        rows
    )

    print(
        f"\nResults saved: {OUTPUT}"
    )

if __name__ == "__main__":
    main()