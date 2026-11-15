
"""Phase 10.4: integrated evidence-guided adaptive deblending.

Compare pre-refinement evidence screening, post-fit
acceptance and uncertainty-based budget prioritization.

All refinement counterfactuals are evaluated offline.
Routing decisions never access simulation ground truth.
"""

import argparse
import csv
import json

from pathlib import Path
from time import perf_counter

import numpy as np

from src.baselines.baseline_flexible_sersic import (
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
)

from src.evaluation.benchmark_uncertainty_calibration import (
    generate_observation,
    uncertainty_features,
    grouped_rows,
)

from src.evaluation.benchmark_uncertainty_adaptive import (
    fit_chromatic,
)

from src.evaluation.benchmark_chromatic_evidence import (
    chromatic_evidence,
    REFERENCE_THRESHOLD,
)


# ---------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT / "results"
    / "benchmark_integrated_adaptive_phase10_4.csv"
)

N_SCENES = 24

MAX_BUDGET = 6

RANDOM_SEED = 8102026

BANDS = ("f150w", "f277w", "f444w")

THRESHOLD = float(REFERENCE_THRESHOLD)

np.testing.assert_allclose(
    THRESHOLD,
    9.487729036781154,
    rtol=1e-10,
)


# ---------------------------------------------------------
# Independent simulation design
# ---------------------------------------------------------

def generate_cases():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    families = np.array(
        ["gaussian", "sersic1",
         "sersic2", "sersic4"] * 2
        + ["achromatic"] * 8
        + ["chromatic"] * 8
    )

    rng.shuffle(families)

    cases = []

    used_seeds = set()

    for i, family in enumerate(
        families, start=1
    ):

        family = str(family)

        seed = int(
            rng.integers(
                2_120_000_000,
                2_130_000_000,
            )
        )

        if seed in used_seeds:
            raise ValueError(
                "Duplicate simulation seed"
            )

        used_seeds.add(seed)

        separation = int(
            rng.choice([2, 4, 6, 8, 12])
        )

        ratio = float(
            rng.choice([0.1, 0.25, 0.5, 1.0])
        )

        fractions = None

        n = None

        gradient = 0.0

        if family in (
            "achromatic", "chromatic"
        ):

            middle = rng.uniform(
                0.25, 0.75, size=2
            )

            if family == "achromatic":

                fractions = np.repeat(
                    middle[:, None],
                    3,
                    axis=1,
                )

            else:

                # Sample color-gradient magnitude
                # independently of separation and flux ratio.
                magnitude = rng.uniform(
                    0.04, 0.23, size=2
                )

                sign = rng.choice(
                    [-1.0, 1.0], size=2
                )

                slope = magnitude * sign

                fractions = np.column_stack([
                    middle - slope,
                    middle,
                    middle + slope,
                ])

                gradient = float(
                    magnitude.mean()
                )

        elif family != "gaussian":

            n = float(
                family.replace("sersic", "")
            )

        else:

            n = 0.5

        cases.append(dict(
            scene_id=i,
            seed=seed,
            split="validation",
            morphology=family,
            separation=separation,
            ratio=ratio,
            sersic_n=n,
            fractions=fractions,
            gradient_strength=gradient,
        ))

    return cases


def make_scene(case):

    if case["morphology"] in (
        "achromatic", "chromatic"
    ):

        config = create_configuration(
            case["separation"],
            case["ratio"],
            case["seed"],
        )

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

    return generate_observation(case)


# ---------------------------------------------------------
# Integrated observation-side diagnostics
# ---------------------------------------------------------

def assess_initial(scene, initial, theta, success):

    start = perf_counter()

    evidence = chromatic_evidence(
        scene,
        theta,
    )

    evidence_seconds = (
        perf_counter() - start
    )

    np.testing.assert_allclose(
        evidence["flux"],
        initial,
        rtol=5e-5,
        atol=1e-5,
    )

    start = perf_counter()

    uncertainty = uncertainty_features(
        scene["observed"],
        scene["psfs"],
        scene["positions"],
        scene["noise_sigma"],
        theta,
    )

    uncertainty_seconds = (
        perf_counter() - start
    )

    np.testing.assert_allclose(
        uncertainty["flux"],
        initial,
        rtol=5e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        evidence["chi2"],
        uncertainty["chi2"],
        rtol=1e-5,
        atol=1e-8,
    )

    scales = uncertainty["scales"]

    relative = scales["relative"]

    reliability = (
        scales["proposed"] / relative
    )

    if not np.isfinite(reliability).all():
        raise ValueError(
            "Invalid reliability scale"
        )

    score = float(
        evidence["evidence_score"]
    )

    evidence_valid = bool(
        evidence["evidence_valid"]
        and np.isfinite(score)
        and success
    )

    uncertainty_valid = bool(
        np.all(
            uncertainty["chromatic_valid"]
        )
    )

    # Invalid initial solutions are not silently
    # classified as low-risk observations.
    forced = not evidence_valid

    eligible = bool(
        forced
        or score > THRESHOLD
    )

    return dict(
        evidence=evidence,
        uncertainty=uncertainty,

        evidence_score=score,

        reliability_score=float(
            reliability.max()
        ),

        covariance_score=float(
            (
                scales["chromatic"]
                / relative
            ).max()
        ),

        evidence_valid=evidence_valid,

        uncertainty_valid=uncertainty_valid,

        forced_refinement=forced,

        eligible=eligible,

        evidence_seconds=evidence_seconds,

        uncertainty_seconds=uncertainty_seconds,
    )


# ---------------------------------------------------------
# Counterfactual nonlinear refinement
# ---------------------------------------------------------

def run_refinement(
    scene,
    theta,
    initial,
    initial_chi2,
):

    try:

        result = fit_chromatic(
            scene["observed"],
            scene["psfs"],
            scene["positions"],
            scene["noise_sigma"],
            theta,
            initial,
        )

    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:

        result = dict(
            flux=initial.copy(),
            chi2=initial_chi2,
            converged=False,
            accepted=False,
            iterations=0,
            message=str(exc),
            parameters=np.full(8, np.nan),
        )

    return result


def post_fit_gate(
    scene,
    initial,
    initial_chi2,
    refined,
):
    """Require numerical validity and model evidence."""

    actual_evidence = (
        initial_chi2
        - float(refined["chi2"])
    ) * scene["observed"].size

    candidate = np.asarray(
        refined["flux"],
        dtype=float,
    )

    valid = bool(
        refined["converged"]
        and refined["accepted"]
        and np.isfinite(actual_evidence)
        and candidate.shape == initial.shape
        and np.isfinite(candidate).all()
        and np.all(candidate >= 0)
    )

    accepted = bool(
        valid
        and actual_evidence > THRESHOLD
    )

    return actual_evidence, accepted


# ---------------------------------------------------------
# Evaluate one independent observation
# ---------------------------------------------------------

def evaluate_scene(case):

    scene = make_scene(case)

    observed = scene["observed"]

    if tuple(
        str(b).lower()
        for b in scene["bands"]
    ) != BANDS:

        raise ValueError(
            "Unexpected band order"
        )

    start = perf_counter()

    initial, _, diagnostics = (
        fit_flexible_sersic(
            observed,
            scene["psfs"],
            scene["positions"],
            scene["noise_sigma"],
        )
    )

    initial_seconds = (
        perf_counter() - start
    )

    initial_success = bool(
        diagnostics["optimizer_success"]
    )

    theta = np.stack([
        shape_vector(shape)
        for shape in diagnostics["shapes"]
    ])

    assessment = assess_initial(
        scene,
        initial,
        theta,
        initial_success,
    )

    initial_chi2 = float(
        assessment["evidence"]["chi2"]
    )

    # The observation-only routing decision is
    # fixed before the counterfactual fit.
    eligible = assessment["eligible"]

    start = perf_counter()

    refined = run_refinement(
        scene,
        theta,
        initial,
        initial_chi2,
    )

    refinement_seconds = (
        perf_counter() - start
    )

    actual_evidence, post_accepted = (
        post_fit_gate(
            scene,
            initial,
            initial_chi2,
            refined,
        )
    )

    raw_refined = np.asarray(
        refined["flux"]
    )

    if (
        raw_refined.shape != initial.shape
        or not np.isfinite(raw_refined).all()
        or np.any(raw_refined < 0)
    ):

        raise RuntimeError(
            "Invalid refinement flux output"
        )

    gated_flux = (
        raw_refined
        if post_accepted
        else initial
    )

    integrated_flux = (
        gated_flux
        if eligible
        else initial
    )

    # Ground truth is accessed only after all
    # observation-side decisions are complete.
    truth = scene["input_flux"]

    if (
        truth.shape != (2, 3)
        or not np.isfinite(truth).all()
        or np.any(truth <= 0)
    ):

        raise ValueError(
            "Invalid simulation truth"
        )

    initial_error = np.abs(
        initial / truth - 1.0
    )

    refined_error = np.abs(
        raw_refined / truth - 1.0
    )

    gated_error = np.abs(
        gated_flux / truth - 1.0
    )

    integrated_error = np.abs(
        integrated_flux / truth - 1.0
    )

    fractions = (
        None
        if case["fractions"] is None
        else case["fractions"].tolist()
    )

    uncertainty = assessment["uncertainty"]

    scales = uncertainty["scales"]

    evidence = assessment["evidence"]

    rows = []

    for s in range(2):

        for b, band in enumerate(BANDS):

            rows.append(dict(
                scene_id=case["scene_id"],
                seed=case["seed"],

                morphology=case["morphology"],

                separation_px=case["separation"],

                flux_ratio_f277w=case["ratio"],

                gradient_strength=case[
                    "gradient_strength"
                ],

                fractions=json.dumps(fractions),

                source=s + 1,
                band=band,

                true_flux=truth[s, b],

                initial_flux=initial[s, b],

                refined_flux=raw_refined[s, b],

                gated_flux=gated_flux[s, b],

                integrated_flux=integrated_flux[s, b],

                initial_error=initial_error[s, b],

                refined_error=refined_error[s, b],

                gated_error=gated_error[s, b],

                integrated_error=integrated_error[s, b],

                initial_chi2=initial_chi2,

                refined_chi2=refined["chi2"],

                pixel_count=observed.size,

                evidence_score=assessment[
                    "evidence_score"
                ],

                actual_evidence=actual_evidence,

                threshold=THRESHOLD,

                evidence_valid=assessment[
                    "evidence_valid"
                ],

                uncertainty_valid=assessment[
                    "uncertainty_valid"
                ],

                null_rank=evidence["null_rank"],

                chromatic_rank=evidence[
                    "chromatic_rank"
                ],

                flux_boundary=evidence[
                    "flux_boundary"
                ],

                shape_boundary=evidence[
                    "shape_boundary"
                ],

                forced_refinement=assessment[
                    "forced_refinement"
                ],

                route_evidence=eligible,

                post_accepted=post_accepted,

                integrated_accepted=bool(
                    eligible and post_accepted
                ),

                initial_converged=initial_success,

                refinement_converged=bool(
                    refined["converged"]
                ),

                refinement_accepted=bool(
                    refined["accepted"]
                ),

                refinement_iterations=refined[
                    "iterations"
                ],

                refinement_message=refined[
                    "message"
                ],

                reliability_score=assessment[
                    "reliability_score"
                ],

                covariance_score=assessment[
                    "covariance_score"
                ],

                initial_scale_relative=scales[
                    "relative"
                ][s, b],

                initial_scale_fixed=scales[
                    "fixed"
                ][s, b],

                initial_scale_chromatic=scales[
                    "chromatic"
                ][s, b],

                initial_scale_proposed=scales[
                    "proposed"
                ][s, b],

                initial_ambiguity=uncertainty[
                    "ambiguity"
                ][s, b],

                initial_theta=json.dumps(
                    theta.tolist()
                ),

                refinement_parameters=json.dumps(
                    [
                        float(x)
                        if np.isfinite(x)
                        else None

                        for x in refined[
                            "parameters"
                        ]
                    ]
                ),

                initial_seconds=initial_seconds,

                evidence_seconds=assessment[
                    "evidence_seconds"
                ],

                uncertainty_seconds=assessment[
                    "uncertainty_seconds"
                ],

                refinement_seconds=refinement_seconds,
            ))

    print(
        f"Scene {case['scene_id']:2d} | "
        f"{case['morphology']:10s} | "
        f"initial={initial_error.mean()*100:6.2f}% | "
        f"refined={refined_error.mean()*100:6.2f}% | "
        f"final={integrated_error.mean()*100:6.2f}% | "
        f"local={assessment['evidence_score']:.3f} | "
        f"actual={actual_evidence:.3f} | "
        f"route={eligible} | "
        f"accepted={post_accepted} | "
        f"refinement={refinement_seconds:.2f}s",
        flush=True,
    )

    return rows


# ---------------------------------------------------------
# Observation-only budget allocation
# ---------------------------------------------------------

def as_bool(value):

    return (
        str(value).strip().lower()
        == "true"
    )


def rank_budget(
    scores,
    allowed,
    forced,
    budget,
):
    """Rank candidates without using ground truth."""

    scores = np.asarray(
        scores, dtype=float
    )

    allowed = np.asarray(
        allowed, dtype=bool
    )

    forced = np.asarray(
        forced, dtype=bool
    )

    clean = np.where(
        np.isfinite(scores),
        scores,
        -np.inf,
    )

    candidates = np.flatnonzero(
        allowed
    )

    # Invalid initial/evidence diagnostics receive
    # priority under every compared budget policy.
    order = sorted(
        candidates,
        key=lambda i: (
            not forced[i],
            -clean[i],
            i,
        ),
    )

    route = np.zeros(
        len(scores), dtype=bool
    )

    route[
        order[:budget]
    ] = True

    return route


# ---------------------------------------------------------
# Matched evaluation
# ---------------------------------------------------------

def summarize(rows):

    groups = sorted(
        grouped_rows(rows).values(),

        key=lambda g: int(
            g[0]["scene_id"]
        ),
    )

    count = len(groups)

    def number(key):

        return np.array([
            float(g[0][key])
            for g in groups
        ])

    def flag(key):

        return np.array([
            as_bool(g[0][key])
            for g in groups
        ])

    def measurements(key):

        return np.array([
            [
                float(r[key])
                for r in g
            ]

            for g in groups
        ])

    initial = measurements(
        "initial_error"
    )

    refined = measurements(
        "refined_error"
    )

    gated = measurements(
        "gated_error"
    )

    integrated_recorded = measurements(
        "integrated_error"
    )

    eligible = flag(
        "route_evidence"
    )

    accepted = flag(
        "post_accepted"
    )

    forced = flag(
        "forced_refinement"
    )

    evidence_score = number(
        "evidence_score"
    )

    reliability_score = number(
        "reliability_score"
    )

    covariance_score = number(
        "covariance_score"
    )

    all_scenes = np.ones(
        count, dtype=bool
    )

    no_scenes = np.zeros(
        count, dtype=bool
    )

    budget = min(
        MAX_BUDGET,
        int(eligible.sum()),
    )

    # Both primary priority methods use the
    # identical evidence-eligible candidate pool.
    evidence_ranked = rank_budget(
        evidence_score,
        eligible,
        forced,
        budget,
    )

    evidence_uncertainty = rank_budget(
        reliability_score,
        eligible,
        forced,
        budget,
    )

    uncertainty_only = rank_budget(
        reliability_score,
        all_scenes,
        forced,
        budget,
    )

    covariance_only = rank_budget(
        covariance_score,
        all_scenes,
        forced,
        budget,
    )

    chi2_only = rank_budget(
        number("initial_chi2"),
        all_scenes,
        forced,
        budget,
    )

    # Oracle is an evaluation-only upper reference.
    benefit = (
        initial.mean(axis=1)
        - gated.mean(axis=1)
    )

    oracle = rank_budget(
        benefit,
        eligible,
        no_scenes,
        budget,
    )

    # Strategy configuration:
    # route, use post-gate, evidence cost, uncertainty cost
    strategies = {
        "initial_only": (
            no_scenes, False, False, False
        ),

        "always_refine": (
            all_scenes, False, False, False
        ),

        "post_gate_only": (
            all_scenes, True, False, False
        ),

        "evidence_adaptive": (
            eligible, True, True, False
        ),

        "uncertainty_only": (
            uncertainty_only, True, True, True
        ),

        "covariance_only": (
            covariance_only, True, True, True
        ),

        "chi2_only": (
            chi2_only, True, True, False
        ),

        "evidence_ranked": (
            evidence_ranked, True, True, False
        ),

        "evidence_uncertainty": (
            evidence_uncertainty, True, True, True
        ),

        "oracle_within_evidence": (
            oracle, True, True, False
        ),
    }

    family = np.array([
        g[0]["morphology"]
        for g in groups
    ])

    achromatic = family == "achromatic"

    chromatic = family == "chromatic"

    single = ~(
        achromatic | chromatic
    )

    faint = np.array([
        [
            int(r["source"]) == 2
            and float(r["flux_ratio_f277w"]) == 0.1

            for r in g
        ]

        for g in groups
    ])

    initial_time = number(
        "initial_seconds"
    )

    evidence_time = number(
        "evidence_seconds"
    )

    uncertainty_time = number(
        "uncertainty_seconds"
    )

    refinement_time = number(
        "refinement_seconds"
    )

    np.testing.assert_allclose(
        integrated_recorded,

        np.where(
            eligible[:, None],
            gated,
            initial,
        ),

        rtol=1e-10,
        atol=1e-12,
    )

    def safe_mean(values):

        return (
            float(np.mean(values))
            if values.size
            else np.nan
        )

    # Remove the same largest-runtime scene from
    # every strategy for a sensitivity analysis.
    timing_keep = np.ones(
        count, dtype=bool
    )

    timing_keep[
        int(np.argmax(refinement_time))
    ] = False

    print("\n--- Experiment inventory ---")

    print(
        f"Scenes: {count}/{N_SCENES}"
    )

    print(
        f"Evidence eligible: "
        f"{eligible.sum()}/{count}"
    )

    print(
        f"Invalid evidence diagnostics: "
        f"{forced.sum()}"
    )

    print(
        f"Invalid uncertainty diagnostics: "
        f"{(~flag('uncertainty_valid')).sum()}"
    )

    print(
        f"Post-fit accepted: "
        f"{accepted.sum()}/{count}"
    )

    print(
        f"Missed post-fit evidence: "
        f"{np.sum(accepted & ~eligible)}"
    )

    print(
        f"Limited refinement budget: "
        f"{budget}"
    )

    print(
        f"Initial convergence: "
        f"{flag('initial_converged').sum()}/{count}"
    )

    print(
        f"Refinement convergence: "
        f"{flag('refinement_converged').sum()}/{count}"
    )

    print("\n--- Morphology screening ---")

    for name, mask in (
        ("single", single),
        ("achromatic", achromatic),
        ("chromatic", chromatic),
    ):

        print(
            f"{name}: "
            f"{np.sum(mask & eligible)}/{mask.sum()} "
            f"eligible; "
            f"{np.sum(mask & eligible & accepted)} "
            f"accepted"
        )

    print("\n--- Strategy comparison ---")

    results = {}

    for name, config in strategies.items():

        route, use_gate, use_evidence, use_uq = (
            config
        )

        alternative = (
            gated
            if use_gate
            else refined
        )

        errors = np.where(
            route[:, None],
            alternative,
            initial,
        )

        cost = (
            initial_time
            + route * refinement_time
        )

        if use_evidence:

            cost += evidence_time

        if use_uq:

            cost += uncertainty_time

        final_accepted = (
            route & accepted

            if use_gate

            else route & flag(
                "refinement_accepted"
            )
        )

        results[name] = dict(
            error=errors.mean(),
            cost=cost.sum(),
            trimmed_cost=cost[
                timing_keep
            ].sum(),
        )

        print(
            f"\n{name}"

            f"\n  Attempts: "
            f"{route.sum()}/{count}"

            f"\n  Accepted: "
            f"{final_accepted.sum()}/{count}"

            f"\n  Mean error: "
            f"{errors.mean()*100:.2f}%"

            f"\n  Median error: "
            f"{np.median(errors)*100:.2f}%"

            f"\n  P90 error: "
            f"{np.percentile(errors,90)*100:.2f}%"

            f"\n  Single-profile error: "
            f"{safe_mean(errors[single])*100:.2f}%"

            f"\n  Achromatic error: "
            f"{safe_mean(errors[achromatic])*100:.2f}%"

            f"\n  Chromatic error: "
            f"{safe_mean(errors[chromatic])*100:.2f}%"

            f"\n  Faint companion error: "
            f"{safe_mean(errors[faint])*100:.2f}%"

            f"\n  Estimated runtime: "
            f"{cost.sum():.2f}s"

            f"\n  Runtime excluding largest refinement scene: "
            f"{cost[timing_keep].sum():.2f}s"
        )

    print("\n--- Priority ablation ---")

    overlap = int(
        np.sum(
            evidence_ranked
            & evidence_uncertainty
        )
    )

    delta = 100 * (
        results["evidence_ranked"]["error"]
        - results["evidence_uncertainty"]["error"]
    )

    print(
        f"Common selected scenes: "
        f"{overlap}/{budget}"
    )

    print(
        f"Uncertainty priority gain: "
        f"{delta:.4f} percentage points"
    )

    if overlap == budget:

        print(
            "Priority ablation is inconclusive: "
            "both strategies selected identical scenes."
        )

    # Guarded random allocation within the
    # identical evidence-eligible candidate pool.
    forced_order = np.flatnonzero(
        forced & eligible
    )

    forced_random = np.zeros(
        count, dtype=bool
    )

    forced_random[
        forced_order[:budget]
    ] = True

    remaining = eligible & ~forced

    remaining_budget = (
        budget - forced_random.sum()
    )

    probability = (
        remaining_budget / remaining.sum()

        if remaining.sum() > 0

        else 0.0
    )

    random_error = (
        initial.mean(axis=1).sum()
        - benefit[forced_random].sum()
        - probability * benefit[remaining].sum()
    ) / count

    print(
        f"Expected guarded random error: "
        f"{random_error*100:.2f}%"
    )

    print("\n--- Robustness check ---")

    improvement = (
        initial.mean(axis=1)
        - integrated_recorded.mean(axis=1)
    )

    keep = np.ones(
        count, dtype=bool
    )

    largest_benefit = int(
        np.argmax(improvement)
    )

    keep[largest_benefit] = False

    print(
        f"Largest-benefit scene: "
        f"{groups[largest_benefit][0]['scene_id']}"
    )

    print(
        f"Initial error without scene: "
        f"{initial[keep].mean()*100:.2f}%"
    )

    print(
        f"Integrated error without scene: "
        f"{integrated_recorded[keep].mean()*100:.2f}%"
    )


# ---------------------------------------------------------
# Resume-safe result storage
# ---------------------------------------------------------

def load_completed(cases):

    if not OUTPUT.exists():

        return []

    with OUTPUT.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        rows = list(
            csv.DictReader(stream)
        )

    expected = {
        c["scene_id"]: c
        for c in cases
    }

    groups = grouped_rows(rows)

    for scene_id, group in groups.items():

        if scene_id not in expected:

            raise ValueError(
                "Unexpected saved scene"
            )

        case = expected[scene_id]

        for row in group:

            if (
                int(row["seed"]) != case["seed"]

                or row["morphology"]
                != case["morphology"]

                or float(row["separation_px"])
                != case["separation"]

                or float(row["flux_ratio_f277w"])
                != case["ratio"]

                or float(row["threshold"])
                != THRESHOLD
            ):

                raise ValueError(
                    "Saved experiment mismatch"
                )

    return rows


def save_completed(rows):

    # Atomic replacement prevents a partially
    # written scene from corrupting completed results.
    temporary = OUTPUT.with_suffix(
        ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()

        writer.writerows(rows)

    temporary.replace(OUTPUT)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most this many new scenes. "
            "Completed scenes remain saved."
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):

        parser.error(
            "--limit must be positive"
        )

    cases = generate_cases()

    rows = load_completed(cases)

    completed = set(
        grouped_rows(rows)
    )

    print(
        "Phase 10.4 — "
        "Integrated evidence-guided adaptive deblending"
    )

    print(
        f"Frozen threshold: {THRESHOLD:.6f}"
    )

    print(
        f"Planned scenes: {N_SCENES}"
    )

    print(
        f"Previously completed: {len(completed)}"
    )

    print(
        f"Remaining scenes: "
        f"{N_SCENES-len(completed)}"
    )

    print(
        f"Expected measurements: {N_SCENES*6}",
        flush=True,
    )

    processed = 0

    for case in cases:

        if case["scene_id"] in completed:

            continue

        if (
            args.limit is not None
            and processed >= args.limit
        ):

            break

        batch = evaluate_scene(case)

        grouped_rows(batch)

        rows.extend(batch)

        save_completed(rows)

        processed += 1

    if len(rows) == N_SCENES * 6:

        summarize(rows)

        print(
            f"\nCompleted scenes: {N_SCENES}"
        )

    else:

        print(
            f"\nSaved scenes: "
            f"{len(rows)//6}/{N_SCENES}"
        )

        print(
            "Run the same command to resume."
        )

    print(
        f"Results saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
