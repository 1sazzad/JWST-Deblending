
"""Phase 10.2: uncertainty-guided adaptive deblending.

Use frozen Phase 10.1 uncertainty features to control
a smooth wavelength-dependent nonlinear refinement.

All candidate refinements are evaluated offline to
measure paired counterfactual outcomes.

This experiment does not establish calibrated coverage
for the final adaptive estimator.
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.fft import rfftn, irfftn
from scipy.optimize import minimize

from src.baselines.baseline_flexible_gaussian import (
    profile_linear,
    MIN_SIGMA,
    MAX_SIGMA,
)

from src.baselines.baseline_flexible_sersic import (
    SersicObjective,
    fit_flexible_sersic,
    MIN_N,
    MAX_N,
)

from src.evaluation.benchmark_risk_refinement import (
    shape_vector,
)

from src.evaluation.benchmark_uncertainty_calibration import (
    uncertainty_features,
    generate_observation,
    grouped_rows,
    conformal_multiplier,
)


# ---------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results"

REFERENCE = (
    RESULTS
    / "benchmark_uncertainty_calibration_phase10_1.csv"
)

OUTPUT = (
    RESULTS
    / "benchmark_uncertainty_adaptive_phase10_2.csv"
)

N_SCENES = 16

RANDOM_SEED = 5102026

MAX_ITERATIONS = 100

BANDS = (
    "f150w",
    "f277w",
    "f444w",
)

# Nominal filter wavelengths in microns.
WAVELENGTHS = np.array([
    1.50,
    2.77,
    4.44,
])

log_wave = np.log(WAVELENGTHS)

t = (
    log_wave - log_wave[0]
) / (
    log_wave[-1] - log_wave[0]
)

# Interpolate morphology parameters linearly
# in log wavelength.
H = np.column_stack([
    1 - t,
    t,
])


# ---------------------------------------------------------
# Smooth chromatic nonlinear objective
# ---------------------------------------------------------

class SmoothChromaticObjective:
    """Optimize radial width and concentration across bands.

    Each source has two wavelength endpoints for
    radial log-width and two for log-Sersic index.

    Axis ratio, orientation and position remain fixed
    at the initial shared-morphology solution.
    """

    def __init__(
        self,
        objective,
        theta,
    ):

        self.obj = objective

        self.theta = np.asarray(
            theta,
            dtype=float,
        )

    def evaluate(
        self,
        x,
        details=False,
    ):

        # Source x shape variable x wavelength endpoint.
        parameters = x.reshape(
            2, 2, 2
        )

        increments = np.einsum(
            "be,ske->bsk",
            H,
            parameters,
        )

        gradient = np.zeros_like(
            parameters
        )

        flux = np.empty(
            (2, 3)
        )

        model = np.empty_like(
            self.obj.observed
        )

        loss = 0.0

        for b in range(3):

            images = []

            for s, quadrature in enumerate(
                self.obj.quad
            ):

                theta = self.theta[s].copy()

                # Radial width change preserves
                # the original elliptical axis ratio.
                theta[:2] += increments[b, s, 0]

                theta[3] += increments[b, s, 1]

                z = quadrature.evaluate(
                    theta
                )

                # Exact normalized morphology
                # and analytical shape derivatives.
                profiles = np.stack([
                    z[0],
                    z[1] + z[2],
                    z[4],
                ])

                transformed = rfftn(
                    profiles,
                    s=self.obj.fft_shape,
                    axes=(-2, -1),
                )

                blurred = irfftn(
                    transformed
                    * self.obj.psf_fft[b],

                    s=self.obj.fft_shape,

                    axes=(-2, -1),
                )

                images.append(
                    blurred[
                        ...,
                        self.obj.crop[0],
                        self.obj.crop[1],
                    ]
                )

            z = np.stack(
                images
            )

            # Exact nonnegative flux estimation.
            flux[:, b], _, model[b] = (
                profile_linear(
                    z[:, 0],
                    self.obj.observed[b],
                )
            )

            residual = (
                model[b]
                - self.obj.observed[b]
            )

            weighted = (
                residual
                / self.obj.noise[b]**2
            )

            loss += 0.5 * np.sum(
                residual * weighted
            )

            # Envelope theorem for the
            # profiled linear flux solution.
            g = np.sum(
                z[:, 1:]
                * weighted[None, None]
                * flux[:, b, None, None, None],

                axis=(-2, -1),
            )

            gradient += (
                g[:, :, None]
                * H[b, None, None, :]
            )

        if details:

            return (
                flux,
                model,
                float(loss),
            )

        return (
            float(loss),
            gradient.ravel(),
        )


def fit_chromatic(
    observed,
    psfs,
    positions,
    noise,
    theta,
    initial_flux,
):
    """Run bounded nonlinear chromatic refinement."""

    objective = SersicObjective(
        observed,
        psfs,
        positions,
        noise,
    )

    smooth = SmoothChromaticObjective(
        objective,
        theta,
    )

    bounds = []

    for lx, ly, _, ln in theta:

        # Common radial shift must preserve
        # both physical width bounds.
        lo_width = max(
            np.log(MIN_SIGMA) - lx,
            np.log(MIN_SIGMA) - ly,
        )

        hi_width = min(
            np.log(MAX_SIGMA) - lx,
            np.log(MAX_SIGMA) - ly,
        )

        lo_n = np.log(MIN_N) - ln

        hi_n = np.log(MAX_N) - ln

        # Tolerate floating-point rounding
        # at an existing parameter boundary.
        bounds.extend([
            (
                min(lo_width, 0.0),
                max(hi_width, 0.0),
            )
        ] * 2)

        bounds.extend([
            (
                min(lo_n, 0.0),
                max(hi_n, 0.0),
            )
        ] * 2)

    x0 = np.zeros(8)

    base_flux, _, base_loss = (
        smooth.evaluate(
            x0,
            details=True,
        )
    )

    # Zero increments must reproduce the
    # existing shared-morphology point estimate.
    np.testing.assert_allclose(
        base_flux,
        initial_flux,
        rtol=5e-5,
        atol=1e-5,
    )

    result = minimize(
        smooth.evaluate,

        x0,

        jac=True,

        method="L-BFGS-B",

        bounds=bounds,

        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 30,
        },
    )

    accepted = False

    flux = base_flux.copy()

    loss = base_loss

    if (
        result.success
        and np.isfinite(result.fun)
        and np.isfinite(result.x).all()
    ):

        candidate, model, candidate_loss = (
            smooth.evaluate(
                result.x,
                details=True,
            )
        )

        # Physical and numerical acceptance guard.
        if (
            np.isfinite(candidate).all()
            and np.isfinite(model).all()
            and np.all(candidate >= 0)
            and candidate_loss <= base_loss
        ):

            flux = candidate

            loss = candidate_loss

            accepted = True

    return dict(
        flux=flux,

        chi2=2 * loss / observed.size,

        converged=bool(result.success),

        accepted=accepted,

        iterations=int(result.nit),

        message=str(result.message),

        parameters=result.x.copy(),
    )


# ---------------------------------------------------------
# Freeze the controller using calibration data only
# ---------------------------------------------------------

def load_controller():
    """Determine observation-side reference thresholds."""

    with REFERENCE.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        archive = list(
            csv.DictReader(stream)
        )

    calibration = [
        row
        for row in archive
        if row["split"] == "calibration"
    ]

    groups = list(
        grouped_rows(
            calibration
        ).values()
    )

    if len(groups) != 39:

        raise ValueError(
            "Expected 39 calibration scenes"
        )

    q95 = conformal_multiplier(
        groups,
        "proposed",
        0.95,
    )

    values = []

    for group in groups:

        relative = np.array([
            float(r["scale_relative"])
            for r in group
        ])

        proposed = np.array([
            float(r["scale_proposed"])
            for r in group
        ])

        chromatic = np.array([
            float(r["scale_chromatic"])
            for r in group
        ])

        ambiguity = np.array([
            float(r["linear_ambiguity"])
            for r in group
        ])

        if not all(
            str(r["chromatic_valid"]).lower()
            == "true"
            for r in group
        ):

            raise ValueError(
                "Unexpected invalid calibration diagnostic"
            )

        U = q95 * proposed / relative

        D = ambiguity / chromatic

        if (
            not np.isfinite(U).all()
            or not np.isfinite(D).all()
        ):

            raise ValueError(
                "Invalid calibration features"
            )

        values.append(
            (U, D)
        )

    u0 = float(np.median([
        U.max()
        for U, D in values
    ]))

    d0 = float(np.median([
        D.max()
        for U, D in values
    ]))

    if min(u0, d0) <= 0:

        raise ValueError(
            "Invalid controller reference scales"
        )

    priorities = [
        np.minimum(
            U / u0,
            D / d0,
        ).max()

        for U, D in values
    ]

    threshold = float(
        np.median(priorities)
    )

    parameters = dict(
        q95=q95,
        u0=u0,
        d0=d0,
        threshold=threshold,
    )

    # Only seed metadata is accessed from
    # the complete archive to avoid duplication.
    used_seeds = {
        int(r["seed"])
        for r in archive
    }

    return (
        parameters,
        used_seeds,
    )


def controller_features(
    result,
    controller,
):
    """Calculate adaptive priority and conventional scores."""

    scales = result["scales"]

    relative = scales["relative"]

    proposed = scales["proposed"]

    chromatic = scales["chromatic"]

    ambiguity = result["ambiguity"]

    valid = bool(
        np.all(result["chromatic_valid"])
        and np.isfinite(ambiguity).all()
    )

    U = (
        controller["q95"]
        * proposed
        / relative
    )

    # Invalid local systems are retained and
    # flagged for review rather than corrected blindly.
    if not valid:

        return dict(
            priority=-1.0,
            uncertainty_score=float(U.max()),
            ambiguity_score=-1.0,
            ambiguity_noise_score=-1.0,
            covariance_score=float(
                (chromatic / relative).max()
            ),
            chi2_score=result["chi2"],
            diagnostic_valid=False,
            route_proposed=False,
        )

    D = ambiguity / chromatic

    priority = float(
        np.minimum(
            U / controller["u0"],
            D / controller["d0"],
        ).max()
    )

    return dict(
        priority=priority,

        uncertainty_score=float(
            U.max()
        ),

        ambiguity_score=float(
            (ambiguity / relative).max()
        ),

        ambiguity_noise_score=float(
            D.max()
        ),

        covariance_score=float(
            (chromatic / relative).max()
        ),

        chi2_score=result["chi2"],

        diagnostic_valid=True,

        route_proposed=bool(
            priority >= controller["threshold"]
        ),
    )


# ---------------------------------------------------------
# Generate new pilot scenes
# ---------------------------------------------------------

def generate_cases(used_seeds):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    cases = []

    for i in range(N_SCENES):

        # Separate seed range from Phase 10.1.
        seed = int(
            rng.integers(
                1_000_000_000,
                2_000_000_000,
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
            split="pilot",
            seed=seed,
            separation=separation,
            ratio=ratio,
            morphology=family,
            sersic_n=n,
            fractions=fractions,
        ))

    return cases


# ---------------------------------------------------------
# Matched candidate evaluation
# ---------------------------------------------------------

def evaluate_case(
    case,
    controller,
):

    scene = generate_observation(
        case
    )

    observed = scene["observed"]
    psfs = scene["psfs"]
    positions = scene["positions"]
    noise = scene["noise_sigma"]

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
            psfs,
            positions,
            noise,
        )
    )

    initial_seconds = (
        perf_counter() - start
    )

    theta = np.stack([
        shape_vector(shape)
        for shape in diagnostics["shapes"]
    ])

    start = perf_counter()

    uncertainty = uncertainty_features(
        observed,
        psfs,
        positions,
        noise,
        theta,
    )

    np.testing.assert_allclose(
        uncertainty["flux"],
        initial,
        rtol=5e-5,
        atol=1e-5,
    )

    features = controller_features(
        uncertainty,
        controller,
    )

    diagnostic_seconds = (
        perf_counter() - start
    )

    # Always run the counterfactual alternative.
    # Its result does not influence the routing score.
    start = perf_counter()

    try:

        refinement = fit_chromatic(
            observed,
            psfs,
            positions,
            noise,
            theta,
            initial,
        )

    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:

        refinement = dict(
            flux=initial.copy(),

            chi2=uncertainty["chi2"],

            converged=False,

            accepted=False,

            iterations=0,

            message=str(exc),

            parameters=np.full(
                8,
                np.nan,
            ),
        )

    refinement_seconds = (
        perf_counter() - start
    )

    refined = refinement["flux"]

    selected = features[
        "route_proposed"
    ]

    adaptive = (
        refined
        if selected
        else initial
    )

    # Ground truth enters only the evaluation.
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
        initial / truth - 1
    )

    refined_error = np.abs(
        refined / truth - 1
    )

    adaptive_error = np.abs(
        adaptive / truth - 1
    )

    rows = []

    for s in range(2):

        for b, band in enumerate(BANDS):

            rows.append(dict(
                scene_id=case["scene_id"],

                seed=case["seed"],

                morphology=case["morphology"],

                separation_px=case["separation"],

                flux_ratio_f277w=case["ratio"],

                source=s + 1,

                band=band,

                true_flux=truth[s, b],

                initial_flux=initial[s, b],

                refined_flux=refined[s, b],

                adaptive_flux=adaptive[s, b],

                initial_error=initial_error[s, b],

                refined_error=refined_error[s, b],

                adaptive_error=adaptive_error[s, b],

                initial_chi2=uncertainty["chi2"],

                refined_chi2=refinement["chi2"],

                refinement_converged=(
                    refinement["converged"]
                ),

                refinement_accepted=(
                    refinement["accepted"]
                ),

                refinement_iterations=(
                    refinement["iterations"]
                ),

                refinement_message=(
                    refinement["message"]
                ),

                initial_converged=bool(
                    diagnostics["optimizer_success"]
                ),

                scale_proposed=(
                    uncertainty["scales"]["proposed"][s, b]
                ),

                scale_chromatic=(
                    uncertainty["scales"]["chromatic"][s, b]
                ),

                scale_relative=(
                    uncertainty["scales"]["relative"][s, b]
                ),

                linear_ambiguity=(
                    uncertainty["ambiguity"][s, b]
                ),

                initial_seconds=initial_seconds,

                diagnostic_seconds=diagnostic_seconds,

                refinement_seconds=refinement_seconds,

                q95=controller["q95"],

                u0=controller["u0"],

                d0=controller["d0"],

                threshold=controller["threshold"],

                **features,
            ))

    print(
        f"Scene {case['scene_id']:2d} | "
        f"{case['morphology']:10s} | "
        f"initial={initial_error.mean()*100:6.2f}% | "
        f"refined={refined_error.mean()*100:6.2f}% | "
        f"adaptive={adaptive_error.mean()*100:6.2f}% | "
        f"priority={features['priority']:.3f} | "
        f"route={selected} | "
        f"accepted={refinement['accepted']} | "
        f"refinement={refinement_seconds:.2f}s",
        flush=True,
    )

    return rows


# ---------------------------------------------------------
# Equal-budget routing comparison
# ---------------------------------------------------------

def summarize(rows):

    groups = list(
        grouped_rows(rows).values()
    )

    count = len(groups)

    initial = np.array([
        np.mean([
            r["initial_error"]
            for r in g
        ])
        for g in groups
    ])

    refined = np.array([
        np.mean([
            r["refined_error"]
            for r in g
        ])
        for g in groups
    ])

    benefit = initial - refined

    proposed = np.array([
        g[0]["route_proposed"]
        for g in groups
    ], dtype=bool)

    budget = int(
        proposed.sum()
    )

    def top_budget(scores):

        scores = np.asarray(
            scores,
            dtype=float,
        )

        order = np.argsort(
            -scores,
            kind="stable",
        )

        selected = np.zeros(
            count,
            dtype=bool,
        )

        selected[
            order[:budget]
        ] = True

        return selected

    strategies = {
        "initial_only": np.zeros(
            count,
            dtype=bool,
        ),

        "always_refine": np.ones(
            count,
            dtype=bool,
        ),

        "proposed": proposed,
    }

    for name, key in (
        ("chi2", "chi2_score"),

        ("covariance", "covariance_score"),

        ("uncertainty", "uncertainty_score"),

        ("ambiguity", "ambiguity_score"),

        ("ambiguity_noise", "ambiguity_noise_score"),
    ):

        strategies[name] = top_budget([
            g[0][key]
            for g in groups
        ])

    # Ground-truth oracle is evaluation only.
    strategies["oracle"] = (
        top_budget(benefit)
    )

    chromatic = np.array([
        g[0]["morphology"] == "chromatic"
        for g in groups
    ])

    initial_cost = np.array([
        g[0]["initial_seconds"]
        for g in groups
    ])

    diagnostic_cost = np.array([
        g[0]["diagnostic_seconds"]
        for g in groups
    ])

    refinement_cost = np.array([
        g[0]["refinement_seconds"]
        for g in groups
    ])

    print(
        f"\nProposed refinement budget: "
        f"{budget}/{count}"
    )

    print(
        f"Expected random routing error: "
        f"{100*(initial.mean() - budget/count*benefit.mean()):.2f}%"
    )

    print(
        "\n--- Matched comparison ---"
    )

    for name, selected in strategies.items():

        errors = np.where(
            selected,
            refined,
            initial,
        )

        single_error = (
            errors[~chromatic].mean()
            if np.any(~chromatic)
            else np.nan
        )

        chromatic_error = (
            errors[chromatic].mean()
            if np.any(chromatic)
            else np.nan
        )

        cost = (
            initial_cost.sum()
            + refinement_cost[selected].sum()
        )

        # Charge the full measured diagnostic
        # cost to uncertainty-based strategies.
        if name not in (
            "initial_only",
            "always_refine",
            "chi2",
            "oracle",
        ):

            cost += diagnostic_cost.sum()

        faint = []

        for i, group in enumerate(groups):

            for row in group:

                if (
                    row["source"] == 2
                    and row["flux_ratio_f277w"]
                    == 0.1
                ):

                    key = (
                        "refined_error"
                        if selected[i]
                        else "initial_error"
                    )

                    faint.append(
                        row[key]
                    )

        faint_error = (
            np.mean(faint)
            if faint
            else np.nan
        )

        print(
            f"\n{name}"
            f"\n  Selected: {selected.sum()}/{count}"
            f"\n  Overall error: {errors.mean()*100:.2f}%"
            f"\n  Single-profile error: {single_error*100:.2f}%"
            f"\n  Chromatic error: {chromatic_error*100:.2f}%"
            f"\n  Faint companion error: {faint_error*100:.2f}%"
            f"\n  Estimated processing cost: {cost:.2f}s"
        )

    converged = sum(
        g[0]["refinement_converged"]
        for g in groups
    )

    accepted = sum(
        g[0]["refinement_accepted"]
        for g in groups
    )

    print(
        f"\nRefinement convergence: "
        f"{converged}/{count}"
    )

    print(
        f"Accepted physical refinements: "
        f"{accepted}/{count}"
    )

    print(
        f"Refinement improved photometry: "
        f"{np.sum(benefit > 0)}/{count}"
    )


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    if OUTPUT.exists():

        raise FileExistsError(
            OUTPUT
        )

    controller, used_seeds = (
        load_controller()
    )

    cases = generate_cases(
        used_seeds
    )

    print(
        "Phase 10.2 — "
        "Uncertainty-guided adaptive refinement"
    )

    print(
        f"Controller: {controller}"
    )

    print(
        f"Planned new scenes: {len(cases)}"
    )

    print(
        f"Expected measurements: {len(cases)*6}",
        flush=True,
    )

    rows = []

    with OUTPUT.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = None

        for case in cases:

            batch = evaluate_case(
                case,
                controller,
            )

            if writer is None:

                writer = csv.DictWriter(
                    stream,

                    fieldnames=list(
                        batch[0]
                    ),
                )

                writer.writeheader()

            writer.writerows(
                batch
            )

            stream.flush()

            rows.extend(
                batch
            )

    summarize(
        rows
    )

    print(
        f"\nCompleted scenes: "
        f"{len(rows)//6}/{N_SCENES}"
    )

    print(
        f"Results saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
