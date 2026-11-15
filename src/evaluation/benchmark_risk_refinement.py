
"""Phase 9.3.1: photometric risk-guided adaptive refinement.

Compare:
    1. Shared-morphology flexible Sersic.
    2. Unregularized local correction.
    3. Fixed ridge regularization.
    4. Pixel-risk-guided correction.
    5. Flux-risk-guided correction.

Reuse archived fitted shapes where available.
Ground truth is used exclusively for evaluation.

Risk estimates are local plug-in approximations.
They are not calibrated photometric uncertainties.
"""

import argparse
import csv

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

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

from src.evaluation.benchmark_flux_sensitivity import (
    regenerate,
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results"

OUTPUT = (
    RESULTS
    / "benchmark_risk_refinement_phase9_3_1.csv"
)

BANDS = (
    "f150w",
    "f277w",
    "f444w",
)

SCENE_IDS = (1, 16, 46, 81, 96)

FAMILIES = (
    "gaussian",
    "sersic1",
    "sersic2",
    "sersic4",
    "chromatic",
)

METHODS = (
    "flexible_sersic",
    "unregularized",
    "fixed_regularization",
    "pixel_risk",
    "flux_risk",
)

# Dimensionless ridge strengths.
# Actual lambda is scaled by the median squared
# singular value of each band's nuisance design.

ALPHAS = (
    0.0,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    np.inf,
)

FIXED_ALPHA = 1.0

# One bounded local correction per band.
TRUST_RADIUS = 0.25

BACKTRACK_FACTORS = (
    1.0,
    0.5,
    0.25,
    0.125,
)

RANK_TOLERANCE = 1e-7


# ---------------------------------------------------------
# Archive loading
# ---------------------------------------------------------

def load_cases():
    """Load matched development scenes and initial fits."""

    groups = defaultdict(list)

    single_path = (
        RESULTS
        / "benchmark_flexible_sersic_phase8_4.csv"
    )

    with single_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        for row in csv.DictReader(stream):

            family = row["morphology"]

            scene_id = int(row["scene_id"])

            if row["status"] != "success":
                raise ValueError(
                    "Invalid archived Sersic fit"
                )

            groups[
                family, scene_id
            ].append(row)

    composite_path = (
        RESULTS
        / "benchmark_composite_morphology_phase9_1.csv"
    )

    with composite_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        for row in csv.DictReader(stream):

            if (
                row["track"] != "chromatic"
                or row["method"]
                != "flexible_sersic"
            ):
                continue

            if row["status"] != "success":
                raise ValueError(
                    "Invalid composite fit"
                )

            groups[
                "chromatic",
                int(row["scene_id"]),
            ].append(row)

    expected = {
        (family, scene_id)
        for family in FAMILIES
        for scene_id in SCENE_IDS
    }

    if set(groups) != expected:
        raise ValueError(
            "Unexpected archived scene inventory"
        )

    # Confirm six unique source-band records.
    for key, rows in groups.items():

        actual = {
            (
                int(row["source"]),
                row["band"].lower(),
            )
            for row in rows
        }

        expected_keys = {
            (s, b)
            for s in (1, 2)
            for b in BANDS
        }

        if (
            len(rows) != 6
            or actual != expected_keys
        ):
            raise ValueError(
                f"Invalid archive group: {key}"
            )

    return groups


def shape_vector(shape):
    """Convert fitted shape metadata to optimizer coordinates."""

    return np.array([
        np.log(float(shape["sigma_x"])),
        np.log(float(shape["sigma_y"])),
        float(shape["rho"]),
        np.log(float(shape["fitted_sersic_n"])),
    ])


def archive_arrays(rows):
    """Recover source-band arrays from archived measurements."""

    lookup = {
        (
            int(row["source"]),
            row["band"].lower(),
        ): row
        for row in rows
    }

    flux = np.array([
        [
            float(
                lookup[s, b]["estimated_flux"]
            )
            for b in BANDS
        ]
        for s in (1, 2)
    ])

    truth = np.array([
        [
            float(
                lookup[s, b]["true_flux"]
            )
            for b in BANDS
        ]
        for s in (1, 2)
    ])

    return flux, truth, lookup


# ---------------------------------------------------------
# Linearized photometric risk
# ---------------------------------------------------------

def center_columns(matrix):
    """Profile out the unrestricted constant background."""

    return matrix - matrix.mean(
        axis=0,
        keepdims=True,
    )


def prepare_band(
    templates,
    derivatives,
    observed,
    noise,
    initial_flux,
):
    """Build the local whitened inverse problem.

    Two flux variables and four morphology variables.

    Each source has:
        - one common radial log-width increment;
        - one log-Sersic-index increment.

    Centroid, orientation and axis ratio remain fixed.
    """

    if np.any(
        initial_flux
        <= 1e-8 * max(
            initial_flux.max(),
            1.0,
        )
    ):
        raise ValueError(
            "Initial flux boundary"
        )

    # X: two unit-flux templates.
    X = templates.reshape(
        2, -1
    ).T / noise

    X = center_columns(X)

    # Z: morphology tangents multiplied by
    # the initial individual source flux.
    Z = derivatives.reshape(
        4, -1
    ).T

    Z = (
        Z
        * np.repeat(initial_flux, 2)
        / noise
    )

    Z = center_columns(Z)

    y = observed.ravel() / noise

    y = y - y.mean()

    # Eliminate the unpenalized flux subspace.
    Q, R = np.linalg.qr(
        X,
        mode="reduced",
    )

    if (
        not np.isfinite(R).all()
        or np.linalg.cond(R) > 1e8
    ):
        raise ValueError(
            "Ill-conditioned flux design"
        )

    A = np.linalg.solve(
        R,
        Q.T,
    )

    # Coupling between morphology and flux.
    C = A @ Z

    # Orthogonalize morphology tangents
    # against the flux subspace.
    Z_perp = Z - Q @ (Q.T @ Z)

    U, singular, Vt = np.linalg.svd(
        Z_perp,
        full_matrices=False,
    )

    if (
        len(singular) != 4
        or singular[0] <= 0
        or singular[-1]
        <= RANK_TOLERANCE * singular[0]
    ):
        raise ValueError(
            "Rank-deficient morphology design"
        )

    reference_scale = float(
        np.median(singular**2)
    )

    B = C @ Vt.T

    # Unregularized local flux operator.
    L_full = (
        A
        - (B / singular) @ U.T
    )

    full_flux = L_full @ y

    baseline_flux = A @ y

    baseline_sigma = np.sqrt(
        np.sum(A*A, axis=1)
    )

    # Observation-derived relative flux weighting.
    # The uncertainty floor prevents divergence
    # for very faint sources.
    scale = np.maximum(
        np.abs(initial_flux),
        baseline_sigma,
    )

    W = 1.0 / np.maximum(
        scale,
        1e-12,
    )

    return dict(
        A=A,
        B=B,
        U=U,
        singular=singular,
        Vt=Vt,
        y=y,
        projected_y=U.T @ y,
        reference_scale=reference_scale,
        L_full=L_full,
        full_flux=full_flux,
        baseline_flux=baseline_flux,
        baseline_sigma=baseline_sigma,
        W=W,
    )


def evaluate_alpha(system, alpha):
    """Calculate local correction and target-risk estimates."""

    A = system["A"]

    B = system["B"]

    U = system["U"]

    d = system["singular"]

    Vt = system["Vt"]

    y = system["y"]

    t = system["projected_y"]

    W = system["W"]

    if np.isinf(alpha):

        shrink = np.zeros_like(d)

    else:

        ridge = (
            alpha
            * system["reference_scale"]
        )

        shrink = (
            d*d / (d*d + ridge)
        )

    # Regularized morphology increment.
    gamma = Vt.T @ (
        shrink * t / d
    )

    # Linear flux estimator.
    L = (
        A
        - (B * (shrink / d)) @ U.T
    )

    flux = L @ y

    L_full = system["L_full"]

    full_flux = system["full_flux"]

    # Flux-targeted SURE-type risk.
    difference = W * (
        flux - full_flux
    )

    WL = W[:, None] * L

    WL_full = (
        W[:, None] * L_full
    )

    flux_risk = (
        np.sum(difference**2)
        + 2.0 * np.sum(WL * WL_full)
        - np.sum(WL_full**2)
    )

    # Pixel-targeted SURE comparison.
    # Terms constant across alpha are omitted.
    pixel_risk = (
        np.sum(
            ((1.0 - shrink) * t)**2
        )
        + 2.0 * np.sum(shrink)
    )

    # Local statistical uncertainty approximation.
    linear_sigma = np.sqrt(
        np.sum(L*L, axis=1)
    )

    # Sensitivity to unrestricted morphology changes.
    ambiguity = np.abs(
        full_flux
        - system["baseline_flux"]
    )

    return dict(
        alpha=alpha,
        gamma=gamma.reshape(2, 2),
        linear_flux=flux,
        linear_sigma=linear_sigma,
        ambiguity=ambiguity,
        flux_risk=float(flux_risk),
        pixel_risk=float(pixel_risk),
    )


def build_candidates(systems):
    """Evaluate a common dimensionless alpha across bands."""

    candidates = {}

    for alpha in ALPHAS:

        bands = [
            evaluate_alpha(
                system,
                alpha,
            )
            for system in systems
        ]

        candidates[alpha] = dict(
            alpha=alpha,

            gamma=np.stack([
                item["gamma"]
                for item in bands
            ]),

            linear_flux=np.stack([
                item["linear_flux"]
                for item in bands
            ], axis=1),

            linear_sigma=np.stack([
                item["linear_sigma"]
                for item in bands
            ], axis=1),

            ambiguity=np.stack([
                item["ambiguity"]
                for item in bands
            ], axis=1),

            flux_risk=sum(
                item["flux_risk"]
                for item in bands
            ),

            pixel_risk=sum(
                item["pixel_risk"]
                for item in bands
            ),
        )

    return candidates


# ---------------------------------------------------------
# Physical forward model
# ---------------------------------------------------------

def reconstruction_chi2(
    observed,
    model,
    noise,
):
    """Match the existing benchmark residual metric."""

    return float(
        np.mean(
            (
                (observed - model)
                / noise[:, None, None]
            )**2
        )
    )


def fit_templates(objective, templates):
    """Refit nonnegative individual fluxes and background."""

    observed = objective.observed

    flux = np.empty(
        (2, len(objective.noise))
    )

    model = np.empty_like(observed)

    for b in range(
        len(objective.noise)
    ):

        flux[:, b], _, model[b] = (
            profile_linear(
                templates[b],
                observed[b],
            )
        )

    if (
        not np.isfinite(flux).all()
        or not np.isfinite(model).all()
        or np.any(flux < 0)
    ):
        raise ValueError(
            "Invalid physical flux fit"
        )

    chi2 = reconstruction_chi2(
        observed,
        model,
        objective.noise,
    )

    return flux, model, chi2


def bounded_step(theta, gamma):
    """Apply trust region and physical parameter bounds.

    gamma has shape:
        bands x sources x 2

    The two increments are radial log-width
    and log-Sersic-index.
    """

    delta = np.clip(
        gamma,
        -TRUST_RADIUS,
        TRUST_RADIUS,
    )

    for b in range(delta.shape[0]):

        for s in range(2):

            lx, ly, _, ln = theta[s]

            # Common width change preserves
            # the axis ratio and orientation.
            lo = max(
                np.log(MIN_SIGMA) - lx,
                np.log(MIN_SIGMA) - ly,
            )

            hi = min(
                np.log(MAX_SIGMA) - lx,
                np.log(MAX_SIGMA) - ly,
            )

            delta[b, s, 0] = np.clip(
                delta[b, s, 0],
                lo,
                hi,
            )

            delta[b, s, 1] = np.clip(
                delta[b, s, 1],
                np.log(MIN_N) - ln,
                np.log(MAX_N) - ln,
            )

    return delta


def render_corrected(
    objective,
    theta,
    delta,
):
    """Render exact positive Sersic profiles after correction."""

    templates = []

    for b in range(
        len(objective.noise)
    ):

        band_templates = []

        for s, quadrature in enumerate(
            objective.quad
        ):

            shape = theta[s].copy()

            # Apply the bounded chromatic increment.
            shape[:2] += delta[b, s, 0]

            shape[3] += delta[b, s, 1]

            # Exact positive normalized profile;
            # not a signed linear morphology.
            profile = quadrature.evaluate(
                shape
            )[0:1]

            if (
                np.any(profile < 0)
                or not np.isclose(
                    profile.sum(),
                    1.0,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    "Invalid intrinsic profile"
                )

            blurred = (
                objective.convolve(profile)
                [0, b]
            )

            band_templates.append(
                blurred
            )

        templates.append(
            np.stack(band_templates)
        )

    return np.stack(templates)


def physical_refinement(
    objective,
    theta,
    candidate,
    baseline_flux,
    baseline_chi2,
):
    """Apply the same physical guard to every correction method."""

    gamma = candidate["gamma"]

    delta = bounded_step(
        theta,
        gamma,
    )

    clipped = not np.allclose(
        delta,
        gamma,
        rtol=1e-8,
        atol=1e-10,
    )

    result = dict(
        flux=baseline_flux.copy(),
        chi2=baseline_chi2,
        accepted=False,
        trust_clipped=clipped,
        backtrack_factor=0.0,
        applied_delta=np.zeros_like(delta),
    )

    if (
        np.isinf(candidate["alpha"])
        or np.max(np.abs(delta)) < 1e-10
    ):
        return result

    # Nonlinear verification of the local step.
    for factor in BACKTRACK_FACTORS:

        applied = factor * delta

        templates = render_corrected(
            objective,
            theta,
            applied,
        )

        flux, _, chi2 = fit_templates(
            objective,
            templates,
        )

        # Do not accept a correction that increases
        # the actual nonlinear reconstruction loss.
        if chi2 < baseline_chi2 - 1e-10:

            result.update(
                flux=flux,
                chi2=chi2,
                accepted=True,
                backtrack_factor=factor,
                applied_delta=applied,
            )

            return result

    return result


# ---------------------------------------------------------
# Proposed adaptive refinement interface
# ---------------------------------------------------------

def refine_scene(
    observed,
    psfs,
    positions,
    noise,
    theta,
):
    """Run the proposed prototype using observation data only."""

    objective = SersicObjective(
        observed,
        psfs,
        positions,
        noise,
    )

    # Compute exact profiles and analytic derivatives
    # at the shared-morphology initial solution.
    blurred = np.stack([
        objective.convolve(
            quadrature.evaluate(theta[s])
        )
        for s, quadrature
        in enumerate(objective.quad)
    ])

    templates = np.stack([
        blurred[:, 0, b]
        for b in range(len(noise))
    ])

    baseline_flux, _, baseline_chi2 = (
        fit_templates(
            objective,
            templates,
        )
    )

    # Prepare a common no-correction fallback.
    fallback = dict(
        flux=baseline_flux.copy(),
        chi2=baseline_chi2,
        accepted=False,
        trust_clipped=False,
        backtrack_factor=0.0,
        applied_delta=np.zeros(
            (len(noise), 2, 2)
        ),
        alpha=np.inf,
        flux_risk=np.nan,
        pixel_risk=np.nan,
        linear_sigma=np.full(
            baseline_flux.shape,
            np.nan,
        ),
        ambiguity=np.full(
            baseline_flux.shape,
            np.nan,
        ),
        diagnostic_valid=False,
        diagnostic_reason="",
    )

    start = perf_counter()

    try:

        systems = []

        for b in range(len(noise)):

            derivatives = []

            for s in range(2):

                # Joint radial log-width tangent.
                derivatives.append(
                    blurred[s, 1, b]
                    + blurred[s, 2, b]
                )

                # Concentration tangent.
                derivatives.append(
                    blurred[s, 4, b]
                )

            systems.append(
                prepare_band(
                    templates[b],
                    np.stack(derivatives),
                    observed[b],
                    noise[b],
                    baseline_flux[:, b],
                )
            )

        candidates = build_candidates(
            systems
        )

    except (
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:

        # Do not silently discard a difficult scene.
        # Preserve the initial fit and mark ambiguity
        # as unresolved.
        fallback["diagnostic_reason"] = str(exc)

        fallback["diagnostic_seconds"] = (
            perf_counter() - start
        )

        fallback["correction_seconds"] = 0.0

        return {
            method: fallback.copy()
            for method in METHODS
        }

    diagnostic_seconds = (
        perf_counter() - start
    )

    # Choose alpha without using simulation truth.
    pixel_alpha = min(
        ALPHAS,
        key=lambda a: (
            candidates[a]["pixel_risk"],
            -a,
        ),
    )

    flux_alpha = min(
        ALPHAS,
        key=lambda a: (
            candidates[a]["flux_risk"],
            -a,
        ),
    )

    selections = {
        "flexible_sersic": np.inf,
        "unregularized": 0.0,
        "fixed_regularization": FIXED_ALPHA,
        "pixel_risk": pixel_alpha,
        "flux_risk": flux_alpha,
    }

    # Multiple methods may select the same alpha.
    # Cache physical corrections to avoid extra work.
    cache = {}

    results = {}

    for method, alpha in selections.items():

        candidate = candidates[alpha]

        if alpha not in cache:

            correction_start = perf_counter()

            physical = physical_refinement(
                objective,
                theta,
                candidate,
                baseline_flux,
                baseline_chi2,
            )

            physical["correction_seconds"] = (
                perf_counter()
                - correction_start
            )

            cache[alpha] = physical

        result = dict(
            cache[alpha]
        )

        result.update(
            alpha=alpha,

            flux_risk=candidate[
                "flux_risk"
            ],

            pixel_risk=candidate[
                "pixel_risk"
            ],

            linear_sigma=candidate[
                "linear_sigma"
            ],

            ambiguity=candidate[
                "ambiguity"
            ],

            diagnostic_valid=True,

            diagnostic_reason="",

            diagnostic_seconds=(
                diagnostic_seconds
            ),
        )

        results[method] = result

    return results


# ---------------------------------------------------------
# Matched experimental evaluation
# ---------------------------------------------------------

def evaluate_scene(
    family,
    scene_id,
    rows,
):
    """Evaluate all methods on one matched observation."""

    first = rows[0]

    metadata = dict(
        separation=float(
            first["separation_px"]
        ),
        ratio=float(
            first["flux_ratio_f277w"]
        ),
        seed=int(first["seed"]),
    )

    scene = regenerate(
        family,
        metadata,
    )

    archived_flux, archived_truth, lookup = (
        archive_arrays(rows)
    )

    np.testing.assert_allclose(
        scene["input_flux"],
        archived_truth,
        atol=1e-8,
        rtol=1e-8,
    )

    observed = scene["observed"]

    psfs = scene["psfs"]

    positions = scene["positions"]

    noise = scene["noise_sigma"]

    initial_start = perf_counter()

    if family == "chromatic":

        # Full fitted shapes were not saved in
        # the Phase 9.1 composite CSV.
        _, _, diagnostics = (
            fit_flexible_sersic(
                observed,
                psfs,
                positions,
                noise,
            )
        )

        theta = np.stack([
            shape_vector(shape)
            for shape
            in diagnostics["shapes"]
        ])

        initialization_source = "refitted"

        initial_optimizer_success = bool(
            diagnostics["optimizer_success"]
        )

    else:

        # Reuse previously fitted shape parameters.
        theta = np.stack([
            shape_vector(
                lookup[s, BANDS[0]]
            )
            for s in (1, 2)
        ])

        initialization_source = "archive"

        initial_optimizer_success = all(
            row["optimizer_success"].lower()
            == "true"
            for row in rows
        )

    initial_seconds = (
        perf_counter() - initial_start
    )

    # The proposed algorithm receives no true
    # morphology, family label or true flux.
    results = refine_scene(
        observed,
        psfs,
        positions,
        noise,
        theta,
    )

    baseline = results[
        "flexible_sersic"
    ]

    # Confirm that the archived baseline has been
    # reconstructed consistently.
    np.testing.assert_allclose(
        baseline["flux"],
        archived_flux,
        rtol=5e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        baseline["chi2"],
        float(first["chi2"]),
        rtol=1e-5,
        atol=1e-6,
    )

    # Ground truth enters only the evaluation.
    truth = scene["input_flux"]

    baseline_error = np.abs(
        baseline["flux"] / truth - 1
    )

    records = []

    for method in METHODS:

        result = results[method]

        flux = result["flux"]

        signed_error = (
            flux - truth
        ) / truth

        combined_error = (
            flux.sum(axis=0)
            - truth.sum(axis=0)
        ) / truth.sum(axis=0)

        if (
            not np.isfinite(
                signed_error
            ).all()
            or not np.isfinite(
                result["chi2"]
            )
        ):
            raise ValueError(
                "Invalid evaluation metrics"
            )

        mean_error = float(
            np.abs(signed_error).mean()
        )

        print(
            f"{family:10s} | "
            f"scene {scene_id:3d} | "
            f"{method:22s} | "
            f"error={mean_error*100:7.2f}% | "
            f"alpha={result['alpha']:7g} | "
            f"accepted={result['accepted']}",
            flush=True,
        )

        for s in range(2):

            for b, band in enumerate(BANDS):

                records.append(dict(
                    morphology=family,

                    scene_id=scene_id,

                    seed=metadata["seed"],

                    separation_px=metadata[
                        "separation"
                    ],

                    flux_ratio_f277w=metadata[
                        "ratio"
                    ],

                    method=method,

                    source=s + 1,

                    band=band,

                    true_flux=truth[s, b],

                    initial_flux=(
                        baseline["flux"][s, b]
                    ),

                    estimated_flux=flux[s, b],

                    signed_error=(
                        signed_error[s, b]
                    ),

                    absolute_error=abs(
                        signed_error[s, b]
                    ),

                    initial_absolute_error=(
                        baseline_error[s, b]
                    ),

                    combined_signed_error=(
                        combined_error[b]
                    ),

                    chi2=result["chi2"],

                    initial_chi2=(
                        baseline["chi2"]
                    ),

                    selected_alpha=(
                        result["alpha"]
                    ),

                    flux_risk=(
                        result["flux_risk"]
                    ),

                    pixel_risk=(
                        result["pixel_risk"]
                    ),

                    correction_accepted=(
                        result["accepted"]
                    ),

                    trust_clipped=(
                        result["trust_clipped"]
                    ),

                    backtrack_factor=(
                        result["backtrack_factor"]
                    ),

                    delta_log_width=(
                        result["applied_delta"]
                        [b, s, 0]
                    ),

                    delta_log_n=(
                        result["applied_delta"]
                        [b, s, 1]
                    ),

                    linear_sigma=(
                        result["linear_sigma"]
                        [s, b]
                    ),

                    linear_ambiguity=(
                        result["ambiguity"]
                        [s, b]
                    ),

                    diagnostic_valid=(
                        result["diagnostic_valid"]
                    ),

                    diagnostic_reason=(
                        result["diagnostic_reason"]
                    ),

                    initialization_source=(
                        initialization_source
                    ),

                    initial_optimizer_success=(
                        initial_optimizer_success
                    ),

                    initialization_seconds=(
                        initial_seconds
                    ),

                    diagnostic_seconds=(
                        result["diagnostic_seconds"]
                    ),

                    correction_seconds=(
                        result["correction_seconds"]
                    ),
                ))

    return records


# ---------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------

def summarize(rows):
    """Report matched photometric performance."""

    tracks = {
        "single_profile": [
            r for r in rows
            if r["morphology"] != "chromatic"
        ],

        "chromatic": [
            r for r in rows
            if r["morphology"] == "chromatic"
        ],

        "overall": rows,
    }

    for track, subset in tracks.items():

        if not subset:
            continue

        print(
            f"\n--- {track} ---"
        )

        for method in METHODS:

            group = [
                r for r in subset
                if r["method"] == method
            ]

            if not group:
                continue

            errors = np.array([
                r["absolute_error"]
                for r in group
            ])

            old_errors = np.array([
                r["initial_absolute_error"]
                for r in group
            ])

            faint = [
                r["absolute_error"]
                for r in group
                if (
                    r["source"] == 2
                    and r["flux_ratio_f277w"]
                    == 0.1
                )
            ]

            scene_records = [
                r for r in group
                if (
                    r["source"] == 1
                    and r["band"] == BANDS[0]
                )
            ]

            accepted = sum(
                r["correction_accepted"]
                for r in scene_records
            )

            valid = sum(
                r["diagnostic_valid"]
                for r in scene_records
            )

            relative_reduction = (
                1 - errors.mean()
                / old_errors.mean()
                if old_errors.mean() > 0
                else np.nan
            )

            print(
                f"\n{method}"
            )

            print(
                f"  Mean error: "
                f"{errors.mean()*100:.2f}%"
            )

            print(
                f"  Median error: "
                f"{np.median(errors)*100:.2f}%"
            )

            print(
                f"  P90 error: "
                f"{np.percentile(errors, 90)*100:.2f}%"
            )

            print(
                f"  Relative error reduction: "
                f"{relative_reduction*100:+.2f}%"
            )

            print(
                f"  Accepted corrections: "
                f"{accepted}/{len(scene_records)}"
            )

            print(
                f"  Valid diagnostics: "
                f"{valid}/{len(scene_records)}"
            )

            if faint:

                print(
                    f"  Faint companion error: "
                    f"{np.mean(faint)*100:.2f}%"
                )


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Evaluate one archived Gaussian scene "
            "without writing a CSV."
        ),
    )

    args = parser.parse_args()

    if not args.smoke and OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    groups = load_cases()

    cases = (
        [("gaussian", 1)]
        if args.smoke
        else [
            (family, i)
            for family in FAMILIES
            for i in SCENE_IDS
        ]
    )

    planned_records = (
        len(cases)
        * len(METHODS)
        * 6
    )

    print(
        "Phase 9.3.1 — "
        "Photometric risk-guided refinement"
    )

    print(
        f"Planned scenes: {len(cases)}"
    )

    print(
        f"New expensive initial fits: "
        f"{sum(f == 'chromatic' for f, _ in cases)}"
    )

    print(
        f"Expected flux measurements: "
        f"{planned_records}",
        flush=True,
    )

    records = []

    for family, scene_id in cases:

        records.extend(
            evaluate_scene(
                family,
                scene_id,
                groups[family, scene_id],
            )
        )

    if len(records) != planned_records:
        raise RuntimeError(
            "Incomplete evaluation"
        )

    summarize(records)

    if not args.smoke:

        with OUTPUT.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=list(
                    records[0]
                ),
            )

            writer.writeheader()

            writer.writerows(
                records
            )

        print(
            f"\nResults saved: {OUTPUT}"
        )

    print(
        f"Completed measurements: "
        f"{len(records)}/{planned_records}"
    )


if __name__ == "__main__":
    main()
