
"""Phase 9.2.1: observation-derived flux sensitivity.

No new nonlinear fits. Reuse archived fixed Gaussian
estimates and regenerate the matched observations.

Linearized morphology perturbations are diagnostics,
not final flux estimates or calibrated uncertainties.
"""

import csv

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.linalg import block_diag
from scipy.signal import fftconvolve
from scipy.special import gammaincinv
from scipy.stats import spearmanr

from src.baselines.baseline_deblend import (
    TEMPLATE_SIGMA,
)

from src.evaluation.benchmark_baseline import (
    create_configuration,
)

from src.evaluation.benchmark_composite_morphology import (
    create_composite_components,
    create_composite_scene,
    BULGE_FRACTIONS,
)

from src.simulation.simulate_blends import (
    simulate_scene,
)


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results"

OUTPUT = (
    RESULTS
    / "benchmark_flux_sensitivity_phase9_2_1.csv"
)

BANDS = ("f150w", "f277w", "f444w")

FAMILIES = {
    "gaussian": None,
    "sersic1": 1.0,
    "sersic2": 2.0,
    "sersic4": 4.0,
}

SCENE_IDS = (1, 16, 46, 81, 96)


# ---------------------------------------------------------
# Load previously evaluated measurements
# ---------------------------------------------------------

def load_archives():

    groups = defaultdict(list)

    path = (
        RESULTS
        / "benchmark_flexible_sersic_phase8_4.csv"
    )

    with path.open(newline="") as stream:

        for row in csv.DictReader(stream):

            if row["status"] != "success":
                raise ValueError("Invalid archived fit")

            key = (
                row["morphology"],
                int(row["scene_id"]),
            )

            groups[key].append({
                "source": int(row["source"]),
                "band": row["band"].lower(),
                "seed": int(row["seed"]),
                "separation": float(
                    row["separation_px"]
                ),
                "ratio": float(
                    row["flux_ratio_f277w"]
                ),
                "truth": float(row["true_flux"]),
                "fixed": float(
                    row["fixed_gaussian_flux"]
                ),
                "sersic": float(
                    row["estimated_flux"]
                ),
                "fixed_chi2": float(
                    row["fixed_gaussian_chi2"]
                ),
            })

    path = (
        RESULTS
        / "benchmark_composite_morphology_phase9_1.csv"
    )

    composite = defaultdict(dict)

    with path.open(newline="") as stream:

        for row in csv.DictReader(stream):

            if row["method"] not in (
                "fixed_gaussian",
                "flexible_sersic",
            ):
                continue

            if row["status"] != "success":
                raise ValueError("Invalid composite fit")

            key = (
                row["track"],
                int(row["scene_id"]),
                int(row["source"]),
                row["band"].lower(),
            )

            if row["method"] in composite[key]:
                raise ValueError("Duplicate measurement")

            composite[key][row["method"]] = row

    for key, methods in composite.items():

        family, scene_id, source, band = key

        fixed = methods["fixed_gaussian"]

        sersic = methods["flexible_sersic"]

        groups[family, scene_id].append({
            "source": source,
            "band": band,
            "seed": int(fixed["seed"]),
            "separation": float(
                fixed["separation_px"]
            ),
            "ratio": float(
                fixed["flux_ratio_f277w"]
            ),
            "truth": float(fixed["true_flux"]),
            "fixed": float(fixed["estimated_flux"]),
            "sersic": float(sersic["estimated_flux"]),
            "fixed_chi2": float(fixed["chi2"]),
        })

    expected = {
        (family, i)
        for family in (
            *FAMILIES,
            *BULGE_FRACTIONS,
        )
        for i in SCENE_IDS
    }

    if set(groups) != expected:
        raise ValueError("Unexpected archived scenes")

    for key, rows in groups.items():

        if len(rows) != 6:
            raise ValueError(
                f"Expected six measurements: {key}"
            )

        keys = {
            (r["source"], r["band"])
            for r in rows
        }

        expected_keys = {
            (s, b)
            for s in (1, 2)
            for b in BANDS
        }

        if keys != expected_keys:
            raise ValueError(
                f"Invalid source-band keys: {key}"
            )

    return groups


# ---------------------------------------------------------
# Normalized morphology perturbations
# ---------------------------------------------------------

def template_derivatives(shape, position, psf):
    """Return the original template and four shape tangents.

    Perturbations:
        log width X
        log width Y
        correlation / ellipticity
        log Sersic concentration

    Each intrinsic derivative has zero integrated mass.
    """

    y, x = np.indices(shape, dtype=float)

    u = (x - position[0]) / TEMPLATE_SIGMA
    v = (y - position[1]) / TEMPLATE_SIGMA

    r2 = u*u + v*v

    p = np.exp(-0.5 * r2)
    p /= p.sum()

    def concentration_profile(n):

        radius = (
            np.sqrt(r2)
            / np.sqrt(2 * np.log(2))
        )

        z = np.exp(
            -gammaincinv(2*n, 0.5)
            * radius**(1/n)
        )

        return z / z.sum()

    eps = 1e-4

    dn = (
        concentration_profile(
            0.5 * np.exp(eps)
        )
        - concentration_profile(
            0.5 * np.exp(-eps)
        )
    ) / (2 * eps)

    derivatives = [
        p * (u*u - np.sum(p*u*u)),
        p * (v*v - np.sum(p*v*v)),
        p * (u*v - np.sum(p*u*v)),
        dn,
    ]

    profiles = np.stack([
        p,
        *derivatives,
    ])

    np.testing.assert_allclose(
        profiles[1:].sum(axis=(1, 2)),
        0,
        atol=1e-8,
    )

    return np.stack([
        fftconvolve(z, psf, mode="same")
        for z in profiles
    ])


# ---------------------------------------------------------
# Stable local linear sensitivity calculation
# ---------------------------------------------------------

def linear_diagnostic(J, residual):
    """Unconstrained local step and covariance approximation.

    Column scaling improves numerical conditioning.
    Rank-deficient systems are explicitly marked.
    """

    norms = np.linalg.norm(J, axis=0)

    threshold = (
        1e-12 * max(norms.max(), 1.0)
    )

    if np.any(norms <= threshold):

        return None, None, 0

    A = J / norms

    U, singular, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    good = singular > 1e-10 * singular[0]

    rank = int(good.sum())

    if rank != J.shape[1]:

        return None, None, rank

    V = Vt.T

    step = (
        V @ (
            (U.T @ residual) / singular
        )
    ) / norms

    variance = (
        np.sum(
            (V / singular)**2,
            axis=1,
        )
        / norms**2
    )

    return step, variance, rank


def flux_sensitivity(
    observed,
    psfs,
    positions,
    noise,
    flux,
):
    """Extract features without accessing ground truth.

    Flux shape is (bands, sources).
    """

    X_blocks = []
    Z_blocks = []
    residuals = []

    for b in range(len(noise)):

        T = np.stack([
            template_derivatives(
                observed[b].shape,
                position,
                psfs[b],
            )
            for position in positions
        ])

        model = np.sum(
            flux[b, :, None, None]
            * T[:, 0],
            axis=0,
        )

        # Profile out the unrestricted
        # constant background.
        background = np.mean(
            observed[b] - model
        )

        r = (
            observed[b]
            - model
            - background
        ) / noise[b]

        X = np.column_stack([
            T[0, 0].ravel(),
            T[1, 0].ravel(),
            np.ones(r.size),
        ]) / noise[b]

        Z = np.column_stack([
            T[s, 1:].reshape(4, -1).T
            * flux[b, s]
            / noise[b]
            for s in range(2)
        ])

        X_blocks.append(X)
        Z_blocks.append(Z)
        residuals.append(r.ravel())

    X = block_diag(*X_blocks)

    # Shared morphology perturbations:
    # eight shape variables across all bands.
    Z_shared = np.vstack(Z_blocks)

    # Chromatic morphology perturbations:
    # eight independent variables per band.
    Z_chromatic = block_diag(*Z_blocks)

    r = np.concatenate(residuals)

    flux_ids = np.array([
        3*b + s
        for b in range(len(noise))
        for s in range(2)
    ])

    # Fixed-template statistical covariance.
    _, var0, rank0 = linear_diagnostic(X, r)

    if var0 is None:

        raise ValueError(
            "Fixed template system is rank deficient"
        )

    sigma0 = np.sqrt(var0[flux_ids])

    # Stabilize relative metrics for very faint sources.
    # This denominator uses estimated quantities only.
    scale = np.maximum(
        np.abs(flux.ravel()),
        sigma0,
    )

    features = {
        "fixed_chi2": float(np.mean(r*r)),
        "fixed_rank": rank0,
        "fixed_relative_sigma": float(
            np.max(sigma0 / scale)
        ),
        "min_estimated_snr": float(
            np.min(flux.ravel() / sigma0)
        ),
        "near_flux_boundary": bool(
            np.any(
                flux.ravel()
                <= 1e-8 * max(flux.max(), 1.0)
            )
        ),
    }

    shifts = {}

    for name, Z in (
        ("shared", Z_shared),
        ("chromatic", Z_chromatic),
    ):

        J = np.column_stack([X, Z])

        step, variance, rank = (
            linear_diagnostic(J, r)
        )

        features[f"{name}_rank"] = rank

        features[f"{name}_full_rank"] = (
            step is not None
        )

        if step is None:

            features[f"{name}_relative_shift"] = (
                np.nan
            )

            features[f"{name}_noise_amplification"] = (
                np.nan
            )

            continue

        delta = step[flux_ids]

        shifts[name] = delta

        features[f"{name}_relative_shift"] = float(
            np.max(np.abs(delta) / scale)
        )

        features[f"{name}_noise_amplification"] = float(
            np.max(
                np.sqrt(variance[flux_ids])
                / sigma0
            )
        )

    features["chromatic_extra_shift"] = (
        float(
            np.max(
                np.abs(
                    shifts["chromatic"]
                    - shifts["shared"]
                ) / scale
            )
        )
        if len(shifts) == 2
        else np.nan
    )

    return features


# ---------------------------------------------------------
# Matched observation generation
# ---------------------------------------------------------

def regenerate(family, metadata):

    config = create_configuration(
        metadata["separation"],
        metadata["ratio"],
        metadata["seed"],
    )

    if family in FAMILIES:

        n = FAMILIES[family]

        if n is None:
            return simulate_scene(**config)

        return simulate_scene(
            **config,
            sersic_n=n,
        )

    bulge, disk = (
        create_composite_components(config)
    )

    return create_composite_scene(
        bulge,
        disk,
        BULGE_FRACTIONS[family],
    )


def evaluate_scene(family, scene_id, rows):

    rows = sorted(
        rows,
        key=lambda r: (
            BANDS.index(r["band"]),
            r["source"],
        ),
    )

    metadata = rows[0]

    start = perf_counter()

    scene = regenerate(
        family,
        metadata,
    )

    fixed = np.array([
        r["fixed"] for r in rows
    ]).reshape(3, 2)

    # Observation-side diagnostic.
    features = flux_sensitivity(
        scene["observed"],
        scene["psfs"],
        scene["positions"],
        scene["noise_sigma"],
        fixed,
    )

    # Validate the archived reconstruction.
    if not np.isclose(
        features["fixed_chi2"],
        metadata["fixed_chi2"],
        rtol=1e-5,
        atol=1e-6,
    ):

        raise ValueError(
            f"Archive reconstruction mismatch: "
            f"{family}, {scene_id}"
        )

    # Ground truth is accessed only after
    # all diagnostic features are computed.
    truth = np.array([
        r["truth"] for r in rows
    ])

    np.testing.assert_allclose(
        truth,
        scene["input_flux"].T.ravel(),
    )

    sersic = np.array([
        r["sersic"] for r in rows
    ])

    fixed_error = np.abs(
        fixed.ravel() / truth - 1
    )

    sersic_error = np.abs(
        sersic / truth - 1
    )

    elapsed = perf_counter() - start

    result = {
        "morphology": family,
        "scene_id": scene_id,
        "seed": metadata["seed"],
        "separation_px": metadata["separation"],

        **features,

        # Evaluation targets, not controller inputs.
        "true_flux_ratio": metadata["ratio"],
        "fixed_mean_error": float(
            fixed_error.mean()
        ),
        "fixed_max_error": float(
            fixed_error.max()
        ),
        "sersic_mean_error": float(
            sersic_error.mean()
        ),
        "refinement_benefit": float(
            fixed_error.mean()
            - sersic_error.mean()
        ),
        "runtime_seconds": elapsed,
    }

    print(
        f"{family} | scene {scene_id} | "
        f"fixed={fixed_error.mean()*100:.2f}% | "
        f"benefit={result['refinement_benefit']*100:.2f} pp | "
        f"shared_shift={features['shared_relative_shift']:.3f} | "
        f"chromatic_shift={features['chromatic_relative_shift']:.3f}",
        flush=True,
    )

    return result


# ---------------------------------------------------------
# Exploratory evaluation
# ---------------------------------------------------------

def summarize(rows):

    indicators = (
        "separation_px",
        "fixed_chi2",
        "fixed_relative_sigma",
        "min_estimated_snr",
        "shared_relative_shift",
        "chromatic_relative_shift",
        "shared_noise_amplification",
        "chromatic_noise_amplification",
        "chromatic_extra_shift",
    )

    targets = (
        "fixed_mean_error",
        "fixed_max_error",
        "refinement_benefit",
    )

    print("\n--- Exploratory Spearman correlations ---")

    for indicator in indicators:

        x = np.array([
            r[indicator] for r in rows
        ], dtype=float)

        for target in targets:

            y = np.array([
                r[target] for r in rows
            ], dtype=float)

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            if (
                valid.sum() >= 3
                and np.ptp(x[valid]) > 0
                and np.ptp(y[valid]) > 0
            ):

                rho = spearmanr(
                    x[valid],
                    y[valid],
                ).statistic

                print(
                    f"{indicator} -> {target}: "
                    f"rho={rho:+.3f}, "
                    f"n={valid.sum()}"
                )


def main():

    if OUTPUT.exists():

        raise FileExistsError(OUTPUT)

    groups = load_archives()

    print(
        f"Planned diagnostics: {len(groups)}"
    )

    print("New nonlinear fits: 0")

    rows = []

    for (family, scene_id), measurements in sorted(
        groups.items()
    ):

        rows.append(
            evaluate_scene(
                family,
                scene_id,
                measurements,
            )
        )

    with OUTPUT.open("x", newline="") as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)

    print(
        f"\nCompleted diagnostics: "
        f"{len(rows)}/{len(groups)}"
    )

    print(f"Results saved: {OUTPUT}")


if __name__ == "__main__":
    main()
