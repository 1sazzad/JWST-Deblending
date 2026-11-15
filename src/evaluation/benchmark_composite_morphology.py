
"""Phase 9.1: composite galaxy morphology benchmark.

Investigate single-profile model mismatch using two-component
bulge-disk galaxies. Reuse the existing simulator and fitters.

Two tracks:
    1. Achromatic composite morphology.
    2. Chromatic composite morphology.

Ground truth is used exclusively for simulation and evaluation.
"""

import csv

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from src.simulation.simulate_blends import (
    simulate_scene,
)

from src.baselines.baseline_deblend import (
    fit_band,
)

from src.baselines.baseline_flexible_gaussian import (
    fit_flexible_gaussian,
)

from src.baselines.baseline_flexible_sersic import (
    fit_flexible_sersic,
)

from src.evaluation.benchmark_baseline import (
    create_configuration,
    SEPARATIONS,
    FLUX_RATIOS,
    REPETITIONS,
    BASE_SEED,
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "results"
    / "benchmark_composite_morphology_phase9_1.csv"
)

SCENE_IDS = (1, 16, 46, 81, 96)

METHODS = (
    "fixed_gaussian",
    "flexible_gaussian",
    "flexible_sersic",
)

# Rows represent sources.
# Columns represent F150W, F277W and F444W.
#
# Both tracks share identical F277W bulge fractions.
# Only the wavelength dependence changes.

BULGE_FRACTIONS = {
    "achromatic": np.array([
        [0.65, 0.65, 0.65],
        [0.35, 0.35, 0.35],
    ]),

    "chromatic": np.array([
        [0.40, 0.65, 0.80],
        [0.15, 0.35, 0.60],
    ]),
}

FIELDS = (
    "track",
    "scene_id",
    "seed",
    "separation_px",
    "flux_ratio_f277w",
    "method",
    "source",
    "band",
    "bulge_fraction",
    "true_flux",
    "estimated_flux",
    "signed_error",
    "absolute_error",
    "combined_signed_error",
    "chi2",
    "fitted_sersic_n",
    "optimizer_success",
    "runtime_seconds",
    "status",
    "failure_reason",
)


# ---------------------------------------------------------
# Composite galaxy simulation
# ---------------------------------------------------------

def create_composite_components(config):
    """Generate co-centered bulge and disk components.

    Bulges use Sersic n=4 with smaller effective radii.
    Disks use Sersic n=1 with larger effective radii.

    Both retain the same source positions, total input flux,
    observational PSFs and noise realization.
    """

    bulge_shape = config["morphology"].copy()

    disk_shape = config["morphology"].copy()

    # Construct distinct radial structures.
    bulge_shape[:, :2] *= 0.55

    disk_shape[:, :2] *= 1.25

    bulge = simulate_scene(
        **{
            **config,
            "morphology": bulge_shape,
        },
        sersic_n=4.0,
    )

    disk = simulate_scene(
        **{
            **config,
            "morphology": disk_shape,
        },
        sersic_n=1.0,
    )

    # Confirm matching simulation conditions.
    for key in (
        "input_flux",
        "positions",
        "psfs",
        "noise_sigma",
    ):

        np.testing.assert_allclose(
            bulge[key],
            disk[key],
        )

    return bulge, disk


def create_composite_scene(
    bulge,
    disk,
    fractions,
):
    """Combine normalized source components before fitting.

    Each component originally carries the full source flux.
    Weighted combination therefore preserves total input
    flux while allowing realistic morphology mismatch.
    """

    fractions = np.asarray(
        fractions,
        dtype=np.float64,
    )

    truth = disk["input_flux"]

    if (
        fractions.shape != truth.shape
        or np.any(fractions < 0)
        or np.any(fractions > 1)
        or not np.isfinite(fractions).all()
    ):
        raise ValueError(
            "Invalid bulge fractions"
        )

    weights = fractions[:, :, None, None]

    # Combine PSF-convolved bulge and disk images.
    components = (
        weights * bulge["components"]
        + (1 - weights) * disk["components"]
    )

    # Preserve the finite-canvas intrinsic flux definition.
    intrinsic = (
        weights * bulge["intrinsic"]
        + (1 - weights) * disk["intrinsic"]
    )

    # Reuse one noise realization for both tracks.
    noise = (
        disk["observed"]
        - disk["noiseless"]
    )

    noiseless = components.sum(axis=0)

    observed = noiseless + noise

    visible_flux = components.sum(
        axis=(-2, -1)
    )

    if not np.isfinite(observed).all():
        raise ValueError(
            "Invalid composite observation"
        )

    if (
        np.any(visible_flux <= 0)
        or np.any(
            visible_flux > truth * (1 + 1e-6)
        )
    ):
        raise ValueError(
            "Invalid composite visible flux"
        )

    return {
        "observed": observed,
        "noiseless": noiseless,
        "components": components,
        "intrinsic": intrinsic,
        "input_flux": truth.copy(),
        "visible_flux": visible_flux,
        "positions": disk["positions"],
        "noise_sigma": disk["noise_sigma"],
        "psfs": disk["psfs"],
        "bands": disk["bands"],
    }


# ---------------------------------------------------------
# Existing baseline interfaces
# ---------------------------------------------------------

def fit_method(scene, method):
    """Use observation-side information only."""

    observed = scene["observed"]

    psfs = scene["psfs"]

    positions = scene["positions"]

    noise = scene["noise_sigma"]

    if method == "fixed_gaussian":

        flux = np.empty(
            (len(positions), len(noise))
        )

        model = np.empty_like(
            observed
        )

        for b in range(len(noise)):

            (
                flux[:, b],
                model[b],
                _,
                _,
            ) = fit_band(
                observed[b],
                psfs[b],
                positions,
                noise[b],
            )

        diagnostics = {
            "optimizer_success": True,
            "shapes": [{}, {}],
        }

        return flux, model, diagnostics

    if method == "flexible_gaussian":

        return fit_flexible_gaussian(
            observed,
            psfs,
            positions,
            noise,
        )

    if method == "flexible_sersic":

        return fit_flexible_sersic(
            observed,
            psfs,
            positions,
            noise,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ---------------------------------------------------------
# Photometric evaluation
# ---------------------------------------------------------

def evaluate(
    scene,
    track,
    case,
    method,
):
    """Measure individual and combined flux recovery."""

    scene_id, separation, ratio, seed = case

    truth = scene["input_flux"]

    fractions = BULGE_FRACTIONS[track]

    start = perf_counter()

    status = "success"

    failure_reason = ""

    try:

        flux, model, diagnostics = fit_method(
            scene,
            method,
        )

        if (
            flux.shape != truth.shape
            or model.shape != scene["observed"].shape
        ):
            raise ValueError(
                "Unexpected model dimensions"
            )

        if (
            not np.isfinite(flux).all()
            or not np.isfinite(model).all()
            or np.any(flux < 0)
        ):
            raise ValueError(
                "Invalid fitted model"
            )

        # Ground truth is accessed only for evaluation.
        signed_error = (
            flux - truth
        ) / truth

        combined_error = (
            flux.sum(axis=0)
            - truth.sum(axis=0)
        ) / truth.sum(axis=0)

        residual = (
            scene["observed"]
            - model
        )

        chi2 = float(
            np.mean(
                (
                    residual
                    / scene["noise_sigma"][
                        :, None, None
                    ]
                ) ** 2
            )
        )

        if (
            not np.isfinite(signed_error).all()
            or not np.isfinite(chi2)
        ):
            raise ValueError(
                "Nonfinite evaluation metrics"
            )

    except Exception as exc:

        status = "failed"

        failure_reason = (
            f"{type(exc).__name__}: {exc}"
        )

    runtime = perf_counter() - start

    rows = []

    # Retain every expected measurement,
    # including failed-fit records.
    for s in range(2):

        for b, band in enumerate(
            scene["bands"]
        ):

            row = dict(
                track=track,
                scene_id=scene_id,
                seed=seed,
                separation_px=separation,
                flux_ratio_f277w=ratio,
                method=method,
                source=s + 1,
                band=str(band).lower(),
                bulge_fraction=fractions[s, b],
                true_flux=truth[s, b],
                runtime_seconds=runtime,
                status=status,
                failure_reason=failure_reason,
            )

            if status == "success":

                row.update(
                    estimated_flux=flux[s, b],

                    signed_error=signed_error[s, b],

                    absolute_error=abs(
                        signed_error[s, b]
                    ),

                    combined_signed_error=(
                        combined_error[b]
                    ),

                    chi2=chi2,

                    fitted_sersic_n=(
                        diagnostics["shapes"][s].get(
                            "fitted_sersic_n",
                            "",
                        )
                    ),

                    optimizer_success=(
                        diagnostics["optimizer_success"]
                    ),
                )

            rows.append(row)

    if status == "success":

        print(
            f"{track} | "
            f"scene {scene_id} | "
            f"{method} | "
            f"error={np.abs(signed_error).mean()*100:.2f}% | "
            f"chi2={chi2:.5f} | "
            f"converged={diagnostics['optimizer_success']} | "
            f"{runtime:.2f}s",
            flush=True,
        )

    else:

        print(
            f"{track} | "
            f"scene {scene_id} | "
            f"{method} | "
            f"FAILED: {failure_reason}",
            flush=True,
        )

    return rows


# ---------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------

def summarize(rows):
    """Summarize each morphology track separately."""

    for track in BULGE_FRACTIONS:

        print(f"\n--- {track} ---")

        for method in METHODS:

            group = [
                r for r in rows
                if (
                    r["track"] == track
                    and r["method"] == method
                )
            ]

            good = [
                r for r in group
                if r["status"] == "success"
            ]

            print(
                f"{method}: "
                f"{len(good)//6}/{len(group)//6} "
                f"successful fits"
            )

            if not good:
                continue

            errors = np.array([
                r["absolute_error"]
                for r in good
            ])

            combined = np.array([
                abs(r["combined_signed_error"])
                for r in good
            ])

            faint = [
                r["absolute_error"]
                for r in good
                if (
                    r["source"] == 2
                    and r["flux_ratio_f277w"] == 0.1
                )
            ]

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
                f"  Combined flux error: "
                f"{combined.mean()*100:.2f}%"
            )

            if faint:

                print(
                    f"  Faint companion error: "
                    f"{np.mean(faint)*100:.2f}%"
                )


def main():
    """Run the composite morphology benchmark."""

    grid = list(
        product(
            SEPARATIONS,
            FLUX_RATIOS,
            range(REPETITIONS),
        )
    )

    cases = [
        (
            i,
            *grid[i - 1][:2],
            BASE_SEED + i,
        )
        for i in SCENE_IDS
    ]

    planned = (
        len(cases)
        * len(BULGE_FRACTIONS)
        * len(METHODS)
    )

    print(
        "Composite Morphology Benchmark"
    )

    print(
        f"Planned fits: {planned} | "
        f"Expected measurements: {planned*6}",
        flush=True,
    )

    rows = []

    # Exclusive creation prevents overwriting
    # previous experimental results.
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

            scene_id, sep, ratio, seed = case

            config = create_configuration(
                sep,
                ratio,
                seed,
            )

            # Generate physical components only once.
            # Reuse them across both morphology tracks.
            bulge, disk = (
                create_composite_components(
                    config
                )
            )

            for track, fractions in (
                BULGE_FRACTIONS.items()
            ):

                scene = create_composite_scene(
                    bulge,
                    disk,
                    fractions,
                )

                for method in METHODS:

                    batch = evaluate(
                        scene,
                        track,
                        case,
                        method,
                    )

                    writer.writerows(batch)

                    stream.flush()

                    rows.extend(batch)

    summarize(rows)

    failed = sum(
        r["status"] == "failed"
        for r in rows
    ) // 6

    print(
        f"\nRecorded fits: "
        f"{len(rows)//6}/{planned}"
    )

    print(
        f"Failed fits: {failed}"
    )

    print(
        f"Results saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
