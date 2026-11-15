
"""Phase 8.4: matched flexible Sersic development benchmark.

Run only the new baseline on the five prespecified base scenes
and four morphology families.

Compare against Phase 8.3 CSV, which includes the Phase 7.5 references.

Use --smoke for one scene; it does not write a CSV.
"""

import argparse
import csv

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

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

from src.simulation.simulate_blends import (
    simulate_scene,
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SCENE_IDS = (1, 16, 46, 81, 96)

FAMILIES = {
    "gaussian": None,
    "sersic1": 1.0,
    "sersic2": 2.0,
    "sersic4": 4.0,
}

METHODS = (
    "fixed_gaussian",
    "scarlet2",
    "scarlet2_neural",
    "flexible_gaussian",
)

BANDS = (
    "f150w",
    "f277w",
    "f444w",
)

FIELDS = [
    "morphology",
    "scene_id",
    "seed",
    "separation_px",
    "flux_ratio_f277w",
    "source",
    "band",
    "true_flux",
    "estimated_flux",
    "signed_error",
    "absolute_error",
    "combined_signed_error",
    "chi2",
    "sigma_x",
    "sigma_y",
    "rho",
    "sigma_major",
    "sigma_minor",
    "angle_deg",
    "fitted_sersic_n",
    "background",
    "optimizer_success",
    "optimizer_message",
    "iterations",
    "function_evaluations",
    "valid_starts",
    "converged_starts",
    "start_losses",
    "runtime_seconds",
    "status",
    "failure_reason",
] + [
    f"{m}_{k}"
    for m in METHODS
    for k in (
        "flux",
        "absolute_error",
        "chi2"
    )
]


# ---------------------------------------------------------
# Load previous benchmark measurements
# ---------------------------------------------------------

def load_reference(family, cases):
    """Validate and reuse Phase 8.3 results.

    No previously evaluated method is rerun.
    """

    path = (
        ROOT
        / "results"
        / "benchmark_flexible_gaussian_phase8_3.csv"
    )

    configs = {
        i: (sep, ratio, seed)
        for i, sep, ratio, seed in cases
    }

    expected = {
        (i, m, s, b)
        for i in configs
        for m in METHODS
        for s in (1, 2)
        for b in BANDS
    }

    refs = {}

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as stream:

        for row in csv.DictReader(stream):

            i = int(row["scene_id"])

            if (
                row["morphology"] != family
                or i not in configs
            ):
                continue

            s = int(row["source"])

            band = row["band"].lower()

            sep, ratio, seed = configs[i]

            # Validate scene identity and
            # original experimental configuration.
            if (
                row["status"] != "success"
                or int(row["seed"]) != seed
                or float(
                    row["separation_px"]
                ) != sep
                or float(
                    row["flux_ratio_f277w"]
                ) != ratio
            ):
                raise ValueError(
                    f"Invalid Phase 8.3 row: "
                    f"{family}, {i}, {s}, {band}"
                )

            truth = float(row["true_flux"])

            for m in METHODS:

                key = i, m, s, band

                if (
                    key not in expected
                    or key in refs
                ):
                    raise ValueError(
                        f"Unexpected or duplicate "
                        f"reference: {key}"
                    )

                if m == "flexible_gaussian":

                    f, e, c = (
                        float(row[k])
                        for k in (
                            "estimated_flux",
                            "absolute_error",
                            "chi2",
                        )
                    )

                else:

                    f, e, c = (
                        float(row[f"{m}_{k}"])
                        for k in (
                            "flux",
                            "absolute_error",
                            "chi2",
                        )
                    )

                # Verify numerical consistency
                # of the archived flux metrics.
                if (
                    not np.isfinite(
                        [truth, f, e, c]
                    ).all()
                    or truth <= 0
                    or c < 0
                    or not np.isclose(
                        e,
                        abs(f / truth - 1),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                ):
                    raise ValueError(
                        f"Invalid reference "
                        f"metrics: {key}"
                    )

                refs[key] = dict(
                    true_flux=truth,
                    estimated_flux=f,
                    absolute_error=e,
                    chi2=c,
                )

    if set(refs) != expected:
        raise ValueError(
            f"Missing reference measurements: "
            f"{expected - set(refs)}"
        )

    return refs


# ---------------------------------------------------------
# Evaluate one matched simulation scene
# ---------------------------------------------------------

def run_scene(
    family,
    n,
    case,
    refs
):
    """Generate matched inputs and evaluate all six fluxes."""

    i, sep, ratio, seed = case

    config = create_configuration(
        sep,
        ratio,
        seed
    )

    # Preserve the original Gaussian simulation.
    # Change only the intrinsic morphology family
    # for Sersic experiments.
    if n is None:

        scene = simulate_scene(
            **config
        )

    else:

        scene = simulate_scene(
            **config,
            sersic_n=n
        )

    truth = scene["input_flux"]

    bands = tuple(
        str(b).lower()
        for b in scene["bands"]
    )

    if (
        bands != BANDS
        or truth.shape != (2, 3)
    ):
        raise ValueError(
            "Unexpected simulator layout"
        )

    # Confirm that all archived methods
    # used the same scene ground truth.
    for s in range(2):

        for b, band in enumerate(bands):

            for m in METHODS:

                if not np.isclose(
                    refs[
                        i, m, s + 1, band
                    ]["true_flux"],
                    truth[s, b],
                    rtol=1e-8,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"Archive truth mismatch: "
                        f"{family}, {i}"
                    )

    start = perf_counter()

    status = "success"
    reason = ""

    try:

        # Only observation-side information
        # is passed to the fitting algorithm.
        flux, model, d = fit_flexible_sersic(
            scene["observed"],
            scene["psfs"],
            scene["positions"],
            scene["noise_sigma"],
        )

        if (
            flux.shape != truth.shape
            or model.shape
            != scene["observed"].shape
        ):
            raise ValueError(
                "Unexpected output shape"
            )

        # Ground truth is used exclusively
        # for evaluating recovered flux.
        signed = (
            flux - truth
        ) / truth

        combined = (
            flux.sum(axis=0)
            - truth.sum(axis=0)
        ) / truth.sum(axis=0)

        chi = float(
            np.mean(
                (
                    (
                        scene["observed"]
                        - model
                    )
                    / scene["noise_sigma"][
                        :, None, None
                    ]
                ) ** 2
            )
        )

        if (
            not np.isfinite(signed).all()
            or not np.isfinite(chi)
        ):
            raise ValueError(
                "Invalid evaluation metrics"
            )

    except Exception as exc:

        status = "failed"

        reason = (
            f"{type(exc).__name__}: {exc}"
        )

    elapsed = perf_counter() - start

    rows = []

    # Preserve all six source-band records,
    # including records for failed fits.
    for s in range(2):

        for b, band in enumerate(bands):

            row = dict(
                morphology=family,
                scene_id=i,
                seed=seed,
                separation_px=sep,
                flux_ratio_f277w=ratio,
                source=s + 1,
                band=band,
                true_flux=truth[s, b],
                runtime_seconds=elapsed,
                status=status,
                failure_reason=reason,
            )

            # Reuse all previous baseline
            # measurements from the archive.
            for m in METHODS:

                ref = refs[
                    i, m, s + 1, band
                ]

                row.update({
                    f"{m}_flux":
                        ref["estimated_flux"],

                    f"{m}_absolute_error":
                        ref["absolute_error"],

                    f"{m}_chi2":
                        ref["chi2"],
                })

            if status == "success":

                row.update(
                    estimated_flux=flux[s, b],
                    signed_error=signed[s, b],
                    absolute_error=abs(
                        signed[s, b]
                    ),
                    combined_signed_error=combined[b],
                    chi2=chi,
                    background=d["background"][b],

                    **d["shapes"][s],

                    **{
                        k: d[k]
                        for k in (
                            "optimizer_success",
                            "optimizer_message",
                            "iterations",
                            "function_evaluations",
                            "valid_starts",
                            "converged_starts",
                            "start_losses",
                        )
                    },
                )

            rows.append(row)

    if status == "success":

        fixed_error = 100 * np.mean([
            r["fixed_gaussian_absolute_error"]
            for r in rows
        ])

        sersic_error = (
            100 * np.abs(signed).mean()
        )

        print(
            f"{family} | scene {i} | "
            f"fixed={fixed_error:.2f}% | "
            f"sersic={sersic_error:.2f}% | "
            f"chi2={chi:.5f} | "
            f"converged={d['optimizer_success']} | "
            f"{elapsed:.2f}s",
            flush=True,
        )

    else:

        print(
            f"{family} | scene {i} | "
            f"FAILED: {reason}",
            flush=True,
        )

    return rows


# ---------------------------------------------------------
# Main benchmark execution
# ---------------------------------------------------------

def main():
    """Execute the controlled Phase 8.4 benchmark."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    grid = list(
        product(
            SEPARATIONS,
            FLUX_RATIOS,
            range(REPETITIONS),
        )
    )

    ids = (
        SCENE_IDS[:1]
        if args.smoke
        else SCENE_IDS
    )

    cases = [
        (
            i,
            *grid[i - 1][:2],
            BASE_SEED + i
        )
        for i in ids
    ]

    families = (
        {"gaussian": None}
        if args.smoke
        else FAMILIES
    )

    # Load and validate previous results
    # before executing the new method.
    refs = {
        family: load_reference(
            family,
            cases
        )
        for family in families
    }

    planned = (
        len(cases)
        * len(families)
    )

    print(
        f"Planned new fits: {planned} | "
        f"Expected measurements: {planned * 6}",
        flush=True,
    )

    rows = []

    def run(
        writer=None,
        stream=None
    ):

        for family, n in families.items():

            for case in cases:

                batch = run_scene(
                    family,
                    n,
                    case,
                    refs[family],
                )

                rows.extend(batch)

                if writer is not None:

                    writer.writerows(batch)

                    stream.flush()

    if args.smoke:

        # Smoke mode does not write a CSV.
        run()

    else:

        path = (
            ROOT
            / "results"
            / "benchmark_flexible_sersic_phase8_4.csv"
        )

        # Exclusive creation protects
        # existing benchmark results.
        with path.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=FIELDS
            )

            writer.writeheader()

            run(
                writer,
                stream
            )

        print(
            f"Results saved: {path}"
        )

    # -----------------------------------------------------
    # Comparative performance summary
    # -----------------------------------------------------

    print(
        "\n--- Matched successful-fit comparison ---"
    )

    for family in families:

        group = [
            r for r in rows
            if (
                r["morphology"] == family
                and r["status"] == "success"
            )
        ]

        if not group:
            continue

        metrics = {
            m: 100 * np.mean([
                r[f"{m}_absolute_error"]
                for r in group
            ])
            for m in METHODS
        }

        metrics["flexible_sersic"] = (
            100 * np.mean([
                r["absolute_error"]
                for r in group
            ])
        )

        print(
            f"{family}: "
            + ", ".join(
                f"{m}={v:.2f}%"
                for m, v in metrics.items()
            )
        )

    failed = sum(
        r["status"] == "failed"
        for r in rows
    ) // 6

    converged = sum(
        r["status"] == "success"
        and r["optimizer_success"]
        for r in rows
    ) // 6

    print(
        f"Recorded fits: {len(rows) // 6}/{planned} | "
        f"Failed fits: {failed} | "
        f"Selected starts converged: {converged}"
    )


if __name__ == "__main__":
    main()
