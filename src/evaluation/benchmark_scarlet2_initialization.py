
"""Phase 8.1–8.2: controlled scarlet2 initialization,
iteration and morphology support study.

Reuse existing simulation and fitting modules.
Run only new fits and compare against archived benchmarks.
"""

import argparse
import csv
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from src.baselines.baseline_scarlet2 import (
    fit_scarlet2,
    INITIALIZATION_MIX,
)

from src.evaluation.benchmark_baseline import (
    create_configuration,
    SEPARATIONS,
    FLUX_RATIOS,
    REPETITIONS,
    BASE_SEED,
)

from src.evaluation.benchmark_scarlet2 import (
    output_path_for,
)

from src.simulation.simulate_blends import simulate_scene


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SCENE_IDS = (1, 16, 46, 81, 96)

FAMILIES = {
    "gaussian": None,
    "sersic1": 1.0,
}

METHODS = (
    "scarlet2",
    "scarlet2_neural",
)

SUPPORT_FAMILIES = {
    "sersic1": 1.0,
    "sersic2": 2.0,
    "sersic4": 4.0,
}

SUPPORT_SIZES = (49, 65)

FIELDS = (
    "morphology",
    "scene_id",
    "seed",
    "separation_px",
    "flux_ratio_f277w",
    "method",
    "initialization",
    "mix_fraction",
    "max_iterations",
    "source",
    "band",
    "true_flux",
    "archive_warm_flux",
    "archive_warm_absolute_error",
    "archive_warm_chi2",
    "warm_start_nominal_flux",
    "initial_model_flux",
    "estimated_flux",
    "signed_error",
    "absolute_error",
    "combined_signed_error",
    "initial_chi2",
    "chi2",
    "iterations",
    "stopping_reason",
    "first_loss",
    "last_loss",
    "last_max_change",
    "peak_offset_px",
    "positive_centroid_offset_px",
    "negative_mass_fraction",
    "runtime_seconds",
    "status",
    "failure_reason",
    "box_size",
    "bbox_height",
    "bbox_width",
    "bbox_y",
    "bbox_x",
    "edge_positive_mass_fraction",
)


def configurations(ids):
    """Reuse original benchmark scene IDs and seeds."""

    grid = list(
        product(
            SEPARATIONS,
            FLUX_RATIOS,
            range(REPETITIONS),
        )
    )

    return [
        (i, *grid[i - 1][:2], BASE_SEED + i)
        for i in ids
    ]


def load_reference(family, cases, methods=METHODS):
    """Read archived warm-start measurements."""

    path = output_path_for(family)

    expected = {
        (i, method, source, band)
        for i, _, _, _ in cases
        for method in methods
        for source in (1, 2)
        for band in ("f150w", "f277w", "f444w")
    }

    case_map = {
        i: (sep, ratio, seed)
        for i, sep, ratio, seed in cases
    }

    reference = {}

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:

        for row in csv.DictReader(file):

            key = (
                int(row["scene_id"]),
                row["method"],
                int(row["source"]),
                row["band"].lower(),
            )

            if key not in expected:
                continue

            sep, ratio, seed = case_map[key[0]]

            if (
                key in reference
                or row["status"] != "success"
                or int(row["seed"]) != seed
                or float(row["separation_px"]) != sep
                or float(row["flux_ratio_f277w"]) != ratio
                or (
                    family != "gaussian"
                    and row.get("morphology") != family
                )
            ):
                raise ValueError(
                    f"Invalid archived reference: {path}: {key}"
                )

            for field in (
                "true_flux",
                "estimated_flux",
                "absolute_error",
                "chi2",
            ):
                row[field] = float(row[field])

                if not np.isfinite(row[field]):
                    raise ValueError(
                        f"Nonfinite archived reference: "
                        f"{key}, {field}"
                    )

            if (
                row["true_flux"] <= 0
                or not np.isclose(
                    row["absolute_error"],
                    abs(
                        row["estimated_flux"]
                        / row["true_flux"] - 1
                    ),
                    rtol=1e-5,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    f"Invalid archived flux metrics: {key}"
                )

            reference[key] = row

    if set(reference) != expected:
        raise ValueError(
            f"Missing archived references in {path}: "
            f"{expected - set(reference)}"
        )

    return reference


def run_fit(
    scene,
    family,
    case,
    method,
    reference,
    args,
    box_size=31,
):
    """Fit using observation-side information only.

    Ground truth is used exclusively for evaluation.
    """

    i, sep, ratio, seed = case

    bands = [
        str(b).lower()
        for b in scene["bands"]
    ]

    truth = scene["input_flux"]

    # Validate matching reference measurements.
    for s in range(2):
        for b, band in enumerate(bands):

            ref = reference[
                i, method, s + 1, band
            ]

            if not np.isclose(
                truth[s, b],
                ref["true_flux"],
                rtol=1e-8,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Archive truth mismatch: "
                    f"{family}, scene {i}"
                )

    start = perf_counter()

    status = "success"
    reason = ""

    try:

        flux, reconstruction, d = fit_scarlet2(
            scene["observed"],
            scene["psfs"],
            scene["positions"],
            scene["noise_sigma"],
            scene["bands"],
            use_neural_prior=(
                method == "scarlet2_neural"
            ),
            max_iterations=args.iterations,
            initialization_mode=args.initialization,
            collect_diagnostics=True,
            box_size=box_size,
        )

        # Verify actual morphology support.
        if any(
            (
                m["bbox_height"],
                m["bbox_width"],
            ) != (box_size, box_size)
            for m in d["morphology_stats"]
        ):
            raise ValueError(
                "Actual source support differs "
                "from requested box"
            )

        if (
            flux.shape != truth.shape
            or reconstruction.shape
            != scene["observed"].shape
        ):
            raise ValueError(
                "Unexpected model output shape"
            )

        # Individual and combined flux errors.
        signed = (flux - truth) / truth

        combined = (
            flux.sum(axis=0)
            - truth.sum(axis=0)
        ) / truth.sum(axis=0)

        # Use the same reconstruction metric
        # as the original benchmark.
        chi2 = float(
            np.mean(
                (
                    (
                        scene["observed"]
                        - reconstruction
                    )
                    / scene["noise_sigma"][:, None, None]
                ) ** 2
            )
        )

        if (
            not np.all(np.isfinite(signed))
            or not np.isfinite(chi2)
        ):
            raise ValueError(
                "Nonfinite evaluation metrics"
            )

    except Exception as exc:

        status = "failed"

        reason = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"FAILED: {reason}",
            flush=True,
        )

    runtime = perf_counter() - start

    rows = []

    # Record every source-band measurement.
    for s in range(2):

        for b, band in enumerate(bands):

            ref = reference[
                i, method, s + 1, band
            ]

            row = dict(
                morphology=family,
                scene_id=i,
                seed=seed,
                separation_px=sep,
                flux_ratio_f277w=ratio,
                method=method,
                initialization=args.initialization,
                mix_fraction=(
                    INITIALIZATION_MIX
                    if args.initialization
                    == "balanced_warm_start"
                    else 0
                ),
                max_iterations=args.iterations,
                box_size=box_size,
                source=s + 1,
                band=band,
                true_flux=truth[s, b],
                archive_warm_flux=ref[
                    "estimated_flux"
                ],
                archive_warm_absolute_error=ref[
                    "absolute_error"
                ],
                archive_warm_chi2=ref[
                    "chi2"
                ],
                runtime_seconds=runtime,
                status=status,
                failure_reason=reason,
            )

            if status == "success":

                row.update(
                    warm_start_nominal_flux=d[
                        "warm_start_flux"
                    ][s, b],
                    initial_model_flux=d[
                        "initial_flux"
                    ][s, b],
                    estimated_flux=flux[s, b],
                    signed_error=signed[s, b],
                    absolute_error=abs(signed[s, b]),
                    combined_signed_error=combined[b],
                    initial_chi2=d["initial_chi2"],
                    chi2=chi2,
                    iterations=d["iterations"],
                    stopping_reason=d["stopping_reason"],
                    first_loss=d["first_loss"],
                    last_loss=d["last_loss"],
                    last_max_change=d["last_max_change"],
                    **d["morphology_stats"][s],
                )

            rows.append(row)

    if status == "success":

        old = np.mean([
            r["archive_warm_absolute_error"]
            for r in rows
        ])

        print(
            f"{family} | scene {i} | {method}: "
            f"archived {old * 100:.2f}% -> "
            f"new {np.abs(signed).mean() * 100:.2f}% | "
            f"chi2={chi2:.4f} | "
            f"{d['iterations']} steps | "
            f"{d['stopping_reason']} | "
            f"max_change={d['last_max_change']:.6g}",
            flush=True,
        )

    return rows


def summarize(rows):
    """Report paired results by morphology and support."""

    print(
        "\n--- Paired comparison against "
        "archived 31x31 warm start ---"
    )

    groups = sorted({
        (
            r["morphology"],
            r["method"],
            r["box_size"],
        )
        for r in rows
    })

    for family, method, size in groups:

        group = [
            r for r in rows
            if (
                r["morphology"],
                r["method"],
                r["box_size"],
            ) == (family, method, size)
        ]

        good = [
            r for r in group
            if r["status"] == "success"
        ]

        print(
            f"{family} | {method} | "
            f"{size}x{size}: "
            f"{len(good) // 6}/{len(group) // 6} "
            f"fits successful"
        )

        if not good:
            continue

        old = np.mean([
            r["archive_warm_absolute_error"]
            for r in good
        ])

        new = np.mean([
            r["absolute_error"]
            for r in good
        ])

        stops = Counter(
            r["stopping_reason"]
            for r in good
            if r["source"] == 1
            and r["band"] == "f150w"
        )

        print(
            f"  Archived: {old * 100:.2f}% | "
            f"New: {new * 100:.2f}% | "
            f"Stops: {dict(stops)}"
        )


def main():
    """Execute initialization or support ablation."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--initialization",
        choices=(
            "warm_start",
            "balanced_warm_start",
        ),
        default="balanced_warm_start",
    )

    parser.add_argument(
        "--scene-ids",
        nargs="+",
        type=int,
        choices=SCENE_IDS,
        default=SCENE_IDS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--support-study",
        action="store_true",
        help=(
            "Compare 49 and 65 boxes against "
            "archived 31x31 results using "
            "vanilla scarlet2 and Sersic n=1/2/4."
        ),
    )

    args = parser.parse_args()

    if (
        args.iterations < 1
        or len(set(args.scene_ids))
        != len(args.scene_ids)
    ):
        parser.error(
            "Use a positive iteration budget "
            "and unique scene IDs"
        )

    # Preserve all other experimental settings
    # during morphology support ablation.
    if args.support_study:

        if args.iterations != 1000:
            parser.error(
                "Support study requires the "
                "archived 1000-iteration budget"
            )

        args.initialization = "warm_start"

    families = (
        SUPPORT_FAMILIES
        if args.support_study
        else FAMILIES
    )

    methods = (
        ("scarlet2",)
        if args.support_study
        else METHODS
    )

    sizes = (
        SUPPORT_SIZES
        if args.support_study
        else (31,)
    )

    cases = configurations(args.scene_ids)

    refs = {
        family: load_reference(
            family,
            cases,
            methods,
        )
        for family in families
    }

    suffix = "_".join(
        map(str, args.scene_ids)
    )

    name = (
        f"scarlet2_support_49_65_{suffix}.csv"
        if args.support_study
        else (
            f"scarlet2_{args.initialization}_"
            f"{args.iterations}_{suffix}.csv"
        )
    )

    output = (
        args.output
        or ROOT / "results" / name
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    planned = (
        len(cases)
        * len(families)
        * len(methods)
        * len(sizes)
    )

    print(
        f"Planned new fits: {planned} | "
        f"Expected measurements: {planned * 6}"
    )

    print(
        f"Output: {output}",
        flush=True,
    )

    # Exclusive creation protects existing results.
    with output.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for family, n in families.items():

            for case in cases:

                i, sep, ratio, seed = case

                config = create_configuration(
                    sep,
                    ratio,
                    seed,
                )

                # Generate one matched scene for
                # all requested support sizes.
                scene = (
                    simulate_scene(**config)
                    if n is None
                    else simulate_scene(
                        **config,
                        sersic_n=n,
                    )
                )

                for method in methods:

                    for size in sizes:

                        print(
                            f"\n{family} | "
                            f"scene {i} | {method} | "
                            f"{size}x{size}",
                            flush=True,
                        )

                        new_rows = run_fit(
                            scene,
                            family,
                            case,
                            method,
                            refs[family],
                            args,
                            box_size=size,
                        )

                        writer.writerows(new_rows)

                        file.flush()

                        rows.extend(new_rows)

    summarize(rows)

    print(
        f"Recorded fits: "
        f"{len(rows) // 6}/{planned} | "
        f"Failed fits: "
        f"{sum(r['status'] == 'failed' for r in rows) // 6}"
    )

    print(f"Results saved: {output}")


if __name__ == "__main__":
    main()
