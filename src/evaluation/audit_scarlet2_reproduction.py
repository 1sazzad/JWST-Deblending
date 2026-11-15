"""
Phase 8.1.1: Read-only API inspection and matched baseline reproduction.

Install this module at src/evaluation/audit_scarlet2_reproduction.py.
The existing simulator, fitter, benchmark, pilot and archive CSVs are
not modified. Every run receives a new audit directory.
"""

import argparse
import hashlib
import inspect
import json
import tempfile
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
from itertools import product
from pathlib import Path

import numpy as np
import scarlet2 as sc2

from src.evaluation import benchmark_scarlet2 as benchmark
from src.evaluation.benchmark_baseline import (
    create_configuration, SEPARATIONS, FLUX_RATIOS, REPETITIONS, BASE_SEED,
)
from src.simulation.simulate_blends import simulate_scene
from src.simulation.psf_utils import PSF_DIR, BANDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_IDS = (1, 16, 46, 81, 96)
FAMILIES = ("gaussian", "sersic1")
# Preliminary numerical reproduction tolerances, not scientific thresholds.
RTOL = 1e-4
ATOL = 1e-5
COMPARE_FIELDS = benchmark.FIELDS + [
    "morphology", "archive_status", "archive_estimated_flux",
    "archive_chi2", "flux_difference", "chi2_difference",
    "reproduction_status",
]


def sha256(path):
    """Compute a content fingerprint without changing the input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    """Record actual local dependency versions."""
    packages = ("scarlet2", "jax", "jaxlib", "galaxygrad", "numpy",
                "scipy", "numpyro", "optax", "equinox", "astropy")
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def inspect_api():
    """Print local source and signatures; never run an optimization."""
    print(json.dumps(environment(), indent=2))
    print("scarlet2 package:", sc2.__file__)
    targets = [
        ("relative_step", sc2.relative_step, True),
        ("Scene.fit", sc2.Scene.fit, True),
        ("Frame.from_observations", sc2.Frame.from_observations, True),
        ("Source.__init__", sc2.Source.__init__, True),
        ("Source.__call__", sc2.Source.__call__, True),
        ("measure.flux", sc2.measure.flux, True),
        ("Observation.goodness_of_fit", sc2.Observation.goodness_of_fit, True),
    ]
    # Scene.fit may delegate the actual optimization to this module.
    try:
        infer = import_module("scarlet2.infer")
        if hasattr(infer, "fit"):
            targets.append(("infer.fit", infer.fit, True))
    except ImportError as error:
        print("Optional infer module inspection unavailable:", error)
    for label, target, show_source in targets:
        print("\n---", label, "---")
        try:
            print("Signature:", inspect.signature(target))
            if show_source:
                print(inspect.getsource(target))
        except (TypeError, ValueError, OSError) as error:
            print("Inspection unavailable:", error)


def selected_scenes():
    """Reconstruct original IDs, preserving the archived configuration."""
    grid = product(SEPARATIONS, FLUX_RATIOS, range(REPETITIONS))
    return [(i, sep, ratio)
            for i, (sep, ratio, repeat) in enumerate(grid, start=1)
            if repeat == 0]


def identity(row):
    """A single source-band measurement identity."""
    return (row["scene_id"], row["method"], row["source"],
            row["band"].upper())


def compare(row, old, morphology):
    """Compare new and old metrics without treating agreement as accuracy."""
    result = dict(row, morphology=morphology,
                  archive_status=old["status"],
                  archive_estimated_flux=old["estimated_flux"],
                  archive_chi2=old["chi2"],
                  flux_difference="", chi2_difference="",
                  reproduction_status="not_comparable")
    if row["status"] != "success" or old["status"] != "success":
        return result

    # Refuse to compare mismatched simulated truth/configurations.
    if not np.isclose(row["true_flux"], old["true_flux"],
                      rtol=1e-8, atol=1e-8):
        raise ValueError(f"Truth mismatch for {identity(row)}")
    for key in ("seed", "separation_px", "flux_ratio_f277w"):
        if row[key] != old[key]:
            raise ValueError(f"Configuration mismatch: {key}")
    result["flux_difference"] = row["estimated_flux"] - old["estimated_flux"]
    result["chi2_difference"] = row["chi2"] - old["chi2"]
    same = all(np.isclose(row[key], old[key], rtol=RTOL, atol=ATOL)
               for key in ("estimated_flux", "chi2"))
    result["reproduction_status"] = "within_tolerance" if same else "different"
    return result


def run(smoke):
    """Reproduce the old fits using the unchanged benchmark evaluator."""
    selected = selected_scenes()
    ids = (1,) if smoke else AUDIT_IDS
    subset = [(i, sep, ratio) for i, sep, ratio in selected if i in ids]
    if tuple(i for i, _, _ in subset) != ids:
        raise RuntimeError("Audit IDs do not match the original grid")

    # Validate the two existing archives before starting any fitting.
    archives = {}
    fingerprints = {}
    for family in FAMILIES:
        path = benchmark.output_path_for(family)
        old_rows, completed = benchmark.load_existing_results(
            path, selected, morphology=family
        )
        expected = {(i, method) for i, _, _ in subset
                    for method in benchmark.METHODS}
        if not expected.issubset(completed):
            raise ValueError(f"Archive lacks selected fits: {path}")
        archives[family] = {identity(row): row for row in old_rows}
        fingerprints[str(path)] = sha256(path)

    # Record the exact local code and old archives for provenance.
    source_paths = [
        "src/baselines/baseline_scarlet2.py",
        "src/baselines/baseline_deblend.py",
        "src/evaluation/benchmark_scarlet2.py",
        "src/evaluation/benchmark_baseline.py",
        "src/simulation/simulate_blends.py",
        "src/simulation/psf_utils.py",
        "src/evaluation/audit_scarlet2_reproduction.py",
    ]
    for relative in source_paths:
        path = PROJECT_ROOT / relative
        fingerprints[str(path)] = sha256(path)

    # Fingerprint the empirical PSFs actually used by the simulator.
    for band in BANDS:
        path = PSF_DIR / f"{band}_psf.fits"
        fingerprints[str(path)] = sha256(path)

    root = PROJECT_ROOT / "results" / "audit"
    root.mkdir(parents=True, exist_ok=True)
    # A new directory per invocation prevents accidental overwrite.
    output = Path(tempfile.mkdtemp(prefix="phase8_1_repro_", dir=root))
    manifest = {
        "environment": environment(), "scene_ids": list(ids),
        "families": list(FAMILIES), "methods": list(benchmark.METHODS),
        "max_iterations": benchmark.MAX_ITERATIONS,
        "rtol": RTOL, "atol": ATOL, "input_sha256": fingerprints,
        "status": "started",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    csv_path = output / "reproduction.csv"
    rows = []
    planned = len(subset) * len(FAMILIES) * len(benchmark.METHODS)
    print(f"Output: {output}", flush=True)
    print(f"Planned fits: {planned}; expected rows: {planned * 6}", flush=True)
    try:
        for family in FAMILIES:
            n, _, _ = benchmark.MORPHOLOGY_CONFIG[family]
            for scene_id, separation, ratio in subset:
                seed = BASE_SEED + scene_id
                config = create_configuration(separation, ratio, seed)
                scene = (simulate_scene(**config) if n is None
                         else simulate_scene(**config, sersic_n=n))
                for method in benchmark.METHODS:
                    print(f"\n{family} | scene {scene_id} | {method}", flush=True)
                    # The original evaluator invokes exactly the archived fitter.
                    fresh = benchmark.evaluate_method(
                        scene, scene_id, seed, separation, ratio, method
                    )
                    if len(fresh) != 6:
                        raise RuntimeError("Unexpected measurement count")
                    rows.extend(compare(row, archives[family][identity(row)], family)
                                for row in fresh)
                    # Reuse the benchmark's atomic writer, but only in audit/.
                    benchmark.save_results(rows, csv_path, COMPARE_FIELDS)
        if len(rows) != planned * 6:
            raise RuntimeError("Unexpected final measurement count")
        counts = {}
        for row in rows:
            status = row["reproduction_status"]
            counts[status] = counts.get(status, 0) + 1
        manifest["comparison_counts"] = counts
        manifest["status"] = "completed"
        print("\nComparison counts:", counts)
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        changed = [name for name, digest in fingerprints.items()
                   if not Path(name).is_file() or sha256(Path(name)) != digest]
        manifest["changed_inputs"] = changed
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Input files changed:", changed, flush=True)
        print("Audit output:", output, flush=True)
    if changed:
        raise RuntimeError("One or more input files changed during the run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect-only", action="store_true",
                       help="Print installed API details; do not fit.")
    modes.add_argument("--smoke", action="store_true",
                       help="Reproduce scene 1 in both families.")
    args = parser.parse_args()
    if args.inspect_only:
        inspect_api()
    else:
        run(args.smoke)


if __name__ == "__main__":
    main()
