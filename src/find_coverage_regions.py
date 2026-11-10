r"""Find small CEERS cutout regions with shared coverage in three NIRCam bands.

Run from the project root:
    .\.venv\Scripts\python.exe src\find_coverage_regions.py

Reads the WHT extensions without copying entire mosaics into memory.
Results are 0-based pixel coordinates, suitable for NumPy image slicing.
"""

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
BANDS = ("f150w", "f277w", "f444w")
COARSE_STEP = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128, help="Square cutout size in pixels")
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--min-spacing", type=int, default=512)
    return parser.parse_args()


def choose_files(folder):
    selected = []
    for band in BANDS:
        matches = list(folder.glob(f"*_{band}_*_i2d.fits"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {band.upper()} i2d FITS in {folder}, found {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def main():
    args = parse_args()
    if args.size <= 0 or args.size % COARSE_STEP:
        raise ValueError(f"--size must be a positive multiple of {COARSE_STEP}")
    if not (0 < args.min_coverage <= 1):
        raise ValueError("--min-coverage must be between 0 and 1")
    if args.max_regions < 1 or args.min_spacing < 0:
        raise ValueError("--max-regions must be positive and --min-spacing nonnegative")

    paths = choose_files(ROOT / "data" / "raw" / "fits")
    with ExitStack() as stack:
        hdus = [stack.enter_context(fits.open(p, memmap=True)) for p in paths]
        weights = [h["WHT"].data for h in hdus]
        shapes = [w.shape for w in weights]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"WHT image dimensions do not match: {shapes}")

        height, width = shapes[0]
        if args.size > min(height, width):
            raise ValueError("Cutout size exceeds the image dimensions")

        # Tiny boolean arrays (about 300 x 663 for the current CEERS mosaics).
        common = np.logical_and.reduce(
            [w[::COARSE_STEP, ::COARSE_STEP] > 0 for w in weights]
        )

        candidates = []
        for y0 in range(0, height - args.size + 1, args.size):
            for x0 in range(0, width - args.size + 1, args.size):
                coarse = common[
                    y0 // COARSE_STEP : (y0 + args.size) // COARSE_STEP,
                    x0 // COARSE_STEP : (x0 + args.size) // COARSE_STEP,
                ]
                if coarse.mean() < args.min_coverage:
                    continue

                # Confirm the complete region at full resolution in all bands.
                shared = np.logical_and.reduce(
                    [
                        w[y0 : y0 + args.size, x0 : x0 + args.size] > 0
                        for w in weights
                    ]
                )
                fraction = float(shared.mean())
                if fraction >= args.min_coverage:
                    candidates.append((fraction, x0, y0))

    # Prefer high-coverage regions while distributing them over the field.
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    selected = []
    for fraction, x0, y0 in candidates:
        if all(
            (x0 - old_x) ** 2 + (y0 - old_y) ** 2 >= args.min_spacing ** 2
            for _, old_x, old_y in selected
        ):
            selected.append((fraction, x0, y0))
            if len(selected) >= args.max_regions:
                break

    if not selected:
        raise RuntimeError("No shared-coverage regions found. Try --min-coverage 0.90")

    output_csv = ROOT / "data" / "processed" / "coverage_regions.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("region_id", "x0", "y0", "size", "shared_coverage"))
        for idx, (fraction, x0, y0) in enumerate(selected, start=1):
            writer.writerow((f"region_{idx:03d}", x0, y0, args.size, f"{fraction:.6f}"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(common, origin="lower", cmap="gray", vmin=0, vmax=1)
    for idx, (_, x0, y0) in enumerate(selected, start=1):
        ax.add_patch(
            Rectangle(
                (x0 / COARSE_STEP, y0 / COARSE_STEP),
                args.size / COARSE_STEP,
                args.size / COARSE_STEP,
                fill=False,
                edgecolor="red",
                linewidth=1.2,
            )
        )
        ax.text(x0 / COARSE_STEP, y0 / COARSE_STEP, str(idx), color="yellow", fontsize=6)
    ax.set_title("Shared F150W / F277W / F444W coverage and candidate regions")
    ax.set_xlabel(f"x / {COARSE_STEP} pixels")
    ax.set_ylabel(f"y / {COARSE_STEP} pixels")
    fig.tight_layout()

    output_png = ROOT / "results" / "coverage_map.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Selected {len(selected)} regions of {args.size} x {args.size} pixels")
    print(f"Saved: {output_csv}")
    print(f"Saved: {output_png}")
    print("Note: coverage alone does not establish that a region contains blended galaxies.")


if __name__ == "__main__":
    main()
