"""
Module: visualize_morphologies.py

Purpose:
    Compare Gaussian and Sersic galaxy morphologies
    under identical observational conditions.

    Investigate how much intrinsic source flux is
    contained within different morphology box sizes.

Outputs:
    1. Comparative morphology visualization.
    2. Intrinsic morphology support diagnostics.

This module does not perform galaxy deblending.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

from src.simulation.simulate_blends import simulate_scene


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "morphology_comparison.png"
)

BAND = "F277W"

MORPHOLOGIES = (
    ("Gaussian", None),
    ("Sersic n=1", 1.0),
    ("Sersic n=2", 2.0),
    ("Sersic n=4", 4.0)
)

BOX_SIZES = (31, 49, 65)


def get_box_bounds(position, box_size, image_shape):
    """
    Calculate a square morphology support region.

    Parameters
    ----------
    position:
        Source center in (x, y) coordinates.

    box_size:
        Width and height of the square support region.

    image_shape:
        Shape of the two-dimensional intrinsic image.

    Returns
    -------
    tuple:
        Clipped boundaries (x_min, x_max, y_min, y_max).

    Notes
    -----
    The nearest pixel is used as the box center.
    This is an illustrative support diagnostic,
    not an exact scarlet2 bounding-box reproduction.
    """

    x, y = position

    center_x = int(np.floor(x + 0.5))
    center_y = int(np.floor(y + 0.5))

    half = box_size // 2

    height, width = image_shape

    x_min = max(0, center_x - half)
    x_max = min(width, center_x + half + 1)

    y_min = max(0, center_y - half)
    y_max = min(height, center_y + half + 1)

    return x_min, x_max, y_min, y_max


def calculate_support_fraction(
    intrinsic_image,
    position,
    box_size,
    input_flux
):
    """
    Measure the fraction of simulated input flux
    contained within a specified morphology box.

    Calculations are performed before PSF convolution.
    """

    x_min, x_max, y_min, y_max = get_box_bounds(
        position,
        box_size,
        intrinsic_image.shape
    )

    enclosed_flux = intrinsic_image[
        y_min:y_max,
        x_min:x_max
    ].sum()

    return float(enclosed_flux / input_flux)


def report_morphology_support(scenes, band_index):
    """
    Report enclosed and excluded intrinsic source flux
    for different morphology support sizes.
    """

    print("\n--- Morphology Support Analysis ---")
    print(f"Band: {BAND}")

    for name, scene in scenes:

        print(f"\nMorphology: {name}")

        for source_index, position in enumerate(
            scene["positions"]
        ):

            intrinsic_image = scene["intrinsic"][
                source_index,
                band_index
            ]

            input_flux = scene["input_flux"][
                source_index,
                band_index
            ]

            print(f"\nSource {source_index + 1}")

            for box_size in BOX_SIZES:

                fraction = calculate_support_fraction(
                    intrinsic_image,
                    position,
                    box_size,
                    input_flux
                )

                enclosed = fraction * 100

                excluded = (1 - fraction) * 100

                print(
                    f"{box_size}x{box_size} box | "
                    f"Enclosed: {enclosed:.6f}% | "
                    f"Excluded: {excluded:.6f}%"
                )


def create_comparison_figure(scenes, band_index):
    """
    Generate a matched morphology comparison.

    Each row represents one morphology family.

    Columns show:
        1. Intrinsic source A.
        2. Intrinsic source B.
        3. Noiseless PSF-convolved blend.
        4. Noisy observed blend.

    A shared symmetric logarithmic intensity scale
    allows comparison of bright cores, faint wings,
    and negative observational noise.
    """

    figure, axes = plt.subplots(
        len(scenes),
        4,
        figsize=(16, 15),
        constrained_layout=True
    )

    column_titles = (
        "Intrinsic Source A",
        "Intrinsic Source B",
        "Noiseless Blend",
        "Observed Blend"
    )

    # Use a common intensity scale across all rows
    # and columns to avoid misleading comparisons.
    max_intensity = max(
        float(
            np.max(
                np.abs(scene[key])
            )
        )
        for _, scene in scenes
        for key in (
            "intrinsic",
            "noiseless",
            "observed"
        )
    )

    noise_level = float(
        scenes[0][1]["noise_sigma"][band_index]
    )

    normalization = SymLogNorm(
        linthresh=noise_level,
        vmin=-max_intensity,
        vmax=max_intensity
    )

    image_handle = None

    for row, (name, scene) in enumerate(scenes):

        images = (
            scene["intrinsic"][0, band_index],
            scene["intrinsic"][1, band_index],
            scene["noiseless"][band_index],
            scene["observed"][band_index]
        )

        for column, image in enumerate(images):

            axis = axes[row, column]

            image_handle = axis.imshow(
                image,
                origin="lower",
                cmap="magma",
                norm=normalization
            )

            # Draw the illustrative 31x31 support box
            # on each individual intrinsic source.
            if column < 2:

                position = scene["positions"][column]

                bounds = get_box_bounds(
                    position,
                    31,
                    image.shape
                )

                x_min, x_max, y_min, y_max = bounds

                rectangle = Rectangle(
                    (x_min - 0.5, y_min - 0.5),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1.2,
                    edgecolor="cyan",
                    facecolor="none",
                    linestyle="--"
                )

                axis.add_patch(rectangle)

            if row == 0:
                axis.set_title(column_titles[column])

            if column == 0:
                axis.set_ylabel(name, fontsize=12)

            axis.set_xticks([])
            axis.set_yticks([])

    figure.colorbar(
        image_handle,
        ax=axes.ravel().tolist(),
        shrink=0.7,
        label="Simulated pixel intensity"
    )

    figure.suptitle(
        f"Matched Galaxy Morphology Comparison — {BAND}\n"
        "Cyan boxes: 31x31 intrinsic morphology support",
        fontsize=16
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"\nFigure saved: {OUTPUT_PATH}")


def main():
    """
    Generate matched pilot scenes, evaluate intrinsic
    morphology support, and save the comparison figure.
    """

    scenes = []

    for name, sersic_n in MORPHOLOGIES:

        print(f"Generating {name}...")

        scene = simulate_scene(
            sersic_n=sersic_n,
            seed=42
        )

        scenes.append((name, scene))

    bands = [
        str(band).upper()
        for band in scenes[0][1]["bands"]
    ]

    band_index = bands.index(BAND)

    report_morphology_support(
        scenes,
        band_index
    )

    create_comparison_figure(
        scenes,
        band_index
    )


if __name__ == "__main__":
    main()