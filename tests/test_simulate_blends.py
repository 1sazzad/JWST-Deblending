"""
Unit and scientific validation tests for the galaxy simulator.

Tests cover:
    1. Backward compatibility with the original Gaussian pilot.
    2. Sersic morphology generation.
    3. Integrated flux accounting.
    4. Simulation reproducibility.
    5. Invalid parameter handling.
"""

import unittest
from pathlib import Path

import numpy as np

from src.simulation.simulate_blends import (
    simulate_scene,
    sersic_galaxy
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PILOT_PATH = (
    PROJECT_ROOT
    / "data/processed/synthetic/pilot_scene_0001.npz"
)


class TestSimulateBlends(unittest.TestCase):
    """Validate Gaussian and Sersic galaxy simulations."""

    @classmethod
    def setUpClass(cls):
        """
        Generate reusable test scenes once to reduce
        repeated simulation costs.
        """
        cls.reference = simulate_scene()

        cls.sersic_scenes = {
            n: simulate_scene(sersic_n=n)
            for n in (1.0, 2.0, 4.0)
        }

    def test_gaussian_regression(self):
        """
        Ensure that adding Sersic support has not changed
        the original Gaussian pilot experiment.
        """
        self.assertTrue(
            PILOT_PATH.exists(),
            "Original Gaussian pilot file is missing."
        )

        with np.load(PILOT_PATH) as saved:

            for key in (
                "observed",
                "noiseless",
                "components",
                "input_flux"
            ):

                with self.subTest(key=key):

                    np.testing.assert_allclose(
                        self.reference[key],
                        saved[key],
                        rtol=1e-7,
                        atol=1e-9
                    )

    def test_sersic_morphologies(self):
        """
        Validate image dimensions, finite values and
        actual morphology changes for each Sersic index.
        """
        for n, scene in self.sersic_scenes.items():

            with self.subTest(sersic_n=n):

                self.assertEqual(
                    scene["observed"].shape,
                    (3, 128, 128)
                )

                self.assertEqual(
                    scene["components"].shape,
                    (2, 3, 128, 128)
                )

                self.assertTrue(
                    np.all(np.isfinite(scene["observed"]))
                )

                self.assertTrue(
                    np.all(np.isfinite(scene["components"]))
                )

                # Verify that Sersic mode actually changes
                # the noiseless galaxy morphology.
                self.assertFalse(
                    np.allclose(
                        scene["noiseless"],
                        self.reference["noiseless"]
                    )
                )

    def test_flux_accounting(self):
        """
        Verify component reconstruction and integrated
        flux accounting for each morphology configuration.
        """
        for n, scene in self.sersic_scenes.items():

            with self.subTest(sersic_n=n):

                components = scene["components"]
                truth = scene["input_flux"]
                visible = scene["visible_flux"]

                # Individual source components must sum
                # to the noiseless blended observation.
                np.testing.assert_allclose(
                    components.sum(axis=0),
                    scene["noiseless"],
                    rtol=1e-6,
                    atol=1e-8
                )

                # Verify the recorded visible flux.
                np.testing.assert_allclose(
                    components.sum(axis=(-2, -1)),
                    visible,
                    rtol=1e-6,
                    atol=1e-8
                )

                # Input flux must remain unchanged.
                np.testing.assert_allclose(
                    truth,
                    self.reference["input_flux"]
                )

                self.assertTrue(
                    np.all(np.isfinite(visible))
                )

                self.assertTrue(np.all(visible > 0))

                # Cropping must not introduce additional
                # physical flux beyond numerical tolerance.
                self.assertTrue(
                    np.all(visible <= truth * (1 + 1e-6))
                )

                # Report flux lost outside the cutout.
                flux_loss = (truth - visible) / truth * 100

                print(
                    f"\nSersic n={n}: Maximum flux loss = "
                    f"{flux_loss.max():.8f}%"
                )

    def test_reproducibility(self):
        """
        Identical simulation parameters and random seeds
        must reproduce the same observed image.
        """
        first = simulate_scene(
            sersic_n=2.0,
            seed=42
        )

        second = simulate_scene(
            sersic_n=2.0,
            seed=42
        )

        np.testing.assert_array_equal(
            first["observed"],
            second["observed"]
        )

    def test_invalid_sersic_index(self):
        """
        Reject non-positive or non-finite Sersic indices.
        """
        for invalid_n in (0, -1, np.nan, np.inf):

            with self.subTest(sersic_n=invalid_n):

                with self.assertRaises(ValueError):

                    simulate_scene(sersic_n=invalid_n)


    def test_sersic_sampling_convergence(self):
        """
        Evaluate numerical convergence of pixel-integrated
        Sersic profiles at different sampling resolutions.

        Compare 8x8 and 32x32 subpixel sampling against
        a higher-resolution 64x64 numerical reference.

        Test compact, intermediate and extended profiles
        at different subpixel center positions.
        """

        # Representative morphology configurations:
        # center_x, center_y, sigma_x, sigma_y.
        configurations = (
            (32.0, 32.0, 1.5, 1.2),
            (32.5, 32.5, 2.5, 1.8),
            (32.23, 32.71, 4.0, 3.0)
        )

        # Preliminary numerical acceptance threshold.
        # The normalized L1 difference must remain
        # below 0.2% for 32x32 versus 64x64 sampling.
        tolerance = 0.002

        print("\n--- Sersic Sampling Convergence ---")

        for n in (1.0, 2.0, 4.0):

            for x0, y0, sx, sy in configurations:

                with self.subTest(
                    sersic_n=n,
                    center=(x0, y0),
                    scale=(sx, sy)
                ):

                    profiles = {}

                    # Generate the same intrinsic morphology
                    # using three sampling resolutions.
                    for sampling in (8, 32, 64):

                        profile = sersic_galaxy(
                            64,
                            x0,
                            y0,
                            sx,
                            sy,
                            25.0,
                            n=n,
                            oversample=sampling
                        )

                        # Verify numerical stability.
                        self.assertTrue(
                            np.all(np.isfinite(profile))
                        )

                        # Verify physical non-negativity.
                        self.assertTrue(
                            np.all(profile >= 0)
                        )

                        # Verify integrated normalization.
                        np.testing.assert_allclose(
                            profile.sum(),
                            1.0,
                            rtol=1e-10,
                            atol=1e-10
                        )

                        profiles[sampling] = profile

                    # Use the finest sampled profile
                    # as the numerical reference.
                    reference = profiles[64]

                    # Measure normalized L1 differences.
                    error_8 = np.abs(
                        profiles[8] - reference
                    ).sum()

                    error_32 = np.abs(
                        profiles[32] - reference
                    ).sum()

                    print(
                        f"n={n} | "
                        f"Scale=({sx}, {sy}) | "
                        f"8x8 error={error_8*100:.6f}% | "
                        f"32x32 error={error_32*100:.6f}%"
                    )

                    # Verify convergence of the higher
                    # resolution sampling configuration.
                    self.assertLessEqual(
                        error_32,
                        tolerance,
                        "Sersic sampling has not converged."
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)