
"""Phase 8.4: PSF-aware flexible Sersic baseline.

Independent continuous Sersic indices, elliptical shapes and shared morphology
across bands. Profile out flux and background using the Phase 8.3 solver.
The finite 256-pixel canvas and crop match the development simulator; no
truth is supplied to the fitter. Numerical quadrature is fixed by spatial
radius (not by the true morphology or scene label).
"""

import numpy as np

from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.optimize import minimize
from scipy.special import gammaincinv

from src.baselines.baseline_flexible_gaussian import (
    profile_linear,
    MIN_SIGMA,
    MAX_SIGMA,
    MAX_RHO,
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------

MIN_N, MAX_N = 0.5, 5.0

MAX_ITERATIONS = 150
PADDING = 64

RADIUS_SCALE = np.sqrt(2 * np.log(2.0))

# Source A width, source B width,
# starting Sersic index A, starting Sersic index B.
STARTS = (
    (2.5, 2.5, 1.0, 1.0),
    (1.7, 3.5, 2.0, 2.0),
    (3.5, 1.7, 4.0, 4.0),
)


# ---------------------------------------------------------
# Pixel-integrated Sersic morphology
# ---------------------------------------------------------

class PixelQuadrature:
    """Precompute midpoint offsets; integrate central pixels more finely.

    Fixed 32x32 sampling within 4 pixels of the supplied center, 8x8
    within 16 pixels and 2x2 elsewhere. Regions are disjoint and independent
    of fitted shape. Chunked processing limits CPU memory consumption.
    """

    def __init__(self, shape, position):

        y, x = np.indices(
            shape,
            dtype=np.float64
        )

        dx = (x - position[0]).ravel()
        dy = (y - position[1]).ravel()

        radius = np.maximum(
            np.abs(dx),
            np.abs(dy)
        )

        self.shape = shape
        self.groups = []

        for mask, samples in (
            (radius <= 4, 32),
            ((radius > 4) & (radius <= 16), 8),
            (radius > 16, 2),
        ):

            ids = np.flatnonzero(mask)

            offsets = (
                (np.arange(samples) + 0.5)
                / samples
                - 0.5
            )

            oy, ox = np.meshgrid(
                offsets,
                offsets,
                indexing="ij"
            )

            # Store only coordinate arrays,
            # not shape-dependent values.
            self.groups.append(
                (
                    ids,
                    dx[ids],
                    dy[ids],
                    ox.ravel(),
                    oy.ravel(),
                )
            )

    def evaluate(self, theta):
        """Return normalized profile and analytical derivatives.

        Parameters:
            theta:
                log(sigma_x), log(sigma_y),
                rho and log(Sersic index).
        """

        sx, sy = np.exp(theta[:2])

        rho = theta[2]
        n = np.exp(theta[3])

        q = 1 - rho * rho

        bn = gammaincinv(2 * n, 0.5)

        # Only the scalar b_n derivative
        # uses a central finite difference.
        eps = 1e-4

        db_dlogn = (
            gammaincinv(
                2 * n * np.exp(eps),
                0.5
            )
            - gammaincinv(
                2 * n * np.exp(-eps),
                0.5
            )
        ) / (2 * eps)

        values = np.zeros(
            (5, np.prod(self.shape)),
            dtype=np.float64
        )

        for ids, dx, dy, ox, oy in self.groups:

            # Keep temporary arrays below
            # roughly 131k sample points.
            chunk = max(
                1,
                131072 // len(ox)
            )

            for start in range(
                0,
                len(ids),
                chunk
            ):

                sl = slice(
                    start,
                    start + chunk
                )

                u = (
                    dx[sl][None, :]
                    + ox[:, None]
                ) / sx

                v = (
                    dy[sl][None, :]
                    + oy[:, None]
                ) / sy

                uv = u * v

                a = np.maximum(
                    u * u
                    - 2 * rho * uv
                    + v * v,
                    1e-300
                )

                logr = (
                    0.5 * np.log(a / q)
                    - np.log(RADIUS_SCALE)
                )

                power = np.exp(logr / n)

                k = bn * power

                p = np.exp(-k)

                common = k / n

                # Apply the chain rule for the
                # elliptical radius and Sersic index.
                dest = ids[sl]

                values[0, dest] = p.mean(
                    axis=0
                )

                values[1, dest] = (
                    p
                    * common
                    * (u * u - rho * uv)
                    / a
                ).mean(axis=0)

                values[2, dest] = (
                    p
                    * common
                    * (v * v - rho * uv)
                    / a
                ).mean(axis=0)

                values[3, dest] = (
                    p
                    * common
                    * (uv / a - rho / q)
                ).mean(axis=0)

                values[4, dest] = (
                    p
                    * (
                        k * logr / n
                        - db_dlogn * power
                    )
                ).mean(axis=0)

        # Normalize the morphology on the
        # finite padded simulation canvas.
        total = values[0].sum()

        if (
            not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError(
                "Invalid Sersic normalization"
            )

        p = values[0] / total

        # Apply the normalization derivative.
        values[1:] = (
            values[1:]
            - p[None]
            * values[1:].sum(axis=1)[:, None]
        ) / total

        values[0] = p

        return values.reshape(
            (5, *self.shape)
        )


# ---------------------------------------------------------
# PSF-aware variable projection objective
# ---------------------------------------------------------

class SersicObjective:
    """Evaluate profiled weighted loss and its gradient."""

    def __init__(
        self,
        observed,
        psfs,
        positions,
        noise_sigma,
    ):

        self.observed = np.asarray(
            observed,
            dtype=np.float64
        )

        psfs = np.asarray(
            psfs,
            dtype=np.float64
        )

        positions = np.asarray(
            positions,
            dtype=np.float64
        )

        self.noise = np.asarray(
            noise_sigma,
            dtype=np.float64
        )

        if (
            self.observed.ndim != 3
            or positions.shape != (2, 2)
        ):
            raise ValueError(
                "Expected a band cube and "
                "two (x,y) positions"
            )

        b, h, w = self.observed.shape

        if (
            psfs.ndim != 3
            or psfs.shape[0] != b
            or self.noise.shape != (b,)
        ):
            raise ValueError(
                "Incompatible dimensions"
            )

        if (
            not all(
                np.isfinite(x).all()
                for x in (
                    self.observed,
                    psfs,
                    positions,
                    self.noise,
                )
            )
            or np.any(self.noise <= 0)
        ):
            raise ValueError("Invalid input")

        if (
            np.any(psfs < 0)
            or not np.allclose(
                psfs.sum(axis=(-2, -1)),
                1,
                rtol=1e-8,
                atol=1e-8,
            )
        ):
            raise ValueError(
                "Expected nonnegative "
                "unit-sum PSFs"
            )

        if (
            np.any(positions < 0)
            or np.any(
                positions
                > np.array([w - 1, h - 1])
            )
        ):
            raise ValueError(
                "Position outside observation"
            )

        # Match the development simulator's
        # finite padded canvas.
        ph = h + 2 * PADDING
        pw = w + 2 * PADDING

        kh, kw = psfs.shape[-2:]

        self.fft_shape = (
            next_fast_len(ph + kh - 1),
            next_fast_len(pw + kw - 1),
        )

        # Precompute the PSF Fourier transforms.
        self.psf_fft = rfftn(
            psfs,
            s=self.fft_shape,
            axes=(-2, -1)
        )

        sy = (kh - 1) // 2 + PADDING
        sx = (kw - 1) // 2 + PADDING

        self.crop = (
            slice(sy, sy + h),
            slice(sx, sx + w),
        )

        self.quad = [
            PixelQuadrature(
                (ph, pw),
                p + PADDING
            )
            for p in positions
        ]

    def convolve(self, profiles):
        """Convolve profiles and derivatives with all PSFs."""

        pf = rfftn(
            profiles,
            s=self.fft_shape,
            axes=(-2, -1)
        )

        out = irfftn(
            pf[:, None]
            * self.psf_fft[None],
            s=self.fft_shape,
            axes=(-2, -1)
        )

        return out[
            ...,
            self.crop[0],
            self.crop[1]
        ].copy()

    def evaluate(
        self,
        theta,
        return_details=False
    ):
        """Evaluate the profiled loss and its gradient."""

        blurred = np.stack([
            self.convolve(
                q.evaluate(
                    theta[4 * s:4 * s + 4]
                )
            )
            for s, q in enumerate(self.quad)
        ])

        flux = np.empty(
            (2, len(self.noise))
        )

        bg = np.empty(
            len(self.noise)
        )

        model = np.empty_like(
            self.observed
        )

        # Reuse the Phase 8.3 nonnegative
        # linear flux and background solver.
        for b in range(len(self.noise)):

            flux[:, b], bg[b], model[b] = (
                profile_linear(
                    blurred[:, 0, b],
                    self.observed[b]
                )
            )

        residual = model - self.observed

        weighted = (
            residual
            / self.noise[:, None, None] ** 2
        )

        loss = 0.5 * np.sum(
            residual * weighted
        )

        if return_details:
            return (
                flux,
                model,
                bg,
                float(loss)
            )

        # Envelope theorem for the profiled
        # nonnegative linear solution.
        grad = np.array([
            np.sum(
                blurred[s, 1:]
                * weighted[None]
                * flux[s][None, :, None, None],
                axis=(1, 2, 3)
            )
            for s in range(2)
        ]).ravel()

        return float(loss), grad


# ---------------------------------------------------------
# Main fitting interface
# ---------------------------------------------------------

def fit_flexible_sersic(
    observed,
    psfs,
    positions,
    noise_sigma,
    *,
    max_iterations=MAX_ITERATIONS,
):
    """Fit eight nonlinear shape parameters.

    Select the final solution using observation loss only.
    No true flux or morphology information is used.
    """

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be positive"
        )

    objective = SersicObjective(
        observed,
        psfs,
        positions,
        noise_sigma,
    )

    bounds = [
        (
            np.log(MIN_SIGMA),
            np.log(MAX_SIGMA)
        ),
        (
            np.log(MIN_SIGMA),
            np.log(MAX_SIGMA)
        ),
        (-MAX_RHO, MAX_RHO),
        (
            np.log(MIN_N),
            np.log(MAX_N)
        ),
    ] * 2

    results = []

    # Multiple deterministic starts reduce
    # sensitivity to initialization.
    for a, b, na, nb in STARTS:

        x0 = np.array([
            np.log(a),
            np.log(a),
            0.0,
            np.log(na),

            np.log(b),
            np.log(b),
            0.0,
            np.log(nb),
        ])

        result = minimize(
            objective.evaluate,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "ftol": 1e-10,
                "gtol": 1e-5,
                "maxls": 30,
            },
        )

        if (
            np.isfinite(result.fun)
            and np.isfinite(result.x).all()
        ):
            results.append(result)

    if not results:
        raise RuntimeError(
            "All starts invalid"
        )

    # Never use ground truth to select a fit.
    best = min(
        results,
        key=lambda r: r.fun
    )

    flux, model, bg, loss = (
        objective.evaluate(
            best.x,
            return_details=True
        )
    )

    if (
        not np.isfinite(flux).all()
        or not np.isfinite(model).all()
        or np.any(flux < 0)
    ):
        raise RuntimeError(
            "Invalid fitted solution"
        )

    shapes = []

    # Convert the fitted covariance
    # parameterization into interpretable
    # galaxy morphology parameters.
    for lx, ly, rho, ln in best.x.reshape(2, 4):

        sx, sy = np.exp([lx, ly])

        n = np.exp(ln)

        covariance = np.array([
            [
                sx * sx,
                rho * sx * sy
            ],
            [
                rho * sx * sy,
                sy * sy
            ],
        ])

        vals, vecs = np.linalg.eigh(
            covariance
        )

        major = vecs[:, 1]

        angle = (
            np.degrees(
                np.arctan2(
                    major[1],
                    major[0]
                )
            )
            + 90
        ) % 180 - 90

        shapes.append(
            dict(
                sigma_x=float(sx),
                sigma_y=float(sy),
                rho=float(rho),
                sigma_major=float(
                    np.sqrt(vals[1])
                ),
                sigma_minor=float(
                    np.sqrt(vals[0])
                ),
                angle_deg=float(angle),
                fitted_sersic_n=float(n),
            )
        )

    diagnostics = dict(
        shapes=shapes,
        background=bg,
        loss=loss,
        chi2=2 * loss / np.size(observed),
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
        iterations=int(best.nit),
        function_evaluations=int(best.nfev),
        valid_starts=len(results),
        converged_starts=sum(
            bool(r.success)
            for r in results
        ),
        start_losses=[
            float(r.fun)
            for r in results
        ],
    )

    return flux, model, diagnostics
