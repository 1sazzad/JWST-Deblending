"""PSF-aware two-source flexible Gaussian baseline (Phase 8.3).

Fit a shared elliptical Gaussian morphology per source across all bands.
Profile out nonnegative fluxes and free constant backgrounds with exact
(two-source) active-set least squares. Optimize only six shape parameters.
No simulation truth is used by this fitter.
"""

import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.optimize import minimize, nnls

MIN_SIGMA = 0.6
MAX_SIGMA = 8.0
MAX_RHO = 0.9
MAX_ITERATIONS = 150
# Symmetric deterministic starts; no source brightness or shape truth used.
START_WIDTHS = ((2.5, 2.5), (1.7, 3.5), (3.5, 1.7))


def gaussian_with_derivatives(dx, dy, theta):
    """Unit-sum Gaussian and derivatives w.r.t. log widths and correlation.

    Covariance = [[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]].
    Marginal widths and correlation describe arbitrary elliptical orientations.
    """
    sx, sy = np.exp(theta[:2])
    rho = theta[2]
    u, v = dx / sx, dy / sy
    q = 1.0 - rho**2
    a = u*u - 2*rho*u*v + v*v
    p = np.exp(-0.5*a/q)
    p /= p.sum()
    score = np.stack(((u*u-rho*u*v)/q, (v*v-rho*u*v)/q,
                      u*v/q - rho*a/q**2))
    deriv = p[None] * (score - (score*p).sum(axis=(-2,-1))[:,None,None])
    return np.concatenate((p[None], deriv), axis=0)


def profile_linear(templates, observed):
    """Solve two nonnegative fluxes and a free constant per band.

    Eliminate the background by centering the data and each template.
    Enumerate active sets in the 2-D NNLS problem. Fall back to SciPy NNLS
    if the two columns are nearly linearly dependent.
    """
    a = templates.reshape(2, -1)
    y = observed.ravel()
    means = a.mean(axis=1)
    ym = y.mean()
    z = a - means[:, None]
    t = y - ym
    g00, g11 = np.sum(z*z, axis=1)
    g01 = np.sum(z[0]*z[1])
    h0, h1 = np.sum(z*t, axis=1)
    det = g00*g11-g01*g01
    if min(g00, g11) <= 0 or det <= 1e-10*g00*g11:
        flux = nnls(z.T, t)[0]
    else:
        candidates = [np.zeros(2), np.array([max(0.,h0/g00),0.]),
                      np.array([0.,max(0.,h1/g11)])]
        both = np.array([(g11*h0-g01*h1)/det,(g00*h1-g01*h0)/det])
        if np.all(both >= 0):
            candidates.append(both)
        # Difference from zero-flux squared error, omitting constant t.T@t.
        def score(f):
            return g00*f[0]**2+2*g01*f[0]*f[1]+g11*f[1]**2-2*(h0*f[0]+h1*f[1])
        flux = min(candidates, key=score)
    background = float(ym - np.sum(flux*means))
    model = np.sum(flux[:,None,None]*templates, axis=0)+background
    return flux, background, model


class GaussianObjective:
    """Cache PSF FFTs and evaluate the profiled loss and analytic gradient."""

    def __init__(self, observed, psfs, positions, noise_sigma):
        self.observed = np.asarray(observed, dtype=np.float64)
        psfs = np.asarray(psfs, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        self.noise = np.asarray(noise_sigma, dtype=np.float64)
        if self.observed.ndim != 3 or positions.shape != (2,2):
            raise ValueError('Expected a band cube and exactly two (x,y) positions')
        b,h,w = self.observed.shape
        if psfs.ndim != 3 or psfs.shape[0] != b or self.noise.shape != (b,):
            raise ValueError('Incompatible PSF or noise dimensions')
        if not all(np.isfinite(a).all() for a in (self.observed, psfs, positions, self.noise)) or np.any(self.noise <= 0):
            raise ValueError('Invalid inputs')
        if np.any(psfs < 0) or not np.allclose(psfs.sum(axis=(-2,-1)),1,atol=1e-8,rtol=1e-8):
            raise ValueError('Expected existing nonnegative unit-sum PSFs')
        # Cutout-normalized Gaussian equals the padded-canvas Gaussian to
        # negligible tail error for these central sources. Do not use on edges.
        margin = np.minimum(positions, np.array([w-1,h-1])-positions)
        if np.any(margin < 6*MAX_SIGMA):
            raise ValueError('Source too close to edge for this central-cutout baseline')
        kh,kw = psfs.shape[-2:]
        self.fft_shape = (next_fast_len(h+kh-1),next_fast_len(w+kw-1))
        self.psf_fft = rfftn(psfs,s=self.fft_shape,axes=(-2,-1))
        sy,sx = (kh-1)//2,(kw-1)//2
        self.crop = (slice(sy,sy+h),slice(sx,sx+w))
        y,x = np.indices((h,w),dtype=np.float64)
        self.offsets = [(x-px,y-py) for px,py in positions]

    def convolve(self, profiles):
        """Linear same-mode convolution, matching the fixed Gaussian baseline."""
        pf = rfftn(profiles,s=self.fft_shape,axes=(-2,-1))
        out = irfftn(pf[:,None]*self.psf_fft[None],s=self.fft_shape,axes=(-2,-1))
        return out[...,self.crop[0],self.crop[1]]

    def evaluate(self, theta, return_details=False):
        # Array order: source, profile/derivative, band, y, x.
        blurred = np.stack([self.convolve(gaussian_with_derivatives(dx,dy,theta[3*s:3*s+3]))
                            for s,(dx,dy) in enumerate(self.offsets)])
        fluxes = np.empty((2,len(self.noise)))
        backgrounds = np.empty(len(self.noise))
        model = np.empty_like(self.observed)
        for b in range(len(self.noise)):
            fluxes[:,b],backgrounds[b],model[b] = profile_linear(blurred[:,0,b],self.observed[b])
        residual = model-self.observed
        weighted = residual/self.noise[:,None,None]**2
        loss = 0.5*np.sum(residual*weighted)
        # Envelope theorem: at the profiled linear optimum, derivatives of
        # fitted amplitudes/backgrounds do not enter the loss gradient.
        grad = np.array([np.sum(blurred[s,1:]*weighted[None]*fluxes[s][None,:,None,None],axis=(1,2,3))
                         for s in range(2)]).ravel()
        if return_details:
            return fluxes, model, backgrounds, float(loss)
        return float(loss), grad


def fit_flexible_gaussian(observed, psfs, positions, noise_sigma, *, max_iterations=MAX_ITERATIONS):
    """Fit both sources jointly across bands; choose lowest objective start.

    Return flux, rendered model and optimization diagnostics. Convergence and
    numerical validity are separate; finite iteration-limited fits are retained.
    """
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    objective = GaussianObjective(observed,psfs,positions,noise_sigma)
    bounds = [(np.log(MIN_SIGMA),np.log(MAX_SIGMA)),(np.log(MIN_SIGMA),np.log(MAX_SIGMA)),(-MAX_RHO,MAX_RHO)]*2
    results = []
    for a,b in START_WIDTHS:
        x0 = np.array([np.log(a),np.log(a),0.,np.log(b),np.log(b),0.])
        result = minimize(objective.evaluate,x0,jac=True,method='L-BFGS-B',bounds=bounds,
                          options={'maxiter':max_iterations,'ftol':1e-10,'gtol':1e-5,'maxls':30})
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            results.append(result)
    if not results:
        raise RuntimeError('All optimization starts returned nonfinite outputs')
    best = min(results,key=lambda r:r.fun)
    flux,model,bg,loss = objective.evaluate(best.x,return_details=True)
    if not np.isfinite(flux).all() or not np.isfinite(model).all() or np.any(flux<0):
        raise RuntimeError('Invalid profiled solution')
    shapes = []
    for logx,logy,rho in best.x.reshape(2,3):
        sx,sy = np.exp([logx,logy])
        covariance = np.array([[sx*sx,rho*sx*sy],[rho*sx*sy,sy*sy]])
        vals,vecs = np.linalg.eigh(covariance)
        major = vecs[:,1]
        angle = (np.degrees(np.arctan2(major[1],major[0]))+90)%180-90
        shapes.append(dict(sigma_x=float(sx),sigma_y=float(sy),rho=float(rho),
                           sigma_major=float(np.sqrt(vals[1])),sigma_minor=float(np.sqrt(vals[0])),
                           angle_deg=float(angle)))
    diagnostics = dict(shapes=shapes,background=bg,loss=loss,
                       chi2=2*loss/np.size(observed),optimizer_success=bool(best.success),
                       optimizer_message=str(best.message),iterations=int(best.nit),
                       function_evaluations=int(best.nfev),valid_starts=len(results),
                       converged_starts=sum(bool(r.success) for r in results),
                       start_losses=[float(r.fun) for r in results])
    return flux,model,diagnostics
