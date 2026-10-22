"""Precipitation phase: how much of a month's precipitation falls as snow.

OGGM's monthly temperature-index model (MonthlyTIModel) splits each month's precipitation with a
linear ramp on the *monthly mean* temperature at each elevation: all solid at or below
temp_all_solid (0 C), all liquid at or above temp_all_liq (2 C). At Wolverine, where monthly means
sit near 0 C for much of the accumulation season and storms are warm atmospheric rivers, that
choice controls how much of the precipitation becomes accumulation.

A phase curve here is any function f(T) -> solid fraction in [0, 1], where T is the air
temperature (C) while it precipitates. `expected_solid_fraction` turns a curve defined on
instantaneous temperatures into a monthly fraction by averaging it over a normal distribution of
temperatures around the monthly mean (sigma = sub-monthly temperature spread, plus an optional
offset for precipitation falling in warmer or colder weather than the monthly mean). With
sigma = 0 and the OGGM ramp it reproduces OGGM exactly.
"""
import numpy as np

# Gauss-Hermite nodes for E[f(T + sigma Z)], Z ~ N(0, 1)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(24)
_GH_X = _GH_X * np.sqrt(2.0)
_GH_W = _GH_W / np.sqrt(np.pi)


class PhaseCurve:
    """Callable solid fraction f(T), T in C. Subclasses implement `__call__`."""
    name = "curve"

    def __call__(self, t):
        raise NotImplementedError

    def t50(self):
        """Temperature of 50% snow, found numerically."""
        t = np.linspace(-10, 10, 4001)
        f = self(t)
        return float(t[np.argmin(np.abs(f - 0.5))])

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}, 50% snow at {self.t50():.2f} C>"


class Ramp(PhaseCurve):
    """OGGM's default: linear between t_solid (all snow) and t_liq (all rain)."""

    def __init__(self, t_solid=0.0, t_liq=2.0, name=None):
        if t_liq <= t_solid:
            raise ValueError("t_liq must be above t_solid")
        self.t_solid, self.t_liq = float(t_solid), float(t_liq)
        self.name = name or f"ramp {t_solid:g} to {t_liq:g} C"

    def __call__(self, t):
        return np.clip(1 - (np.asarray(t, dtype=float) - self.t_solid) / (self.t_liq - self.t_solid), 0, 1)


class Threshold(PhaseCurve):
    """Single rain/snow threshold: all snow below t0, all rain above."""

    def __init__(self, t0=1.0, name=None):
        self.t0 = float(t0)
        self.name = name or f"threshold {t0:g} C"

    def __call__(self, t):
        return (np.asarray(t, dtype=float) < self.t0).astype(float)


class Logistic(PhaseCurve):
    """Smooth S-curve f = 1 / (1 + exp((T - t50) / width)). `width` (K) sets how gradual the
    transition is: 10-90% snow spans about 4.4 x width."""

    def __init__(self, t50=1.0, width=0.5, name=None):
        self.t_half, self.width = float(t50), float(width)
        self.name = name or f"logistic, 50% at {t50:.2f} C, width {width:.2f} K"

    def __call__(self, t):
        return 1.0 / (1.0 + np.exp((np.asarray(t, dtype=float) - self.t_half) / self.width))

    def t50(self):
        return self.t_half


class Jennings2018(PhaseCurve):
    """Bivariate (temperature and relative humidity) logistic regression of Jennings et al. (2018,
    Nature Communications 9:1148), fit to ~17.8 million Northern Hemisphere observations:
    p(snow) = 1 / (1 + exp(a + b T + c RH)), T in C, RH in %. With a fixed RH this is a
    Logistic; the monthly OGGM model has no humidity, so RH is a parameter. Check the coefficients
    against the paper's table before citing numbers from it."""

    a, b, c = -10.04, 1.41, 0.09

    def __init__(self, rh=90.0, name=None):
        self.rh = float(rh)
        self.name = name or f"Jennings et al. 2018 at RH {rh:g}%"

    def __call__(self, t, rh=None):
        rh = self.rh if rh is None else rh
        return 1.0 / (1.0 + np.exp(self.a + self.b * np.asarray(t, dtype=float) + self.c * np.asarray(rh, dtype=float)))


class Empirical(PhaseCurve):
    """A tabulated curve, e.g. binned from SCHNAPS microphysics (see schnaps.phase_curve_from_case).
    Linear interpolation between bin centres, constant beyond the ends."""

    def __init__(self, t, frac, name="empirical"):
        t, frac = np.asarray(t, dtype=float), np.asarray(frac, dtype=float)
        ok = np.isfinite(t) & np.isfinite(frac)
        order = np.argsort(t[ok])
        self.t, self.frac = t[ok][order], np.clip(frac[ok][order], 0, 1)
        self.name = name

    def __call__(self, t):
        return np.interp(np.asarray(t, dtype=float), self.t, self.frac)


def expected_solid_fraction(curve, t_mean, sigma=0.0, offset=0.0):
    """Mean solid fraction when the precipitation-time temperature is normal around
    t_mean + offset with standard deviation sigma (all in C or K). Broadcasts over arrays."""
    t_mean = np.asarray(t_mean, dtype=float) + offset
    sigma = np.asarray(sigma, dtype=float)
    if np.all(sigma == 0):
        return curve(t_mean)
    sigma = np.broadcast_to(sigma, t_mean.shape)
    t = t_mean[..., None] + sigma[..., None] * _GH_X
    return np.sum(curve(t) * _GH_W, axis=-1)


def fit_logistic(t, solid_fraction, weights=None):
    """Weighted least-squares fit of a Logistic curve to samples of temperature (C) and solid
    fraction (0-1), e.g. hourly SCHNAPS cells weighted by their precipitation. Returns a Logistic
    with the fitted t50 and width, and their standard errors in `.stderr`."""
    from scipy.optimize import least_squares
    t, y = np.asarray(t, dtype=float).ravel(), np.asarray(solid_fraction, dtype=float).ravel()
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    t, y, w = t[ok], y[ok], np.sqrt(w[ok])
    if len(t) < 10:
        raise ValueError("Need at least 10 samples to fit a phase curve")

    def resid(p):
        return w * (1.0 / (1.0 + np.exp((t - p[0]) / np.exp(p[1]))) - y)

    sol = least_squares(resid, x0=[1.0, np.log(0.7)])
    t50, width = sol.x[0], float(np.exp(sol.x[1]))
    curve = Logistic(t50, width, name=f"fit: 50% at {t50:.2f} C, width {width:.2f} K")
    try:
        dof = max(len(t) - 2, 1)
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (2 * sol.cost / dof)
        se = np.sqrt(np.diag(cov))
        curve.stderr = {"t50": float(se[0]), "width": float(width * se[1])}
    except np.linalg.LinAlgError:
        curve.stderr = {"t50": np.nan, "width": np.nan}
    return curve


def binned_fraction(t, solid, precip, bins=np.arange(-8, 8.5, 0.5)):
    """Precipitation-weighted solid fraction in temperature bins: returns (centres, fraction,
    precipitation total per bin)."""
    t, solid, precip = (np.asarray(a, dtype=float).ravel() for a in (t, solid, precip))
    k = np.digitize(t, bins) - 1
    n = len(bins) - 1
    ok = (k >= 0) & (k < n) & np.isfinite(t) & (precip > 0)
    ptot = np.bincount(k[ok], weights=precip[ok], minlength=n)
    stot = np.bincount(k[ok], weights=solid[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(ptot > 0, stot / ptot, np.nan)
    return 0.5 * (bins[1:] + bins[:-1]), frac, ptot


STANDARD_CURVES = {
    "oggm": Ramp(0.0, 2.0, name="OGGM default (ramp 0-2 C)"),
    "threshold0": Threshold(0.0, name="threshold 0 C"),
    "threshold1": Threshold(1.0, name="threshold 1 C"),
    "threshold2": Threshold(2.0, name="threshold 2 C"),
    "jennings90": Jennings2018(90.0, name="Jennings 2018, RH 90%"),
}
