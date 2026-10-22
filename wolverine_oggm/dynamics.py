"""Ice velocity in OGGM, and a seasonal-sliding extension for testing a hydrology hypothesis.

OGGM's flowline models use the shallow-ice approximation. The depth-averaged velocity at each
grid point is

    u = fd * H^(n+1) * |rho g dS/dx|^n  +  fs * H^(n-1) * |rho g dS/dx|^n      (n = 3)
        deformation (fd = 2A/(n+2))        sliding (Budd-type, fs in Pa^-3 m^2 s^-1)

with A (glen_a) calibrated so the inverted volume matches the consensus estimate and fs = 0 by
default. So in standard OGGM, precipitation phase changes velocity only through geometry: phase
changes the mass balance, the mass balance changes thickness H and slope dS/dx, and those change
u. There is no water at the bed.

`HydroSliding` is an evolution model that lets fs vary month to month with the liquid water
reaching the glacier surface (rain + melt from the mass-balance model):

    fs(t) = fs0 * max(0, 1 + k * (L(t) / L_ref - 1))

where L(t) is the glacier-wide liquid input of the current month, L_ref its mean over a reference
period, and k the coupling strength (k = 0 is standard OGGM with constant fs0). It is a
conceptual sensitivity test, not a subglacial hydrology model: real sliding responds to the rate
of change of water input and to drainage-system efficiency, which this ignores. Use it to ask
"if sliding scaled with liquid water input, how much would rain-vs-snow partitioning matter for
the velocity and the glacier's evolution?"
"""
import numpy as np
from oggm import cfg
from oggm.core.flowline import SemiImplicitModel, FluxBasedModel

SEC_IN_MONTH = cfg.SEC_IN_YEAR / 12
FS_OERLEMANS = 5.7e-20      # Oerlemans (1997) sliding parameter, the classic OGGM non-zero fs


def _monthly_climate(mb, heights, yr):
    """(temp, tempformelt, prcp, prcpsol) from a flowline MB model, unwrapping Random/Constant MB
    models to the underlying monthly model and year."""
    if hasattr(mb, "get_state_yr"):                       # RandomMassBalance
        y0 = int(np.floor(yr))
        yr = mb.get_state_yr(y0) + (yr - y0)
        mb = mb.mbmod
    if hasattr(mb, "mbmod") and not hasattr(mb, "get_monthly_climate"):
        mb = mb.mbmod
    return mb, mb.get_monthly_climate(heights, year=yr)


def liquid_input(model_or_fls, mb_model, yr):
    """Glacier-wide mean liquid water input (rain + potential melt, kg m-2 month-1) in the month
    containing floating year `yr`, on the current geometry."""
    fls = getattr(model_or_fls, "fls", model_or_fls)
    num = den = 0.0
    for fl_id, fl in enumerate(fls):
        ice = fl.thick > 0
        if not ice.any():
            continue
        mb = mb_model.flowline_mb_models[fl_id] if hasattr(mb_model, "flowline_mb_models") else mb_model
        base, (t, tm, p, ps) = _monthly_climate(mb, fl.surface_h[ice], yr)
        liq = (p - ps) + base.monthly_melt_f * tm
        a = fl.widths_m[ice] * fl.dx_meter
        num += np.sum(liq * a)
        den += np.sum(a)
    return num / den if den > 0 else 0.0


def liquid_climatology(fls, mb_model, years):
    """Mean monthly liquid input over `years` on a fixed geometry (for L_ref)."""
    vals = [liquid_input(fls, mb_model, y + (m + 0.5) / 12) for y in years for m in range(12)]
    return float(np.mean(vals))


class _HydroSlidingMixin:
    def __init__(self, *args, sliding_coupling=0.0, liquid_ref=None, liquid_ref_years=None,
                 max_dt_days=15.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs0 = self.fs
        self.coupling = float(sliding_coupling)
        self._max_dt = max_dt_days * cfg.SEC_IN_DAY
        if self.coupling != 0 and self.fs0 <= 0:
            raise ValueError("HydroSliding needs fs > 0: invert with a non-zero fs first "
                             "(see notebook 04) or pass fs=...")
        if liquid_ref is None and self.coupling != 0:
            if liquid_ref_years is None:
                raise ValueError("Give liquid_ref (kg m-2 month-1) or liquid_ref_years")
            liquid_ref = liquid_climatology(self.fls, self.mb_model, liquid_ref_years)
        self.liquid_ref = liquid_ref
        self._month = None
        self.fs_history = []          # (floating year, liquid input, fs) once per month

    def _update_fs(self):
        if self.coupling == 0:
            return
        month = int(np.floor(self.yr * 12 + 1e-9))
        if month == self._month:
            return
        self._month = month
        L = liquid_input(self, self.mb_model, (month + 0.5) / 12)
        self.fs = self.fs0 * max(0.0, 1.0 + self.coupling * (L / self.liquid_ref - 1.0))
        self.fs_history.append((month / 12, L, self.fs))

    def step(self, dt):
        if self.coupling != 0:
            # never step across a month boundary, so fs follows the seasonal cycle
            to_boundary = ((np.floor(self.yr * 12 + 1e-9) + 1) / 12 - self.yr) * cfg.SEC_IN_YEAR
            dt = min(dt, self._max_dt, max(to_boundary, cfg.SEC_IN_DAY))
            self._update_fs()
        return super().step(dt)


class HydroSliding(_HydroSlidingMixin, SemiImplicitModel):
    """SemiImplicitModel (OGGM 1.6's default, single flowline) with liquid-water-dependent fs."""


class HydroSlidingFluxBased(_HydroSlidingMixin, FluxBasedModel):
    """FluxBasedModel (works with several flowlines) with liquid-water-dependent fs."""


def _invert_once(gdir, fs, vol0):
    from oggm.workflow import calibrate_inversion_from_consensus
    import logging
    lg = logging.getLogger("oggm.workflow")
    lvl = lg.level
    lg.setLevel(logging.ERROR)
    try:
        calibrate_inversion_from_consensus([gdir], fs=fs, volume_m3_reference=vol0,
                                           error_on_mismatch=False, add_to_log_file=False)
    finally:
        lg.setLevel(lvl)
    inv = gdir.read_pickle("inversion_output")
    vol = float(sum(np.sum(c["volume"]) for c in inv))
    diag = gdir.get_diagnostics()
    return diag["inversion_glen_a"], vol


def sliding_fraction(gdir, glen_a, fs):
    """Share of the depth-averaged speed due to sliding, area-weighted over the inverted glacier."""
    num = den = 0.0
    for c in gdir.read_pickle("inversion_output"):
        H, w = c["thick"], c["width"]
        slope = np.clip(c["slope_angle"], np.deg2rad(cfg.PARAMS["min_slope"]), np.pi / 2)   # as in the inversion
        tau = cfg.PARAMS["ice_density"] * cfg.G * slope
        n = cfg.PARAMS["glen_n"]
        ud = 2.0 / (n + 2) * glen_a * H ** (n + 1) * tau ** n
        us = fs * H ** (n - 1) * tau ** n
        ok = H > 0
        num += np.sum((us * w)[ok])
        den += np.sum(((ud + us) * w)[ok])
    return num / den if den > 0 else np.nan


def invert_with_sliding(gdir, sliding_frac=None, fs=None, filesuffix="_fs", tol=0.02):
    """Re-invert the thickness with sliding while keeping the glacier volume of the existing
    (no-sliding) inversion: glen_a is recalibrated for each fs so the volume is unchanged.

    Give either `fs` directly or a target `sliding_frac` (0-1, the area-weighted share of the
    depth-averaged speed due to sliding), and fs is found by bisection. Writes
    model_flowlines{filesuffix} and leaves the original inversion files as they were, so runs
    that do not ask for `filesuffix` are unaffected. Returns a dict with glen_a, fs, the
    resulting sliding fraction and the volume (km3) before and after."""
    import shutil
    from oggm import tasks
    inv = gdir.read_pickle("inversion_output")
    vol0 = float(sum(np.sum(c["volume"]) for c in inv))
    diag0 = gdir.get_diagnostics()
    backup = {}
    for name in ("inversion_output", "model_flowlines"):
        p = gdir.get_filepath(name)
        backup[name] = p + ".bak"
        shutil.copy(p, backup[name])
    try:
        if fs is None:
            if sliding_frac is None or not 0 < sliding_frac < 1:
                raise ValueError("Give fs, or sliding_frac between 0 and 1")
            lo, hi = -24.0, -18.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                glen_a, vol = _invert_once(gdir, 10 ** mid, vol0)
                frac = sliding_fraction(gdir, glen_a, 10 ** mid)
                if abs(frac - sliding_frac) < tol and abs(vol / vol0 - 1) < 0.01:
                    break
                if frac < sliding_frac:
                    lo = mid
                else:
                    hi = mid
            fs = 10 ** mid
        glen_a, vol = _invert_once(gdir, fs, vol0)
        frac = sliding_fraction(gdir, glen_a, fs)
        tasks.init_present_time_glacier(gdir, filesuffix=filesuffix)
    finally:
        for name, b in backup.items():
            shutil.move(b, gdir.get_filepath(name))
        gdir.add_to_diagnostics("inversion_glen_a", diag0.get("inversion_glen_a", cfg.PARAMS["glen_a"]))
        gdir.add_to_diagnostics("inversion_fs", diag0.get("inversion_fs", 0.0))
    if abs(vol / vol0 - 1) > 0.01:
        print(f"WARNING: with fs={fs:.2e} the volume is {vol / 1e9:.3f} km3 instead of {vol0 / 1e9:.3f} km3: "
              "glen_a hit its bounds. Use a smaller fs or sliding fraction.")
    out = {"glen_a": glen_a, "fs": fs, "sliding_fraction": frac, "volume_km3": vol / 1e9,
           "volume_km3_no_sliding": vol0 / 1e9, "filesuffix": filesuffix}
    for k in ("glen_a", "fs", "sliding_fraction"):
        gdir.add_to_diagnostics(f"inversion{filesuffix}_{k}", out[k])
    return out


def velocity_partition(fl, glen_a, fs, n=3):
    """Deformation and sliding parts of the depth-averaged velocity (m/yr) on a flowline's
    staggered grid, from its current thickness and surface slope."""
    rho, g = cfg.PARAMS["ice_density"], cfg.G
    H = fl.thick
    S = fl.surface_h
    dx = fl.dx_meter
    Hs = 0.5 * (H[1:] + H[:-1])
    slope = -np.diff(S) / dx
    tau = rho * g * np.abs(slope)
    fd = 2.0 / (n + 2) * glen_a
    u_def = fd * Hs ** (n + 1) * tau ** n * cfg.SEC_IN_YEAR
    u_sl = fs * Hs ** (n - 1) * tau ** n * cfg.SEC_IN_YEAR
    x = (np.arange(len(Hs)) + 0.5) * dx
    return x, u_def, u_sl
