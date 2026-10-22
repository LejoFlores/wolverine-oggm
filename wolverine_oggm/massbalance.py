"""A phase-aware version of OGGM's monthly temperature-index mass-balance model.

PhaseAwareTIModel is MonthlyTIModel with one change: the solid share of each month's precipitation
comes from a phase curve (wolverine_oggm.phase) averaged over the sub-monthly temperature spread,
instead of OGGM's 0-2 C ramp on the monthly mean. Melt, the temperature lapse rate, the
precipitation factor and the temperature bias are untouched, so differences between runs are the
phase partitioning alone (plus whatever recalibration you choose to do).

OGGM builds mass-balance models itself inside tasks (calibration, runs), passing only its own
arguments, so the phase settings live on the class. `phase_model_class` makes a configured class:

    from wolverine_oggm import massbalance, phase
    Cls = massbalance.phase_model_class(phase.Logistic(1.0, 0.5), temp_std="file")
    mbmod = Cls(gdir)                                     # uses the calibrated melt_f etc.
    Cls = massbalance.recalibrate(gdir, Cls, "_logistic")   # match the geodetic MB again
    tasks.run_from_climate_data(gdir, mb_model_class=Cls, ...)
"""
import numpy as np
import xarray as xr
from oggm import cfg
from oggm.core.massbalance import MonthlyTIModel, mb_calibration_from_scalar_mb
from oggm.utils import floatyear_to_date

from . import phase as _phase


class PhaseAwareTIModel(MonthlyTIModel):
    """MonthlyTIModel with a configurable precipitation-phase scheme.

    Class attributes (set them with `phase_model_class`):
      phase_curve : PhaseCurve or None (None = OGGM's ramp from cfg.PARAMS)
      temp_std    : None or 0 (no spread), a float (K), 12 monthly values, or "file" to use the
                    daily temperature standard deviation in the climate file (W5E5 has it)
      precip_temp_offset : K added to the temperature when computing phase (precipitation in
                    warmer weather than the month as a whole: > 0)
    """
    phase_curve = None
    temp_std = None
    precip_temp_offset = 0.0
    calib_suffix = ""        # which mb_calib{suffix}.json to read by default (see `recalibrate`)
    label = "OGGM ramp"

    def __init__(self, gdir, **kwargs):
        if self.calib_suffix:
            kwargs.setdefault("mb_params_filesuffix", self.calib_suffix)
        super().__init__(gdir, **kwargs)
        if self.phase_curve is None:
            self.phase_curve = _phase.Ramp(self.t_solid, self.t_liq, name="OGGM default (ramp)")
        self.sigma = self._read_sigma(gdir, kwargs.get("filename", "climate_historical"),
                                      kwargs.get("input_filesuffix", ""), kwargs.get("ys"), kwargs.get("ye"))

    def _read_sigma(self, gdir, filename, filesuffix, ys, ye):
        n = len(self.temp)
        ts = self.temp_std
        if ts is None:
            return np.zeros(n)
        if isinstance(ts, str):
            if ts != "file":
                raise ValueError("temp_std must be None, a number, 12 numbers or 'file'")
            with xr.open_dataset(gdir.get_filepath(filename, filesuffix=filesuffix)) as ds:
                if "temp_std" not in ds:
                    raise ValueError(f"{filename} has no temp_std; give temp_std as numbers instead")
                yrs = ds["time.year"].values
                ok = np.ones(len(yrs), bool)
                if ys is not None:
                    ok &= yrs >= ys
                if ye is not None:
                    ok &= yrs <= ye
                return ds["temp_std"].values[ok].astype(float)
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        if ts.size == 1:
            return np.full(n, ts[0])
        if ts.size == 12:
            return ts[self.months - 1]
        raise ValueError("temp_std must have 1 or 12 values")

    def _solid_fraction(self, temp, sigma):
        return _phase.expected_solid_fraction(self.phase_curve, temp, sigma, self.precip_temp_offset)

    def _index(self, year):
        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        return np.where((self.years == y) & (self.months == m))[0][0]

    def get_monthly_climate(self, heights, year=None):
        temp, tempformelt, prcp, _ = super().get_monthly_climate(heights, year=year)
        sig = self.sigma[self._index(year)]
        return temp, tempformelt, prcp, prcp * self._solid_fraction(temp, sig)

    def _get_2d_annual_climate(self, heights, year):
        temp2d, tmelt2d, prcp2d, _ = super()._get_2d_annual_climate(heights, year)
        y = np.floor(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        sig = self.sigma[np.where(self.years == y)[0]]            # (12,)
        frac = self._solid_fraction(temp2d, np.broadcast_to(sig, temp2d.shape))
        return temp2d, tmelt2d, prcp2d, prcp2d * frac


def phase_model_class(phase_curve=None, temp_std=None, precip_temp_offset=0.0, name=None, calib_suffix=""):
    """A PhaseAwareTIModel subclass with these phase settings, to hand to OGGM tasks
    (mb_model_class=...). calib_suffix selects the calibration file it reads by default."""
    label = name or (phase_curve.name if phase_curve is not None else "OGGM ramp")
    attrs = dict(phase_curve=phase_curve, temp_std=temp_std, precip_temp_offset=precip_temp_offset,
                 calib_suffix=calib_suffix, label=label)
    return type(f"PhaseAwareTIModel_{abs(hash((label, calib_suffix))) % 10**6}", (PhaseAwareTIModel,), attrs)


def with_settings(cls, **changes):
    """Copy of a phase model class with some attributes changed, e.g.
    with_settings(Cls, precip_temp_offset=-1.0, label="phase frozen")."""
    attrs = {k: getattr(cls, k) for k in ("phase_curve", "temp_std", "precip_temp_offset", "calib_suffix", "label")}
    attrs.update(changes)
    return type(f"{PhaseAwareTIModel.__name__}_{abs(hash(tuple(sorted(map(str, attrs.items()))))) % 10**6}",
                (PhaseAwareTIModel,), attrs)


def recalibrate(gdir, mb_model_class, filesuffix, param="melt_f", ref_mb=None, ref_period=None):
    """Recalibrate one parameter (melt_f by default) so the model with this phase scheme matches
    the same reference MB as the original calibration, keeping the other parameters. Writes
    mb_calib{filesuffix}.json and returns a copy of the class that reads it by default, so it can
    go straight into OGGM tasks. The calibration itself is in `returned_class.calib`."""
    calib = gdir.read_json("mb_calib")
    ref_mb = calib["reference_mb"] if ref_mb is None else ref_mb
    ref_period = calib["reference_period"] if ref_period is None else ref_period
    kw = {k: calib[k] for k in ("melt_f", "prcp_fac", "temp_bias") if k != param}
    mb_calibration_from_scalar_mb(gdir, ref_mb=ref_mb, ref_period=ref_period, calibrate_param1=param,
                                  mb_model_class=mb_model_class, filesuffix=filesuffix,
                                  write_to_gdir=True, overwrite_gdir=True, **kw)
    out = with_settings(mb_model_class, calib_suffix=filesuffix)
    out.calib = gdir.read_json("mb_calib", filesuffix=filesuffix)
    return out


def accumulation_and_melt(mbmod, heights, years):
    """Monthly climate at the given heights for each year: a Dataset with temp (C), prcp,
    prcp_solid, prcp_liquid, melt (kg m-2 month-1, melt is the potential melt melt_f x positive
    degree-months), dims (time, height)."""
    out = {k: [] for k in ("temp", "prcp", "prcp_solid", "melt")}
    times = []
    for y in years:
        for m in range(1, 13):
            fy = y + (m - 1) / 12
            t, tm, p, ps = mbmod.get_monthly_climate(heights, year=fy)
            out["temp"].append(t)
            out["prcp"].append(p)
            out["prcp_solid"].append(ps)
            out["melt"].append(mbmod.monthly_melt_f * tm)
            times.append(np.datetime64(f"{y:04d}-{m:02d}-01"))
    ds = xr.Dataset({k: (("time", "height"), np.array(v)) for k, v in out.items()},
                    coords={"time": times, "height": np.asarray(heights)})
    ds["prcp_liquid"] = ds["prcp"] - ds["prcp_solid"]
    for v in ("prcp", "prcp_solid", "prcp_liquid", "melt"):
        ds[v].attrs["units"] = "kg m-2 month-1"
    ds["temp"].attrs["units"] = "degC"
    return ds


def flowline_heights_and_weights(gdir):
    """Surface heights and area weights of the present-day model flowlines (all flowlines)."""
    fls = gdir.read_pickle("model_flowlines")
    h = np.concatenate([fl.surface_h for fl in fls])
    w = np.concatenate([fl.widths_m * fl.dx_meter for fl in fls])
    ice = np.concatenate([fl.thick > 0 for fl in fls])
    return h[ice], w[ice]


def specific_mb_by_scheme(gdir, classes, years, **kwargs):
    """Glacier-wide specific MB (kg m-2 yr-1) per year on the fixed present geometry, one column per
    class in `classes` ({label: mb_model_class})."""
    import pandas as pd
    h, w = flowline_heights_and_weights(gdir)
    out = {}
    for label, cls in classes.items():
        mb = cls(gdir, **kwargs)
        out[label] = [np.average(mb.get_annual_mb(h, year=y) * cfg.SEC_IN_YEAR * mb.rho, weights=w) for y in years]
    return pd.DataFrame(out, index=pd.Index(list(years), name="year"))


def seasonal_balances(mbmod, gdir, years, winter_months=(10, 11, 12, 1, 2, 3, 4)):
    """Glacier-wide winter, summer and annual balance (kg m-2) per hydrological year (Oct-Sep,
    labelled by the year it ends), on the fixed present geometry, to compare with WGMS/USGS
    WINTER_BALANCE, SUMMER_BALANCE and ANNUAL_BALANCE. Winter is October to April by default,
    roughly the USGS season at Wolverine. Also returns winter solid and liquid precipitation."""
    import pandas as pd
    h, w = flowline_heights_and_weights(gdir)
    rows = []
    for y in years:
        win = summ = sol = liq = 0.0
        for m in list(range(10, 13)) + list(range(1, 10)):
            yr = (y - 1) if m >= 10 else y
            fy = yr + (m - 1) / 12
            smb = mbmod.get_monthly_mb(h, year=fy) * cfg.SEC_IN_MONTH * mbmod.rho
            val = np.average(smb, weights=w)
            if m in winter_months:
                win += val
                _, _, p, ps = mbmod.get_monthly_climate(h, year=fy)
                sol += np.average(ps, weights=w)
                liq += np.average(p - ps, weights=w)
            else:
                summ += val
        rows.append((y, win, summ, win + summ, sol, liq))
    return pd.DataFrame(rows, columns=["year", "winter", "summer", "annual", "winter_solid_prcp",
                                       "winter_liquid_prcp"]).set_index("year")


__all__ = ["PhaseAwareTIModel", "phase_model_class", "with_settings", "recalibrate", "accumulation_and_melt",
           "flowline_heights_and_weights", "specific_mb_by_scheme", "seasonal_balances"]
