"""Running and summarising experiment ensembles.

Every experiment in notebooks 03 and 04 is one OGGM run with an output suffix. `run_stationary`
runs one under a stationary climate (years drawn at random from the calibration period, with
an optional temperature change), and `summary` / `velocity_profile` read its output back as
tidy tables.
"""
import numpy as np
import pandas as pd
import xarray as xr
from oggm import tasks

from . import config


def run_stationary(gdir, suffix, mb_model_class, nyears, temperature_bias=0.0, seed=1, **kwargs):
    """tasks.run_random_climate from the present glacier, with the same random year sequence for
    every call that uses the same seed. kwargs go to the flowline model (e.g. evolution_model,
    glen_a, fs, init_model_fls)."""
    y0, halfsize = config.random_climate_window()
    return tasks.run_random_climate(gdir, nyears=nyears, y0=y0, halfsize=halfsize, seed=seed,
                                    mb_model_class=mb_model_class, temperature_bias=temperature_bias,
                                    output_filesuffix=suffix, unique_samples=True, **kwargs)


def _fl_group(gdir):
    return f"fl_{len(gdir.read_pickle('model_flowlines')) - 1}"


def fl_diagnostics(gdir, suffix):
    with xr.open_dataset(gdir.get_filepath("fl_diagnostics", filesuffix=suffix), group=_fl_group(gdir)) as ds:
        return ds.load()


def summary(gdir, suffix):
    """Per output time: volume (km3), area (km2), length (km), area-weighted mean and max surface
    velocity (m/yr) on the main flowline."""
    with xr.open_dataset(gdir.get_filepath("model_diagnostics", filesuffix=suffix)) as d:
        diag = d.load()
    fl = fl_diagnostics(gdir, suffix)
    ice = fl["thickness_m"] > 1
    v = fl["ice_velocity_myr"].where(ice)
    area = fl["area_m2"].where(ice)
    vmean = (v * area).sum("dis_along_flowline") / area.sum("dis_along_flowline")
    out = pd.DataFrame({"volume_km3": diag.volume_m3.values / 1e9, "area_km2": diag.area_m2.values / 1e6,
                        "length_km": diag.length_m.values / 1e3}, index=diag.time.values)
    out["velocity_mean_myr"] = pd.Series(vmean.values, index=fl.time.values).reindex(out.index)
    out["velocity_max_myr"] = pd.Series(v.max("dis_along_flowline").values, index=fl.time.values).reindex(out.index)
    out.index.name = "year"
    return out


def velocity_profile(gdir, suffix, years=None):
    """Surface velocity along the flowline (m/yr), mean over the given output years (default:
    the last 10)."""
    fl = fl_diagnostics(gdir, suffix)
    sel = fl.isel(time=slice(-10, None)) if years is None else fl.sel(time=slice(*years))
    v = sel["ice_velocity_myr"].where(sel["thickness_m"] > 1).mean("time")
    return pd.Series(v.values, index=fl.dis_along_flowline.values / 1000, name=suffix)


def compare(gdir, suffixes, labels=None, tail=10):
    """One row per run: change in volume, length and mean velocity over the run, and the means of
    the last `tail` years."""
    rows = []
    for k, s in enumerate(suffixes):
        d = summary(gdir, s)
        last = d.iloc[-tail:].mean()
        rows.append({"run": (labels or suffixes)[k],
                     "volume_end_km3": last.volume_km3,
                     "volume_change_pct": 100 * (last.volume_km3 / d.volume_km3.iloc[0] - 1),
                     "length_end_km": last.length_km,
                     "velocity_mean_end_myr": last.velocity_mean_myr,
                     "velocity_change_pct": 100 * (last.velocity_mean_myr / d.velocity_mean_myr.iloc[1:4].mean() - 1)})
    return pd.DataFrame(rows).set_index("run")


def ensemble(gdir, run_fn, seeds):
    """Run `run_fn(seed) -> suffix` for several seeds and stack the summaries (columns get a
    'seed' level), for spread across random climate sequences."""
    frames = {s: summary(gdir, run_fn(s)) for s in seeds}
    return pd.concat(frames, names=["seed"]).reset_index()


__all__ = ["run_stationary", "fl_diagnostics", "summary", "velocity_profile", "compare", "ensemble", "np"]
