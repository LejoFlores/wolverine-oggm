"""SCHNAPS output -> OGGM.

SCHNAPS writes hourly netCDF files per domain (`*_d03_*.nc`, 24 frames each) on a curvilinear
lat/lon grid. Precipitation variables (`precipitation`, `snowfall`, `graupel`) are accumulated
totals; 2 m air temperature is `taix` in the Noah-MP runs, or split into `temperature_2m_veg` and
`temperature_2m_bare` (0 where a tile is absent) in the SNOWPACK runs. `post/extract_phase.py`
condenses a case into one small file (hourly `precip`, `solid`, `rain`, `t2`, `terrain`, `lat`,
`lon`) that is easy to copy off Borah; `load_case` reads either.

Two pathways into OGGM:

1. Episodes (what exists now, e.g. Oct 9-17 2026 at 50 m): SCHNAPS's microphysics decide the phase
   of every hour's precipitation on every glacier cell. `phase_samples` + `phase_curve_from_case`
   turn that into a phase curve (solid fraction vs 2 m temperature) and `band_summary` compares
   it, elevation band by elevation band, with what OGGM's temperature ramp would say for the same
   hours and for the episode-mean temperature (OGGM's monthly view). The fitted curve plugs into
   massbalance.phase_model_class for long OGGM runs.

2. Long runs (months to years): `monthly_climate` aggregates glacier-cell temperature and
   precipitation to monthly values at a reference height, and `write_oggm_climate` splices the
   fully covered months into a copy of the glacier's baseline climate file, which OGGM then reads
   with `climate_input_filesuffix`. By default the splice divides SCHNAPS precipitation by the
   calibrated precipitation factor (and removes the temperature bias), so the MB model uses
   SCHNAPS amounts as they are rather than scaling them twice.

`make_synthetic_case` writes a fake case in SCHNAPS's raw format so the notebooks can be run and
checked before real output is copied over. Its numbers mean nothing.
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from . import config, phase as _phase



def accum_to_hourly(acc):
    """Accumulated totals (time, y, x) to per-interval amounts; the first interval is lost.
    Same as accum_to_hourly in schnaps-borah's post/extract_phase.py."""
    d = np.diff(acc, axis=0, prepend=acc[:1])
    return np.clip(d, 0, None)   # restarts can reset an accumulator; never negative


# --- reading ---------------------------------------------------------------------------------------
def load_case(path, domain="d03"):
    """Hourly glacier-relevant fields from a SCHNAPS case.

    path: an output directory of a case (raw `*_{domain}_*.nc` files) or a file written by
    schnaps-borah's post/extract_phase.py. Returns a Dataset (time, y, x) with precip, solid, rain (mm in the hour
    ending at time), t2 (C), and terrain, lat, lon (y, x)."""
    if os.path.isfile(path):
        ds = xr.open_dataset(path).load()
        out = xr.Dataset({"precip": ds["precip"], "solid": ds["solid"],
                          "t2": ds["t2"] - 273.15 if float(ds["t2"].max()) > 150 else ds["t2"]},
                         coords={"time": ds["time"]})
        out["terrain"] = ds["terrain"]
        out["lat"], out["lon"] = ds["lat"], ds["lon"]
        out.attrs["source"] = ds.attrs.get("source", path)
    else:
        files = sorted(glob.glob(os.path.join(path, f"*_{domain}_*.nc")))
        if not files:
            raise FileNotFoundError(f"No *_{domain}_*.nc in {path}")
        acc = {"precipitation": [], "snowfall": [], "graupel": []}
        t2, times = [], []
        for f in files:
            with xr.open_dataset(f, decode_times=True) as d:
                times.append(d["time"].values)
                for v in acc:
                    acc[v].append(d[v].values.astype(np.float64) if v in d else None)
                if "taix" in d:
                    t2.append(d["taix"].values.astype(np.float32))
                elif "temperature_2m_veg" in d or "temperature_2m_bare" in d:
                    t2.append(_tiles_t2(lambda v, d=d: d[v].values.astype(np.float32) if v in d else None))
                else:
                    raise KeyError(f"{f} has no taix or temperature_2m_veg/bare: add one to OUTPUT_VARS")
                if f == files[0]:
                    lat, lon = d["lat"].values, d["lon"].values
                    terrain = d["terrain"].values if "terrain" in d else None
        if terrain is not None and terrain.ndim == 3:
            terrain = terrain[0]
        pr = accum_to_hourly(np.concatenate(acc["precipitation"]))
        solid = accum_to_hourly(np.concatenate(acc["snowfall"])) if acc["snowfall"][0] is not None else np.zeros_like(pr)
        if acc["graupel"][0] is not None:
            solid = solid + accum_to_hourly(np.concatenate(acc["graupel"]))
        solid = np.minimum(solid, pr)
        dims = ("time", "y", "x")
        t2 = np.concatenate(t2)
        out = xr.Dataset({"precip": (dims, pr.astype(np.float32)), "solid": (dims, solid.astype(np.float32)),
                          "t2": (dims, (t2 - 273.15).astype(np.float32))},
                         coords={"time": np.concatenate(times)})
        out["terrain"] = (("y", "x"), terrain if terrain is not None else np.full(lat.shape, np.nan))
        out["lat"], out["lon"] = (("y", "x"), lat), (("y", "x"), lon)
        out.attrs["source"] = f"{path} ({domain}, {len(files)} files)"
        out = out.isel(time=slice(1, None))         # the first hour's accumulation is unknown
    out["rain"] = out["precip"] - out["solid"]
    for v in ("precip", "solid", "rain"):
        out[v].attrs["units"] = "mm"
    out["t2"].attrs["units"] = "degC"
    return out


def _tiles_t2(get):
    """As schnaps-borah's post/extract_phase.two_m_temperature: mean of the vegetated and bare tiles, 0 = absent."""
    tv, tb = get("temperature_2m_veg"), get("temperature_2m_bare")
    tiles = np.stack([t for t in (tv, tb) if t is not None])
    tiles = np.where(tiles > 150, tiles, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(tiles, axis=0)


# --- where the glacier is ------------------------------------------------------------------------------
def glacier_mask(ds, outlines=None, rgi_id=None, gdir=None):
    """Boolean (y, x): SCHNAPS cells whose centre is inside the glacier outline.

    outlines: a GeoDataFrame or GeoJSON path (default: the repo's RGI 7.0 outlines, selecting
    `rgi_id`, default Wolverine); or pass an OGGM `gdir` to use its outline."""
    import geopandas as gpd
    import shapely
    if gdir is not None:
        geom = gdir.read_shapefile("outlines").to_crs(4326).union_all()
    else:
        g = gpd.read_file(outlines or config.RGI7_OUTLINES) if not isinstance(outlines, gpd.GeoDataFrame) else outlines
        if rgi_id is None and "rgi_id" in g:
            rgi_id = config.RGI7_ID
        if rgi_id is not None:
            g = g[g["rgi_id"] == rgi_id]
        geom = g.to_crs(4326).union_all()
    m = shapely.contains_xy(geom, ds["lon"].values, ds["lat"].values)
    if not m.any():
        raise ValueError("No SCHNAPS cell centre falls inside the outline: is this the right domain?")
    return xr.DataArray(m, dims=("y", "x"))


# --- pathway 1: episodes -------------------------------------------------------------------------
def phase_samples(ds, mask, min_precip=0.1):
    """Every wet glacier cell-hour as a row: time, elevation z, t2, precip, solid, frac."""
    m = np.asarray(mask)
    jj, ii = np.where(m)
    pr = ds["precip"].values[:, jj, ii]
    so = ds["solid"].values[:, jj, ii]
    t2 = ds["t2"].values[:, jj, ii]
    z = np.broadcast_to(ds["terrain"].values[jj, ii], pr.shape)
    tt = np.broadcast_to(ds["time"].values[:, None], pr.shape)
    wet = pr >= min_precip
    df = pd.DataFrame({"time": tt[wet], "z": z[wet], "t2": t2[wet], "precip": pr[wet], "solid": so[wet]})
    df["frac"] = df["solid"] / df["precip"]
    df.attrs["n_cells"] = int(m.sum())
    df.attrs["hours"] = int(ds.sizes["time"])
    return df


def phase_curve_from_case(df, bins=np.arange(-6, 6.5, 0.5)):
    """(Empirical binned curve, fitted Logistic) from phase_samples, weighted by precipitation."""
    c, frac, ptot = _phase.binned_fraction(df["t2"], df["solid"], df["precip"], bins=bins)
    keep = ptot > 0.01 * ptot.sum() / len(ptot)
    emp = _phase.Empirical(c[keep], frac[keep], name="SCHNAPS microphysics (binned)")
    emp.precip_per_bin = ptot
    fit = _phase.fit_logistic(df["t2"], df["frac"], weights=df["precip"])
    return emp, fit


def precip_temperature_offset(ds, mask):
    """Precipitation-weighted mean 2 m temperature minus the plain mean over the same glacier cells
    and hours (K). Positive: it precipitates in warmer-than-average weather (typical of atmospheric
    rivers). Feeds PhaseAwareTIModel.precip_temp_offset."""
    m = np.asarray(mask)
    t = ds["t2"].values[:, m]
    p = ds["precip"].values[:, m]
    return float(np.sum(t * p) / np.sum(p) - np.mean(t))


def band_summary(df, ds=None, mask=None, band=100.0, curve=None):
    """Per elevation band: precipitation, SCHNAPS solid fraction, the solid fraction OGGM's
    ramp gives from the same hourly temperatures, and (with ds and mask) from the band's
    episode-mean temperature, which is what a monthly model sees."""
    curve = curve or _phase.STANDARD_CURVES["oggm"]
    d = df.copy()
    d["band"] = (np.floor(d["z"] / band) * band + band / 2).astype(int)
    d["solid_oggm_hourly"] = d["precip"] * curve(d["t2"])
    g = d.groupby("band")
    out = pd.DataFrame({
        "precip_sum_mm": g["precip"].sum(),
        "solid_frac_schnaps": g["solid"].sum() / g["precip"].sum(),
        "solid_frac_oggm_hourly_T": g["solid_oggm_hourly"].sum() / g["precip"].sum(),
        "t2_precip_weighted": g.apply(lambda x: np.average(x["t2"], weights=x["precip"]), include_groups=False),
    })
    if ds is not None and mask is not None:
        m = np.asarray(mask)
        z = ds["terrain"].values[m]
        tmean = ds["t2"].values[:, m].mean(axis=0)
        bands = (np.floor(z / band) * band + band / 2).astype(int)
        tb = pd.Series(tmean).groupby(bands).mean()
        out["n_cells"] = pd.Series(bands).value_counts().reindex(out.index)
        out["precip_mm_per_cell"] = out["precip_sum_mm"] / out["n_cells"]
        out["t2_episode_mean"] = tb.reindex(out.index)
        out["solid_frac_oggm_episode_mean_T"] = curve(out["t2_episode_mean"].values)
    out.index.name = "band_centre_m"
    return out


# --- pathway 2: long runs --------------------------------------------------------------------------
def monthly_climate(ds, mask, ref_hgt=None, lapse=-0.0065):
    """Monthly glacier-mean climate from SCHNAPS glacier cells.

    temp: mean 2 m temperature brought to ref_hgt (default: the glacier cells' mean elevation)
    with `lapse` (K/m); prcp: glacier-mean precipitation (mm/month); prcp_solid: same for SCHNAPS
    solid precipitation; temp_std: standard deviation of daily means; coverage: fraction of the
    month's hours present."""
    m = np.asarray(mask)
    z = ds["terrain"].values[m]
    ref_hgt = float(np.mean(z)) if ref_hgt is None else float(ref_hgt)
    t = ds["t2"].values[:, m] + lapse * (ref_hgt - z)[None, :]
    h = pd.DataFrame({"temp": t.mean(axis=1), "prcp": ds["precip"].values[:, m].mean(axis=1),
                      "prcp_solid": ds["solid"].values[:, m].mean(axis=1)},
                     index=pd.DatetimeIndex(ds["time"].values))
    daily = h["temp"].resample("D").mean()
    mon = pd.DataFrame({"temp": h["temp"].resample("MS").mean(), "prcp": h["prcp"].resample("MS").sum(),
                        "prcp_solid": h["prcp_solid"].resample("MS").sum(),
                        "temp_std": daily.resample("MS").std(),
                        "hours": h["temp"].resample("MS").count()})
    mon["coverage"] = mon["hours"] / (mon.index.days_in_month * 24)
    mon.attrs["ref_hgt"] = ref_hgt
    mon.attrs["lon"], mon.attrs["lat"] = float(ds["lon"].values[m].mean()), float(ds["lat"].values[m].mean())
    return mon


def write_oggm_climate(gdir, monthly, filesuffix="_schnaps", min_coverage=0.95, as_is=True, lapse=-0.0065):
    """Splice SCHNAPS months into a copy of gdir's climate_historical and write it as
    climate_historical{filesuffix}. Months with coverage >= min_coverage are replaced; the rest
    keep the baseline values. SCHNAPS temperature is moved to the baseline's reference height
    with `lapse`.

    as_is=True divides SCHNAPS precipitation by the calibrated prcp_fac and subtracts the
    calibrated temp_bias, so that once the MB model applies them the SCHNAPS values are used
    unchanged. as_is=False treats SCHNAPS like the baseline data (the factors apply on top).
    Returns a DataFrame of the replaced months with baseline and SCHNAPS values side by side."""
    base_path = gdir.get_filepath("climate_historical")
    with xr.open_dataset(base_path) as b:
        b = b.load()
    ref_hgt = float(b.attrs["ref_hgt"])
    calib = gdir.read_json("mb_calib")
    use = monthly[monthly["coverage"] >= min_coverage]
    times = pd.DatetimeIndex(b["time"].values)
    use = use[use.index.isin(times)]
    if use.empty:
        raise ValueError(f"No fully covered SCHNAPS month falls in the baseline period {times[0]:%Y-%m} to "
                         f"{times[-1]:%Y-%m}. For recent periods use a baseline that reaches them "
                         "(e.g. cfg.PARAMS['baseline_climate'] = 'ERA5').")
    temp = use["temp"] + lapse * (ref_hgt - monthly.attrs["ref_hgt"])
    prcp = use["prcp"].copy()
    if as_is:
        temp = temp - calib["temp_bias"]
        prcp = prcp / calib["prcp_fac"]
    idx = times.get_indexer(use.index)
    comp = pd.DataFrame({"temp_baseline": b["temp"].values[idx], "temp_schnaps": temp.values,
                         "prcp_baseline": b["prcp"].values[idx], "prcp_schnaps": prcp.values}, index=use.index)
    tnew, pnew = b["temp"].values.copy(), b["prcp"].values.copy()
    tnew[idx], pnew[idx] = temp.values, prcp.values
    kw = {}
    if "temp_std" in b:
        sd = b["temp_std"].values.copy()
        ok = use["temp_std"].notna().values
        sd[idx[ok]] = use["temp_std"].values[ok]
        kw["temp_std"] = sd
    gdir.write_monthly_climate_file(times, pnew, tnew, ref_hgt, b.attrs["ref_pix_lon"], b.attrs["ref_pix_lat"],
                                    filesuffix=filesuffix,
                                    source=f"{b.attrs.get('climate_source', 'baseline')} with {len(use)} months "
                                           f"from SCHNAPS ({'as is' if as_is else 'scaled like baseline'})",
                                    **kw)
    return comp


# --- synthetic case --------------------------------------------------------------------------------
def make_synthetic_case(outdir, gdir, start="2026-10-09", hours=192, dx_deg=(0.0009, 0.0018),
                        t50=0.8, width=0.6, seed=0, domain="d03"):
    """Write a fake SCHNAPS case (raw output format, one file per day) over the glacier in `gdir`,
    with terrain from the glacier's DEM, a storm-like temperature and precipitation history, and
    a known logistic phase curve (t50, width) plus noise, so the notebook's fit can be checked.
    Not real data."""
    import rasterio
    import salem
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    olines = gdir.read_shapefile("outlines").to_crs(4326)
    w, s, e, n = olines.total_bounds
    lats = np.arange(s - 0.01, n + 0.01, dx_deg[0])
    lons = np.arange(w - 0.02, e + 0.02, dx_deg[1])
    lon, lat = np.meshgrid(lons, lats)
    with rasterio.open(gdir.get_filepath("dem")) as r:
        dem = r.read(1).astype(float)
    xx, yy = gdir.grid.transform(lon, lat, crs=salem.wgs84, nearest=True)
    ok = (xx >= 0) & (xx < dem.shape[1]) & (yy >= 0) & (yy < dem.shape[0])
    terrain = np.where(ok, dem[np.clip(yy, 0, dem.shape[0] - 1), np.clip(xx, 0, dem.shape[1] - 1)], np.nan)
    terrain = np.where(np.isfinite(terrain), terrain, np.nanmin(terrain))
    t = pd.date_range(start, periods=hours, freq="h")
    hr = np.arange(hours)
    zref = np.median(terrain)
    t_ref = 1.0 + 3.0 * np.sin(2 * np.pi * hr / 24) * 0.3 + 2.5 * np.sin(2 * np.pi * hr / hours)   # warm mid-storm
    storm = np.exp(-0.5 * ((hr - 0.65 * hours) / (0.12 * hours)) ** 2) + 0.4 * np.exp(-0.5 * ((hr - 0.2 * hours) / 10) ** 2)
    t2 = t_ref[:, None, None] - 0.0065 * (terrain - zref)[None] + rng.normal(0, 0.4, (hours,) + terrain.shape)
    oro = 1 + 0.6 * (terrain - terrain.min()) / np.ptp(terrain)
    pr = np.clip(2.5 * storm[:, None, None] * oro[None] * rng.lognormal(0, 0.4, (hours,) + terrain.shape) - 0.05, 0, None)
    frac = 1 / (1 + np.exp((t2 - t50) / width))
    frac = np.clip(frac + rng.normal(0, 0.05, frac.shape), 0, 1)
    snow = pr * frac * 0.85
    grau = pr * frac * 0.15
    acc = lambda a: np.cumsum(a, axis=0)   # noqa: E731
    P, S, G = acc(pr), acc(snow), acc(grau)
    for d0 in range(0, hours, 24):
        sl = slice(d0, min(d0 + 24, hours))
        ds = xr.Dataset({"precipitation": (("time", "y", "x"), P[sl].astype(np.float32)),
                         "snowfall": (("time", "y", "x"), S[sl].astype(np.float32)),
                         "graupel": (("time", "y", "x"), G[sl].astype(np.float32)),
                         "taix": (("time", "y", "x"), (t2[sl] + 273.15).astype(np.float32)),
                         "terrain": (("y", "x"), terrain.astype(np.float32)),
                         "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
                        coords={"time": t[sl]})
        ds.attrs["comment"] = f"SYNTHETIC test case from wolverine_oggm.schnaps.make_synthetic_case (t50={t50}, width={width}); not model output"
        ds.to_netcdf(os.path.join(outdir, f"synthetic_{domain}_{t[d0]:%Y-%m-%d_%H-%M-%S}.nc"))
    return outdir
