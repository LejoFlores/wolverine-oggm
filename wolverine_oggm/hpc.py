"""Batch experiments for clusters (Borah): glacier directories at a chosen resolution, ensembles of
runs as Slurm array tasks, and one merged output file per experiment.

An experiment is a YAML file (examples in experiments/). It names a *setup* (which glaciers, map
resolution, DEM, climate, spinup) and a set of *factors* (phase scheme, warming, random-climate
seed or GCM/SSP, sliding). Every combination of factors is one *member*: one OGGM run per glacier.

Three steps, each a sub-command of `python -m wolverine_oggm.hpc`:

    prepare  CONFIG --stage fetch    needs internet (borah/prepare.slurm, short queue): outlines,
                                     DEM, flowlines, climate, MB calibration, inversion, GCM files
    prepare  CONFIG --stage spinup   offline: historical run and dynamic spinup
    list     CONFIG                  writes members.csv and prints the member count
    run      CONFIG --members 0-39   offline (Slurm array task): runs those members, `--procs` at a time
    collect  CONFIG                  merges every member into ensemble.nc and summary.csv

Paths: everything goes under $OGGM_ROOT (default /bsuscratch/$USER/oggm, or ~/OGGM_hpc when
/bsuscratch does not exist):

    $OGGM_ROOT/download_cache/        OGGM's download cache (set OGGM_DOWNLOAD_CACHE to move it)
    $OGGM_ROOT/gdirs/<setup name>/    prepared glacier directories, shared by experiments
    $OGGM_ROOT/results/<experiment>/  members.csv, members/*.nc, ensemble.nc, summary.csv

Each member runs on a private copy of the glacier directories in node-local scratch, so members
never write to the same files, and only its outputs are copied back.
"""
import argparse
import copy
import glob
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

# --- paths ---------------------------------------------------------------------------------------


def oggm_root():
    if os.environ.get("OGGM_ROOT"):
        return os.environ["OGGM_ROOT"]
    user = os.environ.get("USER", "user")
    if os.path.isdir("/bsuscratch"):
        return f"/bsuscratch/{user}/oggm"
    return os.path.join(os.path.expanduser("~"), "OGGM_hpc")


def _set_download_cache():
    """Point OGGM's download cache at $OGGM_ROOT unless OGGM_DOWNLOAD_CACHE is set. Must run
    before oggm.cfg.initialize (OGGM reads the variable there)."""
    os.environ.setdefault("OGGM_DOWNLOAD_CACHE", os.path.join(oggm_root(), "download_cache"))
    os.makedirs(os.environ["OGGM_DOWNLOAD_CACHE"], exist_ok=True)


def setup_dir(conf):
    return os.path.join(oggm_root(), "gdirs", conf["setup"]["name"])


def results_dir(conf):
    return os.path.join(oggm_root(), "results", conf["name"])


# --- configuration -------------------------------------------------------------------------------

SETUP_DEFAULTS = dict(glaciers=["RGI60-01.09162"], dx=None, dem=None, flowline_dx=2, border=80,
                      climate="GSWP3_W5E5", spinup=True, shop=[], gcms=[], ssps=[],
                      params={})
SCENARIO_DEFAULTS = dict(type="random", nyears=200, seeds=[1], ys=None, ye=None, gcms=None, ssps=None)
FACTOR_DEFAULTS = dict(phase=["oggm"], temp_std=None, recalibrate=True, temperature_bias=[0.0],
                       precip_temp_offset=[0.0], sliding=None)
OUTPUT_DEFAULTS = dict(monthly=False, fl_diagnostics=True)

ISIMIP3B_GCMS = ["gfdl-esm4_r1i1p1f1", "ipsl-cm6a-lr_r1i1p1f1", "mpi-esm1-2-hr_r1i1p1f1",
                 "mri-esm2-0_r1i1p1f1", "ukesm1-0-ll_r1i1p1f2"]


def _listify(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def load_config(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    conf = {"name": raw["name"], "config_file": os.path.abspath(path)}
    conf["setup"] = {**SETUP_DEFAULTS, **raw.get("setup", {})}
    conf["setup"].setdefault("name", conf["name"])
    conf["scenario"] = {**SCENARIO_DEFAULTS, **raw.get("scenario", {})}
    conf["factors"] = {**FACTOR_DEFAULTS, **raw.get("factors", {})}
    conf["output"] = {**OUTPUT_DEFAULTS, **raw.get("output", {})}
    sc = conf["scenario"]
    if sc["type"] not in ("random", "historical", "gcm"):
        raise ValueError("scenario.type must be random, historical or gcm")
    if sc["type"] == "gcm":
        sc["gcms"] = _listify(sc["gcms"] or ISIMIP3B_GCMS)
        sc["ssps"] = _listify(sc["ssps"] or ["ssp126", "ssp370", "ssp585"])
        sc["ye"] = sc["ye"] or 2100
        if conf["factors"]["sliding"]:
            raise ValueError("sliding runs are for random and historical scenarios only")
        # the setup has to process these GCM files in its fetch stage
        conf["setup"]["gcms"] = sorted(set(_listify(conf["setup"]["gcms"])) | set(sc["gcms"]))
        conf["setup"]["ssps"] = sorted(set(_listify(conf["setup"]["ssps"])) | set(sc["ssps"]))
    dem = conf["setup"]["dem"]
    if dem and _is_path(dem):    # relative DEM paths are relative to the config file
        dem = os.path.expandvars(os.path.expanduser(str(dem)))
        conf["setup"]["dem"] = os.path.normpath(os.path.join(os.path.dirname(conf["config_file"]), dem))
    if isinstance(conf["setup"]["glaciers"], str):
        conf["setup"]["glaciers"] = [conf["setup"]["glaciers"]]
    return conf


def _is_path(dem):
    return os.path.sep in str(dem) or str(dem).lower().endswith((".tif", ".tiff", ".vrt"))


def _setup_key(setup):
    """Hash of the setup settings that change the glacier directories (not the GCM list, which
    only adds files)."""
    keep = {k: setup[k] for k in ("glaciers", "dx", "dem", "flowline_dx", "border", "climate", "shop", "params")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def members(conf):
    """One row per member: every combination of the factors (and of the seeds or GCM/SSP pairs)."""
    sc, f = conf["scenario"], conf["factors"]
    axes = {}
    if sc["type"] == "random":
        axes["seed"] = _listify(sc["seeds"])
    elif sc["type"] == "gcm":
        axes["gcm"] = sc["gcms"]
        axes["ssp"] = sc["ssps"]
    axes["phase"] = _listify(f["phase"])
    axes["temperature_bias"] = [float(x) for x in _listify(f["temperature_bias"])]
    axes["precip_temp_offset"] = [float(x) for x in _listify(f["precip_temp_offset"])]
    if f["sliding"]:
        axes["sliding_coupling"] = [float(x) for x in _listify(f["sliding"].get("coupling", [0.0]))]
    rows = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
    df = pd.DataFrame(rows)
    df.index.name = "member"
    return df


def parse_range(s, n):
    """'0-9,12,20-' -> sorted member indices (open end = up to n-1)."""
    if s in (None, "", "all"):
        return list(range(n))
    out = set()
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a or 0), (int(b) if b else n - 1) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


# --- OGGM initialisation --------------------------------------------------------------------------

def init_oggm(working_dir, setup, offline, mp_processes=1, logging_level="WORKFLOW"):
    _set_download_cache()
    from oggm import cfg, utils
    cfg.initialize(logging_level=logging_level)
    cfg.PATHS["working_dir"] = utils.mkdir(working_dir)
    p = cfg.PARAMS
    p["border"] = int(setup["border"])
    p["flowline_dx"] = setup["flowline_dx"]
    p["baseline_climate"] = setup["climate"]
    p["use_intersects"] = False          # elevation-band flowlines do not need them
    p["continue_on_error"] = False
    p["store_model_geometry"] = True
    p["store_fl_diagnostics"] = True
    p["use_multiprocessing"] = mp_processes > 1
    p["mp_processes"] = mp_processes
    if setup["dx"]:
        p["grid_dx_method"] = "fixed"
        p["fixed_dx"] = float(setup["dx"])
    for k, v in (setup.get("params") or {}).items():
        p[k] = v
    if offline:
        p["has_internet"] = False         # a missing file fails at once instead of hanging
    return cfg


def _ncpus():
    for var in ("SLURM_CPUS_PER_TASK", "OGGM_PROCS"):
        if os.environ.get(var):
            return int(os.environ[var])
    return 1


# --- prepare --------------------------------------------------------------------------------------

def _rgi_entities(setup):
    """RGI 6.2 outlines (lon/lat) of the selected glaciers. A list of ids comes from OGGM's level-1
    preprocessed directories (small downloads); a bbox reads the RGI region file (420 MB, once)."""
    import geopandas as gpd
    from oggm import utils, workflow, cfg
    g = setup["glaciers"]
    if isinstance(g, dict) and "bbox" in g:
        x0, y0, x1, y1 = g["bbox"]
        region = g.get("region", "01")
        df = gpd.read_file(utils.get_rgi_region_file(region, version="62"))
        df = df[(df.CenLon >= x0) & (df.CenLon <= x1) & (df.CenLat >= y0) & (df.CenLat <= y1)]
        if g.get("min_area_km2"):
            df = df[pd.to_numeric(df.Area) >= float(g["min_area_km2"])]
        if df.empty:
            raise ValueError(f"no RGI 6.2 glaciers have their centre in {g['bbox']}")
        return df.reset_index(drop=True)
    ids = _listify(g)
    tmp = os.path.join(cfg.PATHS["working_dir"], "_l1_outlines")
    wd = cfg.PATHS["working_dir"]
    cfg.PATHS["working_dir"] = utils.mkdir(tmp, reset=True)
    try:
        gds = workflow.init_glacier_directories(
            ids, from_prepro_level=1, prepro_border=80,
            prepro_base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/")
        out = pd.concat([gd.read_shapefile("outlines").to_crs("EPSG:4326") for gd in gds], ignore_index=True)
    finally:
        cfg.PATHS["working_dir"] = wd
        shutil.rmtree(tmp, ignore_errors=True)
    return gpd.GeoDataFrame(out, crs="EPSG:4326")


def _dem_source(setup):
    """(source, user file) for define_glacier_region."""
    dem = setup["dem"]
    if dem is None and not setup["dx"]:
        return "OGGM_DEFAULT", None
    if dem and _is_path(dem):
        return "USER", dem
    if dem:
        return str(dem).upper(), None
    # OGGM's own rule for its RGI-TOPO choice: COPDEM30 on grids finer than 60 m
    return ("COPDEM30" if setup["dx"] and float(setup["dx"]) < 60 else "COPDEM90"), None


CHOSEN_DEM_URL = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/rgitopo/2025.4/"
                  "chosen_dem_RGI62_20251029.csv")

SHOP = {
    "consensus_thickness": ("oggm.shop.bedtopo", "add_consensus_thickness", "consensus_ice_thickness"),
    "itslive_velocity": ("oggm.shop.its_live", "itslive_velocity_to_gdir", "itslive_v"),
    "millan_thickness": ("oggm.shop.millan22", "millan_thickness_to_gdir", "millan_ice_thickness"),
    "millan_velocity": ("oggm.shop.millan22", "millan_velocity_to_gdir", "millan_v"),
    "hugonnet_dhdt": ("oggm.shop.hugonnet_maps", "hugonnet_to_gdir", "hugonnet_dhdt"),
}


def _marker(sdir, stage):
    return os.path.join(sdir, f".done_{stage}.json")


def _write_marker(sdir, stage, setup, **info):
    with open(_marker(sdir, stage), "w") as f:
        json.dump({"setup_key": _setup_key(setup), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "host": os.uname().nodename, **info}, f, indent=1)


def _stage_done(sdir, stage, setup):
    try:
        with open(_marker(sdir, stage)) as f:
            return json.load(f)["setup_key"] == _setup_key(setup)
    except (OSError, ValueError, KeyError):
        return False


def _glacier_ids(sdir):
    with open(os.path.join(sdir, "glaciers.txt")) as f:
        return [s.strip() for s in f if s.strip()]


def prepare_fetch(conf, force=False):
    """Everything that downloads: outlines, DEM on the chosen grid, elevation-band flowlines,
    optional observations (velocity, thickness, dh/dt), baseline climate, MB calibration to the
    Hugonnet et al. (2021) geodetic MB, inversion calibrated to the consensus volume, and GCM
    files. Light on CPU, but needs internet."""
    from oggm import cfg, tasks, workflow, utils
    setup = conf["setup"]
    sdir = setup_dir(conf)
    if _stage_done(sdir, "fetch", setup) and not force:
        print(f"{sdir}: fetch already done for this setup; processing GCM files only")
        init_oggm(sdir, setup, offline=False, mp_processes=_ncpus())
        gdirs = workflow.init_glacier_directories(_glacier_ids(sdir))
    else:
        if os.path.exists(_marker(sdir, "fetch")) and not force:
            raise RuntimeError(f"{sdir} was prepared with different setup settings. Give the setup a new "
                               "name, or rerun with --force to rebuild it (this deletes its runs).")
        init_oggm(sdir, setup, offline=False, mp_processes=_ncpus())
        for m in glob.glob(os.path.join(sdir, ".done_*")):
            os.remove(m)
        rgidf = _rgi_entities(setup)
        print(f"{len(rgidf)} glacier(s), {pd.to_numeric(rgidf.Area).sum():.1f} km2: {', '.join(rgidf.RGIId[:10])}"
              + (" ..." if len(rgidf) > 10 else ""))
        source, user_dem = _dem_source(setup)
        if source == "OGGM_DEFAULT":
            # the DEM OGGM chose for each glacier in its 2025.6 preprocessed directories
            fs = utils.file_downloader(CHOSEN_DEM_URL)
            rgidf["dem_source"] = pd.read_csv(fs, index_col=0).loc[rgidf["RGIId"], "dem_source"].values
            source = None
        if user_dem:
            cfg.PATHS["dem_file"] = user_dem
        gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)
        workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source=source)
        workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs)
        bins = []
        for name in setup["shop"]:
            mod, fn, var = SHOP[name]
            workflow.execute_entity_task(getattr(__import__(mod, fromlist=[fn]), fn), gdirs)
            bins.append(var)
        workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs, bin_variables=bins)
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs, bin_variables=bins)
        workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
        workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
        workflow.execute_entity_task(tasks.process_climate_data, gdirs)
        workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs, informed_threestep=True)
        workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)
        workflow.calibrate_inversion_from_consensus(gdirs, apply_fs_on_mismatch=True,
                                                    error_on_mismatch=False, filter_inversion_output=True)
        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
        with open(os.path.join(sdir, "glaciers.txt"), "w") as f:
            f.write("\n".join(gd.rgi_id for gd in gdirs) + "\n")
        utils.compile_glacier_statistics(gdirs, path=os.path.join(sdir, "glacier_statistics.csv"))
        _write_marker(sdir, "fetch", setup, dem_source=[gd.get_diagnostics().get("dem_source") for gd in gdirs],
                      dem_file=user_dem,
                      dx=[float(gd.grid.dx) for gd in gdirs])
    from oggm.shop import gcm_climate
    for gcm, ssp in itertools.product(setup["gcms"], setup["ssps"]):
        sfx = gcm_suffix(gcm, ssp)
        todo = [gd for gd in gdirs if not os.path.exists(gd.get_filepath("gcm_data", filesuffix=sfx))]
        if todo:
            print(f"ISIMIP3b {gcm} {ssp}: {len(todo)} glacier(s)")
            workflow.execute_entity_task(gcm_climate.process_monthly_isimip_data, todo,
                                         member=gcm, ssp=ssp, output_filesuffix=sfx)
    stats = pd.read_csv(os.path.join(sdir, "glacier_statistics.csv"), index_col=0)
    cols = [c for c in ("rgi_area_km2", "dem_source", "grid_dx", "inv_volume_km3", "flowline_type") if c in stats]
    print(stats[cols].to_string())
    print(f"fetch done: {sdir}")


def gcm_suffix(gcm, ssp):
    return f"_{gcm}_{ssp}"


def prepare_spinup(conf, force=False):
    """CPU-heavy, offline: historical run over the baseline climate and, with setup.spinup, the
    dynamic melt_f calibration from 1979 that matches the geodetic MB with the glacier evolving
    (as in OGGM's level-5 directories)."""
    from oggm import tasks, workflow
    setup = conf["setup"]
    sdir = setup_dir(conf)
    if not _stage_done(sdir, "fetch", setup):
        raise RuntimeError(f"run `prepare --stage fetch` for setup '{setup['name']}' first (borah/prepare.slurm does both stages)")
    if _stage_done(sdir, "spinup", setup) and not force:
        print(f"{sdir}: spinup already done")
        return
    init_oggm(sdir, setup, offline=True, mp_processes=_ncpus())
    from oggm import cfg
    gdirs = workflow.init_glacier_directories(_glacier_ids(sdir))
    ci = gdirs[0].get_climate_info()
    y0, ye = ci["baseline_yr_0"], ci["baseline_yr_1"] + 1
    t = time.time()
    workflow.execute_entity_task(tasks.run_from_climate_data, gdirs, min_ys=y0, ye=ye,
                                 output_filesuffix="_historical")
    if setup["spinup"]:
        workflow.execute_entity_task(
            tasks.run_dynamic_melt_f_calibration, gdirs, err_dmdtda_scaling_factor=0.2,
            ys=max(1979, y0), ye=ye, melt_f_max=cfg.PARAMS["melt_f_max"],
            kwargs_run_function={"minimise_for": "area"}, ignore_errors=True,
            kwargs_fallback_function={"minimise_for": "area"}, output_filesuffix="_spinup_historical")
    _write_marker(sdir, "spinup", setup, seconds=round(time.time() - t), spinup=bool(setup["spinup"]))
    print(f"spinup done in {time.time() - t:.0f} s: {sdir}")


# --- members --------------------------------------------------------------------------------------

def _phase_curve(name):
    from . import phase
    if name in phase.STANDARD_CURVES:
        return phase.STANDARD_CURVES[name]
    if name.startswith("logistic"):          # logistic:T50:width, e.g. logistic:1.0:0.5
        _, t50, w = name.split(":")
        return phase.Logistic(float(t50), float(w), name=f"logistic {t50} C, width {w}")
    if name.startswith("threshold:"):
        t0 = float(name.split(":")[1])
        return phase.Threshold(t0, name=f"threshold {t0:g} C")
    if name.startswith("ramp:"):
        _, a, b = name.split(":")
        return phase.Ramp(float(a), float(b), name=f"ramp {a}-{b} C")
    if name.endswith(".csv"):               # tabulated curve: columns t, frac (e.g. from SCHNAPS)
        df = pd.read_csv(os.path.expandvars(os.path.expanduser(name)))
        return phase.Empirical(df["t"].values, df["frac"].values, name=os.path.basename(name))
    raise ValueError(f"unknown phase scheme {name!r}: use one of {list(phase.STANDARD_CURVES)}, "
                     "logistic:T50:width, threshold:T, ramp:Tsolid:Tliquid or a CSV file")


def _temp_std(gdir, setting, climate_filename, climate_suffix):
    """The phase model's temp_std, with 'file' falling back to the monthly climatology of the
    baseline file when the run's climate file (GCM data) has no temp_std."""
    if setting != "file":
        return setting
    import xarray as xr
    with xr.open_dataset(gdir.get_filepath(climate_filename, filesuffix=climate_suffix)) as ds:
        if "temp_std" in ds:
            return "file"
    with xr.open_dataset(gdir.get_filepath("climate_historical")) as ds:
        return ds["temp_std"].groupby("time.month").mean().values.astype(float)


def _run_glacier(gdir, conf, m, suffix):
    from oggm import tasks
    from . import dynamics, massbalance
    sc, f = conf["scenario"], conf["factors"]
    if sc["type"] == "gcm":
        clim = ("gcm_data", gcm_suffix(m["gcm"], m["ssp"]))
    else:
        clim = ("climate_historical", "")
    cls = massbalance.phase_model_class(_phase_curve(m["phase"]),
                                        temp_std=_temp_std(gdir, f["temp_std"], *clim),
                                        precip_temp_offset=m["precip_temp_offset"])
    calib = gdir.read_json("mb_calib")
    if f["recalibrate"]:
        # recalibrate melt_f on fixed geometry so that this scheme matches the same geodetic MB;
        # every scheme (OGGM's ramp too) goes through this, so they are compared on equal terms
        cls = massbalance.recalibrate(gdir, cls, suffix)
        calib = cls.calib
    kw = dict(mb_model_class=cls, output_filesuffix=suffix, store_monthly_step=conf["output"]["monthly"],
              temperature_bias=m["temperature_bias"] or None)
    sliding = f["sliding"]
    if sliding:
        from . import config
        inv = dynamics.invert_with_sliding(gdir, sliding_frac=sliding.get("fraction"), fs=sliding.get("fs"),
                                           filesuffix="_fs")
        fls = gdir.read_pickle("model_flowlines", filesuffix="_fs")
        a, b = config.calibration_years()
        l_ref = dynamics.liquid_climatology(fls, cls(gdir), range(a, b + 1))
        kw.update(evolution_model=dynamics.HydroSliding, glen_a=inv["glen_a"], fs=inv["fs"],
                  sliding_coupling=m["sliding_coupling"], liquid_ref=l_ref, init_model_fls=fls)
    spun = os.path.exists(gdir.get_filepath("model_geometry", filesuffix="_spinup_historical"))
    if sc["type"] == "random":
        from . import config
        y0, half = config.random_climate_window()
        tasks.run_random_climate(gdir, nyears=int(sc["nyears"]), y0=y0, halfsize=half, seed=int(m["seed"]),
                                 unique_samples=True, **kw)
    elif sc["type"] == "historical":
        ci = gdir.get_climate_info()
        ye = int(sc["ye"] or ci["baseline_yr_1"] + 1)
        if "init_model_fls" in kw:
            ys = int(sc["ys"] or gdir.rgi_date + 1)
        elif spun:
            ys = int(sc["ys"] or 2000)
            kw.update(init_model_filesuffix="_spinup_historical", init_model_yr=ys)
        else:
            ys = int(sc["ys"] or gdir.rgi_date + 1)
        tasks.run_from_climate_data(gdir, ys=ys, ye=ye, **kw)
    else:
        init = "_spinup_historical" if spun else "_historical"
        tasks.run_from_climate_data(gdir, climate_filename=clim[0], climate_input_filesuffix=clim[1],
                                    init_model_filesuffix=init, ye=int(sc["ye"]), **kw)
    return {k: calib.get(k) for k in ("melt_f", "prcp_fac", "temp_bias")}


def _local_scratch():
    for var in ("OGGM_LOCAL_SCRATCH", "SLURM_TMPDIR", "TMPDIR"):
        d = os.environ.get(var)
        if d and os.path.isdir(d):
            return d
    return tempfile.gettempdir()


def run_member(conf, i, keep=False):
    """One member: copy the setup's glacier directories to node-local scratch, run every glacier,
    and write members/m####.nc (OGGM's compiled run output: volume, area, length, ... per glacier),
    m####_summary.csv (one row per glacier and output year, including mean and max surface
    velocity on the main flowline) and, with output.fl_diagnostics, each glacier's flowline
    diagnostics (thickness and velocity along the flowline through time)."""
    from oggm import utils, workflow
    from . import experiments
    m = members(conf).loc[i].to_dict()
    rdir = os.path.join(results_dir(conf), "members")
    os.makedirs(rdir, exist_ok=True)
    tag = f"m{i:04d}"
    suffix = f"_{tag}"
    sdir = setup_dir(conf)
    if not _stage_done(sdir, "fetch", conf["setup"]):
        raise RuntimeError(f"setup '{conf['setup']['name']}' is not prepared: run `prepare` first")
    work = tempfile.mkdtemp(prefix=f"oggm_{conf['name']}_{tag}_", dir=_local_scratch())
    t0 = time.time()
    try:
        shutil.copytree(os.path.join(sdir, "per_glacier"), os.path.join(work, "per_glacier"),
                        ignore=shutil.ignore_patterns("*_m[0-9][0-9][0-9][0-9]*"))
        init_oggm(work, conf["setup"], offline=True, mp_processes=1, logging_level="WARNING")
        gdirs = workflow.init_glacier_directories(_glacier_ids(sdir))
        rows, calibs = [], {}
        for gd in gdirs:
            calibs[gd.rgi_id] = _run_glacier(gd, conf, m, suffix)
            if conf["output"]["fl_diagnostics"]:
                shutil.copy(gd.get_filepath("fl_diagnostics", filesuffix=suffix),
                            os.path.join(rdir, f"{tag}_{gd.rgi_id}_fl_diagnostics.nc"))
            s = experiments.summary(gd, suffix)
            s.insert(0, "rgi_id", gd.rgi_id)
            rows.append(s.reset_index())
        utils.compile_run_output(gdirs, input_filesuffix=suffix, path=os.path.join(rdir, f"{tag}.nc"))
        pd.concat(rows).to_csv(os.path.join(rdir, f"{tag}_summary.csv"), index=False)
        info = {"member": i, **m, "calibration": calibs, "seconds": round(time.time() - t0, 1),
                "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID")}
        with open(os.path.join(rdir, f"{tag}.json"), "w") as fh:
            json.dump(info, fh, indent=1, default=float)
        print(f"{tag} done in {time.time() - t0:.0f} s: {m}", flush=True)
        return True
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def _run_one(args):
    conf, i = args
    try:
        return run_member(conf, i)
    except Exception:
        print(f"m{i:04d} FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return False


def run(conf, indices, procs=1, skip_done=True):
    rdir = os.path.join(results_dir(conf), "members")
    if skip_done:
        done = [i for i in indices if os.path.exists(os.path.join(rdir, f"m{i:04d}.json"))]
        if done:
            print(f"{len(done)} member(s) already done, skipping them")
        indices = [i for i in indices if i not in done]
    if not indices:
        return 0
    if procs > 1 and len(indices) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(procs, len(indices))) as pool:
            ok = pool.map(_run_one, [(conf, i) for i in indices], chunksize=1)
    else:
        ok = [_run_one((conf, i)) for i in indices]
    failed = [i for i, k in zip(indices, ok) if not k]
    if failed:
        print(f"{len(failed)} member(s) failed: {failed}", file=sys.stderr)
    return len(failed)


# --- collect --------------------------------------------------------------------------------------

def write_members(conf):
    rdir = results_dir(conf)
    os.makedirs(rdir, exist_ok=True)
    df = members(conf)
    df.to_csv(os.path.join(rdir, "members.csv"))
    shutil.copy(conf["config_file"], os.path.join(rdir, "experiment.yml"))
    return df


def collect(conf):
    """ensemble.nc: OGGM's run output of every member stacked along `member`, with the factors as
    coordinates on that dimension; summary.csv: the per-member summaries with the factor columns."""
    import xarray as xr
    rdir = results_dir(conf)
    df = members(conf)
    present = [i for i in df.index if os.path.exists(os.path.join(rdir, "members", f"m{i:04d}.nc"))]
    missing = sorted(set(df.index) - set(present))
    if not present:
        raise RuntimeError(f"no member output in {rdir}/members")
    dss = []
    for i in present:
        with xr.open_dataset(os.path.join(rdir, "members", f"m{i:04d}.nc")) as ds:
            dss.append(ds.load())
    ens = xr.concat(dss, dim=pd.Index(present, name="member"), coords="minimal", compat="override",
                    combine_attrs="drop_conflicts")
    for col in df.columns:
        vals = df.loc[present, col].values
        ens = ens.assign_coords({col: ("member", vals.astype(str) if vals.dtype == object else vals)})
    ens.attrs["experiment"] = conf["name"]
    ens.attrs["setup"] = conf["setup"]["name"]
    ens.attrs["missing_members"] = ",".join(map(str, missing)) or "none"
    ens.to_netcdf(os.path.join(rdir, "ensemble.nc"))
    summ = []
    for i in present:
        p = os.path.join(rdir, "members", f"m{i:04d}_summary.csv")
        if os.path.exists(p):
            s = pd.read_csv(p)
            for col in df.columns:
                s.insert(0, col, df.loc[i, col])
            s.insert(0, "member", i)
            summ.append(s)
    if summ:
        pd.concat(summ).to_csv(os.path.join(rdir, "summary.csv"), index=False)
    print(f"{len(present)} of {len(df)} members collected into {rdir}/ensemble.nc and summary.csv")
    if missing:
        print(f"missing members: {missing}. Rerun them with: sbatch --array=... borah/run_members.slurm "
              f"{conf['config_file']}  (finished members are skipped)")
    return ens


# --- command line ---------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m wolverine_oggm.hpc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build the glacier directories of the experiment's setup")
    p.add_argument("config")
    p.add_argument("--stage", choices=["fetch", "spinup", "all"], default="all")
    p.add_argument("--force", action="store_true", help="rebuild even if done (deletes the setup's runs)")
    p = sub.add_parser("list", help="write members.csv and print the number of members")
    p.add_argument("config")
    p.add_argument("--quiet", action="store_true", help="print only the number")
    p = sub.add_parser("run", help="run some members")
    p.add_argument("config")
    p.add_argument("--members", default="all", help="e.g. 0-9,12 (default all)")
    p.add_argument("--procs", type=int, default=None, help="members at a time (default $SLURM_CPUS_PER_TASK or 1)")
    p.add_argument("--rerun", action="store_true", help="rerun members that already have output")
    p = sub.add_parser("collect", help="merge member outputs")
    p.add_argument("config")
    p = sub.add_parser("show", help="print the parsed configuration and paths")
    p.add_argument("config")
    a = ap.parse_args(argv)
    conf = load_config(a.config)
    if a.cmd == "prepare":
        if a.stage in ("fetch", "all"):
            prepare_fetch(conf, force=a.force)
        if a.stage in ("spinup", "all"):
            prepare_spinup(conf, force=a.force)
    elif a.cmd == "list":
        df = write_members(conf)
        if a.quiet:
            print(len(df))
        else:
            print(df.to_string())
            print(f"{len(df)} members; table in {results_dir(conf)}/members.csv")
    elif a.cmd == "run":
        df = members(conf)          # (members.csv is written by `list`; array tasks must not race on it)
        idx = parse_range(a.members, len(df))
        return 1 if run(conf, idx, procs=a.procs or _ncpus(), skip_done=not a.rerun) else 0
    elif a.cmd == "collect":
        collect(conf)
    elif a.cmd == "show":
        print(json.dumps(conf, indent=1, default=str))
        print(f"OGGM_ROOT  {oggm_root()}\nsetup dir  {setup_dir(conf)}\nresults    {results_dir(conf)}")
        print(f"members    {len(members(conf))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
