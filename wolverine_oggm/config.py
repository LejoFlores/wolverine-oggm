"""Shared settings for the Wolverine OGGM notebooks: glacier IDs, data sources, paths and the
offline test mode.

Every notebook starts with

    from wolverine_oggm import config
    gdir = config.get_glacier_directory()

which returns an OGGM GlacierDirectory for Wolverine Glacier from OGGM's preprocessed
directories (W5E5 climate, elevation-band flowlines, calibrated mass balance, inverted thickness,
dynamically spun-up to the RGI date). The first call downloads ~20 MB into WORKING_DIR; later
calls reuse it.

Test mode: set the environment variable WOLVERINE_OGGM_TEST=1 to run every notebook on OGGM's
offline sample glacier (Hintereisferner, Austria, with HISTALP climate) instead. That needs no
internet beyond OGGM's sample-data repository and is how the notebooks are checked; the numbers
are not Wolverine's.
"""
import os

import numpy as np

# --- Wolverine Glacier -------------------------------------------------------------------------
RGI6_ID = "RGI60-01.09162"                  # OGGM 1.6 works on RGI 6; the WGMS link is in OGGM's tables
RGI7_ID = "RGI2000-v7.0-G-01-11350"         # same glacier in RGI 7.0 (data/rgi7_glaciers_wolverine.geojson)
WGMS_ID = 94                                # USGS benchmark glacier; annual and seasonal MB since 1966
NAME = "Wolverine Glacier"

# USGS glacier station 15236895 (990 m), as in schnaps-borah's post/plot_domains.py
SITES = {"Glacier station (990 m)": (60.3869, -148.9371)}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RGI 7.0 outlines (NSIDC-0770, region 01) for the SCHNAPS Wolverine d02 box, copied from
# schnaps-borah/setup/wolverine/rgi7_glaciers_wolverine.geojson
RGI7_OUTLINES = os.path.join(REPO, "data", "rgi7_glaciers_wolverine.geojson")

# --- OGGM data ---------------------------------------------------------------------------------
# OGGM 1.6.3's default preprocessed directories: W5E5 baseline climate, elevation-band flowlines,
# per-glacier calibration on Hugonnet et al. (2021) geodetic MB 2000-2020, dynamic spinup.
PREPRO_BASE_URL = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
                   "L3-L5_files/2025.6/elev_bands/W5E5/per_glacier_spinup/")
PREPRO_LEVEL = 5          # 5 = ready to run (model_flowlines and the spun-up glacier included)
PREPRO_BORDER = 80        # grid points around the glacier in the local map

WORKING_DIR = os.environ.get("WOLVERINE_OGGM_WORKDIR",
                             os.path.join(os.path.expanduser("~"), "OGGM_runs", "wolverine"))
FIG_DIR = os.environ.get("WOLVERINE_OGGM_FIGDIR", os.path.join(WORKING_DIR, "figures"))

TEST_MODE = os.environ.get("WOLVERINE_OGGM_TEST", "0") not in ("", "0", "false", "False")


def init_oggm(working_dir=None, reset=False, logging_level="CRITICAL"):
    """cfg.initialize with the settings these notebooks rely on. OGGM logs every task at its
    "WORKFLOW" level, which is above WARNING, so the default here is CRITICAL (quiet). Pass
    logging_level="WORKFLOW" or "INFO" to see what OGGM is doing."""
    from oggm import cfg, utils
    cfg.initialize(logging_level=logging_level)
    cfg.set_logging_config(logging_level)
    wd = working_dir or (WORKING_DIR + ("_test" if TEST_MODE else ""))
    cfg.PATHS["working_dir"] = utils.mkdir(wd, reset=reset)
    cfg.PARAMS["border"] = PREPRO_BORDER
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["store_model_geometry"] = True
    cfg.PARAMS["store_fl_diagnostics"] = True     # per-gridpoint thickness, velocity, MB through time
    cfg.PARAMS["continue_on_error"] = False
    os.makedirs(FIG_DIR, exist_ok=True)
    return cfg.PATHS["working_dir"]


def get_glacier_directory(reset=False):
    """Wolverine's preprocessed GlacierDirectory (or the offline sample glacier in test mode)."""
    init_oggm(reset=reset)
    if TEST_MODE:
        return _test_glacier_directory(reset=reset)
    from oggm import workflow
    gdirs = workflow.init_glacier_directories([RGI6_ID], from_prepro_level=PREPRO_LEVEL,
                                              prepro_border=PREPRO_BORDER,
                                              prepro_base_url=PREPRO_BASE_URL, reset=reset, force=reset)
    return gdirs[0]


def calibration_period():
    """Period of the reference geodetic MB, the window the MB model is calibrated on."""
    return ("1990-01-01_2003-01-01" if TEST_MODE else "2000-01-01_2020-01-01")


def calibration_years():
    a, b = calibration_period().split("_")
    return int(a[:4]), int(b[:4]) - 1


def last_climate_year(gdir):
    from oggm import cfg   # noqa: F401
    import xarray as xr
    with xr.open_dataset(gdir.get_filepath("climate_historical")) as ds:
        return int(ds["time.year"][-1])


# --- offline test glacier ------------------------------------------------------------------------
TEST_REF_MB = -1100.0     # kg m-2 yr-1, roughly Hintereisferner's 1990-2002 mean (WGMS)


def _test_glacier_directory(reset=False):
    """Hintereisferner from OGGM's sample data, processed the same way as the preprocessed
    directories (elevation bands, scalar MB calibration, inversion, spinup-free initial state)."""
    import geopandas as gpd
    from oggm import cfg, tasks, utils, workflow
    from oggm.core import massbalance

    cfg.PATHS["dem_file"] = utils.get_demo_file("hef_srtm.tif")
    cfg.PATHS["climate_file"] = utils.get_demo_file("histalp_merged_hef.nc")
    cfg.PARAMS["baseline_climate"] = "CUSTOM"
    cfg.PARAMS["prcp_fac"] = 2.5
    cfg.PARAMS["border"] = 10
    cfg.PARAMS["use_intersects"] = False
    rgidf = gpd.read_file(utils.get_demo_file("Hintereisferner_RGI5.shp"))
    gdir = workflow.init_glacier_directories(rgidf, reset=reset, force=reset)[0]
    if os.path.exists(gdir.get_filepath("model_flowlines")) and not reset:
        return gdir
    for task in (tasks.define_glacier_region, tasks.simple_glacier_masks,
                 tasks.elevation_band_flowline, tasks.fixed_dx_elevation_band_flowline,
                 tasks.compute_downstream_line, tasks.compute_downstream_bedshape,
                 tasks.process_custom_climate_data):
        task(gdir)
    massbalance.mb_calibration_from_scalar_mb(gdir, ref_mb=TEST_REF_MB, ref_period=calibration_period(),
                                              overwrite_gdir=True)
    tasks.apparent_mb_from_any_mb(gdir, mb_years=calibration_years())
    tasks.prepare_for_inversion(gdir)
    tasks.mass_conservation_inversion(gdir)
    tasks.filter_inversion_output(gdir)
    tasks.init_present_time_glacier(gdir)
    return gdir


def fig_path(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def rng(seed=0):
    return np.random.default_rng(seed)


# --- run windows -----------------------------------------------------------------------------------
def random_climate_window():
    """(y0, halfsize) for stationary-climate experiments: the calibration period, years drawn at
    random from it."""
    a, b = calibration_years()
    return (a + b) // 2, (b - a) // 2


def historical_run(gdir):
    """kwargs for tasks.run_from_climate_data over the recent past: from the dynamically spun-up
    glacier of 2000 (preprocessed directories) to the end of the climate data, or, in test mode,
    from the present glacier over the calibration period."""
    if TEST_MODE:
        a, b = calibration_years()
        return dict(ys=a, ye=b + 1, init_model_fls=gdir.read_pickle("model_flowlines"))
    ye = last_climate_year(gdir) + 1
    if os.path.exists(gdir.get_filepath("model_geometry", filesuffix="_spinup_historical")):
        return dict(ys=2000, ye=ye, init_model_filesuffix="_spinup_historical", init_model_yr=2000)
    return dict(ys=int(gdir.rgi_date) + 1, ye=ye, init_model_fls=gdir.read_pickle("model_flowlines"))


def daily_temp_spread(gdir, default=2.5):
    """'file' when the climate file has the daily temperature s.d. (W5E5 does), else a typical
    value in K (the HISTALP test climate has none)."""
    import xarray as xr
    with xr.open_dataset(gdir.get_filepath("climate_historical")) as ds:
        return "file" if "temp_std" in ds else default
