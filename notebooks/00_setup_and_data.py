# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 · Setup and data: Wolverine Glacier in OGGM
#
# This is the first notebook of the suite. It gets Wolverine Glacier into the
# [Open Global Glacier Model (OGGM)](https://oggm.org) and looks at everything OGGM knows about it
# before any experiment: the outline, the topography, the flowline, the climate, the calibrated
# mass balance and the inverted ice thickness.
#
# | Notebook | What it does |
# | --- | --- |
# | **00 Setup and data** | Download the preprocessed glacier directory and inspect the inputs |
# | 01 OGGM primer | The model in equations: mass balance, phase, inversion, flow, velocity |
# | 02 Mass balance and calibration | How the MB model is calibrated, and checks against the USGS/WGMS series |
# | 03 Precipitation-phase experiments | Alternative rain/snow schemes and warming with and without phase change |
# | 04 Velocity and sliding | OGGM velocities, observed velocities, and a water-driven sliding hypothesis |
# | 05 SCHNAPS forcing | Phase curves from the SCHNAPS microphysics, and splicing SCHNAPS months into the climate |
# | 06 Publication figures | Figure recipes for talks and papers |
#
# **Where to run.** A laptop is fine: Wolverine is one glacier and runs take seconds. The first
# call downloads about 20 MB from OGGM's server (`cluster.klima.uni-bremen.de`). On Borah, run
# the download cell on the login node (compute nodes have no internet), then work anywhere.
#
# **Test mode.** With the environment variable `WOLVERINE_OGGM_TEST=1`, every notebook runs on
# OGGM's offline sample glacier (Hintereisferner, Austria) instead. That's how the notebooks are
# checked. The numbers are then not Wolverine's, and figures carry a red "TEST MODE" note.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))      # the wolverine_oggm package next to this folder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wolverine_oggm import config, plotting

plotting.set_style("paper")
print("Test mode:", config.TEST_MODE)

# %% [markdown]
# ## The glacier directory
#
# OGGM keeps everything about one glacier in a *glacier directory* (`gdir`): the outline, a local
# DEM and map projection, the flowlines, the climate file, calibrated parameters and model
# output. We start from OGGM's **preprocessed level 5** directory. It has W5E5 climate,
# elevation-band flowlines, a mass-balance model calibrated per glacier on the Hugonnet et al.
# (2021) geodetic mass balance for 2000–2020, inverted thickness, and a dynamically spun-up
# glacier. Those are OGGM 1.6's standard choices for global runs, which makes them a defensible
# baseline. The notebooks then change one thing at a time.
#
# `config.get_glacier_directory(reset=True)` starts from scratch. That deletes everything you
# computed in the working directory.

# %%
gdir = config.get_glacier_directory()
gdir

# %%
print("Working directory:", gdir.dir)
sorted(os.listdir(gdir.dir))

# %% [markdown]
# ## Outline and topography
#
# OGGM 1.6 uses the RGI 6.0 outline (`RGI60-01.09162`). The SCHNAPS domains use RGI 7.0 outlines
# (`RGI2000-v7.0-G-01-11350`). The map below draws both, plus the USGS glacier station (990 m),
# to show how much they differ.

# %%
from oggm import graphics

fig, ax = plt.subplots(figsize=(plotting.WIDTH_1COL * 1.4, plotting.WIDTH_1COL * 1.2))
graphics.plot_domain(gdir, ax=ax)
if not config.TEST_MODE:
    import geopandas as gpd
    import salem
    rgi7 = gpd.read_file(config.RGI7_OUTLINES)
    rgi7 = rgi7[rgi7.rgi_id == config.RGI7_ID]
    for geom in rgi7.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            lon, lat = (np.asarray(c) for c in poly.exterior.xy)
            i, j = gdir.grid.transform(lon, lat, crs=salem.wgs84)   # map pixel coordinates
            ax.plot(i, j, color="orange", lw=1)
    ax.plot([], [], color="orange", lw=1, label="RGI 7.0 outline")
    for name, (la, lo) in config.SITES.items():
        i, j = gdir.grid.transform(lo, la, crs=salem.wgs84)
        ax.plot(i, j, "k^", ms=5)
        ax.annotate(name, (i, j), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.legend(loc="lower left", frameon=False)
ax.set_title(config.NAME if not config.TEST_MODE else gdir.name)
plotting.test_mode_note(fig)

# %% [markdown]
# ## Flowline
#
# Preprocessed OGGM 1.6 directories use **elevation-band flowlines** (Huss & Farinotti 2012).
# The glacier is collapsed into a single flowline whose grid points are elevation bands. Each band
# keeps the right area and the area-elevation distribution, but it is not a real path down the
# glacier. That matters when comparing with velocity maps (notebook 04): observed velocities
# have to be averaged by elevation band first.

# %%
fls = gdir.read_pickle("model_flowlines")
fl = fls[-1]
print(f"{len(fls)} flowline(s), {fl.nx} grid points, dx = {fl.dx_meter:.0f} m, "
      f"area {sum(f.area_km2 for f in fls):.2f} km2, volume {sum(f.volume_km3 for f in fls):.3f} km3")

fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 2.4))
plotting.flowline_profile(axes[0], fl)
axes[0].legend(frameon=False)
x = np.arange(fl.nx) * fl.dx_meter / 1000
axes[1].plot(x, fl.widths_m, color=plotting.ICE)
axes[1].set_xlabel("Distance along flowline (km)")
axes[1].set_ylabel("Width (m)")
plotting.label_panels(axes)
plotting.test_mode_note(fig)

# %% [markdown]
# ## Climate
#
# The baseline climate is one monthly time series at the nearest W5E5 grid point (0.5°, about
# 55 km). It has temperature, precipitation and, for W5E5, the standard deviation of daily
# temperature within each month (`temp_std`). Notebook 03 uses `temp_std` for sub-monthly phase.
# OGGM moves temperature to each flowline height with a fixed lapse rate (−6.5 K/km). Precipitation
# is the same at all heights, multiplied by a calibrated precipitation factor.

# %%
clim = xr.open_dataset(gdir.get_filepath("climate_historical")).load()
print(clim.attrs.get("climate_source"), "| reference height", clim.attrs["ref_hgt"], "m")
clim

# %%
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 2.4))
ann = clim.resample(time="YS").mean()
axes[0].bar(ann["time.year"], clim["prcp"].resample(time="YS").sum(), color=plotting.ICE, alpha=0.4)
axes[0].set_ylabel("Annual precipitation (mm)")
ax2 = axes[0].twinx()
ax2.plot(ann["time.year"], ann["temp"], color="k")
ax2.set_ylabel(f"Annual mean T at {clim.attrs['ref_hgt']:.0f} m (°C)")
clm = clim.groupby("time.month").mean()
axes[1].bar(clm["month"], clm["prcp"], color=plotting.ICE, alpha=0.4, label="Precipitation")
axes[1].set_ylabel("Monthly precipitation (mm)")
ax3 = axes[1].twinx()
ax3.plot(clm["month"], clm["temp"], "k-o", ms=3, label="Temperature")
if "temp_std" in clim:
    ax3.fill_between(clm["month"], clm["temp"] - clm["temp_std"], clm["temp"] + clm["temp_std"], color="k", alpha=0.1)
ax3.axhline(0, color=plotting.GREY, lw=0.5)
ax3.set_ylabel("T (°C), ±1 daily s.d.")
axes[1].set_xlabel("Month")
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)

# %% [markdown]
# ## Calibrated mass-balance parameters
#
# `mb_calib.json` holds the three parameters of the temperature-index model:
#
# * `melt_f`: melt factor (kg m⁻² day⁻¹ K⁻¹)
# * `prcp_fac`: multiplies the baseline precipitation. It corrects for gauge undercatch, orographic
#   enhancement and the coarse grid, and is often 2–3 for maritime glaciers.
# * `temp_bias`: added to the baseline temperature
#
# It also holds the reference MB they were fitted to, and the global settings in force:
# `temp_all_solid` and `temp_all_liq` (the rain/snow ramp) and `temp_melt`.

# %%
calib = gdir.read_json("mb_calib")
pd.Series({k: v for k, v in calib.items() if not isinstance(v, dict)})

# %%
calib["mb_global_params"]

# %% [markdown]
# ## Observed mass balance (USGS / WGMS)
#
# Wolverine is a USGS benchmark glacier (WGMS id 94), with glaciological mass balance since 1966.
# OGGM ships the WGMS series for its reference glaciers. Notebook 02 compares it with the model.

# %%
try:
    ref_mb = gdir.get_ref_mb_data()
    print(ref_mb[["ANNUAL_BALANCE"]].describe().T)
    ref_mb.tail()
except Exception as err:          # not every glacier is a WGMS reference glacier
    ref_mb = None
    print("No WGMS series for this glacier:", err)

# %% [markdown]
# ## Inverted thickness
#
# OGGM inverts for thickness by mass conservation. It assumes the glacier is in equilibrium with
# the calibrated "apparent" mass balance, and that the ice flux at each point follows the
# shallow-ice approximation with Glen's A (and no sliding by default). A is tuned so the regional
# volume matches the consensus estimate (Farinotti et al. 2019).

# %%
diag = gdir.get_diagnostics()
for k in diag:
    if k.startswith("inversion") or k in ("flowline_type", "dem_source"):
        print(f"{k:25s} {diag[k]}")

# %%
fig, ax = plt.subplots(figsize=(plotting.WIDTH_1COL * 1.3, 2.4))
ax.plot(x, fl.thick, color=plotting.ICE)
ax.set_xlabel("Distance along flowline (km)")
ax.set_ylabel("Ice thickness (m)")
plotting.test_mode_note(fig)

# %% [markdown]
# ## Next
#
# Notebook 01 opens up the model: how the mass balance, the rain/snow split and the ice flow are
# computed, and where velocity comes from.
