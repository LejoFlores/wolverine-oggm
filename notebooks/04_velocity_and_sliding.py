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
# # 04 · Velocity: deformation, sliding, and a water-driven sliding hypothesis
#
# Notebook 01 showed that standard OGGM has no basal sliding, and that velocity responds to
# precipitation phase only through geometry. Observations at Wolverine and other maritime
# glaciers show strong **seasonal** speed-ups when surface water reaches the bed, from melt and
# from rain. Rain-on-glacier events in autumn are a candidate link between precipitation phase
# and velocity that works over days to months, not decades.
#
# This notebook:
#
# 1. compares OGGM's present-day velocity with observed velocity (ITS_LIVE, Millan et al. 2022);
# 2. re-inverts the thickness with sliding at a chosen sliding fraction, keeping the volume;
# 3. runs **H5**: if the sliding parameter scales with liquid water input (rain + melt), how do
#    seasonal velocity and long-term geometry respond to rain/snow partitioning?
#
# H5 uses `wolverine_oggm.dynamics.HydroSliding`:
# $f_s(t) = f_{s0}\,\max\big(0,\ 1 + k\,(L(t)/L_{ref} - 1)\big)$, where $L$ is the glacier-wide
# monthly liquid input. It is a conceptual sensitivity test with one free parameter $k$, not a
# subglacial hydrology model. Real sliding depends on how fast water input changes, and on whether
# the drainage system is efficient or not. The results bound how much a water–sliding link
# *could* matter. They are not a prediction.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from oggm import cfg

from wolverine_oggm import config, dynamics, experiments, massbalance, phase, plotting

warnings.filterwarnings("ignore")
plotting.set_style("paper")
gdir = config.get_glacier_directory()
fl0 = gdir.read_pickle("model_flowlines")[-1]
diag = gdir.get_diagnostics()
glen_a0 = diag.get("inversion_glen_a", cfg.PARAMS["glen_a"])

# %% [markdown]
# ## Observed velocity by elevation
#
# Elevation-band flowlines have no map position, so observed velocity maps are compared **by
# elevation**: average the observed speed of all glacier pixels in each elevation bin, and plot
# the model's flowline velocity against its surface elevation. OGGM's data shop adds ITS_LIVE
# and Millan velocities to `gridded_data.nc` (needs internet the first time).

# %%
def add_velocity_obs(gdir):
    if config.TEST_MODE:          # no velocity products for the offline test glacier
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            return ds.load()
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        have = [v for v in ("itslive_v", "millan_v") if v in ds]
    if "itslive_v" not in have:
        try:
            from oggm.shop import its_live
            its_live.itslive_velocity_to_gdir(gdir)
        except Exception as err:
            print("ITS_LIVE not added:", err.__class__.__name__, err)
    if "millan_v" not in have:
        try:
            from oggm.shop import millan22
            millan22.millan_velocity_to_gdir(gdir)
        except Exception as err:
            print("Millan 2022 not added:", err.__class__.__name__, err)
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        return ds.load()


grid = add_velocity_obs(gdir)
obs_vars = [v for v in ("itslive_v", "millan_v") if v in grid]
print("Observed velocity fields:", obs_vars or "none")


def by_elevation(grid, var, bins):
    m = grid["glacier_mask"].values == 1
    z = grid["topo"].values[m]
    v = grid[var].values[m]
    k = np.digitize(z, bins) - 1
    out = pd.DataFrame({"k": k, "v": v}).dropna().groupby("k")["v"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    out.columns = ["median", "q25", "q75"]
    out.index = 0.5 * (bins[out.index] + bins[out.index + 1])
    return out


bins = np.arange(np.floor(fl0.surface_h.min() / 50) * 50, fl0.surface_h.max() + 50, 50)
x, u_def, _ = dynamics.velocity_partition(fl0, glen_a0, 0.0)
zs = 0.5 * (fl0.surface_h[1:] + fl0.surface_h[:-1])
fig, ax = plt.subplots(figsize=(plotting.WIDTH_1COL * 1.2, 3))
ice = 0.5 * (fl0.thick[1:] + fl0.thick[:-1]) > 0
ax.plot(u_def[ice] * 5 / 4, zs[ice], color=plotting.ICE, label="OGGM, deformation only (inversion)")
for var, c in zip(obs_vars, ("k", plotting.GREY)):
    b = by_elevation(grid, var, bins)
    ax.plot(b["median"], b.index, color=c, label=f"{var} (median, IQR)")
    ax.fill_betweenx(b.index, b["q25"], b["q75"], color=c, alpha=0.15, lw=0)
ax.set_xlabel("Surface velocity (m yr⁻¹)")
ax.set_ylabel("Elevation (m)")
ax.legend(frameon=False, fontsize=6)
plotting.test_mode_note(fig)

# %% [markdown]
# The comparison is approximate. Observed annual velocities include sliding and are
# speed-averaged over the bin's real flow paths. The elevation-band flowline is a geometric
# abstraction with smoothed slopes. Still, the comparison shows whether deformation alone can
# explain the observed speeds. Where observations are clearly faster, sliding (or a stiffer
# calibrated A) is needed.
#
# ## Re-inverting with sliding
#
# Choose the share of the depth-averaged speed due to sliding, area-weighted over the glacier.
# `invert_with_sliding` searches for $f_s$, recalibrating $A$ each time so the glacier volume is
# unchanged, and writes `model_flowlines_fs*` next to the originals. Varying the sliding fraction
# is itself a hypothesis axis.

# %%
inv = {}
for frac in (0.25, 0.5):
    inv[frac] = dynamics.invert_with_sliding(gdir, sliding_frac=frac, filesuffix=f"_fs{int(frac * 100)}")
pd.DataFrame(inv).T[["glen_a", "fs", "sliding_fraction", "volume_km3", "volume_km3_no_sliding"]].astype(float)

# %% [markdown]
# ## H5: liquid water input drives sliding
#
# Setup: the 50% sliding inversion, stationary climate, monthly time steps and output. The
# reference liquid input $L_{ref}$ comes from the default phase scheme over the calibration
# period, and is the same for all runs, so a scheme with more rain also has more sliding.
#
# Runs:
#
# * coupling $k$ = 0 (constant sliding), 0.5 and 1 with the default scheme;
# * $k$ = 1 with all precipitation as snow below 2 °C vs below 0 °C (thresholds 2 °C and 0 °C,
#   recalibrated), which is more rain vs less rain at the same total precipitation;
# * $k$ = 1 with +1 K applied to the phase only, so snow turns into rain while melt is unchanged.

# %%
NYEARS = 20 if config.TEST_MODE else 40
FRAC = 0.5
sl = inv[FRAC]
fls_fs = gdir.read_pickle("model_flowlines", filesuffix=sl["filesuffix"])
spread = config.daily_temp_spread(gdir)
base = massbalance.recalibrate(gdir, massbalance.phase_model_class(phase.STANDARD_CURVES["oggm"], temp_std=spread), "_ph4")
y0, y1 = config.calibration_years()
L_ref = dynamics.liquid_climatology(fls_fs, base(gdir), range(y0, y1 + 1))
print(f"L_ref = {L_ref:.0f} kg m-2 per month (glacier-wide rain + melt)")

thr2 = massbalance.recalibrate(gdir, massbalance.phase_model_class(phase.Threshold(2.0), temp_std=spread), "_thr2s")
thr0 = massbalance.recalibrate(gdir, massbalance.phase_model_class(phase.Threshold(0.0), temp_std=spread), "_thr0s")
h5 = {
    "k=0": (base, 0.0),
    "k=0.5": (base, 0.5),
    "k=1": (base, 1.0),
    "k=1, threshold 2 °C (less rain)": (thr2, 1.0),
    "k=1, threshold 0 °C (more rain)": (thr0, 1.0),
    "k=1, phase +1 K": (massbalance.with_settings(base, precip_temp_offset=1.0), 1.0),
}
models = {}
for k, (label, (cls, coupling)) in enumerate(h5.items()):
    models[label] = experiments.run_stationary(
        gdir, f"_h5_{k}", cls, NYEARS, seed=1, init_model_fls=fls_fs, store_monthly_step=True,
        evolution_model=dynamics.HydroSliding, glen_a=sl["glen_a"], fs=sl["fs"],
        sliding_coupling=coupling, liquid_ref=L_ref)
experiments.compare(gdir, [f"_h5_{k}" for k in range(len(h5))], labels=list(h5), tail=12 * 5).round(2)

# %% [markdown]
# ### Seasonal cycle of velocity
#
# Monthly output: the value stored at time $t$ is the velocity during the month that ends at $t$.

# %%
def seasonal_cycle(gdir, suffix, last_years=10):
    d = experiments.summary(gdir, suffix)
    d = d[d.index >= d.index.max() - last_years]
    month = (np.round(((d.index.values - 1 / 12) % 1) * 12).astype(int) % 12) + 1
    return d.groupby(month)["velocity_mean_myr"].mean()


fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 2.6))
for k, label in enumerate(h5):
    sc = seasonal_cycle(gdir, f"_h5_{k}")
    axes[0].plot(sc.index, sc.values, "-o", ms=2, label=label)
m = models["k=1"]
fh = pd.DataFrame(m.fs_history, columns=["year", "L", "fs"]).set_index("year")
last = fh[fh.index >= fh.index.max() - 3]
axes[1].plot(last.index, last["L"], color=plotting.RAIN)
axes[1].set_ylabel("Liquid input (kg m⁻² month⁻¹)", color=plotting.RAIN)
ax2 = axes[1].twinx()
ax2.plot(last.index, last["fs"] / sl["fs"], color="k", lw=0.8)
ax2.set_ylabel("f_s / f_s0")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Mean surface velocity (m yr⁻¹)")
axes[1].set_xlabel("Model year")
axes[0].legend(frameon=False, fontsize=5)
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)

# %% [markdown]
# ### Long-term: does the faster flow change the geometry?

# %%
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 2.4))
for k, label in enumerate(h5):
    d = experiments.summary(gdir, f"_h5_{k}")
    ann = d.groupby(np.floor(d.index)).mean()
    axes[0].plot(ann.index, ann.volume_km3, label=label)
    axes[1].plot(ann.index, ann.velocity_mean_myr)
axes[0].set_ylabel("Volume (km³)")
axes[1].set_ylabel("Annual-mean surface velocity (m yr⁻¹)")
for ax in axes:
    ax.set_xlabel("Model year")
axes[0].legend(frameon=False, fontsize=5)
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)

# %% [markdown]
# ## Interpreting H5
#
# * With $k$ > 0, rain and melt months speed the glacier up. The **phase** experiments
#   (threshold 0 vs 2 °C, phase +1 K) show how much of that seasonal signal comes from
#   precipitation arriving as rain rather than snow.
# * Over decades, two effects compete: more rain means less accumulation (thinning, slower flow),
#   while more sliding moves ice to lower, warmer elevations faster (more melt). Both reduce
#   volume, but they act on velocity in opposite directions.
# * Tests against data: the observed seasonal velocity cycle (e.g. ITS_LIVE image pairs, or GNSS
#   if the field programme has it) should constrain $k$ and the sliding fraction. Rain-on-snow
#   events in autumn are the sharpest test, because melt is small then.
#
# **Limitations to state in a thesis.** Sliding here responds to the monthly *amount* of water,
# with no drainage evolution, no storage and no lag. OGGM's mass balance stays monthly, and
# elevation-band flowlines have no bed geometry along a real flow path. A physically based
# test needs a coupled hydrology–sliding model (e.g. GlaDS or SHAKTI in ISSM or Elmer/Ice). This
# notebook helps decide whether that investment is warranted.
